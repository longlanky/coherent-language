"""HTTP API for Cohere Transcribe ASR models.

Serves the locally cached HuggingFace models:
  - CohereLabs/cohere-transcribe-03-2026        (14 languages)
  - CohereLabs/cohere-transcribe-arabic-07-2026 (Arabic + English)
  - ivrit-ai/whisper-large-v3-turbo-ct2         (Hebrew; faster-whisper backend)

Run:  .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
"""

import base64
import gc
import json
import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass

import torch
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

import llm_cleanup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cohere-asr")

SAMPLE_RATE = 16000

# --- Request and job limits -------------------------------------------------
# Uploads are streamed to disk and rejected with 413 past the cap (/tmp is
# tmpfs on this host, so an oversized upload costs RAM, not disk).
MAX_UPLOAD_MB = int(os.environ.get("ASR_MAX_UPLOAD_MB", "200"))
MAX_AUDIO_SECONDS = float(os.environ.get("ASR_MAX_AUDIO_SECONDS", "7200"))
# Backpressure for the background path: one worker, so a deep queue means
# everything behind it waits. Reject early instead of queueing forever.
MAX_QUEUED_JOBS = int(os.environ.get("ASR_MAX_QUEUED_JOBS", "32"))
# Finished/failed jobs are dropped after this long (they hold transcripts and
# any TTS audio), so memory cannot grow without bound.
JOB_TTL_S = int(os.environ.get("ASR_JOB_TTL_S", "3600"))
# Model weights come from the local HF cache; never fetch during a request.
LOCAL_FILES_ONLY = os.environ.get("ASR_LOCAL_FILES_ONLY", "1") not in ("0", "false", "")
# Optional integrity pin: {"model-id": "revision-sha", ...} as JSON.
_MODEL_REVISIONS = json.loads(os.environ.get("ASR_MODEL_REVISIONS", "{}") or "{}")

MODELS = {
    "cohere-transcribe-03-2026": {
        "repo_id": "CohereLabs/cohere-transcribe-03-2026",
        "default_language": "en",
        "quantize_encoder": False,
        "backend": "cohere",
    },
    "cohere-transcribe-arabic-07-2026": {
        "repo_id": "CohereLabs/cohere-transcribe-arabic-07-2026",
        "default_language": "ar",
        "quantize_encoder": False,
        "backend": "cohere",
    },
    # INT8-encoder / FP16-decoder hybrids (FluidInference CoreML q8 scheme,
    # reproduced natively in PyTorch — CoreML itself can't run on Linux).
    "cohere-transcribe-03-2026-int8enc": {
        "repo_id": "CohereLabs/cohere-transcribe-03-2026",
        "default_language": "en",
        "quantize_encoder": True,
        "backend": "cohere",
    },
    "cohere-transcribe-arabic-07-2026-int8enc": {
        "repo_id": "CohereLabs/cohere-transcribe-arabic-07-2026",
        "default_language": "ar",
        "quantize_encoder": True,
        "backend": "cohere",
    },
    # Hebrew ASR: ivrit.ai's whisper-large-v3-turbo fine-tune in CTranslate2
    # format, run via faster-whisper (a separate runtime from the Cohere stack).
    "ivrit-whisper-large-v3-turbo-ct2": {
        "repo_id": "ivrit-ai/whisper-large-v3-turbo-ct2",
        "default_language": "he",
        "quantize_encoder": False,
        "backend": "faster-whisper",
    },
}

# Runtime config: GPU by default when available (fp16 + accelerate CPU offload,
# since a 2B fp16 model barely fits in the P1000's 4GB). Override with env vars.
_device = os.environ.get("ASR_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
_dtype = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}[os.environ.get("ASR_DTYPE") or ("float16" if _device == "cuda" else "float32")]

# ctranslate2 compute type for the faster-whisper backend. int8 (DP4A, needs
# CC >= 6.1) is the fast path on the P1000 — int8_float16 requires CC >= 7.0
# and is rejected on sm_61. ~0.8 GB VRAM for the 809M-param turbo.
_whisper_compute = os.environ.get("ASR_WHISPER_COMPUTE") or "int8"

# Default model id when a request omits `model` (e.g. an -int8enc variant to
# make quantized inference the default). Override with ASR_DEFAULT_MODEL.
_default_model = os.environ.get("ASR_DEFAULT_MODEL") or "cohere-transcribe-03-2026"
if _default_model not in MODELS:
    raise RuntimeError(f"ASR_DEFAULT_MODEL '{_default_model}' not in {sorted(MODELS)}")

app = FastAPI(title="Cohere Transcribe ASR API")

_loaded = {}  # model name -> (processor, model)
# One lock for the whole load -> generate -> unload lifecycle: two models must
# never occupy the 4 GB GPU at once, and a model must never be freed from VRAM
# while another thread is generating with it.
_model_lock = threading.Lock()


def _free_vram():
    gc.collect()
    if _device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def unload_models():
    """Drop all loaded models and release their VRAM.

    Called at the end of every request so co-located GPU processes (e.g.
    ollama) can use the VRAM between requests.
    """
    for old_name in list(_loaded):
        log.info("Unloading %s", old_name)
        del _loaded[old_name]
    _free_vram()


def run_cleanup(raw_text: str, translate_to: str | None = None,
                llm: llm_cleanup.LLMConfig | None = None) -> dict:
    """Post-process a transcript with the configured LLM provider.

    Kept as a thin compatibility wrapper; all provider logic lives in
    llm_cleanup.py so it can also be used outside the HTTP service.
    """
    return llm_cleanup.clean_transcript(raw_text, translate_to, llm)


def get_model(name: str):
    """Return the loaded model, loading it first if needed.

    Caller must hold _model_lock. Anything else resident is evicted first —
    the 4 GB P1000 cannot hold two 2B models plus inference temporaries.
    """
    if name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{name}'. Available: {sorted(MODELS)}",
        )
    if name in _loaded:
        return _loaded[name]
    for old_name in list(_loaded):
        log.info("Evicting %s to make room for %s", old_name, name)
        del _loaded[old_name]
    _free_vram()

    repo_id = MODELS[name]["repo_id"]
    # Pin to a commit when ASR_MODEL_REVISIONS names one, and never hit the
    # network mid-request: a cache miss must fail loudly, not silently download.
    revision = _MODEL_REVISIONS.get(name)
    offline = dict(revision=revision, local_files_only=LOCAL_FILES_ONLY)

    if MODELS[name]["backend"] == "faster-whisper":
        from faster_whisper import WhisperModel

        log.info("Loading %s on %s (ctranslate2 %s, revision %s) ...", repo_id,
                 _device, _whisper_compute, revision or "latest cached")
        model = WhisperModel(repo_id, device=_device,
                             compute_type=_whisper_compute, **offline)
        _loaded[name] = (None, model)
        log.info("Loaded %s", repo_id)
        return _loaded[name]

    log.info("Loading %s on %s (%s, revision %s) ...", repo_id, _device, _dtype,
             revision or "latest cached")
    processor = AutoProcessor.from_pretrained(repo_id, **offline)
    if MODELS[name]["quantize_encoder"]:
        from quant import quantize_encoder_int8

        # Load on CPU and quantize there first: the fp16 model fills
        # the 4 GB GPU, leaving no room for quantization temporaries.
        model = CohereAsrForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=_dtype, **offline
        )
        with torch.no_grad():
            quantize_encoder_int8(model)
        model = model.to(_device)
    elif _device == "cuda":
        model = CohereAsrForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=_dtype, device_map="auto", **offline
        )
    else:
        model = CohereAsrForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=_dtype, **offline
        ).to(_device)
    model.eval()
    _loaded[name] = (processor, model)
    log.info("Loaded %s", repo_id)
    return _loaded[name]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "static", "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    """Liveness plus subsystem health.

    A dead worker thread or a queue that has not drained used to report "ok"
    while every job hung in "queued" forever, so liveness alone was useless.
    """
    queue_depth = _job_queue.qsize()
    worker_alive = (time.time() - _worker_heartbeat) < 300
    stale = queue_depth > 0 and (time.time() - _last_job_completed) > 1800
    return {
        "status": "degraded" if (stale or not worker_alive) else "ok",
        "queue_depth": queue_depth,
        "jobs": len(_jobs),
        "worker_alive": worker_alive,
        "seconds_since_last_job": round(time.time() - _last_job_completed, 1),
    }


@app.get("/v1/models")
def list_models():
    return {
        "device": _device,
        "dtype": str(_dtype),
        "local_files_only": LOCAL_FILES_ONLY,
        "models": [
            {
                "id": name,
                "repo_id": info["repo_id"],
                "backend": info["backend"],
                "default_language": info["default_language"],
                "quantization": "int8-encoder/fp16-decoder" if info["quantize_encoder"] else None,
                "revision": _MODEL_REVISIONS.get(name),
                "loaded": name in _loaded,
            }
            for name, info in MODELS.items()
        ],
    }


def _cohere_chunk_offsets(feature_extractor, audio, total_samples: int
                          ) -> list[tuple[float, float]]:
    """Return (start, end) seconds for the processor's audio chunks.

    The Cohere feature extractor splits long audio into contiguous chunks at
    quiet points near ``max_audio_clip_s``. It only returns the chunk tensors,
    but they tile the input exactly, so cumulative sample counts recover the
    offsets without a second ASR pass. This deliberately mirrors the exact
    chunking that produced the model outputs.
    """
    waveform = torch.as_tensor(audio, dtype=torch.float32)
    chunks = feature_extractor._split_audio_chunks_energy(waveform)
    offsets: list[tuple[float, float]] = []
    position = 0
    for chunk in chunks:
        size = int(chunk.shape[0])
        if size <= 0:
            continue
        offsets.append((position / SAMPLE_RATE, (position + size) / SAMPLE_RATE))
        position += size
    if not offsets or position != total_samples:
        raise ValueError("audio chunks did not cover the input exactly")
    return offsets


def _segments_from_chunks(chunk_texts: list[str],
                          offsets: list[tuple[float, float]],
                          fallback_text: str, duration_s: float) -> list[dict]:
    """Pair per-chunk decoded text with chunk times.

    Any misalignment falls back to one whole-file segment rather than emitting
    incorrectly timed text.
    """
    if not offsets or len(chunk_texts) != len(offsets):
        log.warning("Could not align %d chunk text(s) with %d offset(s); "
                    "using one timestamp segment", len(chunk_texts), len(offsets))
        return [{"start": 0.0, "end": round(duration_s, 2),
                 "text": fallback_text}]

    segments = []
    for (start, end), chunk in zip(offsets, chunk_texts):
        chunk = (chunk or "").strip()
        if chunk:
            segments.append({"start": round(start, 2), "end": round(end, 2),
                             "text": chunk})
    if not segments:
        return [{"start": 0.0, "end": round(duration_s, 2),
                 "text": fallback_text}]
    return segments


def _transcribe(audio_path: str, model: str, language: str | None,
                punctuation: bool, max_new_tokens: int,
                chunk_batch: int | None = None,
                timestamps: bool = False) -> dict:
    """Run one transcription on a local audio file. Holds all model/tensor
    references in this frame, so they are released when it returns — the
    caller can then free the VRAM.

    When ``timestamps`` is true the result gains ``segments``: Whisper uses its
    native segment times; Cohere uses the processor's energy-based chunk
    boundaries (its model card documents no native timestamps).
    """
    processor, asr_model = get_model(model)

    lang = language or MODELS[model]["default_language"]

    if MODELS[model]["backend"] == "faster-whisper":
        # faster-whisper decodes the audio itself (PyAV handles webm/opus/mp3
        # natively) and windows long files internally, so the load_audio
        # fallback and mini-batch loop below don't apply. `punctuation` and
        # `max_new_tokens` are Cohere-only and ignored here.
        whisper_segments, info = asr_model.transcribe(audio_path, language=lang,
                                                      beam_size=5)
        whisper_segments = list(whisper_segments)
        text = " ".join((s.text or "").strip() for s in whisper_segments).strip()
        log.info("Transcribed %.1fs of audio with %s (%s)",
                 info.duration, model, lang)
        result = {
            "text": text,
            "model": model,
            "language": lang,
            "duration_s": round(info.duration, 2),
        }
        if timestamps:
            segments = [
                {"start": round(s.start, 2), "end": round(s.end, 2),
                 "text": (s.text or "").strip()}
                for s in whisper_segments if (s.text or "").strip()
            ]
            result["segments"] = segments or [
                {"start": 0.0, "end": round(info.duration, 2), "text": text}
            ]
        return result

    languages = getattr(asr_model.config, "supported_languages", None) or []
    if languages and lang not in languages:
        raise HTTPException(
            status_code=400,
            detail=f"Language '{lang}' not supported by {model}. Supported: {languages}",
        )

    try:
        audio = load_audio(audio_path, sampling_rate=SAMPLE_RATE)
    except Exception:
        # Some containers (e.g. webm/opus from browser MediaRecorder) aren't
        # decodable by librosa — convert to 16 kHz mono wav with ffmpeg and retry.
        # The intermediate path is unpredictable (it used to be a fixed sibling
        # name), and ffmpeg is restricted to local files since these bytes came
        # from the network.
        fd, wav_path = tempfile.mkstemp(prefix="asr-", suffix=".ffmpeg.wav")
        os.close(fd)
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-protocol_whitelist", "file",
                 "-y", "-v", "error", "-i", audio_path,
                 "-ar", str(SAMPLE_RATE), "-ac", "1",
                 "-t", str(MAX_AUDIO_SECONDS), wav_path],
                check=True, capture_output=True, timeout=300,
            )
            audio = load_audio(wav_path, sampling_rate=SAMPLE_RATE)
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode audio")
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    duration_s = len(audio) / SAMPLE_RATE
    if duration_s > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"audio too long ({duration_s:.0f}s; limit {MAX_AUDIO_SECONDS:.0f}s)",
        )
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        language=lang,
        punctuation=punctuation,
    )
    audio_chunk_index = inputs.get("audio_chunk_index")
    inputs = inputs.to(asr_model.device, dtype=asr_model.dtype)

    # Long audio is chunked by the processor (35 s clips). Running generate()
    # on all chunks as one batch needs VRAM proportional to the chunk count
    # (a 24 min file = 42 chunks > 2.3 GiB of activations — fatal on a 4 GB
    # card), so generate in mini-batches and concatenate.
    chunk_batch = chunk_batch or int(os.environ.get("ASR_CHUNK_BATCH", "4"))
    n_chunks = inputs["input_features"].shape[0]
    if n_chunks <= chunk_batch:
        outputs = asr_model.generate(**inputs, max_new_tokens=max_new_tokens)
    else:
        log.info("Long-form: %d chunks, mini-batches of %d", n_chunks, chunk_batch)
        parts = []
        for i in range(0, n_chunks, chunk_batch):
            sub = {
                k: v[i:i + chunk_batch]
                for k, v in inputs.items()
                if k != "audio_chunk_index" and torch.is_tensor(v) and v.shape[0] == n_chunks
            }
            parts.append(asr_model.generate(**sub, max_new_tokens=max_new_tokens))
        # Each mini-batch returns its own max sequence length; pad to a common
        # length before concatenating (decode skips the pad/special tokens).
        pad_id = (asr_model.generation_config.pad_token_id
                  or asr_model.generation_config.eos_token_id)
        max_len = max(p.shape[1] for p in parts)
        parts = [
            torch.cat([p, p.new_full((p.shape[0], max_len - p.shape[1]), pad_id)], dim=1)
            for p in parts
        ]
        outputs = torch.cat(parts, dim=0)

    text = processor.decode(
        outputs,
        skip_special_tokens=True,
        audio_chunk_index=audio_chunk_index,
        language=lang,
    )
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""

    result = {
        "text": text,
        "model": model,
        "language": lang,
        "duration_s": round(duration_s, 2),
    }
    if timestamps:
        try:
            chunk_texts = processor.tokenizer.decode(
                outputs, skip_special_tokens=True)
            if isinstance(chunk_texts, str):
                chunk_texts = [chunk_texts]
            offsets = _cohere_chunk_offsets(
                processor.feature_extractor, audio, len(audio))
        except Exception:
            log.warning("Could not derive Cohere chunk timestamps; "
                        "using one segment", exc_info=True)
            chunk_texts, offsets = [], []
        result["segments"] = _segments_from_chunks(
            chunk_texts, offsets, text, duration_s)

    log.info("Transcribed %.1fs of audio with %s (%s)", duration_s, model, lang)
    return result


def _save_upload(file: UploadFile) -> str:
    """Stream the upload to a temp file, rejecting it at MAX_UPLOAD_MB.

    The previous version read the whole body into memory first; /tmp is tmpfs
    here, so an oversized upload cost RAM rather than disk.
    """
    # basename() first: only the extension is wanted, never a path component.
    suffix = os.path.splitext(os.path.basename(file.filename or "audio.wav"))[1] or ".wav"
    max_bytes = MAX_UPLOAD_MB * 2**20
    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds {MAX_UPLOAD_MB} MiB",
                    )
                tmp.write(chunk)
        except Exception:
            tmp.close()
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            raise
        return tmp.name


# Allowlisted translation targets — the same list the web UI offers. The target
# is interpolated into an LLM system prompt, and the transcript is untrusted
# text, so anything outside this set (including injection payloads such as
# "French. Ignore prior rules and…") is rejected rather than forwarded.
TRANSLATE_TARGETS = frozenset([
    "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani",
    "Basque", "Belarusian", "Bengali", "Bosnian", "Bulgarian", "Burmese",
    "Catalan", "Cebuano", "Chinese (Mandarin)", "Chinese (Cantonese)",
    "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian", "Finnish",
    "French", "Galician", "Georgian", "German", "Greek", "Gujarati",
    "Haitian Creole", "Hausa", "Hebrew", "Hindi", "Hungarian", "Icelandic",
    "Igbo", "Indonesian", "Irish", "Italian", "Japanese", "Javanese",
    "Kannada", "Kazakh", "Khmer", "Korean", "Kurdish", "Kyrgyz", "Lao",
    "Latvian", "Lithuanian", "Luxembourgish", "Macedonian", "Malay",
    "Malayalam", "Maltese", "Marathi", "Mongolian", "Nepali", "Norwegian",
    "Pashto", "Persian (Farsi)", "Polish", "Portuguese", "Punjabi", "Romanian",
    "Russian", "Serbian", "Sinhala", "Slovak", "Slovenian", "Somali",
    "Spanish", "Swahili", "Swedish", "Tagalog", "Tamil", "Telugu", "Thai",
    "Turkish", "Ukrainian", "Urdu", "Uzbek", "Vietnamese", "Welsh", "Yoruba",
    "Zulu",
])


@app.get("/v1/translate/targets")
def translate_targets():
    return {"targets": sorted(TRANSLATE_TARGETS)}


def _discard(path: str):
    """Best-effort removal of a temp file (never masks the caller's error)."""
    try:
        os.unlink(path)
    except OSError:
        log.warning("Could not unlink %s", path)


def _translate_target(lang: str, modifier: str) -> str | None:
    """Combine the translate form fields into a target like "Levantine Arabic".

    The language must be allowlisted and the modifier restricted to plain
    words: both end up inside an LLM prompt that also carries untrusted
    transcript text, so this is the only place they can be constrained.
    Returns None when no language was given.
    """
    lang, modifier = lang.strip()[:60], modifier.strip()[:40]
    if not lang:
        return None
    if lang not in TRANSLATE_TARGETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported translate_lang '{lang}'. See /v1/translate/targets.",
        )
    if modifier and not re.fullmatch(r"[\w \-']{1,40}", modifier):
        raise HTTPException(status_code=400,
                            detail="Invalid translate_modifier (letters, spaces, - and ' only).")
    return f"{modifier} {lang}".strip() if modifier else lang


# --- Text to speech (Kokoro-82M) -------------------------------------------
# Runs on CPU by design (fast at 82M params; keeps the GPU free for ASR).
# tts.py is imported lazily so the ASR API still starts if kokoro breaks.

_tts_lock = threading.Lock()

_TTS_VOICE_LANGS = {
    "a": "American English", "b": "British English", "e": "Spanish",
    "f": "French", "h": "Hindi", "i": "Italian", "j": "Japanese",
    "p": "Portuguese (BR)", "z": "Mandarin Chinese",
}


def _tts_synthesize(text: str, voice: str, speed: float) -> bytes:
    with _tts_lock:
        if voice.startswith("silma_"):
            import tts_silma
            return tts_silma.synthesize_wav_bytes(text, voice=voice, speed=speed)
        import tts
        return tts.synthesize_wav_bytes(text, voice=voice, speed=speed)


def _all_voices() -> list[dict]:
    """Kokoro voices plus SILMA voice-clone references (silma_*)."""
    import tts
    voices = [
        {"id": v, "engine": "kokoro",
         "gender": "female" if v[1] == "f" else "male",
         "language": _TTS_VOICE_LANGS.get(v[0], v[0])}
        for v in tts.available_voices()
    ]
    import tts_silma
    for v in tts_silma.available_voices():
        lang = "Arabic + English (voice clone)"
        voices.append({"id": v, "engine": "silma", "gender": "clone",
                       "language": lang})
    return voices


_voice_cache: list[dict] | None = None


def _known_voice_ids() -> set[str]:
    """Cached voice ids.

    Discovery resolves an HF cache path and scans voices/, so it runs once
    rather than on every request (it also backs voice validation).
    """
    global _voice_cache
    if _voice_cache is None:
        _voice_cache = _all_voices()
    return {v["id"] for v in _voice_cache}


@app.get("/v1/tts/voices")
def tts_voices():
    import tts
    return {"voices": _all_voices(), "sample_rate": tts.SAMPLE_RATE}


@app.post("/v1/audio/speech")
def create_speech(
    text: str = Form(...),
    voice: str = Form("af_heart"),
    speed: float = Form(1.0),
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > 20000:
        raise HTTPException(status_code=400, detail="text too long (20000 chars max)")
    if voice not in _known_voice_ids():
        raise HTTPException(status_code=400,
                            detail=f"Unknown voice '{voice}'. See /v1/tts/voices.")
    try:
        wav = _tts_synthesize(text, voice, speed)
    except Exception as exc:
        err_id = uuid.uuid4().hex[:12]
        log.exception("TTS failed (%s)", err_id)
        raise HTTPException(status_code=500, detail=f"internal error (id {err_id})")
    return Response(content=wav, media_type="audio/wav")


# --- LLM cleanup providers -------------------------------------------------
# Keys and base URLs are configured server-side (see llm_cleanup.py). These
# discovery endpoints let clients choose among enabled providers and query
# the endpoint's current model list instead of relying on a static list.

@app.get("/v1/llm/providers")
def llm_providers():
    try:
        return llm_cleanup.available_providers()
    except llm_cleanup.LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/v1/llm/models")
def llm_models(provider: str = "", refresh: bool = False):
    try:
        return llm_cleanup.list_models(provider or None, refresh=refresh)
    except llm_cleanup.LLMConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except llm_cleanup.LLMUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception:
        log.exception("Could not list LLM models")
        raise HTTPException(status_code=502, detail="could not list models")


# --- Background jobs -------------------------------------------------------
# Long recordings exceed proxy timeouts (e.g. Cloudflare's 100 s), so clients
# can submit with background=true and poll for the result. One worker thread;
# jobs are serialized with synchronous requests via _model_lock, and VRAM is
# released after every job exactly like after every sync request.

_jobs: dict[str, dict] = {}
_job_queue: queue.Queue = queue.Queue()
# Liveness signals for /health: the worker bumps the heartbeat on every loop
# iteration, so a dead thread (previously invisible) becomes detectable.
_worker_heartbeat = time.time()
_last_job_completed = time.time()


@dataclass(frozen=True)
class JobSpec:
    """Everything one background transcription needs."""
    job_id: str
    audio_path: str
    model: str
    language: str | None
    punctuation: bool
    max_new_tokens: int
    timestamps: bool
    cleanup: bool
    translate_to: str | None
    llm_provider: str
    llm_model: str
    speak_voice: str
    speak_speed: float


def _set_job(job_id: str, **fields):
    """Update a job record and stamp it (used for TTL expiry)."""
    job = _jobs.setdefault(job_id, {})
    job.update(fields)
    job["updated_at"] = time.time()


def _expire_jobs():
    """Drop finished/failed jobs older than JOB_TTL_S.

    Job results hold the transcript (and any TTS audio), so without this the
    store grows without bound for the lifetime of the process.
    """
    cutoff = time.time() - JOB_TTL_S
    for job_id, job in list(_jobs.items()):
        if job.get("status") in ("done", "failed") and job.get("updated_at", 0) < cutoff:
            log.info("Expiring job %s (%s)", job_id, job.get("status"))
            del _jobs[job_id]


def _maybe_speak(result: dict, speak_voice: str, speak_speed: float):
    """Attach TTS audio of the final text (cleaned if cleanup ran, else raw)
    as result["tts"]["audio_wav_b64"]. Enabled when speak_voice is non-empty."""
    if not speak_voice:
        return
    # Validate against the known voices: tts_silma builds
    # os.path.join(VOICES_DIR, voice + ".wav") from this value, so an
    # unvalidated id such as "silma_../../.." would escape voices/.
    try:
        if speak_voice not in _known_voice_ids():
            result["tts"] = {"error": f"Unknown voice '{speak_voice}'. "
                                      "See /v1/tts/voices."}
            return
    except Exception as exc:
        log.warning("Voice list unavailable: %s", exc)
        result["tts"] = {"error": "voice list unavailable"}
        return
    text = result["text"]
    cleanup = result.get("cleanup")
    if cleanup and not cleanup.get("error") and cleanup.get("cleaned_text"):
        text = cleanup["cleaned_text"]
    try:
        wav = _tts_synthesize(text, speak_voice, speak_speed)
        result["tts"] = {"voice": speak_voice,
                         "audio_wav_b64": base64.b64encode(wav).decode()}
    except Exception as exc:
        log.exception("TTS failed")
        result["tts"] = {"error": str(exc)}


def _run_job(item: JobSpec):
    """Process one queued job.

    The GPU lock covers only load -> generate -> unload; LLM cleanup and TTS
    run after it is released so they cannot block every other request for
    minutes at a time.
    """
    global _last_job_completed
    _set_job(item.job_id, status="running")
    result = None
    try:
        with _model_lock:
            try:
                result = _transcribe(item.audio_path, item.model, item.language,
                                     item.punctuation, item.max_new_tokens,
                                     timestamps=item.timestamps)
            except HTTPException as exc:
                _set_job(item.job_id, status="failed", error=exc.detail)
                result = None
            except torch.OutOfMemoryError:
                # Observed failure mode on the 4 GB card: free VRAM and retry
                # with the smallest mini-batch before giving up.
                _free_vram()
                try:
                    result = _transcribe(item.audio_path, item.model, item.language,
                                         item.punctuation, item.max_new_tokens,
                                         chunk_batch=1, timestamps=item.timestamps)
                except torch.OutOfMemoryError:
                    log.exception("Job %s failed (CUDA OOM)", item.job_id)
                    _set_job(item.job_id, status="failed",
                             error="GPU memory exhausted; retry with a shorter "
                                   "file or a lower ASR_CHUNK_BATCH")
                    result = None
            except Exception as exc:
                err_id = uuid.uuid4().hex[:12]
                log.exception("Job %s failed (%s)", item.job_id, err_id)
                _set_job(item.job_id, status="failed", error=f"internal error (id {err_id})")
                result = None
            # Except-block names are cleared before this runs, so no live
            # traceback pins the model in memory here.
            unload_models()

        if result is not None:
            if item.cleanup:
                # ASR model is out of VRAM now; the cleanup model gets the GPU.
                try:
                    llm = llm_cleanup.resolve_config(item.llm_provider or None,
                                                     item.llm_model or None)
                except Exception as exc:
                    log.warning("LLM cleanup configuration failed: %s", exc)
                    result["cleanup"] = {"error": str(exc)}
                else:
                    result["cleanup"] = run_cleanup(result["text"],
                                                    item.translate_to, llm)
            _maybe_speak(result, item.speak_voice, item.speak_speed)
            _set_job(item.job_id, status="done", result=result)
    finally:
        try:
            os.unlink(item.audio_path)
        except OSError:
            log.warning("Could not unlink %s", item.audio_path)
        _last_job_completed = time.time()


def _job_worker():
    """Single background worker.

    The loop body is fully guarded: previously an exception raised outside the
    inner try (e.g. os.unlink) escaped the loop and killed the only worker,
    leaving every later job stuck in "queued" forever with no error anywhere.
    """
    global _worker_heartbeat
    while True:
        item = _job_queue.get()
        _worker_heartbeat = time.time()
        try:
            _run_job(item)
        except Exception:
            job_id = getattr(item, "job_id", "?")
            log.exception("Job %s failed unexpectedly", job_id)
            try:
                _set_job(job_id, status="failed", error="internal error")
            except Exception:
                pass
        finally:
            _job_queue.task_done()


threading.Thread(target=_job_worker, daemon=True, name="asr-jobs").start()


@app.get("/v1/audio/transcriptions/jobs/{job_id}")
def job_status(job_id: str):
    _expire_jobs()  # natural sweep point: polling clients keep this path warm
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
    return {"job_id": job_id, **job}


@app.delete("/v1/audio/transcriptions/jobs/{job_id}")
def delete_job(job_id: str):
    """Drop a job and its transcript immediately (instead of waiting for TTL)."""
    if _jobs.pop(job_id, None) is None:
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
    return {"job_id": job_id, "deleted": True}


@app.post("/v1/audio/transcriptions")
def transcribe(
    file: UploadFile = File(...),
    model: str = Form(_default_model),
    language: str | None = Form(None),
    punctuation: bool = Form(True),
    max_new_tokens: int = Form(256),
    background: bool = Form(False),
    timestamps: bool = Form(False),
    cleanup: bool = Form(False),
    translate_lang: str = Form(""),
    translate_modifier: str = Form(""),
    llm_provider: str = Form(""),
    llm_model: str = Form(""),
    speak_voice: str = Form(""),
    speak_speed: float = Form(1.0),
):
    if model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {sorted(MODELS)}",
        )
    if not 16 <= max_new_tokens <= 1024:
        raise HTTPException(status_code=400,
                            detail="max_new_tokens must be between 16 and 1024")
    if language is not None:
        language = language.strip()[:32]
        if language and not re.fullmatch(r"[A-Za-z+\-]{2,32}", language):
            raise HTTPException(status_code=400, detail=f"Invalid language '{language}'")
    if speak_voice and speak_voice not in _known_voice_ids():
        raise HTTPException(status_code=400,
                            detail=f"Unknown voice '{speak_voice}'. See /v1/tts/voices.")
    translate_to = _translate_target(translate_lang, translate_modifier) if cleanup else None
    # Resolve provider/model before accepting the upload so a bad selection
    # fails fast instead of after minutes of ASR work. A transient upstream
    # failure (e.g. Ollama is down while its model list is being discovered)
    # is deferred: cleanup retries after ASR and reports cleanup.error rather
    # than failing the transcription.
    llm_cfg = None
    if cleanup:
        try:
            llm_cfg = llm_cleanup.resolve_config(llm_provider or None,
                                                 llm_model or None)
        except llm_cleanup.LLMConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except llm_cleanup.LLMUpstreamError as exc:
            log.warning("Could not resolve LLM cleanup config yet: %s", exc)
    audio_path = _save_upload(file)

    if background:
        # Return immediately; the client polls /v1/audio/transcriptions/jobs/<id>.
        _expire_jobs()
        if _job_queue.qsize() >= MAX_QUEUED_JOBS:
            _discard(audio_path)
            raise HTTPException(
                status_code=503,
                detail=f"Too many queued jobs (limit {MAX_QUEUED_JOBS}); retry shortly.",
            )
        job_id = uuid.uuid4().hex
        _set_job(job_id, status="queued", model=model)
        _job_queue.put(JobSpec(
            job_id=job_id,
            audio_path=audio_path,
            model=model,
            language=language,
            punctuation=punctuation,
            max_new_tokens=max_new_tokens,
            timestamps=timestamps,
            cleanup=cleanup,
            translate_to=translate_to,
            llm_provider=llm_cfg.provider if llm_cfg else llm_provider,
            llm_model=llm_cfg.model if llm_cfg else llm_model,
            speak_voice=speak_voice,
            speak_speed=speak_speed,
        ))
        return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued", "model": model})

    # The load -> generate -> unload lifecycle runs under one lock: two models
    # are never in VRAM at once, and a model is never unloaded mid-generation.
    # LLM cleanup and TTS must happen *outside* it — they can take minutes and
    # would otherwise block every other request.
    with _model_lock:
        # Capture failures as plain data and re-raise afterwards: while an
        # exception is in flight its traceback keeps _transcribe's frame (and
        # thus the model) alive, which would defeat the VRAM release below.
        try:
            result = _transcribe(audio_path, model, language, punctuation,
                                 max_new_tokens, timestamps=timestamps)
        except HTTPException as exc:
            status_code, detail = exc.status_code, exc.detail
        except torch.OutOfMemoryError:
            # Real, observed failure on the 4 GB card: release VRAM and retry
            # with the smallest mini-batch before admitting defeat.
            _free_vram()
            try:
                result = _transcribe(audio_path, model, language, punctuation,
                                     max_new_tokens, chunk_batch=1,
                                     timestamps=timestamps)
            except torch.OutOfMemoryError:
                log.exception("Transcription failed (CUDA OOM)")
                status_code, detail = (503, "GPU memory exhausted; retry with a "
                                            "shorter file or a lower ASR_CHUNK_BATCH")
            except Exception as exc:
                err_id = uuid.uuid4().hex[:12]
                log.exception("Transcription failed (%s)", err_id)
                status_code, detail = 500, f"internal error (id {err_id})"
            else:
                status_code, detail = None, None
        except Exception as exc:
            # Log the detail, return only a correlation id.
            err_id = uuid.uuid4().hex[:12]
            log.exception("Transcription failed (%s)", err_id)
            status_code, detail = 500, f"internal error (id {err_id})"
        else:
            status_code, detail = None, None

        # Evict everything from VRAM at the end of every request so co-located
        # GPU processes (e.g. ollama) can use it, and no stale model can linger
        # when the next request switches models. No local in this frame (nor a
        # live traceback) references the model, so its tensors can be freed.
        unload_models()

    _discard(audio_path)

    if status_code is None:
        if cleanup:
            # ASR model is out of VRAM now; the cleanup model gets the GPU.
            # llm_cfg is None when the provider could not be queried during
            # validation (e.g. Ollama was briefly down), so resolve again here
            # and report a cleanup error instead of losing the transcript.
            try:
                cfg = llm_cfg or llm_cleanup.resolve_config(llm_provider or None,
                                                            llm_model or None)
            except Exception as exc:
                log.warning("LLM cleanup configuration failed: %s", exc)
                result["cleanup"] = {"error": str(exc)}
            else:
                result["cleanup"] = run_cleanup(result["text"], translate_to, cfg)
        _maybe_speak(result, speak_voice, speak_speed)
        return result
    raise HTTPException(status_code=status_code, detail=detail)
