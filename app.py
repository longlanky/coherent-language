"""HTTP API for Cohere Transcribe ASR models.

Serves the locally cached HuggingFace models:
  - CohereLabs/cohere-transcribe-03-2026        (14 languages)
  - CohereLabs/cohere-transcribe-arabic-07-2026 (Arabic + English)

Run:  .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
"""

import gc
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import uuid

import requests
import torch
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cohere-asr")

SAMPLE_RATE = 16000

# Optional LLM post-cleanup via a local ollama instance. The prompts live in
# repo files next to this module.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://10.200.100.5:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")
_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "llm-post-processing-system-prompt.txt")) as _f:
    _CLEANUP_SYSTEM_PROMPT = _f.read()
with open(os.path.join(_here, "llm-post-processing-user-prompt.txt")) as _f:
    _CLEANUP_USER_TEMPLATE = _f.read()

MODELS = {
    "cohere-transcribe-03-2026": {
        "repo_id": "CohereLabs/cohere-transcribe-03-2026",
        "default_language": "en",
        "quantize_encoder": False,
    },
    "cohere-transcribe-arabic-07-2026": {
        "repo_id": "CohereLabs/cohere-transcribe-arabic-07-2026",
        "default_language": "ar",
        "quantize_encoder": False,
    },
    # INT8-encoder / FP16-decoder hybrids (FluidInference CoreML q8 scheme,
    # reproduced natively in PyTorch — CoreML itself can't run on Linux).
    "cohere-transcribe-03-2026-int8enc": {
        "repo_id": "CohereLabs/cohere-transcribe-03-2026",
        "default_language": "en",
        "quantize_encoder": True,
    },
    "cohere-transcribe-arabic-07-2026-int8enc": {
        "repo_id": "CohereLabs/cohere-transcribe-arabic-07-2026",
        "default_language": "ar",
        "quantize_encoder": True,
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


def _unload_ollama():
    """Force the ollama model out of RAM/VRAM (keep_alive=0)."""
    try:
        requests.post(f"{OLLAMA_HOST}/api/generate",
                      json={"model": OLLAMA_MODEL, "keep_alive": 0}, timeout=30)
        log.info("Unloaded %s from ollama", OLLAMA_MODEL)
    except Exception:
        log.warning("Failed to unload %s from ollama", OLLAMA_MODEL, exc_info=True)


def _ollama_chat(messages: list[dict]) -> str:
    """One JSON-mode chat call; returns the message content string."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages,
              "stream": False, "format": "json",
              "options": {"num_predict": -1}},  # never truncate mid-JSON
        timeout=900,  # long transcripts on a shared 4 GB GPU can take minutes
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_json(content: str) -> dict:
    # Tolerate stray markdown fences despite format:"json", raw control
    # characters inside strings (strict=False), and any leading/trailing
    # prose around the JSON object.
    cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    data = json.loads(cleaned, strict=False)
    if not isinstance(data, dict):
        raise ValueError(f"unexpected payload: {content[:200]}")
    return data


def _ollama_chat_json(messages: list[dict], attempts: int = 3) -> dict:
    """_ollama_chat + _parse_json with retries: the model intermittently
    emits malformed JSON (observed ~1 in 5 longer runs)."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        content = _ollama_chat(messages)
        try:
            return _parse_json(content)
        except (json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            log.warning("LLM returned invalid JSON (attempt %d/%d): %s", attempt, attempts, exc)
            log.warning("raw LLM content (first 500 chars): %s", content[:500])
    raise last_exc


def run_cleanup(raw_text: str, translate_to: str | None = None) -> dict:
    """Post-process a transcript with the ollama LLM.

    Returns {"cleaned_text", "changes", "warnings"} on success, or
    {"error": ...} on failure. The ollama model is always unloaded again.

    When translate_to is set (e.g. "French" or "Levantine Arabic") a second
    LLM pass translates the cleaned text — a single combined prompt proved
    unreliable (the model returned untranslated cleaned text). cleaned_text
    then holds the translation; changes/warnings come from the cleaning pass.
    """
    log.info("Running LLM cleanup with %s (%d chars, translate_to=%s)",
             OLLAMA_MODEL, len(raw_text), translate_to)
    try:
        data = _ollama_chat_json([
            {"role": "system", "content": _CLEANUP_SYSTEM_PROMPT},
            {"role": "user", "content": _CLEANUP_USER_TEMPLATE.replace("{raw_transcript}", raw_text)},
        ])
        if "cleaned_text" not in data:
            raise ValueError(f"unexpected cleanup payload: {str(data)[:200]}")
        data.setdefault("changes", [])
        data.setdefault("warnings", [])

        if translate_to:
            log.info("Translating cleaned text to %s", translate_to)
            tr_data = _ollama_chat_json([
                {"role": "system", "content": (
                    "You are a professional translator. Translate the user's text "
                    f"into {translate_to}. Translate faithfully and completely — do "
                    "not summarize, condense, or omit anything, and do not add "
                    "commentary. Return ONLY a JSON object of the form "
                    '{"translation": "..."}.')},
                {"role": "user", "content": data["cleaned_text"]},
            ])
            translated = tr_data.get("translation") or tr_data.get("cleaned_text")
            if not translated:
                raise ValueError(f"unexpected translation payload: {str(tr_data)[:200]}")
            data["cleaned_text"] = translated
            data["translation"] = {"target": translate_to}

        log.info("Cleanup done: %d changes, %d warnings",
                 len(data["changes"]), len(data["warnings"]))
        return data
    except Exception as exc:
        log.exception("LLM cleanup failed")
        return {"error": str(exc)}
    finally:
        _unload_ollama()


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
    log.info("Loading %s on %s (%s) ...", repo_id, _device, _dtype)
    processor = AutoProcessor.from_pretrained(repo_id)
    if MODELS[name]["quantize_encoder"]:
        from quant import quantize_encoder_int8

        # Load on CPU and quantize there first: the fp16 model fills
        # the 4 GB GPU, leaving no room for quantization temporaries.
        model = CohereAsrForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=_dtype
        )
        with torch.no_grad():
            quantize_encoder_int8(model)
        model = model.to(_device)
    elif _device == "cuda":
        model = CohereAsrForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=_dtype, device_map="auto"
        )
    else:
        model = CohereAsrForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=_dtype
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
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {
        "device": _device,
        "dtype": str(_dtype),
        "models": [
            {
                "id": name,
                "repo_id": info["repo_id"],
                "default_language": info["default_language"],
                "quantization": "int8-encoder/fp16-decoder" if info["quantize_encoder"] else None,
                "loaded": name in _loaded,
            }
            for name, info in MODELS.items()
        ],
    }


def _transcribe(audio_path: str, model: str, language: str | None,
                punctuation: bool, max_new_tokens: int) -> dict:
    """Run one transcription on a local audio file. Holds all model/tensor
    references in this frame, so they are released when it returns — the
    caller can then free the VRAM."""
    processor, asr_model = get_model(model)

    lang = language or MODELS[model]["default_language"]
    languages = getattr(asr_model.config, "supported_languages", None) or []
    if languages and lang not in languages:
        raise HTTPException(
            status_code=400,
            detail=f"Language '{lang}' not supported by {model}. Supported: {languages}",
        )

    try:
        audio = load_audio(audio_path, sampling_rate=SAMPLE_RATE)
    except Exception as first_exc:
        # Some containers (e.g. webm/opus from browser MediaRecorder) aren't
        # decodable by librosa — convert to 16 kHz mono wav with ffmpeg and retry.
        wav_path = audio_path + ".ffmpeg.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", audio_path,
                 "-ar", str(SAMPLE_RATE), "-ac", "1", wav_path],
                check=True, capture_output=True, timeout=300,
            )
            audio = load_audio(wav_path, sampling_rate=SAMPLE_RATE)
        except Exception:
            raise HTTPException(status_code=400,
                                detail=f"Could not decode audio: {first_exc}")
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    duration_s = len(audio) / SAMPLE_RATE
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
    chunk_batch = int(os.environ.get("ASR_CHUNK_BATCH", "4"))
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

    log.info("Transcribed %.1fs of audio with %s (%s)", duration_s, model, lang)
    return {
        "text": text,
        "model": model,
        "language": lang,
        "duration_s": round(duration_s, 2),
    }


def _save_upload(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        return tmp.name


def _translate_target(lang: str, modifier: str) -> str | None:
    """Combine the translate form fields into a target like "Levantine Arabic".
    Length-capped; returns None when no language was given."""
    lang, modifier = lang.strip()[:60], modifier.strip()[:40]
    if not lang:
        return None
    return f"{modifier} {lang}".strip() if modifier else lang


# --- Background jobs -------------------------------------------------------
# Long recordings exceed proxy timeouts (e.g. Cloudflare's 100 s), so clients
# can submit with background=true and poll for the result. One worker thread;
# jobs are serialized with synchronous requests via _model_lock, and VRAM is
# released after every job exactly like after every sync request.

_jobs: dict[str, dict] = {}
_job_queue: queue.Queue = queue.Queue()


def _job_worker():
    while True:
        (job_id, audio_path, model, language, punctuation, max_new_tokens,
         cleanup, translate_to) = _job_queue.get()
        _jobs[job_id]["status"] = "running"
        with _model_lock:
            try:
                result = _transcribe(audio_path, model, language, punctuation, max_new_tokens)
            except HTTPException as exc:
                _jobs[job_id].update(status="failed", error=exc.detail)
                result = None
            except Exception as exc:
                log.exception("Job %s failed", job_id)
                _jobs[job_id].update(status="failed", error=str(exc))
                result = None
            # Except-block names are cleared before this runs, so no live
            # traceback pins the model in memory here.
            unload_models()
            os.unlink(audio_path)
            if result is not None and cleanup:
                # ASR model is out of VRAM now; gemma gets the GPU to itself.
                result["cleanup"] = run_cleanup(result["text"], translate_to)
            if result is not None:
                _jobs[job_id].update(status="done", result=result)


threading.Thread(target=_job_worker, daemon=True, name="asr-jobs").start()


@app.get("/v1/audio/transcriptions/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
    return {"job_id": job_id, **job}


@app.post("/v1/audio/transcriptions")
def transcribe(
    file: UploadFile = File(...),
    model: str = Form(_default_model),
    language: str | None = Form(None),
    punctuation: bool = Form(True),
    max_new_tokens: int = Form(256),
    background: bool = Form(False),
    cleanup: bool = Form(False),
    translate_lang: str = Form(""),
    translate_modifier: str = Form(""),
):
    if model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {sorted(MODELS)}",
        )
    audio_path = _save_upload(file)
    translate_to = _translate_target(translate_lang, translate_modifier) if cleanup else None

    if background:
        # Return immediately; the client polls /v1/audio/transcriptions/jobs/<id>.
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"status": "queued", "model": model}
        _job_queue.put((job_id, audio_path, model, language, punctuation, max_new_tokens,
                        cleanup, translate_to))
        return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued", "model": model})

    # The whole request lifecycle runs under one lock: two models are never
    # in VRAM at once, and a model is never unloaded mid-generation.
    with _model_lock:
        # Capture failures as plain data and re-raise afterwards: while an
        # exception is in flight its traceback keeps _transcribe's frame (and
        # thus the model) alive, which would defeat the VRAM release below.
        try:
            result = _transcribe(audio_path, model, language, punctuation, max_new_tokens)
        except HTTPException as exc:
            status_code, detail = exc.status_code, exc.detail
        except Exception as exc:
            log.exception("Transcription failed")
            status_code, detail = 500, str(exc)
        else:
            status_code, detail = None, None

        # Evict everything from VRAM at the end of every request so co-located
        # GPU processes (e.g. ollama) can use it, and no stale model can linger
        # when the next request switches models. No local in this frame (nor a
        # live traceback) references the model, so its tensors can be freed.
        unload_models()
        os.unlink(audio_path)
        if status_code is None and cleanup:
            # ASR model is out of VRAM now; gemma gets the GPU to itself.
            result["cleanup"] = run_cleanup(result["text"], translate_to)

    if status_code is None:
        return result
    raise HTTPException(status_code=status_code, detail=detail)
