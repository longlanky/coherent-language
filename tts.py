"""Kokoro-82M text-to-speech on this host's Python 3.14 environment.

The kokoro/misaki packages declare Python <3.13 and misaki's English G2P needs
spacy (no py3.14 build), so this module routes ALL languages — English
included — through misaki's espeak-based G2P, which works everywhere. kokoro
and misaki are installed with --no-deps --ignore-requires-python; the missing
`misaki.en` module (spacy-bound) is stubbed so `kokoro.pipeline` imports.

Voice id prefix determines language: a/b→English (US/GB), e→es, f→fr, h→hi,
i→it, j→ja*, p→pt-br, z→zh* (* need misaki[ja]/[zh] extras if importable).
Synthesis always runs on CPU: the 82M model is near-real-time there, and the
GPU stays free for the ASR models.
"""

import logging
import re
import sys
import types

import numpy as np

log = logging.getLogger("cohere-asr.tts")

# --- shims: make kokoro importable without spacy ---------------------------
import misaki  # noqa: E402  (real package; only its .en submodule is spacy-bound)

_en_stub = types.ModuleType("misaki.en")


class _UnavailableG2P:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("misaki.en requires spacy (unavailable on Python 3.14); "
                           "this host routes English through the espeak G2P instead")


class _MToken:  # only referenced for type hints in kokoro.pipeline
    pass


_en_stub.G2P = _UnavailableG2P
_en_stub.MToken = _MToken
sys.modules["misaki.en"] = _en_stub
misaki.en = _en_stub

import torch  # noqa: E402
import kokoro.pipeline  # noqa: E402
from kokoro import KPipeline  # noqa: E402

# Custom lang codes routing English through EspeakG2P (see pipeline.py:123).
kokoro.pipeline.LANG_CODES["x"] = "en-us"
kokoro.pipeline.LANG_CODES["y"] = "en-gb"

VOICE_LANG = {
    "a": "x", "b": "y",  # English via espeak shim
    "e": "e", "f": "f", "h": "h", "i": "i", "j": "j", "p": "p", "z": "z",
}

SAMPLE_RATE = 24000
_model = None
_pipelines: dict[str, KPipeline] = {}


def _get_model():
    global _model
    if _model is None:
        from kokoro import KModel
        log.info("Loading Kokoro-82M TTS model (CPU) ...")
        _model = KModel().to("cpu").eval()
    return _model


def _get_pipeline(lang_code: str) -> KPipeline:
    if lang_code not in _pipelines:
        _pipelines[lang_code] = KPipeline(lang_code=lang_code, model=_get_model())
    return _pipelines[lang_code]


def available_voices() -> list[str]:
    """Voice ids present in the local HF cache snapshot."""
    from huggingface_hub import hf_hub_download
    import glob
    import os
    path = hf_hub_download("hexgrad/Kokoro-82M", filename="voices/af_heart.pt")
    voices_dir = os.path.dirname(path)
    return sorted(os.path.splitext(f)[0] for f in os.listdir(voices_dir) if f.endswith(".pt"))


def _split_text(text: str, max_len: int = 400) -> list[str]:
    """Split into sentence-ish chunks (espeak path truncates at 510 phonemes)."""
    parts = re.split(r"(?<=[.!?؟،;:\n])\s+", text.strip())
    chunks, cur = [], ""
    for part in parts:
        if len(cur) + len(part) + 1 > max_len and cur:
            chunks.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}".strip()
    # hard-wrap any remaining over-long chunk
    out = []
    for c in chunks + ([cur] if cur else []):
        out.extend(c[i:i + max_len] for i in range(0, len(c), max_len)) if len(c) > max_len else out.append(c)
    return [c for c in out if c]


def synthesize(text: str, voice: str = "af_heart", speed: float = 1.0) -> np.ndarray:
    """Text -> 24 kHz mono float32 waveform."""
    prefix = voice[0]
    if prefix not in VOICE_LANG:
        raise ValueError(f"Unknown voice '{voice}'")
    pipeline = _get_pipeline(VOICE_LANG[prefix])
    speed = min(max(float(speed), 0.5), 2.0)
    audio = []
    for chunk in _split_text(text):
        for result in pipeline(chunk, voice=voice, speed=speed):
            if result.audio is not None:
                audio.append(result.audio.cpu().numpy())
    if not audio:
        raise ValueError("TTS produced no audio (empty or unsupported text?)")
    return np.concatenate(audio)


def synthesize_wav_bytes(text: str, voice: str = "af_heart", speed: float = 1.0) -> bytes:
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, synthesize(text, voice, speed), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()
