"""SILMA TTS (F5-TTS based bilingual Arabic/English voice cloning) wrapper.

Two py3.14 shims are applied before importing silma_tts:
- nemo_text_processing's Normalizer (needs pynini, which can't build on
  py3.14) is replaced with a pass-through — we lose number/date
  normalization, synthesis is unaffected otherwise.
- torchaudio.load (torchcodec-backed in torchaudio 2.11+, and torchcodec's
  native lib won't load on this host) is replaced with a soundfile reader.

Voices are reference-audio clones: each entry in voices/ is
<voice_id>.wav + <voice_id>.txt (its transcript).
"""

import logging
import os
import sys
import types

import numpy as np

log = logging.getLogger("cohere-asr.tts.silma")

_HERE = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(_HERE, "voices")
SAMPLE_RATE = 24000

# --- shim: NeMo normalizer (pass-through) ----------------------------------
_nemo = types.ModuleType("nemo_text_processing")
_ntn = types.ModuleType("nemo_text_processing.text_normalization")
_norm = types.ModuleType("nemo_text_processing.text_normalization.normalize")


class _Normalizer:
    def __init__(self, **kwargs):
        pass

    def normalize(self, text, **kwargs):
        return text


_norm.Normalizer = _Normalizer
_ntn.normalize = _norm
_nemo.text_normalization = _ntn
sys.modules["nemo_text_processing"] = _nemo
sys.modules["nemo_text_processing.text_normalization"] = _ntn
sys.modules["nemo_text_processing.text_normalization.normalize"] = _norm

# --- shim: torchaudio.load via soundfile -----------------------------------
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


def _ta_load(path, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _ta_load

_model = None


def _get_model():
    global _model
    if _model is None:
        from silma_tts.api import SilmaTTS
        device = os.environ.get("SILMA_TTS_DEVICE", "cpu")
        log.info("Loading SILMA TTS model (%s) ...", device)
        _model = SilmaTTS(device=device)
    return _model


def available_voices() -> list[str]:
    if not os.path.isdir(VOICES_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(VOICES_DIR)
                  if f.endswith(".wav") and os.path.exists(os.path.join(VOICES_DIR, f[:-4] + ".txt")))


def synthesize_wav_bytes(text: str, voice: str = "silma_ar_1", speed: float = 1.0) -> bytes:
    import io
    import tempfile

    wav_path = os.path.join(VOICES_DIR, voice + ".wav")
    txt_path = os.path.join(VOICES_DIR, voice + ".txt")
    if not os.path.exists(wav_path):
        raise ValueError(f"Unknown SILMA voice '{voice}'. Available: {available_voices()}")
    with open(txt_path) as f:
        ref_text = f.read().strip()

    model = _get_model()
    speed = min(max(float(speed), 0.5), 2.0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        out_path = tmp.name
    try:
        wav, sr, _spec = model.infer(
            ref_file=wav_path,
            ref_text=ref_text,
            gen_text=text.strip(),
            file_wave=out_path,
            seed=42,
            speed=speed,
        )
        wav = np.asarray(wav, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, wav, int(sr), format="WAV", subtype="PCM_16")
        return buf.getvalue()
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)
