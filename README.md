# Cohere Transcribe ASR API

An HTTP API for transcribing audio files with the locally cached
[Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
models from HuggingFace:

| Model | Repo | Quantization | Languages |
|---|---|---|---|
| `cohere-transcribe-03-2026` | `CohereLabs/cohere-transcribe-03-2026` | — (FP16) | en, fr, de, es, it, pt, nl, pl, el, ar, ja, zh, vi, ko |
| `cohere-transcribe-arabic-07-2026` | `CohereLabs/cohere-transcribe-arabic-07-2026` | — (FP16) | ar, en (incl. Arabic–English code-switching) |
| `cohere-transcribe-03-2026-int8enc` | `CohereLabs/cohere-transcribe-03-2026` | INT8 encoder / FP16 decoder | same as base |
| `cohere-transcribe-arabic-07-2026-int8enc` | `CohereLabs/cohere-transcribe-arabic-07-2026` | INT8 encoder / FP16 decoder | same as base |

All are 2B-parameter Conformer encoder-decoder models, served via
`transformers>=5.4.0` (`AutoProcessor` + `CohereAsrForConditionalGeneration`).
Weights are loaded from the local HuggingFace cache (`~/.cache/huggingface/hub/`)
— no downloads required. No authentication.

## Endpoints

### `POST /v1/audio/transcriptions`

Transcribes an uploaded audio file (OpenAI-style multipart form).
Long-form audio (>35 s) is chunked and reassembled automatically.

| Form field | Type | Default | Description |
|---|---|---|---|
| `file` | file | *required* | Audio file (wav, mp3, …; resampled to 16 kHz) |
| `model` | string | server default (see `ASR_DEFAULT_MODEL`) | Model id from the table above |
| `language` | string | `en` (base) / `ar` (Arabic model) | Language code; validated against the model's supported languages |
| `punctuation` | bool | `true` | `false` gives lower-cased output without punctuation |
| `max_new_tokens` | int | `256` | Generation cap |
| `background` | bool | `false` | `true` → return `202` + `job_id` immediately instead of waiting |

Response:

```json
{
  "text": " If not, there will be a big crisis between you and the European Parliament.",
  "model": "cohere-transcribe-03-2026",
  "language": "en",
  "duration_s": 5.44
}
```

Errors: `400` for unknown model, unsupported language, or undecodable audio; `500` for inference failures.

**Long recordings and proxy timeouts (Cloudflare's 100 s):** synchronous
requests only work for audio short enough to finish within the proxy limit.
For anything longer, submit with `background=true` — the request returns in
under a second — and poll the job endpoint:

```bash
# 1. Submit (returns immediately)
curl -X POST https://asr.ninefive.cc/v1/audio/transcriptions \
  -H "Cookie: CF_Authorization=$CF_JWT" \
  -F "file=@long_recording.mp3" -F "background=true" -F "language=ar"
# → {"job_id": "070c7b86…", "status": "queued"}

# 2. Poll until "done" (each poll is a fast request, no proxy timeout)
curl -H "Cookie: CF_Authorization=$CF_JWT" \
  https://asr.ninefive.cc/v1/audio/transcriptions/jobs/070c7b86…
# → {"status": "queued"|"running"|"done"|"failed", "result": {...}|"error": "..."}
```

Verified through Cloudflare with a 24-minute mp3 (42 chunks): submit 0.8 s,
job completes in the background, VRAM freed afterwards. Long-form generation
runs in mini-batches of `ASR_CHUNK_BATCH` chunks (default 4) so activation
memory stays bounded regardless of recording length.

Jobs are kept in process memory (lost on restart); one worker thread processes
them in FIFO order, serialized with synchronous requests.

### `GET /v1/audio/transcriptions/jobs/{job_id}`

Returns `{"job_id", "status", ...}` — `result` when `done`, `error` when `failed`.

### `GET /v1/models`

Lists registered models, default languages, and whether each is loaded.

### `GET /health`

Returns `{"status": "ok"}`.

## Usage

```bash
# English (default model)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "language=en"

# Arabic model
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=cohere-transcribe-arabic-07-2026" \
  -F "language=ar"

# No punctuation
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "language=en" -F "punctuation=false"
```

Interactive API docs (FastAPI/Swagger): http://localhost:8000/docs

## Client script (`./transcribe`)

Wraps the whole flow — file selection, upload, background polling (immune to
proxy timeouts), printing, and saving. **Zero dependencies** (Python standard
library only, any Python 3.8+): copy this one file to any machine and run it,
no venv or `pip install` needed. A GUI file picker (zenity) is used when no
path is given and available, otherwise a terminal prompt.

```bash
./transcribe sermon.mp3                  # GUI file picker if no path given (zenity)
./transcribe sermon.mp3 --language ar    # saves sermon.cohere-transcribe-arabic-07-2026-int8enc.txt
./transcribe sermon.mp3 --output out.txt --no-punctuation
./transcribe sermon.mp3 --server http://127.0.0.1:8000   # local server, no auth
```

- Defaults to the public endpoint `https://asr.ninefive.cc` (override with
  `--server` or `ASR_SERVER`; also `ASR_MODEL`, `ASR_LANGUAGE`).
- For the Cloudflare-Access endpoint it needs the `CF_Authorization` JWT via
  `--cookie`, the `CF_AUTHORIZATION` env var, or a `./cf.jwt` file (note: the
  cookie expires ~24 h after browser login). Localhost servers skip auth.
- Always submits with `background=true` and polls, so even hour-long files
  never hit the 100 s Cloudflare timeout. Progress dots while waiting;
  Ctrl+C aborts the client without killing the server-side job.
- The transcript is printed and saved next to the audio as
  `<name>.<model-id>.txt` unless `--output` is given.

## Running

The `asr-server` script manages a daemonized (nohup + pidfile) server in
userland — no systemd needed:

```bash
cd /home/nolan/cohere-asr
./asr-server start --device cuda --gpu 0 \
  --default-model cohere-transcribe-arabic-07-2026-int8enc
./asr-server status    # pid + /health + VRAM in use
./asr-server logs      # follow the server log (asr-server.log)
./asr-server stop      # SIGTERM, then SIGKILL if needed
./asr-server restart --device cpu --dtype float32 --port 8080
```

| Option | Env var | Default | Purpose |
|---|---|---|---|
| `--device cuda\|cpu` | `ASR_DEVICE` | `cuda` if available | Inference device |
| `--gpu N` | `CUDA_VISIBLE_DEVICES` | all visible | GPU selection |
| `--dtype …` | `ASR_DTYPE` | fp16 (GPU) / fp32 (CPU) | Model dtype |
| `--default-model ID` | `ASR_DEFAULT_MODEL` | `cohere-transcribe-03-2026` | Model when requests omit `model` — set an `-int8enc` id for quantized-by-default |
| `--chunk-batch N` | `ASR_CHUNK_BATCH` | `4` | Long-audio chunks per generate mini-batch (lower if OOM) |
| `--host` / `--port` | — | `0.0.0.0` / `8000` | Bind address |

Plain `uvicorn app:app --host 0.0.0.0 --port 8000` still works for
foreground/debug runs.

Every request (or background job) loads the model it needs, transcribes, and
then **completely evicts the model from VRAM** before finishing (verified:
VRAM returns to the ~85 MiB CUDA-context baseline after every request,
including error paths). This keeps the GPU free for co-located processes
(e.g. ollama) and makes stale-model eviction bugs impossible — two models are
never in VRAM at once. The trade-off: each request pays a model load
(~10–40 s depending on page cache; INT8 variants also re-quantize). The whole
load → generate → unload lifecycle runs under one lock, so requests are
serialized and a model is never freed mid-generation.

## Quantization (INT8 encoder / FP16 decoder)

The `-int8enc` variants reproduce the
[`FluidInference/cohere-transcribe-03-2026-coreml`](https://huggingface.co/FluidInference/cohere-transcribe-03-2026-coreml)
hybrid scheme natively in PyTorch: per-output-channel symmetric INT8 encoder
weights with FP16 activations; the decoder, embeddings, LayerNorms, and conv
layers stay FP16 (precision-sensitive and small). Implemented in `quant.py`
(`Int8Linear`), applied to every `nn.Linear` of the encoder at load time —
no extra dependencies and no calibration data needed.

**Why not serve the CoreML artifacts directly?** That repo contains Apple
CoreML packages (`.mlpackage`/`.mlmodelc`), and the CoreML runtime only exists
on macOS/iOS — it cannot execute on this Linux host. The files remain in the
HF cache for reference/Mac use; the API derives the same quantization from the
original FP16 weights instead.

Measured on the Quadro P1000:

| Variant | VRAM after load | EN demo transcript |
|---|---:|---|
| FP16 | ~3.9 GB | reference |
| `-int8enc` | ~2.6 GB | identical to FP16 |

Both `-int8enc` variants produced transcripts **identical** to their FP16
counterparts on the demo files (English and Arabic). Encoder weight memory
drops ~2× overall (INT8 vs FP16 on the encoder's linear layers).

## Setup (already done in `.venv/`)

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# Torch cu126 build — required for the Quadro P1000, see notes
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -r requirements.txt
```

## Hardware/software notes

- **GPU: Quadro P1000 (4 GB, Pascal / sm_61).** The stock PyTorch cu130 wheel
  ships no kernels for sm_61 — the **cu126 build is required**. The 2B model in
  fp16 (~4 GB) just fits: `device_map="auto"` places what fits on the GPU
  (observed ~3.9 GB used) and offloads any overflow to CPU RAM. CPU inference
  also works (`ASR_DEVICE=cpu`, fp32, needs ~8 GB RAM per model).
- **Per-request VRAM release.** Two 2B models (even INT8-quantized, ~2.6 GB
  each) plus inference temporaries do not fit in 4 GB together, and this GPU
  is shared with other processes (e.g. ollama) — so no model stays resident:
  VRAM is fully released after every request. One subtlety, if you touch this
  code: the unload must run in a frame that holds no references to the model,
  and not while an exception from inference is in flight (its traceback keeps
  the frame — and the model — alive). That is why `transcribe` is a thin
  wrapper around `_transcribe` that re-raises captured errors after unloading.
- **Triton native kernels: enabled.** Torch 2.13 routes some eager ops (e.g.
  `bmm`) through its native Triton kernels, which need a C compiler and the
  Python dev headers at runtime — `gcc` and `python3.14-dev` are installed on
  this host, so this path works. If you deploy elsewhere without them, set
  `TORCH_DISABLE_NATIVE_JIT=1` to fall back to standard aten kernels.
- Verified stack: Python 3.14.4, torch 2.13.0+cu126, transformers 5.14.1,
  accelerate, fastapi, uvicorn, librosa/soundfile.

## Verified results

- English demo (VoxPopuli, 5.4 s) →
  *" If not, there will be a big crisis between you and the European Parliament."*
  (identical for FP16 and `-int8enc`)
- Arabic demo (dialectal, code-switched, 3.6 s) →
  *" ففي الحالة دي المسألة دي يعني more safe"*
  (identical for FP16 and `-int8enc`)
- Error paths: unknown model → 400 with available ids; non-audio file → 400.
- **Chunked vs single-batch parity** (same `arabic-07-2026-int8enc` model, GPU
  mini-batches of 4 vs CPU single batch, 4 files incl. a 24-min/42-chunk one):
  3 of 4 byte-identical; the 4th differed in exactly one word
  (الغروزني. vs الغروزنة. — an fp16-vs-fp32 rounding artifact, not chunking).
  Repro scripts and transcripts in `comparison/`.
- **GPU vs CPU runtimes** (same files, inference time excl. ~5–8 s model load):
  the P1000 is only ~1.1× faster than this host's CPU on short clips, and on
  the 24-min file the CPU's single big batch *beats* 11 sequential GPU
  mini-batches (303 s vs 314 s, both ≈ 4.7× real-time). Details in
  `comparison/timing_summary.txt`.
- **`--chunk-batch` sweet spot: 4 (the default).** Benchmarked 1–32 on the
  24-min file (`comparison/chunk_bench_summary.txt`): inference time is flat
  (~316–328 s) from batch 1 through 8 — the workload is dominated by
  autoregressive decoder steps, so encoder batching buys nothing on the
  P1000 — while peak VRAM climbs 2.8 → 3.6 GB. Batch 4 is the fastest with
  comfortable headroom (3.1 GB peak); 16+ OOMs the 4 GB card. Drop to 1–2
  only if ollama needs a few hundred extra MB during a transcription.

## Limitations (from the model card)

- Best with a single pre-specified language; no automatic language detection.
- No timestamps, no speaker diarization.
- Eager to transcribe non-speech noise — a VAD/noise gate upstream helps on
  quiet audio.

## Project layout

```
cohere-asr/
├── app.py             # FastAPI service (single file)
├── quant.py           # INT8 encoder quantization (Int8Linear)
├── asr-server         # daemon manager (start/stop/status/logs)
├── transcribe         # CLI client (upload + poll + save)
├── comparison/        # benchmark scripts and transcripts
├── requirements.txt   # Python dependencies
├── .venv/             # virtualenv with the verified stack
├── asr-server.*.log   # daemon logs, per port (created at runtime)
├── asr-server.*.pid   # daemon pidfiles, per port (created at runtime)
└── README.md
```
