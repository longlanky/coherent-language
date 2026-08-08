export ASR_DTYPE="float32"
export ASR_DEVICE="cpu"
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
