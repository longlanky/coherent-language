export ASR_DTYPE="float16"
export ASR_DEVICE="cuda"
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments"
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000

