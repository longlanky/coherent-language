"""Benchmark ASR_CHUNK_BATCH on the GPU daemon with the 24-min / 42-chunk file.

For each batch value: restart the :8000 daemon with --chunk-batch N, run the
file as a background job, record wall/load/inference time (log timestamps) and
peak VRAM (nvidia-smi sampler thread). Writes chunk_bench_summary.txt.
"""

import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

FILE = "/home/nolan/Desktop/ASR_tests/khutbah/khutbah_firstkhutba.mp3"
BASE = "http://127.0.0.1:8000"
LOGFILE = "/home/nolan/cohere-asr/asr-server.8000.log"
SERVER = "/home/nolan/cohere-asr/asr-server"
MODEL = "cohere-transcribe-arabic-07-2026-int8enc"
BATCHES = [1, 2, 4, 8, 16, 24, 32]
OUT = Path(__file__).parent

LOG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO (Loading|Loaded|Transcribed)")

_peak = 0
_sampling = False


def vram_sampler():
    global _peak
    while _sampling:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            _peak = max(_peak, int(out))
        except Exception:
            pass
        time.sleep(0.5)


def log_times(since):
    marks = {}
    for line in Path(LOGFILE).read_text(errors="replace").splitlines():
        m = LOG_RE.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
            if ts >= since and m.group(2) not in marks:
                marks[m.group(2)] = ts
    return marks


def restart(batch):
    subprocess.run([SERVER, "restart", "--device", "cuda", "--gpu", "0",
                    "--default-model", MODEL, "--chunk-batch", str(batch)],
                   check=True, capture_output=True, text=True)


def run_job():
    t0 = time.time()
    with open(FILE, "rb") as f:
        r = requests.post(f"{BASE}/v1/audio/transcriptions",
                          files={"file": (Path(FILE).name, f)},
                          data={"model": MODEL, "language": "ar", "background": "true"},
                          timeout=120)
    job_id = r.json()["job_id"]
    while True:
        job = requests.get(f"{BASE}/v1/audio/transcriptions/jobs/{job_id}", timeout=30).json()
        if job["status"] in ("done", "failed"):
            return job, time.time() - t0
        time.sleep(3)


def main():
    global _peak, _sampling
    rows = []
    for batch in BATCHES:
        print(f"--- chunk_batch={batch}: restarting daemon", flush=True)
        restart(batch)
        _peak = 0
        _sampling = True
        sampler = threading.Thread(target=vram_sampler, daemon=True)
        sampler.start()
        t_start = datetime.now()
        try:
            job, wall = run_job()
        finally:
            _sampling = False
            sampler.join()
        time.sleep(1)
        marks = log_times(t_start)
        infer = (marks["Transcribed"] - marks["Loaded"]).total_seconds() if {"Loaded", "Transcribed"} <= marks.keys() else None
        if job["status"] == "done":
            rows.append((batch, "ok", round(wall, 1), infer, _peak))
            print(f"batch={batch}: OK wall={wall:.0f}s infer={infer:.0f}s peak_vram={_peak}MiB", flush=True)
        else:
            rows.append((batch, str(job.get("error"))[:80], round(wall, 1), None, _peak))
            print(f"batch={batch}: FAILED {str(job.get('error'))[:120]}", flush=True)

    lines = ["chunk_batch | status | wall_s | infer_s | peak_vram_MiB",
             "------------+--------+--------+---------+---------------"]
    for batch, status, wall, infer, peak in rows:
        lines.append(f"{batch:11d} | {status[:6]:6s} | {wall:6.1f} | {infer if infer is not None else '-':>7} | {peak}")
    summary = "\n".join(lines)
    (OUT / "chunk_bench_summary.txt").write_text(summary + "\n")
    print("\n" + summary, flush=True)
    # leave the daemon on the winning/default config
    restart(4)


if __name__ == "__main__":
    main()
