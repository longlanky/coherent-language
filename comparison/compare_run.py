"""Submit comparison jobs: chunked GPU (:8000) vs single-batch CPU (:8002),
same model/quantization (cohere-transcribe-arabic-07-2026-int8enc, language=ar).

Saves transcripts to comparison/{name}.{gpu_chunked,cpu_single}.txt and a
similarity summary to comparison/summary.txt.
"""

import difflib
import json
import time
from pathlib import Path

import requests

FILES = {
    "opening_88s": "/home/nolan/Desktop/ASR_tests/khutbah/khutbah_opening_supplication.mp3",
    "second_khutba_103s": "/home/nolan/Desktop/ASR_tests/khutbah/khutbah_SecondKhutbaAndDua.mp3",
    "convo_159s": "/home/nolan/Desktop/ASR_tests/ar_convo_beginner.mp3",
    "first_khutba_1465s": "/home/nolan/Desktop/ASR_tests/khutbah/khutbah_firstkhutba.mp3",
}
MODES = {
    "gpu_chunked": "http://127.0.0.1:8000",   # ASR_CHUNK_BATCH=4 (default)
    "cpu_single": "http://127.0.0.1:8002",    # ASR_CHUNK_BATCH=100000 (one batch)
}
MODEL = "cohere-transcribe-arabic-07-2026-int8enc"
OUT = Path(__file__).parent
POLL_EVERY = 20
MAX_WAIT_S = 4 * 3600


def submit(base, path):
    with open(path, "rb") as f:
        r = requests.post(
            f"{base}/v1/audio/transcriptions",
            files={"file": (Path(path).name, f)},
            data={"model": MODEL, "language": "ar", "background": "true"},
            timeout=120,
        )
    r.raise_for_status()
    return r.json()["job_id"]


def wait(base, job_id):
    deadline = time.time() + MAX_WAIT_S
    while time.time() < deadline:
        r = requests.get(f"{base}/v1/audio/transcriptions/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        job = r.json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(POLL_EVERY)
    return {"status": "timeout"}


def main():
    jobs = {}
    texts = {}
    for name, path in FILES.items():
        for mode, base in MODES.items():
            out = OUT / f"{name}.{mode}.txt"
            if out.exists():  # survived from a previous (crashed) run
                texts[(name, mode)] = out.read_text()
                print(f"cached {name} [{mode}], {len(out.read_text())} chars", flush=True)
                continue
            jobs[(name, mode)] = submit(base, path)
            print(f"submitted {name} [{mode}] -> {jobs[(name, mode)]}", flush=True)

    for (name, mode), job_id in jobs.items():
        base = MODES[mode]
        t0 = time.time()
        job = wait(base, job_id)
        elapsed = time.time() - t0
        if job["status"] == "done":
            text = job["result"]["text"]
            texts[(name, mode)] = text
            (OUT / f"{name}.{mode}.txt").write_text(text)
            print(f"done {name} [{mode}] in {elapsed:.0f}s, {len(text)} chars", flush=True)
        else:
            texts[(name, mode)] = None
            print(f"{job['status']} {name} [{mode}]: {str(job.get('error'))[:200]}", flush=True)

    lines = []
    for name in FILES:
        a, b = texts.get((name, "gpu_chunked")), texts.get((name, "cpu_single"))
        lines.append(f"=== {name} ===")
        if a is None or b is None:
            lines.append("  skipped (a job did not complete)")
            continue
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        wa, wb = a.split(), b.split()
        sm = difflib.SequenceMatcher(None, wa, wb)
        word_ratio = sm.ratio()
        lines.append(f"  char similarity: {ratio:.4f}   word similarity: {word_ratio:.4f}")
        lines.append(f"  lengths: gpu_chunked {len(a)} chars/{len(wa)} words, "
                     f"cpu_single {len(b)} chars/{len(wb)} words")
        if ratio < 1.0:
            diffs = [(op, " ".join(wa[i1:i2])[:80], " ".join(wb[j1:j2])[:80])
                     for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
            lines.append(f"  {len(diffs)} differing word spans; first 5:")
            for op, sa, sb in diffs[:5]:
                lines.append(f"    {op}: GPU[{sa}] CPU[{sb}]")
        else:
            lines.append("  IDENTICAL")
    summary = "\n".join(lines)
    (OUT / "summary.txt").write_text(summary + "\n")
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
