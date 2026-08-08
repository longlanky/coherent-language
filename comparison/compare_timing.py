"""Runtime + accuracy comparison: chunked GPU (:8000) vs single-batch CPU (:8002).

Runs each sample sequentially on both modes (no queue contention), records
wall time per job, and splits model-load vs inference time by parsing the
instance's log timestamps (Loading.../Loaded.../Transcribed...).
Writes comparison/timing_summary.txt and refreshes the transcript .txt files.
"""

import difflib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests

FILES = {
    "opening_88s": "/home/nolan/Desktop/ASR_tests/khutbah/khutbah_opening_supplication.mp3",
    "second_khutba_103s": "/home/nolan/Desktop/ASR_tests/khutbah/khutbah_SecondKhutbaAndDua.mp3",
    "convo_159s": "/home/nolan/Desktop/ASR_tests/ar_convo_beginner.mp3",
    "first_khutba_1465s": "/home/nolan/Desktop/ASR_tests/khutbah/khutbah_firstkhutba.mp3",
}
MODES = {
    "gpu_chunked": ("http://127.0.0.1:8000", "/home/nolan/cohere-asr/asr-server.8000.log"),
    "cpu_single": ("http://127.0.0.1:8002", "/home/nolan/cohere-asr/asr-server.8002.log"),
}
MODEL = "cohere-transcribe-arabic-07-2026-int8enc"
OUT = Path(__file__).parent

LOG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO (Loading|Loaded|Transcribed)")


def log_times(logfile, since: datetime):
    """Last Loading/Loaded/Transcribed timestamps at or after `since`."""
    marks = {}
    for line in Path(logfile).read_text(errors="replace").splitlines():
        m = LOG_RE.match(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
        if ts >= since and m.group(2) not in marks:
            marks[m.group(2)] = ts
    return marks


def run_job(base, path):
    t0 = time.time()
    with open(path, "rb") as f:
        r = requests.post(
            f"{base}/v1/audio/transcriptions",
            files={"file": (Path(path).name, f)},
            data={"model": MODEL, "language": "ar", "background": "true"},
            timeout=120,
        )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    while True:
        job = requests.get(f"{base}/v1/audio/transcriptions/jobs/{job_id}", timeout=30).json()
        if job["status"] in ("done", "failed"):
            return job, time.time() - t0
        time.sleep(2)


def main():
    results = {}
    for name, path in FILES.items():
        for mode, (base, logfile) in MODES.items():
            t_start = datetime.now()
            job, wall = run_job(base, path)
            if job["status"] != "done":
                results[(name, mode)] = {"error": str(job.get("error"))[:200]}
                print(f"FAILED {name} [{mode}]", flush=True)
                continue
            text = job["result"]["text"]
            (OUT / f"{name}.{mode}.txt").write_text(text)
            time.sleep(1)  # let the server flush log lines
            marks = log_times(logfile, t_start)
            load_s = (marks["Loaded"] - marks["Loading"]).total_seconds() if {"Loading", "Loaded"} <= marks.keys() else None
            infer_s = (marks["Transcribed"] - marks["Loaded"]).total_seconds() if {"Loaded", "Transcribed"} <= marks.keys() else None
            results[(name, mode)] = {
                "wall_s": round(wall, 1), "load_s": load_s, "infer_s": infer_s,
                "chars": len(text), "words": len(text.split()),
                "duration_s": job["result"]["duration_s"],
            }
            print(f"done {name} [{mode}]: {results[(name, mode)]}", flush=True)

    lines = []
    for name in FILES:
        g, c = results.get((name, "gpu_chunked"), {}), results.get((name, "cpu_single"), {})
        lines.append(f"=== {name} (audio {g.get('duration_s', '?')}s) ===")
        if "error" in g or "error" in c:
            lines.append(f"  error: gpu={g.get('error')} cpu={c.get('error')}")
            continue
        lines.append(f"  GPU chunked : wall {g['wall_s']:7.1f}s  (load {g['load_s']:.1f}s + inference {g['infer_s']:.1f}s)")
        lines.append(f"  CPU single  : wall {c['wall_s']:7.1f}s  (load {c['load_s']:.1f}s + inference {c['infer_s']:.1f}s)")
        lines.append(f"  inference speedup GPU vs CPU: {c['infer_s'] / max(g['infer_s'], 0.01):.2f}x")
        a = (OUT / f"{name}.gpu_chunked.txt").read_text()
        b = (OUT / f"{name}.cpu_single.txt").read_text()
        wr = difflib.SequenceMatcher(None, a.split(), b.split()).ratio()
        lines.append(f"  word similarity: {wr:.4f} ({len(a.split())} vs {len(b.split())} words)")
    summary = "\n".join(lines)
    (OUT / "timing_summary.txt").write_text(summary + "\n")
    (OUT / "timing_raw.json").write_text(json.dumps({f"{n}|{m}": v for (n, m), v in results.items()}, indent=2))
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
