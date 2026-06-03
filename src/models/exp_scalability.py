"""Scalability analysis — CAMP recommendation/inference cost vs problem size.

ADD-ONLY measurement on the EXISTING trained per-institution policies (loaded
from ``models/camp_{uni}.zip`` — no retraining). Measures wall-clock inference
time and peak memory as a function of:

  * **problem size (number of students/samples)** — CAMP's deterministic
    sequential rollout is timed over increasing test-sample counts within one
    institution (Khalifa), giving wall-clock seconds and milliseconds/sample;
  * **candidate-course vocabulary / graph size** — the three institutions have
    different action-space (vocabulary) and prerequisite-graph sizes (Khalifa 59,
    AUS 61, UNC 74 courses); inference cost per sample is measured for each at a
    fixed sample count.

Wall-clock numbers are hardware-dependent measurements (not a locked result and
not compared against any committed metric). CPU-only; seed 42; British English.

Writes ``results/scalability.json``. No committed result value is read or
changed; this only times existing models.
"""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
from statistics import median
from typing import Any

import numpy as np
import psutil
from sb3_contrib import MaskablePPO

from src.models import camp
from src.models import task
from src.utils.seed import set_seed

RESULTS_DIR = task.BUNDLE / "results"
OUT_PATH = RESULTS_DIR / "scalability.json"

SEED = 42
PROBLEM_SIZES = [50, 100, 250, 500, 1000, 2000]  # within Khalifa test split
VOCAB_FIXED_N = 300                                # samples per institution for vocab sweep
N_REPEATS = 3                                      # median of repeats for timing stability


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def _time_scores(uni: str, model: MaskablePPO, sub: list[dict[str, Any]]) -> float:
    """Median wall-clock seconds to score ``sub`` (deterministic rollout)."""
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        camp._camp_scores(uni, model, sub)
        times.append(time.perf_counter() - t0)
    return float(median(times))


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples_test = task.load_samples("test")
    by_uni = {u: [s for s in samples_test if s["university"] == u] for u in camp.INSTITUTIONS}

    # --- model load cost (peak memory delta + time across the 3 policies) ---
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    models = {u: MaskablePPO.load(task.MODELS_DIR / f"camp_{u}.zip") for u in camp.INSTITUTIONS}
    load_seconds = time.perf_counter() - t0
    rss_after_load = _rss_mb()

    # --- environment build cost (fixed per-institution setup, amortised over N) ---
    t0 = time.perf_counter()
    env = camp.make_env("khalifa", "test", seed=SEED)
    env_build_seconds = time.perf_counter() - t0
    del env

    # --- (a) inference time + peak memory vs number of samples (Khalifa) ---
    # Timing is measured with tracing OFF (tracemalloc would inflate wall-clock
    # several-fold); peak Python memory is measured in a separate single pass.
    khal = by_uni["khalifa"]
    by_problem_size = []
    for n in PROBLEM_SIZES:
        if n > len(khal):
            continue
        sub = khal[:n]
        wall = _time_scores("khalifa", models["khalifa"], sub)  # clean timing
        tracemalloc.start()                                     # separate memory pass
        camp._camp_scores("khalifa", models["khalifa"], sub)
        peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        by_problem_size.append({
            "n_samples": n,
            "wall_seconds": round(wall, 4),
            "ms_per_sample": round(wall / n * 1000, 4),
            "peak_python_mb": round(peak_mb, 3),
            "process_rss_mb": round(_rss_mb(), 1),
        })
        print(f"  N={n:5d}  {wall:.3f}s  {wall/n*1000:.2f} ms/sample  peakPy {peak_mb:.1f} MB")

    # --- (b) inference time per sample vs vocabulary / graph size (3 institutions) ---
    by_vocabulary_size = []
    for u in camp.INSTITUTIONS:
        sub = by_uni[u][:VOCAB_FIXED_N]
        n = len(sub)
        wall = _time_scores(u, models[u], sub)
        by_vocabulary_size.append({
            "institution": u,
            "vocabulary_size": len(camp._institution_courses(u)),
            "n_samples": n,
            "wall_seconds": round(wall, 4),
            "ms_per_sample": round(wall / n * 1000, 4),
        })
        print(f"  {u:8s} vocab={len(camp._institution_courses(u))}  {wall/n*1000:.2f} ms/sample (N={n})")

    out = {
        "experiment": "CAMP inference scalability (existing policies; no retraining)",
        "seed": SEED,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "numpy": np.__version__,
            "device": "CPU-only",
            "note": "wall-clock timings are hardware-dependent measurements, not a "
            "locked result; rollout depth is camp._ROLLOUT_DEPTH="
            f"{camp._ROLLOUT_DEPTH}.",
        },
        "model_load": {
            "n_policies": len(models),
            "load_seconds": round(load_seconds, 4),
            "rss_before_mb": round(rss_before, 1),
            "rss_after_load_mb": round(rss_after_load, 1),
            "rss_delta_mb": round(rss_after_load - rss_before, 1),
        },
        "env_build_seconds_khalifa": round(env_build_seconds, 4),
        "by_problem_size": by_problem_size,
        "by_vocabulary_size": by_vocabulary_size,
    }
    with OUT_PATH.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
