"""Machine-specific timing harness for frozen mixing configurations."""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from build_panels import DATA, load_instance
from exp_mixing import evaluate
import prop_damping_lib as L

ROOT = Path(__file__).resolve().parent


def run(panel: str):
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    frozen = json.loads((ROOT / "results/mixing/selected_mixing.json").read_text())
    paths = sorted((DATA / panel).glob("[0-9][0-9][0-9][0-9].npz"))[:9]
    rows = []
    for cfg in frozen["winners"]:
        method_cfg = cfg["config"]
        times = []
        for _ in range(5):
            for path in paths:
                inst, yA, yB, _ = load_instance(path)
                t0 = time.perf_counter()
                evaluate(inst, yA, yB, method_cfg)
                times.append(1000 * (time.perf_counter() - t0))
        rows.append({"winner": cfg["name"], "median_ms": statistics.median(times),
                     "cv": statistics.pstdev(times) / statistics.mean(times)})
    mc = []
    for k in (10, 20, 50, 100, 200):
        times, scores = [], []
        for path in paths:
            inst, yA, _, _ = load_instance(path)
            t0 = time.perf_counter()
            score = L.mc_reach(inst, k, L.stable_seed(inst.name, "bench_mc", k))
            times.append(1000 * (time.perf_counter() - t0))
            scores.append(L.spearman(score, L.ranks_of(yA)))
        mc.append({"method": f"MC_{k}", "mean_rho": float(sum(scores) / len(scores)),
                   "median_ms": statistics.median(times),
                   "cv": statistics.pstdev(times) / statistics.mean(times)})
    (ROOT / "results/mixing").mkdir(exist_ok=True, parents=True)
    (ROOT / "results/mixing" / f"runtime_{panel}.json").write_text(
        json.dumps({"winners": rows, "mc": mc}, indent=2))


if __name__ == "__main__":
    run(sys.argv[1])
