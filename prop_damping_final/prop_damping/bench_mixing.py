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
    (ROOT / "results/mixing").mkdir(exist_ok=True, parents=True)
    (ROOT / "results/mixing" / f"runtime_{panel}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run(sys.argv[1])
