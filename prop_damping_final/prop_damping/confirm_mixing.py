"""Evaluate a previously frozen mixing configuration on held-out panels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from exp_mixing import evaluate
from build_panels import DATA, load_instance

ROOT = Path(__file__).resolve().parent


def run(panel: str):
    frozen = json.loads((ROOT / "results/mixing/selected_mixing.json").read_text())
    rows = []
    for path in sorted((DATA / panel).glob("[0-9][0-9][0-9][0-9].npz")):
        inst, yA, yB, _ = load_instance(path)
        for cfg in frozen["winners"]:
            rows.append({"id": int(path.stem), "winner": cfg["name"],
                         "rho": evaluate(inst, yA, yB, cfg)})
    out = ROOT / "results/mixing" / f"confirm_{panel}.csv.gz"
    pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")


if __name__ == "__main__":
    run(sys.argv[1])
