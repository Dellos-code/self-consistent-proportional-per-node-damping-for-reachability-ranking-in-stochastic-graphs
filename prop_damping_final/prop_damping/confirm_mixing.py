"""Evaluate a previously frozen mixing configuration on held-out panels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import prop_damping_lib as L
import mixing_lib as M
from exp_mixing import evaluate
from build_panels import DATA, load_instance

ROOT = Path(__file__).resolve().parent


def run(panel: str):
    frozen = json.loads((ROOT / "results/mixing/selected_mixing.json").read_text())
    rows = []
    for path in sorted((DATA / panel).glob("[0-9][0-9][0-9][0-9].npz")):
        inst, yA, yB, _ = load_instance(path)
        for cfg in frozen["winners"]:
            method_cfg = cfg["config"]
            score = evaluate(inst, yA, yB, method_cfg, return_vector=True)
            rows.append({"id": int(path.stem), "method": cfg["name"],
                         **L.full_metrics(score, yA, L.ranks_of(yA))})
        baseline = M.baseline_scores(inst, yB, names=(
            "rule_EW", "rule_WT", "fixed085", "MP2", "IRIE_exact", "oracle:scalar"))
        for name in ("rule_EW", "rule_WT", "fixed085", "MP2", "IRIE_exact",
                     "oracle:scalar"):
            score = baseline[name]
            rows.append({"id": int(path.stem), "method": name,
                         **L.full_metrics(score, yA, L.ranks_of(yA))})
    out = ROOT / "results/mixing" / f"confirm_{panel}.csv.gz"
    pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")


if __name__ == "__main__":
    run(sys.argv[1])
