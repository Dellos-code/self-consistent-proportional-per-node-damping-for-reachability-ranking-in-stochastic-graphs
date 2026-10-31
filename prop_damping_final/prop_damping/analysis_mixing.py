"""Statistical summaries for mixing result files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a) - np.asarray(b)
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    return {"mean_diff": float(d.mean()), "w": int(np.sum(d > 0)),
            "t": int(np.sum(d < 0)), "l": int(np.sum(d == 0)), "p": p}


def summarise(path: str):
    df = pd.concat([pd.read_csv(p) for p in Path(path).glob("*.csv.gz")], ignore_index=True)
    out = df.groupby("config_id", as_index=False)["rho"].agg(["mean", "count"]).reset_index()
    out.to_csv(Path(path) / "summary.csv", index=False)
    return out


if __name__ == "__main__":
    summarise(sys.argv[1])
