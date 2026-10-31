"""Generate compact, reproducible Markdown/CSV reporting assets."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def run(stage: str):
    root = Path(__file__).resolve().parent
    src = root / "results/mixing" / stage
    files = list(src.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(src)
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    summary = df.groupby("config_id", as_index=False).rho.mean().sort_values("rho", ascending=False)
    out = root / "docs/mixing"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / f"{stage}_summary.csv", index=False)
    (out / f"{stage}_summary.md").write_text(summary.head(20).to_markdown(index=False) + "\n")


if __name__ == "__main__":
    run(sys.argv[1])
