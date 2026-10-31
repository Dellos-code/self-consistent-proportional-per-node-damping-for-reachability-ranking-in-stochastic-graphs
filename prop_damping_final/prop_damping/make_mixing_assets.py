"""Generate compact, reproducible Markdown/CSV reporting assets."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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
    figdir = root / "figures_mixing"
    figdir.mkdir(exist_ok=True)
    top = summary.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top.config_id.astype(str), top.rho)
    plt.xlabel("Mean Spearman rho")
    plt.tight_layout()
    plt.savefig(figdir / f"{stage}_top20.png", dpi=160)
    plt.close()
    thesis = root / "thesis/mixing"
    thesis.mkdir(parents=True, exist_ok=True)
    text = (
        "\\chapter{Mixing strategies for the dual-operator hybrid}\\n"
        "\\label{ch:mixing}\\n\\n"
        "This chapter reports the preregistered mixing study. "
        "Selection was performed on the dev panel only, with independent "
        "evaluation ground truth used for scoring. The ranked configurations "
        "and exact configuration identifiers are stored in the accompanying "
        "CSV and JSONL files.\\n\\n"
        f"The {stage} screen evaluated {len(df)} configuration-instance rows. "
        f"The best observed mean Spearman correlation was "
        f"{summary.rho.iloc[0]:.4f}.\\n"
    )
    (thesis / f"{stage}_en.tex").write_text(text)
    (thesis / f"{stage}_gr.tex").write_text(
        "\\chapter{Στρατηγικές ανάμιξης για το υβρίδιο δύο τελεστών}\\n"
        "Η επιλογή έγινε αποκλειστικά στο dev και η αξιολόγηση χρησιμοποίησε "
        "ανεξάρτητη εκτίμηση της αλήθειας.\\n"
    )


if __name__ == "__main__":
    run(sys.argv[1])
