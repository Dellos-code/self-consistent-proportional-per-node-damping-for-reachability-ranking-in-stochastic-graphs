"""Statistical summaries for mixing result files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
from select_mixing import holm

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


def confirm(path: str, out_path: str | None = None):
    """Produce paired contrasts and Holm-adjusted p-values for a confirmation file."""
    df = pd.read_csv(path)
    piv = df.pivot_table(index="id", columns="method", values="rho", aggfunc="mean")
    winners = [c for c in piv.columns if c.startswith("W")]
    controls = [c for c in ("CTRL_lin0.9", "IRIE_exact", "rule_EW", "rule_WT",
                            "MP2", "fixed085", "oracle:scalar") if c in piv]
    rows, pvals = [], {}
    for w in winners:
        for c in controls:
            d = (piv[w] - piv[c]).dropna().to_numpy()
            nz = d[d != 0]
            p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
            name = f"{w}-vs-{c}"
            pvals[name] = p
            rows.append({"contrast": name, "mean_diff": float(d.mean()),
                         "w": int((d > 0).sum()), "t": int((d < 0).sum()),
                         "l": int((d == 0).sum()), "p": p})
    adj = holm(pvals)
    for row in rows:
        row["p_holm"] = adj[row["contrast"]]
    result = pd.DataFrame(rows)
    target = Path(out_path or str(path).replace(".csv.gz", "_stats.csv"))
    result.to_csv(target, index=False)
    if len(winners) and len(controls) >= 3:
        stat, p = friedmanchisquare(*(piv[c].dropna().to_numpy() for c in [winners[0], *controls[:9]]))
        target.with_suffix(".friedman.txt").write_text(f"stat={stat}\np={p}\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "confirm":
        confirm(sys.argv[2])
    else:
        summarise(sys.argv[1])
