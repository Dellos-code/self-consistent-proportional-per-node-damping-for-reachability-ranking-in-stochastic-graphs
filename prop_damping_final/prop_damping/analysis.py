"""Analysis helpers: loading results, selection on dev, paired statistics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
INF = 10 ** 6


def load_stage(stage, panel):
    d = RES / f"{stage}_{panel}"
    frames, metas, diags = [], [], {}
    for f in sorted(d.glob("[0-9][0-9][0-9][0-9].csv.gz")):
        iid = int(f.name[:4])
        frames.append(pd.read_csv(f))
        j = json.loads((d / f"{iid:04d}.json").read_text())
        m = dict(j["meta"])
        m["id"] = iid
        for k in ("alpha_rule", "alpha_scalar_B", "alpha_scalar_A", "seconds"):
            m[k] = j.get(k)
        pn = j.get("pernode")
        if pn:
            m["pn_fit"] = pn["fit_rho"]
            m["pn_fit_scalar"] = pn.get("fit_scalar_B")
            m["pn_accepted"] = pn["accepted"]
            m["pn_seconds"] = pn["seconds"]
            m["pn_frac_changed"] = pn["frac_changed"]
            for k, v in pn["corr"].items():
                m["pn_corr_" + k] = v
        metas.append(m)
        diags[iid] = j
    R = pd.concat(frames, ignore_index=True)
    M = pd.DataFrame(metas).set_index("id")
    return R, M, diags


def key_cols(R):
    R = R.copy()
    R["beta"] = R["beta"].fillna(-1.0)
    return R


def config_means(R, prefix, M=None, by=None):
    """Mean rho per configuration (method, T, beta, blend) for methods starting with prefix."""
    S = key_cols(R[R["method"].str.startswith(prefix)])
    if by is not None:
        S = S.join(M[[by]], on="id")
        g = S.groupby(["method", "T", "beta", "blend", by])["rho"].mean().unstack(by)
        return g
    return S.groupby(["method", "T", "beta", "blend"])["rho"].agg(["mean", "count"]).sort_values("mean", ascending=False)


def per_instance(R, method, T=0, beta=None, blend=None):
    S = key_cols(R[(R["method"] == method) & (R["T"] == T)])
    if beta is not None:
        S = S[S["beta"] == beta]
    if blend is not None:
        S = S[S["blend"] == blend]
    return S.set_index("id")["rho"]


def paired(a, b, n_boot=20000, seed=0):
    """Paired comparison a - b over common instances: mean, bootstrap CI, Wilcoxon, W/T/L."""
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).values
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    nz = d[np.abs(d) > 1e-9]
    p = wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan
    return dict(n=len(d), mean_a=float(a.loc[idx].mean()), mean_b=float(b.loc[idx].mean()),
                diff=float(d.mean()), lo=float(lo), hi=float(hi), p=float(p),
                win=int((d > 1e-9).sum()), tie=int((np.abs(d) <= 1e-9).sum()), loss=int((d < -1e-9).sum()))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj
