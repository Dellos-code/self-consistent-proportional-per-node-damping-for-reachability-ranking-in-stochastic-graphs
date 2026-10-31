"""Proposition 2 diagnostics computed directly from ground truth (prior panel).

For each instance: kappa(x); the dynamic range of the proportionality ratio
r_i = (y*_i - x_i)/x_i; saturation of alpha_i = 1 - x_i/y*_i; and whether the
law keeps ORDINAL information among vertices sharing the same prior value
(Spearman between alpha^prop and y* restricted to non-seed vertices)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import prop_damping_lib as L
from build_panels import DATA, load_instance

rows = []
for f in sorted((DATA / "prior").glob("[0-9][0-9][0-9][0-9].npz")):
    inst, yA, yB, _ = load_instance(f)
    x, y = inst.x, yB
    r = np.maximum(y - x, 1e-300) / x
    a = 1 - x / y
    tau_rows = {}
    for tau in (0.0, 0.1, 0.5, 0.9, 1.0):
        xr = (1 - tau) * x + tau / inst.n
        rr = np.maximum(y - x, 1e-300) / xr
        tau_rows[f"log10_range_r_tau{tau:g}"] = float(np.log10(np.percentile(rr, 99) / max(np.percentile(rr, 1), 1e-300)))
        tau_rows[f"sat099_tau{tau:g}"] = float(np.mean(rr / (1 + rr) > 0.99))
    # ordinal content among vertices with (near-)identical prior: the most common prior value
    vals, cnt = np.unique(np.round(x / x.max(), 9), return_counts=True)
    common = cnt.max() >= 10
    if common:
        sel = np.round(x / x.max(), 9) == vals[np.argmax(cnt)]
        ord_rho = L.corr_ranks(L.ranks_of(a[sel]), L.ranks_of(y[sel]))
    else:
        ord_rho = np.nan
    rows.append(dict(id=int(f.stem), family=inst.meta["family"], prange=inst.meta["prange"],
                     prior=inst.meta["prior"], log10_kappa=np.log10(x.max() / x.min()),
                     Hnorm=inst.meta["Hnorm"], ceiling=inst.meta["ceiling"],
                     mean_alpha_prop=float(a.mean()), ordinal_rho_equal_prior=ord_rho,
                     corr_r_with_ystar=L.corr_ranks(L.ranks_of(r), L.ranks_of(y)),
                     corr_r_with_inv_prior=L.corr_ranks(L.ranks_of(r), L.ranks_of(1 / x)),
                     **tau_rows))
D = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
D.to_csv("results/prop2_stats.csv", index=False)
order = ["uniform", "dir10", "logn0.5", "dir3", "logn1", "dir1", "logn2", "dir0.3", "seed5_0.1", "seed1_0.1"]
g = D.groupby("prior")[["log10_kappa", "log10_range_r_tau0", "sat099_tau0", "corr_r_with_ystar",
                        "corr_r_with_inv_prior", "ordinal_rho_equal_prior", "log10_range_r_tau0.5",
                        "sat099_tau0.5"]].mean().reindex(order)
pd.set_option("display.width", 220)
print(g.round(3))
