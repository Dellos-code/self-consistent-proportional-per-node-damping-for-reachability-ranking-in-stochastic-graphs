"""Sign-condition diagnostic: at the calibrated-rule fixed point, the fraction of
vertices whose neighbour signal exceeds their prior (S_i > x_i) predicts the
direction in which per-node damping should move with reachability."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import prop_damping_lib as L
from build_panels import DATA, load_instance
import analysis as A

rows = []
for panel in ("dev", "test"):
    R, M, D = A.load_stage("grid", panel)
    P = R[R["T"] == 0].pivot_table(index="id", columns="method", values="rho")
    for iid in M.index:
        inst, yA, yB, _ = load_instance(DATA / panel / f"{iid:04d}.npz")
        ops = inst.ops()
        a0 = L.rule_alpha(ops["Pbar"])
        y, _ = L.diffuse(ops["EW"], inst.x, a0)
        S = ops["EW"] @ y
        fpos = float(np.mean(S > inst.x))
        # x-weighted share (mass-weighted)
        rows.append(dict(panel=panel, id=iid, family=M.loc[iid, "family"], prior=M.loc[iid, "prior"],
                         prange=M.loc[iid, "prange"], Pbar=M.loc[iid, "Pbar"], frac_S_gt_x=fpos,
                         pn_corr_prop=M.loc[iid].get("pn_corr_prop_true", np.nan),
                         d_trueprop=P.loc[iid, "true:prop"] - P.loc[iid, "base:rule_EW"],
                         d_trueap1=P.loc[iid, "true:ap1.0"] - P.loc[iid, "base:rule_EW"]))
Dd = pd.DataFrame(rows)
Dd.to_csv("results/sign_condition.csv", index=False)
ok = Dd.dropna(subset=["pn_corr_prop"])
_r1 = spearmanr(ok.frac_S_gt_x, ok.pn_corr_prop)
_ms = Path("results/tables/misc_stats.json")
_d = json.loads(_ms.read_text()) if _ms.exists() else {}
_d.update(sign_spearman=float(_r1[0]), sign_spearman_p=float(_r1[1]), sign_n=int(len(ok)))
_ms.parent.mkdir(exist_ok=True)
_ms.write_text(json.dumps(_d, indent=1))
print("Spearman(frac S>x, corr(alpha*_node, 1-x/y*)) =", round(spearmanr(ok.frac_S_gt_x, ok.pn_corr_prop)[0], 3),
      " n =", len(ok))
print("Spearman(frac S>x, rho(true:prop) - rho(rule)) =", round(spearmanr(Dd.frac_S_gt_x, Dd.d_trueprop)[0], 3))
Dd["bin"] = pd.cut(Dd.frac_S_gt_x, [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
print(Dd.groupby("bin", observed=True)[["pn_corr_prop", "d_trueprop", "d_trueap1", "Pbar"]].agg(["mean", "count"]).round(3).to_string())
print(Dd.groupby(["prior", "prange"])[["frac_S_gt_x", "pn_corr_prop", "d_trueprop"]].mean().round(3).unstack("prior").to_string())
