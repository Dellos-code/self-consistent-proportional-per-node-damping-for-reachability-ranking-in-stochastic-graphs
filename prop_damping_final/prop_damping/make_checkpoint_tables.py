"""Generate every Checkpoint-1 table from saved results (no number hard-coded).

Writes results/tables/*.csv and results/tables/*.md."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as A

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "tables"
OUT.mkdir(exist_ok=True)
SEL = json.loads((RES / "selected.json").read_text())
INF = 10 ** 6

LABEL = {
    "base:rule_EW": "calibrated scalar rule [4] (EW)",
    "base:fixed085": "fixed alpha = 0.85 (EW)",
    "base:rule_WT": "calibrated push PPR [4]",
    "base:local_pesc": "naive local alpha_i = P_esc(i)",
    "oracle:scalar": "scalar oracle (fit B, eval A)",
    "oracle:scalar_in": "scalar oracle (in-sample on A)",
    "oracle:scalar_WT": "push-PPR scalar oracle (fit B)",
    "oracle:pernode": "per-node oracle (fit B, eval A)",
    "ref:ceiling": "split-sample reference: rank by independent y*_B",
    "ref:prior": "rank by prior x",
    "ref:Pesc": "rank by P_esc",
    "ref:MP2": "KNZ message passing, 2 it.",
    "ref:MPinf": "KNZ message passing, converged",
    "true:prop": "law with TRUE y*: alpha_i = 1 - x_i/y*_i",
    "true:ap0.5": "anchored law with TRUE y*, eta=0.5",
    "true:ap1.0": "anchored law with TRUE y*, eta=1",
}


def md(df, floatfmt=3):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join([d.index.name or ""] + [str(c) for c in d.columns]) + " |"
    sep = "|" + "---|" * (len(d.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(d.index, d.values)]
    return "\n".join([head, sep] + body)


def save(name, df, note=""):
    df.to_csv(OUT / f"{name}.csv")
    (OUT / f"{name}.md").write_text((note + "\n\n" if note else "") + md(df))
    return df


def pfmt(p):
    return "" if pd.isna(p) else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


# ----------------------------------------------------------------------------
def confirmatory(panel="test"):
    R, M, D = A.load_stage("confirm", panel)
    rule = A.per_instance(R, "base:rule_EW")
    orc = A.per_instance(R, "oracle:scalar")
    pn = A.per_instance(R, "oracle:pernode")
    rows = []
    methods = [m for m in LABEL if (R["method"] == m).any()] + ["sel:" + k for k in SEL]
    for m in methods:
        s = A.per_instance(R, m)
        if s.empty:
            continue
        a = A.paired(s, rule)
        b = A.paired(s, orc)
        byp = s.to_frame("r").join(M["prior"]).groupby("prior")["r"].mean()
        rows.append(dict(method=m, label=LABEL.get(m, m.replace("sel:", "")), mean=s.mean(),
                         dir1=byp.get("dir1"), uniform=byp.get("uniform"),
                         d_rule=a["diff"], ci=f"[{a['lo']:+.3f}, {a['hi']:+.3f}]", p_rule=a["p"],
                         wtl=f"{a['win']}/{a['tie']}/{a['loss']}", d_scalar_oracle=b["diff"],
                         pct_of_scalar_oracle=100 * s.mean() / orc.mean(),
                         pct_scalar_gap_closed=100 * (s.mean() - rule.mean()) / (orc.mean() - rule.mean()),
                         pct_pernode_gap_closed=100 * (s.mean() - rule.mean()) / (pn.mean() - rule.mean())))
    T = pd.DataFrame(rows).set_index("method")
    conf = T.index.str.startswith("sel:")
    T.loc[conf, "p_holm"] = A.holm(T.loc[conf, "p_rule"].fillna(1.0).values)
    T["p_rule"] = T["p_rule"].map(pfmt)
    T["p_holm"] = T["p_holm"].map(pfmt)
    return T, R, M


def baseline_frame(stage, panels):
    out = []
    for panel in panels:
        R, M, D = A.load_stage(stage, panel)
        P = R[R["T"] == 0].pivot_table(index="id", columns="method", values="rho")
        P = P.join(M)
        P["panel"] = panel
        P["diag"] = [D[i] for i in P.index]
        out.append(P)
    return pd.concat(out)


def t2_replication(P):
    """[4]'s rule against the scalar oracle, and the hindsight-best constant alpha."""
    rows = []
    for grp, G in [("all", P)] + list(P.groupby("prior")):
        curves = np.array([d["scalar_curve_A"] for d in G["diag"]])
        mean_curve = curves.mean(0)
        k = int(np.argmax(mean_curve))
        rows.append(dict(group=grp, n=len(G), rule=G["base:rule_EW"].mean(),
                         fixed085=G["base:fixed085"].mean(),
                         best_constant=mean_curve[k], best_constant_alpha=0.025 * (k + 1),
                         scalar_oracle_in=G["oracle:scalar_in"].mean(),
                         pct_of_oracle_in=100 * G["base:rule_EW"].mean() / G["oracle:scalar_in"].mean(),
                         mean_alpha_rule=G["alpha_rule"].mean(),
                         mean_alpha_oracle=G["alpha_scalar_A"].mean()))
    return pd.DataFrame(rows).set_index("group")


def t3_headroom(P):
    P = P.copy()
    P["pernode_gain"] = P["oracle:pernode"] - P["oracle:scalar"]
    P["apparent_gain"] = P["pn_fit"] - P["pn_fit_scalar"]
    cols = ["oracle:scalar", "oracle:pernode", "ref:ceiling", "pernode_gain", "apparent_gain"]
    fam = P.groupby("family")[cols].mean()
    fam.loc["ALL"] = P[cols].mean()
    corner = P.groupby(["prange", "prior"])[cols].mean()
    corner.index = [f"{a} / {b}" for a, b in corner.index]
    corner.index.name = "range / prior"
    return fam, corner


def t5_eta_curve(stage_panels):
    out = []
    for panel in stage_panels:
        R, M, D = A.load_stage("grid", panel)
        S = R[R["method"].str.startswith("ap:") & (R["T"] == INF)].join(M["prior"], on="id")
        g = S.groupby(["method", "prior"])["rho"].mean().unstack()
        g.index = [float(i.split(":")[1]) for i in g.index]
        g.columns = [f"{panel}_{c}" for c in g.columns]
        out.append(g.sort_index())
    T = pd.concat(out, axis=1)
    T.index.name = "eta"
    return T


def t7_meanfield():
    out = []
    for panel in ("dev", "test"):
        C = pd.read_csv(RES / f"mfcontrols_{panel}.csv.gz")
        R, M, D = A.load_stage("grid", panel)
        C = C.join(M["prior"], on="id")
        g = C.groupby(["config", "prior", "control"])["rho"].mean().unstack("control")
        g["law_minus_mean"] = g["law"] - g["mean"]
        g.columns = [f"{panel}_{c}" for c in g.columns]
        out.append(g)
    T = pd.concat(out, axis=1)
    T.index = [f"{c} / {p}" for c, p in T.index]
    T.index.name = "config / prior"
    return T


def t8_sign():
    D = pd.read_csv(RES / "sign_condition.csv")
    D["frac_S_gt_x_bin"] = pd.cut(D.frac_S_gt_x, [-0.01, 0.1, 0.3, 0.5, 0.7, 1.0]).astype(str)
    g = D.groupby("frac_S_gt_x_bin")[["pn_corr_prop", "d_trueprop", "d_trueap1", "Pbar"]].mean()
    g["n"] = D.groupby("frac_S_gt_x_bin").size()
    return g


def t11_secondary(panel="test"):
    R, M, D = A.load_stage("confirm", panel)
    rows = [r for iid in D for r in D[iid]["full_metrics"]]
    F = pd.DataFrame(rows)
    g = F.groupby("method")[["rho", "tau", "topk", "ndcg"]].mean()
    keep = [m for m in list(LABEL) + ["sel:" + k for k in SEL] if m in g.index]
    return g.loc[keep]


def t13_dynamics():
    rows = []
    for panel in ("dev",):
        R, M, D = A.load_stage("grid", panel)
        for iid, d in D.items():
            for tag, v in d["dynamics"].items():
                for t in ("1", "5", "20", "200"):
                    if t in v:
                        rows.append(dict(config=tag.replace("sa:", ""), prior=M.loc[iid, "prior"], t=int(t),
                                         active=v[t][0], trivial=v[t][1],
                                         diverged=("diverged_at" in v)))
    Dy = pd.DataFrame(rows)
    g = Dy.groupby(["config", "prior", "t"])[["active", "trivial"]].mean().unstack("t")
    g.columns = [f"{a}@t{t}" for a, t in g.columns]
    div = Dy.groupby(["config", "prior"])["diverged"].mean()
    g["diverged_frac"] = div
    return g


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    T, R, M = confirmatory("test")
    save("T6_confirmatory_test", T.drop(columns=["label"]),
         "Held-out test panel (n=100, 180 instances). Configurations frozen on dev "
         "(results/selected.json, sha256 in selected.sha256). d_rule: paired mean difference "
         "vs calibrated rule with 95% bootstrap CI; p: Wilcoxon signed-rank; Holm over the "
         f"{int(T.index.str.startswith('sel:').sum())} frozen configurations. Gap-closed "
         "percentages are ratios of panel means relative to the calibrated rule.")
    P = baseline_frame("grid", ["dev", "test"])
    save("T2_replication_of_rule", t2_replication(P),
         "Rule clip(0.502+0.558 Pbar, 0.4, 1) vs scalar oracle on dev+test (n=100, 360 instances).")
    fam, corner = t3_headroom(P)
    save("T3a_pernode_headroom_by_family", fam, "Per-node oracle fitted on y*_B, evaluated on y*_A.")
    save("T3b_pernode_headroom_by_regime", corner)
    save("T5_anchored_eta_curve", t5_eta_curve(["dev", "test"]),
         "Anchored proportional law AP(eta), self-consistent fixed point; eta=0 is the scalar rule.")
    save("T7_meanfield_attribution", t7_meanfield(),
         "Each frozen realization config (law) vs the identical pipeline with alpha replaced by its "
         "per-step mean (mean) or randomly reassigned (perm).")
    save("T8_sign_condition", t8_sign())
    save("T11_secondary_metrics_test", t11_secondary("test"),
         "Kendall tau-b, top-10% overlap and NDCG@10% (relevance = y*_A), test panel.")
    save("T13_dynamics_dev", t13_dynamics(),
         "Fraction of vertices on the active branch (S > 2x) and on the trivial branch (y = x).")
    for name in ("T2_replication_of_rule", "T3a_pernode_headroom_by_family", "T5_anchored_eta_curve",
                 "T7_meanfield_attribution", "T8_sign_condition", "T11_secondary_metrics_test"):
        print("\n=====", name)
        print(pd.read_csv(OUT / f"{name}.csv", index_col=0).round(4).to_string())


PRIOR_ORDER = ["uniform", "dir10", "logn0.5", "dir3", "logn1", "dir1", "logn2", "dir0.3", "seed5_0.1", "seed1_0.1"]


def t9_prior():
    R, M, D = A.load_stage("confirm", "prior")
    P = R[R["T"] == 0].pivot_table(index="id", columns="method", values="rho").join(M[["prior", "log10_kappa", "family", "prange"]])
    rule = P["base:rule_EW"]
    cols = {"rule": "base:rule_EW", "scalar_oracle": "oracle:scalar", "pernode_oracle": "oracle:pernode",
            "ceiling": "ref:ceiling", "true_prop": "true:prop", "true_ap1": "true:ap1.0",
            "AP_eta1": "sel:AP_eta1", "AP_eta1_tau0.5": "sel:AP_eta1@tau0.5", "AP_eta1_tau1": "sel:AP_eta1@tau1",
            "PBnom": "sel:PB_rule_nominal", "PBnom_tau0.5": "sel:PB_rule_nominal@tau0.5",
            "PBnom_tau1": "sel:PB_rule_nominal@tau1", "PBX_best": "sel:PBX_rule_best",
            "CTRL_lin0.9": "sel:CTRL_lin0.9", "CTRL_Cmean": "sel:CTRL_Cmean", "B_unif": "sel:B_unif",
            "C_dir1": "sel:C_dir1"}
    T = pd.DataFrame({k: P[v] for k, v in cols.items() if v in P}).join(P[["prior", "log10_kappa"]])
    g = T.groupby("prior").mean().reindex(PRIOR_ORDER)
    # differences vs the rule, per prior type, with Wilcoxon p
    dd = {}
    for k in ["AP_eta1", "AP_eta1_tau0.5", "AP_eta1_tau1", "PBnom", "PBnom_tau1", "PBX_best", "CTRL_lin0.9"]:
        if k not in T:
            continue
        diffs = {}
        for pr in PRIOR_ORDER:
            G = T[T.prior == pr]
            r = A.paired(G[k], G["rule"])
            diffs[pr] = r["diff"]
        dd["d_" + k] = pd.Series(diffs)
    return g, pd.DataFrame(dd).reindex(PRIOR_ORDER), T


if __name__ == "__main__" and (RES / "confirm_prior").exists() and \
        len(list((RES / "confirm_prior").glob("*.csv.gz"))) >= 150:
    g, dd, Tp = t9_prior()
    save("T9a_prior_panel_methods", g, "Prior-uniformity panel (5 families x 3 ranges x 10 priors, n=144-150).")
    save("T9b_prior_panel_diff_vs_rule", dd, "Mean paired difference vs calibrated rule, per prior type.")
    from scipy.stats import spearmanr
    _ms = OUT / "misc_stats.json"
    _d = json.loads(_ms.read_text()) if _ms.exists() else {}
    for k in ("AP_eta1", "PBnom", "true_ap1"):
        if k in Tp:
            r = spearmanr(Tp["log10_kappa"], Tp[k] - Tp["rule"])
            _d[f"prior_spearman_{k}"] = float(r[0])
            _d[f"prior_spearman_{k}_p"] = float(r[1])
            _ms.write_text(json.dumps(_d, indent=1))
            print(f"Spearman(log10 kappa, {k} - rule) = {r[0]:.3f} (p={r[1]:.2g})")
    print(g.round(3).to_string())
    print(dd.round(4).to_string())
    pd.read_csv(RES / "prop2_stats.csv").to_csv(OUT / "T13b_prop2_ground_truth_stats.csv", index=False)


def t10_transfer(panel):
    R, M, D = A.load_stage("confirm", panel)
    rule = A.per_instance(R, "base:rule_EW")
    keys = ["base:rule_EW", "oracle:scalar", "oracle:pernode", "ref:ceiling", "ref:MP2", "ref:Pesc",
            "true:prop", "sel:AP_eta1", "sel:PB_rule_nominal", "sel:PB_rule_best", "sel:PB_mp2_best",
            "sel:PB_mp2_eta0", "sel:PBX_rule_best", "sel:CTRL_Cmean", "sel:CTRL_lin0.9", "sel:A_best",
            "sel:B_best", "sel:C_best", "sel:C_dir1", "sel:B_unif"]
    group = "genre" if panel == "real" else "prior"
    rows = []
    for k in keys:
        s = A.per_instance(R, k)
        if s.empty:
            continue
        a = A.paired(s, rule)
        byg = s.to_frame("r").join(M[[group]]).groupby(group)["r"].mean()
        rows.append(dict(method=k, n=len(s), mean=s.mean(), d_rule=a["diff"],
                         ci=f"[{a['lo']:+.3f}, {a['hi']:+.3f}]", p=pfmt(a["p"]),
                         wtl=f"{a['win']}/{a['tie']}/{a['loss']}", **{f"{group}={g}": v for g, v in byg.items()}))
    return pd.DataFrame(rows).set_index("method")


if __name__ == "__main__":
    for panel in ("real", "scale"):
        d = RES / f"confirm_{panel}"
        if d.exists() and len(list(d.glob("*.csv.gz"))) >= {"real": 88, "scale": 90}[panel]:
            T10 = t10_transfer(panel)
            save(f"T10_transfer_{panel}", T10, f"Frozen configurations on the {panel} panel.")
            print(f"\n===== T10 {panel}")
            print(T10.round(4).to_string())


def t7b_attribution(panel):
    """Per-node law vs its mean-field twin on a transfer panel, with paired tests."""
    C = pd.read_csv(RES / f"mfcontrols_{panel}.csv.gz")
    W = C.pivot_table(index=["id", "config"], columns="control", values="rho").reset_index()
    rows = []
    for cfg, G in W.groupby("config"):
        law, mean = G.set_index("id")["law"], G.set_index("id")["mean"]
        r = A.paired(law, mean)
        rows.append(dict(config=cfg, law=law.mean(), mean=mean.mean(), perm=G["perm"].mean(),
                         d_law_minus_mean=r["diff"], ci=f"[{r['lo']:+.4f}, {r['hi']:+.4f}]", p=r["p"],
                         wtl=f"{r['win']}/{r['tie']}/{r['loss']}"))
    T = pd.DataFrame(rows).set_index("config")
    T["p_holm"] = A.holm(T["p"].fillna(1.0).values)
    T["p"] = T["p"].map(pfmt)
    T["p_holm"] = T["p_holm"].map(pfmt)
    return T


if __name__ == "__main__":
    for panel, name in (("real", "T7b_meanfield_attribution_real"), ("scale", "T7c_meanfield_attribution_scale")):
        if (RES / f"mfcontrols_{panel}.csv.gz").exists():
            save(name, t7b_attribution(panel),
                 f"Per-node law vs per-step-mean twin ({panel} panel); Holm over the configurations listed.")


if __name__ == "__main__":
    for panel in ("real", "scale"):
        d = RES / f"confirm_{panel}"
        if d.exists() and len(list(d.glob("*.csv.gz"))) >= {"real": 88, "scale": 90}[panel]:
            save(f"T11_secondary_metrics_{panel}", t11_secondary(panel),
                 f"Kendall tau-b, top-10% overlap and NDCG@10% on the {panel} panel.")


def t12_cost():
    """Accuracy (mean rho on the panel) against median single-process wall clock."""
    rt = pd.read_csv(RES / "runtime.csv")
    mc = pd.read_csv(RES / "mc_baseline.csv")
    ms = rt.groupby(["panel", "method"])["ms"].median()
    rows = []
    for panel in ("test", "scale"):
        R, M, D = A.load_stage("confirm", panel)
        rho = lambda m: A.per_instance(R, m).mean()
        base = ms[(panel, "rule_EW (calibrated scalar)")]
        ops_ms = ms[(panel, "operators (P_esc, W, EW, P)")]
        entries = [
            ("calibrated rule [4]", rho("base:rule_EW"), ms[(panel, "E2E calibrated rule")]),
            ("KNZ MP, 2 it.", rho("ref:MP2"), ops_ms + ms[(panel, "KNZ MP, 2 iterations")]),
            ("AP(eta=1), fixed point", rho("sel:AP_eta1"), ops_ms + ms[(panel, "AP(eta=1) standalone, to fixed point")]),
            ("rule + anchored polish (K=1)", rho("sel:PB_rule_nominal"), ops_ms + base + ms[(panel, "Mode (b) nominal polish, K=1 (excl. base)")]),
            ("rule + C-law on raw P (K=5)", rho("sel:PBX_rule_best"), ms[(panel, "E2E rule + C-law raw P (PBX_rule_best)")]),
            ("rule + constant-alpha raw P (K=5)", rho("sel:CTRL_lin0.9"), ms[(panel, "E2E rule + constant-alpha raw P (CTRL_lin0.9)")]),
            ("scalar oracle (needs ground truth)", rho("oracle:scalar"), np.nan),
        ]
        for k in (50, 200, 1000):
            G = mc[(mc.panel == panel) & (mc.k == k)]
            entries.append((f"Monte Carlo, {k} worlds", G.rho.mean(), ms[(panel, f"Monte Carlo, {k} worlds")]))
        for name, r, t in entries:
            rows.append(dict(panel=panel, method=name, rho=r, ms=t))
    T = pd.DataFrame(rows).pivot(index="method", columns="panel", values=["rho", "ms"])
    T.columns = [f"{a}_{b}" for a, b in T.columns]
    return T.sort_values("rho_test")


if __name__ == "__main__" and (RES / "mc_baseline.csv").exists() and (RES / "runtime.csv").exists():
    T12 = t12_cost()
    save("T12_accuracy_vs_cost", T12, "Mean Spearman rho on the whole panel against median single-process, "
         "end-to-end wall clock (ms) over one uniform-prior instance per family and probability range.")
    print(T12.round(4).to_string())
