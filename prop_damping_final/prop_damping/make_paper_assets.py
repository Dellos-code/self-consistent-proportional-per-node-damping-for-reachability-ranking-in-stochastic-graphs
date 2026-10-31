"""make_paper_assets.py — every number, table and figure used by the papers.

Reads results/ only and writes
  paper/generated/macros.tex      numeric macros (shared by EN and GR documents)
  paper/generated/tables_en.tex   table macros, English
  paper/generated/tables_gr.tex   table macros, Greek
  figures/fig{1..5}_*.pdf         English figures
  figures_gr/fig{1..5}_*.pdf      Greek figures
No result is typed by hand: change results/ and re-run to update all documents.
"""
import json
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as A
from build_panels import DATA, M_GT

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
TB = RES / "tables"
GEN = ROOT / "paper" / "generated"
FIGDIR = {"en": ROOT / "figures", "gr": ROOT / "figures_gr"}
SEL = json.loads((RES / "selected.json").read_text())
INF = 10 ** 6
MINUS = "−"

# Quoted from the prior work [4] (not results of this study).
PREV_HEADROOM_ALL, PREV_HEADROOM_GRID, PREV_PCT_ORACLE = 0.062, 0.341, 98.2


# =============================================================================
# formatting
# =============================================================================
def num(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "---"
    return f"{v:.{d}f}".replace("-", MINUS)


def sgn(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "---"
    s = f"{v:+.{d}f}"
    if float(s) == 0.0:
        s = ("+" if v >= 0 else "-") + s[1:]
    return s.replace("-", MINUS)


def pval(p):
    if p is None or not np.isfinite(p):
        return "---"
    if p <= 0:
        return "$<$10\\textsuperscript{" + MINUS + "300}"
    if p >= 1e-3:
        return f"{p:.3f}"
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.1f}\\texttimes10\\textsuperscript{{{MINUS}{-e}}}"


def sci(v, d=1):
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** e
    return f"{m:.{d}f}\\texttimes10\\textsuperscript{{{MINUS if e < 0 else ''}{abs(e)}}}"


def pct(v, d=1):
    return f"{v:.{d}f}".replace("-", MINUS)


# =============================================================================
# data
# =============================================================================
class Panel:
    def __init__(self, stage, panel):
        self.R, self.M, self.D = A.load_stage(stage, panel)
        self.rule = A.per_instance(self.R, "base:rule_EW")

    def s(self, m):
        return A.per_instance(self.R, m)

    def mean(self, m):
        return float(self.s(m).mean())

    def vs_rule(self, m):
        return A.paired(self.s(m), self.rule)

    def by(self, m, col="prior"):
        return self.s(m).to_frame("r").join(self.M[[col]]).groupby(col)["r"].mean()


CONF = {p: Panel("confirm", p) for p in ("test", "real", "scale", "prior")}
GRID = {p: Panel("grid", p) for p in ("dev", "test")}


def holm_family(panel):
    """Holm-adjusted p (vs rule) over the frozen configurations on one panel."""
    P = CONF[panel]
    keys = ["sel:" + k for k in SEL]
    ps = [P.vs_rule(k)["p"] for k in keys]
    adj = A.holm(np.nan_to_num(np.array(ps, float), nan=1.0))
    return dict(zip(keys, adj))


HOLM = {p: holm_family(p) for p in ("test", "real", "scale")}

tf = json.loads((RES / "tests_foundations.json").read_text())
td = json.loads((RES / "tests_dynamics.json").read_text())
ms_ = json.loads((TB / "misc_stats.json").read_text())
T2 = pd.read_csv(TB / "T2_replication_of_rule.csv", index_col=0)
T3 = pd.read_csv(TB / "T3a_pernode_headroom_by_family.csv", index_col=0)
T3b = pd.read_csv(TB / "T3b_pernode_headroom_by_regime.csv", index_col=0)
T5 = pd.read_csv(TB / "T5_anchored_eta_curve.csv", index_col=0)
T8 = pd.read_csv(TB / "T8_sign_condition.csv", index_col=0)
T9 = pd.read_csv(TB / "T9a_prior_panel_methods.csv", index_col=0)
T9d = pd.read_csv(TB / "T9b_prior_panel_diff_vs_rule.csv", index_col=0)
T12 = pd.read_csv(TB / "T12_accuracy_vs_cost.csv", index_col=0)
T13 = pd.read_csv(TB / "T13_dynamics_dev.csv", index_col=[0, 1])
P2 = pd.read_csv(RES / "prop2_stats.csv")
SIGN = pd.read_csv(RES / "sign_condition.csv")
TRACES = pd.read_csv(RES / "traces_dev.csv.gz")
COMP = pd.read_csv(RES / "complementarity.csv")
SWEEP = pd.read_csv(RES / "size_sweep.csv")
EXPL = pd.concat([pd.read_csv(RES / "explore_rawP_dev_test.csv.gz"),
                  pd.read_csv(RES / "explore_rawP_real_scale.csv.gz")])
MCB = pd.read_csv(RES / "mc_baseline.csv")
RT = pd.read_csv(RES / "runtime.csv")
MF = {p: pd.read_csv(RES / f"mfcontrols_{p}.csv.gz") for p in ("dev", "test", "real", "scale")}
SHA = (RES / "selected.sha256").read_text().split()

PRIOR_ORDER = ["uniform", "dir10", "logn0.5", "dir3", "logn1", "dir1", "logn2", "dir0.3", "seed5_0.1", "seed1_0.1"]


def mf_paired(panel, cfg):
    C = MF[panel]
    W = C[C.config == cfg].pivot_table(index="id", columns="control", values="rho")
    return A.paired(W["law"], W["mean"]), W


# --------------------------------------------------------------------- size sweep
def sweep_tables():
    D = SWEEP.copy()
    ks = np.array([50, 200, 1000])
    lk = np.log(ks)

    def rho_at(r, k):
        y = np.array([r.rho_mc50, r.rho_mc200, r.rho_mc1000])
        return float(np.interp(np.log(np.clip(k, 50, 1000)), lk, y))

    D["k_eq"] = 200 * D.ms_ensemble / D.ms_mc200
    D["rho_mc_eq"] = [rho_at(r, r.k_eq) for r in D.itertuples()]
    D["d_eq"] = D.rho_ensemble - D.rho_mc_eq
    D["ratio"] = D.ms_mc200 / D.ms_ensemble
    D["bucket"] = pd.cut(D.n, [0, 160, 350, 700, 1400, 2100],
                         labels=["100--150", "196--300", "400--600", "900--1300", "2000--2025"])
    return D


SW = sweep_tables()


def crossover_n(fam):
    """Smallest n beyond which the ensemble is faster than MC-200 at every larger size (median over reps)."""
    G = SW[SW.family == fam].groupby("n")["ratio"].median().sort_index()
    ns = list(G.index)
    for i, n in enumerate(ns):
        if all(G.loc[m] > 1 for m in ns[i:]):
            return n, (ns[i - 1] if i > 0 else None)
    return None, ns[-1]


# =============================================================================
# macros
# =============================================================================
MAC = {}


def mac(name, val):
    assert name.isalpha(), name
    assert name not in MAC, name
    MAC[name] = str(val)


def build_macros():
    # ---- design / protocol
    counts = {p: len(list((DATA / p).glob("[0-9][0-9][0-9][0-9].npz"))) for p in ("dev", "test", "prior", "real", "scale")}
    mac("NDev", counts["dev"]); mac("NTest", counts["test"]); mac("NPrior", counts["prior"])
    mac("NReal", counts["real"]); mac("NScale", counts["scale"]); mac("NTotal", sum(counts.values()))
    mac("NDevTest", counts["dev"] + counts["test"])
    mac("MGT", f"{M_GT:,}".replace(",", "{,}")); mac("MGTsp", f"{M_GT:,}".replace(",", "\\,"))
    mac("NFrozen", len(SEL))
    mac("SelHash", SHA[0][:16])
    mac("SelTime", SHA[2].replace("T", " ")[:16] if len(SHA) > 2 else "")
    sizes = {f: len(A.config_means(GRID["dev"].R, pre)) for f, pre in
             (("SA", "sa:"), ("AP", "ap:"), ("PB", "pb:"), ("PBX", "pbx:"))}
    mac("CfgSA", sizes["SA"]); mac("CfgAP", sizes["AP"]); mac("CfgPB", sizes["PB"]); mac("CfgPBX", sizes["PBX"])
    mac("CfgTotal", sum(sizes.values()))
    Md = GRID["dev"].M
    mac("ERPbarMin", num(Md[Md.family == "er"].Pbar.min(), 2))
    mac("ERPbarMax", num(Md[Md.family == "er"].Pbar.max(), 2))
    mac("NFamilies", Md.family.nunique())
    # ---- foundations
    mac("ThmResidual", sci(tf["theorem1_max_residual"]))
    mac("MCExactZ", num(tf["mc_vs_exact_max_z"], 1))
    mac("KNZzeroDiff", sci(tf["knz0_minus_pesc"]))
    mac("CollapseExcess", sci(tf["collapse_A_uniform_max_rel_excess_after_1_step"]))
    # ---- dynamics
    mac("TrivAOneUnif", pct(100 * T13.loc[("A|EW|raw|1", "uniform"), "trivial@t1"], 0))
    mac("TrivBFiveUnif", pct(100 * T13.loc[("B|EW|raw|1", "uniform"), "trivial@t5"], 0))
    mac("TrivBTwentyUnif", pct(100 * T13.loc[("B|EW|raw|1", "uniform"), "trivial@t20"], 0))
    mac("ActADirEnd", pct(100 * T13.loc[("A|EW|raw|1", "dir1"), "active@t200"], 0))
    p3 = sorted([(float(k.split("=")[1].split(":")[0]), v) for k, v in td.items() if k.startswith("P3_lam")])
    for (lam, v), w in zip(p3, ("One", "Two", "Three")):
        mac(f"LamVal{w}", num(lam, 1)); mac(f"RhoJB{w}", num(v[1], 2)); mac(f"RhoJA{w}", num(v[0], 2))
    mac("CStartsDiff", sci(td["P4_C_EW_three_starts_same_fixed_point_maxdiff"], 0))
    mac("SparseFracSbelowX", pct(100 * td["P5_sparse_ER_fraction_S_below_x"], 0))
    mac("SignAgree", td["P5_sign_rule_agreement_(of_20)"])
    # ---- sign obstruction
    mac("SignSpearman", num(ms_["sign_spearman"], 2)); mac("SignSpearmanP", sci(ms_["sign_spearman_p"]))
    mac("SignN", ms_["sign_n"])
    mac("SignLowCorr", num(T8.iloc[0]["pn_corr_prop"], 2)); mac("SignLowDrop", num(abs(T8.iloc[0]["d_trueprop"]), 2))
    mac("SignLowN", int(T8.iloc[0]["n"])); mac("SignMaxCorr", num(T8["pn_corr_prop"].max(), 2))
    # ---- Proposition F (pull PPR)
    Rg, Mg = GRID["dev"].R, GRID["dev"].M
    Sg = Rg[Rg.method.str.startswith("sa:")].copy()
    Sg["op"] = Sg.method.str.split("|").str[1]
    Sg = Sg.join(Mg["prior"], on="id")
    w = Sg[(Sg.op == "W") & (Sg.prior == "uniform")]["rho"]
    mac("PullZeroPct", pct(100 * (w.abs() < 1e-12).mean(), 0)); mac("PullMean", num(w.mean(), 3))
    # ---- Proposition 2 ground truth
    g = P2.groupby("prior")[["corr_r_with_ystar", "corr_r_with_inv_prior", "sat099_tau0", "log10_kappa"]].mean()
    mac("RinvDirTen", num(g.loc["dir10", "corr_r_with_inv_prior"], 2))
    mac("RinvDirSmall", num(g.loc["dir0.3", "corr_r_with_inv_prior"], 2))
    mac("RyDirTen", num(g.loc["dir10", "corr_r_with_ystar"], 2))
    mac("RyDirSmall", num(g.loc["dir0.3", "corr_r_with_ystar"], 2))
    mac("RySeedFive", num(g.loc["seed5_0.1", "corr_r_with_ystar"], 2))
    mac("RySeedOne", num(g.loc["seed1_0.1", "corr_r_with_ystar"], 2))
    mac("SatSeedOne", pct(100 * g.loc["seed1_0.1", "sat099_tau0"], 0))
    mac("KappaDirSmall", num(g.loc["dir0.3", "log10_kappa"], 1))
    # ---- replication of [4]
    mac("PrevHeadAll", sgn(PREV_HEADROOM_ALL)); mac("PrevHeadGrid", sgn(PREV_HEADROOM_GRID))
    mac("PrevPctOracle", pct(PREV_PCT_ORACLE))
    mac("RepRuleDir", num(T2.loc["dir1", "rule"])); mac("RepRuleUnif", num(T2.loc["uniform", "rule"]))
    mac("RepBestConstDir", num(T2.loc["dir1", "best_constant"]))
    mac("RepBestConstAlphaDir", num(T2.loc["dir1", "best_constant_alpha"], 3))
    mac("RepPctDir", pct(T2.loc["dir1", "pct_of_oracle_in"])); mac("RepPctUnif", pct(T2.loc["uniform", "pct_of_oracle_in"]))
    mac("RepOracleAlphaUnif", num(T2.loc["uniform", "mean_alpha_oracle"], 2))
    mac("RepOracleAlphaDir", num(T2.loc["dir1", "mean_alpha_oracle"], 2))
    mac("RepRuleAlpha", num(T2.loc["uniform", "mean_alpha_rule"], 2))
    mac("RepConstAlphaUnif", num(T2.loc["uniform", "best_constant_alpha"], 3))
    mac("RepConstUnif", num(T2.loc["uniform", "best_constant"]))
    mac("RepFixedUnif", num(T2.loc["uniform", "fixed085"]))
    # ---- headroom
    mac("HeadAll", sgn(T3.loc["ALL", "pernode_gain"])); mac("HeadApparent", sgn(T3.loc["ALL", "apparent_gain"]))
    mac("HeadOverfitPct", pct(100 * (1 - T3.loc["ALL", "pernode_gain"] / T3.loc["ALL", "apparent_gain"]), 0))
    mac("HeadGrid", sgn(T3.loc["grid", "pernode_gain"])); mac("HeadTree", sgn(T3.loc["tree", "pernode_gain"]))
    mac("HeadRect", sgn(T3.loc["rect", "pernode_gain"]))
    mac("HeadCorner", sgn(T3b.loc["R5 / dir1", "pernode_gain"]))
    pm = []
    for p in ("dev", "test"):
        Pp = GRID[p].R[GRID[p].R["T"] == 0].pivot_table(index="id", columns="method", values="rho").join(GRID[p].M["family"])
        Pp = Pp[Pp.family.isin(["grid", "rect"])]
        pm.append(float((Pp["oracle:pernode"] - Pp["oracle:scalar"]).max()))
    mac("HeadMax", sgn(max(pm)))
    # ---- held-out test panel
    T = CONF["test"]
    keymap = {"Rule": "base:rule_EW", "OracleScalar": "oracle:scalar", "OraclePernode": "oracle:pernode",
              "Ceiling": "ref:ceiling", "TrueProp": "true:prop", "TrueAPone": "true:ap1.0", "Fixed": "base:fixed085",
              "PushRule": "base:rule_WT", "LocalPesc": "base:local_pesc", "MPtwo": "ref:MP2", "Pesc": "ref:Pesc",
              "Abest": "sel:A_best", "Bbest": "sel:B_best", "Cbest": "sel:C_best", "Cdir": "sel:C_dir1",
              "Bunif": "sel:B_unif", "APquarter": "sel:AP_eta0.25", "APone": "sel:AP_eta1",
              "PBbest": "sel:PB_rule_best", "PBnom": "sel:PB_rule_nominal", "PBone": "sel:PB_rule_eta1",
              "MPbest": "sel:PB_mp2_best", "MPzero": "sel:PB_mp2_eta0", "MPone": "sel:PB_mp2_eta1",
              "PBXbest": "sel:PBX_rule_best", "PBXunif": "sel:PBX_rule_unif", "PBXdir": "sel:PBX_rule_dir1",
              "PBXnomB": "sel:PBX_rule_nominalB", "Cmean": "sel:CTRL_Cmean", "Cperm": "sel:CTRL_Cperm",
              "LinHalf": "sel:CTRL_lin0.5", "LinNine": "sel:CTRL_lin0.9"}
    orc, pn, rule = T.mean("oracle:scalar"), T.mean("oracle:pernode"), T.mean("base:rule_EW")
    for short, m in keymap.items():
        a = T.vs_rule(m)
        mac(f"Test{short}", num(a["mean_a"]))
        if short != "Rule":
            mac(f"Test{short}D", sgn(a["diff"]))
            mac(f"Test{short}CI", f"[{sgn(a['lo'])}, {sgn(a['hi'])}]")
            mac(f"Test{short}P", pval(a["p"]))
            mac(f"Test{short}WTL", f"{a['win']}/{a['tie']}/{a['loss']}")
        if m in HOLM["test"]:
            mac(f"Test{short}Holm", pval(HOLM["test"][m]))
        mac(f"Test{short}PctOracle", pct(100 * a["mean_a"] / orc))
        mac(f"Test{short}GapScalar", pct(100 * (a["mean_a"] - rule) / (orc - rule)))
        mac(f"Test{short}GapNode", pct(100 * (a["mean_a"] - rule) / (pn - rule)))
    mac("TestPBbestDfour", sgn(T.vs_rule("sel:PB_rule_best")["diff"], 4))
    byp = T.by("sel:C_dir1")
    mac("TestCdirOnDir", num(byp["dir1"])); mac("TestCdirOnUnif", num(byp["uniform"]))
    byp = T.by("sel:B_unif")
    mac("TestBunifOnDir", num(byp["dir1"])); mac("TestBunifOnUnif", num(byp["uniform"]))
    mac("TestRuleOnDir", num(T.by("base:rule_EW")["dir1"])); mac("TestRuleOnUnif", num(T.by("base:rule_EW")["uniform"]))
    # ---- AP(eta) curve
    mac("EtaBestDir", f"{T5['test_dir1'].idxmax():g}".replace("-", "\u2212"))
    mac("EtaZeroTestDir", num(T5.loc[0.0, "test_dir1"])); mac("EtaOneTestDir", num(T5.loc[1.0, "test_dir1"]))
    mac("EtaTwoTestDir", num(T5.loc[2.0, "test_dir1"]))
    mac("EtaUnifRange", num(T5["test_uniform"].max() - T5["test_uniform"].min(), 4))
    # ---- attribution
    for panel, P in (("Test", "test"), ("Real", "real"), ("Scale", "scale")):
        for cfg, short in (("PBX_rule_best", "PBX"), ("PBX_rule_unif", "PBXunif"), ("B_unif", "Bunif"),
                           ("C_dir1", "Cdir"), ("A_best", "Abest")):
            r, W = mf_paired(P, cfg)
            mac(f"Att{panel}{short}D", sgn(r["diff"], 4)); mac(f"Att{panel}{short}P", pval(r["p"]))
            mac(f"Att{panel}{short}CI", f"[{sgn(r['lo'], 4)}, {sgn(r['hi'], 4)}]")
            mac(f"Att{panel}{short}Law", num(W["law"].mean())); mac(f"Att{panel}{short}Mean", num(W["mean"].mean()))
    for panel, P in (("Test", "test"), ("Real", "real"), ("Scale", "scale")):
        cfgs = sorted(MF[P].config.unique())
        ps = [mf_paired(P, c)[0]["p"] for c in cfgs]
        adj = dict(zip(cfgs, A.holm(np.nan_to_num(np.array(ps, float), nan=1.0))))
        mac(f"Att{panel}PBXHolm", pval(adj["PBX_rule_best"]))
        mac(f"Att{panel}PBXunifHolm", pval(adj["PBX_rule_unif"]))
    # ---- prior panel
    mac("PriorAPUnif", sgn(T9d.loc["uniform", "d_AP_eta1"], 4)); mac("PriorAPSeedOne", sgn(T9d.loc["seed1_0.1", "d_AP_eta1"]))
    mac("PriorAPDirTen", sgn(T9d.loc["dir10", "d_AP_eta1"])); mac("PriorAPDirSmall", sgn(T9d.loc["dir0.3", "d_AP_eta1"]))
    mac("PriorAPLognTwo", sgn(T9d.loc["logn2", "d_AP_eta1"]))
    mac("PriorSpearman", num(ms_["prior_spearman_AP_eta1"], 2)); mac("PriorSpearmanP", sci(ms_["prior_spearman_AP_eta1_p"]))
    mac("PriorTauOneMin", sgn(T9d["d_AP_eta1_tau1"].min())); mac("PriorTauOneMax", sgn(T9d["d_AP_eta1_tau1"].max()))
    mac("PriorTauHalfLognTwo", sgn(T9d.loc["logn2", "d_AP_eta1_tau0.5"]))
    mac("PriorLinMin", sgn(T9d["d_CTRL_lin0.9"].min())); mac("PriorLinMax", sgn(T9d["d_CTRL_lin0.9"].max()))
    mac("PriorPBXMin", sgn(T9d["d_PBX_best"].min())); mac("PriorPBXMax", sgn(T9d["d_PBX_best"].max()))
    mac("PriorRuleUnif", num(T9.loc["uniform", "rule"])); mac("PriorOracleUnif", num(T9.loc["uniform", "scalar_oracle"]))
    # ---- transfer panels
    for panel, P in (("Real", "real"), ("Scale", "scale")):
        X = CONF[P]
        for short, m in (("Rule", "base:rule_EW"), ("OracleScalar", "oracle:scalar"), ("OraclePernode", "oracle:pernode"),
                         ("Ceiling", "ref:ceiling"), ("APone", "sel:AP_eta1"), ("PBnom", "sel:PB_rule_nominal"),
                         ("PBbest", "sel:PB_rule_best"), ("PBXbest", "sel:PBX_rule_best"), ("LinNine", "sel:CTRL_lin0.9"),
                         ("Cmean", "sel:CTRL_Cmean"), ("MPzero", "sel:PB_mp2_eta0"), ("TrueProp", "true:prop"),
                         ("Cbest", "sel:C_best")):
            a = X.vs_rule(m)
            mac(f"{panel}{short}", num(a["mean_a"]))
            if short != "Rule":
                mac(f"{panel}{short}D", sgn(a["diff"]))
                mac(f"{panel}{short}P", pval(a["p"]))
                mac(f"{panel}{short}CI", f"[{sgn(a['lo'])}, {sgn(a['hi'])}]")
        mac(f"{panel}PBbestDfour", sgn(X.vs_rule("sel:PB_rule_best")["diff"], 4))
        mac(f"{panel}Headroom", sgn(X.mean("oracle:pernode") - X.mean("oracle:scalar")))
    Rr = CONF["real"]
    mac("RealHeadCora", sgn(Rr.by("oracle:pernode", "genre")["cora"] - Rr.by("oracle:scalar", "genre")["cora"]))
    mac("NPernodeScale", len(CONF["scale"].s("oracle:pernode")))
    # ---- secondary metrics (test, real)
    for panel, P in (("Test", "test"), ("Real", "real")):
        F = pd.DataFrame([r for iid in CONF[P].D for r in CONF[P].D[iid]["full_metrics"]]).groupby("method").mean(numeric_only=True)
        for short, m in (("Rule", "base:rule_EW"), ("LinNine", "sel:CTRL_lin0.9"), ("PBXbest", "sel:PBX_rule_best"),
                         ("OracleScalar", "oracle:scalar"), ("Cmean", "sel:CTRL_Cmean")):
            mac(f"Sec{panel}{short}Tau", num(F.loc[m, "tau"])); mac(f"Sec{panel}{short}Top", num(F.loc[m, "topk"]))
            mac(f"Sec{panel}{short}Ndcg", num(F.loc[m, "ndcg"]))
    # ---- cost (single-process, end-to-end)
    def c(method, col):
        return T12.loc[method, col]
    mac("CostTestRuleMs", num(c("calibrated rule [4]", "ms_test"), 1)); mac("CostScaleRuleMs", num(c("calibrated rule [4]", "ms_scale"), 1))
    mac("CostTestEnsMs", num(c("rule + constant-alpha raw P (K=5)", "ms_test"), 1))
    mac("CostScaleEnsMs", num(c("rule + constant-alpha raw P (K=5)", "ms_scale"), 1))
    mac("CostTestPBXMs", num(c("rule + C-law on raw P (K=5)", "ms_test"), 1))
    mac("CostScalePBXMs", num(c("rule + C-law on raw P (K=5)", "ms_scale"), 1))
    for k, w in ((50, "Fifty"), (200, "TwoHundred"), (1000, "Thousand")):
        mac(f"CostTestMC{w}Ms", num(c(f"Monte Carlo, {k} worlds", "ms_test"), 1))
        mac(f"CostScaleMC{w}Ms", num(c(f"Monte Carlo, {k} worlds", "ms_scale"), 1))
        mac(f"CostTestMC{w}Rho", num(c(f"Monte Carlo, {k} worlds", "rho_test")))
        mac(f"CostScaleMC{w}Rho", num(c(f"Monte Carlo, {k} worlds", "rho_scale")))
    mac("CostScaleSpeedup", num(c("Monte Carlo, 200 worlds", "ms_scale") / c("rule + constant-alpha raw P (K=5)", "ms_scale"), 1))
    mac("CostTestAPMs", num(c("AP(eta=1), fixed point", "ms_test"), 1))
    # ---- size sweep / crossover
    for fam, w in (("er", "ER"), ("ba", "BA"), ("grid", "Grid"), ("ws", "WS")):
        n1, n0 = crossover_n(fam)
        mac(f"Cross{w}", n1 if n1 is not None else "---")
        mac(f"Cross{w}Below", n0 if n0 is not None else "---")
        mac(f"EqBudget{w}", sgn(SW[SW.family == fam].d_eq.median()))
    gb = SW.groupby("bucket", observed=True)
    mac("EqSmall", sgn(gb.d_eq.median().iloc[0])); mac("EqLarge", sgn(gb.d_eq.median().iloc[-1]))
    mac("EqMid", sgn(gb.d_eq.median().iloc[2]))
    mac("EqWinLarge", pct(100 * gb.d_eq.apply(lambda s: (s > 0).mean()).iloc[-1], 0))
    mac("EqWinSmall", pct(100 * gb.d_eq.apply(lambda s: (s > 0).mean()).iloc[0], 0))
    mac("SweepNmin", SW.n.min()); mac("SweepNmax", SW.n.max()); mac("SweepN", len(SW))
    mac("SweepMCtwoRho", num(SW.rho_mc200.median())); mac("SweepEnsRho", num(SW.rho_ensemble.median()))
    fm = SW.groupby("family")[["rho_ensemble", "rho_rule", "rho_mc200"]].median()
    for fam, w in (("er", "ER"), ("ba", "BA"), ("grid", "Grid"), ("ws", "WS")):
        mac(f"Sweep{w}EnsRho", num(fm.loc[fam, "rho_ensemble"])); mac(f"Sweep{w}RuleRho", num(fm.loc[fam, "rho_rule"]))
        mac(f"Sweep{w}MCRho", num(fm.loc[fam, "rho_mc200"]))
    # ---- complementarity
    cg = COMP.groupby("panel")
    for panel, w in (("test", "Test"), ("real", "Real"), ("scale", "Scale")):
        mac(f"Err{w}", num(cg.err_corr.mean()[panel], 2))
        mac(f"RawP{w}", num(cg.rho_rawP.mean()[panel])); mac(f"Blend{w}", num(cg.rho_blend.mean()[panel]))
        mac(f"RawP{w}D", sgn(cg.rho_rawP.mean()[panel] - cg.rho_rule.mean()[panel]))
        mac(f"Blend{w}D", sgn(cg.rho_blend.mean()[panel] - cg.rho_rule.mean()[panel]))
    # ---- MC baseline accuracy
    mb = MCB.groupby(["panel", "k"]).rho.mean()
    mac("RealMCFiftyRho", num(mb[("real", 50)])); mac("RealMCTwoHundredRho", num(mb[("real", 200)]))
    # ---- ground-truth noise
    Mdev = GRID["dev"].M
    mac("WSDenseCeiling", num(Mdev[(Mdev.family == "ws") & (Mdev.prange == "R5")]["ceiling"].mean(), 2))
    # ---- stage run times recorded in logs/ (used by the documentation)
    import re as _re
    logs = ROOT / "logs"

    def elapsed(fname, pattern):
        p = logs / fname
        if not p.exists():
            return "---"
        hits = _re.findall(pattern, p.read_text())
        return str(int(hits[-1])) if hits else "---"
    for st, pan, w in (("grid", "dev", "GridDev"), ("grid", "test", "GridTest"), ("confirm", "test", "ConfTest"),
                       ("confirm", "prior", "ConfPrior"), ("confirm", "real", "ConfReal"),
                       ("confirm", "scale", "ConfScale")):
        f = "grid_devtest.log" if st == "grid" else "confirm.log"
        mac(f"Run{w}", elapsed(f, rf"\[{st}/{pan}\] \d+/\d+ elapsed (\d+)s"))
    tot = 0
    for pan in ("dev", "test", "prior", "real", "scale"):
        v = elapsed("build_panels.log", rf"\[{pan}\] done: \d+ instances in (\d+)s")
        tot += int(v) if v != "---" else 0
    mac("RunBuild", tot)
    import platform
    mac("PyVersion", platform.python_version())
    import scipy, networkx, matplotlib
    mac("NpVersion", np.__version__); mac("SpVersion", scipy.__version__); mac("PdVersion", pd.__version__)
    mac("NxVersion", networkx.__version__); mac("MplVersion", matplotlib.__version__)
    mac("NCores", os.cpu_count())


# =============================================================================
# tables
# =============================================================================
LBL = {
    "en": {
        "base:rule_EW": "Calibrated scalar rule [4]",
        "base:fixed085": "Fixed ${\\alpha=0.85}$",
        "base:rule_WT": "Calibrated push PPR [4]",
        "base:local_pesc": "Local ${\\alpha_i=P_{\\mathrm{esc}}(i)}$",
        "oracle:scalar": "Scalar oracle",
        "oracle:pernode": "Per-node oracle",
        "ref:ceiling": "Split-sample reference",
        "ref:MP2": "KNZ message passing (2 it.)",
        "true:prop": "Law with true ${y^*}$: ${\\alpha_i=1-x_i/y^*_i}$",
        "true:ap1.0": "Anchored law with true ${y^*}$ (${\\eta=1}$)",
        "sel:A_best": "A: raw ${P}$, excess ${\\gamma=16}$, ${T=2}$",
        "sel:B_best": "B: ${EP}$, excess ${\\gamma=16}$, ${T=2}$",
        "sel:C_best": "C: ${EP}$, excess ${\\gamma=16}$, ${T=2}$",
        "sel:C_dir1": "C: raw ${P}$, ${T=3}$ (Dirichlet pick)",
        "sel:B_unif": "B: push, excess ${\\gamma=16}$, ${T=50}$ (uniform pick)",
        "sel:AP_eta0.25": "${\\mathrm{AP}(\\eta=0.25)}$, fixed point",
        "sel:AP_eta1": "${\\mathrm{AP}(\\eta=1)}$, fixed point",
        "sel:PB_rule_best": "Rule ${+}$ AP(${\\eta=-0.25}$), ${K=1}$, ${\\beta=0.25}$",
        "sel:PB_rule_nominal": "Rule ${+}$ AP(${\\eta=1}$), ${K=1}$, ${\\beta=0.5}$",
        "sel:PB_rule_eta1": "Rule ${+}$ AP(${\\eta=1}$) s.c., ${K=\\infty}$, ${\\beta=0.25}$ (rank)",
        "sel:PB_mp2_best": "MP ${+}$ AP(${\\eta=0.25}$), ${K=5}$, ${\\beta=0.75}$ (rank)",
        "sel:PB_mp2_eta0": "MP ${+}$ scalar rule, ${K=5}$, ${\\beta=0.75}$ (rank)",
        "sel:PB_mp2_eta1": "MP ${+}$ AP(${\\eta=1}$), ${K=5}$, ${\\beta=0.75}$ (rank)",
        "sel:PBX_rule_best": "Rule ${+}$ C on raw ${P}$, ${\\gamma=4}$, ${K=5}$, ${\\beta=0.5}$",
        "sel:PBX_rule_unif": "Rule ${+}$ A on raw ${P}$, ${\\gamma=1}$, ${K=1}$, ${\\beta=0.5}$",
        "sel:PBX_rule_dir1": "Rule ${+}$ C on raw ${P}$, ${\\gamma=1}$, ${K=5}$, ${\\beta=0.5}$ (rank)",
        "sel:PBX_rule_nominalB": "Mode (b) as written: rule ${+}$ B, ${K=1}$, ${\\beta=0.5}$",
        "sel:CTRL_Cmean": "Control: C law ${\\to}$ per-step mean",
        "sel:CTRL_Cperm": "Control: C law ${\\to}$ permuted",
        "sel:CTRL_lin0.5": "Control: constant ${\\alpha=0.5}$ on raw ${P}$",
        "sel:CTRL_lin0.9": "Control: constant ${\\alpha=0.9}$ on raw ${P}$ (rank)",
    },
    "gr": {
        "base:rule_EW": "Βαθμονομημένος βαθμωτός κανόνας [4]",
        "base:fixed085": "Σταθερό ${\\alpha=0.85}$",
        "base:rule_WT": "Βαθμονομημένο PPR ώθησης [4]",
        "base:local_pesc": "Τοπικό ${\\alpha_i=P_{\\mathrm{esc}}(i)}$",
        "oracle:scalar": "Βαθμωτό μαντείο",
        "oracle:pernode": "Μαντείο ανά κόμβο",
        "ref:ceiling": "Αναφορά διαιρεμένου δείγματος",
        "ref:MP2": "Μετάδοση μηνυμάτων KNZ (2 επαν.)",
        "true:prop": "Νόμος με αληθές ${y^*}$: ${\\alpha_i=1-x_i/y^*_i}$",
        "true:ap1.0": "Αγκυρωμένος νόμος με αληθές ${y^*}$ (${\\eta=1}$)",
        "sel:A_best": "Α: ακατέργαστος ${P}$, περίσσεια ${\\gamma=16}$, ${T=2}$",
        "sel:B_best": "Β: ${EP}$, περίσσεια ${\\gamma=16}$, ${T=2}$",
        "sel:C_best": "Γ: ${EP}$, περίσσεια ${\\gamma=16}$, ${T=2}$",
        "sel:C_dir1": "Γ: ακατέργαστος ${P}$, ${T=3}$ (επιλογή Dirichlet)",
        "sel:B_unif": "Β: ώθηση, περίσσεια ${\\gamma=16}$, ${T=50}$ (επιλογή ομοιόμ.)",
        "sel:AP_eta0.25": "${\\mathrm{AP}(\\eta=0.25)}$, σταθερό σημείο",
        "sel:AP_eta1": "${\\mathrm{AP}(\\eta=1)}$, σταθερό σημείο",
        "sel:PB_rule_best": "Κανόνας ${+}$ AP(${\\eta=-0.25}$), ${K=1}$, ${\\beta=0.25}$",
        "sel:PB_rule_nominal": "Κανόνας ${+}$ AP(${\\eta=1}$), ${K=1}$, ${\\beta=0.5}$",
        "sel:PB_rule_eta1": "Κανόνας ${+}$ AP(${\\eta=1}$) α.σ., ${K=\\infty}$, ${\\beta=0.25}$ (τάξ.)",
        "sel:PB_mp2_best": "MP ${+}$ AP(${\\eta=0.25}$), ${K=5}$, ${\\beta=0.75}$ (τάξ.)",
        "sel:PB_mp2_eta0": "MP ${+}$ βαθμωτός κανόνας, ${K=5}$, ${\\beta=0.75}$ (τάξ.)",
        "sel:PB_mp2_eta1": "MP ${+}$ AP(${\\eta=1}$), ${K=5}$, ${\\beta=0.75}$ (τάξ.)",
        "sel:PBX_rule_best": "Κανόνας ${+}$ Γ σε ακατέργ. ${P}$, ${\\gamma=4}$, ${K=5}$, ${\\beta=0.5}$",
        "sel:PBX_rule_unif": "Κανόνας ${+}$ Α σε ακατέργ. ${P}$, ${\\gamma=1}$, ${K=1}$, ${\\beta=0.5}$",
        "sel:PBX_rule_dir1": "Κανόνας ${+}$ Γ σε ακατέργ. ${P}$, ${\\gamma=1}$, ${K=5}$, ${\\beta=0.5}$ (τάξ.)",
        "sel:PBX_rule_nominalB": "Εκδοχή (β) ως έχει: κανόνας ${+}$ Β, ${K=1}$, ${\\beta=0.5}$",
        "sel:CTRL_Cmean": "Έλεγχος: νόμος Γ ${\\to}$ μέση τιμή ανά βήμα",
        "sel:CTRL_Cperm": "Έλεγχος: νόμος Γ ${\\to}$ μετάθεση",
        "sel:CTRL_lin0.5": "Έλεγχος: σταθερό ${\\alpha=0.5}$ σε ακατέργ. ${P}$",
        "sel:CTRL_lin0.9": "Έλεγχος: σταθερό ${\\alpha=0.9}$ σε ακατέργ. ${P}$ (τάξ.)",
    },
}

TXT = {
    "en": dict(method="Method", rho="$\\bar\\rho$", dir="Dir.", unif="Unif.", delta="$\\Delta\\rho$ vs rule [95\\% CI]",
               wtl="W/T/L", holm="$p_{\\mathrm{Holm}}$", p="$p$", gap="Gap\\,\\%", group_ref="References and oracles",
               group_law="The proportional law (frozen on dev)", group_ens="Mode (b) polishers and matched controls",
               panel="Panel", law="Law", mean="Mean twin", perm="Permuted", dlaw="Law $-$ mean [95\\% CI]",
               prior="Prior", kappa="$\\log_{10}\\kappa$", n="$n$", inst="Instances", fam="Families", priors="Priors",
               role="Role", cfg="Configuration", mode="Mode", desc="Description"),
    "gr": dict(method="Μέθοδος", rho="$\\bar\\rho$", dir="Dir.", unif="Ομοιόμ.", delta="$\\Delta\\rho$ ως προς κανόνα [95\\% ΔΕ]",
               wtl="Ν/Ι/Η", holm="$p_{\\mathrm{Holm}}$", p="$p$", gap="Κάλυψη\\,\\%", group_ref="Αναφορές και μαντεία",
               group_law="Ο αναλογικός νόμος (παγωμένος στο σύνολο ανάπτυξης)", group_ens="Εξομαλυντές Εκδοχής (β) και αντίστοιχοι έλεγχοι",
               panel="Σύνολο", law="Νόμος", mean="\\shortstack{Δίδυμο\\\\μέσης τιμής}", perm="Μετάθεση", dlaw="\\shortstack{Νόμος $-$ μέση τιμή\\\\{[95\\% ΔΕ]}}",
               prior="Αρχική κατανομή", kappa="$\\log_{10}\\kappa$", n="$n$", inst="Στιγμιότυπα", fam="Οικογένειες", priors="Αρχ. κατανομές",
               role="Ρόλος", cfg="Διαμόρφωση", mode="Εκδοχή", desc="Περιγραφή"),
}


def tabular(colspec, header, rows, size="\\footnotesize", sep="3.5pt"):
    colspec = colspec.replace("p{", ">{\\raggedright\\arraybackslash}p{")
    out = [f"{{{size}\\setlength{{\\tabcolsep}}{{{sep}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if isinstance(r, str):
            out.append(r)
        else:
            out.append(" & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(out)


def table_env(name, caption, label, body, star=False, notes=None):
    env = "table*" if star else "table"
    s = [f"\\newcommand{{\\{name}}}{{%", f"\\begin{{{env}}}[!htbp]", "\\centering",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}", body]
    if notes:
        s.append(f"\\par\\smallskip\\begin{{minipage}}{{0.97\\linewidth}}\\footnotesize {notes}\\end{{minipage}}")
    s += [f"\\end{{{env}}}}}", ""]
    return "\n".join(s)


def tab_panels(lang):
    gr = lang == "gr"
    rows = [
        (("Ανάπτυξη (dev)" if gr else "Development (dev)"), "100", MAC["NDev"], "9",
         "ομοιόμ., Dir(1)" if gr else "uniform, Dir(1)", "επιλογή διαμορφώσεων" if gr else "configuration search"),
        (("Έλεγχος (test)" if gr else "Test"), "100", MAC["NTest"], "9", "ομοιόμ., Dir(1)" if gr else "uniform, Dir(1)",
         "επιβεβαιωτική αξιολόγηση" if gr else "confirmatory evaluation"),
        (("Αρχικές κατανομές" if gr else "Priors"), "144--150", MAC["NPrior"], "5", "10 τύποι" if gr else "10 types",
         "Πρόταση 2" if gr else "Proposition 2"),
        (("Πραγματικά δίκτυα" if gr else "Real networks"), "15--500", MAC["NReal"], "7",
         "ομοιόμ., Dir(1)" if gr else "uniform, Dir(1)", "μεταφορά (τύπος δικτύου)" if gr else "transfer (genre)"),
        (("Κλίμακα" if gr else "Scale"), "900", MAC["NScale"], "9", "ομοιόμ., Dir(1)" if gr else "uniform, Dir(1)",
         "μεταφορά (μέγεθος)" if gr else "transfer (size)"),
    ]
    t = TXT[lang]
    hdr = [t["panel"], t["n"], t["inst"], t["fam"] if not gr else "Οικογ./γραφ.", t["priors"], t["role"]]
    body = tabular("lrrrll", hdr, [list(r) for r in rows])
    cap = ("Τα πέντε σύνολα στιγμιοτύπων. Κάθε στιγμιότυπο φέρει δύο ανεξάρτητα δείγματα Monte Carlo "
           f"πραγματικής τιμής αναφοράς ({MAC['MGTsp']} κόσμοι το καθένα)." if gr else
           f"The five benchmark panels. Every instance carries two independent Monte-Carlo ground-truth samples "
           f"({MAC['MGT']} worlds each).")
    return table_env("TabPanels", cap, "tab:panels", body)


def ci_str(a):
    return f"{sgn(a['diff'])} [{sgn(a['lo'])}, {sgn(a['hi'])}]"


def conf_rows(lang, methods, panel="test"):
    P = CONF[panel]
    orc, rule = P.mean("oracle:scalar"), P.mean("base:rule_EW")
    rows = []
    for m in methods:
        a = P.vs_rule(m)
        byp = P.by(m)
        is_rule = m == "base:rule_EW"
        holm = HOLM[panel].get(m)
        rows.append([LBL[lang][m], num(a["mean_a"]), num(byp.get("dir1")), num(byp.get("uniform")),
                     "---" if is_rule else ci_str(a), "---" if is_rule else f"{a['win']}/{a['tie']}/{a['loss']}",
                     pval(holm) if holm is not None else ("---" if is_rule else pval(a["p"]) + "$^\\dagger$"),
                     pct(100 * (a["mean_a"] - rule) / (orc - rule), 0)])
    return rows


def tab_confirm(lang):
    t = TXT[lang]
    hdr = [t["method"], t["rho"], t["dir"], t["unif"], t["delta"], t["wtl"], t["holm"], t["gap"]]
    refs = ["base:rule_EW", "base:fixed085", "base:rule_WT", "base:local_pesc", "ref:MP2", "oracle:scalar",
            "oracle:pernode", "ref:ceiling", "true:prop", "true:ap1.0"]
    law = ["sel:A_best", "sel:B_best", "sel:C_best", "sel:C_dir1", "sel:B_unif", "sel:AP_eta0.25", "sel:AP_eta1",
           "sel:PB_rule_best", "sel:PB_rule_nominal", "sel:PB_rule_eta1", "sel:PBX_rule_nominalB"]
    ens = ["sel:PB_mp2_best", "sel:PB_mp2_eta0", "sel:PB_mp2_eta1", "sel:PBX_rule_best", "sel:PBX_rule_unif",
           "sel:PBX_rule_dir1", "sel:CTRL_Cmean", "sel:CTRL_Cperm", "sel:CTRL_lin0.5", "sel:CTRL_lin0.9"]
    rows = [f"\\multicolumn{{8}}{{l}}{{\\textit{{{t['group_ref']}}}}} \\\\"] + conf_rows(lang, refs) + \
           ["\\midrule", f"\\multicolumn{{8}}{{l}}{{\\textit{{{t['group_law']}}}}} \\\\"] + conf_rows(lang, law) + \
           ["\\midrule", f"\\multicolumn{{8}}{{l}}{{\\textit{{{t['group_ens']}}}}} \\\\"] + conf_rows(lang, ens)
    body = tabular(("p{5.0cm}" if lang == "gr" else "p{5.3cm}") + "rrrlrrr", hdr, rows, size="\\scriptsize", sep="2.6pt")
    if lang == "gr":
        cap = (f"Επιβεβαιωτική αξιολόγηση στο ανεξάρτητο σύνολο ελέγχου ($n=100$, {MAC['NTest']} στιγμιότυπα). "
               "Όλες οι διαμορφώσεις «παγώθηκαν» στο σύνολο ανάπτυξης πριν από την εξέταση οποιουδήποτε αποτελέσματος ελέγχου.")
        notes = ("$\\bar\\rho$: μέσος συντελεστής Spearman· Dir./Ομοιόμ.: ανά τύπο αρχικής κατανομής· $\\Delta\\rho$: μέση διαφορά κατά ζεύγη ως προς "
                 "τον βαθμονομημένο κανόνα με 95\\% διάστημα εμπιστοσύνης bootstrap· Ν/Ι/Η: νίκες/ισοπαλίες/ήττες· "
                 "$p_{\\mathrm{Holm}}$: έλεγχος προσημασμένων τάξεων Wilcoxon με διόρθωση Holm επί των "
                 f"{MAC['NFrozen']} παγωμένων διαμορφώσεων ($^\\dagger$: μη διορθωμένο $p$ για αναφορές)· "
                 "Κάλυψη: ποσοστό του χάσματος κανόνα–βαθμωτού μαντείου που καλύπτεται· (τάξ.): συνδυασμός τάξεων· α.σ.: αυτοσυνεπής.")
    else:
        cap = (f"Confirmatory evaluation on the held-out test panel ($n=100$, {MAC['NTest']} instances). "
               "Every configuration was frozen on the development panel before any test result was examined.")
        notes = ("$\\bar\\rho$: mean Spearman correlation; Dir./Unif.: by prior type; $\\Delta\\rho$: paired mean difference to "
                 "the calibrated rule with a 95\\% bootstrap confidence interval; W/T/L: wins/ties/losses; "
                 "$p_{\\mathrm{Holm}}$: Wilcoxon signed-rank test, Holm-corrected over the "
                 f"{MAC['NFrozen']} frozen configurations ($^\\dagger$: uncorrected $p$ for references); "
                 "Gap: share of the rule-to-scalar-oracle gap closed; (rank): rank blend; s.c.: self-consistent.")
    return table_env("TabConfirm", cap, "tab:confirm", body, notes=notes)


def tab_replication(lang):
    gr = lang == "gr"
    hdr = (["\\shortstack[l]{Αρχική\\\\κατανομή}", "$n$", "Κανόνας", "$\\alpha=0.85$", "\\shortstack{Καλύτερη\\\\σταθερά}",
            "\\shortstack{$\\alpha$\\\\σταθεράς}", "Μαντείο", "\\shortstack{\\%\\\\μαντείου}", "\\shortstack{$\\bar\\alpha$\\\\κανόνα}",
            "\\shortstack{$\\bar\\alpha$\\\\μαντείου}"] if gr else
           ["Prior", "$n$", "Rule", "$\\alpha=0.85$", "Best constant", "Its $\\alpha$", "Oracle", "\\% oracle",
            "Rule $\\bar\\alpha$", "Oracle $\\bar\\alpha$"])
    names = {"all": "όλα" if gr else "all", "dir1": "Dirichlet(1)", "uniform": "ομοιόμορφη" if gr else "uniform"}
    rows = []
    for g in ("all", "dir1", "uniform"):
        r = T2.loc[g]
        rows.append([names[g], str(int(r["n"])), num(r["rule"]), num(r["fixed085"]), num(r["best_constant"]),
                     num(r["best_constant_alpha"], 3), num(r["scalar_oracle_in"]), pct(r["pct_of_oracle_in"]),
                     num(r["mean_alpha_rule"], 2), num(r["mean_alpha_oracle"], 2)])
    body = tabular("lrrrrrrrrr", hdr, rows)
    cap = ("Αναπαραγωγή του κανόνα του [4] στα σύνολα ανάπτυξης και ελέγχου ($n=100$). «Καλύτερη σταθερά»: η μία τιμή $\\alpha$ "
           "που μεγιστοποιεί το μέσο $\\rho$ στο σύνολο (επιλογή εκ των υστέρων)· «Μαντείο»: βέλτιστο $\\alpha$ ανά στιγμιότυπο." if gr else
           "Replication of the rule of [4] on the development and test panels ($n=100$). ``Best constant'': the single $\\alpha$ "
           "maximising the panel-mean $\\rho$ (chosen with hindsight); ``Oracle'': best $\\alpha$ per instance.")
    return table_env("TabReplication", cap, "tab:replication", body)


def tab_headroom(lang):
    gr = lang == "gr"
    fam_gr = {"ba": "BA", "barbell": "αλτήρας", "cp": "πυρήνας--περιφέρεια", "er": "ER", "grid": "πλέγμα",
              "rect": "ορθογ. πλέγμα", "sbm": "SBM", "tree": "δένδρο", "ws": "WS", "ALL": "όλα"}
    fam_en = {"ba": "BA", "barbell": "barbell", "cp": "core--periphery", "er": "ER", "grid": "grid",
              "rect": "rectangular grid", "sbm": "SBM", "tree": "tree", "ws": "WS", "ALL": "all"}
    hdr = (["Οικογένεια", "\\shortstack{Βαθμωτό\\\\μαντείο}", "\\shortstack{Μαντείο\\\\ανά κόμβο}",
            "\\shortstack{Αναφορά διαιρ.\\\\δείγματος}", "\\shortstack{Κέρδος\\\\(εκτός δείγματος)}",
            "\\shortstack{Κέρδος\\\\(εντός δείγματος)}"] if gr else
           ["Family", "Scalar oracle", "Per-node oracle", "Split-sample ref.", "Gain (out of sample)", "Gain (in sample)"])
    rows = []
    for f in list(T3.index):
        r = T3.loc[f]
        rows.append([(fam_gr if gr else fam_en)[f], num(r["oracle:scalar"]), num(r["oracle:pernode"]),
                     num(r["ref:ceiling"]), sgn(r["pernode_gain"]), sgn(r["apparent_gain"])])
    rows.insert(len(rows) - 1, "\\midrule")
    body = tabular("lrrrrr", hdr, rows)
    cap = ("Περιθώριο βελτίωσης της απόσβεσης ανά κόμβο: μαντείο ανάβασης κατά συντεταγμένες που προσαρμόζεται στο $y^*_B$ "
           "και αξιολογείται στο ανεξάρτητο $y^*_A$ (ανάπτυξη και έλεγχος)." if gr else
           "Per-node damping headroom: a coordinate-ascent oracle fitted on $y^*_B$ and scored on the independent $y^*_A$ "
           "(development and test panels).")
    return table_env("TabHeadroom", cap, "tab:headroom", body)


def tab_attribution(lang):
    gr = lang == "gr"
    t = TXT[lang]
    cfgs = ["PBX_rule_best", "PBX_rule_unif", "PBX_rule_dir1", "C_best", "B_best", "A_best", "B_unif", "C_dir1",
            "PBX_rule_nominalB"]
    pan = {"test": "έλεγχος" if gr else "test", "real": "πραγματικά" if gr else "real", "scale": "$n=900$"}
    hdr = [t["cfg"], t["panel"], t["law"], t["mean"], t["perm"], t["dlaw"], t["p"]]
    rows = []
    for cfg in cfgs:
        for i, p in enumerate(("test", "real", "scale")):
            r, W = mf_paired(p, cfg)
            rows.append([LBL[lang]["sel:" + cfg] if i == 0 else "", pan[p], num(W["law"].mean()), num(W["mean"].mean()),
                         num(W["perm"].mean()), f"{sgn(r['diff'])} [{sgn(r['lo'])}, {sgn(r['hi'])}]", pval(r["p"])])
        rows.append("\\addlinespace[2pt]")
    rows = rows[:-1]
    body = tabular("p{5.0cm}lrrrlr", hdr, rows, size="\\scriptsize", sep="3pt")
    cap = ("Απόδοση του κέρδους στον νόμο ανά κόμβο: κάθε παγωμένη διαμόρφωση που εκτελεί μια υλοποίηση (νόμος) έναντι της ίδιας "
           "ροής επεξεργασίας με το διάνυσμα $\\alpha$ αντικατεστημένο σε κάθε βήμα από τη μέση τιμή του (δίδυμο μέσης τιμής) ή από τυχαία μετάθεσή του." if gr else
           "Attributing gains to the per-node law: every frozen configuration that runs a realization (law) against the identical "
           "pipeline with its $\\alpha$ vector replaced at every step by its mean (mean twin) or by a random permutation.")
    notes = ("$p$: έλεγχος Wilcoxon για τη διαφορά νόμου--δίδυμου." if gr else "$p$: Wilcoxon test of the law-minus-twin difference.")
    return table_env("TabAttribution", cap, "tab:attribution", body, notes=notes)


def tab_prior(lang):
    gr = lang == "gr"
    names_gr = {"uniform": "ομοιόμορφη", "dir10": "Dirichlet(10)", "logn0.5": "λογαριθμοκανονική(0.5)", "dir3": "Dirichlet(3)",
                "logn1": "λογαριθμοκανονική(1)", "dir1": "Dirichlet(1)", "logn2": "λογαριθμοκανονική(2)",
                "dir0.3": "Dirichlet(0.3)", "seed5_0.1": "5 σπόροι, $\\varepsilon=0.1$", "seed1_0.1": "1 σπόρος, $\\varepsilon=0.1$"}
    names_en = {"uniform": "uniform", "dir10": "Dirichlet(10)", "logn0.5": "log-normal(0.5)", "dir3": "Dirichlet(3)",
                "logn1": "log-normal(1)", "dir1": "Dirichlet(1)", "logn2": "log-normal(2)", "dir0.3": "Dirichlet(0.3)",
                "seed5_0.1": "5 seeds, $\\varepsilon=0.1$", "seed1_0.1": "1 seed, $\\varepsilon=0.1$"}
    hdr = (["Αρχική κατανομή", "$\\log_{10}\\kappa$", "Κανόνας", "AP $\\tau{=}0$", "AP $\\tau{=}0.5$", "AP $\\tau{=}1$",
            "Εξομ. (β) $\\tau{=}0$", "C σε $P$", "$\\alpha{=}0.9$ σε $P$"] if gr else
           ["Prior", "$\\log_{10}\\kappa$", "Rule", "AP $\\tau{=}0$", "AP $\\tau{=}0.5$", "AP $\\tau{=}1$",
            "Polish (b) $\\tau{=}0$", "C on $P$", "$\\alpha{=}0.9$ on $P$"])
    rows = []
    for p in PRIOR_ORDER:
        rows.append([(names_gr if gr else names_en)[p], num(T9.loc[p, "log10_kappa"], 1), num(T9.loc[p, "rule"]),
                     sgn(T9d.loc[p, "d_AP_eta1"]), sgn(T9d.loc[p, "d_AP_eta1_tau0.5"]), sgn(T9d.loc[p, "d_AP_eta1_tau1"]),
                     sgn(T9d.loc[p, "d_PBnom"]), sgn(T9d.loc[p, "d_PBX_best"]), sgn(T9d.loc[p, "d_CTRL_lin0.9"])])
    body = tabular("lrrrrrrrr", hdr, rows, size="\\footnotesize", sep="3pt")
    cap = (f"Το σύνολο ομοιομορφίας της αρχικής κατανομής ({MAC['NPrior']} στιγμιότυπα· ίδιο γράφημα και πιθανότητες ακμών για όλους τους τύπους αρχικής κατανομής). "
           "Η στήλη «Κανόνας» δίνει το μέσο $\\rho$· οι υπόλοιπες τη διαφορά κατά ζεύγη $\\Delta\\rho$ ως προς τον κανόνα." if gr else
           f"The prior-uniformity panel ({MAC['NPrior']} instances; the same graph and edge probabilities across prior types). "
           "``Rule'' is the mean $\\rho$; the other columns are paired differences $\\Delta\\rho$ to the rule.")
    return table_env("TabPrior", cap, "tab:prior", body)


def tab_transfer(lang):
    gr = lang == "gr"
    methods = ["base:rule_EW", "oracle:scalar", "oracle:pernode", "true:prop", "sel:AP_eta1", "sel:PB_rule_nominal",
               "sel:PB_rule_best", "sel:PBX_rule_nominalB", "sel:C_best", "sel:PB_mp2_eta0", "sel:PBX_rule_best",
               "sel:CTRL_Cmean", "sel:CTRL_lin0.9"]
    hdr = ([TXT[lang]["method"], "$\\bar\\rho$ πραγμ.", "$\\Delta\\rho$ πραγμ. [95\\% ΔΕ]", "$p$",
            "$\\bar\\rho$ $n{=}900$", "$\\Delta\\rho$ $n{=}900$ [95\\% ΔΕ]", "$p$"] if gr else
           [TXT[lang]["method"], "$\\bar\\rho$ real", "$\\Delta\\rho$ real [95\\% CI]", "$p$",
            "$\\bar\\rho$ $n{=}900$", "$\\Delta\\rho$ $n{=}900$ [95\\% CI]", "$p$"])
    rows = []
    for m in methods:
        cells = [LBL[lang][m]]
        for p in ("real", "scale"):
            P = CONF[p]
            s = P.s(m)
            if s.empty:
                cells += ["---", "---", "---"]
                continue
            a = P.vs_rule(m)
            if m == "base:rule_EW":
                cells += [num(a["mean_a"]), "---", "---"]
            else:
                hp = HOLM[p].get(m)
                cells += [num(a["mean_a"]), ci_str(a), pval(hp) if hp is not None else pval(a["p"]) + "$^\\dagger$"]
        rows.append(cells)
    body = tabular(("p{4.4cm}" if gr else "p{4.6cm}") + "rlrrlr", hdr, rows, size="\\scriptsize", sep="2.6pt")
    cap = (f"Μεταφορά στα πραγματικά δίκτυα ({MAC['NReal']} στιγμιότυπα) και στο $n=900$ ({MAC['NScale']} στιγμιότυπα· "
           f"το μαντείο ανά κόμβο σε {MAC['NPernodeScale']} από αυτά). $p$: Wilcoxon με διόρθωση Holm επί των παγωμένων διαμορφώσεων "
           "($^\\dagger$: μη διορθωμένο)." if gr else
           f"Transfer to real networks ({MAC['NReal']} instances) and to $n=900$ ({MAC['NScale']} instances; the per-node oracle on "
           f"{MAC['NPernodeScale']} of them). $p$: Wilcoxon, Holm-corrected over the frozen configurations ($^\\dagger$: uncorrected).")
    return table_env("TabTransfer", cap, "tab:transfer", body)


def tab_cost(lang):
    gr = lang == "gr"
    order = ["KNZ MP, 2 it.", "Monte Carlo, 50 worlds", "AP(eta=1), fixed point", "rule + anchored polish (K=1)",
             "calibrated rule [4]", "rule + C-law on raw P (K=5)", "rule + constant-alpha raw P (K=5)",
             "Monte Carlo, 200 worlds", "Monte Carlo, 1000 worlds", "scalar oracle (needs ground truth)"]
    lab_en = {"KNZ MP, 2 it.": "KNZ message passing (2 it.)", "Monte Carlo, 50 worlds": "Batched Monte Carlo, 50 worlds",
              "AP(eta=1), fixed point": "AP($\\eta=1$), fixed point", "rule + anchored polish (K=1)": "Rule $+$ anchored polish ($K=1$)",
              "calibrated rule [4]": "Calibrated rule [4]", "rule + C-law on raw P (K=5)": "Rule $+$ C on raw $P$ ($K=5$)",
              "rule + constant-alpha raw P (K=5)": "Rule $+$ constant $\\alpha$ on raw $P$ ($K=5$)",
              "Monte Carlo, 200 worlds": "Batched Monte Carlo, 200 worlds", "Monte Carlo, 1000 worlds": "Batched Monte Carlo, 1000 worlds",
              "scalar oracle (needs ground truth)": "Scalar oracle (needs ground truth)"}
    lab_gr = {"KNZ MP, 2 it.": "Μετάδοση μηνυμάτων KNZ (2 επαν.)", "Monte Carlo, 50 worlds": "Monte Carlo κατά δέσμες, 50 κόσμοι",
              "AP(eta=1), fixed point": "AP($\\eta=1$), σταθερό σημείο", "rule + anchored polish (K=1)": "Κανόνας $+$ αγκυρωμένη εξομάλυνση ($K=1$)",
              "calibrated rule [4]": "Βαθμονομημένος κανόνας [4]", "rule + C-law on raw P (K=5)": "Κανόνας $+$ Γ σε ακατέργ. $P$ ($K=5$)",
              "rule + constant-alpha raw P (K=5)": "Κανόνας $+$ σταθερό $\\alpha$ σε ακατέργ. $P$ ($K=5$)",
              "Monte Carlo, 200 worlds": "Monte Carlo κατά δέσμες, 200 κόσμοι", "Monte Carlo, 1000 worlds": "Monte Carlo κατά δέσμες, 1000 κόσμοι",
              "scalar oracle (needs ground truth)": "Βαθμωτό μαντείο (απαιτεί αληθή τιμή)"}
    hdr = (["Μέθοδος", "$\\bar\\rho$ ($n{=}100$)", "ms ($n{=}100$)", "$\\bar\\rho$ ($n{=}900$)", "ms ($n{=}900$)"] if gr else
           ["Method", "$\\bar\\rho$ ($n{=}100$)", "ms ($n{=}100$)", "$\\bar\\rho$ ($n{=}900$)", "ms ($n{=}900$)"])
    rows = []
    for k in order:
        r = T12.loc[k]
        rows.append([(lab_gr if gr else lab_en)[k], num(r["rho_test"]), num(r["ms_test"], 2) if np.isfinite(r["ms_test"]) else "---",
                     num(r["rho_scale"]), num(r["ms_scale"], 2) if np.isfinite(r["ms_scale"]) else "---"])
    body = tabular("lrrrr", hdr, rows)
    cap = ("Ακρίβεια (μέσο $\\rho$ σε ολόκληρο το σύνολο) έναντι χρόνου εκτέλεσης από άκρο σε άκρο, σε μία διεργασία "
           "(διάμεσος, ένα στιγμιότυπο με ομοιόμορφη αρχική κατανομή ανά οικογένεια και εύρος πιθανοτήτων)." if gr else
           "Accuracy (mean $\\rho$ over the whole panel) against single-process, end-to-end wall clock "
           "(median over one uniform-prior instance per family and probability range).")
    return table_env("TabCost", cap, "tab:cost", body)


def tab_sweep(lang):
    gr = lang == "gr"
    g = SW.groupby("bucket", observed=True)
    hdr = (["$n$", "\\shortstack{ms\\\\συνδυασμού}", "ms MC-200", "Λόγος", "\\shortstack{$k$ ίσου\\\\κόστους}", "\\shortstack{$\\rho$\\\\συνδυασμού}", "\\shortstack{$\\rho$ MC ίσου\\\\κόστους}",
            "Διαφορά", "Νίκες"] if gr else
           ["$n$", "ms ensemble", "ms MC-200", "Ratio", "Equal-cost $k$", "$\\rho$ ensemble", "$\\rho$ equal-cost MC",
            "Difference", "Wins"])
    rows = []
    for b, G in g:
        rows.append([str(b), num(G.ms_ensemble.median(), 2), num(G.ms_mc200.median(), 2), num(G.ratio.median(), 2),
                     f"{G.k_eq.median():.0f}", num(G.rho_ensemble.median()), num(G.rho_mc_eq.median()),
                     sgn(G.d_eq.median()), pct(100 * (G.d_eq > 0).mean(), 0) + "\\%"])
    body = tabular("lrrrrrrrr", hdr, rows)
    cap = (f"Σάρωση μεγέθους ({MAC['SweepN']} στιγμιότυπα ER, BA, WS και πλέγματος, εύρος R3, μία διεργασία). «Λόγος»: χρόνος MC-200 "
           "προς χρόνο συνδυασμού· «ίσου κόστους»: το Monte Carlo με τον ίδιο χρονικό προϋπολογισμό με τον συνδυασμό (παρεμβολή σε $\\log k$). "
           "Διάμεσοι ανά κλάση μεγέθους." if gr else
           f"Size sweep ({MAC['SweepN']} ER, BA, WS and grid instances, range R3, single process). ``Ratio'': MC-200 time over "
           "ensemble time; ``equal-cost'': Monte Carlo given the ensemble's time budget (interpolated in $\\log k$). Medians per size class.")
    return table_env("TabSweep", cap, "tab:sweep", body)


def tab_secondary(lang):
    gr = lang == "gr"
    methods = ["base:rule_EW", "oracle:scalar", "oracle:pernode", "sel:AP_eta1", "sel:PB_rule_nominal", "sel:PBX_rule_best",
               "sel:CTRL_Cmean", "sel:CTRL_lin0.9"]
    hdr = ([TXT[lang]["method"], "$\\tau$ έλεγχ.", "κορυφ.\\ 10\\% έλεγχ.", "NDCG έλεγχ.", "$\\tau$ πραγμ.", "κορυφ.\\ 10\\% πραγμ.", "NDCG πραγμ."] if gr else
           [TXT[lang]["method"], "$\\tau$ test", "top-10\\% test", "NDCG test", "$\\tau$ real", "top-10\\% real", "NDCG real"])
    F = {}
    for p in ("test", "real"):
        F[p] = pd.DataFrame([r for iid in CONF[p].D for r in CONF[p].D[iid]["full_metrics"]]).groupby("method").mean(numeric_only=True)
    rows = [[LBL[lang][m]] + [num(F[p].loc[m, c]) for p in ("test", "real") for c in ("tau", "topk", "ndcg")] for m in methods]
    body = tabular("p{5.0cm}rrrrrr", hdr, rows, size="\\scriptsize", sep="3pt")
    cap = ("Δευτερεύουσες μετρικές: $\\tau$-b του Kendall, επικάλυψη των κορυφαίων 10\\% και NDCG@10\\% (συνάφεια $=y^*_A$)." if gr else
           "Secondary metrics: Kendall's $\\tau$-b, top-10\\% overlap and NDCG@10\\% (relevance $=y^*_A$).")
    return table_env("TabSecondary", cap, "tab:secondary", body)


def tab_complement(lang):
    gr = lang == "gr"
    g = COMP.groupby("panel")
    hdr = (["Σύνολο", "$\\rho$ κανόνα", "$\\rho$ ακατέργ. $P$", "$\\rho$ συνδυασμού τάξεων", "Συσχ. σφαλμάτων τάξης"] if gr else
           ["Panel", "$\\rho$ rule", "$\\rho$ raw $P$", "$\\rho$ rank blend", "Rank-error correlation"])
    nm = {"test": "έλεγχος" if gr else "test", "real": "πραγματικά" if gr else "real", "scale": "$n=900$"}
    rows = [[nm[p], num(g.rho_rule.mean()[p]), num(g.rho_rawP.mean()[p]), num(g.rho_blend.mean()[p]),
             num(g.err_corr.mean()[p], 2)] for p in ("test", "real", "scale")]
    body = tabular("lrrrr", hdr, rows)
    cap = ("Συμπληρωματικότητα των δύο τελεστών: διάδοση $y\\leftarrow0.1x+0.9Py$ για $K=5$ από την αρχή (ακατέργαστος $P$) και "
           "ο βαθμονομημένος κανόνας. Η χαμηλή συσχέτιση των σφαλμάτων τάξης εξηγεί το κέρδος του συνδυασμού." if gr else
           "Complementarity of the two operators: propagation $y\\leftarrow0.1x+0.9Py$ for $K=5$ from scratch (raw $P$) and the "
           "calibrated rule. The low correlation of their rank errors explains the gain of the blend.")
    return table_env("TabComplement", cap, "tab:complement", body)


def tab_configs(lang):
    """Dictionary of the frozen configurations (for the documentation)."""
    gr = lang == "gr"
    hdr = (["Κλειδί", "Τύπος", "Παράμετροι"] if gr else ["Key", "Type", "Parameters"])
    rows = []
    for k, c in SEL.items():
        par = ", ".join(f"{a}={b}" if a != "T" else f"T={'inf' if b == INF else b}" for a, b in c.items() if a != "type")
        rows.append(["\\path{" + k + "}", c["type"], "\\path{" + par + "}"])
    body = tabular("lll", hdr, rows, size="\\scriptsize", sep="3pt")
    cap = ("Οι παγωμένες διαμορφώσεις (\\texttt{results/selected.json})." if gr else
           "The frozen configurations (\\texttt{results/selected.json}).")
    return table_env("TabConfigs", cap, "tab:configs", body)


def tab_dynamics(lang):
    gr = lang == "gr"
    rw, ex = ("ακατέργ.", "περίσσεια ${\\gamma=16}$") if gr else ("raw", "excess ${\\gamma=16}$")
    kA, kB, kC = ("Α", "Β", "Γ") if gr else ("A", "B", "C")
    sel = [("A|EW|raw|1", f"{kA}, $EW$, {rw}"), ("B|EW|raw|1", f"{kB}, $EW$, {rw}"), ("C|EW|raw|1", f"{kC}, $EW$, {rw}"),
           ("A|EW|l1|1", f"{kA}, $EW$, $L_1$"), ("B|EW|l1|1", f"{kB}, $EW$, $L_1$"), ("A|EW|ex|16", f"{kA}, $EW$, {ex}"),
           ("C|P|raw|1", f"{kC}, $P$, {rw}")]
    hdr = (["Διαμόρφωση", "Αρχ. κατανομή", "τετρ. $t{=}1$", "τετρ. $t{=}5$", "τετρ. $t{=}20$", "τετρ. $t{=}200$", "ενεργ. $t{=}200$"] if gr else
           ["Configuration", "Prior", "trivial $t{=}1$", "trivial $t{=}5$", "trivial $t{=}20$", "trivial $t{=}200$", "active $t{=}200$"])
    rows = []
    for key, name in sel:
        for prior in ("uniform", "dir1"):
            r = T13.loc[(key, prior)]
            rows.append([name if prior == "uniform" else "", ("ομοιόμ." if gr else "uniform") if prior == "uniform" else "Dir(1)",
                         num(r["trivial@t1"], 2), num(r["trivial@t5"], 2), num(r["trivial@t20"], 2),
                         num(r["trivial@t200"], 2), num(r["active@t200"], 2)])
    body = tabular("llrrrrr", hdr, rows)
    cap = ("Κλάσματα κορυφών στον τετριμμένο κλάδο ($y_i=x_i$) και στον ενεργό κλάδο ($S_i>2x_i$) κατά τις επαναλήψεις "
           "(σύνολο ανάπτυξης)." if gr else
           "Fractions of vertices on the trivial branch ($y_i=x_i$) and the active branch ($S_i>2x_i$) over the iterations "
           "(development panel).")
    return table_env("TabDynamics", cap, "tab:dynamics", body)


def build_tables(lang):
    parts = [f"% generated by make_paper_assets.py --- do not edit ({lang})", "",
             tab_panels(lang), tab_confirm(lang), tab_replication(lang), tab_headroom(lang), tab_attribution(lang),
             tab_prior(lang), tab_transfer(lang), tab_cost(lang), tab_sweep(lang), tab_secondary(lang),
             tab_complement(lang), tab_configs(lang), tab_dynamics(lang)]
    return "\n".join(parts)


def write_text_assets():
    GEN.mkdir(parents=True, exist_ok=True)
    build_macros()
    lines = ["% generated by make_paper_assets.py --- do not edit", ""]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in MAC.items()]
    (GEN / "macros.tex").write_text("\n".join(lines) + "\n")
    for lang in ("en", "gr"):
        (GEN / f"tables_{lang}.tex").write_text(build_tables(lang))
    print(f"macros: {len(MAC)}  tables: en, gr")


if __name__ == "__main__":
    import sys
    write_text_assets()
    if "--no-figures" not in sys.argv:
        import paper_figures
        paper_figures.main(sys.modules[__name__])
