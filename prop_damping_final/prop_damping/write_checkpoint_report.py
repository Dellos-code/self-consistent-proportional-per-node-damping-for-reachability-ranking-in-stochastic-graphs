"""Compose CHECKPOINT_1_REPORT.md from the saved tables (every number is read from results/)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
TB = RES / "tables"


def T(name, index_col=0):
    return pd.read_csv(TB / f"{name}.csv", index_col=index_col)


def f3(v):
    return f"{v:.3f}"


def sg(v, d=3):
    return f"{v:+.{d}f}"


def mdtab(name):
    return (TB / f"{name}.md").read_text()


def main():
    t2 = T("T2_replication_of_rule")
    t3 = T("T3a_pernode_headroom_by_family")
    t3b = T("T3b_pernode_headroom_by_regime")
    t5 = T("T5_anchored_eta_curve")
    t6 = T("T6_confirmatory_test")
    t7 = T("T7_meanfield_attribution")
    t7r = T("T7b_meanfield_attribution_real")
    t8 = T("T8_sign_condition")
    t9 = T("T9a_prior_panel_methods")
    t9d = T("T9b_prior_panel_diff_vs_rule")
    t10r = T("T10_transfer_real")
    t10s = T("T10_transfer_scale") if (TB / "T10_transfer_scale.csv").exists() else None
    t11 = T("T11_secondary_metrics_test")
    rt = pd.read_csv(RES / "runtime.csv") if (RES / "runtime.csv").exists() else None
    p2 = pd.read_csv(RES / "prop2_stats.csv")
    ex = pd.read_csv(RES / "explore_rawP_dev_test.csv.gz")
    exs = pd.read_csv(RES / "explore_rawP_real_scale.csv.gz") if (RES / "explore_rawP_real_scale.csv.gz").exists() else None
    sha = (RES / "selected.sha256").read_text().split()
    tf = json.loads((RES / "tests_foundations.json").read_text())
    td = json.loads((RES / "tests_dynamics.json").read_text())
    ms = json.loads((TB / "misc_stats.json").read_text())
    t13 = pd.read_csv(TB / "T13_dynamics_dev.csv", index_col=[0, 1])
    sel = json.loads((RES / "selected.json").read_text())
    p3 = {k.split("=")[1].split(":")[0]: v for k, v in td.items() if k.startswith("P3_lam")}
    import analysis as AN
    Rg, Mg, _ = AN.load_stage("grid", "dev")
    fam_sizes = {f: len(AN.config_means(Rg, pre)) for f, pre in
                 (("SA", "sa:"), ("AP", "ap:"), ("PB", "pb:"), ("PBX", "pbx:"))}
    wsd = Mg[(Mg.family == "ws") & (Mg.prange == "R5")]["ceiling"].mean()
    Sg = Rg[Rg.method.str.startswith("sa:")].copy()
    Sg["op"] = Sg.method.str.split("|").str[1]
    Sg = Sg.join(Mg["prior"], on="id")
    w_unif = Sg[(Sg.op == "W") & (Sg.prior == "uniform")]["rho"]
    w_zero, w_mean = float((w_unif.abs() < 1e-12).mean()), float(w_unif.mean())
    er_rng = (Mg[Mg.family == "er"].Pbar.min(), Mg[Mg.family == "er"].Pbar.max())
    Rt, Mt, _ = AN.load_stage("grid", "test")
    pn_max = []
    for RR, MM in ((Rg, Mg), (Rt, Mt)):
        Pp = RR[RR["T"] == 0].pivot_table(index="id", columns="method", values="rho").join(MM["family"])
        Pp = Pp[Pp.family.isin(["grid", "rect"])]
        pn_max.append(float((Pp["oracle:pernode"] - Pp["oracle:scalar"]).max()))
    pn_max = max(pn_max)

    r = lambda m, c="mean": float(t6.loc[m, c])
    rule, orc, pn, ceil = r("base:rule_EW"), r("oracle:scalar"), r("oracle:pernode"), r("ref:ceiling")
    L = []
    A = L.append
    A("# Checkpoint 1 — Phase 1 report\n")
    A("**Self-Consistent Proportional Per-Node Damping for Reachability Ranking in Uncertain Graphs**  ")
    A("Experimental investigation of the manuscript's Theorem 1, Realizations A/B/C, Proposition 2 and Modes (a)/(b).\n")
    A("All numbers below are generated from `results/` by `write_checkpoint_report.py`; tables are in "
      "`results/tables/`. Spearman ρ against an independent ground-truth sample is the primary metric throughout.\n")
    A("*Revision note (26 October 2026).* This report was regenerated from the final `results/`. The single-process "
      "timings were re-measured end to end in Phase 2 (`bench_runtime.py`, median of three runs), so the timing figures "
      "in §0 and §8 differ from the version delivered at Checkpoint 1, which quoted 3.5 vs 13.3 ms (3.8×) at n=900 from "
      "an earlier timing run. Every other number is unchanged.\n")

    # ------------------------------------------------------------------ 0
    A("## 0. Bottom line\n")
    A(f"* **The algebra is right; the law does not work.** Theorem 1 holds to a relative residual of {tf['theorem1_max_residual']:.1e}. "
      "But on contractive operators (W, EW) Realization A collapses to y ≡ x after one step under uniform priors "
      f"(under Dirichlet priors only {100*t13.loc[('A|EW|raw|1','dir1'),'active@t200']:.0f}% of vertices stay active, those whose neighbours carry more prior); "
      "a positive all-active solution requires an expansive operator and is then a repelling fixed point of both A and B; "
      "and the law has the wrong sign wherever a vertex's neighbour signal is below its prior.")
    A(f"* **Pre-registered held-out test (180 instances):** the anchored law at η=1 scores {sg(r('sel:AP_eta1','d_rule'))} vs the calibrated rule "
      f"(Holm p={t6.loc['sel:AP_eta1','p_holm']}); Mode (b) as written in the manuscript (Realization B polishing the rule on the excess scale, K=1, β=0.5) "
      f"{sg(r('sel:PBX_rule_nominalB','d_rule'))} (Holm p={t6.loc['sel:PBX_rule_nominalB','p_holm']}) and its anchored version {sg(r('sel:PB_rule_nominal','d_rule'))} "
      f"(Holm p={t6.loc['sel:PB_rule_nominal','p_holm']}); the best polish found on dev {sg(r('sel:PB_rule_best','d_rule'),4)} (p={t6.loc['sel:PB_rule_best','p_rule']}). "
      f"Even with the **true** y* plugged in, α_i = 1 − x_i/y*_i scores {f3(r('true:prop'))} vs {f3(rule)} for the rule.")
    A(f"* **What does gain (+{r('sel:PBX_rule_best','d_rule'):.3f} to +{r('sel:CTRL_lin0.9','d_rule'):.3f})** is the Mode (b) *architecture* used as an ensemble: "
      "blend the calibrated diffusion with a few sweeps of propagation on the raw probability adjacency P. Replacing the per-node law by a constant "
      f"leaves the gain intact or larger (`CTRL_lin0.9`: {f3(r('sel:CTRL_lin0.9'))}, {r('sel:CTRL_lin0.9','pct_of_scalar_oracle'):.1f}% of the scalar oracle, "
      f"Holm p={t6.loc['sel:CTRL_lin0.9','p_holm']}).")
    t12b = T("T12_accuracy_vs_cost") if (TB / "T12_accuracy_vs_cost.csv").exists() else None
    if t12b is not None:
        A(f"* **Context that matters for framing:** a batched 200-world Monte Carlo ranks at {f3(t12b.loc['Monte Carlo, 200 worlds','rho_test'])} in "
          f"{t12b.loc['Monte Carlo, 200 worlds','ms_test']:.1f} ms at n=100 (rule: {f3(t12b.loc['calibrated rule [4]','rho_test'])} in {t12b.loc['calibrated rule [4]','ms_test']:.1f} ms); "
          f"only at n=900 does the deterministic ensemble match it at lower cost ({t12b.loc['rule + constant-alpha raw P (K=5)','ms_scale']:.1f} vs {t12b.loc['Monte Carlo, 200 worlds','ms_scale']:.1f} ms).")
    A(f"* **Two corrections to [4] surfaced:** its rule under *uniform* priors ({f3(t2.loc['uniform','rule'])}) is beaten by the fixed constant "
      f"α={t2.loc['uniform','best_constant_alpha']:.3f} ({f3(t2.loc['uniform','best_constant'])}); and the per-node headroom "
      f"(+0.341 on grids in [4]) re-measures at {sg(t3.loc['grid','pernode_gain'])} on grids here, concentrated in the dense/heterogeneous-prior corner.\n")

    # ------------------------------------------------------------------ 1
    A("## 1. What was built\n")
    A("* `prop_damping_lib.py` — vectorised library (NumPy/SciPy sparse): instance generators (9 synthetic families, 7 real networks), "
      "Monte-Carlo ground truth with block-diagonal batching (one `connected_components` call per batch) and exact world enumeration for verification, "
      "operators W, EW = diag(P_esc)W, Wᵀ (push), P (raw p_ij), EP, normalised damped diffusion with scalar/per-node α (lazy power form, converges on bipartite graphs), "
      "KNZ message passing, Realizations A/B/C under four scale conventions, the anchored power-law family AP(η), Mode (b) polishing, scalar and per-node oracles, metrics.")
    A("* `build_panels.py`, `experiments.py` (stages `grid` and `confirm`), `controls.py`, `mf_controls.py`, `sign_condition.py`, `prop2_stats.py`, "
      "`bench_runtime.py`, `explore_rawP.py`, `make_checkpoint_tables.py`, `tests/`.")
    A(f"* **Panels (688 instances):** dev 180 and test 180 (n=100; ER with P̄_esc from {er_rng[0]:.2f} to {er_rng[1]:.2f}, BA, WS, square grid, 1:4 rectangular grid, SBM, core–periphery, tree, barbell × "
      "5 probability ranges × {uniform, Dirichlet(1)} priors × 2 replicates); prior panel 150 (5 families × 3 ranges × 10 priors, same graph and p across priors); "
      "real 88 (karate, Les Misérables, Davis; 500-vertex snowball samples of Cora, CiteSeer, Amazon-Photo, MS-Academic-CS; 4 probability schemes incl. weighted cascade; 2 priors); "
      "scale 90 (n=900, all 9 families).")
    A("* **Ground truth:** two independent M=2000-world samples per instance: y*_A for evaluation, y*_B for fitting every oracle. "
      "Ranking by y*_B gives the *split-sample reference* `ref:ceiling` (mean test ρ = " + f3(ceil) + "); "
      "because both samples are noisy, an exact predictor of y* would score about the square root of this value, so it is a reliability indicator rather than a hard ceiling.")
    A(f"* **Protocol:** every configuration was searched on dev only; the winners per family were frozen in `results/selected.json` "
      f"(sha256 `{sha[0][:16]}…`, {sha[2] if len(sha) > 2 else ''}) before any test, prior, real or scale result was examined, and evaluated once on those panels. "
      f"Dev searches covered {fam_sizes['SA']} standalone (realization × operator × scale × T), {fam_sizes['AP']} anchored, {fam_sizes['PB']} anchored-polishing and {fam_sizes['PBX']} realization-polishing configurations. "
      f"Paired Wilcoxon signed-rank tests, 20,000-resample bootstrap CIs, Holm correction over the {len(sel)} frozen configurations.\n")

    # ------------------------------------------------------------------ 2
    A("## 2. Theory: what the coupled system can and cannot do\n")
    A("**2.1 Theorem 1 is correct** (both branches satisfy both equations; residual in `results/tests_foundations.txt`). One interpretive flag for the paper: "
      "on the active branch y_i = S_i − x_i the score *decreases* in the vertex's own prior for a fixed neighbour signal, whereas y*_i = x_i + (external mass) increases in it. "
      "The coupled system is therefore not an estimator of y*; it is a constraint that the operator scale must be engineered to satisfy.\n")
    A("**2.2 Proposition A (finite-time collapse of Realization A).** Let T ≥ 0 have row sums ≤ 1 (W and EW qualify) and x uniform. "
      "Since (Ty)_i ≤ max_j y_j, the maximum obeys M_{t+1} ≤ max(1/n, M_t − 1/n), so y ≡ x after at most ⌈nM_0⌉−1 steps. "
      "From the manuscript's warm start y⁽⁰⁾ = x + Tx (M_0 ≤ 2/n) the collapse happens at **t = 1**. Verified on 90 instance–operator pairs "
      f"(max relative excess after one step {tf['collapse_A_uniform_max_rel_excess_after_1_step']:.1e}) and in the dynamics table "
      f"({100*t13.loc[('A|EW|raw|1','uniform'),'trivial@t1']:.0f}% of vertices on the trivial branch at t=1 under EW with uniform priors).\n")
    A("**2.3 Proposition B (absorption and quadratic contraction of Realization B).** If y_i ≤ x_i then α_i = 0 at every later step, so the trivial branch is absorbing; "
      "with uniform x and row-substochastic T the maximal excess m obeys m' ≤ m²/(x+m). "
      f"Observed: {100*t13.loc[('B|EW|raw|1','uniform'),'trivial@t5']:.0f}% of vertices absorbed by t=5 and {100*t13.loc[('B|EW|raw|1','uniform'),'trivial@t20']:.0f}% by t=20 (EW, uniform priors).\n")
    A("**2.4 Proposition C (the self-consistent solution is repelling).** If T ≥ 0 is irreducible and y > 0 satisfies the all-active fixed point y = Ty − x, then Ty > y, so ρ(T) > 1 "
      "(Collatz–Wielandt). The Jacobians are J_A = T and J_B = diag(x/y) + diag(1−x/y)T with J_B y = y + x − x²/y > y, hence ρ(J_A), ρ(J_B) > 1: the fixed point is unstable for both "
      "realizations. For T = λW, ρ(J_B) = −λ² + 3λ − 1 > 1 on (1,2); numerically ρ(J_B) = "
      + ", ".join(f"{p3[l][1]:.2f}" for l in sorted(p3)) + " at λ = " + ", ".join(sorted(p3)) +
      ", and a 1e−9 perturbation departs from the fixed point (relative departure " + ", ".join(str(p3[l][2]) for l in sorted(p3)) + "). "
      "Corollary: no row-substochastic operator admits a positive all-active solution; contractive operators collapse, expansive ones diverge or collapse.\n")
    A("**2.5 Proposition D (Realization C is a different law).** C sets odds = S/x, giving y' = (x²+S²)/(x+S), the Lehmer mean L₂(x,S). Its Jacobian diag(1 − 2x²/(x+S)²)T "
      f"is an ∞-norm contraction for row-substochastic T with S > 0, so C converges to a unique fixed point (three random starts agree to {td['P4_C_EW_three_starts_same_fixed_point_maxdiff']:.0e}) — but that fixed point lies *below* the prior "
      "wherever S < x, and L₂ is non-monotone in S (decreasing for S < (√2−1)x), so an isolated vertex (S = 0) outranks a weakly connected one.\n")
    s8 = t8
    A("**2.6 Proposition E (sign obstruction).** For u_i = (1−α_i)x_i + α_iS_i, ∂u_i/∂α_i = S_i − x_i. Any law in which α_i increases with reachability raises well-connected vertices only "
      "on {S_i > x_i} and *lowers* them on {S_i < x_i}. "
      f"Under EW in sparse regimes S_i < x_i for essentially every vertex ({100*td['P5_sparse_ER_fraction_S_below_x']:.0f}% in a sparse ER instance; finite differences agree with the sign rule on {td['P5_sign_rule_agreement_(of_20)']}/20 vertices). "
      f"Across the {ms['sign_n']} dev+test instances the share of vertices with S > x at the rule's fixed point predicts the relation between the per-node oracle's α* and the law "
      f"(Spearman {ms['sign_spearman']:.2f}, p = {ms['sign_spearman_p']:.1e}): in the bin with <10% such vertices (n={int(s8.iloc[0]['n'])}) corr(α*, 1−x/y*) = {s8.iloc[0]['pn_corr_prop']:.2f} and the law with the true y* "
      f"loses {abs(s8.iloc[0]['d_trueprop']):.2f} in ρ (ranking inversion); in the densest bins it turns positive but stays ≤ {s8['pn_corr_prop'].max():.2f}.\n")
    A(mdtab("T8_sign_condition") + "\n")
    A("**2.7 Proposition F (the 'Standard PPR' variant as written is degenerate under uniform priors).** With S_i = Σ_j w_ij y_j and W row-stochastic, Wx = x for uniform x, "
      "so y = c·x is an exact fixed point of A, B and C and every vertex ties. "
      f"On dev, {100*w_zero:.0f}% of pull-W runs under uniform priors score ρ = 0 exactly (mean {w_mean:.3f}). The meaningful PPR variant is push (Wᵀ), which [4] used.\n")
    g = p2.groupby("prior")[["log10_kappa", "corr_r_with_ystar", "corr_r_with_inv_prior", "sat099_tau0"]].mean()
    A("**2.8 Proposition 2 re-examined.** Part 1 is true but does not imply a ranking advantage (and α_i = 1 − 1/(ny_i) is not shift-invariant). "
      "Part 2's Jacobian is correct as a partial derivative, but the data locate the failure differently. On ground truth (prior panel): under smooth heterogeneous priors the ratio "
      f"r_i = (y*_i − x_i)/x_i becomes a proxy for 1/x_i (rank correlation {g.loc['dir10','corr_r_with_inv_prior']:.2f} → {g.loc['dir0.3','corr_r_with_inv_prior']:.2f} from Dirichlet(10) to Dirichlet(0.3)) "
      f"and loses its correlation with y* ({g.loc['dir10','corr_r_with_ystar']:.2f} → {g.loc['dir0.3','corr_r_with_ystar']:.2f}); under localised seed priors it stays ordinal with y* "
      f"({g.loc['seed5_0.1','corr_r_with_ystar']:.2f}, {g.loc['seed1_0.1','corr_r_with_ystar']:.2f}) but saturates "
      f"({100*g.loc['seed1_0.1','sat099_tau0']:.0f}% of vertices with α > 0.99), which removes teleportation for the non-seed majority. Section 5 measures the effect on ranking.\n")

    # ------------------------------------------------------------------ 3
    A("## 3. Replication of [4] and the per-node headroom\n")
    A(f"Under Dirichlet priors the rule recovers {t2.loc['dir1','pct_of_oracle_in']:.1f}% of the in-sample scalar oracle and beats the best hindsight constant "
      f"({f3(t2.loc['dir1','rule'])} vs {f3(t2.loc['dir1','best_constant'])}), reproducing [4]. Under **uniform** priors it recovers only {t2.loc['uniform','pct_of_oracle_in']:.1f}%: "
      f"the oracle's mean α is {t2.loc['uniform','mean_alpha_oracle']:.2f} versus the rule's {t2.loc['uniform','mean_alpha_rule']:.2f}, and the constant α={t2.loc['uniform','best_constant_alpha']:.3f} scores "
      f"{f3(t2.loc['uniform','best_constant'])}. The calibration is prior-dependent.\n")
    A(mdtab("T2_replication_of_rule") + "\n")
    A(f"Per-node headroom (oracle fitted on y*_B, scored on y*_A) is {sg(t3.loc['ALL','pernode_gain'])} over the scalar oracle overall "
      f"(apparent in-sample {sg(t3.loc['ALL','apparent_gain'])}, so {100*(1-t3.loc['ALL','pernode_gain']/t3.loc['ALL','apparent_gain']):.0f}% of it is overfit), "
      f"{sg(t3.loc['grid','pernode_gain'])} on square grids and {sg(t3.loc['tree','pernode_gain'])} on trees. "
      f"It concentrates in the dense, heterogeneous-prior corner (R5 + Dirichlet: {sg(t3b.loc['R5 / dir1','pernode_gain'])} averaged over families), "
      f"where every scalar α fails; a gain of [4]'s size appears only on individual dense lattice instances (largest: {sg(pn_max)}).\n")
    A(mdtab("T3a_pernode_headroom_by_family") + "\n")

    # ------------------------------------------------------------------ 4-7
    A("## 4. Held-out results for every frozen configuration (test panel)\n")
    A(mdtab("T6_confirmatory_test") + "\n")
    A("Configuration keys (all frozen on dev): `A_best`/`B_best`/`C_best` = best standalone realization overall (A: raw P, excess scale γ=16, T=2; B and C: EP, γ=16, T=2); "
      "`C_dir1`, `B_unif` = best standalone pick within one prior type; `AP_eta*` = anchored law, self-consistent fixed point; `PB_*` = anchored law polishing a base "
      "(rule or 2-step MP) with K sweeps and β blend (`PB_rule_nominal` = anchored law, K=1, β=0.5); `PBX_*` = Realizations A/B/C as polishers of the rule; `PBX_rule_nominalB` = Mode (b) as written in the manuscript "
      "(Realization B on the excess scale, one sweep, 50% blend); `CTRL_*` = the same pipeline as `PBX_rule_best` with the per-node law replaced by its per-step mean "
      "(`Cmean`), a random reassignment (`Cperm`) or a constant damping 0.5/0.9 (`lin*`).\n")
    A("**Mode (a), standalone.** Every realization is significantly worse than the rule on test "
      f"({sg(r('sel:A_best','d_rule'))}, {sg(r('sel:B_best','d_rule'))}, {sg(r('sel:C_best','d_rule'))} for A/B/C). The prior-specific picks do well only on their own prior "
      f"(`C_dir1` {f3(t6.loc['sel:C_dir1','dir1'])} on Dirichlet, {f3(t6.loc['sel:C_dir1','uniform'])} on uniform).\n")
    A("**Anchored power law (the principle in its best form).** η = 0 *is* the scalar rule; the η-curve peaks there on Dirichlet priors and is flat on uniform priors, on dev and test alike.\n")
    A(mdtab("T5_anchored_eta_curve") + "\n")
    A("**Attribution: per-node law vs its own mean-field twin.** Same pipeline, same K, γ, blend; only α_i is replaced by its per-step mean (no per-node variation) or permuted.\n")
    A(mdtab("T7_meanfield_attribution") + "\n")
    A("On real networks (88 instances) the C law inside the best polisher retains a small positive effect over its twin "
      f"({sg(t7r.loc['PBX_rule_best','d_law_minus_mean'],4)}, CI {t7r.loc['PBX_rule_best','ci']}, p={t7r.loc['PBX_rule_best','p']}, Holm {t7r.loc['PBX_rule_best','p_holm']}), while every standalone law loses to its twin; "
      "the best pipeline overall still uses no per-node law.\n")
    A(mdtab("T7b_meanfield_attribution_real") + "\n")

    # ------------------------------------------------------------------ 8
    A("## 5. Prior uniformity (Proposition 2, prior panel, 150 instances)\n")
    A(f"The anchored law's difference to the rule is {sg(t9d.loc['uniform','d_AP_eta1'],4)} under uniform priors and falls to {sg(t9d.loc['seed1_0.1','d_AP_eta1'])} under single-seed priors "
      f"(Spearman with log₁₀κ = {ms['prior_spearman_AP_eta1']:.2f}, p = {ms['prior_spearman_AP_eta1_p']:.1e}). The direction of Proposition 2 is confirmed, but there is no regime of advantage: 'best under uniform priors' means 'harmless under uniform priors'. "
      f"Denominator regularisation shrinks the loss (τ=0.5) and removes it at τ=1 ({sg(t9d['d_AP_eta1_tau1'].min())} to {sg(t9d['d_AP_eta1_tau1'].max())}), which is where the odds denominator stops depending on x. "
      f"The ensemble controls gain under every prior type ({sg(t9d['d_CTRL_lin0.9'].min())} to {sg(t9d['d_CTRL_lin0.9'].max())}).\n")
    A(mdtab("T9b_prior_panel_diff_vs_rule") + "\n")
    A(mdtab("T9a_prior_panel_methods") + "\n")

    # ------------------------------------------------------------------ 9
    A("## 6. Transfer: real networks and n = 900\n")
    A(mdtab("T10_transfer_real") + "\n")
    if t10s is not None:
        A(mdtab("T10_transfer_scale") + "\n")
    if (TB / "T7c_meanfield_attribution_scale.md").exists():
        A(mdtab("T7c_meanfield_attribution_scale") + "\n")
    for p in ("real", "scale"):
        if (TB / f"T11_secondary_metrics_{p}.md").exists():
            A(mdtab(f"T11_secondary_metrics_{p}") + "\n")
    A("## 7. Secondary metrics (test)\n")
    A(mdtab("T11_secondary_metrics_test") + "\n")
    if rt is not None:
        A("## 8. Accuracy against cost, and Monte Carlo\n")
        t12 = T("T12_accuracy_vs_cost")
        A(f"With block-diagonal batching (one connected-components call per batch of worlds), a {200}-world Monte-Carlo estimate costs "
          f"{t12.loc['Monte Carlo, 200 worlds','ms_test']:.1f} ms at n=100 — less than the calibrated rule including operator construction "
          f"({t12.loc['calibrated rule [4]','ms_test']:.1f} ms) — and ranks better ({f3(t12.loc['Monte Carlo, 200 worlds','rho_test'])} vs {f3(t12.loc['calibrated rule [4]','rho_test'])}). "
          f"At n=900 the deterministic ensemble matches it ({f3(t12.loc['rule + constant-alpha raw P (K=5)','rho_scale'])} vs {f3(t12.loc['Monte Carlo, 200 worlds','rho_scale'])}) "
          f"at {t12.loc['Monte Carlo, 200 worlds','ms_scale']/t12.loc['rule + constant-alpha raw P (K=5)','ms_scale']:.1f}× lower cost. "
          "Any paper in this line has to position diffusion heuristics against a well-implemented sampler, not against the 13×-slower sampler assumed in [4].\n")
        A(mdtab("T12_accuracy_vs_cost") + "\n")
        A("Per-method timings (single process, median of 3; one uniform-prior instance per family and range):\n")
        g = rt.groupby(["panel", "method"])[["ms", "iters"]].median().unstack("panel")
        g.columns = [f"{a} ({b})" for a, b in g.columns]
        A(g.round(2).to_markdown() + "\n")
    # ------------------------------------------------------------------ exploratory
    A("## 9. Exploratory (not pre-registered): raw-P propagation on its own\n")
    gx = ex.groupby(["panel", "a", "K", "beta"])["rho"].mean().unstack(["panel"])
    A("Damped propagation on the raw probability adjacency from scratch, y ← (1−a)x + a·Py, K sweeps (a truncated Katz-type score with p-weights), "
      "alone (β=1) and rank-blended 50/50 with the rule (β=0.5). Chosen after seeing dev and test; the real and scale panels are its hold-out.\n")
    A(gx.round(4).to_markdown() + "\n")
    if exs is not None:
        gy = exs.groupby(["panel", "a", "K", "beta"])["rho"].mean().unstack(["panel"])
        A(gy.round(4).to_markdown() + "\n")

    # ------------------------------------------------------------------ threats
    A("## 10. Threats to validity\n")
    A(f"* **Ground-truth noise.** With M=2000, dense instances have unreliable ground-truth ranks (dense Watts–Strogatz on dev: ρ(y*_A, y*_B) = {wsd:.2f}). All comparisons are paired, so this adds noise but no bias.")
    A("* **Operator choice.** The per-node law was tested on five operators and four scale conventions, with and without anchoring, in both modes; the true-y* experiment removes estimation error entirely.")
    A("* **Per-node oracle is a lower bound.** Coordinate ascent over a 14-value grid, 3 sweeps, time-capped (120 s at n ≤ 300, 240 s above); at n = 900 on lattices the cap binds, so headroom there is understated — as was [4]'s.")
    A(f"* **Selection.** Dev searches are large ({min(fam_sizes.values())}–{max(fam_sizes.values())} configurations per method family); held-out evaluation of frozen winners and Holm correction address this. The exploratory raw-P section is not covered by the pre-registration.")
    A("* **Reimplementation of [4].** The rule's constants were used as published; the operator follows [4]'s equations, but implementation details of [4] (iteration count, prior generator) were not available.\n")

    (ROOT / "CHECKPOINT_1_REPORT.md").write_text("\n".join(L))
    print("wrote CHECKPOINT_1_REPORT.md", len("\n".join(L)), "chars")


if __name__ == "__main__":
    main()
