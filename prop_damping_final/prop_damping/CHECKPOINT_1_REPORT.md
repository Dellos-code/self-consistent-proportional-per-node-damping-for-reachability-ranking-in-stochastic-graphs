# Checkpoint 1 — Phase 1 report

**Self-Consistent Proportional Per-Node Damping for Reachability Ranking in Uncertain Graphs**  
Experimental investigation of the manuscript's Theorem 1, Realizations A/B/C, Proposition 2 and Modes (a)/(b).

All numbers below are generated from `results/` by `write_checkpoint_report.py`; tables are in `results/tables/`. Spearman ρ against an independent ground-truth sample is the primary metric throughout.

*Revision note (26 October 2026).* This report was regenerated from the final `results/`. The single-process timings were re-measured end to end in Phase 2 (`bench_runtime.py`, median of three runs), so the timing figures in §0 and §8 differ from the version delivered at Checkpoint 1, which quoted 3.5 vs 13.3 ms (3.8×) at n=900 from an earlier timing run. Every other number is unchanged.

## 0. Bottom line

* **The algebra is right; the law does not work.** Theorem 1 holds to a relative residual of 3.2e-12. But on contractive operators (W, EW) Realization A collapses to y ≡ x after one step under uniform priors (under Dirichlet priors only 30% of vertices stay active, those whose neighbours carry more prior); a positive all-active solution requires an expansive operator and is then a repelling fixed point of both A and B; and the law has the wrong sign wherever a vertex's neighbour signal is below its prior.
* **Pre-registered held-out test (180 instances):** the anchored law at η=1 scores -0.012 vs the calibrated rule (Holm p=3.4e-06); Mode (b) as written in the manuscript (Realization B polishing the rule on the excess scale, K=1, β=0.5) -0.116 (Holm p=2.9e-08) and its anchored version -0.005 (Holm p=5e-06); the best polish found on dev +0.0002 (p=0.766). Even with the **true** y* plugged in, α_i = 1 − x_i/y*_i scores 0.488 vs 0.844 for the rule.
* **What does gain (+0.051 to +0.056)** is the Mode (b) *architecture* used as an ensemble: blend the calibrated diffusion with a few sweeps of propagation on the raw probability adjacency P. Replacing the per-node law by a constant leaves the gain intact or larger (`CTRL_lin0.9`: 0.901, 99.6% of the scalar oracle, Holm p=4.7e-15).
* **Context that matters for framing:** a batched 200-world Monte Carlo ranks at 0.906 in 1.7 ms at n=100 (rule: 0.844 in 2.2 ms); only at n=900 does the deterministic ensemble match it at lower cost (4.6 vs 12.5 ms).
* **Two corrections to [4] surfaced:** its rule under *uniform* priors (0.851) is beaten by the fixed constant α=0.600 (0.894); and the per-node headroom (+0.341 on grids in [4]) re-measures at +0.018 on grids here, concentrated in the dense/heterogeneous-prior corner.

## 1. What was built

* `prop_damping_lib.py` — vectorised library (NumPy/SciPy sparse): instance generators (9 synthetic families, 7 real networks), Monte-Carlo ground truth with block-diagonal batching (one `connected_components` call per batch) and exact world enumeration for verification, operators W, EW = diag(P_esc)W, Wᵀ (push), P (raw p_ij), EP, normalised damped diffusion with scalar/per-node α (lazy power form, converges on bipartite graphs), KNZ message passing, Realizations A/B/C under four scale conventions, the anchored power-law family AP(η), Mode (b) polishing, scalar and per-node oracles, metrics.
* `build_panels.py`, `experiments.py` (stages `grid` and `confirm`), `controls.py`, `mf_controls.py`, `sign_condition.py`, `prop2_stats.py`, `bench_runtime.py`, `explore_rawP.py`, `make_checkpoint_tables.py`, `tests/`.
* **Panels (688 instances):** dev 180 and test 180 (n=100; ER with P̄_esc from 0.15 to 0.98, BA, WS, square grid, 1:4 rectangular grid, SBM, core–periphery, tree, barbell × 5 probability ranges × {uniform, Dirichlet(1)} priors × 2 replicates); prior panel 150 (5 families × 3 ranges × 10 priors, same graph and p across priors); real 88 (karate, Les Misérables, Davis; 500-vertex snowball samples of Cora, CiteSeer, Amazon-Photo, MS-Academic-CS; 4 probability schemes incl. weighted cascade; 2 priors); scale 90 (n=900, all 9 families).
* **Ground truth:** two independent M=2000-world samples per instance: y*_A for evaluation, y*_B for fitting every oracle. Ranking by y*_B gives the *split-sample reference* `ref:ceiling` (mean test ρ = 0.958); because both samples are noisy, an exact predictor of y* would score about the square root of this value, so it is a reliability indicator rather than a hard ceiling.
* **Protocol:** every configuration was searched on dev only; the winners per family were frozen in `results/selected.json` (sha256 `b55fe19a10d3e5de…`, 2026-10-25T22:41:26+02:00) before any test, prior, real or scale result was examined, and evaluated once on those panels. Dev searches covered 760 standalone (realization × operator × scale × T), 40 anchored, 840 anchored-polishing and 864 realization-polishing configurations. Paired Wilcoxon signed-rank tests, 20,000-resample bootstrap CIs, Holm correction over the 21 frozen configurations.

## 2. Theory: what the coupled system can and cannot do

**2.1 Theorem 1 is correct** (both branches satisfy both equations; residual in `results/tests_foundations.txt`). One interpretive flag for the paper: on the active branch y_i = S_i − x_i the score *decreases* in the vertex's own prior for a fixed neighbour signal, whereas y*_i = x_i + (external mass) increases in it. The coupled system is therefore not an estimator of y*; it is a constraint that the operator scale must be engineered to satisfy.

**2.2 Proposition A (finite-time collapse of Realization A).** Let T ≥ 0 have row sums ≤ 1 (W and EW qualify) and x uniform. Since (Ty)_i ≤ max_j y_j, the maximum obeys M_{t+1} ≤ max(1/n, M_t − 1/n), so y ≡ x after at most ⌈nM_0⌉−1 steps. From the manuscript's warm start y⁽⁰⁾ = x + Tx (M_0 ≤ 2/n) the collapse happens at **t = 1**. Verified on 90 instance–operator pairs (max relative excess after one step 1.4e-15) and in the dynamics table (100% of vertices on the trivial branch at t=1 under EW with uniform priors).

**2.3 Proposition B (absorption and quadratic contraction of Realization B).** If y_i ≤ x_i then α_i = 0 at every later step, so the trivial branch is absorbing; with uniform x and row-substochastic T the maximal excess m obeys m' ≤ m²/(x+m). Observed: 77% of vertices absorbed by t=5 and 100% by t=20 (EW, uniform priors).

**2.4 Proposition C (the self-consistent solution is repelling).** If T ≥ 0 is irreducible and y > 0 satisfies the all-active fixed point y = Ty − x, then Ty > y, so ρ(T) > 1 (Collatz–Wielandt). The Jacobians are J_A = T and J_B = diag(x/y) + diag(1−x/y)T with J_B y = y + x − x²/y > y, hence ρ(J_A), ρ(J_B) > 1: the fixed point is unstable for both realizations. For T = λW, ρ(J_B) = −λ² + 3λ − 1 > 1 on (1,2); numerically ρ(J_B) = 1.16, 1.25, 1.09 at λ = 1.2, 1.5, 1.9, and a 1e−9 perturbation departs from the fixed point (relative departure 8.00e-01, 5.00e-01, 1.00e-01). Corollary: no row-substochastic operator admits a positive all-active solution; contractive operators collapse, expansive ones diverge or collapse.

**2.5 Proposition D (Realization C is a different law).** C sets odds = S/x, giving y' = (x²+S²)/(x+S), the Lehmer mean L₂(x,S). Its Jacobian diag(1 − 2x²/(x+S)²)T is an ∞-norm contraction for row-substochastic T with S > 0, so C converges to a unique fixed point (three random starts agree to 8e-17) — but that fixed point lies *below* the prior wherever S < x, and L₂ is non-monotone in S (decreasing for S < (√2−1)x), so an isolated vertex (S = 0) outranks a weakly connected one.

**2.6 Proposition E (sign obstruction).** For u_i = (1−α_i)x_i + α_iS_i, ∂u_i/∂α_i = S_i − x_i. Any law in which α_i increases with reachability raises well-connected vertices only on {S_i > x_i} and *lowers* them on {S_i < x_i}. Under EW in sparse regimes S_i < x_i for essentially every vertex (100% in a sparse ER instance; finite differences agree with the sign rule on 20/20 vertices). Across the 360 dev+test instances the share of vertices with S > x at the rule's fixed point predicts the relation between the per-node oracle's α* and the law (Spearman 0.61, p = 1.9e-38): in the bin with <10% such vertices (n=85) corr(α*, 1−x/y*) = -0.34 and the law with the true y* loses 1.21 in ρ (ranking inversion); in the densest bins it turns positive but stays ≤ 0.11.

| frac_S_gt_x_bin | pn_corr_prop | d_trueprop | d_trueap1 | Pbar | n |
|---|---|---|---|---|---|
| (-0.01, 0.1] | -0.338 | -1.210 | -0.014 | 0.309 | 85 |
| (0.1, 0.3] | -0.147 | -0.046 | -0.011 | 0.425 | 67 |
| (0.3, 0.5] | -0.027 | -0.097 | -0.038 | 0.711 | 77 |
| (0.5, 0.7] | 0.018 | -0.119 | -0.027 | 0.906 | 126 |
| (0.7, 1.0] | 0.106 | 0.001 | 0.000 | 0.985 | 5 |

**2.7 Proposition F (the 'Standard PPR' variant as written is degenerate under uniform priors).** With S_i = Σ_j w_ij y_j and W row-stochastic, Wx = x for uniform x, so y = c·x is an exact fixed point of A, B and C and every vertex ties. On dev, 85% of pull-W runs under uniform priors score ρ = 0 exactly (mean 0.012). The meaningful PPR variant is push (Wᵀ), which [4] used.

**2.8 Proposition 2 re-examined.** Part 1 is true but does not imply a ranking advantage (and α_i = 1 − 1/(ny_i) is not shift-invariant). Part 2's Jacobian is correct as a partial derivative, but the data locate the failure differently. On ground truth (prior panel): under smooth heterogeneous priors the ratio r_i = (y*_i − x_i)/x_i becomes a proxy for 1/x_i (rank correlation 0.79 → 0.98 from Dirichlet(10) to Dirichlet(0.3)) and loses its correlation with y* (0.30 → -0.16); under localised seed priors it stays ordinal with y* (0.84, 0.96) but saturates (61% of vertices with α > 0.99), which removes teleportation for the non-seed majority. Section 5 measures the effect on ranking.

## 3. Replication of [4] and the per-node headroom

Under Dirichlet priors the rule recovers 96.2% of the in-sample scalar oracle and beats the best hindsight constant (0.851 vs 0.684), reproducing [4]. Under **uniform** priors it recovers only 91.2%: the oracle's mean α is 0.56 versus the rule's 0.84, and the constant α=0.600 scores 0.894. The calibration is prior-dependent.

Rule clip(0.502+0.558 Pbar, 0.4, 1) vs scalar oracle on dev+test (n=100, 360 instances).

| group | n | rule | fixed085 | best_constant | best_constant_alpha | scalar_oracle_in | pct_of_oracle_in | mean_alpha_rule | mean_alpha_oracle |
|---|---|---|---|---|---|---|---|---|---|
| all | 360 | 0.851 | 0.780 | 0.780 | 0.875 | 0.909 | 93.617 | 0.842 | 0.706 |
| dir1 | 180 | 0.851 | 0.678 | 0.684 | 0.900 | 0.884 | 96.188 | 0.842 | 0.852 |
| uniform | 180 | 0.851 | 0.882 | 0.894 | 0.600 | 0.933 | 91.180 | 0.842 | 0.560 |

Per-node headroom (oracle fitted on y*_B, scored on y*_A) is +0.028 over the scalar oracle overall (apparent in-sample +0.051, so 46% of it is overfit), +0.018 on square grids and +0.069 on trees. It concentrates in the dense, heterogeneous-prior corner (R5 + Dirichlet: +0.133 averaged over families), where every scalar α fails; a gain of [4]'s size appears only on individual dense lattice instances (largest: +0.347).

Per-node oracle fitted on y*_B, evaluated on y*_A.

| family | oracle:scalar | oracle:pernode | ref:ceiling | pernode_gain | apparent_gain |
|---|---|---|---|---|---|
| ba | 0.949 | 0.973 | 0.988 | 0.024 | 0.032 |
| barbell | 0.926 | 0.949 | 0.974 | 0.023 | 0.048 |
| cp | 0.961 | 0.979 | 0.990 | 0.018 | 0.024 |
| er | 0.916 | 0.939 | 0.968 | 0.023 | 0.045 |
| grid | 0.933 | 0.950 | 0.960 | 0.018 | 0.040 |
| rect | 0.919 | 0.955 | 0.970 | 0.036 | 0.056 |
| sbm | 0.870 | 0.898 | 0.947 | 0.028 | 0.057 |
| tree | 0.920 | 0.989 | 0.994 | 0.069 | 0.072 |
| ws | 0.773 | 0.786 | 0.821 | 0.013 | 0.086 |
| ALL | 0.907 | 0.935 | 0.957 | 0.028 | 0.051 |

## 4. Held-out results for every frozen configuration (test panel)

Held-out test panel (n=100, 180 instances). Configurations frozen on dev (results/selected.json, sha256 in selected.sha256). d_rule: paired mean difference vs calibrated rule with 95% bootstrap CI; p: Wilcoxon signed-rank; Holm over the 21 frozen configurations. Gap-closed percentages are ratios of panel means relative to the calibrated rule.

| method | mean | dir1 | uniform | d_rule | ci | p_rule | wtl | d_scalar_oracle | pct_of_scalar_oracle | pct_scalar_gap_closed | pct_pernode_gap_closed | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:rule_EW | 0.844 | 0.842 | 0.846 | 0.000 | [+0.000, +0.000] |  | 0/180/0 | -0.060 | 93.346 | 0.000 | 0.000 |  |
| base:fixed085 | 0.781 | 0.681 | 0.881 | -0.064 | [-0.093, -0.035] | 7.3e-05 | 68/0/112 | -0.124 | 86.313 | -105.716 | -68.300 |  |
| base:rule_WT | 0.811 | 0.838 | 0.785 | -0.033 | [-0.066, +0.001] | 0.005 | 56/0/124 | -0.093 | 89.708 | -54.683 | -35.329 |  |
| base:local_pesc | 0.441 | 0.690 | 0.193 | -0.403 | [-0.503, -0.307] | 1.0e-18 | 41/1/138 | -0.463 | 48.769 | -669.976 | -432.852 |  |
| oracle:scalar | 0.904 | 0.877 | 0.932 | 0.060 | [+0.044, +0.078] | 5.6e-21 | 129/27/24 | 0.000 | 100.000 | 100.000 | 64.607 |  |
| oracle:scalar_in | 0.905 | 0.877 | 0.933 | 0.061 | [+0.045, +0.079] | 2.9e-25 | 144/24/12 | 0.001 | 100.069 | 101.034 | 65.275 |  |
| oracle:scalar_WT | 0.886 | 0.894 | 0.879 | 0.042 | [+0.017, +0.068] | 0.664 | 70/0/110 | -0.018 | 98.007 | 70.045 | 45.254 |  |
| oracle:pernode | 0.937 | 0.931 | 0.944 | 0.093 | [+0.071, +0.117] | 1.9e-18 | 136/1/43 | 0.033 | 103.645 | 154.781 | 100.000 |  |
| ref:ceiling | 0.958 | 0.964 | 0.952 | 0.114 | [+0.089, +0.141] | 9.7e-19 | 135/0/45 | 0.054 | 105.954 | 189.481 | 122.419 |  |
| ref:prior | 0.205 | 0.410 | 0.000 | -0.639 | [-0.688, -0.589] | 4.2e-31 | 1/0/179 | -0.699 | 22.670 | -1062.238 | -686.283 |  |
| ref:Pesc | 0.736 | 0.592 | 0.880 | -0.108 | [-0.160, -0.057] | 0.002 | 68/0/112 | -0.168 | 81.410 | -179.401 | -115.906 |  |
| ref:MP2 | 0.754 | 0.630 | 0.878 | -0.090 | [-0.145, -0.037] | 0.074 | 85/0/95 | -0.151 | 83.356 | -150.155 | -97.011 |  |
| ref:MPinf | 0.681 | 0.590 | 0.773 | -0.163 | [-0.227, -0.102] | 5.6e-04 | 81/0/99 | -0.223 | 75.334 | -270.726 | -174.908 |  |
| true:prop | 0.488 | 0.707 | 0.269 | -0.356 | [-0.452, -0.265] | 1.3e-17 | 43/0/137 | -0.416 | 53.974 | -591.757 | -382.317 |  |
| true:ap0.5 | 0.832 | 0.820 | 0.843 | -0.012 | [-0.018, -0.008] | 2.9e-15 | 28/52/100 | -0.073 | 91.975 | -20.607 | -13.314 |  |
| true:ap1.0 | 0.817 | 0.801 | 0.834 | -0.027 | [-0.035, -0.019] | 4.1e-20 | 18/49/113 | -0.087 | 90.365 | -44.813 | -28.953 |  |
| sel:A_best | 0.755 | 0.647 | 0.864 | -0.089 | [-0.129, -0.049] | 2.0e-05 | 60/0/120 | -0.149 | 83.519 | -147.707 | -95.429 | 2.8e-04 |
| sel:B_best | 0.762 | 0.717 | 0.807 | -0.082 | [-0.123, -0.041] | 2.7e-04 | 68/0/112 | -0.142 | 84.278 | -136.291 | -88.054 | 0.002 |
| sel:C_best | 0.766 | 0.717 | 0.815 | -0.078 | [-0.117, -0.039] | 6.4e-05 | 64/0/116 | -0.138 | 84.697 | -129.998 | -83.988 | 7.7e-04 |
| sel:C_dir1 | 0.634 | 0.872 | 0.395 | -0.211 | [-0.311, -0.116] | 0.034 | 75/0/105 | -0.271 | 70.070 | -349.839 | -226.021 | 0.169 |
| sel:B_unif | 0.623 | 0.369 | 0.876 | -0.222 | [-0.278, -0.165] | 2.2e-12 | 43/0/137 | -0.282 | 68.842 | -368.292 | -237.943 | 4.4e-11 |
| sel:AP_eta0.25 | 0.843 | 0.839 | 0.846 | -0.002 | [-0.005, +0.003] | 7.3e-04 | 55/48/77 | -0.062 | 93.155 | -2.875 | -1.857 | 0.006 |
| sel:AP_eta1 | 0.832 | 0.818 | 0.846 | -0.012 | [-0.019, -0.005] | 1.9e-07 | 54/43/83 | -0.073 | 91.983 | -20.491 | -13.239 | 3.4e-06 |
| sel:PB_rule_best | 0.844 | 0.843 | 0.846 | 0.000 | [+0.000, +0.000] | 0.766 | 29/119/32 | -0.060 | 93.372 | 0.381 | 0.246 | 1.000 |
| sel:PB_rule_nominal | 0.840 | 0.833 | 0.846 | -0.005 | [-0.007, -0.002] | 3.1e-07 | 14/113/53 | -0.065 | 92.842 | -7.581 | -4.898 | 5.0e-06 |
| sel:PB_rule_eta1 | 0.843 | 0.839 | 0.846 | -0.002 | [-0.003, -0.001] | 0.004 | 49/65/66 | -0.062 | 93.176 | -2.564 | -1.657 | 0.025 |
| sel:PB_mp2_best | 0.894 | 0.884 | 0.903 | 0.049 | [+0.032, +0.069] | 1.0e-04 | 103/0/77 | -0.011 | 98.806 | 82.058 | 53.015 | 0.001 |
| sel:PB_mp2_eta0 | 0.894 | 0.887 | 0.902 | 0.050 | [+0.033, +0.070] | 4.1e-05 | 103/0/77 | -0.010 | 98.902 | 83.490 | 53.941 | 5.3e-04 |
| sel:PB_mp2_eta1 | 0.887 | 0.875 | 0.899 | 0.042 | [+0.024, +0.062] | 0.009 | 99/0/81 | -0.018 | 98.040 | 70.540 | 45.574 | 0.055 |
| sel:PBX_rule_best | 0.895 | 0.894 | 0.897 | 0.051 | [+0.033, +0.070] | 1.0e-04 | 99/3/78 | -0.009 | 98.969 | 84.503 | 54.595 | 0.001 |
| sel:PBX_rule_unif | 0.851 | 0.783 | 0.919 | 0.007 | [-0.014, +0.028] | 0.224 | 63/24/93 | -0.053 | 94.086 | 11.112 | 7.179 | 0.672 |
| sel:PBX_rule_dir1 | 0.775 | 0.899 | 0.652 | -0.069 | [-0.120, -0.020] | 0.148 | 96/0/84 | -0.129 | 85.745 | -114.247 | -73.812 | 0.592 |
| sel:PBX_rule_nominalB | 0.728 | 0.602 | 0.854 | -0.116 | [-0.147, -0.087] | 1.5e-09 | 50/21/109 | -0.176 | 80.519 | -192.787 | -124.554 | 2.9e-08 |
| sel:CTRL_Cmean | 0.898 | 0.895 | 0.900 | 0.053 | [+0.036, +0.073] | 3.4e-07 | 112/3/65 | -0.007 | 99.257 | 88.832 | 57.392 | 5.0e-06 |
| sel:CTRL_Cperm | 0.871 | 0.882 | 0.861 | 0.027 | [+0.006, +0.050] | 0.670 | 77/1/102 | -0.033 | 96.348 | 45.108 | 29.143 | 1.000 |
| sel:CTRL_lin0.5 | 0.896 | 0.891 | 0.901 | 0.052 | [+0.034, +0.071] | 2.8e-07 | 112/3/65 | -0.008 | 99.063 | 85.923 | 55.513 | 4.8e-06 |
| sel:CTRL_lin0.9 | 0.901 | 0.903 | 0.898 | 0.056 | [+0.043, +0.071] | 2.2e-16 | 135/0/45 | -0.004 | 99.581 | 93.699 | 60.536 | 4.7e-15 |

Configuration keys (all frozen on dev): `A_best`/`B_best`/`C_best` = best standalone realization overall (A: raw P, excess scale γ=16, T=2; B and C: EP, γ=16, T=2); `C_dir1`, `B_unif` = best standalone pick within one prior type; `AP_eta*` = anchored law, self-consistent fixed point; `PB_*` = anchored law polishing a base (rule or 2-step MP) with K sweeps and β blend (`PB_rule_nominal` = anchored law, K=1, β=0.5); `PBX_*` = Realizations A/B/C as polishers of the rule; `PBX_rule_nominalB` = Mode (b) as written in the manuscript (Realization B on the excess scale, one sweep, 50% blend); `CTRL_*` = the same pipeline as `PBX_rule_best` with the per-node law replaced by its per-step mean (`Cmean`), a random reassignment (`Cperm`) or a constant damping 0.5/0.9 (`lin*`).

**Mode (a), standalone.** Every realization is significantly worse than the rule on test (-0.089, -0.082, -0.078 for A/B/C). The prior-specific picks do well only on their own prior (`C_dir1` 0.872 on Dirichlet, 0.395 on uniform).

**Anchored power law (the principle in its best form).** η = 0 *is* the scalar rule; the η-curve peaks there on Dirichlet priors and is flat on uniform priors, on dev and test alike.

Anchored proportional law AP(eta), self-consistent fixed point; eta=0 is the scalar rule.

| eta | dev_dir1 | dev_uniform | test_dir1 | test_uniform |
|---|---|---|---|---|
| -1.0 | 0.788 | 0.852 | 0.785 | 0.842 |
| -0.5 | 0.834 | 0.855 | 0.830 | 0.846 |
| -0.25 | 0.849 | 0.855 | 0.845 | 0.846 |
| 0.0 | 0.856 | 0.856 | 0.846 | 0.846 |
| 0.25 | 0.853 | 0.856 | 0.839 | 0.846 |
| 0.5 | 0.848 | 0.856 | 0.829 | 0.846 |
| 1.0 | 0.840 | 0.856 | 0.818 | 0.846 |
| 2.0 | 0.829 | 0.855 | 0.804 | 0.845 |

**Attribution: per-node law vs its own mean-field twin.** Same pipeline, same K, γ, blend; only α_i is replaced by its per-step mean (no per-node variation) or permuted.

Each frozen realization config (law) vs the identical pipeline with alpha replaced by its per-step mean (mean) or randomly reassigned (perm).

| config / prior | dev_law | dev_mean | dev_perm | dev_law_minus_mean | test_law | test_mean | test_perm | test_law_minus_mean |
|---|---|---|---|---|---|---|---|---|
| A_best / dir1 | 0.643 | 0.617 | 0.579 | 0.027 | 0.647 | 0.618 | 0.604 | 0.029 |
| A_best / uniform | 0.852 | 0.871 | 0.713 | -0.018 | 0.864 | 0.880 | 0.731 | -0.016 |
| B_best / dir1 | 0.723 | 0.715 | 0.639 | 0.008 | 0.717 | 0.710 | 0.650 | 0.007 |
| B_best / uniform | 0.793 | 0.792 | 0.690 | 0.000 | 0.807 | 0.817 | 0.711 | -0.010 |
| B_unif / dir1 | 0.333 | 0.593 | 0.403 | -0.260 | 0.369 | 0.610 | 0.408 | -0.241 |
| B_unif / uniform | 0.883 | 0.850 | 0.716 | 0.033 | 0.876 | 0.855 | 0.769 | 0.021 |
| C_best / dir1 | 0.723 | 0.715 | 0.693 | 0.008 | 0.717 | 0.710 | 0.701 | 0.008 |
| C_best / uniform | 0.790 | 0.792 | 0.768 | -0.002 | 0.815 | 0.817 | 0.793 | -0.003 |
| C_dir1 / dir1 | 0.870 | 0.876 | 0.834 | -0.007 | 0.872 | 0.879 | 0.836 | -0.006 |
| C_dir1 / uniform | 0.388 | 0.885 | 0.671 | -0.497 | 0.395 | 0.898 | 0.689 | -0.503 |
| PBX_rule_best / dir1 | 0.902 | 0.904 | 0.886 | -0.002 | 0.894 | 0.895 | 0.879 | -0.002 |
| PBX_rule_best / uniform | 0.896 | 0.900 | 0.856 | -0.004 | 0.897 | 0.900 | 0.859 | -0.004 |
| PBX_rule_dir1 / dir1 | 0.905 | 0.904 | 0.891 | 0.001 | 0.899 | 0.900 | 0.886 | -0.001 |
| PBX_rule_dir1 / uniform | 0.657 | 0.899 | 0.836 | -0.242 | 0.652 | 0.897 | 0.833 | -0.244 |
| PBX_rule_nominalB / dir1 | 0.596 | 0.601 | 0.596 | -0.005 | 0.602 | 0.607 | 0.603 | -0.005 |
| PBX_rule_nominalB / uniform | 0.864 | 0.865 | 0.853 | -0.001 | 0.854 | 0.855 | 0.845 | -0.001 |
| PBX_rule_unif / dir1 | 0.785 | 0.790 | 0.693 | -0.005 | 0.783 | 0.788 | 0.699 | -0.005 |
| PBX_rule_unif / uniform | 0.920 | 0.919 | 0.854 | 0.001 | 0.919 | 0.918 | 0.852 | 0.000 |

On real networks (88 instances) the C law inside the best polisher retains a small positive effect over its twin (+0.0066, CI [+0.0031, +0.0108], p=0.011, Holm 0.032), while every standalone law loses to its twin; the best pipeline overall still uses no per-node law.

Per-node law vs per-step-mean twin (real panel); Holm over the configurations listed.

| config | law | mean | perm | d_law_minus_mean | ci | p | wtl | p_holm |
|---|---|---|---|---|---|---|---|---|
| A_best | 0.809 | 0.828 | 0.688 | -0.019 | [-0.0308, -0.0089] | 0.001 | 23/4/61 | 0.005 |
| B_best | 0.820 | 0.825 | 0.698 | -0.005 | [-0.0102, -0.0000] | 0.007 | 28/0/60 | 0.029 |
| B_unif | 0.724 | 0.772 | 0.272 | -0.047 | [-0.0695, -0.0251] | 2.6e-05 | 29/0/59 | 2.1e-04 |
| C_best | 0.823 | 0.825 | 0.792 | -0.002 | [-0.0035, +0.0002] | 0.183 | 33/2/53 | 0.365 |
| C_dir1 | 0.739 | 0.844 | 0.777 | -0.105 | [-0.1680, -0.0513] | 4.2e-04 | 24/4/60 | 0.003 |
| PBX_rule_best | 0.889 | 0.882 | 0.864 | 0.007 | [+0.0031, +0.0108] | 0.011 | 49/2/37 | 0.032 |
| PBX_rule_dir1 | 0.849 | 0.889 | 0.866 | -0.039 | [-0.0680, -0.0145] | 0.813 | 43/1/44 | 0.813 |
| PBX_rule_nominalB | 0.696 | 0.701 | 0.688 | -0.005 | [-0.0068, -0.0028] | 3.3e-10 | 16/3/69 | 2.9e-09 |
| PBX_rule_unif | 0.862 | 0.855 | 0.773 | 0.008 | [+0.0030, +0.0124] | 4.8e-04 | 61/2/25 | 0.003 |

## 5. Prior uniformity (Proposition 2, prior panel, 150 instances)

The anchored law's difference to the rule is +0.0003 under uniform priors and falls to -0.098 under single-seed priors (Spearman with log₁₀κ = -0.31, p = 1.2e-04). The direction of Proposition 2 is confirmed, but there is no regime of advantage: 'best under uniform priors' means 'harmless under uniform priors'. Denominator regularisation shrinks the loss (τ=0.5) and removes it at τ=1 (-0.004 to +0.005), which is where the odds denominator stops depending on x. The ensemble controls gain under every prior type (+0.033 to +0.052).

Mean paired difference vs calibrated rule, per prior type.

|  | d_AP_eta1 | d_AP_eta1_tau0.5 | d_AP_eta1_tau1 | d_PBnom | d_PBnom_tau1 | d_PBX_best | d_CTRL_lin0.9 |
|---|---|---|---|---|---|---|---|
| uniform | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.039 | 0.040 |
| dir10 | -0.007 | -0.001 | 0.002 | -0.003 | 0.000 | 0.032 | 0.043 |
| logn0.5 | -0.017 | -0.005 | 0.003 | -0.007 | 0.000 | 0.036 | 0.044 |
| dir3 | -0.013 | -0.003 | 0.003 | -0.006 | 0.001 | 0.038 | 0.047 |
| logn1 | -0.031 | -0.004 | -0.004 | -0.011 | 0.001 | 0.049 | 0.051 |
| dir1 | -0.031 | -0.006 | 0.005 | -0.011 | 0.001 | 0.038 | 0.046 |
| logn2 | -0.057 | -0.017 | -0.004 | -0.013 | -0.000 | 0.032 | 0.049 |
| dir0.3 | -0.056 | -0.017 | 0.003 | -0.021 | 0.002 | 0.039 | 0.052 |
| seed5_0.1 | -0.079 | -0.040 | 0.005 | -0.000 | 0.000 | 0.007 | 0.033 |
| seed1_0.1 | -0.098 | -0.072 | -0.003 | 0.001 | 0.000 | 0.029 | 0.042 |

Prior-uniformity panel (5 families x 3 ranges x 10 priors, n=144-150).

| prior | rule | scalar_oracle | pernode_oracle | ceiling | true_prop | true_ap1 | AP_eta1 | AP_eta1_tau0.5 | AP_eta1_tau1 | PBnom | PBnom_tau0.5 | PBnom_tau1 | PBX_best | CTRL_lin0.9 | CTRL_Cmean | B_unif | C_dir1 | log10_kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 0.869 | 0.951 | 0.975 | 0.978 | 0.608 | 0.857 | 0.869 | 0.869 | 0.869 | 0.869 | 0.869 | 0.869 | 0.907 | 0.909 | 0.907 | 0.904 | 0.750 | 0.000 |
| dir10 | 0.864 | 0.882 | 0.957 | 0.981 | 0.737 | 0.842 | 0.857 | 0.863 | 0.866 | 0.861 | 0.863 | 0.864 | 0.896 | 0.907 | 0.893 | 0.728 | 0.844 | 0.770 |
| logn0.5 | 0.860 | 0.891 | 0.956 | 0.983 | 0.759 | 0.821 | 0.843 | 0.855 | 0.863 | 0.853 | 0.858 | 0.861 | 0.896 | 0.904 | 0.896 | 0.511 | 0.865 | 1.191 |
| dir3 | 0.854 | 0.885 | 0.961 | 0.982 | 0.765 | 0.824 | 0.840 | 0.851 | 0.857 | 0.848 | 0.852 | 0.855 | 0.892 | 0.901 | 0.892 | 0.504 | 0.871 | 1.476 |
| logn1 | 0.866 | 0.898 | 0.961 | 0.981 | 0.726 | 0.813 | 0.835 | 0.862 | 0.862 | 0.855 | 0.862 | 0.867 | 0.915 | 0.917 | 0.912 | 0.489 | 0.878 | 2.334 |
| dir1 | 0.873 | 0.903 | 0.964 | 0.983 | 0.746 | 0.817 | 0.842 | 0.867 | 0.878 | 0.862 | 0.869 | 0.874 | 0.911 | 0.919 | 0.905 | 0.423 | 0.865 | 3.353 |
| logn2 | 0.844 | 0.892 | 0.952 | 0.982 | 0.678 | 0.758 | 0.787 | 0.827 | 0.840 | 0.831 | 0.838 | 0.844 | 0.876 | 0.893 | 0.879 | 0.464 | 0.840 | 4.629 |
| dir0.3 | 0.852 | 0.898 | 0.958 | 0.981 | 0.677 | 0.773 | 0.795 | 0.835 | 0.855 | 0.830 | 0.843 | 0.853 | 0.891 | 0.904 | 0.887 | 0.254 | 0.845 | 8.526 |
| seed5_0.1 | 0.867 | 0.904 | 0.946 | 0.978 | 0.614 | 0.764 | 0.788 | 0.828 | 0.873 | 0.867 | 0.867 | 0.868 | 0.874 | 0.901 | 0.872 | 0.730 | 0.807 | 2.429 |
| seed1_0.1 | 0.818 | 0.899 | 0.936 | 0.952 | 0.453 | 0.653 | 0.720 | 0.747 | 0.816 | 0.819 | 0.818 | 0.818 | 0.847 | 0.860 | 0.849 | 0.651 | 0.705 | 3.127 |

## 6. Transfer: real networks and n = 900

Frozen configurations on the real panel.

| method | n | mean | d_rule | ci | p | wtl | genre=amazon_electronics_photo | genre=citeseer | genre=classic | genre=cora | genre=ms_academic_cs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base:rule_EW | 88 | 0.840 | 0.000 | [+0.000, +0.000] |  | 0/88/0 | 0.781 | 0.846 | 0.886 | 0.813 | 0.852 |
| oracle:scalar | 88 | 0.897 | 0.057 | [+0.041, +0.074] | 1.7e-13 | 71/13/4 | 0.828 | 0.909 | 0.935 | 0.881 | 0.912 |
| oracle:pernode | 88 | 0.950 | 0.109 | [+0.091, +0.130] | 3.9e-16 | 87/0/1 | 0.848 | 0.984 | 0.965 | 0.982 | 0.960 |
| ref:ceiling | 88 | 0.985 | 0.145 | [+0.121, +0.171] | 3.9e-16 | 87/0/1 | 0.957 | 0.997 | 0.988 | 0.997 | 0.987 |
| ref:MP2 | 88 | 0.871 | 0.031 | [-0.017, +0.075] | 0.005 | 64/0/24 | 0.835 | 0.831 | 0.899 | 0.867 | 0.908 |
| ref:Pesc | 88 | 0.814 | -0.026 | [-0.073, +0.020] | 0.881 | 48/0/40 | 0.833 | 0.735 | 0.872 | 0.749 | 0.855 |
| true:prop | 88 | 0.547 | -0.293 | [-0.393, -0.202] | 7.2e-11 | 22/0/66 | 0.589 | 0.415 | 0.619 | 0.422 | 0.654 |
| sel:AP_eta1 | 88 | 0.825 | -0.015 | [-0.023, -0.009] | 5.4e-05 | 26/14/48 | 0.771 | 0.831 | 0.869 | 0.796 | 0.836 |
| sel:PB_rule_nominal | 88 | 0.834 | -0.007 | [-0.010, -0.004] | 3.6e-07 | 5/51/32 | 0.776 | 0.840 | 0.880 | 0.804 | 0.844 |
| sel:PB_rule_best | 88 | 0.840 | 0.000 | [-0.000, +0.001] | 0.278 | 19/54/15 | 0.781 | 0.846 | 0.887 | 0.813 | 0.852 |
| sel:PB_mp2_best | 88 | 0.899 | 0.059 | [+0.045, +0.075] | 2.4e-12 | 76/0/12 | 0.829 | 0.907 | 0.928 | 0.900 | 0.918 |
| sel:PB_mp2_eta0 | 88 | 0.900 | 0.060 | [+0.045, +0.077] | 4.8e-12 | 74/0/14 | 0.830 | 0.908 | 0.929 | 0.902 | 0.919 |
| sel:PBX_rule_best | 88 | 0.889 | 0.048 | [+0.033, +0.065] | 7.3e-09 | 67/0/21 | 0.812 | 0.885 | 0.947 | 0.895 | 0.876 |
| sel:CTRL_Cmean | 88 | 0.882 | 0.042 | [+0.026, +0.058] | 2.2e-06 | 63/0/25 | 0.810 | 0.873 | 0.942 | 0.883 | 0.871 |
| sel:CTRL_lin0.9 | 88 | 0.901 | 0.060 | [+0.047, +0.075] | 4.7e-14 | 79/0/9 | 0.827 | 0.913 | 0.936 | 0.916 | 0.892 |
| sel:A_best | 88 | 0.809 | -0.031 | [-0.070, +0.005] | 0.597 | 44/0/44 | 0.757 | 0.871 | 0.836 | 0.807 | 0.760 |
| sel:B_best | 88 | 0.820 | -0.020 | [-0.057, +0.014] | 0.966 | 47/1/40 | 0.774 | 0.858 | 0.867 | 0.821 | 0.757 |
| sel:C_best | 88 | 0.823 | -0.017 | [-0.051, +0.016] | 0.963 | 47/0/41 | 0.770 | 0.864 | 0.863 | 0.827 | 0.773 |
| sel:C_dir1 | 88 | 0.739 | -0.101 | [-0.163, -0.046] | 0.004 | 36/0/52 | 0.673 | 0.750 | 0.852 | 0.583 | 0.778 |
| sel:B_unif | 88 | 0.724 | -0.116 | [-0.166, -0.069] | 2.6e-05 | 26/0/62 | 0.760 | 0.709 | 0.786 | 0.644 | 0.692 |

Frozen configurations on the scale panel.

| method | n | mean | d_rule | ci | p | wtl | prior=dir1 | prior=uniform |
|---|---|---|---|---|---|---|---|---|
| base:rule_EW | 90 | 0.832 | 0.000 | [+0.000, +0.000] |  | 0/90/0 | 0.820 | 0.843 |
| oracle:scalar | 90 | 0.894 | 0.062 | [+0.038, +0.090] | 8.4e-13 | 71/14/5 | 0.863 | 0.926 |
| oracle:pernode | 40 | 0.927 | 0.099 | [+0.062, +0.142] | 6.4e-06 | 30/0/10 | 0.914 | 0.940 |
| ref:ceiling | 90 | 0.951 | 0.119 | [+0.087, +0.152] | 1.2e-10 | 70/0/20 | 0.958 | 0.943 |
| ref:MP2 | 90 | 0.730 | -0.102 | [-0.181, -0.025] | 0.103 | 42/0/48 | 0.606 | 0.854 |
| ref:Pesc | 90 | 0.710 | -0.122 | [-0.197, -0.049] | 0.007 | 30/0/60 | 0.568 | 0.851 |
| true:prop | 90 | 0.529 | -0.303 | [-0.432, -0.186] | 2.4e-07 | 23/0/67 | 0.762 | 0.296 |
| sel:AP_eta1 | 90 | 0.827 | -0.005 | [-0.014, +0.005] | 0.001 | 35/6/49 | 0.804 | 0.849 |
| sel:PB_rule_nominal | 90 | 0.828 | -0.004 | [-0.007, -0.002] | 3.1e-06 | 9/42/39 | 0.812 | 0.844 |
| sel:PB_rule_best | 90 | 0.832 | -0.000 | [-0.001, +0.000] | 0.507 | 19/47/24 | 0.820 | 0.843 |
| sel:PB_mp2_best | 90 | 0.877 | 0.045 | [+0.021, +0.072] | 0.008 | 51/0/39 | 0.858 | 0.896 |
| sel:PB_mp2_eta0 | 90 | 0.877 | 0.045 | [+0.020, +0.072] | 0.009 | 51/0/39 | 0.858 | 0.896 |
| sel:PBX_rule_best | 90 | 0.873 | 0.041 | [+0.019, +0.063] | 0.005 | 51/1/38 | 0.865 | 0.880 |
| sel:CTRL_Cmean | 90 | 0.873 | 0.041 | [+0.019, +0.064] | 2.2e-04 | 59/1/30 | 0.862 | 0.884 |
| sel:CTRL_lin0.9 | 90 | 0.886 | 0.054 | [+0.038, +0.073] | 1.4e-07 | 70/0/20 | 0.882 | 0.891 |
| sel:A_best | 90 | 0.728 | -0.104 | [-0.147, -0.061] | 7.0e-06 | 25/0/65 | 0.646 | 0.811 |
| sel:B_best | 90 | 0.717 | -0.115 | [-0.167, -0.064] | 9.9e-06 | 25/0/65 | 0.674 | 0.760 |
| sel:C_best | 90 | 0.722 | -0.110 | [-0.160, -0.062] | 8.2e-06 | 22/0/68 | 0.682 | 0.762 |
| sel:C_dir1 | 90 | 0.615 | -0.217 | [-0.351, -0.097] | 0.063 | 34/0/56 | 0.835 | 0.395 |
| sel:B_unif | 90 | 0.655 | -0.177 | [-0.234, -0.120] | 5.5e-09 | 14/0/76 | 0.462 | 0.848 |

Per-node law vs per-step-mean twin (scale panel); Holm over the configurations listed.

| config | law | mean | perm | d_law_minus_mean | ci | p | wtl | p_holm |
|---|---|---|---|---|---|---|---|---|
| A_best | 0.728 | 0.733 | 0.598 | -0.004 | [-0.0197, +0.0107] | 0.564 | 37/0/53 | 1.000 |
| B_best | 0.717 | 0.720 | 0.608 | -0.003 | [-0.0098, +0.0033] | 0.798 | 45/0/45 | 1.000 |
| B_unif | 0.655 | 0.691 | 0.470 | -0.036 | [-0.0620, -0.0102] | 0.002 | 34/0/56 | 0.014 |
| C_best | 0.722 | 0.720 | 0.700 | 0.001 | [-0.0019, +0.0044] | 0.722 | 40/3/47 | 1.000 |
| C_dir1 | 0.615 | 0.850 | 0.722 | -0.235 | [-0.3636, -0.1207] | 0.004 | 27/0/63 | 0.020 |
| PBX_rule_best | 0.873 | 0.873 | 0.847 | -0.000 | [-0.0043, +0.0043] | 0.002 | 21/7/62 | 0.011 |
| PBX_rule_dir1 | 0.772 | 0.879 | 0.841 | -0.107 | [-0.1695, -0.0518] | 8.6e-04 | 27/1/62 | 0.007 |
| PBX_rule_nominalB | 0.709 | 0.712 | 0.707 | -0.003 | [-0.0042, -0.0019] | 3.8e-06 | 25/10/55 | 3.4e-05 |
| PBX_rule_unif | 0.824 | 0.823 | 0.752 | 0.001 | [-0.0045, +0.0063] | 0.315 | 43/8/39 | 1.000 |

Kendall tau-b, top-10% overlap and NDCG@10% on the real panel.

| method | rho | tau | topk | ndcg |
|---|---|---|---|---|
| base:rule_EW | 0.840 | 0.702 | 0.464 | 0.932 |
| base:fixed085 | 0.780 | 0.645 | 0.434 | 0.910 |
| base:rule_WT | 0.817 | 0.686 | 0.574 | 0.893 |
| base:local_pesc | 0.414 | 0.327 | 0.273 | 0.698 |
| oracle:scalar | 0.897 | 0.776 | 0.570 | 0.959 |
| oracle:scalar_WT | 0.878 | 0.752 | 0.672 | 0.941 |
| oracle:pernode | 0.950 | 0.869 | 0.668 | 0.976 |
| ref:ceiling | 0.985 | 0.943 | 0.926 | 0.998 |
| ref:prior | 0.125 | 0.092 | 0.314 | 0.737 |
| ref:Pesc | 0.814 | 0.687 | 0.593 | 0.894 |
| ref:MP2 | 0.871 | 0.769 | 0.678 | 0.929 |
| ref:MPinf | 0.838 | 0.738 | 0.629 | 0.912 |
| true:prop | 0.547 | 0.434 | 0.344 | 0.801 |
| true:ap0.5 | 0.809 | 0.667 | 0.444 | 0.913 |
| true:ap1.0 | 0.765 | 0.620 | 0.403 | 0.885 |
| sel:A_best | 0.809 | 0.665 | 0.578 | 0.931 |
| sel:B_best | 0.820 | 0.676 | 0.588 | 0.927 |
| sel:C_best | 0.823 | 0.677 | 0.586 | 0.928 |
| sel:C_dir1 | 0.739 | 0.598 | 0.565 | 0.917 |
| sel:B_unif | 0.724 | 0.591 | 0.516 | 0.858 |
| sel:AP_eta0.25 | 0.837 | 0.698 | 0.466 | 0.930 |
| sel:AP_eta1 | 0.825 | 0.682 | 0.451 | 0.918 |
| sel:PB_rule_best | 0.840 | 0.702 | 0.466 | 0.933 |
| sel:PB_rule_nominal | 0.834 | 0.694 | 0.458 | 0.925 |
| sel:PB_rule_eta1 | 0.838 | 0.700 | 0.464 | 0.931 |
| sel:PB_mp2_best | 0.899 | 0.773 | 0.518 | 0.949 |
| sel:PB_mp2_eta0 | 0.900 | 0.773 | 0.514 | 0.948 |
| sel:PB_mp2_eta1 | 0.893 | 0.763 | 0.522 | 0.949 |
| sel:PBX_rule_best | 0.889 | 0.756 | 0.634 | 0.971 |
| sel:PBX_rule_unif | 0.862 | 0.726 | 0.634 | 0.942 |
| sel:PBX_rule_dir1 | 0.849 | 0.699 | 0.590 | 0.970 |
| sel:PBX_rule_nominalB | 0.696 | 0.568 | 0.415 | 0.889 |
| sel:CTRL_Cmean | 0.882 | 0.744 | 0.626 | 0.963 |
| sel:CTRL_Cperm | 0.869 | 0.718 | 0.597 | 0.951 |
| sel:CTRL_lin0.5 | 0.884 | 0.747 | 0.635 | 0.965 |
| sel:CTRL_lin0.9 | 0.901 | 0.774 | 0.581 | 0.962 |

Kendall tau-b, top-10% overlap and NDCG@10% on the scale panel.

| method | rho | tau | topk | ndcg |
|---|---|---|---|---|
| base:rule_EW | 0.832 | 0.723 | 0.601 | 0.959 |
| base:fixed085 | 0.761 | 0.644 | 0.551 | 0.934 |
| base:rule_WT | 0.759 | 0.613 | 0.559 | 0.866 |
| base:local_pesc | 0.406 | 0.316 | 0.372 | 0.744 |
| oracle:scalar | 0.894 | 0.797 | 0.676 | 0.975 |
| oracle:scalar_WT | 0.832 | 0.704 | 0.634 | 0.921 |
| oracle:pernode | 0.927 | 0.832 | 0.724 | 0.980 |
| ref:ceiling | 0.951 | 0.877 | 0.880 | 0.995 |
| ref:prior | 0.190 | 0.159 | 0.320 | 0.746 |
| ref:Pesc | 0.710 | 0.583 | 0.543 | 0.876 |
| ref:MP2 | 0.730 | 0.613 | 0.548 | 0.876 |
| ref:MPinf | 0.647 | 0.550 | 0.497 | 0.836 |
| true:prop | 0.529 | 0.430 | 0.431 | 0.797 |
| true:ap0.5 | 0.817 | 0.703 | 0.570 | 0.943 |
| true:ap1.0 | 0.787 | 0.665 | 0.544 | 0.927 |
| sel:A_best | 0.728 | 0.593 | 0.592 | 0.930 |
| sel:B_best | 0.717 | 0.589 | 0.597 | 0.921 |
| sel:C_best | 0.722 | 0.591 | 0.593 | 0.927 |
| sel:C_dir1 | 0.615 | 0.502 | 0.573 | 0.884 |
| sel:B_unif | 0.655 | 0.528 | 0.482 | 0.810 |
| sel:AP_eta0.25 | 0.836 | 0.725 | 0.588 | 0.955 |
| sel:AP_eta1 | 0.827 | 0.710 | 0.554 | 0.944 |
| sel:PB_rule_best | 0.832 | 0.722 | 0.601 | 0.960 |
| sel:PB_rule_nominal | 0.828 | 0.717 | 0.598 | 0.952 |
| sel:PB_rule_eta1 | 0.835 | 0.725 | 0.601 | 0.957 |
| sel:PB_mp2_best | 0.877 | 0.740 | 0.568 | 0.951 |
| sel:PB_mp2_eta0 | 0.877 | 0.739 | 0.570 | 0.950 |
| sel:PB_mp2_eta1 | 0.868 | 0.729 | 0.560 | 0.950 |
| sel:PBX_rule_best | 0.873 | 0.754 | 0.691 | 0.976 |
| sel:PBX_rule_unif | 0.824 | 0.707 | 0.667 | 0.963 |
| sel:PBX_rule_dir1 | 0.772 | 0.653 | 0.597 | 0.943 |
| sel:PBX_rule_nominalB | 0.709 | 0.605 | 0.529 | 0.910 |
| sel:CTRL_Cmean | 0.873 | 0.757 | 0.698 | 0.976 |
| sel:CTRL_Cperm | 0.847 | 0.710 | 0.630 | 0.962 |
| sel:CTRL_lin0.5 | 0.868 | 0.752 | 0.693 | 0.975 |
| sel:CTRL_lin0.9 | 0.886 | 0.772 | 0.677 | 0.979 |

## 7. Secondary metrics (test)

Kendall tau-b, top-10% overlap and NDCG@10% (relevance = y*_A), test panel.

| method | rho | tau | topk | ndcg |
|---|---|---|---|---|
| base:rule_EW | 0.844 | 0.737 | 0.583 | 0.963 |
| base:fixed085 | 0.781 | 0.659 | 0.543 | 0.945 |
| base:rule_WT | 0.811 | 0.666 | 0.579 | 0.919 |
| base:local_pesc | 0.441 | 0.350 | 0.349 | 0.736 |
| oracle:scalar | 0.904 | 0.808 | 0.658 | 0.981 |
| oracle:scalar_WT | 0.886 | 0.757 | 0.661 | 0.976 |
| oracle:pernode | 0.937 | 0.852 | 0.708 | 0.987 |
| ref:ceiling | 0.958 | 0.892 | 0.869 | 0.995 |
| ref:prior | 0.205 | 0.170 | 0.358 | 0.763 |
| ref:Pesc | 0.736 | 0.608 | 0.506 | 0.866 |
| ref:MP2 | 0.754 | 0.636 | 0.514 | 0.863 |
| ref:MPinf | 0.681 | 0.578 | 0.478 | 0.827 |
| true:prop | 0.488 | 0.392 | 0.394 | 0.779 |
| true:ap0.5 | 0.832 | 0.718 | 0.566 | 0.953 |
| true:ap1.0 | 0.817 | 0.695 | 0.546 | 0.943 |
| sel:A_best | 0.755 | 0.627 | 0.546 | 0.887 |
| sel:B_best | 0.762 | 0.638 | 0.546 | 0.878 |
| sel:C_best | 0.766 | 0.639 | 0.552 | 0.889 |
| sel:C_dir1 | 0.634 | 0.523 | 0.565 | 0.872 |
| sel:B_unif | 0.623 | 0.508 | 0.461 | 0.827 |
| sel:AP_eta0.25 | 0.843 | 0.734 | 0.577 | 0.962 |
| sel:AP_eta1 | 0.832 | 0.717 | 0.559 | 0.953 |
| sel:PB_rule_best | 0.844 | 0.737 | 0.583 | 0.963 |
| sel:PB_rule_nominal | 0.840 | 0.731 | 0.577 | 0.957 |
| sel:PB_rule_eta1 | 0.843 | 0.735 | 0.581 | 0.962 |
| sel:PB_mp2_best | 0.894 | 0.764 | 0.569 | 0.949 |
| sel:PB_mp2_eta0 | 0.894 | 0.765 | 0.567 | 0.948 |
| sel:PB_mp2_eta1 | 0.887 | 0.754 | 0.571 | 0.948 |
| sel:PBX_rule_best | 0.895 | 0.779 | 0.688 | 0.980 |
| sel:PBX_rule_unif | 0.851 | 0.735 | 0.643 | 0.967 |
| sel:PBX_rule_dir1 | 0.775 | 0.664 | 0.578 | 0.935 |
| sel:PBX_rule_nominalB | 0.728 | 0.624 | 0.531 | 0.925 |
| sel:CTRL_Cmean | 0.898 | 0.785 | 0.700 | 0.981 |
| sel:CTRL_Cperm | 0.871 | 0.735 | 0.647 | 0.969 |
| sel:CTRL_lin0.5 | 0.896 | 0.784 | 0.697 | 0.980 |
| sel:CTRL_lin0.9 | 0.901 | 0.793 | 0.656 | 0.978 |

## 8. Accuracy against cost, and Monte Carlo

With block-diagonal batching (one connected-components call per batch of worlds), a 200-world Monte-Carlo estimate costs 1.7 ms at n=100 — less than the calibrated rule including operator construction (2.2 ms) — and ranks better (0.906 vs 0.844). At n=900 the deterministic ensemble matches it (0.886 vs 0.887) at 2.7× lower cost. Any paper in this line has to position diffusion heuristics against a well-implemented sampler, not against the 13×-slower sampler assumed in [4].

Mean Spearman rho on the whole panel against median single-process, end-to-end wall clock (ms) over one uniform-prior instance per family and probability range.

| method | rho_scale | rho_test | ms_scale | ms_test |
|---|---|---|---|---|
| KNZ MP, 2 it. | 0.730 | 0.754 | 1.346 | 0.805 |
| Monte Carlo, 50 worlds | 0.785 | 0.802 | 3.328 | 0.588 |
| AP(eta=1), fixed point | 0.827 | 0.832 | 17.350 | 7.550 |
| rule + anchored polish (K=1) | 0.828 | 0.840 | 3.513 | 2.007 |
| calibrated rule [4] | 0.832 | 0.844 | 3.828 | 2.189 |
| rule + C-law on raw P (K=5) | 0.873 | 0.895 | 4.084 | 2.332 |
| rule + constant-alpha raw P (K=5) | 0.886 | 0.901 | 4.574 | 2.737 |
| scalar oracle (needs ground truth) | 0.894 | 0.904 |  |  |
| Monte Carlo, 200 worlds | 0.887 | 0.906 | 12.549 | 1.676 |
| Monte Carlo, 1000 worlds | 0.939 | 0.945 | 59.945 | 6.234 |

Per-method timings (single process, median of 3; one uniform-prior instance per family and range):

| method                                        |   ms (scale) |   ms (test) |   iters (scale) |   iters (test) |
|:----------------------------------------------|-------------:|------------:|----------------:|---------------:|
| AP(eta=1) standalone, to fixed point          |        16.18 |        6.81 |             157 |            132 |
| E2E calibrated rule                           |         3.83 |        2.19 |               0 |              0 |
| E2E rule + C-law raw P (PBX_rule_best)        |         4.08 |        2.33 |               0 |              0 |
| E2E rule + constant-alpha raw P (CTRL_lin0.9) |         4.57 |        2.74 |               0 |              0 |
| KNZ MP, 2 iterations                          |         0.17 |        0.06 |               2 |              2 |
| KNZ MP, converged                             |         1.45 |        0.29 |              23 |             24 |
| Mode (b) nominal polish, K=1 (excl. base)     |         0.02 |        0.01 |               1 |              1 |
| Monte Carlo, 1000 worlds                      |        59.94 |        6.23 |            1000 |           1000 |
| Monte Carlo, 200 worlds                       |        12.55 |        1.68 |             200 |            200 |
| Monte Carlo, 50 worlds                        |         3.33 |        0.59 |              50 |             50 |
| Realization C on raw P, K=5 (excl. base)      |         0.28 |        0.21 |               5 |              5 |
| Realization C standalone, T=3                 |         0.17 |        0.13 |               3 |              3 |
| linear raw-P propagation, K=5 (excl. base)    |         0.18 |        0.13 |               5 |              5 |
| operators (P_esc, W, EW, P)                   |         1.17 |        0.74 |               1 |              1 |
| rule_EW (calibrated scalar)                   |         2.32 |        1.25 |             183 |            159 |
| scalar oracle sweep (40 alphas, batched)      |       409.12 |       38.9  |               0 |              0 |

## 9. Exploratory (not pre-registered): raw-P propagation on its own

Damped propagation on the raw probability adjacency from scratch, y ← (1−a)x + a·Py, K sweeps (a truncated Katz-type score with p-weights), alone (β=1) and rank-blended 50/50 with the rule (β=0.5). Chosen after seeing dev and test; the real and scale panels are its hold-out.

|                |    dev |   test |
|:---------------|-------:|-------:|
| (0.5, 2, 0.5)  | 0.8875 | 0.8861 |
| (0.5, 2, 1.0)  | 0.837  | 0.8492 |
| (0.5, 5, 0.5)  | 0.8991 | 0.8988 |
| (0.5, 5, 1.0)  | 0.8721 | 0.8806 |
| (0.5, 10, 0.5) | 0.8988 | 0.8999 |
| (0.5, 10, 1.0) | 0.8757 | 0.8861 |
| (0.9, 2, 0.5)  | 0.8958 | 0.8941 |
| (0.9, 2, 1.0)  | 0.8492 | 0.8601 |
| (0.9, 5, 0.5)  | 0.9044 | 0.904  |
| (0.9, 5, 1.0)  | 0.8821 | 0.8907 |
| (0.9, 10, 0.5) | 0.8999 | 0.9023 |
| (0.9, 10, 1.0) | 0.8765 | 0.8907 |

|                |   real |   scale |
|:---------------|-------:|--------:|
| (0.5, 2, 0.5)  | 0.8835 |  0.8619 |
| (0.5, 2, 1.0)  | 0.8372 |  0.8055 |
| (0.5, 5, 0.5)  | 0.8893 |  0.8766 |
| (0.5, 5, 1.0)  | 0.8516 |  0.838  |
| (0.5, 10, 0.5) | 0.8844 |  0.8768 |
| (0.5, 10, 1.0) | 0.8286 |  0.839  |
| (0.9, 2, 0.5)  | 0.8956 |  0.8744 |
| (0.9, 2, 1.0)  | 0.8447 |  0.824  |
| (0.9, 5, 0.5)  | 0.902  |  0.8881 |
| (0.9, 5, 1.0)  | 0.8644 |  0.8582 |
| (0.9, 10, 0.5) | 0.8952 |  0.8854 |
| (0.9, 10, 1.0) | 0.8346 |  0.8482 |

## 10. Threats to validity

* **Ground-truth noise.** With M=2000, dense instances have unreliable ground-truth ranks (dense Watts–Strogatz on dev: ρ(y*_A, y*_B) = 0.29). All comparisons are paired, so this adds noise but no bias.
* **Operator choice.** The per-node law was tested on five operators and four scale conventions, with and without anchoring, in both modes; the true-y* experiment removes estimation error entirely.
* **Per-node oracle is a lower bound.** Coordinate ascent over a 14-value grid, 3 sweeps, time-capped (120 s at n ≤ 300, 240 s above); at n = 900 on lattices the cap binds, so headroom there is understated — as was [4]'s.
* **Selection.** Dev searches are large (40–864 configurations per method family); held-out evaluation of frozen winners and Holm correction address this. The exploratory raw-P section is not covered by the pre-registration.
* **Reimplementation of [4].** The rule's constants were used as published; the operator follows [4]'s equations, but implementation details of [4] (iteration count, prior generator) were not available.
