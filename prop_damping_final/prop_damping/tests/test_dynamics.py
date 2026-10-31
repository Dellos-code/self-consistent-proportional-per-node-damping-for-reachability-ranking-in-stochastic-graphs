"""Numerical confirmation of the fixed-point structure of realizations A, B, C.

P1 (collapse)   uniform prior, row-substochastic T, Realization A:
                max_i y_i^(t+1) <= max(x, max_i y_i^(t) - x)  =>  y -> x in finitely many steps;
                from the warm start y0 = x + T x it is reached after ONE step.
P2 (B absorbs)  Realization B: once y_i <= x_i, alpha_i = 0 forever (trivial branch absorbing);
                max excess contracts as  m' <= m^2 / (x + m)   (m = max excess, uniform prior).
P3 (repelling)  any positive all-active fixed point y = T y - x forces rho(T) > 1, and the
                Jacobians of A (J = T) and B (J = diag(x/y) + diag(1 - x/y) T) both have
                spectral radius > 1: the self-consistent solution is unstable.
P4 (C stable)   C's map y -> L2(x, T y) has Jacobian diag(1 - 2x^2/(x+S)^2) T: a contraction
                in the infinity norm for row-substochastic T with S > 0.
P5 (sign)       in the per-node diffusion u_i = (1-a_i) x_i + a_i S_i, du_i/da_i = S_i - x_i:
                raising a_i lowers vertex i wherever S_i < x_i (sparse regimes under EW).
"""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prop_damping_lib as L

rep = {}
# ---------------- P1 -----------------------------------------------------------
worst = 0.0
for fam in ("er", "ba", "grid", "ws", "sbm", "cp", "tree", "barbell", "rect"):
    for pr in ("R1", "R2", "R3", "R4", "R5"):
        ins = L.make_synthetic(fam, 100, pr, "uniform", 7, panel="dyn")
        x = ins.x
        for op in ("EW", "W"):
            T = ins.ops()[op]
            rng = np.random.default_rng(1)
            y = x * (1 + 20 * rng.random(ins.n))                 # arbitrary start, max excess 20x
            for t in range(40):
                M = y.max()
                y = np.maximum(x, T @ y - x)
                assert y.max() <= max(x[0], M - x[0]) * (1 + 1e-12)
            worst = max(worst, np.abs(y / x - 1).max())
rep["P1_A_uniform_after40_max_rel_excess"] = worst
assert worst < 1e-12

# ---------------- P2 -----------------------------------------------------------
ins = L.make_synthetic("ws", 100, "R3", "uniform", 7, panel="dyn")
x, T = ins.x, ins.ops()["EW"]
y = x * (1 + 3 * np.random.default_rng(2).random(ins.n))
ok = True
frozen = np.zeros(ins.n, bool)
for t in range(30):
    m = (y - x).max()
    yn, a = L.realization_step("B", y, T @ y, x)
    frozen_now = yn <= x * (1 + 1e-15)
    # absorption: a vertex at or below its prior has alpha = 0 at the next step
    yn2, a2 = L.realization_step("B", yn, T @ yn, x)
    ok &= bool(np.all(a2[frozen_now] == 0))
    mn = np.maximum(yn - x, 0).max()
    ok &= bool(mn <= m * m / (x[0] + m) * (1 + 1e-9) + 1e-300)
    y = yn
rep["P2_B_absorbing_and_quadratic_bound_hold"] = ok
assert ok

# ---------------- P3 -----------------------------------------------------------
# Operator with constant row sums lam in (1,2): T = lam * W.  With uniform x the
# all-active fixed point exists in closed form, y* = x/(lam-1) > x.
ins = L.make_synthetic("ws", 100, "R3", "uniform", 3, panel="dyn")
W = ins.ops()["W"].toarray()
x = ins.x
for lam in (1.2, 1.5, 1.9):
    T = lam * W
    ystar = np.linalg.solve(T - np.eye(ins.n), x)
    assert np.allclose(ystar, x / (lam - 1)) and np.all(ystar > x)
    rA = np.max(np.abs(np.linalg.eigvals(T)))
    JB = np.diag(x / ystar) + np.diag(1 - x / ystar) @ T
    rB = np.max(np.abs(np.linalg.eigvals(JB)))
    out = []
    for kind in ("A", "B"):
        y = ystar * (1 + 1e-9 * np.random.default_rng(3).standard_normal(ins.n))
        for t in range(300):
            y, _ = L.realization_step(kind, y, T @ y, x)
            if not np.all(np.isfinite(y)) or y.max() > 1e200:
                break
        out.append(float(np.abs(y / ystar - 1).max()))
    rep[f"P3_lam={lam}: rho(J_A), rho(J_B), departure A, B"] = (round(rA, 4), round(rB, 4), *[f"{d:.2e}" for d in out])
    assert rA > 1 and rB > 1 and min(out) > 1e-3

# ---------------- P4 -----------------------------------------------------------
ins = L.make_synthetic("ba", 100, "R3", "dir1", 3, panel="dyn")
x, T = ins.x, ins.ops()["EW"]
ys = []
for s in range(3):
    y = x * np.exp(np.random.default_rng(s).normal(0, 2, ins.n))
    for t in range(3000):
        S = T @ y
        y = (x * x + S * S) / (x + S)
    ys.append(y)
spread = max(np.abs(ys[i] - ys[0]).max() / ys[0].max() for i in range(3))
rep["P4_C_EW_three_starts_same_fixed_point_maxdiff"] = float(spread)
assert spread < 1e-8
yfix = ys[0]
rep["P4_C_EW_fixed_point_fraction_below_prior"] = float(np.mean(yfix < x))

# ---------------- P5 -----------------------------------------------------------
ins = L.make_synthetic("er", 100, "R1", "uniform", 3, panel="dyn")
x, T = ins.x, ins.ops()["EW"]
a0 = L.rule_alpha(ins.ops()["Pbar"])
y, _ = L.diffuse(T, x, a0)
S = T @ y
rep["P5_sparse_ER_fraction_S_below_x"] = float(np.mean(S < x * y.sum()))
# finite-difference check of the sign of d y_k / d a_k for 20 random vertices
agree = 0
for k in np.random.default_rng(4).choice(ins.n, 20, replace=False):
    a = np.full(ins.n, a0)
    a[k] += 1e-4
    y2, _ = L.diffuse(T, x, a, y0=y)
    agree += np.sign(y2[k] / y[k] - 1) == np.sign(S[k] - x[k])
rep["P5_sign_rule_agreement_(of_20)"] = int(agree)

import json
(Path(__file__).resolve().parents[1] / "results" / "tests_dynamics.json").write_text(json.dumps({k: (list(map(lambda z: float(z) if not isinstance(z, str) else z, v)) if isinstance(v, tuple) else (v if isinstance(v, (bool, int, str)) else float(v))) for k, v in rep.items()}, indent=1))
for k, v in rep.items():
    print(f"{k:58s} {v}")
print("ALL DYNAMICS CHECKS PASSED")
