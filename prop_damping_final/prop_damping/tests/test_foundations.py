"""Foundational checks: algebra of Theorem 1, Monte-Carlo vs exact ground truth,
KNZ zeroth iterate = P_esc, diffusion fixed point, collapse lemma."""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prop_damping_lib as L

rng = np.random.default_rng(0)
report = {}

# 1. Theorem 1: both branches satisfy both equations ----------------------------
x = rng.uniform(1e-4, 1.0, 100000)
S = rng.uniform(0, 5.0, 100000)
(y1, a1), (y2, a2) = L.theorem1_roots(x, S)
r_diff1 = np.abs(y1 - ((1 - a1) * x + a1 * S)).max()
act = S > 2 * x
r_diff2 = np.abs(y2[act] - ((1 - a2[act]) * x[act] + a2[act] * S[act])).max() / S[act].max()
r_prop2 = np.abs(a2[act] / (1 - a2[act]) - (y2[act] - x[act]) / x[act]).max() / ((y2[act] - x[act]) / x[act]).max()
quad = np.abs(y2**2 - S * y2 + x * S - x**2).max()
assert r_diff1 < 1e-12 and r_diff2 < 1e-12 and r_prop2 < 1e-10 and quad < 1e-9
assert np.all((a2[act] > 0) & (a2[act] < 1))
report["theorem1_max_residual"] = float(max(r_diff2, r_prop2))

# 2. Monte Carlo vs exact enumeration on small graphs ---------------------------
errs = []
for t in range(6):
    n = 7
    E = L.canon_edges([(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.45])
    p = rng.uniform(0.05, 0.95, len(E))
    xp = rng.dirichlet(np.ones(n))
    inst = L.Instance(f"tiny{t}", n, E, p, xp)
    ye = L.exact_reach(inst)
    ym, ym2 = L.mc_reach(inst, 40000, seed=t, return_second_moment=True)
    se = np.sqrt(np.maximum(ym2 - ym**2, 0) / 40000)
    z = np.abs(ym - ye) / np.maximum(se, 1e-12)
    errs.append(z.max())
report["mc_vs_exact_max_z"] = float(max(errs))
assert max(errs) < 5.0, errs

# 3. KNZ zeroth iterate equals P_esc -------------------------------------------
inst = L.make_synthetic("er", 200, "R3", "uniform", 0, panel="t")
mu0, _ = L.knz_mp(inst, iters=0)
report["knz0_minus_pesc"] = float(np.abs(mu0 - inst.ops()["Pesc"]).max())
assert report["knz0_minus_pesc"] < 1e-12

# 4. Diffusion: lazy and plain power iterations agree on a non-bipartite graph,
#    and the fixed point satisfies y = N((1-a)x + a T y) ---------------------------
inst = L.make_synthetic("ws", 150, "R3", "dir1", 0, panel="t")
T = inst.ops()["EW"]
for a in (0.5, 0.9, 0.99):
    y1, _ = L.diffuse(T, inst.x, a, lazy=True)
    y2, _ = L.diffuse(T, inst.x, a, lazy=False)
    r = (1 - a) * inst.x + a * (T @ y1); r /= r.sum()
    assert np.abs(y1 - y2).sum() < 1e-9 and np.abs(r - y1).sum() < 1e-10
# bipartite grid with alpha=1 (plain power method oscillates, lazy converges)
g = L.make_synthetic("grid", 100, "R4", "uniform", 0, panel="t")
yg, itg = L.diffuse(g.ops()["EW"], g.x, 1.0)
rg = g.ops()["EW"] @ yg; rg /= rg.sum()
report["grid_alpha1_residual"] = float(np.abs(rg - yg).sum())
assert report["grid_alpha1_residual"] < 1e-9

# 5. Collapse lemma: realization A, uniform prior, row-substochastic operator,
#    paper's warm start y0 = x + T x  ->  y1 = x exactly, for every vertex ------
coll = []
for fam in ("er", "ba", "ws", "grid", "sbm", "cp", "tree", "barbell", "rect"):
    for pr in ("R1", "R3", "R5"):
        ins = L.make_synthetic(fam, 100, pr, "uniform", 0, panel="t")
        for op in ("EW", "W"):
            T = ins.ops()[op]
            out, y, _ = L.run_realization(T, ins.x, "A", scale="raw", T_max=1, checkpoints=(1,))
            coll.append(np.abs(y - ins.x).max() / ins.x[0])
report["collapse_A_uniform_max_rel_excess_after_1_step"] = float(max(coll))
assert max(coll) < 1e-12, max(coll)

import json
(Path(__file__).resolve().parents[1] / "results" / "tests_foundations.json").write_text(json.dumps({k: (list(map(lambda z: float(z) if not isinstance(z, str) else z, v)) if isinstance(v, tuple) else (v if isinstance(v, (bool, int, str)) else float(v))) for k, v in report.items()}, indent=1))
for k, v in report.items():
    print(f"{k:55s} {v:.3e}" if isinstance(v, float) else f"{k:55s} {v}")
print("ALL FOUNDATION CHECKS PASSED")
