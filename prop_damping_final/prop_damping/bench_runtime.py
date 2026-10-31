"""Single-process wall-clock benchmark of the main methods (median of 3 runs).

Instances: one per (family, range) with uniform prior from the test panel
(n=100) and the same from the scale panel (n=900).  Writes
results/runtime.csv with milliseconds and iteration counts."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import prop_damping_lib as L
from build_panels import DATA, load_instance
from experiments import ap_run, polish_fixed, excess_embed, nrm, INF
from controls import sweeps

RES = Path(__file__).resolve().parent / "results"


def timed(f, reps=3):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts)), out


def bench(path):
    inst, yA, yB, _ = load_instance(path)
    x = inst.x
    rows = []
    t_ops, ops = timed(lambda: L.build_operators(inst))
    inst._ops = ops
    EW, P = ops["EW"], ops["P"]
    a0 = L.rule_alpha(ops["Pbar"])
    ms, (yr, it) = timed(lambda: L.diffuse(EW, x, a0))
    base = dict(id=int(Path(path).stem), n=inst.n, m=inst.m, family=inst.meta["family"],
                prange=inst.meta["prange"])
    rows.append(dict(base, method="operators (P_esc, W, EW, P)", ms=t_ops, iters=1))
    rows.append(dict(base, method="rule_EW (calibrated scalar)", ms=ms, iters=it))
    ms, (mu, k) = timed(lambda: L.knz_mp(inst, iters=2))
    rows.append(dict(base, method="KNZ MP, 2 iterations", ms=ms, iters=2))
    ms, (mu, k) = timed(lambda: L.knz_mp(inst, iters=None))
    rows.append(dict(base, method="KNZ MP, converged", ms=ms, iters=k + 1))
    ms, out = timed(lambda: ap_run(EW, x, a0, 1.0, x + EW @ x, Ts=(INF,)))
    rows.append(dict(base, method="AP(eta=1) standalone, to fixed point", ms=ms, iters=out["iters"]))
    al = L.anchored_alpha(yr / x, a0, 1.0)
    ms, _ = timed(lambda: polish_fixed(EW, x, yr, al, (1,)))
    rows.append(dict(base, method="Mode (b) nominal polish, K=1 (excl. base)", ms=ms, iters=1))
    ms, _ = timed(lambda: L.run_realization(P, x, "C", scale="raw", y_init=excess_embed(yr, x, 4.0),
                                            T_max=5, checkpoints=(5,)))
    rows.append(dict(base, method="Realization C on raw P, K=5 (excl. base)", ms=ms, iters=5))
    ms, _ = timed(lambda: sweeps(P, x, excess_embed(yr, x, 1.0), "lin0.9", 5))
    rows.append(dict(base, method="linear raw-P propagation, K=5 (excl. base)", ms=ms, iters=5))
    ms, _ = timed(lambda: L.run_realization(P, x, "C", scale="raw", T_max=3, checkpoints=(3,)))
    rows.append(dict(base, method="Realization C standalone, T=3", ms=ms, iters=3))
    ms, _ = timed(lambda: L.scalar_sweep(EW, x), reps=1)
    rows.append(dict(base, method="scalar oracle sweep (40 alphas, batched)", ms=ms, iters=0))
    for k in (50, 200, 1000):
        ms, _ = timed(lambda: L.mc_reach(inst, k, 0), reps=3)
        rows.append(dict(base, method=f"Monte Carlo, {k} worlds", ms=ms, iters=k))

    # ---- end-to-end pipelines (operators built from scratch, ranking/blending included)
    def e2e_rule():
        inst._ops = None
        o = inst.ops()
        return L.diffuse(o["EW"], x, L.rule_alpha(o["Pbar"]))[0]

    def e2e_lin09():
        inst._ops = None
        o = inst.ops()
        y_r = L.diffuse(o["EW"], x, L.rule_alpha(o["Pbar"]))[0]
        out, _ = sweeps(o["P"], x, excess_embed(y_r, x, 1.0), "lin0.9", 5)
        return 0.5 * L.ranks_of(y_r) + 0.5 * L.ranks_of(out[5])

    def e2e_pbx():
        inst._ops = None
        o = inst.ops()
        y_r = L.diffuse(o["EW"], x, L.rule_alpha(o["Pbar"]))[0]
        out, y_p, _ = L.run_realization(o["P"], x, "C", scale="raw", y_init=excess_embed(y_r, x, 4.0),
                                        T_max=5, checkpoints=(5,), record_alpha=True)
        return 0.5 * nrm(y_r) + 0.5 * nrm(out[5]["y"])

    for name, f in (("E2E calibrated rule", e2e_rule), ("E2E rule + constant-alpha raw P (CTRL_lin0.9)", e2e_lin09),
                    ("E2E rule + C-law raw P (PBX_rule_best)", e2e_pbx)):
        ms, _ = timed(f)
        rows.append(dict(base, method=name, ms=ms, iters=0))
    inst._ops = ops
    return rows


if __name__ == "__main__":
    rows = []
    for panel in ("test", "scale"):
        specs = json.loads((DATA / panel / "specs.json").read_text())
        seen = set()
        for i, s in enumerate(specs):
            key = (s["fam"], s["pr"])
            if s["prior"] != "uniform" or key in seen:
                continue
            seen.add(key)
            rows += [dict(r, panel=panel) for r in bench(DATA / panel / f"{i:04d}.npz")]
    D = pd.DataFrame(rows)
    D.to_csv(RES / "runtime.csv", index=False)
    print(D.groupby(["panel", "method"])[["ms", "iters"]].median().round(2).to_string())
