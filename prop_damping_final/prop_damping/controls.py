"""Matched controls for the Mode (b) realization polishers.

The selected polisher (Realization C on the raw probability operator P, from an
excess embedding of the calibrated-rule output, K sweeps, 50% blend) is
compared against the SAME pipeline with C's per-node law alpha_i = S_i/(x_i+S_i)
replaced by
  * Plin(a):   a constant damping a in {0.5, 0.9, 1.0}  (linear propagation on P)
  * Cmean:     the per-sweep mean of C's alpha (same average damping, no per-node law)
  * Cperm:     C's alpha vector randomly reassigned to vertices each sweep
so that any gain attributable to the proportional law itself is isolated.

Usage: python controls.py <panel>[,<panel>...]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import prop_damping_lib as L
from build_panels import DATA, load_instance
from experiments import excess_embed, nrm

RES = Path(__file__).resolve().parent / "results"
KS = (1, 2, 5)


def sweeps(P, x, y0, law, K, seed=None):
    rng = np.random.default_rng(seed)
    y = y0.copy()
    out, amean = {}, {}
    for k in range(1, K + 1):
        S = P @ y
        if law == "C":
            a = S / (x + S)
        elif law == "Cmean":
            a = np.full_like(x, np.mean(S / (x + S)))
        elif law == "Cperm":
            a = (S / (x + S))[rng.permutation(len(x))]
        elif law.startswith("lin"):
            a = np.full_like(x, float(law[3:]))
        else:
            raise ValueError(law)
        y = (1 - a) * x + a * S
        if not np.all(np.isfinite(y)):
            break
        out[k] = y.copy()
        amean[k] = (float(a.mean()), float(np.mean(a > 0.99)))
    return out, amean


def one(path):
    inst, yA, yB, _ = load_instance(path)
    rA = L.ranks_of(yA)
    x, ops = inst.x, inst.ops()
    a0 = L.rule_alpha(ops["Pbar"])
    yr, _ = L.diffuse(ops["EW"], x, a0)
    rows = []
    for g in (1.0, 4.0):
        y0 = excess_embed(yr, x, g)
        for law in ("C", "Cmean", "Cperm", "lin0.5", "lin0.9", "lin1.0"):
            out, am = sweeps(ops["P"], x, y0, law, max(KS), seed=L.stable_seed(inst.name, "cperm"))
            for k in KS:
                if k not in out:
                    continue
                yp = out[k]
                for beta, blend, ys in ((1.0, "s", yp),
                                        (0.5, "s", 0.5 * nrm(yr) + 0.5 * nrm(yp)),
                                        (0.5, "r", 0.5 * L.ranks_of(yr) + 0.5 * L.ranks_of(yp))):
                    rows.append(dict(id=int(Path(path).stem), law=law, gamma=g, K=k, beta=beta, blend=blend,
                                     rho=L.spearman(ys, rA), amean=am[k][0], asat=am[k][1]))
    rows.append(dict(id=int(Path(path).stem), law="rule", gamma=0, K=0, beta=0, blend="-",
                     rho=L.spearman(yr, rA), amean=a0, asat=float(a0 > 0.99)))
    return rows


if __name__ == "__main__":
    for panel in sys.argv[1].split(","):
        paths = sorted((DATA / panel).glob("[0-9][0-9][0-9][0-9].npz"))
        t0 = time.time()
        with Pool(2) as pool:
            rows = [r for rr in pool.map(one, [str(p) for p in paths], chunksize=4) for r in rr]
        pd.DataFrame(rows).to_csv(RES / f"controls_{panel}.csv.gz", index=False, compression="gzip")
        print(panel, len(paths), "instances", f"{time.time() - t0:.0f}s", flush=True)
