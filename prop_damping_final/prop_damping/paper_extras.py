"""Supplementary computations for the paper.

  traces      per-step dynamics of Realizations A/B/C on the dev panel (Fig. 1)
  complement  rank-error complementarity of the calibrated diffusion and raw-P
              propagation on test/real/scale (why dual-operator polishing works)
  sweep       single-process size sweep: end-to-end wall clock and accuracy of
              the dual-operator ensemble vs batched Monte Carlo (speed crossover)

Usage: python paper_extras.py traces|complement|sweep
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


# ----------------------------------------------------------------------------- traces
TRACE_CONFS = [("A", "EW", "raw", 1.0), ("B", "EW", "raw", 1.0), ("C", "EW", "raw", 1.0),
               ("A", "EW", "l1", 1.0), ("B", "EW", "l1", 1.0),
               ("A", "EW", "ex", 16.0), ("B", "EW", "ex", 16.0), ("C", "EW", "ex", 16.0),
               ("C", "P", "raw", 1.0)]
T_TRACE = 60


def trace_one(path):
    inst, yA, yB, _ = load_instance(path)
    rA = L.ranks_of(yA)
    x, ops = inst.x, inst.ops()
    rows = []
    for kind, op, scale, g in TRACE_CONFS:
        out, y, div = L.run_realization(ops[op], x, kind, scale=scale, gamma=g, T_max=T_TRACE,
                                        checkpoints=tuple(range(1, T_TRACE + 1)), target_ranks=rA)
        for t in range(1, T_TRACE + 1):
            r = out[t]
            rows.append(dict(id=int(Path(path).stem), prior=inst.meta["prior"], kind=kind, op=op, scale=scale,
                             gamma=g, t=t, rho=r["rho"], trivial=r["trivial"], active=r["act"]))
    return rows


# ----------------------------------------------------------------------------- complementarity
def raw_p_scratch(P, x, a=0.9, K=5):
    y = x.copy()
    for _ in range(K):
        y = (1 - a) * x + a * (P @ y)
    return y


def complement_one(args):
    panel, path = args
    inst, yA, yB, _ = load_instance(path)
    rA = L.ranks_of(yA)
    x, ops = inst.x, inst.ops()
    yr, _ = L.diffuse(ops["EW"], x, L.rule_alpha(ops["Pbar"]))
    yp = raw_p_scratch(ops["P"], x)
    rr, rp = L.ranks_of(yr), L.ranks_of(yp)
    blend = 0.5 * rr + 0.5 * rp
    n = inst.n
    er, ep = (rr - rA) / n, (rp - rA) / n
    c = np.corrcoef(er, ep)[0, 1] if er.std() > 0 and ep.std() > 0 else np.nan
    return dict(panel=panel, id=int(Path(path).stem), prior=inst.meta["prior"], family=inst.meta["family"],
                Pbar=inst.meta["Pbar"], rho_rule=L.spearman(yr, rA), rho_rawP=L.spearman(yp, rA),
                rho_blend=L.spearman(blend, rA), err_corr=c,
                rmse_rule=float(np.sqrt(np.mean(er ** 2))), rmse_rawP=float(np.sqrt(np.mean(ep ** 2))))


# ----------------------------------------------------------------------------- size sweep
SWEEP_N = (100, 150, 200, 300, 400, 600, 900, 1300, 2000)
SWEEP_FAM = ("er", "ba", "ws", "grid")


def timed(f, reps=5):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts)), out


def ensemble_end_to_end(inst):
    """Operators + calibrated rule + excess embedding + 5 sweeps of constant-alpha raw-P
    propagation + 50/50 rank blend (the CTRL_lin0.9 pipeline), from scratch."""
    inst._ops = None
    ops = inst.ops()
    x = inst.x
    yr, _ = L.diffuse(ops["EW"], x, L.rule_alpha(ops["Pbar"]))
    y = excess_embed(yr, x, 1.0)
    P = ops["P"]
    for _ in range(5):
        y = 0.1 * x + 0.9 * (P @ y)
    return 0.5 * L.ranks_of(yr) + 0.5 * L.ranks_of(y)


def rule_end_to_end(inst):
    inst._ops = None
    ops = inst.ops()
    return L.diffuse(ops["EW"], inst.x, L.rule_alpha(ops["Pbar"]))[0]


def sweep():
    rows = []
    for fam in SWEEP_FAM:
        for n in SWEEP_N:
            for prior in ("uniform", "dir1"):
                for rep in (0, 1):
                    inst = L.make_synthetic(fam, n, "R3", prior, rep, panel="sweep")
                    yA = L.mc_reach(inst, 2000, L.stable_seed(inst.name, "A"))
                    rA = L.ranks_of(yA)
                    t_ens, y_ens = timed(lambda: ensemble_end_to_end(inst))
                    t_rule, y_rule = timed(lambda: rule_end_to_end(inst))
                    row = dict(family=fam, n=inst.n, m=inst.m, prior=prior, rep=rep,
                               Pbar=float(inst.ops()["Pbar"]),
                               ms_ensemble=t_ens, rho_ensemble=L.spearman(y_ens, rA),
                               ms_rule=t_rule, rho_rule=L.spearman(y_rule, rA))
                    for k in (50, 200, 1000):
                        t_mc, y_mc = timed(lambda: L.mc_reach(inst, k, L.stable_seed(inst.name, "sweepmc", k)),
                                           reps=3)
                        row[f"ms_mc{k}"] = t_mc
                        row[f"rho_mc{k}"] = L.spearman(y_mc, rA)
                    rows.append(row)
                    print(fam, inst.n, prior, rep, f"ens {t_ens:.2f}ms mc200 {row['ms_mc200']:.2f}ms", flush=True)
    pd.DataFrame(rows).to_csv(RES / "size_sweep.csv", index=False)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "traces":
        paths = sorted((DATA / "dev").glob("[0-9][0-9][0-9][0-9].npz"))
        with Pool(2) as pool:
            rows = [r for rr in pool.map(trace_one, [str(p) for p in paths], chunksize=4) for r in rr]
        pd.DataFrame(rows).to_csv(RES / "traces_dev.csv.gz", index=False, compression="gzip")
        print("traces", len(rows))
    elif what == "complement":
        jobs = [(p, str(f)) for p in ("test", "real", "scale")
                for f in sorted((DATA / p).glob("[0-9][0-9][0-9][0-9].npz"))]
        with Pool(2) as pool:
            rows = pool.map(complement_one, jobs, chunksize=4)
        D = pd.DataFrame(rows)
        D.to_csv(RES / "complementarity.csv", index=False)
        print(D.groupby("panel")[["rho_rule", "rho_rawP", "rho_blend", "err_corr"]].mean().round(4))
    elif what == "sweep":
        sweep()
