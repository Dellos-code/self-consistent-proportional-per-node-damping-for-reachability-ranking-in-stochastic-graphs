"""Monte-Carlo estimators as baselines: rank by an independent k-world estimate
(seed disjoint from both ground-truth samples).  Writes results/mc_baseline.csv."""
import sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
import prop_damping_lib as L
from build_panels import DATA, load_instance

def one(args):
    panel, path = args
    inst, yA, yB, _ = load_instance(path)
    rA = L.ranks_of(yA)
    rows = []
    for k in (50, 200, 1000):
        t0 = time.perf_counter()
        y = L.mc_reach(inst, k, L.stable_seed(inst.name, "mcbase", k))
        ms = 1000 * (time.perf_counter() - t0)
        rows.append(dict(panel=panel, id=int(Path(path).stem), k=k, rho=L.spearman(y, rA), ms=ms))
    return rows

if __name__ == "__main__":
    jobs = [(p, str(f)) for p in sys.argv[1].split(",") for f in sorted((DATA / p).glob("[0-9][0-9][0-9][0-9].npz"))]
    with Pool(2) as pool:
        rows = [r for rr in pool.map(one, jobs, chunksize=4) for r in rr]
    D = pd.DataFrame(rows)
    D.to_csv("results/mc_baseline.csv", index=False)
    print(D.groupby(["panel", "k"])[["rho", "ms"]].mean().round(4).to_string())
