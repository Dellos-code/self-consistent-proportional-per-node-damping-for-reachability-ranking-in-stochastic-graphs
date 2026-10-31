"""EXPLORATORY (not pre-registered): is the Mode (b) gain the raw-P propagation itself,
or the ensemble with the calibrated rule?  Raw-P damped propagation from scratch
(y <- (1-a) x + a P y, y0 = x, K sweeps; a truncated Katz-type score with
p-weights), alone and rank-blended with the rule."""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
import prop_damping_lib as L
from build_panels import DATA, load_instance
from experiments import nrm

def one(path):
    inst, yA, yB, _ = load_instance(path)
    rA = L.ranks_of(yA); x, ops = inst.x, inst.ops(); P = ops["P"]
    yr, _ = L.diffuse(ops["EW"], x, L.rule_alpha(ops["Pbar"]))
    rows = []
    for a in (0.5, 0.9):
        y = x.copy()
        for k in range(1, 11):
            y = (1 - a) * x + a * (P @ y)
            if k in (2, 5, 10):
                for b, ys in ((1.0, y), (0.5, 0.5 * L.ranks_of(yr) + 0.5 * L.ranks_of(y))):
                    rows.append(dict(id=int(Path(path).stem), a=a, K=k, beta=b, rho=L.spearman(ys, rA)))
    return rows

if __name__ == "__main__":
    panels = sys.argv[1].split(",") if len(sys.argv) > 1 else ["dev", "test"]
    out = []
    for panel in panels:
        paths = sorted((DATA / panel).glob("[0-9][0-9][0-9][0-9].npz"))
        with Pool(2) as pool:
            rows = [dict(r, panel=panel) for rr in pool.map(one, [str(p) for p in paths], chunksize=4) for r in rr]
        out += rows
    D = pd.DataFrame(out)
    import analysis as A
    stage = lambda p: "grid" if p in ("dev", "test") else "confirm"
    pri = pd.concat([A.load_stage(stage(p), p)[1][["prior"]].assign(panel=p) for p in panels]).reset_index()
    D = D.merge(pri, on=["id", "panel"])
    D.to_csv(f"results/explore_rawP_{'_'.join(panels)}.csv.gz", index=False, compression="gzip")
    g = D.groupby(["panel", "a", "K", "beta", "prior"])["rho"].mean().unstack("prior")
    g["all"] = g.mean(1)
    pd.set_option("display.width", 200)
    print(g.round(4).to_string())
