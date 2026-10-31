"""Mean-field and permutation controls for every frozen realization config.

For each frozen configuration that runs a realization (standalone 'sa' or
polisher 'pbx'), re-run the identical pipeline with the per-node alpha replaced
by its per-step mean ('mean') or randomly reassigned ('perm').  The construction
is mechanical (no free choices), so it can be evaluated on dev and test alike.
Usage: python mf_controls.py <panel>[,<panel>]"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
import prop_damping_lib as L
from build_panels import DATA, load_instance
from experiments import excess_embed, nrm, INF

RES = Path(__file__).resolve().parent / "results"
SEL = json.loads((RES / "selected.json").read_text())


def one(path):
    inst, yA, yB, _ = load_instance(path)
    rA = L.ranks_of(yA)
    x, ops = inst.x, inst.ops()
    a0 = L.rule_alpha(ops["Pbar"])
    yr, _ = L.diffuse(ops["EW"], x, a0)
    rows = []
    for name, c in SEL.items():
        if c["type"] not in ("sa", "pbx"):
            continue
        for tr in (None, "mean", "perm"):
            if c["type"] == "sa":
                out, y, _ = L.run_realization(ops[c["op"]], x, c["kind"], scale=c["scale"], gamma=c["gamma"],
                                              T_max=c["T"], checkpoints=(c["T"],), record_alpha=True,
                                              alpha_transform=tr, seed=L.stable_seed(inst.name, "mf"))
                yy = out[c["T"]].get("y", y)
            else:
                Tm = 200 if c["T"] == INF else c["T"]
                out, y, _ = L.run_realization(ops[c["op"]], x, c["kind"], scale=c["scale"], gamma=c["gamma"],
                                              T_max=Tm, checkpoints=(Tm,), y_init=excess_embed(yr, x, c["gamma"]),
                                              record_alpha=True, alpha_transform=tr,
                                              seed=L.stable_seed(inst.name, "mf"))
                yp = out[Tm].get("y", y)
                b = c["beta"]
                yy = yp if b == 1.0 else ((1 - b) * nrm(yr) + b * nrm(yp) if c["blend"] == "s"
                                          else (1 - b) * L.ranks_of(yr) + b * L.ranks_of(yp))
            rows.append(dict(id=int(Path(path).stem), config=name, control=tr or "law", rho=L.spearman(yy, rA)))
    return rows


if __name__ == "__main__":
    for panel in sys.argv[1].split(","):
        paths = sorted((DATA / panel).glob("[0-9][0-9][0-9][0-9].npz"))
        t0 = time.time()
        with Pool(2) as pool:
            rows = [r for rr in pool.map(one, [str(p) for p in paths], chunksize=4) for r in rr]
        pd.DataFrame(rows).to_csv(RES / f"mfcontrols_{panel}.csv.gz", index=False, compression="gzip")
        print(panel, len(paths), f"{time.time() - t0:.0f}s", flush=True)
