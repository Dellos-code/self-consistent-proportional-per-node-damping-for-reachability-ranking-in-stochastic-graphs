"""Build all benchmark panels and their Monte-Carlo ground truth.

Each instance is stored as data/<panel>/<id>.npz with edges, p, x, two
independent ground-truth estimates (yA: evaluation, yB: oracle fitting, each
M=2000 worlds) and the second moment of A (for standard errors).

Usage:  python build_panels.py [panel ...]     (default: all)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import prop_damping_lib as L

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REAL = ROOT / "data_real"
M_GT = 2000

FAMILIES = ["er", "ba", "ws", "grid", "rect", "sbm", "cp", "tree", "barbell"]
RANGES = ["R1", "R2", "R3", "R4", "R5"]


def spec_list(panel):
    S = []
    if panel in ("dev", "test"):
        reps = (0, 1) if panel == "dev" else (2, 3)
        for fam in FAMILIES:
            for pr in RANGES:
                for prior in ("uniform", "dir1"):
                    for r in reps:
                        S.append(dict(kind="syn", fam=fam, n=100, pr=pr, prior=prior, rep=r, panel="main"))
    elif panel == "scale":
        for fam in FAMILIES:
            for pr in RANGES:
                for prior in ("uniform", "dir1"):
                    sp_ = dict(kind="syn", fam=fam, n=900, pr=pr, prior=prior, rep=0, panel="scale")
                    if fam == "barbell":
                        sp_["kw"] = {"m1": 90}
                    S.append(sp_)
    elif panel == "prior":
        priors = ["uniform", "dir10", "dir3", "dir1", "dir0.3", "logn0.5", "logn1", "logn2",
                  "seed5_0.1", "seed1_0.1"]
        for fam, n in (("grid", 144), ("er", 150), ("ba", 150), ("ws", 150), ("sbm", 150)):
            for pr in ("R2", "R3", "R4"):
                for prior in priors:
                    S.append(dict(kind="syn", fam=fam, n=n, pr=pr, prior=prior, rep=0, panel="prior"))
    elif panel == "real":
        schemes = ["unif0.05-0.2", "unif0.2-0.5", "unif0.5-0.9", "wc"]
        for g in ("karate", "lesmis", "davis"):
            for sc in schemes:
                for prior in ("uniform", "dir1"):
                    S.append(dict(kind="classic", g=g, scheme=sc, prior=prior, rep=0))
        for g in ("cora", "citeseer", "amazon_electronics_photo", "ms_academic_cs"):
            for rep in (0, 1):
                for sc in schemes:
                    for prior in ("uniform", "dir1"):
                        S.append(dict(kind="snow", g=g, size=500, scheme=sc, prior=prior, rep=rep))
    else:
        raise ValueError(panel)
    return S


def make_instance(spec):
    if spec["kind"] == "syn":
        return L.make_synthetic(spec["fam"], spec["n"], spec["pr"], spec["prior"], spec["rep"],
                                panel=spec["panel"], **spec.get("kw", {}))
    if spec["kind"] == "classic":
        import networkx as nx
        G = {"karate": nx.karate_club_graph, "lesmis": nx.les_miserables_graph,
             "davis": nx.davis_southern_women_graph}[spec["g"]]()
        E, n = L._nx_edges(G)
    else:
        A = L.load_npz_graph(str(REAL / f"{spec['g']}.npz"))
        nodes = L.snowball(A, spec["size"], L.stable_seed("snow", spec["g"], spec["rep"]))
        E = L.subgraph_edges(A, nodes)
        n = len(nodes)
    sd = L.stable_seed(spec["kind"], spec["g"], spec.get("rep", 0), spec["scheme"])
    p = L.real_probs(E, n, spec["scheme"], sd)
    x = L.make_prior(spec["prior"], n, L.stable_seed(sd, "x", spec["prior"]))
    name = f"real:{spec['g']}:{spec['scheme']}:{spec['prior']}:r{spec.get('rep', 0)}"
    meta = dict(panel="real", family=spec["g"], n=n, prange=spec["scheme"], prior=spec["prior"],
                rep=spec.get("rep", 0), genre=("classic" if spec["kind"] == "classic" else spec["g"]))
    return L.Instance(name, n, E, p, x, meta)


def build_one(args):
    panel, idx, spec = args
    out = DATA / panel / f"{idx:04d}.npz"
    if out.exists():
        return idx, 0.0
    t0 = time.time()
    inst = make_instance(spec)
    sA = L.stable_seed(inst.name, "A")
    sB = L.stable_seed(inst.name, "B")
    yA, yA2 = L.mc_reach(inst, M_GT, sA, return_second_moment=True)
    yB = L.mc_reach(inst, M_GT, sB)
    ops = inst.ops()
    meta = dict(inst.meta, name=inst.name, m=inst.m, Pbar=ops["Pbar"], **L.prior_stats(inst.x),
                ceiling=L.corr_ranks(L.ranks_of(yB), L.ranks_of(yA)), gt_seconds=time.time() - t0)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, edges=inst.edges, p=inst.p, x=inst.x, yA=yA, yA2=yA2, yB=yB,
                        meta=json.dumps(meta, default=str))
    tmp.rename(out)
    return idx, time.time() - t0


def load_instance(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    inst = L.Instance(meta["name"], len(d["x"]), d["edges"], d["p"], d["x"], meta)
    return inst, d["yA"], d["yB"], d["yA2"]


def main(panels):
    for panel in panels:
        (DATA / panel).mkdir(parents=True, exist_ok=True)
        specs = spec_list(panel)
        (DATA / panel / "specs.json").write_text(json.dumps(specs, indent=0))
        jobs = [(panel, i, s) for i, s in enumerate(specs)]
        t0 = time.time()
        with Pool(2) as pool:
            for k, (idx, dt) in enumerate(pool.imap_unordered(build_one, jobs, chunksize=1)):
                if (k + 1) % 20 == 0:
                    print(f"[{panel}] {k + 1}/{len(jobs)}  elapsed {time.time() - t0:.0f}s", flush=True)
        print(f"[{panel}] done: {len(jobs)} instances in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["dev", "test", "prior", "real", "scale"])
