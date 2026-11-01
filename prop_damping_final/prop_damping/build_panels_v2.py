"""Additive fresh-panel specifications for the mixing study.

Ground-truth generation is intentionally separate from ``build_panels.py``.
"""
from __future__ import annotations

import json
import hashlib
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import build_panels as v1
import prop_damping_lib as L

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def _digest(path: Path, key: str) -> str:
    import numpy as np
    with np.load(path, allow_pickle=False) as d:
        return hashlib.sha256(np.asarray(d[key]).tobytes()).hexdigest()


def duplicate_report(panels: list[str]) -> dict[str, list[str]]:
    """Compare edge and prior hashes of new instances with all existing data."""
    existing = [p for panel in ("dev", "test", "prior", "real", "scale")
                for p in (DATA / panel).glob("[0-9][0-9][0-9][0-9].npz")]
    old = {(_digest(p, "edges"), _digest(p, "p"), _digest(p, "x")) for p in existing}
    duplicates = {}
    for panel in panels:
        for path in (DATA / panel).glob("[0-9][0-9][0-9][0-9].npz"):
            pair = (_digest(path, "edges"), _digest(path, "p"), _digest(path, "x"))
            if pair in old:
                duplicates[str(path)] = ["edges_and_prior"]
    return duplicates


def spec_list(panel: str) -> list[dict]:
    if panel == "test2":
        return [dict(kind="syn", fam=fam, n=100, pr=pr, prior=prior, rep=rep, panel="main")
                for fam in v1.FAMILIES for pr in v1.RANGES for prior in ("uniform", "dir1")
                for rep in (4, 5)]
    if panel == "scale2":
        return [dict(kind="syn", fam=fam, n=900, pr=pr, prior=prior, rep=1, panel="scale")
                for fam in v1.FAMILIES for pr in v1.RANGES for prior in ("uniform", "dir1")]
    if panel == "prior2":
        priors = ["uniform", "dir10", "dir3", "dir1", "dir0.3", "logn0.5", "logn1", "logn2",
                  "seed5_0.1", "seed1_0.1"]
        return [dict(kind="syn", fam=fam, n=n, pr=pr, prior=prior, rep=1, panel="prior")
                for fam, n in (("grid", 144), ("er", 150), ("ba", 150), ("ws", 150), ("sbm", 150))
                for pr in ("R2", "R3", "R4") for prior in priors]
    if panel == "big":
        return [dict(kind="syn", fam=fam, n=n, pr=pr, prior="uniform", rep=0, panel="big")
                for fam in ("er", "ba", "grid", "ws", "sbm")
                for n in (2000, 5000, 10000) for pr in ("R2", "R4")]
    if panel == "real2":
        schemes = ["unif0.05-0.2", "unif0.2-0.5", "unif0.5-0.9", "wc"]
        specs = []
        for g in ("karate", "lesmis", "davis"):
            for sc in schemes:
                for prior in ("uniform", "dir1"):
                    specs.append(dict(kind="classic", g=g, scheme=sc, prior=prior, rep=1))
        available = {p.stem for p in (ROOT / "data_real").glob("*.npz")}
        for g in ("cora", "citeseer", "amazon_electronics_photo", "ms_academic_cs"):
            if g not in available:
                continue
            for size, reps in ((500, (2, 3)), (1000, (4,))):
                for rep in reps:
                    for sc in schemes:
                        for prior in ("uniform", "dir1"):
                            specs.append(dict(kind="snow", g=g, size=size, scheme=sc,
                                              prior=prior, rep=rep))
        return specs
    raise ValueError(panel)


def make_instance(spec):
    return v1.make_instance(spec)


def build_one(args):
    return v1.build_one(args)


def main(panels):
    for panel in panels:
        specs = spec_list(panel)
        out = DATA / panel
        out.mkdir(parents=True, exist_ok=True)
        (out / "specs.json").write_text(json.dumps(specs, indent=0))
        jobs = [(panel, i, spec) for i, spec in enumerate(specs)]
        t0 = time.time()
        with Pool(2) as pool:
            for done, (_, seconds) in enumerate(pool.imap_unordered(v1.build_one, jobs, chunksize=1), 1):
                if done % 20 == 0:
                    print(f"[{panel}] {done}/{len(jobs)} elapsed={time.time() - t0:.0f}s",
                          flush=True)
        print(f"{panel}: {len(specs)} ground-truth instances ready in "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["test2", "scale2", "prior2", "real2", "big"])
