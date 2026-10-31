"""Resumable staged runner for the mixing study."""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import prop_damping_lib as L
import mixing_lib as M
from build_panels import DATA, load_instance

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "mixing"


def configs(stage: str) -> list[dict]:
    if stage == "s0":
        return [
            {"name": "rule_EW", "kind": "A", "operator": "EW", "damping": "rule_EW", "steps": "FP"},
            {"name": "rule_WT", "kind": "A", "operator": "WT", "damping": "rule_WT", "steps": "FP"},
            {"name": "CTRL_lin0.9", "kind": "hybrid", "a": 0.9, "steps": 5, "beta": 0.5},
            {"name": "IRIE_exact", "kind": "irie"},
        ]
    if stage in ("s1", "s2", "s3"):
        ops = ("P", "Plog", "Prow", "EP", "W", "EW")
        dams = tuple([0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.925,
                      0.95, 0.975, 0.99, 1.0] +
                     [f"spectral:{c}" for c in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)])
        return [{"name": "S1", "kind": "hybrid", "operator": op, "damping": d,
                 "steps": t, "beta": b, "gamma": g}
                for op in ops for d in dams for t in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50)
                for b in ((0.3, 0.4, 0.5, 0.6, 0.7) if stage == "s2" else (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0))
                for g in ((1.0, 4.0) if stage == "s2" else (0.25, 0.5, 1.0, 2.0, 4.0, 16.0))]
    raise ValueError(stage)


def _ops(inst):
    ops = dict(inst.ops())
    P = ops["P"].tocsr()
    rows = np.repeat(np.arange(inst.n), np.diff(P.indptr))
    plog = P.copy()
    plog.data = -np.log1p(-np.minimum(plog.data, 1.0 - 1e-12))
    ops["Plog"] = plog
    ops["Prow"] = (sp.diags(np.divide(1.0, np.asarray(P.sum(1)).ravel(),
                                      out=np.zeros(inst.n), where=np.asarray(P.sum(1)).ravel() > 0)) @ P).tocsr()
    return ops


def evaluate(inst, yA, yB, cfg):
    ops = _ops(inst)
    rA = L.ranks_of(yA)
    x = inst.x
    if cfg["kind"] == "irie":
        y = M.irie_exact(inst)
    elif cfg["kind"] == "A":
        y, _ = M.phase_A(ops[cfg["operator"]], x, ops, cfg["operator"], cfg["damping"], cfg["steps"])
    else:
        a0, _ = M.phase_A(ops["EW"], x, ops, "EW", "rule_EW", "FP")
        z = np.maximum(a0, 1e-300)
        z = z / np.mean(z / x)
        start = x + cfg.get("gamma", 1.0) * z
        operator = cfg.get("operator", "P")
        damping = cfg.get("damping", cfg.get("a", 0.9))
        traj, _ = M.phase_B_trajectory(ops[operator], x, ops, damping, start,
                                       (cfg["steps"],), "none")
        b = traj[cfg["steps"]]
        y = M.fuse([a0, b], "rank_mean", beta=cfg.get("beta", 0.5))
    return float(L.spearman(y, rA))


def run(stage: str, panel: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    outdir = RES / f"{stage}_{panel}"
    outdir.mkdir(parents=True, exist_ok=True)
    cfgs = configs(stage)
    cfgpath = RES / "configs.jsonl"
    known = {json.loads(line)["config_id"] for line in cfgpath.open()} if cfgpath.exists() else set()
    with cfgpath.open("a") as f:
        for cfg in cfgs:
            cid = M.config_id(cfg)
            if cid not in known:
                f.write(json.dumps({"config_id": cid, "config": cfg}, sort_keys=True) + "\n")
    paths = sorted(DATA.joinpath(panel).glob("[0-9][0-9][0-9][0-9].npz"))
    for path in paths:
        dest = outdir / f"{path.stem}.csv.gz"
        if dest.exists():
            continue
        inst, yA, yB, _ = load_instance(path)
        rows = []
        for cfg in cfgs:
            t0 = time.perf_counter()
            rows.append({"id": int(path.stem), "config_id": M.config_id(cfg),
                         "rho": evaluate(inst, yA, yB, cfg),
                         "ms_est": 1000 * (time.perf_counter() - t0)})
        pd.DataFrame(rows).to_csv(dest, index=False, compression="gzip")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python exp_mixing.py <stage> <panel>")
    run(sys.argv[1], sys.argv[2])
