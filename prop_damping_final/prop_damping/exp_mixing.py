"""Resumable staged runner for the mixing study."""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
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
    if stage == "baselines":
        return [{"name": name, "kind": "baseline", "baseline": name}
                for name in ("rule_EW", "rule_WT", "fixed085", "Pesc", "MP1", "MP2",
                             "MPinf", "IRIE_exact", "IRIE_wc", "oracle:scalar",
                             "oracle:scalar_WT", "heat_kernel:1", "heat_kernel:2",
                             "heat_kernel:5", "heat_kernel:10", "TotalRank",
                             "RAPr:0.5,0.5", "RAPr:1,1", "RAPr:2,1")]
    if stage in ("s1", "s2"):
        ops = ("P", "Plog", "Prow", "EP", "W", "EW")
        dams = tuple([0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.925,
                      0.95, 0.975, 0.99, 1.0] +
                     [f"spectral:{c}" for c in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)])
        return [{"name": "S1", "kind": "hybrid", "operator": op, "damping": d,
                 "steps": t, "beta": b, "gamma": g}
                for op in ops for d in dams for t in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50)
                for b in ((0.3, 0.4, 0.5, 0.6, 0.7) if stage == "s2" else (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0))
                for g in ((1.0, 4.0) if stage == "s2" else (0.25, 0.5, 1.0, 2.0, 4.0, 16.0))]
    if stage == "s3":
        from select_mixing import load_configs
        top_path = RES / "s2_dev_top50.csv"
        if not top_path.exists():
            raise FileNotFoundError("run s2 dev and write s2_dev_top50.csv first")
        top = pd.read_csv(top_path)
        known = load_configs()
        out = []
        betas = tuple(round(i / 20, 2) for i in range(21))
        for cid in top.config_id:
            base = known[cid]
            for beta in betas:
                for gamma in (0.25, 0.5, 1.0, 2.0, 4.0, 16.0):
                    c = dict(base, beta=beta, gamma=gamma)
                    out.append(c)
                for law in ("rule_EW", "rule_WT", "D2:0.25", "D2:1", "D2:4",
                            "D3:1", "D4:-0.25", "D4:0.25", "D4:1"):
                    out.append(dict(base, beta=beta, gamma=4.0, damping=law))
        return out
    if stage == "s4":
        return [
            {"name": "S1_current", "kind": "schedule",
             "blocks": [
                 {"operator": "EW", "damping": "rule_EW", "steps": "FP",
                  "normalisation": "L1", "name": "A"},
                 {"operator": "P", "damping": 0.9, "steps": 5,
                  "normalisation": "none", "name": "B"}],
             "snapshots": ["A", "B"], "fusion": "rank_mean", "beta": 0.5},
            {"name": "S3_alt", "kind": "schedule",
             "blocks": [
                 {"operator": "EW", "damping": "rule_EW", "steps": 1,
                  "normalisation": "L1", "name": "A1"},
                 {"operator": "P", "damping": 0.9, "steps": 1,
                  "normalisation": "none", "name": "B1"},
                 {"operator": "EW", "damping": "rule_EW", "steps": 1,
                  "normalisation": "L1", "name": "A2"},
                 {"operator": "P", "damping": 0.9, "steps": 1,
                  "normalisation": "none", "name": "B2"}],
             "snapshots": ["A2", "B2"], "fusion": "rank_mean", "beta": 0.5},
        ]
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


def evaluate(inst, yA, yB, cfg, cache=None):
    cache = {} if cache is None else cache
    ops = cache.setdefault("ops", _ops(inst))
    rA = L.ranks_of(yA)
    x = inst.x
    if cfg["kind"] == "irie":
        y = M.irie_exact(inst)
    elif cfg["kind"] == "baseline":
        y = M.baseline_scores(inst, yB).get(cfg["baseline"])
        if y is None:
            raise KeyError(cfg["baseline"])
    elif cfg["kind"] == "schedule":
        blocks = tuple(M.Block(**block) for block in cfg["blocks"])
        snapshots = M.run_schedule(ops, x, M.Schedule(blocks, tuple(cfg["snapshots"])))
        y = M.fuse(snapshots, cfg.get("fusion", "rank_mean"),
                   beta=cfg.get("beta"), k=cfg.get("k", 60))
    elif cfg["kind"] == "A":
        y, _ = M.phase_A(ops[cfg["operator"]], x, ops, cfg["operator"], cfg["damping"], cfg["steps"])
    else:
        a0 = cache.get("a0")
        if a0 is None:
            a0, _ = M.phase_A(ops["EW"], x, ops, "EW", "rule_EW", "FP")
            cache["a0"] = a0
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


def run(stage: str, panel: str, shard: tuple[int, int] | None = None):
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
    if shard is not None:
        index, total = shard
        paths = paths[index::total]
    for path in paths:
        dest = outdir / f"{path.stem}.csv.gz"
        if dest.exists():
            continue
        inst, yA, yB, _ = load_instance(path)
        cache = {}
        rows = []
        for cfg in cfgs:
            t0 = time.perf_counter()
            rows.append({"id": int(path.stem), "config_id": M.config_id(cfg),
                         "rho": evaluate(inst, yA, yB, cfg, cache),
                         "ms_est": 1000 * (time.perf_counter() - t0)})
        pd.DataFrame(rows).to_csv(dest, index=False, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("panel")
    parser.add_argument("--shard", type=str)
    parser.add_argument("--procs", type=int, default=1,
                        help="reserved for scheduler compatibility; each output is process-safe")
    args = parser.parse_args()
    shard = None
    if args.shard:
        i, n = args.shard.split("/", 1)
        shard = (int(i), int(n))
    run(args.stage, args.panel, shard)
