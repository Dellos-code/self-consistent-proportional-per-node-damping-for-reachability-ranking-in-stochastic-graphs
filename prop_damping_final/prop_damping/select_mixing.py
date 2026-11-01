"""Selection and freeze-file helpers for the mixing study."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "mixing"


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment, preserving named contrasts."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adj = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * float(p)))
        adj[name] = running
    return adj


def bootstrap_mean_difference(a: np.ndarray, b: np.ndarray, seed_parts,
                             resamples: int = 20000) -> tuple[float, float, float]:
    """Stratification is supplied as integer labels in ``seed_parts``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal shape")
    rng = np.random.default_rng(__import__("prop_damping_lib").stable_seed(*seed_parts))
    n = len(a)
    draws = rng.integers(0, n, size=(resamples, n))
    means = (a[draws] - b[draws]).mean(axis=1)
    return float((a - b).mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def select_top(result_dir: Path, limit: int = 50) -> pd.DataFrame:
    files = sorted(result_dir.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(result_dir)
    df = pd.concat((pd.read_csv(p) for p in files), ignore_index=True)
    return (df.groupby("config_id", as_index=False)["rho"].mean()
              .sort_values("rho", ascending=False).head(limit))


def load_configs(path: Path | None = None) -> dict[str, dict]:
    path = path or RES / "configs.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            out[row["config_id"]] = row["config"]
    return out


def write_stage_summary(stage: str, panel: str = "dev", limit: int = 50) -> Path:
    """Persist the ranked screen and the exact configuration count."""
    directory = RES / f"{stage}_{panel}"
    top = select_top(directory, limit)
    cfgs = load_configs()
    top.insert(1, "config", top.config_id.map(lambda cid: json.dumps(cfgs[cid], sort_keys=True)))
    out = RES / f"{stage}_{panel}_top{limit}.csv"
    top.to_csv(out, index=False)
    return out


def choose_winners(stage: str = "s2", panel: str = "dev") -> list[dict]:
    """Apply W1/W3's declared dev-only rules to a completed S1--S3 screen."""
    top = select_top(RES / f"{stage}_{panel}", 100)
    cfgs = load_configs()
    best = float(top.rho.iloc[0])
    candidates = top[top.rho >= best - 0.002].copy()
    candidates["free_parameters"] = candidates.config_id.map(
        lambda cid: sum(k in cfgs[cid] for k in ("damping", "steps", "beta", "gamma", "operator")))
    w1 = top.iloc[0]
    simple = candidates.sort_values(["free_parameters", "rho"], ascending=[True, False]).iloc[0]
    return [
        dict(name="W1", config_id=w1.config_id, dev_rho=float(w1.rho),
             config=cfgs[w1.config_id]),
        dict(name="W3", config_id=simple.config_id, dev_rho=float(simple.rho),
             config=cfgs[simple.config_id]),
    ]


def freeze(winners: list[dict], baselines: dict, path: Path | None = None) -> Path:
    path = path or RES / "selected_mixing.json"
    if path.exists():
        raise FileExistsError(f"freeze file already exists: {path}")
    payload = {"winners": winners, "baselines": baselines,
               "confirmatory_family": ["CTRL_lin0.9", "IRIE_exact", "rule_EW", "rule_WT",
                                       "MP2", "best_const_alpha_dev", "oracle:scalar"],
               "created_utc": datetime.now(timezone.utc).isoformat()}
    raw = json.dumps(payload, indent=2, sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (path.with_suffix(".sha256")).write_text(f"{digest}  {path.name}\n")
    return path
