"""Reusable engine for the preregistered dual-operator mixing study.

The module is deliberately additive: it imports the published library but
does not alter its behaviour.  Configurations are JSON serialisable and are
identified by the SHA-1 digest of their canonical representation.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import prop_damping_lib as L


@dataclass(frozen=True)
class Block:
    operator: str
    damping: str | float
    steps: int | str
    normalisation: str = "none"
    start: str = "previous"
    name: str = "snapshot"
    gamma: float = 1.0
    eta: float = 1.0


@dataclass(frozen=True)
class Schedule:
    blocks: tuple[Block, ...]
    snapshots: tuple[str, ...] = ()


@dataclass
class ScaledVector:
    """A non-negative vector represented as ``exp(log_scale) * unit``."""

    unit: np.ndarray
    log_scale: float = 0.0

    def ranks(self) -> np.ndarray:
        return L.ranks_of(self.unit)

    def finite(self) -> np.ndarray:
        if self.log_scale > math.log(np.finfo(float).max):
            return np.full_like(self.unit, np.inf)
        return np.exp(self.log_scale) * self.unit


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (Block, Schedule)):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (tuple, list)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def canonical_json(cfg: Any) -> str:
    return json.dumps(_jsonable(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_id(cfg: Any) -> str:
    return hashlib.sha1(canonical_json(cfg).encode("utf-8")).hexdigest()


def _normalise(y: np.ndarray) -> np.ndarray:
    s = float(np.sum(y))
    if not np.isfinite(s) or s <= 0:
        raise FloatingPointError("non-positive or non-finite propagation mass")
    return y / s


def _alpha(x: np.ndarray, ops: dict, law: str | float, y: np.ndarray | None = None,
           T: sp.csr_matrix | None = None, eta: float = 1.0) -> np.ndarray | float:
    if isinstance(law, (float, int)):
        return float(law)
    if law == "rule_EW":
        return L.rule_alpha(ops["Pbar"], "EW")
    if law == "rule_WT":
        return L.rule_alpha(ops["Pbar"], "WT")
    if law.startswith("constant:"):
        return float(law.split(":", 1)[1])
    if law.startswith("spectral:"):
        return float(law.split(":", 1)[1]) / spectral_radius(T, 0)
    if y is None or T is None:
        raise ValueError(f"{law} requires the current vector and operator")
    q = np.divide(y, np.maximum(x, 1e-300))
    if law.startswith("D2:"):
        gamma = float(law.split(":", 1)[1])
        S = T @ y
        return np.clip(gamma * S / (x + gamma * S), 0.0, 1.0 - 1e-12)
    if law.startswith("D3:"):
        gamma = float(law.split(":", 1)[1])
        S = T @ y
        a = gamma * S / (x + gamma * S)
        return float(np.mean(a))
    if law.startswith("D4:"):
        return L.anchored_alpha(q, L.rule_alpha(ops["Pbar"], "EW"), eta)
    raise ValueError(f"unknown damping law: {law}")


def phase_A(T: sp.csr_matrix, x: np.ndarray, ops: dict, operator: str = "EW",
            damping: str | float = "rule_EW", steps: int | str = "FP",
            y0: np.ndarray | None = None, eta: float = 1.0) -> tuple[np.ndarray, int]:
    """Run lazy normalised Phase A, retaining the exact published FP semantics."""
    y = _normalise(x if y0 is None else np.asarray(y0, float))
    if steps == "FP":
        alpha = _alpha(x, ops, damping)
        return L.diffuse(T, x, alpha, y0=y)
    for _ in range(int(steps)):
        a = _alpha(x, ops, damping, y, T, eta)
        z = (1.0 - a) * x + np.asarray(a) * (T @ y)
        y = _normalise(0.5 * (y + z))
    return y, int(steps)


def _raw_step_scaled(state: ScaledVector, x: np.ndarray, T: sp.csr_matrix,
                     alpha: np.ndarray | float) -> ScaledVector:
    """One raw affine step without ever forming an overflowing intermediate."""
    a = np.broadcast_to(np.asarray(alpha, float), x.shape)
    b = np.maximum((1.0 - a) * x, 0.0)
    v = np.maximum(a * (T @ state.unit), 0.0)
    ls = float(state.log_scale)
    if not np.isfinite(ls):
        raise FloatingPointError("invalid log scale")
    if ls > 0:
        b_scaled = b * math.exp(-ls) if ls < 700 else np.zeros_like(b)
        q = b_scaled + v
        scale = ls + math.log(max(float(np.max(q)), np.finfo(float).tiny))
        unit = q / max(float(np.max(q)), np.finfo(float).tiny)
    else:
        scaled = math.exp(ls) * v
        q = b + scaled
        scale = math.log(max(float(np.max(q)), np.finfo(float).tiny))
        unit = q / max(float(np.max(q)), np.finfo(float).tiny)
    return ScaledVector(unit, scale)


def phase_B_trajectory(T: sp.csr_matrix, x: np.ndarray, ops: dict,
                       damping: str | float, start: np.ndarray,
                       steps: tuple[int | str, ...], normalisation: str = "none",
                       eta: float = 1.0) -> tuple[dict[int | str, np.ndarray], dict]:
    """Compute all requested prefixes once.

    Raw recursion uses :class:`ScaledVector`; returned arrays are ordinary
    values when representable and rank-equivalent unit vectors otherwise.
    """
    requested = tuple(steps)
    finite_steps = [int(k) for k in requested if k != "FP"]
    max_steps = max(finite_steps, default=0)
    raw_start = np.asarray(start, float)
    if normalisation == "L1":
        y = _normalise(raw_start)
        state = ScaledVector(y.copy(), 0.0)
    elif normalisation == "none":
        scale0 = float(np.max(np.abs(raw_start)))
        if not np.isfinite(scale0) or scale0 <= 0:
            raise FloatingPointError("invalid propagation start")
        y = raw_start / scale0
        state = ScaledVector(y.copy(), math.log(scale0))
    else:
        raise ValueError(normalisation)
    out: dict[int | str, np.ndarray] = {}
    contractive = False
    rho = spectral_radius(T, 0)
    if isinstance(damping, (float, int)):
        contractive = float(damping) * rho < 1.0
    for k in range(1, max_steps + 1):
        query_y = state.finite()
        if not np.isfinite(query_y).all():
            query_y = state.unit
        a = _alpha(x, ops, damping, query_y, T, eta)
        if normalisation == "L1":
            y = _normalise((1.0 - a) * x + np.asarray(a) * (T @ y))
            state = ScaledVector(y.copy(), 0.0)
        elif normalisation == "none":
            state = _raw_step_scaled(state, x, T, a)
            y = state.unit
        else:
            raise ValueError(normalisation)
        if k in requested:
            out[k] = state.finite() if np.isfinite(state.log_scale) and state.log_scale < 700 else state.unit.copy()
    if "FP" in requested and contractive:
        if normalisation == "L1":
            z, it = phase_A(T, x, ops, damping=damping, steps="FP", y0=y, eta=eta)
        else:
            z = state.unit
            it = max_steps
            for _ in range(20000):
                nxt = _raw_step_scaled(state, x, T, _alpha(x, ops, damping, z, T, eta))
                if np.max(np.abs(nxt.unit - state.unit)) < 1e-12:
                    z, state = nxt.unit, nxt
                    break
                state, z = nxt, nxt.unit
                it += 1
        out["FP"] = z.copy()
    return out, {"rho": float(rho), "contractive": contractive, "max_steps": max_steps}


def run_schedule(ops: dict, x: np.ndarray, schedule: Schedule) -> dict[str, np.ndarray]:
    """Run a block schedule and return named snapshots."""
    current = x.copy()
    snapshots: dict[str, np.ndarray] = {}
    for block in schedule.blocks:
        T = ops[block.operator]
        start = current if block.start == "previous" else x
        if block.start.startswith("embed:"):
            current = x + float(block.start.split(":", 1)[1]) * np.maximum(start, 0)
        if block.operator in ("EW", "WT", "W") and block.normalisation == "L1":
            current, _ = phase_A(T, x, ops, block.operator, block.damping, block.steps,
                                 y0=start, eta=block.eta)
        else:
            traj, _ = phase_B_trajectory(T, x, ops, block.damping, start,
                                          (block.steps,), block.normalisation, block.eta)
            current = traj[block.steps]
        snapshots[block.name] = current.copy()
    names = schedule.snapshots or tuple(snapshots)
    return {name: snapshots[name] for name in names}


def fuse(members: dict[str, np.ndarray] | list[np.ndarray], method: str = "rank_mean",
         weights: np.ndarray | None = None, beta: float | None = None,
         k: int = 60) -> np.ndarray:
    vals = list(members.values()) if isinstance(members, dict) else list(members)
    if not vals:
        raise ValueError("at least one member is required")
    R = np.column_stack([L.ranks_of(v) for v in vals])
    if beta is not None and len(vals) == 2:
        weights = np.array([1.0 - beta, beta])
    if weights is None:
        weights = np.full(len(vals), 1.0 / len(vals))
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    if method in ("rank_mean", "borda"):
        return R @ weights
    if method == "rrf":
        top = len(vals[0]) + 1 - R
        return (1.0 / (k + top)) @ weights
    if method == "geometric":
        S = np.column_stack([np.maximum(v, 1e-300) / np.sum(np.maximum(v, 1e-300)) for v in vals])
        return np.exp(np.log(S) @ weights)
    if method == "l1":
        S = np.column_stack([v / (np.sum(v) or 1.0) for v in vals])
        return S @ weights
    if method == "minmax":
        S = np.column_stack([(v - np.min(v)) / (np.ptp(v) or 1.0) for v in vals])
        return S @ weights
    if method == "zlog":
        S = np.column_stack([np.log(np.maximum(v, 1e-300)) for v in vals])
        S = (S - S.mean(0)) / np.where(S.std(0) > 0, S.std(0), 1.0)
        return S @ weights
    raise ValueError(method)


@lru_cache(maxsize=2048)
def _spectral_radius_cached(key: bytes, shape: tuple[int, int], data: tuple, indices: tuple,
                            indptr: tuple, seed: int) -> float:
    T = sp.csr_matrix((np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=shape)
    if T.shape[0] == 1:
        return float(abs(T[0, 0]))
    try:
        return float(abs(eigsh(T, k=1, which="LM", tol=1e-10, maxiter=5000, v0=np.ones(T.shape[0]))[0][0]))
    except Exception:
        v = np.random.default_rng(seed).random(T.shape[0])
        v /= np.linalg.norm(v)
        r = 0.0
        for _ in range(250):
            w = T @ v
            nw = np.linalg.norm(w)
            if nw == 0:
                return 0.0
            v, r = w / nw, nw
        return float(r)


def spectral_radius(T: sp.spmatrix, seed: int) -> float:
    A = T.tocsr()
    return _spectral_radius_cached(
        hashlib.sha1(A.data.tobytes() + A.indices.tobytes() + A.indptr.tobytes()).digest(),
        A.shape, tuple(A.data), tuple(A.indices), tuple(A.indptr), int(seed))


def irie_exact(inst) -> np.ndarray:
    T = inst.ops()["P"]
    rng = np.random.default_rng(L.stable_seed(inst.name, "irie", "rho"))
    v = rng.random(inst.n)
    v /= np.linalg.norm(v)
    rho = 0.0
    for _ in range(60):
        w = T @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            rho = 0.0
            break
        v, rho = w / nw, float(nw)
    a = min(0.7, 0.95 / rho) if rho > 0 else 0.0
    y = inst.x.copy()
    for _ in range(400):
        yn = inst.x + a * (inst.ops()["P"] @ y)
        if np.max(np.abs(yn - y)) < 1e-12 * max(1.0, np.max(np.abs(y))):
            y = yn
            break
        y = yn
    return y


def irie_wc(inst) -> np.ndarray:
    T = inst.ops()["W"]
    y = inst.x.copy()
    for _ in range(400):
        yn = inst.x + 0.7 * (T @ y)
        if np.max(np.abs(yn - y)) < 1e-12 * max(1.0, np.max(np.abs(y))):
            y = yn
            break
        y = yn
    return y


def baseline_scores(inst, yB: np.ndarray | None = None, *,
                    include_mc: bool = False, mc_k: tuple[int, ...] = (10, 20, 50, 100,
                    200, 500, 1000)) -> dict[str, np.ndarray]:
    """Return the common baseline family; oracle fits use ``yB`` only."""
    x, ops = inst.x, inst.ops()
    out: dict[str, np.ndarray] = {
        "rule_EW": L.diffuse(ops["EW"], x, L.rule_alpha(ops["Pbar"], "EW"))[0],
        "rule_WT": L.diffuse(ops["WT"], x, L.rule_alpha(ops["Pbar"], "WT"))[0],
        "fixed085": L.diffuse(ops["EW"], x, 0.85)[0],
        "Pesc": ops["Pesc"].copy(),
        "MP1": L.knz_mp(inst, iters=1)[0],
        "MP2": L.knz_mp(inst, iters=2)[0],
        "MPinf": L.knz_mp(inst, iters=None)[0],
        "IRIE_exact": irie_exact(inst),
        "IRIE_wc": irie_wc(inst),
    }
    if yB is not None:
        ys, _ = L.scalar_sweep(ops["EW"], x)
        idx = int(np.argmax(L.corr_ranks(L.ranks_of(ys), L.ranks_of(yB))))
        out["oracle:scalar"] = ys[:, idx]
        yw, _ = L.scalar_sweep(ops["WT"], x)
        idx = int(np.argmax(L.corr_ranks(L.ranks_of(yw), L.ranks_of(yB))))
        out["oracle:scalar_WT"] = yw[:, idx]
    for t in (1, 2, 5, 10):
        out[f"heat_kernel:{t}"] = heat_kernel(ops["W"], x, float(t))
    out["TotalRank"] = total_rank(ops["EW"], x)
    for a, b in ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0)):
        out[f"RAPr:{a:g},{b:g}"] = rapr(ops["EW"], x, a, b)
    if include_mc:
        for k in mc_k:
            out[f"MC_{k}"] = L.mc_reach(inst, k, L.stable_seed(inst.name, "MC", k))
    return out


def heat_kernel(T: sp.csr_matrix, x: np.ndarray, t: float, tol: float = 1e-12) -> np.ndarray:
    y = x.copy()
    term = x.copy()
    for k in range(1, 10000):
        term = (t / k) * (T @ term)
        y += term
        if np.max(np.abs(term)) <= tol * max(1.0, np.max(np.abs(y))):
            break
    return y


def linear_rank(T: sp.csr_matrix, x: np.ndarray, alpha: float, steps: int) -> np.ndarray:
    y = x.copy()
    out = np.zeros_like(y)
    for k in range(1, steps + 1):
        y = (1.0 - alpha) * x + alpha * (T @ y)
        out += y
    return out


def operator_mixture(EW: sp.csr_matrix, P: sp.csr_matrix, x: np.ndarray,
                     alpha: float, lam: float, c: float = 0.9,
                     steps: int | str = "FP") -> tuple[np.ndarray, int]:
    """Lazy normalised diffusion on ``(1-lambda) EW + lambda c P/rho(P)``."""
    rho = spectral_radius(P, 0)
    mixed = ((1.0 - lam) * EW + lam * c * P / max(rho, 1e-300)).tocsr()
    ops = {"Pbar": 0.0}
    return phase_A(mixed, x, ops, "EW", alpha, steps)


def total_rank(T: sp.csr_matrix, x: np.ndarray, nodes: int = 16) -> np.ndarray:
    q, w = np.polynomial.legendre.leggauss(nodes)
    out = np.zeros_like(x)
    for z, wt in zip(q, w):
        a = (z + 1.0) / 2.0
        y, _ = L.diffuse(T, x, a, tol=1e-10, maxit=5000)
        out += 0.5 * wt * y
    return out


def rapr(T: sp.csr_matrix, x: np.ndarray, a: float, b: float, nodes: int = 16) -> np.ndarray:
    q, w = np.polynomial.legendre.leggauss(nodes)
    out = np.zeros_like(x)
    for z, wt in zip(q, w):
        alpha = (z + 1.0) / 2.0
        density = alpha ** (a - 1.0) * (1 - alpha) ** (b - 1.0)
        y, _ = L.diffuse(T, x, alpha, tol=1e-10, maxit=5000)
        out += 0.5 * wt * density * y
    return out
