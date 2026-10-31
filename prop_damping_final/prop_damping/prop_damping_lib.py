"""
prop_damping_lib.py
===================

Self-consistent proportional per-node damping for reachability ranking in
uncertain graphs: a vectorised experimental library.

Problem.  G=(V,E,p) undirected, each edge present independently with
probability p_ij; x > 0 a prior with sum(x) = 1; C(i) the component of i in a
sampled world.  Target:  y*_i = E[ sum_{v in C(i)} x_v ].

Contents
--------
* Instance generation (synthetic families, real networks, priors).
* Monte-Carlo ground truth (block-diagonal batching, one connected-components
  call per batch) and exact ground truth by world enumeration (tiny graphs).
* Operators: W (row-stochastic, w_ij ~ -ln(1-p_ij)), EW = diag(P_esc) W
  (escape-scaled pull), WT = W^T (push PPR), P (raw probability adjacency),
  EP = diag(P_esc) P.
* Damped diffusion with scalar or per-node damping (normalised power method,
  lazy form so bipartite graphs converge), batched over many alpha vectors.
* Karrer-Newman-Zdeborova message passing (membership form, log space).
* Realizations A (simultaneous root), B (time-lagged), C (direct
  neighbour-signal) of the proportional principle, under several scale
  conventions; anchored power-law family AP(eta); Mode (b) polishing.
* Oracles: scalar grid oracle, per-node coordinate-ascent oracle (fit on one
  ground-truth sample, evaluated on an independent one).
* Metrics: Spearman, Kendall tau-b, top-k overlap, NDCG@k.

All randomness is derived from explicit seeds (`stable_seed`).
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.special import expit, logit
from scipy.stats import kendalltau, rankdata

P_CLIP = 1.0 - 1e-12          # p used inside logarithms is clipped below 1
RANK_REL = 1e-12              # scores equal to within this relative tolerance are ties

# Calibrated scalar rules from [4]
RULE_EW = (0.502, 0.558)      # escape-scaled pull
RULE_WT = (0.241, 0.859)      # push personalised PageRank
RULE_CLIP = (0.4, 1.0)


def stable_seed(*parts) -> int:
    """Deterministic 63-bit seed from arbitrary parts."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


# =============================================================================
# Instances
# =============================================================================
@dataclass
class Instance:
    name: str
    n: int
    edges: np.ndarray            # (m, 2) int64, i < j, unique
    p: np.ndarray                # (m,) edge probabilities in (0, 1]
    x: np.ndarray                # (n,) prior, > 0, sums to 1
    meta: dict = field(default_factory=dict)
    _ops: dict | None = field(default=None, repr=False)

    @property
    def m(self) -> int:
        return int(len(self.p))

    def ops(self) -> dict:
        if self._ops is None:
            self._ops = build_operators(self)
        return self._ops

    def with_prior(self, x: np.ndarray, name: str | None = None) -> "Instance":
        return Instance(name or self.name, self.n, self.edges, self.p,
                        np.asarray(x, float), dict(self.meta))


def canon_edges(E) -> np.ndarray:
    E = np.asarray(E, dtype=np.int64).reshape(-1, 2)
    E = E[E[:, 0] != E[:, 1]]
    E = np.sort(E, axis=1)
    if len(E):
        E = np.unique(E, axis=0)
    return E


def _nx_edges(G) -> np.ndarray:
    import networkx as nx
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return canon_edges(np.array(list(G.edges()), dtype=np.int64).reshape(-1, 2)), G.number_of_nodes()


def topology(kind: str, n: int, seed: int, **kw):
    """Return (edges, n_actual, info) for a synthetic topology family."""
    import networkx as nx
    s = int(seed % (2 ** 31 - 1))
    rng = np.random.default_rng(seed)
    info = {}
    if kind == "er":
        d = kw.get("deg", 5.0)
        G = nx.gnm_random_graph(n, int(round(n * d / 2)), seed=s)
    elif kind == "ba":
        G = nx.barabasi_albert_graph(n, kw.get("m", 2), seed=s)
    elif kind == "ws":
        G = nx.connected_watts_strogatz_graph(n, kw.get("k", 6), kw.get("beta", 0.1), seed=s)
    elif kind == "grid":
        L = int(round(math.sqrt(n)))
        G = nx.grid_2d_graph(L, L)
        info["shape"] = (L, L)
    elif kind == "rect":
        L1 = int(round(math.sqrt(n / 4.0)))
        L2 = int(round(n / L1))
        G = nx.grid_2d_graph(L1, L2)
        info["shape"] = (L1, L2)
    elif kind == "sbm":
        B = kw.get("blocks", 4)
        sizes = [n // B] * B
        sizes[-1] += n - sum(sizes)
        din, dout = kw.get("din", 5.0), kw.get("dout", 1.0)
        bs = sizes[0]
        pin, pout = din / (bs - 1), dout / (n - bs)
        P = [[pin if a == b else pout for b in range(B)] for a in range(B)]
        G = nx.stochastic_block_model(sizes, P, seed=s)
    elif kind == "cp":                       # core-periphery
        nc = max(4, int(round(0.15 * n)))
        E = [(i, j) for i in range(nc) for j in range(i + 1, nc) if rng.random() < 0.5]
        for v in range(nc, n):
            k = 1 + int(rng.random() < 0.5)
            for c in rng.choice(nc, size=k, replace=False):
                E.append((v, int(c)))
        npp = int(round(0.3 * (n - nc)))
        a = rng.integers(nc, n, size=npp)
        b = rng.integers(nc, n, size=npp)
        E += list(zip(a.tolist(), b.tolist()))
        info["core"] = nc
        return canon_edges(E), n, info
    elif kind == "tree":
        G = nx.random_labeled_tree(n, seed=s)
    elif kind == "barbell":
        m1 = kw.get("m1", int(round(0.3 * n)))
        m2 = n - 2 * m1
        G = nx.barbell_graph(m1, m2)
        info["m1"], info["m2"] = m1, m2
    else:
        raise ValueError(kind)
    E, nn = _nx_edges(G)
    return E, nn, info


PROB_RANGES = {
    "R1": (0.01, 0.06),
    "R2": (0.06, 0.18),
    "R3": (0.18, 0.40),
    "R4": (0.40, 0.70),
    "R5": (0.70, 0.95),
}


def make_prior(kind: str, n: int, seed: int, floor_rel: float = 1e-12) -> np.ndarray:
    """Priors: uniform; dirT (Dirichlet theta=T); lognS (log-normal sigma=S);
    seedK_E (K seeds sharing 1-E, E spread uniformly)."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        x = np.ones(n)
    elif kind.startswith("dir"):
        th = float(kind[3:])
        # Dirichlet via Gamma draws in log space to avoid underflow at small theta
        g = rng.standard_gamma(th, size=n)
        x = g
    elif kind.startswith("logn"):
        sg = float(kind[4:])
        x = np.exp(sg * rng.standard_normal(n))
    elif kind.startswith("seed"):
        k_str, e_str = kind[4:].split("_")
        k, eps = int(k_str), float(e_str)
        x = np.full(n, eps / n)
        seeds = rng.choice(n, size=k, replace=False)
        x[seeds] += (1.0 - eps) / k
    else:
        raise ValueError(kind)
    x = np.asarray(x, float)
    x = np.maximum(x, floor_rel * x.max())
    return x / x.sum()


def prior_stats(x: np.ndarray) -> dict:
    n = len(x)
    H = -np.sum(x * np.log(x))
    return dict(kappa=float(x.max() / x.min()), Hnorm=float(H / math.log(n)),
                log10_kappa=float(np.log10(x.max() / x.min())))


def assign_probs(m: int, rng_seed: int, lo: float, hi: float) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(lo, hi, size=m)


def make_synthetic(kind: str, n: int, prange: str, prior: str, rep: int,
                   panel: str = "", **kw) -> Instance:
    name = f"{panel}:{kind}:n{n}:{prange}:{prior}:r{rep}"
    sd = stable_seed(panel, kind, n, prange, rep)          # topology/probs do not depend on prior
    E, nn, info = topology(kind, n, sd, **kw)
    lo, hi = PROB_RANGES[prange]
    p = assign_probs(len(E), stable_seed(sd, "p"), lo, hi)
    x = make_prior(prior, nn, stable_seed(sd, "x", prior))
    meta = dict(panel=panel, family=kind, n=nn, prange=prange, prior=prior, rep=rep, **info)
    return Instance(name, nn, E, p, x, meta)


# --------------------------- real networks ----------------------------------
def load_npz_graph(path: str):
    d = np.load(path, allow_pickle=False)
    A = sp.csr_matrix((d["adj_data"], d["adj_indices"], d["adj_indptr"]), shape=tuple(d["adj_shape"]))
    A = ((A + A.T) > 0).astype(np.int8).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def snowball(A: sp.csr_matrix, size: int, seed: int) -> np.ndarray:
    """BFS snowball sample from a random start inside the largest component."""
    rng = np.random.default_rng(seed)
    nc, lab = connected_components(A, directed=False)
    big = np.bincount(lab).argmax()
    cand = np.flatnonzero(lab == big)
    start = int(rng.choice(cand))
    seen = {start}
    order = [start]
    frontier = [start]
    indptr, ind = A.indptr, A.indices
    while frontier and len(order) < size:
        nxt = []
        for u in frontier:
            nb = ind[indptr[u]:indptr[u + 1]]
            nb = nb[rng.permutation(len(nb))]
            for v in nb:
                v = int(v)
                if v not in seen:
                    seen.add(v)
                    order.append(v)
                    nxt.append(v)
                    if len(order) >= size:
                        break
            if len(order) >= size:
                break
        frontier = nxt
    return np.array(order[:size])


def subgraph_edges(A: sp.csr_matrix, nodes: np.ndarray) -> np.ndarray:
    S = A[nodes][:, nodes].tocoo()
    return canon_edges(np.c_[S.row, S.col])


def real_probs(E: np.ndarray, n: int, scheme: str, seed: int) -> np.ndarray:
    m = len(E)
    rng = np.random.default_rng(seed)
    if scheme.startswith("const"):
        return np.full(m, float(scheme[5:]))
    if scheme.startswith("unif"):
        lo, hi = map(float, scheme[4:].split("-"))
        return rng.uniform(lo, hi, size=m)
    if scheme == "wc":                       # weighted cascade, symmetrised
        deg = np.bincount(E.ravel(), minlength=n)
        return 1.0 / np.maximum(deg[E[:, 0]], deg[E[:, 1]])
    if scheme.startswith("wcx"):              # scaled weighted cascade, capped
        c = float(scheme[3:])
        deg = np.bincount(E.ravel(), minlength=n)
        return np.minimum(0.95, c / np.maximum(deg[E[:, 0]], deg[E[:, 1]]))
    raise ValueError(scheme)


# =============================================================================
# Operators
# =============================================================================
def escape_probability(n: int, E: np.ndarray, p: np.ndarray) -> np.ndarray:
    pc = np.minimum(p, P_CLIP)
    l1m = np.log1p(-pc)
    logq = np.bincount(E[:, 0], weights=l1m, minlength=n) + np.bincount(E[:, 1], weights=l1m, minlength=n)
    return -np.expm1(logq)


def build_operators(inst: Instance) -> dict:
    n, E, p = inst.n, inst.edges, inst.p
    i, j = E[:, 0], E[:, 1]
    pc = np.minimum(p, P_CLIP)
    lw = -np.log1p(-pc)
    rows, cols = np.r_[i, j], np.r_[j, i]
    Alw = sp.csr_matrix((np.r_[lw, lw], (rows, cols)), shape=(n, n))
    P = sp.csr_matrix((np.r_[p, p], (rows, cols)), shape=(n, n))
    dw = np.asarray(Alw.sum(1)).ravel()
    inv = np.divide(1.0, dw, out=np.zeros(n), where=dw > 0)
    W = (sp.diags(inv) @ Alw).tocsr()
    Pesc = escape_probability(n, E, p)
    ops = dict(
        Pesc=Pesc,
        Pbar=float(Pesc.mean()),
        deg=np.bincount(E.ravel(), minlength=n),
        deff=np.asarray(P.sum(1)).ravel(),
        W=W,
        EW=(sp.diags(Pesc) @ W).tocsr(),
        WT=W.T.tocsr(),
        P=P,
        EP=(sp.diags(Pesc) @ P).tocsr(),
    )
    return ops


def rule_alpha(Pbar: float, family: str = "EW") -> float:
    c0, c1 = RULE_EW if family == "EW" else RULE_WT
    return float(np.clip(c0 + c1 * Pbar, *RULE_CLIP))


# =============================================================================
# Ground truth
# =============================================================================
def mc_reach(inst: Instance, M: int, seed: int, max_nodes: int = 400_000,
             max_edges: int = 4_000_000, return_second_moment: bool = False):
    """Monte-Carlo estimate of y*_i = E[sum_{v in C(i)} x_v] over M sampled worlds.

    Worlds are processed in batches stacked as a block-diagonal graph so a
    single connected_components call labels all of them."""
    rng = np.random.default_rng(seed)
    n, m, x = inst.n, inst.m, inst.x
    if m == 0:
        return (x.copy(), np.zeros(n)) if return_second_moment else x.copy()
    B = int(max(1, min(M, max_nodes // n, max_edges // m)))
    tot = np.zeros(n)
    tot2 = np.zeros(n)
    i0, j0 = inst.edges[:, 0], inst.edges[:, 1]
    xt = np.tile(x, B)
    done = 0
    while done < M:
        b = min(B, M - done)
        keep = rng.random((b, m)) < inst.p
        bi, ei = np.nonzero(keep)
        off = bi * n
        r, c = i0[ei] + off, j0[ei] + off
        N = b * n
        A = sp.csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(N, N))
        nc, lab = connected_components(A, directed=False)
        mass = np.bincount(lab, weights=xt[:N], minlength=nc)
        v = mass[lab].reshape(b, n)
        tot += v.sum(0)
        if return_second_moment:
            tot2 += (v * v).sum(0)
        done += b
    if return_second_moment:
        return tot / M, tot2 / M
    return tot / M


def exact_reach(inst: Instance) -> np.ndarray:
    """Exact y* by enumerating all 2^m worlds (m <= 20)."""
    n, m = inst.n, inst.m
    assert m <= 20
    W = 1 << m
    bits = ((np.arange(W)[:, None] >> np.arange(m)[None, :]) & 1).astype(bool)
    pr = np.prod(np.where(bits, inst.p[None, :], 1 - inst.p[None, :]), axis=1)
    y = np.zeros(n)
    i0, j0 = inst.edges[:, 0], inst.edges[:, 1]
    for w in range(W):
        e = bits[w]
        A = sp.csr_matrix((np.ones(e.sum()), (i0[e], j0[e])), shape=(n, n))
        _, lab = connected_components(A, directed=False)
        mass = np.bincount(lab, weights=inst.x)
        y += pr[w] * mass[lab]
    return y


# =============================================================================
# Metrics
# =============================================================================
def ranks_of(y: np.ndarray, rel: float = RANK_REL) -> np.ndarray:
    """Average ranks after merging scores equal to within a relative tolerance."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        s = np.nanmax(np.abs(y))
        if not np.isfinite(s) or s == 0:
            return np.full(len(y), (len(y) + 1) / 2.0)
        return rankdata(np.round(y / s / rel))
    s = np.nanmax(np.abs(y), axis=0)
    s[~np.isfinite(s) | (s == 0)] = 1.0
    return rankdata(np.round(y / s[None, :] / rel), axis=0)


def corr_ranks(R: np.ndarray, rt: np.ndarray) -> np.ndarray | float:
    """Pearson correlation of rank vector(s) R (n,) or (n,K) with target ranks rt.
    Constant predictions get correlation 0 (no ranking information)."""
    rt = rt - rt.mean()
    nt = np.sqrt((rt * rt).sum())
    if R.ndim == 1:
        r = R - R.mean()
        nr = np.sqrt((r * r).sum())
        return 0.0 if nr == 0 or nt == 0 else float((r @ rt) / (nr * nt))
    Rc = R - R.mean(0, keepdims=True)
    nr = np.sqrt((Rc * Rc).sum(0))
    out = np.zeros(R.shape[1])
    ok = (nr > 0) & (nt > 0)
    out[ok] = (Rc[:, ok].T @ rt) / (nr[ok] * nt)
    return out


def spearman(y: np.ndarray, target_ranks: np.ndarray) -> float:
    return corr_ranks(ranks_of(y), target_ranks)


def full_metrics(y: np.ndarray, ystar: np.ndarray, target_ranks: np.ndarray, kfrac: float = 0.1) -> dict:
    n = len(y)
    yr = np.round(y / (np.max(np.abs(y)) or 1.0) / RANK_REL)
    rho = corr_ranks(rankdata(yr), target_ranks)
    if np.ptp(yr) == 0:
        tau = 0.0
    else:
        tau = float(kendalltau(yr, ystar).statistic)
    k = max(1, int(math.ceil(kfrac * n)))
    o_pred = np.lexsort((np.arange(n), -yr))[:k]
    o_true = np.lexsort((np.arange(n), -ystar))[:k]
    topk = len(np.intersect1d(o_pred, o_true)) / k
    g = (ystar - ystar.min()) / (np.ptp(ystar) or 1.0)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((g[o_pred] * disc).sum())
    idcg = float((np.sort(g)[::-1][:k] * disc).sum())
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return dict(rho=rho, tau=tau, topk=topk, ndcg=ndcg)


# =============================================================================
# Diffusions
# =============================================================================
def diffuse(T: sp.csr_matrix, x: np.ndarray, alpha, y0: np.ndarray | None = None,
            tol: float = 1e-12, maxit: int = 20000, lazy: bool = True):
    """Normalised damped diffusion  y <- N((1-a) x + a T y)  (scalar or per-node a).

    The fixed point is the Perron vector of B = ((1-a) x) 1^T + diag(a) T.
    The lazy form iterates (I+B)/2, which has the same Perron vector and
    converges on bipartite graphs where the plain power method oscillates."""
    a = np.broadcast_to(np.asarray(alpha, float), x.shape)
    u = (1.0 - a) * x
    y = (x if y0 is None else np.asarray(y0, float)).copy()
    y /= y.sum()
    it = 0
    for it in range(1, maxit + 1):
        By = u + a * (T @ y)
        yn = 0.5 * (y + By) if lazy else By
        s = yn.sum()
        if not s > 0:
            return x.copy(), it
        yn /= s
        d = np.abs(yn - y).sum()
        y = yn
        if d < tol:
            break
    return y, it


def diffuse_batch(T: sp.csr_matrix, x: np.ndarray, A: np.ndarray, Y0: np.ndarray | None = None,
                  tol: float = 1e-12, maxit: int = 20000):
    """Batched normalised diffusion; A is (K,) scalars or (n,K) per-node alphas."""
    n = len(x)
    if A.ndim == 1:
        A = np.broadcast_to(A[None, :], (n, len(A)))
    U = (1.0 - A) * x[:, None]
    Y = (np.repeat(x[:, None], A.shape[1], 1) if Y0 is None else np.array(Y0, float)).copy()
    Y /= Y.sum(0, keepdims=True)
    it = 0
    for it in range(1, maxit + 1):
        Yn = 0.5 * (Y + U + A * (T @ Y))
        Yn /= Yn.sum(0, keepdims=True)
        d = np.abs(Yn - Y).sum(0).max()
        Y = Yn
        if d < tol:
            break
    return Y, it


def knz_mp(inst: Instance, iters: int | None = None, tol: float = 1e-12, maxit: int = 2000):
    """Karrer-Newman-Zdeborova message passing in membership variables.
    iters=None runs to convergence; iters=0 returns P_esc exactly."""
    n, E = inst.n, inst.edges
    m = len(E)
    pc = np.minimum(inst.p, P_CLIP)
    src = np.r_[E[:, 0], E[:, 1]]
    dst = np.r_[E[:, 1], E[:, 0]]
    pe = np.r_[pc, pc]
    rev = np.r_[np.arange(m) + m, np.arange(m)]
    nu = np.ones(2 * m)
    K = maxit if iters is None else iters
    k = 0
    for k in range(K):
        lg = np.log1p(-pe * nu)
        L = np.bincount(dst, weights=lg, minlength=n)
        nu_new = -np.expm1(L[src] - lg[rev])
        d = np.abs(nu_new - nu).max()
        nu = nu_new
        if iters is None and d < tol:
            break
    mu = -np.expm1(np.bincount(dst, weights=np.log1p(-pe * nu), minlength=n))
    return mu, k


# =============================================================================
# The proportional principle
# =============================================================================
def theorem1_roots(x: np.ndarray, S: np.ndarray):
    """Both algebraic branches of the coupled system (Theorem 1)."""
    y1, a1 = x.copy(), np.zeros_like(x)
    y2 = S - x
    with np.errstate(divide="ignore", invalid="ignore"):
        a2 = (S - 2 * x) / (S - x)
    return (y1, a1), (y2, a2)


def rescale(y: np.ndarray, x: np.ndarray, scale: str, gamma: float = 1.0) -> np.ndarray:
    if scale == "raw":
        return y
    if scale == "l1":
        s = y.sum()
        return y / s if s > 0 else y
    if scale == "ex":
        z = np.maximum(y - x, 0.0)
        mz = float(np.mean(z / x))
        if not mz > 0:
            return x.copy()
        return x + (gamma / mz) * z
    raise ValueError(scale)


def realization_step(kind: str, y: np.ndarray, S: np.ndarray, x: np.ndarray,
                     gammaC: float = 1.0, alpha_max: float = 1.0 - 1e-9, xr: np.ndarray | None = None,
                     alpha_transform: str | None = None, rng=None):
    """One update of realization A, B or C. Returns (y_new, alpha).

    alpha_transform (matched controls): 'mean' replaces the per-node alpha by its
    mean (same average damping, no per-node law); 'perm' reassigns the alpha
    vector to vertices uniformly at random."""
    xr = x if xr is None else xr
    if kind == "A":
        alpha = np.zeros_like(x)
        act = S > 2 * x
        alpha[act] = (S[act] - 2 * x[act]) / (S[act] - x[act])
    elif kind == "B":
        ex = np.maximum(y - x, 0.0)
        alpha = ex / (ex + xr)                       # = 1 - x/y when xr = x
    elif kind == "C":
        alpha = gammaC * S / (x + gammaC * S)
    else:
        raise ValueError(kind)
    alpha = np.clip(alpha, 0.0, alpha_max)
    if alpha_transform == "mean":
        alpha = np.full_like(alpha, alpha.mean())
    elif alpha_transform == "perm":
        alpha = alpha[rng.permutation(len(alpha))]
    return (1 - alpha) * x + alpha * S, alpha


def run_realization(T, x, kind, scale="raw", gamma=1.0, gammaC=1.0, y0_gamma=1.0,
                    T_max=200, checkpoints=(1, 2, 3, 5, 10, 20, 50, 100, 200),
                    target_ranks=None, xr=None, y_init=None, record_alpha=False,
                    alpha_transform=None, seed=0):
    """Mode (a) standalone run of realization `kind` from the one-hop warm start
    y0 = x + y0_gamma * T x (or from y_init).  Records Spearman and dynamical
    diagnostics at checkpoints."""
    y = (x + y0_gamma * (T @ x)) if y_init is None else np.array(y_init, float)
    y = rescale(y, x, scale, gamma)
    out = {}
    cps = set(checkpoints)
    diverged = None
    alpha = np.zeros_like(x)
    rng = np.random.default_rng(seed)
    for t in range(1, T_max + 1):
        S = T @ y
        act_frac = float(np.mean(S > 2 * x))
        y_prev = y
        y, alpha = realization_step(kind, y, S, x, gammaC=gammaC, xr=xr,
                                    alpha_transform=alpha_transform, rng=rng)
        y = rescale(y, x, scale, gamma)
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e250:
            # Raw-scale growth on an expanding operator: the ranking has long
            # converged to the dominant direction; carry the last finite state.
            diverged = t
            y = y_prev
            for tc in sorted(c for c in cps if c >= t):
                rec = dict(act=act_frac, trivial=0.0, amean=float("nan"), astd=float("nan"),
                           excess_max=float("inf"), ysum=float(y.sum()), diverged=True)
                if target_ranks is not None:
                    rec["rho"] = spearman(y, target_ranks)
                out[tc] = rec
            break
        if t in cps:
            rec = dict(
                act=act_frac,
                trivial=float(np.mean(np.abs(y - x) <= 1e-14 * x)),
                amean=float(alpha.mean()), astd=float(alpha.std()),
                excess_max=float(np.max((y - x) / x)),
                ysum=float(y.sum()),
            )
            if target_ranks is not None:
                rec["rho"] = spearman(y, target_ranks)
            if record_alpha:
                rec["alpha"] = alpha.copy()
                rec["y"] = y.copy()
            out[t] = rec
    return out, y, diverged


def anchored_alpha(q: np.ndarray, alpha0: float, eta: float, amax: float = 1.0 - 1e-6,
                   amin: float = 1e-6) -> np.ndarray:
    """AP(eta): odds_i = exp(c + eta * (log q_i - mean log q)), c chosen so that
    mean_i alpha_i = alpha0.  eta = 0 returns alpha0 everywhere (scalar rule);
    eta = 1 is the proportional law with its global scale anchored."""
    n = len(q)
    if eta == 0.0:
        return np.full(n, float(alpha0))
    a0 = float(np.clip(alpha0, amin, amax))
    lq = np.log(np.maximum(q, 1e-300))
    l = eta * (lq - lq.mean())
    # mean(expit(c + l)) is increasing in c: safeguarded Newton with a bracket
    lo, hi = logit(amin) - np.abs(l).max() - 1.0, logit(amax) + np.abs(l).max() + 1.0
    c = float(logit(a0))
    for _ in range(60):
        s = expit(c + l)
        f = s.mean() - a0
        if abs(f) < 1e-13:
            break
        if f < 0:
            lo = c
        else:
            hi = c
        d = float((s * (1.0 - s)).mean())
        cn = c - f / d if d > 1e-300 else 0.5 * (lo + hi)
        c = cn if lo < cn < hi else 0.5 * (lo + hi)
    return np.clip(expit(c + l), amin, amax)


def power_alpha(q: np.ndarray, gamma: float, eta: float, amax: float = 1.0 - 1e-6) -> np.ndarray:
    """Un-anchored generalised law: alpha/(1-alpha) = gamma * q^eta."""
    odds = gamma * np.power(np.maximum(q, 1e-300), eta)
    return np.clip(odds / (1.0 + odds), 0.0, amax)


def polish_ap(T, x, y_base, alpha0, eta, K, tau=0.0, self_consistent=False, permute_seed=None,
              tol=1e-12, maxit=20000):
    """Mode (b) polishing with the anchored proportional law.
    q_i = y_i / x_reg_i with x_reg = (1-tau) x + tau/n.  K = None runs to the
    fixed point.  permute_seed shuffles the alpha vector (control)."""
    n = len(x)
    xr = (1 - tau) * x + tau / n
    y = np.asarray(y_base, float) / np.sum(y_base)
    alpha = anchored_alpha(y / xr, alpha0, eta)
    if permute_seed is not None:
        alpha = alpha[np.random.default_rng(permute_seed).permutation(n)]
    if not self_consistent:
        if K is None:
            y, _ = diffuse(T, x, alpha, y0=y, tol=tol, maxit=maxit)
        else:
            for _ in range(K):
                y = (1 - alpha) * x + alpha * (T @ y)
                y /= y.sum()
        return y, alpha
    kmax = maxit if K is None else K
    for k in range(kmax):
        yn = (1 - alpha) * x + alpha * (T @ y)
        yn /= yn.sum()
        if K is None:
            yn = 0.5 * (y + yn)
            yn /= yn.sum()
        d = np.abs(yn - y).sum()
        y = yn
        alpha = anchored_alpha(y / xr, alpha0, eta)
        if K is None and d < tol:
            break
    return y, alpha


# =============================================================================
# Oracles
# =============================================================================
ALPHA_GRID = np.round(np.arange(1, 41) * 0.025, 4)          # 0.025 ... 1.000


def scalar_sweep(T, x, grid=ALPHA_GRID, tol=1e-12, maxit=20000):
    Y, it = diffuse_batch(T, x, np.asarray(grid, float), tol=tol, maxit=maxit)
    return Y, it


def pernode_oracle(T, x, fit_ranks, alpha_init, grid=None, sweeps=3, seed=0,
                   tol=1e-10, maxit=400, time_limit=None):
    """Coordinate-ascent over per-node damping alpha in [0,1]^n, maximising
    Spearman against `fit_ranks` (an independent ground-truth sample).
    Candidates for one coordinate are evaluated in a single batched diffusion."""
    if grid is None:
        grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0])
    n = len(x)
    a = np.array(alpha_init, float).copy()
    y, _ = diffuse(T, x, a)
    best = spearman(y, fit_ranks)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    n_acc = 0
    for sw in range(sweeps):
        improved = 0
        for i in rng.permutation(n):
            c = grid[np.abs(grid - a[i]) > 1e-12]
            K = len(c)
            A = np.repeat(a[:, None], K, 1)
            A[i, :] = c
            Y, _ = diffuse_batch(T, x, A, Y0=np.repeat(y[:, None], K, 1), tol=tol, maxit=maxit)
            sc = corr_ranks(ranks_of(Y), fit_ranks)
            k = int(np.argmax(sc))
            if sc[k] > best + 1e-12:
                best = float(sc[k])
                a[i] = c[k]
                y = Y[:, k].copy()
                improved += 1
            if time_limit and time.time() - t0 > time_limit:
                break
        n_acc += improved
        if improved == 0 or (time_limit and time.time() - t0 > time_limit):
            break
    y, _ = diffuse(T, x, a, y0=y)
    return a, y, dict(fit_rho=spearman(y, fit_ranks), accepted=n_acc, sweeps=sw + 1,
                      seconds=time.time() - t0)
