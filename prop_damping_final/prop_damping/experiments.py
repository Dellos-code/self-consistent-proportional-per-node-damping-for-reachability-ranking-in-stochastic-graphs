"""Phase-1 experiment runner.

Stages
------
grid <panel>      full factorial over realizations, operators, scales, T, the
                  anchored family AP(eta), Mode (b) polishing and controls,
                  plus all baselines and oracles.  Run on `dev` (selection) and
                  `test` (held-out report).
confirm <panel>   baselines, oracles and the configurations frozen in
                  results/selected.json (chosen on dev only); used for test,
                  scale, real and prior panels.  `prior` also sweeps tau.

Per instance the runner writes results/<stage>_<panel>/<id>.csv.gz (one row
per method/config) and <id>.json (diagnostics).  Runs are resumable.
"""
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import prop_damping_lib as L
from build_panels import DATA, load_instance

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
INF = 10 ** 6                                   # stands for T, K = infinity

KINDS = ("A", "B", "C")
OPS = ("EW", "W", "WT", "P", "EP")
SCALES = (("raw", 1.0), ("l1", 1.0), ("ex", 1.0), ("ex", 4.0), ("ex", 16.0))
CPS = (1, 2, 3, 5, 10, 20, 50, 200)
ETAS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
KS = (1, 2, 5, INF)
BETAS = (0.25, 0.5, 0.75, 1.0)
TAUS = (0.0, 0.1, 0.5, 0.9, 1.0)


def nrm(y):
    y = np.asarray(y, float)
    s = y.sum()
    return y / s if s != 0 else y


class Recorder:
    def __init__(self, rA, iid):
        self.rA, self.iid, self.rows = rA, iid, []

    def add(self, method, y=None, rho=None, T=0, beta=np.nan, blend="-"):
        if rho is None:
            rho = L.spearman(y, self.rA)
        self.rows.append((self.iid, method, int(T), float(beta), blend, float(rho)))
        return rho

    def blends(self, method, y_base, y_prop, T):
        rb, rp = L.ranks_of(y_base), L.ranks_of(y_prop)
        nb, np_ = nrm(y_base), nrm(y_prop)
        for b in BETAS:
            if b == 1.0:
                self.add(method, y_prop, T=T, beta=1.0, blend="s")
                continue
            self.add(method, (1 - b) * nb + b * np_, T=T, beta=b, blend="s")
            self.add(method, (1 - b) * rb + b * rp, T=T, beta=b, blend="r")

    def frame(self):
        return pd.DataFrame(self.rows, columns=["id", "method", "T", "beta", "blend", "rho"])


# ---------------------------------------------------------------------------
def ap_run(T, x, alpha0, eta, y_init, Ts, tau=0.0, maxit=5000, tol=1e-11, gamma=None):
    """Self-consistent anchored (or free-gamma) proportional diffusion from y_init.
    Returns {T: y}.  Finite T are plain sweeps; INF continues with lazy sweeps to
    the fixed point."""
    n = len(x)
    xr = (1 - tau) * x + tau / n
    law = (lambda q: L.anchored_alpha(q, alpha0, eta)) if gamma is None else \
          (lambda q: L.power_alpha(q, gamma, eta))
    y = nrm(y_init)
    out = {}
    tmax_finite = max([t for t in Ts if t < INF], default=0)
    for t in range(1, tmax_finite + 1):
        a = law(y / xr)
        y = nrm((1 - a) * x + a * (T @ y))
        if t in Ts:
            out[t] = y.copy()
    if INF in Ts:
        it = 0
        for it in range(maxit):
            a = law(y / xr)
            yn = nrm(0.5 * (y + nrm((1 - a) * x + a * (T @ y))))
            d = np.abs(yn - y).sum()
            y = yn
            if d < tol:
                break
        out[INF] = y.copy()
        out["iters"] = it + 1
    return out


def polish_fixed(T, x, y_base, alpha, Ks):
    out = {}
    y = nrm(y_base)
    for k in range(1, max([k for k in Ks if k < INF], default=0) + 1):
        y = nrm((1 - alpha) * x + alpha * (T @ y))
        if k in Ks:
            out[k] = y.copy()
    if INF in Ks:
        out[INF], _ = L.diffuse(T, x, alpha, y0=y_base)
    return out


def excess_embed(y_base, x, gamma):
    z = np.maximum(nrm(y_base), 1e-300)
    z = z / np.mean(z / x)
    return x + gamma * z


# ---------------------------------------------------------------------------
def baselines_and_oracles(inst, yA, yB, rec, diag, do_pernode=True, pernode_limit=None):
    x, ops = inst.x, inst.ops()
    EW = ops["EW"]
    rA, rB = rec.rA, L.ranks_of(yB)
    Y = {}
    # references
    Y["ref:prior"] = x
    Y["ref:Pesc"] = ops["Pesc"]
    Y["ref:MP2"] = L.knz_mp(inst, iters=2)[0]
    mpinf, it = L.knz_mp(inst, iters=None)
    Y["ref:MPinf"] = mpinf
    Y["ref:ceiling"] = yB
    # scalar diffusions
    a0 = L.rule_alpha(ops["Pbar"], "EW")
    Y["base:rule_EW"], _ = L.diffuse(EW, x, a0)
    Y["base:fixed085"], _ = L.diffuse(EW, x, 0.85)
    Y["base:rule_WT"], _ = L.diffuse(ops["WT"], x, L.rule_alpha(ops["Pbar"], "WT"))
    Y["base:local_pesc"], _ = L.diffuse(EW, x, ops["Pesc"])
    # scalar oracle over the alpha grid: fit on B, evaluate on A
    Ys, _ = L.scalar_sweep(EW, x)
    R = L.ranks_of(Ys)
    scA, scB = L.corr_ranks(R, rA), L.corr_ranks(R, rB)
    kB = int(np.argmax(scB))
    Y["oracle:scalar"] = Ys[:, kB]
    diag["scalar_curve_A"] = scA.round(6).tolist()
    diag["scalar_curve_B"] = scB.round(6).tolist()
    diag["alpha_rule"] = a0
    diag["alpha_scalar_B"] = float(L.ALPHA_GRID[kB])
    diag["alpha_scalar_A"] = float(L.ALPHA_GRID[int(np.argmax(scA))])
    rec.add("oracle:scalar_in", rho=float(scA.max()))
    # push-PPR scalar oracle (fit B) for completeness
    Yw, _ = L.scalar_sweep(ops["WT"], x)
    Rw = L.ranks_of(Yw)
    kw = int(np.argmax(L.corr_ranks(Rw, rB)))
    Y["oracle:scalar_WT"] = Yw[:, kw]
    # per-node oracle
    if do_pernode:
        a_node, y_node, info = L.pernode_oracle(EW, x, rB, np.full(inst.n, L.ALPHA_GRID[kB]),
                                                time_limit=pernode_limit)
        Y["oracle:pernode"] = y_node
        info["fit_scalar_B"] = float(scB.max())
        # what does the optimal per-node damping correlate with?
        feats = {
            "prop_true": np.clip(1 - x / yB, 0, 1),
            "log_enrich_true": np.log(yB / x),
            "Pesc": ops["Pesc"],
            "degree": ops["deg"].astype(float),
            "log_enrich_rule": np.log(Y["base:rule_EW"] / x),
            "prior": x,
        }
        ra = L.ranks_of(a_node)
        info["corr"] = {k: L.corr_ranks(ra, L.ranks_of(v)) for k, v in feats.items()}
        info["alpha_mean"] = float(a_node.mean())
        info["alpha_std"] = float(a_node.std())
        info["frac_changed"] = float(np.mean(np.abs(a_node - L.ALPHA_GRID[kB]) > 1e-12))
        diag["pernode"] = info
    # the principle evaluated with the TRUE y* (independent sample B)
    Y["true:prop"], _ = L.diffuse(EW, x, np.clip(1 - x / yB, 0, 1 - 1e-6))
    for eta in (0.5, 1.0):
        Y[f"true:ap{eta}"], _ = L.diffuse(EW, x, L.anchored_alpha(yB / x, a0, eta))
    for k, y in Y.items():
        rec.add(k, y)
    return Y, a0


def full_metric_rows(iid, Y, yA, rA, keys):
    rows = []
    for k in keys:
        if k in Y:
            m = L.full_metrics(Y[k], yA, rA)
            rows.append(dict(id=iid, method=k, **m))
    return rows


# ---------------------------------------------------------------------------
def grid_instance(inst, yA, yB, rec, diag, Ybase, a0):
    x, ops = inst.x, inst.ops()
    EW = ops["EW"]
    rA = rec.rA
    y_rule = Ybase["base:rule_EW"]
    # ---- Mode (a): realizations A/B/C ----
    dyn = {}
    for kind in KINDS:
        for op in OPS:
            T = ops[op]
            confs = [(s, g, 1.0) for s, g in SCALES]
            if kind == "C":
                confs += [(s, 1.0, gc) for s in ("raw", "l1") for gc in (0.25, 4.0)]
            for scale, g, gc in confs:
                out, y, div = L.run_realization(T, x, kind, scale=scale, gamma=g, gammaC=gc,
                                                T_max=200, checkpoints=CPS, target_ranks=rA)
                tag = f"sa:{kind}|{op}|{scale}|{g:g}" + (f"|gc{gc:g}" if gc != 1.0 else "")
                for t in CPS:
                    rec.add(tag, rho=out[t]["rho"], T=t)
                if op in ("EW", "P") and gc == 1.0:
                    dyn[tag] = {t: (out[t].get("act"), out[t].get("trivial")) for t in (1, 5, 20, 200)}
                    if div:
                        dyn[tag]["diverged_at"] = div
    diag["dynamics"] = dyn
    # ---- Mode (a): anchored family AP(eta), self-consistent from the warm start ----
    y_warm = x + EW @ x
    for eta in ETAS:
        out = ap_run(EW, x, a0, eta, y_warm, Ts=(1, 2, 5, 10, INF))
        for t in (1, 2, 5, 10, INF):
            rec.add(f"ap:{eta:g}", out[t], T=t)
    # free-gamma generalised law (not anchored)
    for gamma in (0.25, 1.0, 4.0, 16.0):
        for eta in (0.5, 1.0):
            out = ap_run(EW, x, None, eta, y_warm, Ts=(5, INF), gamma=gamma)
            for t in (5, INF):
                rec.add(f"free:{gamma:g}|{eta:g}", out[t], T=t)
    # ---- Mode (b): polishing ----
    bases = {"rule": y_rule, "mp2": Ybase["ref:MP2"]}
    for bname, yb in bases.items():
        if bname == "mp2" and not np.all(yb > 0):
            yb = np.maximum(yb, 1e-300)
        for eta in ETAS:
            if eta == 0.0 and bname == "rule":
                continue                                   # identical to the base
            alpha = L.anchored_alpha(nrm(yb) / x, a0, eta)
            outf = polish_fixed(EW, x, yb, alpha, KS)
            for k in KS:
                rec.blends(f"pb:{bname}|{eta:g}|fix", yb, outf[k], T=k)
            outs = ap_run(EW, x, a0, eta, yb, Ts=KS)
            for k in KS:
                rec.blends(f"pb:{bname}|{eta:g}|sc", yb, outs[k], T=k)
        # control: same alpha distribution (eta=1), randomly reassigned to vertices
        alpha = L.anchored_alpha(nrm(yb) / x, a0, 1.0)
        alpha = alpha[np.random.default_rng(L.stable_seed(inst.name, "perm")).permutation(inst.n)]
        outp = polish_fixed(EW, x, yb, alpha, KS)
        for k in KS:
            rec.blends(f"pbperm:{bname}", yb, outp[k], T=k)
        # realizations A/B/C as polishers, from an excess embedding of the base
        for kind in KINDS:
            for op in ("EW", "P"):
                for scale in ("raw", "ex"):
                    for g in (1.0, 4.0, 16.0):
                        y0 = excess_embed(yb, x, g)
                        out, y, div = L.run_realization(ops[op], x, kind, scale=scale, gamma=g,
                                                        T_max=200, checkpoints=(1, 2, 5, 200),
                                                        y_init=y0, record_alpha=True)
                        for k, t in ((1, 1), (2, 2), (5, 5), (INF, 200)):
                            yp = out[t].get("y", y)
                            tag = f"pbx:{bname}|{kind}|{op}|{scale}|{g:g}"
                            rec.add(tag, yp, T=k, beta=1.0, blend="s")
                            rec.add(tag, 0.5 * nrm(yb) + 0.5 * nrm(yp), T=k, beta=0.5, blend="s")
                            rec.add(tag, 0.5 * L.ranks_of(yb) + 0.5 * L.ranks_of(yp), T=k, beta=0.5, blend="r")


def selected_instance(inst, yA, yB, rec, diag, Ybase, a0, selected, taus=(0.0,)):
    """Evaluate the frozen configurations from results/selected.json."""
    x, ops = inst.x, inst.ops()
    EW = ops["EW"]
    Ysel = {}
    for name, c in selected.items():
        uses_odds = c["type"] in ("ap", "pb") or (c["type"] == "sa" and c["kind"] == "B")
        for tau in (taus if uses_odds else (0.0,)):
            sfx = "" if tau == 0.0 else f"@tau{tau:g}"
            xr = (1 - tau) * x + tau / inst.n
            if c["type"] == "sa":
                y_init = None
                out, y, div = L.run_realization(ops[c["op"]], x, c["kind"], scale=c["scale"],
                                                gamma=c["gamma"], gammaC=c.get("gc", 1.0),
                                                T_max=max(c["T"], 1), checkpoints=(c["T"],),
                                                record_alpha=True,
                                                xr=(xr if c["kind"] == "B" else None))
                yy = out[c["T"]].get("y", y)
            elif c["type"] == "ap":
                y_warm = x + EW @ x
                out = ap_run(EW, x, a0, c["eta"], y_warm, Ts=(c["T"],), tau=tau)
                yy = out[c["T"]]
            elif c["type"] == "pb":
                yb = Ybase["base:rule_EW"] if c["base"] == "rule" else np.maximum(Ybase["ref:MP2"], 1e-300)
                if c["mode"] == "fix":
                    alpha = L.anchored_alpha(nrm(yb) / xr, a0, c["eta"])
                    yp = polish_fixed(EW, x, yb, alpha, (c["T"],))[c["T"]]
                else:
                    yp = ap_run(EW, x, a0, c["eta"], yb, Ts=(c["T"],), tau=tau)[c["T"]]
                b = c["beta"]
                if b == 1.0:
                    yy = yp
                elif c["blend"] == "s":
                    yy = (1 - b) * nrm(yb) + b * nrm(yp)
                else:
                    yy = (1 - b) * L.ranks_of(yb) + b * L.ranks_of(yp)
            elif c["type"] == "pbx":
                yb = Ybase["base:rule_EW"] if c["base"] == "rule" else np.maximum(Ybase["ref:MP2"], 1e-300)
                y0 = excess_embed(yb, x, c["gamma"])
                Tm = 200 if c["T"] == INF else c["T"]
                out, y, div = L.run_realization(ops[c["op"]], x, c["kind"], scale=c["scale"],
                                                gamma=c["gamma"], T_max=Tm, checkpoints=(Tm,),
                                                y_init=y0, record_alpha=True)
                yp = out[Tm].get("y", y)
                b = c["beta"]
                if b == 1.0:
                    yy = yp
                elif c["blend"] == "s":
                    yy = (1 - b) * nrm(yb) + b * nrm(yp)
                else:
                    yy = (1 - b) * L.ranks_of(yb) + b * L.ranks_of(yp)
            elif c["type"] == "ctrl":
                from controls import sweeps
                yb = Ybase["base:rule_EW"]
                out, _ = sweeps(ops["P"], x, excess_embed(yb, x, c["gamma"]), c["law"], c["T"],
                                seed=L.stable_seed(inst.name, "cperm"))
                yp = out[c["T"]]
                b = c["beta"]
                yy = (1 - b) * nrm(yb) + b * nrm(yp) if c["blend"] == "s" else \
                    (1 - b) * L.ranks_of(yb) + b * L.ranks_of(yp)
            else:
                raise ValueError(c)
            Ysel["sel:" + name + sfx] = yy
    for k, y in Ysel.items():
        rec.add(k, y)
    return Ysel


# ---------------------------------------------------------------------------
HEADLINE = ["ref:prior", "ref:Pesc", "ref:MP2", "ref:MPinf", "ref:ceiling", "base:rule_EW",
            "base:fixed085", "base:rule_WT", "base:local_pesc", "oracle:scalar", "oracle:scalar_WT",
            "oracle:pernode", "true:prop", "true:ap0.5", "true:ap1.0"]


def run_one(args):
    stage, panel, path = args
    path = Path(path)
    iid = int(path.stem)
    outdir = RES / f"{stage}_{panel}"
    outdir.mkdir(parents=True, exist_ok=True)
    fcsv = outdir / f"{iid:04d}.csv.gz"
    if fcsv.exists():
        return iid, 0.0
    t0 = time.time()
    inst, yA, yB, yA2 = load_instance(path)
    rA = L.ranks_of(yA)
    rec = Recorder(rA, iid)
    diag = dict(meta=inst.meta)
    big = inst.n > 300
    # per-node oracle at n=900 only on the families where its headroom was claimed
    do_pn = not (panel == "scale" and inst.meta.get("family") not in ("grid", "rect", "er", "ba"))
    Ybase, a0 = baselines_and_oracles(inst, yA, yB, rec, diag, do_pernode=do_pn,
                                      pernode_limit=(240 if big else 120))
    Yall = dict(Ybase)
    if stage == "grid":
        grid_instance(inst, yA, yB, rec, diag, Ybase, a0)
    else:
        selected = json.loads((RES / "selected.json").read_text())
        taus = TAUS if panel == "prior" else (0.0,)
        Yall.update(selected_instance(inst, yA, yB, rec, diag, Ybase, a0, selected, taus=taus))
    keys = HEADLINE + [k for k in Yall if k.startswith("sel:")]
    fm = full_metric_rows(iid, Yall, yA, rA, keys)
    diag["full_metrics"] = fm
    diag["seconds"] = time.time() - t0
    # store score vectors of headline/selected methods for later analysis (small panels only)
    if inst.n <= 600:
        np.savez_compressed(outdir / f"{iid:04d}_scores.npz",
                            **{k.replace(":", "__").replace("|", "_").replace("@", "_at_"): np.asarray(v, float)
                               for k, v in Yall.items() if k in keys})
    (outdir / f"{iid:04d}.json").write_text(json.dumps(diag, default=float))
    rec.frame().to_csv(fcsv, index=False, compression="gzip")
    return iid, time.time() - t0


def main(stage, panels, procs=2):
    for panel in panels:
        outdir = RES / f"{stage}_{panel}"
        outdir.mkdir(parents=True, exist_ok=True)
        paths = sorted((DATA / panel).glob("[0-9][0-9][0-9][0-9].npz"))
        jobs = [(stage, panel, str(p)) for p in paths]
        t0 = time.time()
        with Pool(procs) as pool:
            for k, (iid, dt) in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
                if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                    print(f"[{stage}/{panel}] {k + 1}/{len(jobs)} elapsed {time.time() - t0:.0f}s (last {dt:.1f}s)",
                          flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    pans = sys.argv[2].split(",")
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    main(st, pans, procs)
