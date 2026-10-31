"""Re-derive the frozen configurations from the development panel and verify
that results/selected.json agrees with the stated selection rules.

The frozen file is never modified.  Every entry is recomputed from
results/grid_dev/ (and results/controls_dev.csv.gz for the constant-alpha
control) with the rule below, and compared with the frozen entry.  An entry is
reported as MATCH when the frozen configuration is the computed one, and as
TIE when it differs only among configurations with identical development means
(for example eta = 0, where the 'fix' and 'sc' modes coincide).

Selection rules (development panel only, mean Spearman rho over instances):
  A_best, B_best, C_best  best standalone configuration of each realization
  C_dir1, B_unif          best standalone configuration of any realization on
                          the Dirichlet(1) / uniform instances only
  AP_eta0.25              best eta != 0 of the anchored law at its fixed point
                          (eta = 0 is the calibrated rule itself)
  AP_eta1                 the manuscripts' law, eta = 1
  PB_rule_best            best anchored polish of the calibrated rule
  PB_rule_nominal         eta = 1, fixed alpha, K = 1, beta = 0.5, score blend
  PB_rule_eta1            best anchored polish of the rule with eta = 1
  PB_mp2_best             best anchored polish of 2-step message passing
  PB_mp2_eta0, _eta1      best polish of message passing with eta = 0 / 1
  PBX_rule_best           best realization polisher of the rule
  PBX_rule_unif, _dir1    best realization polisher on uniform / Dirichlet(1)
  PBX_rule_nominalB       the manuscripts' Mode (b): B, EW, excess gamma = 1,
                          K = 1, beta = 0.5, score blend
  CTRL_Cmean, CTRL_Cperm, CTRL_lin0.5
                          the PBX_rule_best pipeline with the law replaced by its
                          per-step mean, a permutation, or alpha = 0.5
  CTRL_lin0.9             best pipeline with constant alpha = 0.9 on raw P

Usage:  python select_configs.py [--write OUT.json]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as A

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
INF = 10 ** 6
TOL = 1e-12


def g(v):
    return f"{float(v):g}"


def main():
    R, M, _ = A.load_stage("grid", "dev")
    R = R.join(M[["prior"]], on="id")
    R["fam"] = R.method.str.split(":").str[0]
    keys = ["method", "T", "beta", "blend"]

    def means(df):
        return df.groupby(keys, dropna=False).rho.mean().sort_values(ascending=False, kind="stable")

    def best(df):
        m = means(df)
        return m.index[0], float(m.iloc[0]), m

    out, cand = {}, {}
    sa = R[R.fam == "sa"].copy()
    sa["kind"] = sa.method.str.split(":").str[1].str.split("|").str[0]
    for k in "ABC":
        cand[f"{k}_best"] = best(sa[sa.kind == k])
    cand["C_dir1"] = best(sa[sa.prior == "dir1"])
    cand["B_unif"] = best(sa[sa.prior == "uniform"])
    ap = R[(R.fam == "ap") & (R["T"] == INF)]
    cand["AP_eta0.25"] = best(ap[ap.method != "ap:0"])
    pb = R[R.fam == "pb"].copy()
    parts = pb.method.str.split(":").str[1].str.split("|")
    pb["base"], pb["eta"] = parts.str[0], parts.str[1].astype(float)
    cand["PB_rule_best"] = best(pb[pb.base == "rule"])
    cand["PB_rule_eta1"] = best(pb[(pb.base == "rule") & (pb.eta == 1.0)])
    cand["PB_mp2_best"] = best(pb[pb.base == "mp2"])
    cand["PB_mp2_eta0"] = best(pb[(pb.base == "mp2") & (pb.eta == 0.0)])
    cand["PB_mp2_eta1"] = best(pb[(pb.base == "mp2") & (pb.eta == 1.0)])
    pbx = R[R.fam == "pbx"]
    cand["PBX_rule_best"] = best(pbx)
    cand["PBX_rule_unif"] = best(pbx[pbx.prior == "uniform"])
    cand["PBX_rule_dir1"] = best(pbx[pbx.prior == "dir1"])

    # translate a winning (method, T, beta, blend) into the selected.json schema
    def entry(name, key):
        meth, T, beta, blend = key
        fam, rest = meth.split(":", 1)
        p = rest.split("|")
        if fam == "sa":
            return dict(type="sa", kind=p[0], op=p[1], scale=p[2], gamma=float(p[3]), T=int(T))
        if fam == "ap":
            return dict(type="ap", eta=float(p[0]), T=int(T))
        if fam == "pb":
            return dict(type="pb", base=p[0], eta=float(p[1]), mode=p[2], T=int(T), beta=float(beta), blend=blend)
        if fam == "pbx":
            return dict(type="pbx", base=p[0], kind=p[1], op=p[2], scale=p[3], gamma=float(p[4]), T=int(T),
                        beta=float(beta), blend=blend)
        raise ValueError(name)

    for name, (key, val, _) in cand.items():
        out[name] = entry(name, key)
    out["AP_eta1"] = dict(type="ap", eta=1.0, T=INF)
    out["PB_rule_nominal"] = dict(type="pb", base="rule", eta=1.0, mode="fix", T=1, beta=0.5, blend="s")
    out["PBX_rule_nominalB"] = dict(type="pbx", base="rule", kind="B", op="EW", scale="ex", gamma=1.0, T=1,
                                    beta=0.5, blend="s")
    pbxb = out["PBX_rule_best"]
    for law in ("Cmean", "Cperm", "lin0.5"):
        out[f"CTRL_{law}"] = dict(type="ctrl", law=law, gamma=pbxb["gamma"], T=pbxb["T"], beta=pbxb["beta"],
                                  blend=pbxb["blend"])
    C = pd.read_csv(RES / "controls_dev.csv.gz")
    c9 = C[C.law == "lin0.9"].groupby(["gamma", "K", "beta", "blend"]).rho.mean().sort_values(ascending=False)
    gm, K, b, bl = c9.index[0]
    out["CTRL_lin0.9"] = dict(type="ctrl", law="lin0.9", gamma=float(gm), T=int(K), beta=float(b), blend=bl)

    frozen = json.loads((RES / "selected.json").read_text())
    digest = hashlib.sha256((RES / "selected.json").read_bytes()).hexdigest()
    print(f"frozen file sha256 {digest}")
    rec = (RES / "selected.sha256").read_text().split()[0] if (RES / "selected.sha256").exists() else ""
    print("recorded digest   ", rec, "(agrees)" if rec == digest else "(DIFFERS)")
    bad = 0
    for name in frozen:
        f, d = frozen[name], out.get(name)
        if f == d:
            status = "MATCH"
        elif name in cand and f.get("type") in ("pb", "pbx", "sa", "ap"):
            m = cand[name][2]
            fk = frozen_key(f)
            fv = float(m.get(fk, np.nan)) if fk in m.index else np.nan
            status = "TIE" if abs(fv - cand[name][1]) < TOL else "DIFFERENT"
        else:
            status = "DIFFERENT"
        bad += status == "DIFFERENT"
        print(f"{name:20s} {status}")
    print(f"{len(frozen)} frozen configurations, {bad} inconsistent with the rules")
    if "--write" in sys.argv:
        dst = Path(sys.argv[sys.argv.index("--write") + 1])
        if dst.resolve() == (RES / "selected.json").resolve():
            raise SystemExit("refusing to overwrite the frozen selection")
        dst.write_text(json.dumps(out, indent=1))
        print("derived selection written to", dst)
    return bad


def frozen_key(f):
    t = f["type"]
    if t == "sa":
        return (f"sa:{f['kind']}|{f['op']}|{f['scale']}|{g(f['gamma'])}", f["T"], np.nan, "-")
    if t == "ap":
        return (f"ap:{g(f['eta'])}", f["T"], np.nan, "-")
    if t == "pb":
        return (f"pb:{f['base']}|{g(f['eta'])}|{f['mode']}", f["T"], f["beta"], f["blend"])
    if t == "pbx":
        return (f"pbx:{f['base']}|{f['kind']}|{f['op']}|{f['scale']}|{g(f['gamma'])}", f["T"], f["beta"], f["blend"])
    return None


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
