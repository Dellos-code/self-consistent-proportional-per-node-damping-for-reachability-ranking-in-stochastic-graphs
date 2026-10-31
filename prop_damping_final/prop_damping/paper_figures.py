"""Bilingual publication figures (called by make_paper_assets.py).

Every figure is drawn once per language from the same code path; only the text
labels differ (dictionary L below). Colours: the validated reference palette
(light mode), first three slots for scatter-type panels, method identity on axes
or by direct labels elsewhere."""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

C1, C2, C3, C4, C5, C6 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, GRID_C, REF = "#0b0b0b", "#52514e", "#e6e5e1", "#8f8e88"

plt.rcParams.update({
    "font.family": "FreeSerif", "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.8, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID_C,
    "grid.linewidth": 0.5, "lines.linewidth": 1.5, "legend.frameon": False, "savefig.dpi": 300,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
})

L = {
    "en": {
        "t": "iteration $t$", "triv": "fraction on trivial branch ($y_i=x_i$)", "rho": "mean Spearman $\\rho$",
        "f1a": "(a) Collapse on $EW$ (raw scale)", "f1b": "(b) Ranking quality along iterations",
        "f1c": "(c) The self-consistent point repels", "f1d": "(d) Realization C: Lehmer mean",
        "unif": "uniform prior", "dir": "Dirichlet(1) prior", "rule": "calibrated rule [4]",
        "lam": "Perron root $\\lambda$ of $T=\\lambda W$", "spec": "spectral radius",
        "JA": "$\\rho(J_A)=\\lambda$", "JB": "$\\rho(J_B)=-\\lambda^2+3\\lambda-1$", "meas": "measured",
        "stab": "stability threshold", "s": "neighbour signal $S/x$", "l2": "$L_2(x,S)/x$",
        "prior_lvl": "prior level", "below": "fixed point below prior", "dip": "minimum at $S=(\\sqrt{2}-1)x$",
        "raw": "raw", "ex": "excess $\\gamma=16$", "cp": "C on raw $P$",
        "f2a": "(a) Optimal per-node $\\alpha^*$ vs the law", "f2b": "(b) The law with the true $y^*$",
        "f2c": "(c) Anchored law AP($\\eta$)", "fracS": "fraction of vertices with $S_i>x_i$",
        "corr": "corr$(\\alpha^*_i,\\;1-x_i/y^*_i)$", "dtrue": "$\\Delta\\rho$ (true-$y^*$ law $-$ rule)",
        "binned": "binned mean", "eta": "exponent $\\eta$", "test": "test", "devp": "dev",
        "lawlab": "proportional law", "rulelab": "scalar rule",
        "f3a": "(a) Loss grows with prior dispersion", "f3b": "(b) What the ratio measures",
        "f3c": "(c) Denominator regularisation", "kappa": "$\\log_{10}\\kappa(x)$",
        "dap": "$\\Delta\\rho$ (AP($\\eta{=}1$) $-$ rule)", "g_unif": "uniform", "g_smooth": "Dirichlet /\nlog-normal",
        "g_seed": "seed priors", "r_inv": "corr$(r_i,\\,1/x_i)$", "r_y": "corr$(r_i,\\,y^*_i)$",
        "tau": "regularisation $\\tau$", "rank_corr": "rank correlation",
        "f4a": "(a) Mean $\\rho$ per panel", "f4b": "(b) Per-node law minus its mean-field twin",
        "p_test": "test ($n{=}100$)", "p_real": "real", "p_scale": "$n=900$", "p_prior": "prior panel",
        "role_ref": "references", "role_law": "uses the proportional law", "role_free": "law removed (controls)",
        "dlaw": "$\\Delta\\rho$ (law $-$ mean twin), 95% CI",
        "f5a": "(a) Accuracy vs cost, $n=100$", "f5b": "(b) Accuracy vs cost, $n=900$",
        "f5c": "(c) Wall clock vs size", "f5d": "(d) Ensemble vs equal-budget Monte Carlo",
        "ms": "single-process wall clock (ms, log scale)", "n": "number of vertices $n$",
        "ms_short": "wall clock (ms)", "deq": "$\\rho_{\\mathrm{ens}}-\\rho_{\\mathrm{MC}}$ at equal cost",
        "ens": "rule + raw-$P$ ensemble", "mc": "batched Monte Carlo", "oracle": "scalar oracle",
        "mclab": "MC-{k}", "fam_er": "ER", "fam_ba": "BA", "fam_grid": "grid", "fam_ws": "WS",
    },
    "gr": {
        "t": "επανάληψη $t$", "triv": "κλάσμα στον τετριμμένο κλάδο ($y_i=x_i$)", "rho": "μέσο $\\rho$ Spearman",
        "f1a": "(α) Κατάρρευση στον $EW$ (ακατέργαστη κλίμακα)", "f1b": "(β) Ποιότητα κατάταξης ανά επανάληψη",
        "f1c": "(γ) Το αυτοσυνεπές σημείο απωθεί", "f1d": "(δ) Υλοποίηση Γ: μέσος Lehmer",
        "unif": "ομοιόμορφο $x$", "dir": "$x\\sim$ Dirichlet(1)", "rule": "βαθμονομημένος κανόνας [4]",
        "lam": "ρίζα Perron $\\lambda$ του $T=\\lambda W$", "spec": "φασματική ακτίνα",
        "JA": "$\\rho(J_A)=\\lambda$", "JB": "$\\rho(J_B)=-\\lambda^2+3\\lambda-1$", "meas": "μετρήσεις",
        "stab": "όριο ευστάθειας", "s": "σήμα γειτόνων $S/x$", "l2": "$L_2(x,S)/x$",
        "prior_lvl": "στάθμη $y=x$", "below": "σταθερό σημείο κάτω από το $x$", "dip": "ελάχιστο στο $S=(\\sqrt{2}-1)x$",
        "raw": "ακατέργαστη", "ex": "περίσσεια $\\gamma=16$", "cp": "Γ σε ακατέργ. $P$",
        "f2a": "(α) Βέλτιστο $\\alpha^*$ έναντι νόμου", "f2b": "(β) Νόμος με το αληθές $y^*$",
        "f2c": "(γ) Αγκυρωμένος νόμος AP($\\eta$)", "fracS": "κλάσμα κορυφών με $S_i>x_i$",
        "corr": "συσχ.$(\\alpha^*_i,\\;1-x_i/y^*_i)$", "dtrue": "$\\Delta\\rho$ (νόμος με $y^*$ $-$ κανόνας)",
        "binned": "μέσος ανά κλάση", "eta": "εκθέτης $\\eta$", "test": "έλεγχος", "devp": "ανάπτυξη",
        "lawlab": "αναλογικός νόμος", "rulelab": "βαθμωτός κανόνας",
        "f3a": "(α) Απώλεια έναντι διασποράς του $x$", "f3b": "(β) Τι μετρά ο λόγος",
        "f3c": "(γ) Ομαλοποίηση παρονομαστή", "kappa": "$\\log_{10}\\kappa(x)$",
        "dap": "$\\Delta\\rho$ (AP($\\eta{=}1$) $-$ κανόνας)", "g_unif": "ομοιόμορφη", "g_smooth": "Dirichlet /\nλογαριθμοκανονική",
        "g_seed": "κατανομές σπόρων", "r_inv": "συσχ.$(r_i,\\,1/x_i)$", "r_y": "συσχ.$(r_i,\\,y^*_i)$",
        "tau": "ομαλοποίηση $\\tau$", "rank_corr": "συσχέτιση τάξης",
        "f4a": "(α) Μέσο $\\rho$ ανά σύνολο", "f4b": "(β) Νόμος ανά κόμβο μείον το δίδυμο μέσης τιμής",
        "p_test": "έλεγχος ($n{=}100$)", "p_real": "πραγματικά", "p_scale": "$n=900$", "p_prior": "αρχικές κατανομές",
        "role_ref": "αναφορές", "role_law": "με τον αναλογικό νόμο", "role_free": "χωρίς τον νόμο (έλεγχοι)",
        "dlaw": "$\\Delta\\rho$ (νόμος $-$ δίδυμο), 95% ΔΕ",
        "f5a": "(α) Ακρίβεια έναντι κόστους, $n=100$", "f5b": "(β) Ακρίβεια έναντι κόστους, $n=900$",
        "f5c": "(γ) Χρόνος έναντι μεγέθους", "f5d": "(δ) Συνδυασμός έναντι Monte Carlo ίσου κόστους",
        "ms": "χρόνος μίας διεργασίας (ms, λογαριθμική κλίμακα)", "n": "πλήθος κορυφών $n$",
        "ms_short": "χρόνος (ms)", "deq": "$\\rho_{\\mathrm{συνδ}}-\\rho_{\\mathrm{MC}}$ σε ίσο κόστος",
        "ens": "συνδυασμός κανόνα + ακατέργ. $P$", "mc": "Monte Carlo κατά δέσμες", "oracle": "βαθμωτό μαντείο",
        "mclab": "MC-{k}", "fam_er": "ER", "fam_ba": "BA", "fam_grid": "πλέγμα", "fam_ws": "WS",
    },
}

METHOD_SHORT = {
    "en": {"base:rule_EW": "calibrated rule [4]", "sel:AP_eta1": "AP($\\eta{=}1$)", "sel:PB_rule_nominal": "rule + AP polish",
           "sel:PBX_rule_best": "rule + C on $P$", "sel:CTRL_Cmean": "C law $\\to$ mean", "sel:CTRL_Cperm": "C law $\\to$ permuted",
           "sel:CTRL_lin0.9": "rule + $\\alpha{=}0.9$ on $P$", "oracle:scalar": "scalar oracle", "oracle:pernode": "per-node oracle",
           "ref:MP2": "KNZ MP (2 it.)", "rule": "rule", "ap": "AP($\\eta{=}1$)", "pol": "rule + AP polish",
           "pbx": "rule + C on $P$", "lin": "rule + $\\alpha{=}0.9$ on $P$", "mp": "KNZ MP"},
    "gr": {"base:rule_EW": "βαθμονομ. κανόνας [4]", "sel:AP_eta1": "AP($\\eta{=}1$)", "sel:PB_rule_nominal": "κανόνας + εξομάλ. AP",
           "sel:PBX_rule_best": "κανόνας + Γ σε $P$", "sel:CTRL_Cmean": "νόμος Γ $\\to$ μέση τιμή", "sel:CTRL_Cperm": "νόμος Γ $\\to$ μετάθεση",
           "sel:CTRL_lin0.9": "κανόνας + $\\alpha{=}0.9$ σε $P$", "oracle:scalar": "βαθμωτό μαντείο", "oracle:pernode": "μαντείο ανά κόμβο",
           "ref:MP2": "KNZ MP (2 επ.)", "rule": "κανόνας", "ap": "AP($\\eta{=}1$)", "pol": "κανόνας + εξομάλ. AP",
           "pbx": "κανόνας + Γ σε $P$", "lin": "κανόνας + $\\alpha{=}0.9$ σε $P$", "mp": "KNZ MP"},
}

CFG_SHORT = {
    "en": {"PBX_rule_best": "rule + C on $P$ ($K{=}5$)", "PBX_rule_unif": "rule + A on $P$ ($K{=}1$)",
           "PBX_rule_dir1": "rule + C on $P$ (rank)", "PBX_rule_nominalB": "Mode (b) as written (B)",
           "C_best": "C standalone ($EP$)", "B_best": "B standalone ($EP$)", "A_best": "A standalone ($P$)",
           "B_unif": "B standalone (push)", "C_dir1": "C standalone ($P$)"},
    "gr": {"PBX_rule_best": "κανόνας + Γ σε $P$ ($K{=}5$)", "PBX_rule_unif": "κανόνας + Α σε $P$ ($K{=}1$)",
           "PBX_rule_dir1": "κανόνας + Γ σε $P$ (τάξεις)", "PBX_rule_nominalB": "Εκδοχή (β) ως έχει (Β)",
           "C_best": "Γ αυτόνομη ($EP$)", "B_best": "Β αυτόνομη ($EP$)", "A_best": "Α αυτόνομη ($P$)",
           "B_unif": "Β αυτόνομη (ώθηση)", "C_dir1": "Γ αυτόνομη ($P$)"},
}


def save(fig, lang, name, PA):
    out = PA.FIGDIR[lang]
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================================ Fig 1
def fig1(lang, PA):
    T = L[lang]
    TR = PA.TRACES
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.7))
    # (a) trivial-branch fraction
    ax = axs[0, 0]
    for kind, col in (("A", C1), ("B", C2), ("C", C3)):
        for prior, ls in (("uniform", "-"), ("dir1", "--")):
            G = TR[(TR.kind == kind) & (TR.op == "EW") & (TR.scale == "raw") & (TR.prior == prior)]
            g = G.groupby("t").trivial.mean()
            ax.plot(g.index, g.values, ls, color=col, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel(T["t"]); ax.set_ylabel(T["triv"]); ax.set_title(T["f1a"]); ax.set_ylim(-0.03, 1.05)
    kl = {"A": "Α", "B": "Β", "C": "Γ"} if lang == "gr" else {"A": "A", "B": "B", "C": "C"}
    h = [Line2D([], [], color=c, lw=1.5, label=kl[k]) for k, c in (("A", C1), ("B", C2), ("C", C3))]
    h += [Line2D([], [], color=INK2, ls="-", lw=1.2, label=T["unif"]), Line2D([], [], color=INK2, ls="--", lw=1.2, label=T["dir"])]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.07), ncol=1, handlelength=1.8)
    # (b) rho along iterations
    ax = axs[0, 1]
    for kind, col in (("A", C1), ("B", C2), ("C", C3)):
        for scale, gam, ls in (("raw", 1.0, "-"), ("ex", 16.0, "--")):
            G = TR[(TR.kind == kind) & (TR.op == "EW") & (TR.scale == scale) & (TR.gamma == gam)]
            g = G.groupby("t").rho.mean()
            ax.plot(g.index, g.values, ls, color=col, lw=1.4)
    G = TR[(TR.kind == "C") & (TR.op == "P") & (TR.scale == "raw")]
    g = G.groupby("t").rho.mean()
    ax.plot(g.index, g.values, ":", color=C3, lw=1.8)
    rule = PA.GRID["dev"].mean("base:rule_EW")
    ax.axhline(rule, color=REF, ls=(0, (4, 2)), lw=1.0)
    ax.text(1.05, rule + 0.012, T["rule"], color=INK2, fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel(T["t"]); ax.set_ylabel(T["rho"]); ax.set_title(T["f1b"]); ax.set_ylim(-0.16, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    h = [Line2D([], [], color=INK2, ls="-", lw=1.2, label="$EW$, " + T["raw"]),
         Line2D([], [], color=INK2, ls="--", lw=1.2, label="$EW$, " + T["ex"]),
         Line2D([], [], color=C3, ls=":", lw=1.8, label=T["cp"])]
    ax.legend(handles=h, loc="lower right", ncol=2, handlelength=1.8, columnspacing=0.9, borderaxespad=0.25)
    # (c) spectra
    ax = axs[1, 0]
    lam = np.linspace(1.0, 2.0, 200)
    ax.plot(lam, lam, color=C1, label=T["JA"])
    ax.plot(lam, -lam ** 2 + 3 * lam - 1, color=C2, label=T["JB"])
    ax.axhline(1.0, color=REF, ls=(0, (4, 2)), lw=1.0)
    ax.text(1.32, 0.935, T["stab"], color=INK2, fontsize=7)
    pts = sorted([(float(k.split("=")[1].split(":")[0]), v) for k, v in PA.td.items() if k.startswith("P3_lam")])
    for i, (lm, v) in enumerate(pts):
        ax.plot([lm], [v[0]], "o", ms=4.5, mfc="white", mec=C1, mew=1.2)
        ax.plot([lm], [v[1]], "o", ms=4.5, mfc="white", mec=C2, mew=1.2,
                label=T["meas"] if i == 0 else None)
    ax.set_xlabel(T["lam"]); ax.set_ylabel(T["spec"]); ax.set_title(T["f1c"]); ax.set_ylim(0.9, 2.05)
    ax.legend(loc="upper left", handlelength=1.8)
    # (d) Lehmer mean
    ax = axs[1, 1]
    s = np.linspace(0, 3, 400)
    l2 = (1 + s ** 2) / (1 + s)
    ax.axvspan(0, 1, color=GRID_C, alpha=0.6, lw=0)
    ax.text(0.04, 1.86, T["below"], fontsize=7, color=INK2)
    ax.plot(s, s, ":", color=REF, lw=1.0)
    ax.axhline(1.0, color=REF, ls=(0, (4, 2)), lw=1.0)
    ax.text(2.2, 1.04, T["prior_lvl"], fontsize=7, color=INK2)
    ax.plot(s, l2, color=C3, lw=1.8)
    smin = np.sqrt(2) - 1
    ax.plot([smin], [(1 + smin ** 2) / (1 + smin)], "o", ms=4.5, color=C3)
    ax.annotate(T["dip"], xy=(smin, (1 + smin ** 2) / (1 + smin)), xytext=(0.55, 0.42), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.6)
    ax.set_xlabel(T["s"]); ax.set_ylabel(T["l2"]); ax.set_title(T["f1d"])
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, lang, "fig1_dynamics", PA)


# ============================================================================ Fig 2
def binned(x, y, edges):
    b = pd.cut(x, edges)
    g = pd.DataFrame({"x": x, "y": y, "b": b}).groupby("b", observed=True)
    return g.x.mean().values, g.y.mean().values


def fig2(lang, PA):
    T = L[lang]
    D = PA.SIGN
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.35))
    edges = [-0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
    for ax, col, lab, title in ((axs[0], "pn_corr_prop", T["corr"], T["f2a"]), (axs[1], "d_trueprop", T["dtrue"], T["f2b"])):
        for prior, c in (("uniform", C1), ("dir1", C2)):
            G = D[D.prior == prior]
            ax.scatter(G.frac_S_gt_x, G[col], s=7, color=c, alpha=0.55, lw=0,
                       label=T["unif"] if prior == "uniform" else T["dir"])
        bx, by = binned(D.frac_S_gt_x, D[col], edges)
        ax.plot(bx, by, "-o", color=INK, ms=3.5, lw=1.2, label=T["binned"])
        ax.axhline(0, color=REF, lw=0.8)
        ax.set_xlabel(T["fracS"]); ax.set_ylabel(lab); ax.set_title(title, fontsize=8.2)
    axs[0].legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2, markerscale=1.6, frameon=True,
                  facecolor="white", edgecolor="none", framealpha=0.9)
    axs[1].set_ylim(-2.05, 0.35)
    ax = axs[2]
    E = PA.T5
    for prior, c in (("uniform", C1), ("dir1", C2)):
        ax.plot(E.index, E[f"test_{prior}"], "-o", color=c, ms=3.2, lw=1.4)
        ax.plot(E.index, E[f"dev_{prior}"], "--", color=c, lw=1.1)
    ax.axvline(0, color=REF, lw=0.8); ax.axvline(1, color=REF, lw=0.8, ls=":")
    h = [Line2D([], [], color=C1, lw=1.4, label=T["unif"]), Line2D([], [], color=C2, lw=1.4, label=T["dir"]),
         Line2D([], [], color=INK2, lw=1.2, marker="o", ms=3, label=T["test"]),
         Line2D([], [], color=INK2, lw=1.1, ls="--", label=T["devp"])]
    ax.legend(handles=h, loc="lower right", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    lo = min(E[[f"{a}_{b}" for a in ("test", "dev") for b in ("uniform", "dir1")]].min())
    hi = max(E[[f"{a}_{b}" for a in ("test", "dev") for b in ("uniform", "dir1")]].max())
    ax.set_ylim(lo - 0.62 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_xlabel(T["eta"]); ax.set_ylabel(T["rho"]); ax.set_title(T["f2c"], fontsize=8.2)
    fig.tight_layout(w_pad=0.9)
    save(fig, lang, "fig2_sign_obstruction", PA)


# ============================================================================ Fig 3
def prior_group(p):
    if p == "uniform":
        return "g_unif"
    if p.startswith("seed"):
        return "g_seed"
    return "g_smooth"


def fig3(lang, PA):
    T = L[lang]
    P = PA.CONF["prior"]
    rule = P.rule
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.45), gridspec_kw=dict(width_ratios=[1.05, 1.15, 0.95]))
    ax = axs[0]
    d = (P.s("sel:AP_eta1") - rule).to_frame("d").join(P.M[["prior", "log10_kappa"]])
    for g, c in (("g_unif", C1), ("g_smooth", C2), ("g_seed", C3)):
        G = d[d.prior.map(prior_group) == g]
        ax.scatter(G.log10_kappa, G.d, s=8, color=c, alpha=0.65, lw=0, label=T[g])
    ax.axhline(0, color=REF, lw=0.8)
    ax.set_xlabel(T["kappa"]); ax.set_ylabel(T["dap"]); ax.set_title(T["f3a"], fontsize=8.2)
    ax.set_xlim(-0.6, 13.6); ax.set_ylim(-0.76, 0.33)
    ax.legend(loc="lower right", handletextpad=0.1, borderaxespad=0.15, markerscale=1.6, fontsize=6.6,
              labelspacing=0.35)
    ax = axs[1]
    g = PA.P2.groupby("prior")[["corr_r_with_inv_prior", "corr_r_with_ystar"]].mean().reindex(PA.PRIOR_ORDER)
    xs = np.arange(len(g))
    ax.plot(xs, g.corr_r_with_inv_prior, "-o", color=C2, ms=3.2, label=T["r_inv"])
    ax.plot(xs, g.corr_r_with_ystar, "-s", color=C1, ms=3.2, label=T["r_y"])
    ax.axhline(0, color=REF, lw=0.8)
    names = {"en": ["unif", "D10", "LN.5", "D3", "LN1", "D1", "LN2", "D.3", "5 seeds", "1 seed"],
             "gr": ["ομοιόμ.", "D10", "LN.5", "D3", "LN1", "D1", "LN2", "D.3", "5 σπόρ.", "1 σπόρ."]}[lang]
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=55, ha="right", fontsize=6.8)
    ax.set_ylabel(T["rank_corr"]); ax.set_title(T["f3b"], fontsize=8.2)
    ax.legend(loc="center", bbox_to_anchor=(0.49, 0.46), borderaxespad=0.2)
    ax = axs[2]
    taus = [0.0, 0.1, 0.5, 0.9, 1.0]
    for gname, c in (("g_unif", C1), ("g_smooth", C2), ("g_seed", C3)):
        ys = []
        for tau in taus:
            m = "sel:AP_eta1" if tau == 0 else f"sel:AP_eta1@tau{tau:g}"
            dd = (P.s(m) - rule).to_frame("d").join(P.M[["prior"]])
            ys.append(dd[dd.prior.map(prior_group) == gname].d.mean())
        ax.plot(taus, ys, "-o", color=c, ms=3.2, label=T[gname])
    ax.axhline(0, color=REF, lw=0.8)
    ax.set_xlabel(T["tau"]); ax.set_ylabel(T["dap"]); ax.set_title(T["f3c"], fontsize=8.2)
    fig.tight_layout(w_pad=0.7)
    save(fig, lang, "fig3_prior_uniformity", PA)


# ============================================================================ Fig 4
def fig4(lang, PA):
    T = L[lang]
    MS = METHOD_SHORT[lang]
    methods = ["base:rule_EW", "sel:AP_eta1", "sel:PB_rule_nominal", "sel:PBX_rule_best", "sel:CTRL_Cmean",
               "sel:CTRL_Cperm", "sel:CTRL_lin0.9", "oracle:scalar", "oracle:pernode"]
    role = {"base:rule_EW": REF, "oracle:scalar": REF, "oracle:pernode": REF, "sel:AP_eta1": C2,
            "sel:PB_rule_nominal": C2, "sel:PBX_rule_best": C2, "sel:CTRL_Cmean": C1, "sel:CTRL_Cperm": C1,
            "sel:CTRL_lin0.9": C1}
    fig = plt.figure(figsize=(6.3, 5.4))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.12], hspace=0.5, wspace=0.26)
    panels = [("test", T["p_test"]), ("real", T["p_real"]), ("scale", T["p_scale"]), ("prior", T["p_prior"])]
    ys = np.arange(len(methods))[::-1]
    for j, (p, ptitle) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        P = PA.CONF[p]
        vals = [P.mean(m) for m in methods]
        rule = vals[0]
        ax.axvline(rule, color=INK2, lw=0.6, ls=":")
        for y, v, m in zip(ys, vals, methods):
            ax.plot([v], [y], "o", ms=5.2, color=role[m], mec="white", mew=0.6, zorder=3)
            ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=6.2, color=INK2)
        ax.set_xlim(0.80, 1.0)
        ax.set_yticks(ys)
        ax.set_yticklabels([MS[m] for m in methods] if j == 0 else [], fontsize=7)
        ax.tick_params(axis="y", length=0)
        ax.set_title(ptitle, fontsize=8, loc="center", fontweight="normal")
        ax.grid(axis="y", color=GRID_C, lw=0.4)
        ax.set_xticks([0.8, 0.9, 1.0])
        if j == 0:
            ax.text(-0.02, 1.13, T["f4a"], transform=ax.transAxes, fontweight="bold", fontsize=8.8, ha="right")
    h = [Line2D([], [], color=c, marker="o", ls="", ms=5, label=T[k]) for k, c in (("role_ref", REF), ("role_law", C2), ("role_free", C1))]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.62, 1.005), handlelength=1.2)
    ax = fig.add_subplot(gs[1, :])
    cfgs = ["PBX_rule_best", "PBX_rule_unif", "PBX_rule_dir1", "PBX_rule_nominalB", "C_best", "B_best", "A_best",
            "B_unif", "C_dir1"]
    yy = np.arange(len(cfgs))[::-1]
    lo_lim, hi_lim = -0.10, 0.05
    for k, (p, c, ptitle) in enumerate((("test", C1, T["p_test"]), ("real", C2, T["p_real"]), ("scale", C3, T["p_scale"]))):
        for y, cfg in zip(yy, cfgs):
            r, _ = PA.mf_paired(p, cfg)
            yv = y + (1 - k) * 0.27
            d, lo, hi = r["diff"], r["lo"], r["hi"]
            if d < lo_lim:
                ax.text(lo_lim + 0.006 + (0.011 if k == 1 else 0.0), yv, f"{d:.2f}".replace("-", "\u2212"),
                        fontsize=5.8, color=c, va="center")
                ax.plot([lo_lim + 0.002], [yv], marker="<", color=c, ms=4)
                continue
            ax.plot([max(lo, lo_lim), min(hi, hi_lim)], [yv, yv], color=c, lw=1.2)
            ax.plot([d], [yv], "o", color=c, ms=3.8, label=ptitle if y == yy[0] else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_yticks(yy); ax.set_yticklabels([CFG_SHORT[lang][c] for c in cfgs], fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(T["dlaw"]); ax.set_title(T["f4b"])
    ax.legend(loc="lower right", ncol=3, borderaxespad=0.2)
    save(fig, lang, "fig4_modeb_and_controls", PA)


# ============================================================================ Fig 5
def fig5(lang, PA):
    T = L[lang]
    MS = METHOD_SHORT[lang]
    C = PA.T12
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.9))
    entries = [("calibrated rule [4]", "rule", C1), ("AP(eta=1), fixed point", "ap", C1),
               ("rule + anchored polish (K=1)", "pol", C1), ("rule + C-law on raw P (K=5)", "pbx", C2),
               ("rule + constant-alpha raw P (K=5)", "lin", C2), ("KNZ MP, 2 it.", "mp", REF)]
    offs = {"test": {"rule": (7, -10), "ap": (0, 7), "pol": (-7, 4), "pbx": (12, -9), "lin": (12, 8), "mp": (6, -2),
                     50: (6, -8), 200: (-6, 8), 1000: (5, -9)},
            "scale": {"rule": (8, -10), "ap": (6, -2), "pol": (-7, -10), "pbx": (-8, -10), "lin": (-8, -3), "mp": (6, -2),
                      50: (-6, -5), 200: (7, -10), 1000: (-6, -10)}}
    ticks = {"test": [0.5, 1, 2, 5, 10], "scale": [1, 2, 5, 10, 20, 50]}
    for ax, panel, title in ((axs[0, 0], "test", T["f5a"]), (axs[0, 1], "scale", T["f5b"])):
        mc = [C.loc[f"Monte Carlo, {k} worlds"] for k in (50, 200, 1000)]
        ax.plot([r[f"ms_{panel}"] for r in mc], [r[f"rho_{panel}"] for r in mc], "-o", color=C3, ms=4, lw=1.3)
        for k, r in zip((50, 200, 1000), mc):
            o = offs[panel][k]
            ax.annotate(T["mclab"].format(k=k), (r[f"ms_{panel}"], r[f"rho_{panel}"]), textcoords="offset points",
                        xytext=o, fontsize=6.5, color=INK2, ha="left" if o[0] >= 0 else "right")
        for key, short, col in entries:
            r = C.loc[key]
            o = offs[panel][short]
            ax.plot([r[f"ms_{panel}"]], [r[f"rho_{panel}"]], "o" if col != REF else "D", color=col, ms=4.5,
                    mec="white", mew=0.6, zorder=3)
            ax.annotate(MS[short], (r[f"ms_{panel}"], r[f"rho_{panel}"]), textcoords="offset points",
                        xytext=o, fontsize=6.3, color=INK2, ha="left" if o[0] >= 0 else "right",
                        arrowprops=dict(arrowstyle="-", color=REF, lw=0.4, shrinkA=0, shrinkB=2)
                        if abs(o[0]) > 9 else None)
        orc = C.loc["scalar oracle (needs ground truth)", f"rho_{panel}"]
        ax.axhline(orc, color=REF, ls=(0, (4, 2)), lw=0.9)
        ax.set_xscale("log")
        ax.set_xticks(ticks[panel]); ax.set_xticklabels([f"{t:g}" for t in ticks[panel]])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(T["ms_short"]); ax.set_ylabel(T["rho"]); ax.set_title(title, fontsize=8.2)
        xl = ax.get_xlim()
        if panel == "test":
            ax.text(xl[0] * 1.08, orc + 0.004, T["oracle"], fontsize=6.5, color=INK2)
        else:
            ax.text(xl[1] / 1.05, orc - 0.004, T["oracle"], fontsize=6.5, color=INK2, ha="right", va="top")
    h = [Line2D([], [], color=C1, marker="o", ls="", ms=4, label="EW diffusions"),
         Line2D([], [], color=C2, marker="o", ls="", ms=4, label=T["ens"]),
         Line2D([], [], color=C3, marker="o", ls="-", ms=4, label=T["mc"])]
    if lang == "gr":
        h[0].set_label("διαχύσεις EW")
    axs[0, 1].legend(handles=h, loc="lower right", borderaxespad=0.2)
    SW = PA.SW
    ax = axs[1, 0]
    g = SW.groupby("n")
    keys = [("ms_ensemble", T["ens"], C2, "-"), ("ms_mc50", "MC-50", C3, ":"), ("ms_mc200", "MC-200", C3, "-"),
            ("ms_mc1000", "MC-1000", C3, "--")]
    bins = [0, 160, 350, 700, 1400, 2100]
    SWb = SW.assign(nb=pd.cut(SW.n, bins))
    for col, lab, c, ls in keys:
        gg = SWb.groupby("nb", observed=True)
        x = gg.n.median().values
        ax.plot(x, gg[col].median().values, ls, color=c, marker="o", ms=3, lw=1.4, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000]); ax.set_xticklabels(["100", "200", "500", "1000", "2000"])
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(T["n"]); ax.set_ylabel(T["ms_short"]); ax.set_title(T["f5c"], fontsize=8.2)
    ax.legend(loc="upper left", borderaxespad=0.2)
    ax = axs[1, 1]
    for fam, c in (("er", C1), ("ba", C2), ("grid", C3), ("ws", C4)):
        G = SW[SW.family == fam].groupby("n").d_eq.median()
        ax.plot(G.index, G.values, "-o", color=c, ms=3, lw=1.3, label=T[f"fam_{fam}"])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log"); ax.set_xticks([100, 200, 500, 1000, 2000]); ax.set_xticklabels(["100", "200", "500", "1000", "2000"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(T["n"]); ax.set_ylabel(T["deq"]); ax.set_title(T["f5d"], fontsize=8.2)
    ax.legend(loc="upper left", ncol=2, borderaxespad=0.2)
    fig.tight_layout(h_pad=1.1, w_pad=1.0)
    save(fig, lang, "fig5_cost_and_monte_carlo", PA)


def main(PA=None):
    if PA is None:
        import make_paper_assets as PA
    for lang in ("en", "gr"):
        fig1(lang, PA)
        fig2(lang, PA)
        fig3(lang, PA)
        fig4(lang, PA)
        fig5(lang, PA)
    print("figures written:", sorted(p.name for p in PA.FIGDIR["en"].glob("*.pdf")))
