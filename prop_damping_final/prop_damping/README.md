# Self-Consistent Proportional Per-Node Damping for Reachability Ranking in Uncertain Graphs

Code, instance data, results and manuscripts for *Self-Consistent Proportional
Per-Node Damping for Reachability Ranking in Uncertain Graphs: Exact Theory,
Dynamical Obstructions, and Hybrid Dual-Operator Polishing* (Ioannis
Konstantellos, Computer Engineering and Informatics Department, University of
Patras).

| document | content |
|---|---|
| `paper/paper_en.pdf` | the paper (English) |
| `paper/paper_gr.pdf` | the paper (Greek) |
| `DOCUMENTATION_EN.pdf`, `DOCUMENTATION_GR.pdf` | technical documentation: architecture, full proofs of Theorem 1 and Propositions A–F and 2′, module-by-module walkthrough, complexity analysis, reproduction guide |
| `CHECKPOINT_1_REPORT.md` | the Phase-1 report, regenerated from the final `results/` (see its revision note) |

Every number in the papers and in the documentation is a LaTeX macro in
`paper/generated/macros.tex`, and every table a macro in
`paper/generated/tables_en.tex` / `tables_gr.tex`; both are written by
`make_paper_assets.py` from `results/`. No number is typed by hand.

## Layout

| path | content |
|---|---|
| `prop_damping_lib.py` | the library: instances, priors, real-network loading, operators, Monte-Carlo and exact ground truth, metrics, diffusions, message passing, Realizations A/B/C, anchored law AP(η), polishing, oracles |
| `build_panels.py` | generates the five panels and their two independent Monte-Carlo ground-truth samples |
| `experiments.py` | `grid` stage (full search on dev and test) and `confirm` stage (references, oracles and frozen configurations on the held-out panels) |
| `select_configs.py` | re-derives the frozen configurations from the dev results by the stated rules and verifies `results/selected.json` (never overwrites it) |
| `analysis.py` | loading of stage results, configuration means, paired statistics (Wilcoxon, bootstrap), Holm correction |
| `controls.py`, `mf_controls.py` | matched controls: per-node law replaced by a constant, its per-step mean (mean-field twin) or a permutation |
| `sign_condition.py`, `prop2_stats.py` | sign-obstruction diagnostic (Proposition E); Proposition-2 statistics on ground truth |
| `explore_rawP.py` | exploratory (not pre-registered) raw-P propagation |
| `mc_baseline.py`, `bench_runtime.py` | Monte-Carlo k-world baselines; single-process timings |
| `paper_extras.py` | dynamics traces, operator complementarity, size sweep |
| `make_checkpoint_tables.py`, `write_checkpoint_report.py` | the Phase-1 tables and report |
| `make_paper_assets.py`, `paper_figures.py` | every macro, table and figure of the papers (English and Greek) |
| `finish_phase1.sh` | Phase-1 post-processing helper (controls on the scale panel, raw-P exploration, timings, tables, report) |
| `tests/test_foundations.py` | Theorem 1, Monte Carlo vs exact enumeration, message passing with zero iterations ≡ P_esc, diffusion fixed point, collapse after one step |
| `tests/test_dynamics.py` | Propositions A–E numerically |
| `data/<panel>/NNNN.npz` | instance (edges, p, x) with ground truth `yA` (evaluation), `yB` (oracle fitting), `yA2`; `data/<panel>/specs.json` lists the specifications |
| `results/` | per-instance results, controls, diagnostics, timings, the frozen selection `selected.json` and its digest `selected.sha256` |
| `results/tables/` | the Phase-1 tables (`.csv` and `.md`) |
| `paper/` | `paper_en.tex`, `paper_gr.tex`, the generated assets in `paper/generated/` and the compiled PDFs |
| `figures/`, `figures_gr/` | the five figures with English and Greek labels |
| `logs/` | progress logs of the long stages |
| `data_real/` | not shipped: `cora.npz`, `citeseer.npz`, `amazon_electronics_photo.npz`, `ms_academic_cs.npz` from github.com/shchur/gnn-benchmark (`data/npz`); only needed to rebuild the real panel, whose sampled subgraphs are already in `data/real/` |

## Reproduce

Python 3.11 or later (results produced with Python 3.13, NumPy 2.5, SciPy 1.18,
pandas 3.0, NetworkX 3.7, Matplotlib 3.11); XeLaTeX (TeX Live 2023) with the
FreeSerif, FreeSans and DejaVu Sans Mono fonts for the documents.

```bash
pip install numpy scipy pandas networkx matplotlib tabulate
python tests/test_foundations.py
python tests/test_dynamics.py
python build_panels.py                       # 688 instances, about 40 s on 2 cores
python experiments.py grid dev,test          # about 13 min on 2 cores
python controls.py dev                       # dev-only control grid
python select_configs.py                     # verifies the frozen selection
sha256sum results/selected.json              # must equal results/selected.sha256
python experiments.py confirm test,prior,real,scale   # about 40 min on 2 cores
python mf_controls.py dev,test,real,scale
python sign_condition.py && python prop2_stats.py
python explore_rawP.py dev,test && python explore_rawP.py real,scale
python mc_baseline.py test,real,scale
python bench_runtime.py                      # timings depend on the machine
python paper_extras.py traces && python paper_extras.py complement && python paper_extras.py sweep
python make_checkpoint_tables.py && python write_checkpoint_report.py
python make_paper_assets.py                  # macros, tables and figures (--no-figures skips figures)
cd paper
xelatex paper_en.tex                         # three times (cross-references)
xelatex paper_gr.tex                         # three times
cd ..
xelatex DOCUMENTATION_EN.tex                 # three times (table of contents and paper references)
xelatex DOCUMENTATION_GR.tex                 # three times
```

The stored `data/` and `results/` already contain every output, so any step
can be rerun in isolation. All randomness is seeded from instance names
(`stable_seed`), so every run reproduces the stored numbers; only the timings
depend on the machine. Real networks are read with `allow_pickle=False`. The
documentation takes the numbers of the paper's tables, figures and sections
from `paper/*.aux`, so the papers are compiled first.

## Conventions

* Primary metric: Spearman ρ between a method's scores and `yA`; scores equal to
  within 1e-12 (relative) are ties; a constant score vector has ρ = 0.
* Oracles are always fitted on `yB` and scored on `yA`.
* `EW` is the escape-scaled pull operator of [4]; the calibrated rule is
  α = clip(0.502 + 0.558·P̄_esc, 0.4, 1.0) with the normalised power iteration.
* `T = 1000000` in result files means "run to the fixed point".
* Selection uses the development panel only; the 21 frozen configurations were
  fixed (SHA-256 recorded in `results/selected.sha256`) before any held-out run.
