# Mixing-strategy study status

This study is additive to the published implementation. Immutable assets under
`results/`, `data/`, `paper/`, `figures/`, `figures_gr/`, `logs/`, and the
checkpoint/documentation files are not modified.

## 2026-10-31 — scaffold and S0 audit

Created branch `exp/mixing-study` and added the general mixing engine,
canonical configuration IDs, overflow-safe raw propagation, fusion methods,
baseline ports, and the staged runner scaffolding. No fresh-panel method has
been evaluated.

S0 completed on the existing panels with 4 configurations and 180 instances
per panel. Mean Spearman values were:

| panel | CTRL_lin0.9 | rule_EW | rule_WT | IRIE_exact |
|---|---:|---:|---:|---:|
| dev | 0.9037 | 0.8571 | 0.8104 | 0.8800 |
| test | 0.9006 | 0.8442 | 0.8113 | 0.8960 |
| prior | 0.9016 | 0.8568 | 0.8554 | 0.8906 |
| real | 0.9005 | 0.8402 | 0.8169 | 0.8475 |
| scale | 0.8863 | 0.8320 | 0.7594 | 0.8434 |

The required targets match to four decimals. The current-hybrid per-instance
values match the published controls on the checked dev instances exactly.
The vendored IRIE port agrees with the pinned external implementation to
`1e-12` on the checked instance. No fresh panel has been evaluated.

Fresh ground truth generation completed for `test2` (180 instances),
`scale2` (90), and `prior2` (150). The repository lacks the four required
`data_real/*.npz` snowball sources; `build_panels_v2.py` therefore omits those
unavailable specifications rather than substituting other graphs. Classic
`real2` specifications remain available.

The full `(edges, p, prior)` duplicate audit found three unavoidable classic
weighted-cascade repeats: that scheme is deterministic from the degree
sequence, so changing its probability seed cannot change the instance. They
are retained and explicitly reported rather than silently altering the
published probability scheme.

## 2026-10-31 — S2/S3/S4 and freeze

S2 evaluated 16,800 configurations on all 180 dev instances. S3 evaluated the
top-50 S2 survivors with the full beta/gamma refinement and dynamic-law
variants, 15,750 configuration rows per dev instance. S4 evaluated two
schedule configurations. The S3 winner was `EP`, constant `a=0.7`, three
Phase-B steps, excess embedding `gamma=16`, and rank-fusion `beta=0.7`
(mean dev rho 0.9133). The best S4 schedule was 0.9029 and therefore did not
qualify for W2. A 2,000-resample dev bootstrap gave top-1 frequencies 0.274
for W1 and 0.184 for W3. The freeze contains W1 and W3 only; no confirmatory
panel has been evaluated.
