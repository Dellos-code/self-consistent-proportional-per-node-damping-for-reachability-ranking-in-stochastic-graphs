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
