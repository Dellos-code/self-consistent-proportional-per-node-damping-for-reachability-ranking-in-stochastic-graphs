Mean Spearman rho on the whole panel against median single-process, end-to-end wall clock (ms) over one uniform-prior instance per family and probability range.

| method | rho_scale | rho_test | ms_scale | ms_test |
|---|---|---|---|---|
| KNZ MP, 2 it. | 0.730 | 0.754 | 1.346 | 0.805 |
| Monte Carlo, 50 worlds | 0.785 | 0.802 | 3.328 | 0.588 |
| AP(eta=1), fixed point | 0.827 | 0.832 | 17.350 | 7.550 |
| rule + anchored polish (K=1) | 0.828 | 0.840 | 3.513 | 2.007 |
| calibrated rule [4] | 0.832 | 0.844 | 3.828 | 2.189 |
| rule + C-law on raw P (K=5) | 0.873 | 0.895 | 4.084 | 2.332 |
| rule + constant-alpha raw P (K=5) | 0.886 | 0.901 | 4.574 | 2.737 |
| scalar oracle (needs ground truth) | 0.894 | 0.904 |  |  |
| Monte Carlo, 200 worlds | 0.887 | 0.906 | 12.549 | 1.676 |
| Monte Carlo, 1000 worlds | 0.939 | 0.945 | 59.945 | 6.234 |