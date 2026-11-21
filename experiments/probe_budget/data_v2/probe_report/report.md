# Difficulty probe report

problems=1988 steps=7791 non-constant-label problems=1513

| method | evaluated_level R² | MAE | share MAE (pp) | within-problem Spearman | top-1 hit |
|---|---|---|---|---|---|
| uniform | — | — | 5.806 | nan | 0.063 |
| global_mean | -0.000 | 0.810 | 5.806 | nan | 0.063 |
| position_mean | 0.218 | 0.703 | 5.442 | 0.508 | 0.586 |
| ridge F1 (n_seeds=1) | 0.584 | 0.488 | 4.806 | 0.669 | 0.783 |
| ridge F2 (n_seeds=1) | 0.460 | 0.564 | 5.303 | 0.621 | 0.746 |
| ridge F3 (n_seeds=1) | 0.559 | 0.503 | 4.751 | 0.680 | 0.782 |
| ridge F4 (n_seeds=1) | 0.524 | 0.525 | 4.901 | 0.646 | 0.765 |
| mlp F1 (n_seeds=3) | 0.582 ± 0.001 | 0.483 ± 0.002 | 4.641 ± 0.042 | 0.668 ± 0.006 | 0.778 ± 0.010 |
| mlp F2 (n_seeds=3) | 0.540 ± 0.000 | 0.507 ± 0.001 | 4.842 ± 0.021 | 0.636 ± 0.002 | 0.752 ± 0.008 |
| mlp F3 (n_seeds=3) | 0.551 ± 0.006 | 0.501 ± 0.001 | 4.649 ± 0.030 | 0.666 ± 0.007 | 0.778 ± 0.004 |
| mlp F4 (n_seeds=3) | 0.528 ± 0.005 | 0.516 ± 0.005 | 4.796 ± 0.064 | 0.643 ± 0.003 | 0.763 ± 0.002 |

Spearman/top-1 computed only on problems whose labels are not all equal.

## vs position-mean
- ridge F1: spearman 0.669 vs pos 0.508 (BEATS), top1 0.783 vs pos 0.586 (BEATS)
- mlp F1: spearman 0.668 vs pos 0.508 (BEATS), top1 0.778 vs pos 0.586 (BEATS)
- ridge F2: spearman 0.621 vs pos 0.508 (BEATS), top1 0.746 vs pos 0.586 (BEATS)
- mlp F2: spearman 0.636 vs pos 0.508 (BEATS), top1 0.752 vs pos 0.586 (BEATS)
- ridge F3: spearman 0.680 vs pos 0.508 (BEATS), top1 0.782 vs pos 0.586 (BEATS)
- mlp F3: spearman 0.666 vs pos 0.508 (BEATS), top1 0.778 vs pos 0.586 (BEATS)
- ridge F4: spearman 0.646 vs pos 0.508 (BEATS), top1 0.765 vs pos 0.586 (BEATS)
- mlp F4: spearman 0.643 vs pos 0.508 (BEATS), top1 0.763 vs pos 0.586 (BEATS)
