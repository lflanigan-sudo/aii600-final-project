# Reproducibility report

Date: 2026-11-16. Branch `notebook-on-nov11`, commit `585c301`, which is the commit already pushed to `Nov11_team`. Python 3.14.7.

## What was run

```text
python -m scripts.closing_diagnostics
```

This refits the frozen log-price gradient booster and the linear q33 model. It does not retune, and it does not rerun the IQN search, the CatBoost grid, or the feature search. Those recorded 2024 losses stay $3,207.60 and $3,321.24. A top-to-bottom execution of `analysis.ipynb` was not run. The feature-experiment and closing-diagnostic notebook cells still have empty saved outputs. The numbers below come from the script.

## Frozen settings the script uses

- Loss: dollar pinball at tau = 1/3, fit on log price.
- Predictors: `norm_model`, `case_body`, `movement`, `age_at_sale`, `brand`. Rare model names with fewer than 15 training sales become Other.
- Booster: learning rate 0.1, max depth 6, minimum leaf 20, l2 penalty 0, 112 trees, seed 20261024.
- Eligible file checksum expected by the scripts: `cf3b41d890af1feba9348fa9a73d954fcab193ed890a39044bd0355198fc5a8a`.
- Pinned stack in `requirements.txt`: pandas 3.1.0, numpy 2.5.4, matplotlib 3.11.2, scikit-learn 1.9.1, SciPy 1.18.1, catboost 1.2.10.

## Result of this run

| Check | Expected | This run |
|---|---:|---:|
| 2024 booster pinball | 3,168.7295 | 3,168.7295 |
| 2026 booster pinball | 5,540.14 | 5,540.1367 |
| 2026 booster share at or below q33 | 23.80% | 0.2380 |
| 2026 linear pinball | 6,537.68 | 6,537.6822 |
| 2026 linear share | 28.95% | 0.2895 |
| 2024 interval coverage after the $650.96 cutoff | 78.43% | 0.7843 |
| 2026 interval coverage, same cutoff | 72.65% | 0.7265 |
| Cutoff rank | 620 of 774 | 620, value 650.96 |

Screening counts on the 762 validation sales with a usable midpoint: linear flags 269, booster flags 344, both 205, linear-only 64, booster-only 139, neither 354, changed 203. Median recorded-price minus midpoint on the 139 booster-only sales: $2,018.07. Median booster q33 minus recorded price on those 139: $897.32. That median is not the $650.96 interval adjustment.

The same command does not print the 2024 linear pinball or the 2024 calibration share. Those two figures were printed by a second call that uses the same frozen fit, `python _spotcheck_lots.py`, and then that temporary script was deleted. It reported training of 2,490 sales through 2023-12-12, evaluation of 802 sales from 2024-01-30 to 2024-12-19, and no shared record ids. The 2024 linear pinball was 3,975.7439. The 2024 booster share at or below q33 was 0.3541, and the linear share was 0.3042. The reduction is (3,975.7439 − 3,168.7295) / 3,975.7439 = 20.30%. No expected figure disagreed with either run.
