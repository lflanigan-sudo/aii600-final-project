# Results reconciliation

Date: 2026-11-16. The selected model, the five predictors, the seed, the loss, and the chronological split were not changed.

## The figures the project now uses

| Statement | Where it is stated |
|---|---|
| 2024 booster pinball $3,168.73 is 20.30% below the linear model's $3,975.74 | `analysis.ipynb` final comparison table; `presentation.qmd`; decision log |
| None of the tested feature configurations improved the observed 2024 validation dollar pinball loss | Feature-experiment section; final decision; slides. This does not say an attribute can never help. |
| 2026 booster calibration is 23.80% against a 33.33% target, with pinball still below the linear model ($5,540.14 vs $6,537.68) | Closing section; slides; 2026-11-15 decision |
| The reported interval covers 78.43% of 2024 sales and 72.65% of 2026 sales, against 80%, after a $650.96 adjustment read from 2023 only | Closing section; slides. The 70.07% and 67.96% figures are other fits and are labeled as such. |
| Screening flags 269 linear and 344 booster lots; 203 flags change. A sale above the estimate is not a profit | Closing section; slides |
| The target is the recorded USD sale price, not a verified hammer | `final_audit/catalogue_price_audit.md`; new price-basis section in the notebook; `data/README.md`; 2026-11-16 decision |

## What was corrected

`presentation.qmd` had still led with the linear model against a brand-only q33 baseline ($3,975.74, a 9.25% reduction, and the 2025–26 Omega figure of 39.47%). The presented deck is the original custom `presentation.html`, with those slides rewritten to the 20.30% booster result, the feature-bundle result, the 23.80% 2026 calibration, and the two interval coverages. A Quarto render of `presentation.qmd` would replace that design, so the HTML was put back in the original layout.

The closing section of `analysis.ipynb` called the target a hammer. That final section now says recorded sale price and points at the catalogue audit. Earlier cells still use the word hammer. They are the history of the analysis. The final section is the one to present.

`scripts/closing_diagnostics.py` print lines now say recorded sale price. The calculations are unchanged.

## What was left in place

The decision log was not rewritten. The 2026-11-16 entry sits above the 2026-11-15 entries, which still say "hammer." A later entry can correct an earlier one, and the earlier wording stays.

IQN and CatBoost were not refit. Their losses in the comparison table are the saved script results.

`CHECKLIST.md` was not edited. Its ticks were made on `Nov11_team` by an earlier commit. This audit does not treat those ticks as a new claim about `main`.
