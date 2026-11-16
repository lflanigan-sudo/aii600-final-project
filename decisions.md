# Decisions

This file is the history of the project. Append an entry when a choice changes the work: the quantity, the data, the split, the model, the loss, or the action. Leave old entries in place. A later entry can reverse an earlier one. If a result depends on a random draw, record the seed under Why. The split entry records the cutoff date, or the seed and the fraction. The pull request that merges the code names this entry.

## 2026-11-15: 2026 calibration, the screening gaps, and one interval

**Decision.** Keep the five-feature gradient booster as the predictive model and the linear q33 model as the interpretable reference. Do not switch models to repair the 2026 calibration. Report one interval: q10 and q90 fit on sales before 2023, with the 650.96 dollar cutoff read from 2023 and applied unchanged to later years. Keep the catalogue screen as a proposed investigation rule. A hammer above the estimate is not a profitable purchase.

**Why.** `scripts/closing_diagnostics.py` refit the frozen booster on the 4,099 sales through 2025-12-17 and scored the 563 sales from 2026-01-28 to 2026-07-28. Pinball was $5,540.14 for the booster and $6,537.68 for the linear model. The share at or below q33 was 23.80% and 28.95%. The same script reproduced the 2024 booster pinball of $3,168.73.

The too-low quantile is not confined to one predicted-price band. With training cuts at $3,072.21 and $7,994.89, the 2026 booster shares are 25.68%, 22.01%, and 23.48%. It is concentrated by brand. Omega is 31.39% (n=137) and its pinball is $1,057.89 against $1,276.03. Cartier is 21.31% (n=183), pinball $6,110.78 against $6,623.49. Rolex is 21.40% (n=243), pinball $7,637.42 against $9,439.50. The linear Rolex share is 31.69%. Watches with no model name are 16.19% on 105 sales. Oyster Perpetual Date, n=18, is the listed slice where the linear pinball is lower ($5,561.85 versus $5,856.17).

On the 762 validation sales with a catalogue range, the midpoint flag still changes 203 decisions: 139 booster-only and 64 linear-only. The share that sold above the midpoint is 87.32%, 82.81%, 79.14%, and 74.29% for both, only linear, only booster, and neither. Those rates do not separate the groups. The median hammer-minus-midpoint is $2,018.07 for the newly flagged watches, $825.84 for the dropped watches, and $1,169.33 for the watches neither model flags. The newly flagged p10 is −$1,424.00. The median booster q33 on those 139 watches is $897.32 above the hammer, which is the costly direction of the 2:1 loss and is still not a profit. The low-estimate rule changes 193 flags, and the neither group has a higher median excess over the low estimate ($4,517.82) than the booster-only group ($4,099.20). The midpoint is the benchmark to keep. Neither benchmark validates a purchase.

The reported interval fits q10 and q90 on the 1,716 sales through 2022-12-14. The 2023 score uses 774 sales from 2023-02-02 to 2023-12-12. Rank ceil((774+1)×0.8) = 620, and that order statistic is $650.96, added to each end. 2024 coverage is 78.43%. The same cutoff covers 72.65% of 2026 sales: Cartier 72.13%, Omega 80.29%, Rolex 68.72%. The cutoff was not refit on 2026. The 70.07% and 67.96% figures are other fits and are not this interval. q33 is unchanged.

The IQN search and the CatBoost grid were not refit in this pass. Their recorded 2024 losses remain $3,207.60 and $3,321.24.

**Alternative.** Replace the booster with the linear model for Rolex screening because the linear 2026 Rolex share is near one third. That was rejected for the project model because the booster's Rolex pinball is lower in 2026 and its overall pinball is lower in every fixed-setting year. Report the raw 70.07% band as the project interval because it comes from the same training window as the selected q33. That was rejected because those tails were scored on the selection year, while the time-ordered cutoff is read from an earlier year and can be applied forward without being re-estimated.

**Revisit.** Reconsider the booster as the screening quantile for Rolex or Cartier if a later year repeats a share near 21% while the linear share stays near one third and the team decides that calibration outranks the pinball gap. Reconsider the model itself if a later window puts the booster's dollar pinball above the linear model's. Re-estimate the interval cutoff only on a new calibration year that was not used to choose it.

## 2026-11-15: Additional watch attributes, and the five-feature booster stays

**Decision.** Do not add papers, box, cleaned case material, case diameter, gold karat, gemstones, production year, or the cleaned model and movement fields to the gradient booster. Keep the five predictors and the frozen settings.

**Why.** `scripts/feature_experiments.py` first refit the existing design. The 2024 dollar pinball was $3,168.73, matching the recorded booster. The same five columns in the experiment builder matched that figure. Settings stayed at log price, learning rate 0.1, max depth 6, minimum leaf 20, l2 penalty 0, 112 trees, and seed 20261024. No group was retuned. The assigned 4,662 sales were unchanged. The cleaned file on disk is `data/expanded_pull_eligible_cleaned.csv`, 4,664 rows and 278 columns. Its prices match the eligible target. Two cleaned rows are the documented omissions and were not added back.

Papers and box are not columns in that cleaned file. They come from `expanded_pull_eligible_feature_candidates_v2.csv`, extracted from `item_title` and `item_description`. A blank stays "Not stated" rather than "no." Precious-stones is the same yes/blank flag as gemstones, so it was not entered twice. Cleaned case material replaces `case_body` in groups C and F because it is a finer label of the same metal, not a second metal field. Diameter is the first number parsed from `norm_case_diameter_cleaned`, with a missing indicator. It matches `case_measurement_mm` on 1,890 of 1,900 rows where both exist. Gold karat and `production_year_parsed` are numeric, with missing indicators. Production year is the year already used to compute age. Cleaned movement is identical to the raw movement column and replaces the audited movement, after collapsing levels with fewer than 15 training sales. Designer equals brand on all 4,662 assigned sales and was excluded. Clarity, cut, and carats were excluded because they are empty or nearly empty. `modeling_eligibility` and `norm_usd_price_decimal_cleaned` were excluded. The file does not record when a listing field was published, so every added attribute is flagged rather than treated as a verified pre-auction input.

On the same 802 sales, dollar pinball was $3,168.73 for the baseline, $3,201.14 with papers and box, $3,312.14 with the physical specifications, $3,211.26 with gemstones and production year, $3,243.87 with the cleaned model and movement, and $3,351.15 with all of those together. Every date-clustered interval for the paired difference includes zero, and every point estimate is a loss. The expanded models were worse on more sales than they were better. Group D lowered Cartier pinball from $4,285.05 to $4,265.88 and Omega from $1,222.49 to $1,206.72, and raised Rolex from $3,318.31 to $3,471.53. Overall q33 calibration moved from 35.41% to between 36.28% and 38.28%. The high predicted-price tertile, above a baseline prediction of $7,346.78, is where each expanded group added loss.

Each evaluation year was fit only on earlier sales. The baseline had the lowest pinball in 2023 and 2024. Group F was lower in 2022 ($5,349.68 versus $5,426.97) and higher afterward. Group E was lower in 2025 ($5,445.51 versus $5,505.05) and higher in the other four years. These windows are exploratory. The 2024, 2025, and 2026 sales were already in the development file.

No expanded group lowered 2024 pinball, so the q10–q90 check stayed on the five-feature model. Holding q10 and q90 at 112 trees covers 67.96% of the 802 prices, with a median width of $11,625.04 and crossings on 3.74% of sales. That is not the earlier 70.07% band, whose tail models chose their own tree counts. The separate pre-2023 fit, with the 2023 cutoff of $650.96, again covers 78.43% of 2024 prices. That cutoff was not used to choose features.

**Alternative.** Add the combined set because it had the lowest 2022 pinball, or add the cleaned model and movement because they had the lowest 2025 pinball. Both were rejected. Neither gain appears in the 2024 selection window, where every added group raised dollar pinball, and neither gain repeats across the other years.

**Revisit.** Add an attribute if a later window, fit only on earlier sales, reduces q33 dollar pinball, the date-clustered interval for that reduction stays above zero, and the field is confirmed to have been on the listing before the hammer. A lower pinball in one brand or one predicted-price band is not enough while the overall pinball rises.

## 2026-11-15: Freeze the comparison

**Decision.** Freeze the analysis. The selected predictive model is the log-scale gradient booster. The linear q33 model remains the interpretable reference. The IQN and CatBoost are not selected. This reverses the earlier "do not replace the linear model yet" only for the predictive candidate. The linear equation stays the model the team can read coefficient by coefficient.

**Why.** Seed 20261024. Predictors are brand, norm_model, case_body, movement, and age_at_sale. Model names with fewer than 15 training sales are collapsed to Other for the linear model, the IQN, and the booster. CatBoost's search kept those names. Blank categories are the string Missing, and the booster leaves missing age as missing. Training is the 2,490 sales before 2024-01-01, last sale 2023-12-12. Selection is the 802 sales in 2024, from 2024-01-30 through 2024-12-19. The primary metric is mean dollar pinball at tau = 1/3.

The booster settings are learning rate 0.1, max depth 6, minimum leaf 20, l2 penalty 0, and 112 trees, fit on the log price in `scripts/fit_gbqr.py`. Its 2024 dollar pinball is $3,168.73, 20.30% below the linear model's $3,975.74. The linear model is `QuantileRegressor` with quantile 1/3, alpha 0.01, and solver highs. The brand q33 is $4,380.90. The original unadjusted q10–q90 coverages on these 802 sales are 76.81% for the linear model, 74.94% for the IQN, 70.07% for the booster, and 75.06% for CatBoost. q33 calibration on the same sales is 30.42%, 28.55%, 35.41%, and 32.92%.

The IQN is not selected. Its 2024 pinball is $3,207.60. The booster's paired advantage averages $38.87, and a 2,000-draw bootstrap over the 64 auction dates gives −$59.46 to $153.15, which includes zero. Quantiles cross on 103 of 802 sales (12.84%). Cartier is 24.05% and the middle predicted-price band is 22.55%, both outside the 5-point bar around 33.33%.

CatBoost is not selected. Its selected 2024 pinball is $3,321.24, from the log scale, learning rate 0.05, depth 4, l2 leaf regularization 3, and 399 trees in `scripts/fit_catboost.py`. That is above both the booster and the IQN. Keeping rare model names lowered CatBoost's loss relative to a collapsed refit at the same settings ($3,440.45) and still left it behind the booster, which collapses those names. Cartier is 25.00% and fails the 5-point bar. Rolex is 38.52%.

`scripts/finalize_review.py` checks the fixed-setting windows. Each evaluation year is fit only on earlier sales, and the last training sale is before the first evaluation sale: 2021-12-15 before 2022-01-26, 2022-12-14 before 2023-02-02, 2023-12-12 before 2024-01-30, 2024-12-19 before 2025-01-30, and 2025-12-17 before 2026-01-28. Booster pinball stays below the linear pinball in every window. The booster share at or below q33 is 39.34% in 2022, 41.21% in 2023, 35.41% in 2024, 32.47% in 2025, and 23.80% in 2026. The 2026 rate means the predicted quantile sat too low, so a 10% screen would flag fewer watches. The linear share that year is 28.95%, closer to one third, while its pinball is higher ($6,537.68 versus $5,540.14). The 2025 and 2026 sales are already in the development file. Of the 1,370 chronological test lots, 1,088 were used in earlier development, so those years are a sensitivity check and are not an untouched test set.

The screening benchmark is the catalogue range `sale_estimates_low_usd_num` and `sale_estimates_high_usd_num`. On the full file, 4,391 of 4,664 rows have a positive low and high, the low is never above the high, and the hammer equals the low on 2 rows and the high on 3. The file does not record when the estimate was written. The realized price is not used to build the benchmark. On the 762 validation sales with a usable range, a flag means predicted q33 is at least 10% above the midpoint. The linear model flags 269 and the booster flags 344, so the flag changes on 203 sales: 139 booster-only and 64 linear-only. Among booster-only flags, 79.14% sold above the midpoint and the median hammer was $2,018.07 above it. Among linear-only flags, 82.81% sold above it and the median gap was $825.84. The 354 watches neither model flagged also sold above the midpoint 74.29% of the time. The hammer is used only to describe those groups. This is a proposed application of the existing 10% rule, not a validated purchasing strategy. The low-estimate alternative changes 193 flags and is a weak screen, because every group sold above the low estimate at least 92% of the time. Forty validation sales have no usable range and receive no flag.

The 78.43% interval is a separate time-ordered procedure. q10 and q90 are fit on the 1,716 sales before 2023. The additive cutoff is read from the 774 sales in 2023, rank 620, $650.96, and applied to 2024. Coverage of that pre-2023 fit moves from 65.09% to 78.43%, and the median width moves from $11,099.32 to $12,401.25. By brand, calibrated coverage and median width are Cartier 79.11% and $15,990.61, Omega 78.82% and $5,146.71, and Rolex 77.39% and $21,473.38. Omega's raw coverage on this pre-2023 fit was 51.23%, so the same dollar cutoff moves Omega more than Rolex. This band is not the original booster interval, whose 2024 coverage is 70.07%. The cutoff does not change q33, and coverage was not optimized at the expense of q33 pinball.

`requirements.txt` pins pandas 3.1.0, numpy 2.5.4, matplotlib 3.11.2, scikit-learn 1.9.1, SciPy 1.18.1, and catboost 1.2.10. The fits live in `scripts/fit_gbqr.py`, `scripts/audit_gbqr.py`, `scripts/fit_catboost.py`, and `scripts/finalize_review.py`.

**Alternative.** Select the IQN or CatBoost because their unadjusted interval coverage, 74.94% and 75.06%, is higher than the booster's 70.07%, or keep the linear model as the only reported predictor because every brand still misses the 5-point bar. Both were rejected. Dollar pinball at q33 is the primary metric, and the booster has the lowest recorded 2024 loss. The linear model remains available as the interpretable reference.

**Revisit.** Reopen the freeze if sales that were not used to choose the 112-tree settings put another model's q33 pinball below the booster's, or if the team decides the brand calibration misses or the 70.07% raw interval outrank the pinball reduction. Rejecting the IQN would be wrong if those unused sales also bring its crossing rate low enough to report q10 and q90 as a band. Rejecting CatBoost would be wrong on the same pinball terms.

## 2026-11-15: CatBoost quantile regression

**Decision.** Keep the linear q33 model as the benchmark. Record CatBoost as a challenger behind the gradient booster and the IQN.

**Why.** CatBoost 1.2.10 fits separate quantile objectives, in `scripts/fit_catboost.py`. The search keeps rare model names. Eight settings on the log price and eight on the dollar price were scored by dollar pinball on the 802 sales from 2024. Tree count was stopped on a 20% training holdout, seed 20261024, then refit on all 2,490 training rows. The 2025–2026 partition was not loaded.

The lowest loss in the search is $3,321.24, from the log scale, learning rate 0.05, depth 4, l2 leaf regularization 3, and 399 trees. The best dollar-scale fit is $3,328.75 and made one nonpositive prediction. The linear model is $3,975.74 and the brand q33 is $4,380.90. The mean advantage over the linear model is $654.50. A 2,000-draw bootstrap over the 64 auction dates gives $423.34 to $931.31. Rolex is 61.44% of that gain. The recorded IQN loss is $3,207.60 and the recorded gradient-booster loss is $3,168.73, so CatBoost is behind both on this selection sample. Every log-scale setting landed between $3,321.24 and $3,396.74. The selected fit stopped at 399 of the 400-tree cap. Collapsing rare model names at these same settings raises the loss to $3,440.45, so keeping the rare names helped CatBoost. It did not overtake the booster, which collapses those names.

The selected fit puts 32.92% of validation prices at or below q33. Cartier is 25.00%, Omega 37.44%, and Rolex 38.52%. Under the 5-point bar, the overall rate and Omega pass. Cartier fails, and Rolex misses the bar by a small margin. Separate q10 and q90 fits cover 75.06% of prices, against a nominal 80%. The share at or below q10 is 13.47%, and the share at or below q90 is 88.53%. The three quantiles cross on 3 sales (0.37%).

**Alternative.** Adopt CatBoost because native categories beat its own collapsed fit and the date-clustered interval against the linear model lies above zero. That was rejected because the selected loss is above the recorded booster and IQN losses, Cartier misses the 5-point bar, and the 80% band covers 75.06%.

**Revisit.** Reconsider CatBoost if a frozen comparison on sales that were not used to choose these settings puts it ahead of the booster, the brand rates move inside the 5-point bar, and the q10–q90 coverage is close enough to 80% to report the band.

## 2026-11-15: Booster audit, and the linear model stays

**Decision.** Do not replace the linear q33 reference model yet. Keep the log-scale gradient booster as a challenger with its settings frozen.

**Why.** The frozen settings are a log-price quantile booster, learning rate 0.1, max depth 6, minimum leaf 20, no l2 penalty, and 112 trees, seed 20261024. On the 802 sales from 2024 its dollar pinball is $3,168.73, against $3,975.74 for the linear model. That is a 20.30% reduction, $807.01 per watch. The same run reproduces the recorded IQN pinball of $3,207.60. `scripts/audit_gbqr.py` produced the checks below. scikit-learn 1.9.1.

The training pinball of $4,207.74 is not a calculation error and it is not the usual sign of overfitting. Mean pinball is exactly one third of the 2:1 mean loss on both partitions. The 2024 median price is $7,321.72, above the training median of $5,988.64. The training maximum is $3,141,173 and the 2024 maximum is $308,205. The five highest training prices, $3,141,173, $1,521,792, $1,330,320, $985,904, and $885,583, account for 24.77% of the training pinball sum. Without those five sales the training pinball is $3,171.86, in line with the 2024 figure. Preprocessing does not explain the gap: 203 of 802 validation model names are collapsed to Other, 165 training ages and 73 validation ages are missing, and one validation category cell is absent from training. The booster minimizes pinball on the log price. The IQN minimizes pinball on the dollar price. Both scores above are dollar pinball.

The booster’s advantage over the IQN averages $38.87 per watch. A 2,000-draw bootstrap that resamples the 64 auction dates gives −$59.46 to $153.15, which includes zero. The booster has the lower loss on 448 sales and the IQN on 354. By brand the mean IQN-minus-booster difference is $150.93 for Cartier, $58.97 for Omega, and −$100.68 for Rolex. On predicted-price tertiles of the linear training q33, at $3,186.20 and $5,633.28, the booster’s advantage is $68.24, $47.51, and $4.44. Among model names with at least 15 validation sales, the IQN is better on Submariner by $586.18 (31 sales) and the booster is better on Datejust by $249.71 (40 sales) and on Cosmograph by $377.71 (21 sales). That is not a decisive winner.

A sensitivity run keeps these settings fixed and does not retune them on the result. The booster’s pinball is lower than the linear model’s in every window: 2022, $5,426.97 versus $5,860.81; 2023, $3,611.28 versus $4,307.70; 2024, $3,168.73 versus $3,975.74; 2025, $5,505.05 versus $6,339.90; 2026, $5,540.14 versus $6,537.68. The share of prices at or below q33 is 39.34%, 41.21%, 35.41%, 32.47%, and 23.80% in those windows. The 2026 rate is far below one third. These windows are sensitivity results. The file, including 2025–2026, was already used in development.

q33 calibration still fails the 5-point bar by brand: Cartier 27.53%, Omega 41.87%, and Rolex 39.58%, against a target of 33.33% and a passing overall rate of 35.41%. The q10–q90 band covers 70.07% of the 802 validation prices, against a nominal 80%, and the quantiles cross on 2.62%. The shortfall is the tails, not the crossings. 16.96% of validation prices fall at or below q10, against a 10% target, and 87.03% fall at or below q90, against 90%, so both ends sit too far in. Training coverage of the same band is 79.32%. Omega’s validation coverage is 62.56%. A separate interval procedure fits q10 and q90 on sales before 2023 with the same 112 trees, reads the additive dollar cutoff from the 774 sales in 2023, and applies it to 2024. The cutoff is $650.96. Coverage on 2024 moves from 65.09% for that pre-2023 fit to 78.43%, and the median width moves from $11,099.32 to $12,401.25. That adjustment does not change the q33 prediction, and it was not used to select the model.

**Alternative.** Replace the linear model with the booster because the 20.30% pinball reduction is repeated, without retuning, in every earlier-to-later window. That was rejected for now. The date-clustered interval against the IQN includes zero, every brand misses the 5-point q33 bar, and the nominal 80% band covers 70.07%.

**Revisit.** Replace the linear model if a frozen comparison on sales that were not used to choose these settings still shows the pinball advantage, the brand q33 rates are inside the 5-point bar or the team adopts a different bar for a stated reason, and the reported interval, raw or with the separate 2023 cutoff, is close enough to its nominal coverage to use.

## 2026-11-15: Gradient-boosted quantile regression

**Decision.** Keep the linear q33 model as the benchmark. Record one gradient-boosted quantile regression as a challenger, not a replacement.

**Why.** The booster is scikit-learn 1.9.1 `HistGradientBoostingRegressor` with quantile loss, in `scripts/fit_gbqr.py`. It uses the same five predictors, the same 2,490 training sales, and the same 802 validation sales. Seed 20261024. The 2025–2026 partition was not loaded. Sixteen settings were fit on the log price and sixteen on the dollar price. Stopping used a 20% holdout drawn from training only; the selected tree count was then refit on all 2,490 training rows. The reported validation loss is the best loss in that search.

Dollar pinball on the 802 sales was $4,380.90 for the brand q33, $3,975.74 for the linear model, $3,202.98 for the best dollar-scale booster, and $3,168.73 for the best log-scale booster. The log-scale winner used learning rate 0.1, max depth 6, minimum leaf 20, no l2 penalty, and 112 trees. Its training dollar pinball was $4,207.74. Every log-scale setting landed between $3,168.73 and $3,336.13, so the gain over the linear model is not one cell of the grid. The best dollar-scale fit was $34.25 worse than the best log-scale fit and made one nonpositive validation prediction. The recorded IQN pinball on these same sales is $3,207.60. That comparison is also on the selection sample.

The selected log booster put 35.41% of validation prices at or below q33, against 33.33% and the linear model's 30.42%. By brand: Cartier 27.53% (linear 19.94%), Omega 41.87% (linear 43.35%), Rolex 39.58% (linear 32.86%). Under the 5-point bar from the 2026-11-14 entry, the overall rate passes and all three brands fail. The mean paired advantage over the linear model is $807.01. A 2,000-draw bootstrap that resamples the 64 auction dates gives $425.72 to $1,269.97. Rolex is 69.22% of the summed gain. The booster loss is higher on 287 of the 802 sales. Separate q10 and q90 fits that reuse the q33 settings cover 70.07% of prices, against a nominal 80%, and the three quantiles cross on 21 sales (2.62%).

**Alternative.** Adopt the log-scale booster because every searched setting beat the linear validation loss and the date-clustered interval for the gain lies above zero. That was rejected because the settings were chosen on these 802 sales, all three brands miss the 5-point bar, and the 80% band covers 70.07%.

**Revisit.** Replace the linear model only if the pinball advantage remains on sales that were not used to choose these settings, the brand rates move inside the 5-point bar or the team adopts a different bar for a stated reason, and the q10–q90 coverage is close enough to 80% to report the band as an interval.

## 2026-11-15: Gradient-Boosted Quantile Regression Evaluation

**Decision.** Retain the regularized linear q33 regression as the reference model. Record gradient-boosted quantile regression as the leading challenger by 2024 validation dollar pinball loss, alongside the previously evaluated Implicit Quantile Network (IQN). Do not replace the reference model until the booster's calibration, interval coverage, and temporal robustness have been evaluated further.

**Why.** We evaluated gradient-boosted quantile regression using the existing chronological split: 2,490 training sales from 2021–2023 and 802 validation sales from 2024. We compared 32 model settings using log-price and dollar-price targets.

The selected model uses log auction price, a learning rate of 0.1, maximum tree depth of 6, minimum of 20 observations per leaf, no L2 regularization, and 112 boosting trees.

On the same 802 validation sales, average dollar pinball loss was:

| Model | Dollar pinball loss |
|---|---:|
| Brand-q33 baseline | $4,380.90 |
| Linear q33 regression | $3,975.74 |
| IQN | $3,207.60 |
| Best dollar-scale booster | $3,202.98 |
| Best log-scale booster | $3,168.73 |

The selected booster reduced validation dollar pinball loss by $807.01 (20.30%) relative to linear regression and by $38.87 (1.21%) relative to IQN. All tested log-scale settings achieved lower validation loss than the linear reference model, suggesting the improvement was not limited to a single hyperparameter combination.

**Calibration and

## 2026-11-15: Model Comparison, Feature Expansion, and Next Steps

**Decision.** Retain the regularized linear q33 regression as the reference model. Preserve the three-value movement refit and Implicit Quantile Network (IQN) as evaluated alternatives. Do not add recovered specifications or additional predictors to the reference model until their data quality and predictive contribution have been tested.

**Why.** All model comparisons used the same 802 sales from the 2024 chronological validation period. The brand-q33 baseline achieved an average dollar pinball loss of $4,380.90, the original linear q33 model $3,975.74, the three-value movement refit $3,969.35, and the selected IQN $3,207.60.

The movement refit reduced loss by only $6.39 per sale (0.16%) relative to the original linear model. Its q33 calibration was 30.30%, compared with 30.42% for the original model, and its nominal 80% interval coverage was 77.06%, compared with 76.81%. These differences do not establish a material improvement.

The IQN reduced average dollar pinball loss by $768.14 per sale (19.32%) relative to linear q33. However, its overall q33 calibration was 28.55%, its nominal 80% interval coverage was 74.94%, and 103 of 802 validation predictions (12.84%) had crossing quantiles. Cartier calibration was 24.05%, and the middle predicted-price band was 22.55%, both outside the team's proposed ±5 percentage-point tolerance.

A date-clustered bootstrap supported the IQN's observed validation-loss advantage, but the same 2024 observations were used to select network hyperparameters. Therefore, the results are not an independent confirmation of future performance.

**Feature-expansion decision.** The cleaned candidate dataset recovered 1,734 missing field values across six normalized fields, affecting 664 records. These recovered values were not included in the current fitted regression. Potential additions include case diameter, gold karat, and precious-stone information. Their usefulness remains untested.

**Alternative.** Immediately adopt the IQN because it achieved the lowest validation loss, or expand the linear model using all available cleaned fields. We are not adopting either change yet because the IQN's uncertainty estimates require further validation and the recovered specifications may contain extraction errors. Adding predictors without a controlled comparison could increase complexity without improving decision performance.

**Tradeoff.** Retaining the linear model sacrifices the IQN's observed validation-loss advantage but preserves an interpretable reference specification while the IQN's reliability concerns are investigated. The simpler model is not uniformly better calibrated across brands, so this decision does not establish that it is the best future model.

**Next steps.** Complete a reproducible notebook run and verify all required outputs. Preserve the existing chronological split and model results. If time permits, test a small, justified set of additional watch characteristics using identical evaluation records. Investigate quantile-crossing correction and brand-specific calibration before considering the IQN for operational screening.

**Revisit.** Reconsider the model choice if genuinely independent future auction data confirm the IQN's lower dollar pinball loss, its quantile ordering becomes reliable, and calibration meets the team's stated requirements or a revised, justified criterion. Reconsider additional predictors if validated extraction and controlled comparisons demonstrate a meaningful improvement.

**Limitation.** The 2025–2026 retrospective partition is not a fully untouched holdout because 1,088 of its 1,370 observations were exposed during earlier development. The 2024 validation results are also subject to repeated model-selection effects. Neither set of reported scores establishes realized investment profit.

**Files affected.** `analysis.ipynb`, `decisions.md`. Any future changes to predictor construction or model scripts should be recorded in a separate decision entry.

## 2026-11-14: IQN interval, calibration, and robustness

**Decision.** Keep the linear q33 model as the benchmark. Keep the IQN as a recorded challenger.

**Why.** The checks use the same 802 validation sales and the same selected network (width 32, learning rate 0.003, weight decay 0.0001, epoch 250, seed 20261024). The q10–q90 band covers 74.94% of those prices, or 75.06% after ordering the endpoints, against a nominal 80%. The median ordered width is $14,466.09. Quantiles cross on 103 sales (12.84%); q10 is above q90 on 4 sales. The calibration bar used here is 5 percentage points from 33.33% (28.33% to 38.33%). A 3-point bar would fail the IQN overall and still pass the linear model's 30.42%. Under the 5-point bar the IQN passes overall (28.55%), Omega, Rolex, and the low and high predicted-price bands. It fails for Cartier (24.05%) and the middle predicted-price band (22.55%). The linear model is farther from one third for Cartier, Omega, and the high predicted-price band. IQN pinball is lower in every reported slice. The mean paired advantage is $768.14. A 2,000-draw bootstrap that resamples the 64 auction dates gives $322.12 to $1,299.59. Rolex is 77.34% of the summed gain. Removing the five largest Rolex gains leaves a mean advantage of $629.66. Category swaps are limited to cells with at least 15 training sales. Supported model-name swaps are Speedmaster to Seamaster (−$555) and Submariner to GMT-Master II (−$3,222). The yellow-gold quartz Cartier Tortue has no other model with 15 training sales. The 2025–2026 partition was not scored. scikit-learn 1.9.1 and SciPy 1.18.1 match requirements.txt.

**Alternative.** Adopt the IQN because the date-clustered interval for the pinball gain lies above zero and several slices are better calibrated than the linear model. That was rejected because the settings were chosen on these 802 sales, Cartier and the middle band miss the 5-point bar, and 12.84% of the quantile triples cross.

**Revisit.** Replace the linear model only if the pinball advantage remains on sales that were not used to choose the settings, Cartier and the middle predicted-price band move inside the 5-point bar or the team adopts a different bar for a stated reason, and the crossing rate is low enough that q10 and q90 can be reported as an interval.

## 2026-11-13: IQN fitted beside the frozen linear benchmark

**Decision.** Keep the regularized linear q33 model as the benchmark. Record the implicit quantile network as a challenger, not a replacement.

**Why.** The network was fit on the 2,490 training sales only. Width, learning rate, weight decay, and the stopping epoch were selected on the 802 validation sales from 2024. Seed 20261024. On those same 802 sales, dollar pinball loss was $4,380.90 for the brand-q33 baseline, $3,975.74 for the linear model, and $3,207.60 for the selected IQN (width 32, learning rate 0.003, weight decay 0.0001, epoch 250). The IQN loss is the best loss in that search. The share of validation prices at or below the IQN q33 was 28.55%, compared with 30.42% for the linear model and a target of 33.33%. Cartier was 24.05%. Most of the pinball gain was Rolex. The 2025–2026 partition was not scored.

**Alternative.** Replace the linear model because the IQN's 2024 pinball loss is lower. That was rejected because the lower loss selected the network, and the quantile calibration got worse.

**Revisit.** Reconsider the IQN if the lower dollar pinball persists on sales that were not used to choose these settings, and if the share of prices at or below q33 is near one third within each brand.

## 2026-11-12: Implicit Quantile Network challenger

**Decision.** Retain the regularized linear quantile regression as the benchmark and develop one small Implicit Quantile Network (IQN) as an alternative prediction model, following Professor Sokolov's feedback.

**Rationale.** The linear model provides interpretable, cost-sensitive predictions but may not capture nonlinear relationships among watch characteristics. The IQN will learn conditional auction-price quantiles by taking watch features and a cosine embedding of the quantile level as inputs. Training will use dollar-price pinball loss with freshly sampled quantile levels.

**Evaluation.** Fit the network using only the 2,490 training sales from 2021–2023. Select all hyperparameters using the 802 validation sales from 2024. Compare the IQN's q33 average dollar pinball loss against the existing linear model ($3,975.74) and brand-q33 baseline ($4,380.90), using identical observations and loss definitions. Exclude all 2025–2026 sales from training and model selection.

**Alternative.** Retain the linear model without testing a neural network. This avoids additional complexity but leaves possible nonlinear improvements unexplored.

**Interpretation.** Because the IQN has no directly interpretable regression coefficients, estimate the local age effect in USD per year at a stated watch and calculate q33 price contrasts by changing one categorical feature at a time. Report these as predictive associations rather than causal effects.

**Tradeoff.** The IQN may capture nonlinear patterns and interactions but has greater overfitting risk with 2,490 training records and is harder to interpret. Rare model categories will remain collapsed.

**Revisit.** Adopt the IQN only if it improves 2024 validation dollar pinball loss and provides acceptable stability and calibration. Otherwise, retain the linear quantile regression. Record actual IQN results after fitting; no improvement is assumed.

**Status.** Planned experiment; IQN has not yet been fitted or evaluated.

## 2026-11-12: Movement-category standardization and quantile regression refit

**Decision.** Retain the existing regularized 33rd-percentile quantile regression as the primary model while evaluating a revised specification using standardized movement categories: automatic, manual winding, and quartz. Treat the revised specification as an experimental candidate until its data quality and reproducibility are verified.

**Why.** Movement type may help explain auction-price differences, but inconsistent or missing classifications can weaken its predictive value. The revised feature extraction changed movement information for 420 training records and 879 records overall. We tested whether these changes improved prediction accuracy without changing the model family, chronological split, or asymmetric loss function.

**Evidence.** Both specifications used 2,490 training records from 2021–2023 and 802 validation records from 2024. The original regression produced a validation dollar pinball loss of $3,975.74. The standardized-movement refit reduced this to $3,969.35, an improvement of $6.39 per watch (approximately 0.16%). The brand-q33 baseline remained at $4,380.90, giving the revised model a 9.40% lower validation loss.

The revised model's 33rd-percentile calibration was 30.30%, compared with 30.42% for the original model and a nominal target of 33.33%. Its 80% prediction interval coverage improved slightly from 76.81% to 77.06%. These results suggest a small predictive benefit but do not establish a substantial or consistent improvement.

**Alternative.** Adopt the standardized-movement refit immediately as the primary model. We rejected immediate adoption because the validation improvement was small, calibration did not improve, and the revised specification has not yet been evaluated retrospectively. We also considered expanding the model with case diameter and gemstone characteristics, but those features require further validation because of missing values and uncertain classifications.

**Tradeoff.** Standardized movement information may improve predictive accuracy, but it introduces additional preprocessing and data-quality dependencies. The simpler existing specification is easier to reproduce and already performs better than the brand-specific quantile baseline.

**Revisit.** Reconsider adopting the revised model if the movement classifications pass validation, the notebook runs reproducibly, and additional genuinely unseen observations support lower decision-weighted loss without unacceptable deterioration in brand-specific calibration. Reconsider the value of movement enrichment if the improvement disappears after correcting extraction errors or evaluating future auctions.

**Limitations.** The 2025–2026 retrospective evaluation is partially exposed because 1,088 of its 1,370 observations influenced earlier model development. It should not be described as an untouched test set. Dollar pinball loss measures asymmetric prediction error, not realized purchasing profit.

**Next step.** Preserve the original model results, isolate the experimental model variables to prevent notebook-state contamination, verify the revised feature extraction, and document any subsequent model comparison before changing the primary specification.


## 2026-11-09 — Model Validation, Screening Policy, and Retrospective Evaluation

Decision: Retain quantile regression at the 33rd percentile (q33) as the preferred conservative auction-price estimation method for screening luxury watches for further investigation. Do not use the model to automatically recommend purchases or determine maximum bids.

Rationale: The model reflects the assumption that overestimating auction prices is twice as costly as underestimating them. On the 2024 validation set (802 records), quantile regression reduced average dollar pinball loss by 9.25% compared with the brand-q33 baseline ($3,975.74 versus $4,380.90).

The 2025–2026 retrospective evaluation (1,370 records) showed a smaller but positive improvement of 4.38% ($6,455.02 versus $6,750.88). Brand-specific loss improvements were 2.79% for Cartier, 8.64% for Omega, and 4.99% for Rolex.

Alternative rejected: Use the simpler brand-q33 baseline as the primary valuation method. Although the baseline is easier to interpret and maintain, quantile regression achieved lower average dollar pinball loss in both evaluation periods and across all three brands in the retrospective test. The baseline will be retained as a benchmark for future comparisons.

Screening rule: Flag a watch for further investigation when its predicted q33 auction price is at least 10% above an independently available pre-auction price benchmark. The 10% margin is illustrative and has not been optimized. If no reliable pre-auction benchmark exists, report the valuation without issuing a screening flag.

Simulation evidence: A simulation using 3,000 synthetic watch records recovered coefficients close to their known values. Observed q33 calibration was 33.37% against the 33.33% target, and the mean absolute difference between estimated and true conditional q33 predictions was $40.63. This supports the simplified method's implementation but does not establish real-world model validity.

Uncertainty and limitations: The 2025–2026 retrospective evaluation produced overall q33 calibration of 29.78% and 78.25% coverage for the nominal 80% prediction interval. Brand-specific q33 calibration varied: Cartier 21.41%, Omega 39.47%, and Rolex 30.61%.

Omega achieved the largest relative improvement in dollar pinball loss despite calibration above the target. This demonstrates that lower cost-weighted prediction error does not necessarily imply better quantile calibration.

Additionally, 1,088 of the 1,370 chronological test records had been exposed during earlier random-split development. The retrospective results cannot be presented as evidence from a completely untouched holdout.

Revisit condition: Reconsider the model if genuinely unseen auction data show no improvement over the brand-q33 baseline, if brand-specific calibration deteriorates materially, or if actual pre-auction benchmark and investigation-cost data show that the screening policy does not improve decisions. Investigate brand-specific calibration or alternative models before changing the current approach.

Files affected: analysis.ipynb, decisions.md.

## 2026-11-09 — Conservative Description-Based Cleaning and Candidate Validation

**Decision:** Use conservative rule-based extraction to recover missing watch specifications from auction descriptions while preserving the approved source dataset. Save the enriched values in a separate candidate CSV rather than modifying existing fields.

**Rationale:** The revised cleaning procedure recovered 1,734 missing values across six normalized fields, improving 664 of 4,664 auction records (14.24%). Missingness decreased by 13.43% across the six affected fields. The integrity audit confirmed that all 212 original columns and 667 literal `None` values were preserved without modifying original source values.

**Alternative rejected:** Broader text extraction that identifies dial colors from references to hands or markers, or case materials from references to bezels or case backs. Although broader extraction produced more populated values, it increased the risk of incorrect specifications. The preliminary procedure recovered 1,805 values, compared with 1,734 under the revised conservative rules.

**Modeling decision:** Do not include the newly enriched fields in the current fitted quantile-regression model. The candidate dataset has improved completeness, but the accuracy of recovered specifications and their effect on out-of-sample prediction have not yet been established.

**Revisit condition:** Reconsider using the enriched fields after manual validation of recovered dial colors and case materials, confirmation that extraction errors are acceptably low, and an evaluation showing improved out-of-sample dollar pinball loss relative to the current model using the same validation design. Retain the simpler model if the enrichment does not provide a reliable improvement.

**Files affected:** `analysis.ipynb`, `data/README.md`; candidate dataset `expanded_pull_eligible_cleaned.csv` (stored outside the public repository).

## 2026-11-07: Asymmetric loss and quantile regression

Decision. Use a 2:1 cost ratio for overestimating versus underestimating luxury watch auction prices. Fit regularized quantile regression at τ = 1/3 (approximately 0.333) on log sale price. Evaluate predictions in USD using asymmetric pinball loss on the chronological validation set.

Why. The model supports watch acquisition screening, where overestimating resale value can lead to paying too much, while underestimating value may result in a missed opportunity. With overestimation costing twice as much as underestimation, the appropriate target is the 33rd conditional percentile rather than the mean or median. This produces more conservative price estimates.

On the 2024 validation set (802 records), the model achieved a dollar-scale pinball loss of $3,975.74, improving 12.05% over the brand-median baseline ($4,520.62) and 9.25% over the brand-specific 33rd-percentile baseline ($4,380.90).

Alternative. Use ordinary least squares or median regression with equal error costs. These approaches are simpler but do not directly reflect the team's assumed asymmetric acquisition risk. A brand-specific 33rd-percentile baseline is a simpler alternative that directly targets the chosen quantile. A dollar-price quantile model is also reasonable because fitting on log prices does not directly minimize dollar-scale pinball loss.

Checks. The three quantile models use independent preprocessing pipelines. No quantile crossings were observed across the 802 validation records.

Overall, 30.42% of actual prices fell below the predicted 33rd percentile, compared with the nominal target of 33.33%. The estimated 10th–90th percentile intervals achieved 76.81% coverage, slightly below the nominal 80%.

Brand-level calibration differed substantially:

Cartier: 19.94% below q33; 81.01% interval coverage.

Omega: 43.35% below q33; 65.52% interval coverage.

Rolex: 32.86% below q33; 80.21% interval coverage.

These results indicate that improvements in aggregate validation loss do not guarantee reliable calibration for every brand.

Revisit. Reconsider the 2:1 cost ratio if actual acquisition costs suggest a different relationship between overestimation and underestimation. Reevaluate the model if brand-specific calibration remains poor, particularly for Omega, or if future chronological evaluation fails to show an improvement over the brand-specific 33rd-percentile baseline.

Investigate whether alternative specifications, including dollar-price quantile regression or brand-specific models, improve calibration and decision performance without introducing excessive complexity. Preserve the reserved test partition for final evaluation, while acknowledging that some test records were exposed during earlier model development.


## 2026-11-07: Audit preliminary description-based data cleaning

Decision. Create a separate cleaned dataset, data/expanded_pull_eligible_cleaned.csv, using the approved expanded_pull_eligible.csv as the source. Preserve all 4,664 records and 212 original columns, and add 66 columns with the suffix _cleaned, corresponding to the existing norm_* fields. Copy existing normalized values and attempt to populate missing values using explicitly labeled information from norm_item_description. Do not overwrite populated normalized values or modify the approved source dataset. Treat the resulting file as preliminary and do not replace the team's approved modeling predictors until validation is complete.

Why. Missing watch specifications may limit auction-price prediction accuracy. Catalogue descriptions contain additional information about watch models, reference numbers, dials, cases, and bracelets that can potentially recover missing predictor values. Retaining separate cleaned columns allows the team to compare the new extraction against the existing normalized and reviewed fields without losing the original information.

Checks. The initial audit found 1,805 newly populated cells across 675 of the 4,664 records (14.5%). Six normalized fields improved:

norm_dial: 650 values recovered.

norm_watch_case: 568 values recovered.

norm_reference: 245 values recovered.

norm_strap_bracelet: 216 values recovered.

norm_dial_color: 78 values recovered.

norm_model: 48 values recovered.

Across these six fields, missing cells decreased from 12,911 to 11,106, a 14.0% reduction. No existing populated normalized values were overwritten. The cleaned dataset contains 278 columns.

Quality issues. The audit identified 667 differences in the original item_provenance column, associated with conversion of the literal string None into blank values during CSV processing. The import process should use keep_default_na=False to preserve these source values. The dial-color extraction also requires review because some colors describe hands, numerals, or markers rather than the dial itself. These issues have been identified but are not yet confirmed as corrected.

Modeling implications. Additional populated fields do not establish that the extracted specifications are accurate or that prediction performance will improve. Continue using the previously approved predictor-construction rules in scripts.predictors, including reviewed case-body and movement decisions. Do not automatically substitute the new _cleaned fields. Any later comparison must use the existing chronological training and validation assignments without changing the split or scoring the reserved test partition.

Alternative. Continue using only the existing normalized and reviewed predictors, leaving missing values unresolved. This avoids introducing extraction errors but may discard useful information available in catalogue descriptions. Automatically replacing existing values with newly extracted specifications was rejected because it could override reviewed corrections and introduce incorrect predictor values.

Revisit. Correct the item_provenance preservation issue, review questionable dial-color extractions, and rerun the quality audit. Compare newly recovered specifications against the existing reviewed fields and document unresolved conflicts. Record the final cleaned-file SHA-256 checksum after corrections. Evaluate whether validated cleaned predictors improve performance on the chronological validation partition before adopting them for modeling. Preserve the approved original dataset and existing chronological assignments throughout.


## 2026-11-04: Integrate the chronological split into the analysis notebook

Decision. Read `processed/expanded_pull_eligible.csv` from `WATCH_DATA_DIR`, keep source fields as strings, and require SHA-256 `cf3b41d890af1feba9348fa9a73d954fcab193ed890a39044bd0355198fc5a8a`. Build the final predictors with `construct_record` from `scripts.predictors`. Take partition membership from the saved chronological assignment and omission files, joined on unique `record_id`, without regenerating or overwriting those files. The active sale-price summary uses the 2,490 chronological training rows. The training-plot cell is written to save `fig/chronological_training_price_boxplot.png` when it is run. That cell has no saved output, and the file is not present. Keep `fig/watch_price_boxplot.png` as the preserved full-sample figure and do not regenerate it. Keep the earlier full-sample medians and catalogue missingness as historical description. Do not fit a model, generate predictions, or score validation or test.

Why. Pull request #10 made `analysis.ipynb` readable again and left chronological integration undone. The purchasing comparison is about later auctions, so the training rows are the sales before 2024-01-01. Approved blanks and unresolved conflicts stay missing. On the training rows, 3 approved-blank case bodies and 2 unresolved case-body conflicts stay missing, and 4 unresolved movement conflicts stay missing. No approved-blank movement is present in training. A finish recorded as `not_stated` stays a populated label, and 2,210 training rows have that label. The machine-specific data path is not written into the notebook.

Checks. The eligible file has 4,664 unique record IDs and 212 columns. Coverage is 4,664 = 2,490 training + 802 validation + 1,370 test + 2 omitted. The partitions are disjoint. Training dates run 2021-02-05 through 2023-12-12 and are all before 2024-01-01. Validation dates run 2024-01-30 through 2024-12-19 and stay inside 2024. Test dates run 2025-01-30 through 2026-07-28 and are on or after 2025-01-01. The omitted sales are the Cartier reference 2510, serial 362696CD, sold on 2024-04-19, whose 2025-03-31 sale stays in test, and the Omega reference 3570.40, serial 77132178, sold on 2022-04-30, whose 2024-10-16 sale stays in validation. `tests/test_predictors.py` passed. After the latest edit, the notebook parses as JSON, passes nbformat validation, and its code cells pass a Python syntax check. Those checks do not execute the notebook. Saved outputs in the setup, load, catalogue-missingness, split, and training-summary cells are from an earlier Python 3.12.10 run with pandas 3.0.5, NumPy 2.5.2, and Matplotlib 3.11.1. The training-plot cell has no output. The preserved full-sample figure was restored to the `origin/main` file and its cell was not rerun. Saved model files and the chronological assignment files were not rewritten. The repository pin is Python 3.14.7 with pandas 3.1.0, NumPy 2.5.4, and Matplotlib 3.11.2. That interpreter is not installed here, so the notebook was not run from the top in the pinned environment. No model was fit and no partition was scored.

Retrospective disclosure. Of the 1,370 chronological test lots, 1,088 were already in the random training set (817) or the random validation set (271). The other 282 were in the unscored random test partition. This test is not an untouched holdout, and this notebook does not score it.

Protocol discrepancy. This disagreement is unresolved and is flagged for team review. It is not a settled instruction. Issue #6 asks for the conditional quantile to be compared with the brand-median baseline on the same tilted absolute loss. The 2026-11-01 decision records a brand-specific quantile baseline as the primary comparison and keeps the brand-median baseline as an additional reference if useful. This notebook does not choose between those statements, does not select a cost ratio or a quantile, and does not fit either baseline.

Alternative. Regenerate the preserved full-sample figure, or treat one of the two baseline statements as already adopted. Regenerating the figure would replace the historical file. Adopting either baseline before team review would settle a disagreement this notebook is meant to leave open. Rebuilding the assignments in the notebook could overwrite the saved chronological files.

Revisit. Revisit when the team adopts a cost ratio and quantile, when it decides whether the primary baseline is the brand median requested in Issue #6 or the brand-specific quantile recorded on 2026-11-01, or if the eligible file or saved assignments change. Do not score these 1,370 test lots as unseen. Do not move the date boundaries to improve a score.

## 2026-11-03: Repair the analysis notebook after the merge

Decision. Restore a valid `analysis.ipynb` from the last notebooks that still parse: `a0399f0` on `lflanigan-sudo` and `b336585` on `main`, whose notebook matches Anna Kelly's `1436bab`. Keep the research question, checksum check, missingness explanation, and figures. Label the 2026-10-26 brand-median comparison under mean absolute error as the earlier plan. The current plan is the 2026-11-01 entry: quantile regression against a brand-specific quantile baseline under tilted absolute loss, with the brand-median baseline retained as an additional reference if useful. The cost ratio and quantile stay pending. This repair does not fit a model, score a partition, or integrate the chronological assignments into the notebook.

Why. Commit `c258d94` merged `main` into `lflanigan-sudo` and left two notebook source strings adjacent without a comma. JSON parsing fails at line 303. That file is the notebook in pull request #7's merge, `e00b2b9`, and in current `origin/main`. Both parents parse. `a0399f0` carries the quantile question. `1436bab` carries the brand-summary prose, the boxplot lead-in, and the cleaned-field missingness explanation. Pull requests #8 and #9 are merged. Their scripts and decision entries are on `main` and did not edit the notebook.

Alternative. Keep one parent and drop the other. That would discard either the quantile question or the brand-median reference and the boxplot lead-in.

Revisit. Revisit the notebook prose if the team selects a cost ratio and quantile, or if a later edit changes the audited missingness explanation.

## 2026-11-02: Reuse the audited predictor construction

Decision. Reuse the audited expanded-pull construction for the predictors: normalized model, case body, case finish, movement, production year, age at sale, the approximate-year indicator, and the existing case diameter, width, and length. Package that logic in an import-safe module. This branch is separate from the open chronological-split pull request. Notebook integration follows that merge. This change does not fit a model or score a partition.

Why. `modeling_case_body` contains the reviewed case-body overrides and stays blank until a review writes one. `modeling_movement` is an intermediate column: cleaning fills it with a non-conflicting extracted movement, and the final movement predictor uses it only when `movement_review_status` is approved. An approved blank stays missing. An unresolved source conflict stays missing, with no fallback to the raw field. The construction also keeps plating distinct from a solid-gold body and keeps a bezel, dial, bracelet, buckle, or case back distinct from the case body. `not_stated` means no finish was stated.

The reused rules match the saved modeling rows on every compared field. The input was `processed/expanded_pull_eligible.csv`, SHA-256 `cf3b41d890af1feba9348fa9a73d954fcab193ed890a39044bd0355198fc5a8a`, 4,664 rows. The comparison file was `modeling/expanded_pull_modeling_rows.csv`, SHA-256 `d2ee0efa168114f5e88a381943cf9969afb0032816ff3b1cf9b299e79777fad7`. Populated results: case body 4,194; movement 3,241; age at sale 4,318; case finish 4,249. Those finishes contain 4,113 values of `not_stated`, 131 `gold-plated`, 4 `vermeil`, and 1 `rose-gold-plated`. The earlier audit count of 4,065 is the source label `no_finish_stated`, and every one of those rows has the value `not_stated`. The other 48 `not_stated` values are approved corrections whose reviewed finish was blank. Production year matched `production_year_parsed` on every row, and each populated age was the sale year minus that year.

Alternative. Build the predictors from the raw material and movement fields, or treat `modeling_case_body` and `modeling_movement` as the finished predictors on every row. The raw fields skip the reviewed corrections. Using the intermediate movement column on unreviewed rows would replace the structured-and-extracted movement rule.

Revisit. Revisit this construction if the eligible-file checksum changes, a later correction changes an approved or unresolved value, or the team adopts a different extraction rule. A mismatch with the saved modeling rows is a reason to stop and report it, not a reason to change these rules.

## 2026-11-02: Chronological split

Decision. Assign the chronological experiment by sale date. Training is every eligible sale before 2024-01-01: 2,490 lots, from 2021-02-05 through 2023-12-12. Validation is 2024-01-01 through 2024-12-31: 802 lots, from 2024-01-30 through 2024-12-19. Test is 2025-01-01 through the last sale in the eligible file: 1,370 lots, from 2025-01-30 through 2026-07-28. The eligible file stays at 4,664 lots. These counts are the result of that date rule plus the two omissions below.

Repeat-watch policy. Omit an earlier sale only when the saved diagnostics confirm the same physical watch, by brand, reference, and an explicit case serial, and that watch also sells in a later period. The later sale stays in the period of its own sale date. When both confirmed sales fall in one period, both stay and keep the shared physical-watch identifier. A shared reference without that serial is not a reason to omit a lot. The omitted sales are `cartier_sothebys_2024_o000050_l050.json#15`, the Cartier reference 2510 serial 362696CD sold on 2024-04-19, and `omega_sothebys_2022_o000000_l050.json#30`, the Omega reference 3570.40 serial 77132178 sold on 2022-04-30. The 2025-03-31 sale of the Cartier stays in test. The 2024-10-16 sale of the Omega stays in validation. After these omissions, none of the six confirmed physical-watch groups has sales in more than one partition. The master file remains 4,664 lots, and the experiment uses 4,662.

Why. The purchasing decision is about later auctions, so the partitions follow sale date. All three brands and both auction houses appear in every period. The thinnest saved brand-house cells are still Omega at Sotheby's: 54 validation lots and 66 test lots. The input file is `processed/expanded_pull_eligible.csv`, SHA-256 `cf3b41d890af1feba9348fa9a73d954fcab193ed890a39044bd0355198fc5a8a`. Assignments come from `scripts/split_chronological.py` version 1. Its opening note lists the eligible file, the chronological diagnostics, and the saved random-split files it reads under the shared data root. The assignment and omission files were written by script SHA-256 `f870d30ab1d25a07ea41e3360af2d67dac84e7aa6b1c89f92d4e7db09ecaf982`. The usage note is documentation only and has SHA-256 `f6d03df2574350ed37ae13a89d04e886982819d614381d6bcfe28f4df8edd195`. The saved metadata now records the current script hash `f3be9fa6324583ebfd91d3f8fd885eeaeee44df9b070a2217934e1699583b53c` and retains the previous hash. The previous metadata file is kept beside it and was not replaced in Git. The assignment file, omission ledger, and checksums are in the shared data root under `modeling/chronological/` and are not committed. Of the 1,370 chronological test lots, 1,088 were already in the random training set (817) or the random validation set (271). Those 1,088 lots influenced earlier development, so this test is not an untouched holdout. The other 282 chronological test lots were in the unscored random test partition.

Alternative. Keep the random 60/20/20 split, seed 20261024, as the primary evaluation. That split remains saved, and its final test remains unscored. It balances brands, but it mixes earlier and later sales, so it measures error inside the sampled auction population rather than error on later auctions. Treating a shared reference as the same physical watch, or moving a later sale into training, was set aside.

Revisit. Revisit this split if a refreshed eligible file changes these dates or counts, if new explicit case-serial evidence shows another physical watch in more than one period, or if a brand or auction house becomes too thin for the purchasing comparison. Revisit using these 1,370 test lots as a final performance number if the prior use of 1,088 of them is not an acceptable safeguard. Do not describe those 1,088 lots as unseen. Do not move the date boundaries to improve a model score.

## 2026-11-01: Asymmetric loss and quantile regression

Decision. Following Professor Sokolov's feedback in Issue #6, replace MAE as the primary evaluation metric with tilted absolute loss (pinball loss). Fit conditional quantile regression on log-transformed auction prices and convert predictions back to USD for purchasing comparisons.

Why. Overestimating a watch's value is more costly than underestimating it. MAE assigns equal penalties to both errors and therefore does not represent our intended purchasing decision.

Quantile selection. Choose the quantile based on the relative costs of overestimation and underestimation. The team will document and justify the selected cost ratio before fitting the final model.

Baseline. Compare quantile regression with a brand-specific quantile baseline using the same tilted loss. Retain the brand-median baseline as an additional reference if useful.

Evaluation. Preserve the proposed chronological split. Examine residuals by watch model and production year to identify systematic errors or price discontinuities.

Limitations. The dataset contains completed sales only and may not represent watches that failed to sell. The existing test-set exposure limitation also remains.

Alternative. Continue using ordinary regression and MAE. This approach is simpler but does not account for asymmetric purchasing costs.

Revisit. Reconsider the cost ratio, quantile, or model specification if validation results or purchasing assumptions change.

## 2026-10-28: Chronological split feasibility and test-set limitations

Decision. Continue evaluating the proposed chronological split: training on 2021–2023 sales (2,490 observations), validation on 2024 sales (802), and testing on 2025–2026 sales (1,370). These counts reflect the proposed exclusion of two earlier sales of confirmed repeat physical watches. The split remains provisional until the team finalizes the evaluation protocol.

Why. Jordan's feasibility check confirmed that all three brands and both auction houses are represented in each period. The smallest brand–auction-house subgroup is Omega at Sotheby's, with 54 validation and 66 test observations. Chronological evaluation better reflects the intended task of predicting prices for later auctions.

Repeat-watch handling. The two proposed exclusions concern confirmed repeat sales of the same physical watch, identified using reference and case-serial evidence. Watches sharing a model reference are not automatically treated as duplicates. The master dataset remains unchanged at 4,664 observations; the proposed experimental dataset contains 4,662.

Test-set limitation. Of the 1,370 proposed chronological test observations, 1,088 were previously included in training or validation during preliminary random-split development. Therefore, the chronological test set cannot be described as completely untouched. We will reserve chronological test scoring until the evaluation approach is finalized and seek Professor Sokolov's guidance on additional safeguards.

Alternative. Continue using the preliminary random 60/20/20 split as the primary evaluation. This preserves the original experiment but is less representative of predicting later auction prices.

Revisit. Reconsider the split or evaluation procedure if prior test exposure compromises the credibility of final performance estimates, repeat-watch handling introduces bias, or chronological performance varies substantially across brands and auction houses.

## 2026-10-26: Prediction target and purchasing decision

Decision. Use recorded realized auction sale price in USD as the prediction target and mean absolute error (MAE) to evaluate performance against a baseline that predicts each brand’s median sale price calculated from training data only. Use predictions to screen potential purchases rather than automatically set maximum bids.

Why. Completed auction sales provide observable transaction prices. MAE expresses prediction errors in dollars, making them easier to interpret for purchasing decisions. Overestimating a watch's value is more costly because it can lead to overpaying and resale losses.

Alternative. Predict asking prices or use model predictions directly to determine maximum bids. We prefer completed sales and purchase screening because auction prices do not fully represent net resale proceeds, transaction costs, or uncertainty.

Revisit. Reconsider the evaluation metric or decision rule if held-out errors are large, especially for expensive watches, or if predictions do not improve purchase screening.

## 2026-10-26: Chronological evaluation strategy

Decision. Propose using sales before January 1, 2024, for training, sales during 2024 for validation, and sales from January 1, 2025, onward for testing. These boundaries remain provisional pending partition-count and repeat-watch checks.

Why. The intended use is predicting prices for later auctions. Chronological evaluation better reflects this use than randomly mixing earlier and later sales.

Preliminary development used a random 60/20/20 training, validation, and test split with seed 20261024. Those assignments and validation results are preserved, and the original final test remains unscored. Some observations in the proposed later periods have already influenced model development through training or validation, so chronological evaluation will not be described as wholly untouched.

Alternative. Retain the random 60/20/20 split as the primary evaluation. It provides approximately balanced brand representation but measures prediction within the sampled auction population rather than performance on later periods as directly.

Revisit. Before finalizing the chronological split, confirm observation counts by period, brand, and auction house. Specify how confirmed repeat sales of the same physical watch crossing date boundaries will be handled without moving later sales into training. Select boundaries based on the intended use and adequate coverage, not model scores.

## 2026-10-26: Finalize initial eligible watch sample

Decision. Use the 4,664 eligible individual wristwatch sales remaining after cleaning: 1,786 Rolex, 1,594 Cartier, and 1,284 Omega.

Why. The original 5,000 downloaded records included duplicates, accessories, non-watch objects, and bundled lots. Restricting the sample to individual wristwatch sales makes the observations more consistent with the prediction target.

Alternative. Retain all downloaded records to maximize sample size. We rejected this because bundled lots and non-watch objects do not represent comparable individual wristwatch transactions.

Revisit. Recheck eligibility rules if data-quality reviews identify incorrectly excluded or included lots.

## 2026-10-23: Auction data and initial modeling scope

- Decision. Use ALT/FNData completed auction sales for Cartier, Rolex, and Omega at Sotheby’s and Bonhams, within the January 2021–October 2026 query window, to predict realized sale prices in USD.
- Why. The saved pull contains 5,000 rows representing 4,964 unique lots before eligibility cleaning. It provides observed sale outcomes and usable watch characteristics, although coverage varies and retrieval is capped and nonrandom.
- Alternative. Use asking-price listings or immediately include additional luxury brands. We selected completed sales to ground the target in observed transactions and limited the initial scope to establish a workable baseline.
- Revisit. Reconsider the scope if cleaning leaves insufficient coverage, source validation reveals unreliable prices, or held-out evaluation shows that the available characteristics cannot support useful predictions.
  
## 2026-10-23: Team Collaboration and GitHub Setup

- Decision. Use a shared GitHub repository with individual branches and pull requests for team contributions. Add all team members' names and GitHub usernames to README.md.

- Why. This follows Professor Sokolov's project requirements and allows team members to review changes before they are merged into the main branch.

- Alternative. Allow team members to edit the main branch directly. We rejected this because it would bypass the required peer-review process.

- Revisit. Update the team information if membership or GitHub usernames change. Continue using branches and pull requests throughout the project.

