# Defense summary

## The six parts

**Decision.** Use the five-feature log-price gradient booster as the predictive screen, and keep linear quantile regression as the interpretable reference. The screen flags a lot for investigation when predicted q33 is at least 10% above the catalogue midpoint. It does not set a bid.

**Finding.** On the 802 sales in 2024, the booster's mean dollar pinball is $3,168.73, which is $807.01 and 20.30% below the linear model's $3,975.74. None of the tested feature configurations improved that 2024 score. In 2026 the booster's pinball is still lower, $5,540.14 against $6,537.68, but only 23.80% of sales fall at or below its q33, versus 33.33% expected.

**Evidence.** The 2024 table is the selection result. The same frozen settings, fit only on earlier sales, stay ahead of the linear model on pinball in 2022, 2023, 2025, and 2026. The 2026 brand table is the calibration evidence: Rolex is 21.40% on 243 sales and Cartier is 21.31% on 183, while Omega is 31.39% on 137. The interval is a different object: q10 and q90 fit before 2023, cutoff $650.96 from 2023 only, coverage 78.43% in 2024 and 72.65% in 2026.

**Alternatives rejected.** The linear model, because its 2024 pinball is higher and its 2026 Rolex pinball is higher by $1,802.08 even though its Rolex share, 31.69%, is closer to one third. The IQN, because its 2024 pinball is $3,207.60 and the paired comparison against the booster includes zero. CatBoost, because its 2024 pinball is $3,321.24. The extra listing attributes, because every tested group had higher 2024 pinball and the date-clustered intervals included zero. Stripping a buyer's premium and refitting, because the premium is verified only on a sample and Sotheby's results were not public.

**Strongest check.** The pinball advantage did not reverse in a later year. The check that blocks a naive screening claim is the 2026 Rolex and Cartier calibration, the interval coverage falling from 78.43% to 72.65%, and the price basis: the recorded sale price includes premium on the Bonhams lots that could be opened, while the catalogue estimate is not verified to be on the same basis or to have been published before bidding.

**What would change it.** A later window, fit only on earlier sales, in which the booster's pinball exceeds the linear model's. Or an explicit team decision that a share near 33% outranks pinball for Rolex and Cartier. Or a confirmed hammer series for both houses, followed by a refit that changes the ranking. Or a listing field, confirmed on the page before the sale, that reduces q33 pinball with a date-clustered interval above zero.

## Two-minute opening

The project screens watches for further investigation. It does not set a bid. The selected predictor is the five-feature gradient booster because its 2024 dollar pinball is $3,168.73, 20.30% below linear quantile regression, and that pinball advantage repeats in every later year we scored. We rejected the linear model, the IQN, CatBoost, and the extra listing attributes because each had higher 2024 pinball. We would reconsider if new sales removed the pinball advantage, or if the team required the 2026 pattern to be fixed first: for Rolex and Cartier the booster's q33 sat too low, at about 21%. A lower loss is not the same thing as a quantile those brands can screen on, and a sale above the catalogue estimate is not a profit. The recorded target is the USD sale price. On the lots we could check, that amount includes buyer's premium.

## Questions the team should be able to answer

1. Why keep the booster when 2026 Rolex calibration is 21.40% and the linear model's is 31.69%? Because the decision loss is pinball, and the booster's Rolex pinball is lower by $1,802.08. We would switch if that loss reversed, or if the team decided the share outranks the loss.
2. Why is 20.30% the headline rather than the 2026 result? Because 2024 is the selection window. 2026 is a sensitivity check. It still favors the booster on pinball and it fails a claim that the quantile is calibrated.
3. Which interval is the project's? The time-ordered band: fit before 2023, cutoff $650.96 from 2023, 78.43% in 2024 and 72.65% in 2026. The 70.07% band is the unadjusted 2024 fit and is not this interval.
4. Did any added attribute help? None of the tested configurations improved the observed 2024 validation dollar pinball. That is not a claim that an attribute can never help.
5. What can the 10% screen be said to do? It flags lots whose predicted q33 of the recorded sale price clears the catalogue midpoint by 10%. It does not show that buying the watch would have made money, and the estimate's publication time and price basis are not verified for the file as a whole.
