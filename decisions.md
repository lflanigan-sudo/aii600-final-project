# Decisions

This file is the history of the project. Append an entry when a choice changes the work: the quantity, the data, the split, the model, the loss, or the action. Leave old entries in place. A later entry can reverse an earlier one. If a result depends on a random draw, record the seed under Why. The split entry records the cutoff date, or the seed and the fraction. The pull request that merges the code names this entry.


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

