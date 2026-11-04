# Decisions

This file is the history of the project. Append an entry when a choice changes the work: the quantity, the data, the split, the model, the loss, or the action. Leave old entries in place. A later entry can reverse an earlier one. If a result depends on a random draw, record the seed under Why. The split entry records the cutoff date, or the seed and the fraction. The pull request that merges the code names this entry.

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

