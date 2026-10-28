# Proposal


## Hypothesis

We hypothesize that a regression model using brand, model, approximate watch age, case material, movement type, and other available characteristics will predict realized auction sale prices with lower mean absolute error (MAE) in U.S. dollars than a baseline that predicts each brand’s median sale price, as calculated from our training data. Our target is the realized sale price in U.S. dollars for individual Cartier, Rolex, and Omega wristwatches sold at Sotheby’s and Bonhams, using auction records from 2021–2026 to evaluate predictions for later sales.

## Data

We collected auction records through the ALT/FNData API for Cartier, Rolex, and Omega at Sotheby’s and Bonhams, filtering for lots marked Sold with positive USD prices and sale dates from January 1, 2021, through October 22, 2026. The download contained 5,000 rows representing 4,964 unique auction lots before eligibility cleaning. After cleaning, 4,664 individual wristwatch sales remain: 1,786 Rolex, 1,594 Cartier, and 1,284 Omega. Observed sales extend through July 28, 2026. Requests were partitioned by brand, auction house, and year, with capped retrieval in ascending sale-date order; this is not a random sample or complete market coverage.

The analytical unit is one completed sale of a single wristwatch. Cleaning excluded duplicate copies, non-watch objects, accessories, and bundled lots. Predictors include brand, model, approximate production year, case material, and movement. Case measurements and condition reports are too incomplete to be required predictors.

Published auction results generally include buyer's premium, and we will check the imported price fields against selected original results. Recorded prices do not directly measure net resale proceeds or an appropriate acquisition ceiling; fees, service costs, and uncertainty must also inform purchasing decisions. Source documentation, retrieval commands, cleaning rules, and dataset-access instructions will be recorded in data/README.md. Transaction files will remain outside Git history; any public release will depend on explicit redistribution permission.

## Decision affected by the model

A watch collector or reseller will use predicted auction sale prices to decide which watches warrant further investigation before purchase, rather than automatically calculating maximum bids. The buyer will compare the expected total acquisition cost—including applicable buyer’s premium, taxes, shipping, and anticipated service—with the predicted price and its uncertainty without double-counting any premium already included in the recorded auction price. Resellers must also account for selling costs and their required margin.

We will compare a regression model with a brand-median baseline using held-out sales and evaluate MAE in U.S. dollars. We propose chronological evaluation: sales before 2024 for training, 2024 for validation, and 2025–2026 for testing, to better reflect future purchasing decisions. Overestimating value is the more costly mistake because it can justify overpaying and produce a resale loss; underestimating value may cause a missed opportunity. Predictions will support purchase screening, with additional review for unusual watches or poorly represented references.
