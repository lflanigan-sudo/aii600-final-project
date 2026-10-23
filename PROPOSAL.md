# Proposal


## Hypothesis

We hypothesize that a regression model using brand, model, approximate watch age, case material, movement type, and other available characteristics will predict realized auction sale prices more accurately than a baseline that predicts each brand’s median sale price. Our target is the realized sale price in U.S. dollars for individual Cartier, Rolex, and Omega wristwatches sold at Sotheby’s and Bonhams between January 1, 2021, and October 22, 2026.

## Data

We collected auction records through the ALT/FNData API for Cartier, Rolex, and Omega at Sotheby’s and Bonhams, filtering for lots marked Sold with positive USD prices and sale dates from January 1, 2021, through October 22, 2026. The download contains 5,000 rows representing 4,964 unique auction lots before eligibility cleaning. Requests were partitioned by brand, auction house, and year, with capped retrieval in ascending sale-date order; this is not a random sample or complete market coverage.

The analytical unit will be one completed sale of a single wristwatch. Cleaning will remove duplicate copies and exclude non-watch objects and bundled lots, while flagging conflicting specifications. Predictors include brand, model, approximate production year, case material, movement, and available case measurements. Condition-report coverage is insufficient for a core predictor.

Recorded auction prices do not directly measure net resale proceeds or an appropriate acquisition ceiling; fees, service costs, and uncertainty must also inform purchasing decisions. Source documentation, retrieval commands, cleaning rules, and dataset-access arrangements will be recorded in data/README.md. Transaction files will remain outside Git history; any public release will depend on explicit redistribution permission.


## Decision affected by the model

A watch collector or reseller will use predicted auction sale prices to decide which watches warrant further investigation and whether a proposed purchase price is reasonable. The buyer will compare the expected total acquisition cost—including buyer’s premium, taxes, shipping, and anticipated service—with the predicted price and its uncertainty. Resellers must also account for selling costs and their required margin.

We will compare a regression model with a brand-median baseline using held-out sales and evaluate errors in U.S. dollars. Overestimating value is the more costly mistake because it can justify overpaying and produce a resale loss; underestimating value may cause a missed opportunity. Predictions will support purchase screening and conservative bid limits, with additional review for unusual watches or poorly represented references.

