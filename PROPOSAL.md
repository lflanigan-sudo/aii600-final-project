# Proposal

At most one page. Replace the prompts under each heading. Leave the headings.

## Hypothesis

State the claim in one sentence. Then name the single quantity you will forecast or estimate, with its units and the time window.

Luxury watches with similar characteristics can have substantially different listing prices, creating opportunities to identify potentially undervalued watches. We hypothesize that a regression model using brand, model, age, condition, and other available characteristics can estimate typical market listing prices more accurately than a simple baseline. Our target quantity is the estimated listing price in U.S. dollars for watches advertised during [data collection period].

## Data

Name the source, the unit of observation, the sample size, the time span, and how the data were collected. Say how what you measure differs from the quantity the decision needs. Say whether the terms let you republish the files. A release asset is a republished copy. If the terms do not allow that, the files stay off GitHub and `data/README.md` gives the command that downloads them from the source.

We plan to use [dataset name and source], containing individual luxury watch listings. Each observation represents one watch listing, with potential variables including brand, model, production year, condition, and asking price. The dataset contains [number] observations collected between [start date] and [end date] using [collection method].

We will verify the dataset's completeness, reliability, and licensing restrictions before analysis. Listed asking prices may differ from completed sale prices, meaning our estimates will reflect advertised market values rather than confirmed resale values. We will document the source and collection process in data/README.md and will not republish the data unless permitted by its terms.


## Decision affected by the model

Name who acts, and what they will do differently because of the prediction. Say which mistake costs more.

The model will help luxury watch buyers or resellers decide which listings deserve further investigation. We will compare each watch's asking price with its predicted market listing price and identify listings with potentially attractive discounts.

A simple baseline will be compared with a regression model using held-out data. We will evaluate prediction errors in U.S. dollars and assess how those errors affect purchasing decisions. Overestimating a watch's value could lead to an unprofitable purchase, while underestimating its value could cause a missed opportunity. For a buyer seeking to avoid financial losses, overestimation is the more costly mistake. Model predictions will support further evaluation rather than automatically trigger purchases.

