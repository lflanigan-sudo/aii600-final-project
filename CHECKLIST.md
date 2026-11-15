# Checklist

Tick every line that the pushed `main` covers. A line you cannot tick needs one sentence saying why. This file is the list that counts. The ticks on the lab page do not.

## 1. Question

- [ ] The question is one sentence, then one quantity with units and a time window.
- [ ] The decision is named, and the loss or profit is a formula. The costlier mistake is named.
- [ ] A layer diagram: data, model, prediction, decision.

## 2. Data

- [ ] Source, unit of observation, sample size, time span, and how it was collected.
- [ ] How the measurement differs from the quantity the decision needs, and whether that gap is bias or noise.
- [ ] One plot of the raw data and one table of summary statistics, before any model.

## 3. Model

- [ ] The model as equations. Each parameter in words, and the mechanism it stands for.
- [ ] The assumptions, and the one you trust least.
- [ ] Simulated data from the model, checked against the real data. If the fit can recover known parameters, show that it does.

## 4. Uncertainty

- [ ] Every prediction carries a standard error, an interval, or a predictive distribution.
- [ ] The error is measured out of sample, and you say how it grows as the forecast moves further from the data.
- [ ] Residuals plotted against time or against the prediction, and a statement of whether the error looks like noise or like bias.

## 5. Decision

- [ ] An explicit rule from the prediction to the action, and why the loss picks that rule.
- [ ] How the action changes when the main inputs change.
- [ ] What a better model or more data would be worth, in the units of the loss.

## 6. Evaluation

- [ ] A held-out set the model never saw, and a calibration check.
- [ ] The method scored against a simple baseline in the units of the decision, not only RMSE or R².
- [ ] Where the method wins, and where it ties.

## Decisions

- [ ] `decisions.md` is the history: each choice, why, the alternative rejected, and what would change it. Old entries are still in the file.

## 7. Communicate

- [ ] `presentation.html` is a title slide plus the action, the reason, the risk with a number, and the value of being less wrong. It was rendered from `presentation.qmd` after the last edit to that file.
- [ ] The key effects are in plain language and in real units.
- [ ] The limits: where the model should not be trusted, what was assumed, and what you would do with more time or data.

## Repository

- [ ] Names and GitHub usernames are in `README.md`.
- [ ] A group reached `main` through pull requests, and each request names a `decisions.md` entry.
- [ ] `requirements.txt` pins the packages, and `.python-version` is the interpreter that ran the notebook. An R project uses `renv.lock` instead, which records the R version.
- [ ] The `data` release holds the data files, or `data/README.md` gives the command that downloads them from the source. Each file has a SHA-256 there, and the notebook checks it. The notebook on `main` was run from the top and committed with its outputs. Secrets and fitted weights are not in git.
- [ ] The split entry in `decisions.md` records the cutoff date, or the seed and the fraction.

## Disclosure

- [ ] One line in `README.md`: which AI tools you used, and for what.
