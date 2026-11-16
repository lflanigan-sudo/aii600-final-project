"""Gradient-boosted quantile regression beside the frozen linear q33 benchmark.

The booster is scikit-learn's HistGradientBoostingRegressor with quantile loss.
Two fitting scales are searched. A log-price fit matches the linear model: the
conditional quantile transforms back by exp. A dollar fit minimizes the pinball
loss the decision uses, and expensive watches can dominate the splits.

Every setting is chosen on the 802 validation sales from 2024 by dollar pinball
at tau = 1/3. That selected loss is not an untouched check. The 2025-2026
partition is not loaded. The linear model is refit only as a check.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts.fit_iqn import (
    CATEGORICAL,
    SEED,
    TAU,
    assemble_features,
    collapse_rare_models,
    fit_frozen_linear,
    load_partitions,
    pinball_loss,
    resolve_data_root,
    robustness_report,
)

MAX_ITER = 400
VALIDATION_FRACTION = 0.2
N_ITER_NO_CHANGE = 15
GRID = [
    {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
    }
    for learning_rate in (0.05, 0.1)
    for max_depth in (3, 6)
    for min_samples_leaf in (20, 50)
    for l2_regularization in (0.0, 1.0)
]


def categorical_frames(x_train, x_val):
    """Code categories from training only. An unseen validation level stays missing."""
    train = x_train.copy()
    val = x_val.copy()
    for col in CATEGORICAL:
        levels = sorted(train[col].unique())
        train[col] = pd.Categorical(train[col], categories=levels)
        known = val[col].isin(levels)
        val[col] = pd.Categorical(val[col].where(known), categories=levels)
    return train, val


def fit_booster(x_train, y_fit, params, quantile):
    """Pick the stopping iteration on a training holdout, then refit on all rows.

    The 2024 validation sales are not used to stop the trees. They are used
    only when the caller compares finished models.
    """
    search = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=MAX_ITER,
        early_stopping=True,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
        categorical_features="from_dtype",
        random_state=SEED,
        **params,
    )
    search.fit(x_train, y_fit)
    n_iter = int(search.n_iter_)
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=n_iter,
        early_stopping=False,
        categorical_features="from_dtype",
        random_state=SEED,
        **params,
    )
    model.fit(x_train, y_fit)
    return model, n_iter


def predict_dollars(model, frame, scale):
    raw = model.predict(frame)
    if scale == "log":
        return np.exp(raw)
    return raw


def search_scale(x_train, x_val, y_train, y_val, scale):
    y_fit = np.log(y_train) if scale == "log" else y_train
    rows = []
    best = None
    for params in GRID:
        model, n_iter = fit_booster(x_train, y_fit, params, TAU)
        prediction = predict_dollars(model, x_val, scale)
        loss = pinball_loss(y_val, prediction, TAU)
        row = {
            "scale": scale,
            "n_iter": n_iter,
            "validation_pinball_usd": loss,
            "nonpositive_predictions": int(np.sum(prediction <= 0)),
            **params,
        }
        rows.append(row)
        if best is None or loss < best["loss"]:
            best = {"loss": loss, "model": model, "n_iter": n_iter, "params": params, "prediction": prediction}
    table = pd.DataFrame(rows).sort_values("validation_pinball_usd").reset_index(drop=True)
    return table, best


def slice_report(y_val, prediction, linear_prediction, brands):
    rows = []
    masks = [("All", np.ones(len(y_val), dtype=bool))]
    for brand in ["Cartier", "Omega", "Rolex"]:
        masks.append((brand, brands.eq(brand).to_numpy()))
    for label, mask in masks:
        rows.append(
            {
                "slice": label,
                "n": int(mask.sum()),
                "gb_pinball_usd": pinball_loss(y_val[mask], prediction[mask], TAU),
                "linear_pinball_usd": pinball_loss(y_val[mask], linear_prediction[mask], TAU),
                "gb_fraction_below_q33": float(np.mean(y_val[mask] <= prediction[mask])),
                "linear_fraction_below_q33": float(np.mean(y_val[mask] <= linear_prediction[mask])),
            }
        )
    return pd.DataFrame(rows)


def main():
    import sklearn

    _root, eligible_path, assignment_path = resolve_data_root()
    train, validation, predictors = load_partitions(eligible_path, assignment_path)
    print("GRADIENT-BOOSTED QUANTILE REGRESSION")
    print("Data root rule: WATCH_DATA_DIR if set, otherwise <repository>/data.")
    print(f"Eligible file location: {eligible_path.parent.name}/{eligible_path.name}")
    print(f"scikit-learn {sklearn.__version__}")
    print(f"Grid size: {len(GRID)} settings on each of two scales. Seed {SEED}.")
    print("Test partition was not loaded.")

    x_train = assemble_features(train, predictors)
    x_val = assemble_features(validation, predictors)
    collapse_rare_models(x_train, x_val)
    x_train, x_val = categorical_frames(x_train, x_val)
    y_train = train["usd_price"].to_numpy(dtype=float)
    y_val = validation["usd_price"].to_numpy(dtype=float)

    linear_loss, linear_prediction = fit_frozen_linear(x_train, x_val, y_train, y_val)
    brand_q33 = train.groupby("brand")["usd_price"].quantile(TAU)
    baseline_prediction = validation["brand"].map(brand_q33).to_numpy(dtype=float)
    baseline_loss = pinball_loss(y_val, baseline_prediction, TAU)

    tables = []
    fitted = {}
    for scale in ("log", "dollar"):
        table, best = search_scale(x_train, x_val, y_train, y_val, scale)
        tables.append(table)
        fitted[scale] = best
        print(f"\n{scale.upper()} SCALE, 2024 DOLLAR PINBALL, BEST ROW FIRST")
        print(
            table.to_string(
                index=False,
                float_format=lambda value: f"{value:,.4f}" if isinstance(value, float) else str(value),
            )
        )

    search = pd.concat(tables, ignore_index=True).sort_values("validation_pinball_usd")
    winner_scale = search.iloc[0]["scale"]
    winner = fitted[winner_scale]
    train_prediction = predict_dollars(winner["model"], x_train, winner_scale)
    train_loss = pinball_loss(y_train, train_prediction, TAU)

    print("\nQ33 DOLLAR PINBALL ON THE SAME 802 VALIDATION SALES")
    comparison = pd.DataFrame(
        {
            "Method": [
                "Brand q33",
                "Linear q33",
                "Boosted q33, log scale",
                "Boosted q33, dollar scale",
            ],
            "Validation pinball loss (USD)": [
                baseline_loss,
                linear_loss,
                fitted["log"]["loss"],
                fitted["dollar"]["loss"],
            ],
        }
    )
    print(
        comparison.to_string(
            index=False,
            formatters={"Validation pinball loss (USD)": lambda value: f"{value:,.4f}"},
        )
    )
    params = winner["params"]
    print(
        "Lowest validation loss in this search: "
        f"{winner_scale} scale, learning rate {params['learning_rate']}, "
        f"max depth {params['max_depth']}, min leaf {params['min_samples_leaf']}, "
        f"l2 {params['l2_regularization']}, trees {winner['n_iter']}."
    )
    print(f"That model's training dollar pinball: {train_loss:,.4f}")
    print("The validation loss selected these settings. It is not an untouched check.")
    print(f"Nonpositive dollar-scale predictions on validation: {int(np.sum(fitted['dollar']['prediction'] <= 0))}")

    print("\nSELECTED BOOSTER AGAINST THE LINEAR MODEL")
    slices = slice_report(y_val, winner["prediction"], linear_prediction, validation["brand"])
    print(slices.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    print("\nPAIRED LOSS DIFFERENCE, LINEAR MINUS SELECTED BOOSTER, CLUSTERED BY AUCTION DATE")
    robust = robustness_report(
        y_val, linear_prediction, winner["prediction"], validation["brand"], validation["sale_date"]
    )
    print(f"Mean difference (USD): {robust['mean_difference']:,.4f}")
    print(f"95% date-cluster bootstrap interval: {robust['interval_low']:,.4f} to {robust['interval_high']:,.4f}")
    print(f"Auction dates: {robust['n_dates']}")
    print(f"Share of the total gain from Rolex: {robust['rolex_share']:.4f}")
    print(f"Validation sales where the booster loss is higher: {robust['sales_where_iqn_is_worse']}")

    print("\nQ10 AND Q90 REFIT WITH THE SELECTED Q33 SETTINGS")
    print("These two fits are a coverage check. Their settings were not chosen to improve coverage.")
    y_fit = np.log(y_train) if winner_scale == "log" else y_train
    endpoints = {}
    for quantile in (0.1, 0.9):
        model, n_iter = fit_booster(x_train, y_fit, params, quantile)
        endpoints[quantile] = predict_dollars(model, x_val, winner_scale)
        print(f"tau {quantile:.1f} stopped at {n_iter} trees.")
    q10 = endpoints[0.1]
    q33 = winner["prediction"]
    q90 = endpoints[0.9]
    crossing = (q10 > q33) | (q33 > q90) | (q10 > q90)
    ordered_low = np.minimum(q10, q90)
    ordered_high = np.maximum(q10, q90)
    print(f"Raw q10-q90 coverage: {float(np.mean((y_val >= q10) & (y_val <= q90))):.4f}")
    print(f"Coverage after ordering the endpoints: {float(np.mean((y_val >= ordered_low) & (y_val <= ordered_high))):.4f}")
    print(f"Quantile crossings: {int(crossing.sum())} ({float(np.mean(crossing)):.4f})")
    print("The linear q33 model stays the benchmark until a later decision says otherwise.")
    return {
        "baseline_loss": baseline_loss,
        "linear_loss": linear_loss,
        "log_loss": fitted["log"]["loss"],
        "dollar_loss": fitted["dollar"]["loss"],
        "winner_scale": winner_scale,
        "winner_params": params,
        "winner_n_iter": winner["n_iter"],
        "train_loss": train_loss,
        "slices": slices,
        "robust": robust,
        "search": search,
    }


if __name__ == "__main__":
    main()
