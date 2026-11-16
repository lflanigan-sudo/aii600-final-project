"""CatBoost quantile regression beside the frozen linear q33 benchmark.

CatBoost keeps brand, model, case, and movement as categories. The search below
does not collapse rare model names. A second fit, at the selected settings only,
collapses names with fewer than 15 training sales, matching the booster.

Settings are chosen on the 802 validation sales from 2024 by dollar pinball at
tau = 1/3. Tree count is chosen on a 20% holdout of the training sales. The
2025-2026 partition is not loaded. q10 and q90 reuse the selected depth, rate,
and penalty. Their own tree counts are stopped on that same training holdout.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from scripts.fit_gbqr import categorical_frames
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
HOLDOUT_FRACTION = 0.2
OD_WAIT = 15
GRID = [
    {"learning_rate": learning_rate, "depth": depth, "l2_leaf_reg": l2_leaf_reg}
    for learning_rate in (0.05, 0.1)
    for depth in (4, 6)
    for l2_leaf_reg in (1.0, 3.0)
]


def training_holdout(n_rows):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n_rows)
    n_hold = int(round(HOLDOUT_FRACTION * n_rows))
    return order[n_hold:], order[:n_hold]


def fit_quantile(x_train, y_fit, params, quantile, n_iter=None):
    """Stop on a training holdout, then refit on every training row.

    The 2024 sales are not an eval_set here. Passing them would use the
    selection sample to choose the tree count.
    """
    loss = f"Quantile:alpha={quantile:.10f}"
    common = dict(
        loss_function=loss,
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
        cat_features=CATEGORICAL,
    )
    if n_iter is None:
        fit_index, hold_index = training_holdout(len(x_train))
        search = CatBoostRegressor(iterations=MAX_ITER, od_type="Iter", od_wait=OD_WAIT, use_best_model=True, **common)
        search.fit(
            x_train.iloc[fit_index],
            y_fit[fit_index],
            eval_set=(x_train.iloc[hold_index], y_fit[hold_index]),
        )
        n_iter = int(search.tree_count_)
    model = CatBoostRegressor(iterations=n_iter, **common)
    model.fit(x_train, y_fit)
    return model, n_iter


def predict_dollars(model, frame, scale):
    raw = np.asarray(model.predict(frame), dtype=float)
    if scale == "log":
        return np.exp(raw)
    return raw


def search_scale(x_train, x_val, y_train, y_val, scale):
    y_fit = np.log(y_train) if scale == "log" else y_train
    rows = []
    best = None
    for params in GRID:
        model, n_iter = fit_quantile(x_train, y_fit, params, TAU)
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
            best = {
                "loss": loss,
                "model": model,
                "n_iter": n_iter,
                "params": params,
                "prediction": prediction,
                "scale": scale,
            }
    table = pd.DataFrame(rows).sort_values("validation_pinball_usd").reset_index(drop=True)
    return table, best


def slice_table(y_val, prediction, brands):
    rows = []
    masks = [("All", np.ones(len(y_val), dtype=bool))]
    for brand in ["Cartier", "Omega", "Rolex"]:
        masks.append((brand, brands.eq(brand).to_numpy()))
    for label, mask in masks:
        rows.append(
            {
                "slice": label,
                "n": int(mask.sum()),
                "pinball_usd": pinball_loss(y_val[mask], prediction[mask], TAU),
                "fraction_below_q33": float(np.mean(y_val[mask] <= prediction[mask])),
            }
        )
    return pd.DataFrame(rows)


def main():
    import catboost

    _root, eligible_path, assignment_path = resolve_data_root()
    train, validation, predictors = load_partitions(eligible_path, assignment_path)
    print("CATBOOST QUANTILE REGRESSION")
    print(f"catboost {catboost.__version__}")
    print("Rare model names are kept in the search. The 2025-2026 partition was not loaded.")
    print(f"Grid size: {len(GRID)} settings on each of two scales. Seed {SEED}.")
    x_train = assemble_features(train, predictors)
    x_val = assemble_features(validation, predictors)
    y_train = train["usd_price"].to_numpy(dtype=float)
    y_val = validation["usd_price"].to_numpy(dtype=float)

    collapsed_train = x_train.copy()
    collapsed_val = x_val.copy()
    collapse_rare_models(collapsed_train, collapsed_val)
    collapsed_train, collapsed_val = categorical_frames(collapsed_train, collapsed_val)
    linear_loss, linear_prediction = fit_frozen_linear(collapsed_train, collapsed_val, y_train, y_val)
    brand_q33 = train.groupby("brand")["usd_price"].quantile(TAU)
    baseline_prediction = validation["brand"].map(brand_q33).to_numpy(dtype=float)
    baseline_loss = pinball_loss(y_val, baseline_prediction, TAU)

    tables = []
    fitted = {}
    for scale in ("log", "dollar"):
        table, best = search_scale(x_train, x_val, y_train, y_val, scale)
        tables.append(table)
        fitted[scale] = best
        print(f"\n{scale.upper()} SCALE, RARE MODELS KEPT, BEST ROW FIRST")
        print(table.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    search = pd.concat(tables, ignore_index=True).sort_values("validation_pinball_usd")
    winner = fitted[search.iloc[0]["scale"]]
    params = winner["params"]
    scale = winner["scale"]
    y_fit = np.log(y_train) if scale == "log" else y_train
    train_prediction = predict_dollars(winner["model"], x_train, scale)
    train_loss = pinball_loss(y_train, train_prediction, TAU)

    print("\nQ33 DOLLAR PINBALL ON THE SAME 802 VALIDATION SALES")
    comparison = pd.DataFrame(
        {
            "Method": ["Brand q33", "Linear q33", "CatBoost q33, log scale", "CatBoost q33, dollar scale"],
            "Validation pinball loss (USD)": [
                baseline_loss,
                linear_loss,
                fitted["log"]["loss"],
                fitted["dollar"]["loss"],
            ],
        }
    )
    print(comparison.to_string(index=False, formatters={"Validation pinball loss (USD)": lambda value: f"{value:,.4f}"}))
    print(
        "Lowest validation loss in this search: "
        f"{scale} scale, learning rate {params['learning_rate']}, depth {params['depth']}, "
        f"l2 {params['l2_leaf_reg']}, trees {winner['n_iter']}."
    )
    print(f"That model's training dollar pinball: {train_loss:,.4f}")
    print(f"Recorded IQN pinball on these sales: 3,207.5994")
    print(f"Recorded gradient-booster pinball on these sales: 3,168.7295")
    print("The validation loss selected these settings. It is not an untouched check.")
    print(f"Nonpositive dollar-scale predictions: {int(np.sum(fitted['dollar']['prediction'] <= 0))}")

    print("\nSELECTED CATBOOST BY BRAND")
    print(slice_table(y_val, winner["prediction"], validation["brand"]).to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    print("\nPAIRED LOSS DIFFERENCE, LINEAR MINUS SELECTED CATBOOST, CLUSTERED BY AUCTION DATE")
    robust = robustness_report(y_val, linear_prediction, winner["prediction"], validation["brand"], validation["sale_date"])
    print(f"Mean difference (USD): {robust['mean_difference']:,.4f}")
    print(f"95% date-cluster bootstrap interval: {robust['interval_low']:,.4f} to {robust['interval_high']:,.4f}")
    print(f"Share of the total gain from Rolex: {robust['rolex_share']:.4f}")
    print(f"Validation sales where the CatBoost loss is higher: {robust['sales_where_iqn_is_worse']}")

    print("\nSAME SETTINGS AFTER COLLAPSING RARE MODEL NAMES")
    print("This fit is not a second search. It uses the settings chosen while rare names were kept.")
    collapsed_strings_train = x_train.copy()
    collapsed_strings_val = x_val.copy()
    collapse_rare_models(collapsed_strings_train, collapsed_strings_val)
    collapsed_model, collapsed_trees = fit_quantile(
        collapsed_strings_train, y_fit, params, TAU, n_iter=winner["n_iter"]
    )
    collapsed_prediction = predict_dollars(collapsed_model, collapsed_strings_val, scale)
    collapsed_loss = pinball_loss(y_val, collapsed_prediction, TAU)
    print(f"Collapsed-model validation pinball: {collapsed_loss:,.4f}")
    print(f"Collapsed-model fraction at or below q33: {float(np.mean(y_val <= collapsed_prediction)):.4f}")
    print(f"Trees refit: {collapsed_trees}")

    print("\nQ10 AND Q90 WITH THE SELECTED Q33 SETTINGS")
    print("Tree counts are stopped on the training holdout. They were not chosen to improve 2024 coverage.")
    endpoints = {}
    for quantile in (0.1, 0.9):
        model, n_iter = fit_quantile(x_train, y_fit, params, quantile)
        endpoints[quantile] = predict_dollars(model, x_val, scale)
        print(f"tau {quantile:.1f} stopped at {n_iter} trees.")
    q10 = endpoints[0.1]
    q33 = winner["prediction"]
    q90 = endpoints[0.9]
    crossing = (q10 > q33) | (q33 > q90) | (q10 > q90)
    print(f"Share at or below q10: {float(np.mean(y_val <= q10)):.4f}, target 0.10")
    print(f"Share at or below q33: {float(np.mean(y_val <= q33)):.4f}, target 0.3333")
    print(f"Share at or below q90: {float(np.mean(y_val <= q90)):.4f}, target 0.90")
    print(f"Raw q10-q90 coverage: {float(np.mean((y_val >= q10) & (y_val <= q90))):.4f}")
    print(f"Quantile crossings: {int(crossing.sum())} ({float(np.mean(crossing)):.4f})")
    print(f"Median interval width (USD): {float(np.median(q90 - q10)):,.2f}")
    print("The linear q33 model stays the benchmark until a later decision says otherwise.")
    return {
        "baseline_loss": baseline_loss,
        "linear_loss": linear_loss,
        "log_loss": fitted["log"]["loss"],
        "dollar_loss": fitted["dollar"]["loss"],
        "winner_scale": scale,
        "winner_params": params,
        "winner_n_iter": winner["n_iter"],
        "train_loss": train_loss,
        "collapsed_loss": collapsed_loss,
        "robust": robust,
        "search": search,
    }


if __name__ == "__main__":
    main()
