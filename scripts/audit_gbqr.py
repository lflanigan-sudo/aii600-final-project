"""Audit the gradient booster against the IQN and the linear q33 model.

The selected booster settings stay fixed. Nothing in this script chooses a new
setting from a later window, and nothing here replaces the linear benchmark.
Windows that use 2024, 2025, or 2026 are a sensitivity analysis: those sales
were already in the development file.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from scripts.fit_gbqr import categorical_frames, fit_booster, predict_dollars
from scripts.fit_iqn import (
    CATEGORICAL,
    FEATURE_COLS,
    SEED,
    TAU,
    FeatureCoder,
    assemble_features,
    collapse_rare_models,
    load_partitions,
    pinball_loss,
    predict_usd,
    resolve_data_root,
    row_pinball,
    train_one,
)
from scripts.predictors import INPUT_SHA256, construct_record

FROZEN_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}
FROZEN_TREES = 112
FROZEN_LINEAR_LOSS = 3975.7439
FROZEN_GB_LOSS = 3168.7295
FROZEN_IQN_LOSS = 3207.5994
IQN_WIDTH = 32
IQN_LEARNING_RATE = 0.003
IQN_WEIGHT_DECAY = 0.0001


def two_to_one_loss(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    over = np.maximum(predicted - actual, 0.0)
    under = np.maximum(actual - predicted, 0.0)
    return float(np.mean(2.0 * over + under))


def price_table(frame, label):
    price = frame["usd_price"].to_numpy(dtype=float)
    quantiles = np.quantile(price, [0.1, 0.33, 0.5, 0.9])
    return {
        "slice": label,
        "n": int(len(price)),
        "median": float(quantiles[2]),
        "p10": float(quantiles[0]),
        "p33": float(quantiles[1]),
        "p90": float(quantiles[3]),
        "mean": float(np.mean(price)),
        "max": float(np.max(price)),
        "share_above_50k": float(np.mean(price > 50_000)),
    }


def load_assigned(eligible_path, assignment_path):
    import csv
    import hashlib

    digest = hashlib.sha256(eligible_path.read_bytes()).hexdigest()
    if digest != INPUT_SHA256:
        raise RuntimeError("Eligible file checksum does not match the approved file.")
    with eligible_path.open(newline="", encoding="utf-8-sig") as handle:
        eligible_rows = list(csv.DictReader(handle))
    with assignment_path.open(newline="", encoding="utf-8-sig") as handle:
        assignments = list(csv.DictReader(handle))
    eligible_by_id = {row["record_id"]: row for row in eligible_rows}
    rows = []
    predictors = {}
    for assignment in assignments:
        record_id = assignment["record_id"]
        source = eligible_by_id[record_id]
        if source["designer"] != assignment["brand"]:
            raise RuntimeError(f"Brand mismatch for record {record_id}")
        predictors[record_id] = construct_record(source)
        rows.append(
            {
                "record_id": record_id,
                "brand": assignment["brand"],
                "partition": assignment["partition"],
                "sale_date": assignment["sale_date"],
                "usd_price": float(source["usd_price_decimal_num"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["sale_date"] = pd.to_datetime(frame["sale_date"])
    frame["year"] = frame["sale_date"].dt.year
    return frame, predictors


def design(train_df, eval_df, predictors):
    x_train = assemble_features(train_df, predictors)
    x_eval = assemble_features(eval_df, predictors)
    collapse_rare_models(x_train, x_eval)
    x_train_cat, x_eval_cat = categorical_frames(x_train, x_eval)
    return x_train, x_eval, x_train_cat, x_eval_cat


def fit_fixed_booster(x_train, y_train, quantile=TAU, n_trees=FROZEN_TREES):
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=n_trees,
        early_stopping=False,
        categorical_features="from_dtype",
        random_state=SEED,
        **FROZEN_PARAMS,
    )
    model.fit(x_train, np.log(y_train))
    return model


def fit_linear_window(x_train, x_eval, y_train):
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("age", SimpleImputer(strategy="median", add_indicator=True), ["age_at_sale"]),
        ]
    )
    model = Pipeline(
        [
            ("features", preprocessor),
            ("quantile", QuantileRegressor(quantile=TAU, alpha=0.01, solver="highs")),
        ]
    )
    model.fit(x_train, np.log(y_train))
    return np.exp(model.predict(x_eval)), np.exp(model.predict(x_train))


def date_cluster_mean(values, dates, draws=2000):
    codes, _unique = pd.factorize(dates, sort=False)
    groups = [np.flatnonzero(codes == code) for code in range(int(codes.max()) + 1)]
    rng = np.random.default_rng(SEED)
    replicates = np.empty(draws)
    n_dates = len(groups)
    for draw in range(draws):
        chosen = rng.integers(0, n_dates, n_dates)
        sampled = np.concatenate([groups[index] for index in chosen])
        replicates[draw] = values[sampled].mean()
    low, high = np.quantile(replicates, [0.025, 0.975])
    return float(values.mean()), float(low), float(high), n_dates


def conformal_cutoff(scores, alpha=0.2):
    ordered = np.sort(np.asarray(scores, dtype=float))
    n_cal = len(ordered)
    rank = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n_cal)
    return float(ordered[rank - 1]), rank, n_cal


def coverage(y, low, high):
    return float(np.mean((y >= low) & (y <= high)))


def print_frame(frame):
    print(
        frame.to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}" if abs(value) < 100 else f"{value:,.2f}",
        )
    )


def audit_gap(train, validation, x_train, x_val, x_train_cat, x_val_cat, y_train, y_val, gb_model):
    print("\n1. TRAINING VERSUS VALIDATION LOSS")
    train_prediction = np.exp(gb_model.predict(x_train_cat))
    val_prediction = np.exp(gb_model.predict(x_val_cat))
    train_pinball = pinball_loss(y_train, train_prediction, TAU)
    val_pinball = pinball_loss(y_val, val_prediction, TAU)
    train_two = two_to_one_loss(y_train, train_prediction)
    val_two = two_to_one_loss(y_val, val_prediction)
    print(f"Fixed-setting validation pinball: {val_pinball:,.4f}")
    print(f"Recorded search pinball: {FROZEN_GB_LOSS:,.4f}")
    if abs(val_pinball - FROZEN_GB_LOSS) > 0.05:
        raise RuntimeError("Fixed 112-tree booster does not reproduce the recorded validation loss.")
    print(f"Training pinball: {train_pinball:,.4f}")
    print(f"Validation pinball: {val_pinball:,.4f}")
    print(f"Training 2:1 mean loss: {train_two:,.4f}")
    print(f"Validation 2:1 mean loss: {val_two:,.4f}")
    print(
        "Pinball equals the 2:1 mean loss divided by 3: "
        f"train {train_pinball / train_two:.6f}, validation {val_pinball / val_two:.6f}."
    )
    print("The reported score is mean pinball, not the sum, and not the raw 2:1 loss.")

    rows = [price_table(train, "Train 2021-2023"), price_table(validation, "Validation 2024")]
    for brand in ["Cartier", "Omega", "Rolex"]:
        rows.append(price_table(train.loc[train["brand"].eq(brand)], f"Train {brand}"))
        rows.append(price_table(validation.loc[validation["brand"].eq(brand)], f"Validation {brand}"))
    print("\nPRICE DISTRIBUTION")
    print_frame(pd.DataFrame(rows))

    brand_rows = []
    for brand in ["Cartier", "Omega", "Rolex"]:
        train_mask = train["brand"].eq(brand).to_numpy()
        val_mask = validation["brand"].eq(brand).to_numpy()
        brand_rows.append(
            {
                "brand": brand,
                "train_pinball": pinball_loss(y_train[train_mask], train_prediction[train_mask], TAU),
                "val_pinball": pinball_loss(y_val[val_mask], val_prediction[val_mask], TAU),
                "train_pinball_over_median": pinball_loss(y_train[train_mask], train_prediction[train_mask], TAU)
                / float(train.loc[train_mask, "usd_price"].median()),
                "val_pinball_over_median": pinball_loss(y_val[val_mask], val_prediction[val_mask], TAU)
                / float(validation.loc[val_mask, "usd_price"].median()),
            }
        )
    print("\nPINBALL BY BRAND, AND PINBALL DIVIDED BY THAT SLICE'S MEDIAN PRICE")
    print_frame(pd.DataFrame(brand_rows))
    top_positions = np.argsort(y_train)[-5:]
    kept = np.ones(len(y_train), dtype=bool)
    kept[top_positions] = False
    capped = y_train <= float(np.max(y_val))
    print(
        f"Five highest training prices: {', '.join(f'{price:,.0f}' for price in np.sort(y_train[top_positions])[::-1])}"
    )
    print(f"Training pinball without those five sales: {pinball_loss(y_train[kept], train_prediction[kept], TAU):,.4f}")
    print(
        "Training pinball on sales at or below the validation maximum "
        f"{float(np.max(y_val)):,.0f}: {pinball_loss(y_train[capped], train_prediction[capped], TAU):,.4f}"
    )
    print(f"Share of training pinball sum from those five sales: {float(row_pinball(y_train[top_positions], train_prediction[top_positions], TAU).sum() / row_pinball(y_train, train_prediction, TAU).sum()):.4f}")

    collapsed = int((x_val["norm_model"] == "Other").sum())
    age_missing_train = int(x_train["age_at_sale"].isna().sum())
    age_missing_val = int(x_val["age_at_sale"].isna().sum())
    unseen = 0
    for col in CATEGORICAL:
        train_levels = set(x_train[col].astype(str))
        unseen += int((~x_val[col].astype(str).isin(train_levels)).sum())
    print("\nPREPROCESSING ON THE OFFICIAL SPLIT")
    print(f"Validation rows collapsed to Other model: {collapsed} of {len(x_val)}")
    print(f"Age missing: train {age_missing_train}, validation {age_missing_val}")
    print(f"Validation category cells whose level is absent from training: {unseen}")
    print("Blank source categories are the string Missing before either model is fit.")
    print("The booster leaves missing age as missing. The IQN median-imputes age, standardizes it, and adds a missing indicator.")
    print("The log booster minimizes pinball on log price. The IQN minimizes pinball on the dollar price. Both are scored below in dollars.")
    return val_prediction, train_prediction


def compare_models(
    train, validation, x_train, x_val, y_train, y_val, gb_prediction, linear_prediction, train_linear_prediction
):
    print("\n2. BOOSTER VERSUS IQN ON THE SAME 802 SALES")
    coder = FeatureCoder().fit(x_train)
    rng = np.random.default_rng(SEED + IQN_WIDTH + int(IQN_LEARNING_RATE * 1e6) + int(IQN_WEIGHT_DECAY * 1e8))
    chosen = train_one(
        coder.transform(x_train),
        y_train,
        coder.transform(x_val),
        y_val,
        IQN_WIDTH,
        IQN_LEARNING_RATE,
        IQN_WEIGHT_DECAY,
        rng,
    )
    if chosen is None:
        raise RuntimeError("Frozen IQN diverged.")
    iqn_prediction = predict_usd(coder.transform(x_val), TAU, chosen["params"])
    iqn_loss = pinball_loss(y_val, iqn_prediction, TAU)
    gb_loss = pinball_loss(y_val, gb_prediction, TAU)
    print(
        f"Reproduced IQN: width {chosen['width']}, lr {chosen['learning_rate']}, "
        f"decay {chosen['weight_decay']}, epoch {chosen['epoch']}, pinball {iqn_loss:,.4f}"
    )
    print(f"Recorded IQN pinball: {FROZEN_IQN_LOSS:,.4f}")
    if abs(iqn_loss - FROZEN_IQN_LOSS) > 0.05:
        raise RuntimeError("Reproduced IQN loss does not match the recorded loss.")
    difference = row_pinball(y_val, iqn_prediction, TAU) - row_pinball(y_val, gb_prediction, TAU)
    mean_diff, low, high, n_dates = date_cluster_mean(difference, validation["sale_date"])
    print(f"Mean IQN pinball minus booster pinball: {mean_diff:,.4f}")
    print(f"95% date-cluster interval: {low:,.4f} to {high:,.4f} over {n_dates} auction dates")
    print(f"Median paired difference: {float(np.median(difference)):,.4f}")
    print(f"Sales where the booster loss is lower: {int(np.sum(difference > 0))}")
    print(f"Sales where the IQN loss is lower: {int(np.sum(difference < 0))}")
    print(f"Sales where the losses tie: {int(np.sum(np.isclose(difference, 0.0)))}")
    linear_loss = pinball_loss(y_val, linear_prediction, TAU)
    improvement = (linear_loss - gb_loss) / linear_loss
    print(f"Booster improvement versus the linear model: {100 * improvement:.2f}%")
    print(f"Linear validation pinball: {linear_loss:,.4f}")

    brand_rows = []
    for brand in ["Cartier", "Omega", "Rolex"]:
        mask = validation["brand"].eq(brand).to_numpy()
        brand_rows.append(
            {
                "slice": brand,
                "n": int(mask.sum()),
                "iqn_pinball": pinball_loss(y_val[mask], iqn_prediction[mask], TAU),
                "gb_pinball": pinball_loss(y_val[mask], gb_prediction[mask], TAU),
                "mean_iqn_minus_gb": float(difference[mask].mean()),
                "gb_wins": int(np.sum(difference[mask] > 0)),
            }
        )
    print("\nPAIRED LOSS BY BRAND")
    print_frame(pd.DataFrame(brand_rows))

    cuts = np.quantile(train_linear_prediction, [1.0 / 3.0, 2.0 / 3.0])
    band = np.where(
        linear_prediction <= cuts[0],
        "low predicted price",
        np.where(linear_prediction <= cuts[1], "middle predicted price", "high predicted price"),
    )
    print(f"\nPREDICTED-PRICE BANDS FROM THE LINEAR TRAINING Q33: {cuts[0]:,.2f} and {cuts[1]:,.2f}")
    print("These bands use the prediction available before the sale. They do not use the realized price.")
    band_rows = []
    for label in ["low predicted price", "middle predicted price", "high predicted price"]:
        mask = band == label
        band_rows.append(
            {
                "slice": label,
                "n": int(mask.sum()),
                "iqn_pinball": pinball_loss(y_val[mask], iqn_prediction[mask], TAU),
                "gb_pinball": pinball_loss(y_val[mask], gb_prediction[mask], TAU),
                "mean_iqn_minus_gb": float(difference[mask].mean()),
                "gb_wins": int(np.sum(difference[mask] > 0)),
            }
        )
    print_frame(pd.DataFrame(band_rows))

    price_cuts = np.quantile(train["usd_price"], [1.0 / 3.0, 2.0 / 3.0])
    realized = np.where(
        y_val <= price_cuts[0],
        "low realized price",
        np.where(y_val <= price_cuts[1], "middle realized price", "high realized price"),
    )
    print(f"\nREALIZED-PRICE BANDS FROM THE TRAINING PRICE: {price_cuts[0]:,.2f} and {price_cuts[1]:,.2f}")
    print("A realized-price band conditions on the outcome. It describes where the dollar errors landed.")
    realized_rows = []
    for label in ["low realized price", "middle realized price", "high realized price"]:
        mask = realized == label
        realized_rows.append(
            {
                "slice": label,
                "n": int(mask.sum()),
                "iqn_pinball": pinball_loss(y_val[mask], iqn_prediction[mask], TAU),
                "gb_pinball": pinball_loss(y_val[mask], gb_prediction[mask], TAU),
                "mean_iqn_minus_gb": float(difference[mask].mean()),
            }
        )
    print_frame(pd.DataFrame(realized_rows))

    models = x_val["norm_model"].astype(str).to_numpy()
    model_rows = []
    for name in sorted(set(models)):
        mask = models == name
        if int(mask.sum()) < 15:
            continue
        model_rows.append(
            {
                "model": name,
                "n": int(mask.sum()),
                "mean_iqn_minus_gb": float(difference[mask].mean()),
                "gb_pinball": pinball_loss(y_val[mask], gb_prediction[mask], TAU),
                "iqn_pinball": pinball_loss(y_val[mask], iqn_prediction[mask], TAU),
            }
        )
    model_table = pd.DataFrame(model_rows).sort_values("mean_iqn_minus_gb")
    print("\nMODEL NAMES WITH AT LEAST 15 VALIDATION SALES")
    print("Positive mean_iqn_minus_gb means the booster has the lower loss.")
    print_frame(model_table)
    return {
        "iqn_loss": iqn_loss,
        "gb_loss": gb_loss,
        "mean_diff": mean_diff,
        "interval": (low, high),
        "improvement": improvement,
        "iqn_epoch": chosen["epoch"],
        "linear_loss": linear_loss,
    }


def temporal(frame, predictors):
    print("\n3. TEMPORAL SENSITIVITY, FIXED SETTINGS, NO RETUNING")
    print(
        "Settings: log price, learning rate 0.1, depth 6, minimum leaf 20, "
        f"l2 0, {FROZEN_TREES} trees, seed {SEED}."
    )
    print("The linear reference keeps alpha 0.01 and tau 1/3. Neither model is retuned on the window result.")
    print("This is a sensitivity analysis. The file, including 2025-2026, was already used in development.")
    windows = [
        ("Train 2021, evaluate 2022", frame["year"].eq(2021), frame["year"].eq(2022)),
        ("Train 2021-2022, evaluate 2023", frame["year"].le(2022), frame["year"].eq(2023)),
        ("Train 2021-2023, evaluate 2024", frame["year"].le(2023), frame["year"].eq(2024)),
        ("Train through 2024, evaluate 2025", frame["year"].le(2024), frame["year"].eq(2025)),
        ("Train through 2025, evaluate 2026", frame["year"].le(2025), frame["year"].eq(2026)),
    ]
    rows = []
    for label, train_mask, eval_mask in windows:
        train_df = frame.loc[train_mask].reset_index(drop=True)
        eval_df = frame.loc[eval_mask].reset_index(drop=True)
        x_train, x_eval, x_train_cat, x_eval_cat = design(train_df, eval_df, predictors)
        y_train = train_df["usd_price"].to_numpy(dtype=float)
        y_eval = eval_df["usd_price"].to_numpy(dtype=float)
        booster = fit_fixed_booster(x_train_cat, y_train)
        gb_prediction = np.exp(booster.predict(x_eval_cat))
        linear_prediction, _train_prediction = fit_linear_window(x_train, x_eval, y_train)
        gb_loss = pinball_loss(y_eval, gb_prediction, TAU)
        linear_loss = pinball_loss(y_eval, linear_prediction, TAU)
        rows.append(
            {
                "window": label,
                "n_train": len(train_df),
                "n_eval": len(eval_df),
                "gb_pinball": gb_loss,
                "linear_pinball": linear_loss,
                "gb_minus_linear": gb_loss - linear_loss,
                "gb_fraction_below_q33": float(np.mean(y_eval <= gb_prediction)),
                "linear_fraction_below_q33": float(np.mean(y_eval <= linear_prediction)),
            }
        )
        print(
            f"{label}: train {len(train_df)}, eval {len(eval_df)}, "
            f"booster {gb_loss:,.2f}, linear {linear_loss:,.2f}, "
            f"booster below q33 {np.mean(y_eval <= gb_prediction):.4f}"
        )
    print_frame(pd.DataFrame(rows))
    return rows


def interval_audit(train, validation, x_train_cat, x_val_cat, y_train, y_val, frame, predictors):
    print("\n4. INTERVAL COVERAGE")
    endpoints = {"0.33": np.exp(fit_fixed_booster(x_train_cat, y_train).predict(x_val_cat))}
    train_endpoints = {"0.33": np.exp(fit_fixed_booster(x_train_cat, y_train).predict(x_train_cat))}
    for quantile, name in ((0.1, "0.10"), (0.9, "0.90")):
        model, n_iter = fit_booster(x_train_cat, np.log(y_train), FROZEN_PARAMS, quantile)
        endpoints[name] = predict_dollars(model, x_val_cat, "log")
        train_endpoints[name] = predict_dollars(model, x_train_cat, "log")
        print(f"tau {quantile:.2f} early-stopped at {n_iter} trees, using the q33 depth, rate, leaf, and penalty.")
    q10, q33, q90 = endpoints["0.10"], endpoints["0.33"], endpoints["0.90"]
    print(f"Validation share at or below q10: {float(np.mean(y_val <= q10)):.4f}, target 0.10")
    print(f"Validation share at or below q33: {float(np.mean(y_val <= q33)):.4f}, target 0.3333")
    print(f"Validation share at or below q90: {float(np.mean(y_val <= q90)):.4f}, target 0.90")
    print(f"Validation raw q10-q90 coverage: {coverage(y_val, q10, q90):.4f}")
    print(f"Training raw q10-q90 coverage: {coverage(y_train, train_endpoints['0.10'], train_endpoints['0.90']):.4f}")
    print(f"Median validation width: {float(np.median(q90 - q10)):,.2f}")
    crossing = (q10 > q33) | (q33 > q90) | (q10 > q90)
    print(f"Validation crossings: {int(crossing.sum())} ({float(np.mean(crossing)):.4f})")
    tail_rows = []
    for brand in ["All", "Cartier", "Omega", "Rolex"]:
        mask = np.ones(len(y_val), dtype=bool) if brand == "All" else validation["brand"].eq(brand).to_numpy()
        tail_rows.append(
            {
                "slice": brand,
                "n": int(mask.sum()),
                "below_q10": float(np.mean(y_val[mask] <= q10[mask])),
                "below_q90": float(np.mean(y_val[mask] <= q90[mask])),
                "coverage": coverage(y_val[mask], q10[mask], q90[mask]),
                "median_width": float(np.median(q90[mask] - q10[mask])),
            }
        )
    print_frame(pd.DataFrame(tail_rows))

    print("\nSEPARATE CALIBRATION: FIT BEFORE 2023, CALIBRATE ON 2023, EVALUATE ON 2024")
    print("Tree settings stay fixed at 112 trees. 2024 is not used to choose the cutoff.")
    fit_df = frame.loc[frame["year"].lt(2023)].reset_index(drop=True)
    cal_df = frame.loc[frame["year"].eq(2023)].reset_index(drop=True)
    eval_df = frame.loc[frame["year"].eq(2024)].reset_index(drop=True)
    print(f"Fit {len(fit_df)}, calibrate {len(cal_df)}, evaluate {len(eval_df)}")
    y_fit = fit_df["usd_price"].to_numpy(dtype=float)
    y_cal = cal_df["usd_price"].to_numpy(dtype=float)
    y_eval = eval_df["usd_price"].to_numpy(dtype=float)
    x_fit, x_cal, x_fit_cat, x_cal_cat = design(fit_df, cal_df, predictors)
    x_eval = assemble_features(eval_df, predictors)
    collapse_rare_models(x_fit, x_eval)
    x_fit_cat, x_eval_cat = categorical_frames(x_fit, x_eval)
    low_model = fit_fixed_booster(x_fit_cat, y_fit, quantile=0.1)
    high_model = fit_fixed_booster(x_fit_cat, y_fit, quantile=0.9)
    cal_low = np.exp(low_model.predict(x_cal_cat))
    cal_high = np.exp(high_model.predict(x_cal_cat))
    eval_low = np.exp(low_model.predict(x_eval_cat))
    eval_high = np.exp(high_model.predict(x_eval_cat))
    scores = np.maximum(cal_low - y_cal, y_cal - cal_high)
    cutoff, rank, n_cal = conformal_cutoff(scores, alpha=0.2)
    adjusted_low = eval_low - cutoff
    adjusted_high = eval_high + cutoff
    print(f"Uncalibrated 2024 coverage of the pre-2023 fit: {coverage(y_eval, eval_low, eval_high):.4f}")
    print(f"2023 raw coverage: {coverage(y_cal, cal_low, cal_high):.4f}")
    print(f"Conformal cutoff at rank {rank} of {n_cal}: {cutoff:,.2f} dollars")
    print(f"Calibrated 2024 coverage: {coverage(y_eval, adjusted_low, adjusted_high):.4f}")
    print(f"Median uncalibrated 2024 width: {float(np.median(eval_high - eval_low)):,.2f}")
    print(f"Median calibrated 2024 width: {float(np.median(adjusted_high - adjusted_low)):,.2f}")
    print(f"2024 share below the uncalibrated q10: {float(np.mean(y_eval <= eval_low)):.4f}")
    print(f"2024 share below the uncalibrated q90: {float(np.mean(y_eval <= eval_high)):.4f}")
    return {
        "below_q10": float(np.mean(y_val <= q10)),
        "below_q90": float(np.mean(y_val <= q90)),
        "raw_coverage": coverage(y_val, q10, q90),
        "train_coverage": coverage(y_train, train_endpoints["0.10"], train_endpoints["0.90"]),
        "calibrated_coverage": coverage(y_eval, adjusted_low, adjusted_high),
        "cutoff": cutoff,
    }


def main():
    _root, eligible_path, assignment_path = resolve_data_root()
    frame, predictors = load_assigned(eligible_path, assignment_path)
    print("AUDIT OF THE GRADIENT BOOSTER")
    print(f"Assigned sales: {len(frame)}. Years: {sorted(frame['year'].unique().tolist())}")
    train, validation, _predictors = load_partitions(eligible_path, assignment_path)
    x_train, x_val, x_train_cat, x_val_cat = design(train, validation, predictors)
    y_train = train["usd_price"].to_numpy(dtype=float)
    y_val = validation["usd_price"].to_numpy(dtype=float)
    for col in FEATURE_COLS:
        if col not in x_train.columns:
            raise RuntimeError(f"Missing predictor column {col}")
    gb_model = fit_fixed_booster(x_train_cat, y_train)
    gb_prediction, _train_prediction = audit_gap(
        train, validation, x_train, x_val, x_train_cat, x_val_cat, y_train, y_val, gb_model
    )
    linear_prediction, train_linear_prediction = fit_linear_window(x_train, x_val, y_train)
    linear_loss = pinball_loss(y_val, linear_prediction, TAU)
    print(f"\nRecomputed linear validation pinball: {linear_loss:,.4f}")
    if abs(linear_loss - FROZEN_LINEAR_LOSS) > 0.05:
        raise RuntimeError("Linear window fit does not reproduce the frozen validation loss.")
    compared = compare_models(
        train,
        validation,
        x_train,
        x_val,
        y_train,
        y_val,
        gb_prediction,
        linear_prediction,
        train_linear_prediction,
    )
    windows = temporal(frame, predictors)
    intervals = interval_audit(
        train, validation, x_train_cat, x_val_cat, y_train, y_val, frame, predictors
    )
    print("\nThe linear q33 model remains the reference. This audit does not retune it.")
    return {"compared": compared, "windows": windows, "intervals": intervals}


if __name__ == "__main__":
    main()
