"""Closing checks: 2026 calibration slices, screening gaps, and one interval.

The gradient-booster settings stay fixed. The 2026 sales and the 2024
validation sales were already in the development file. Nothing here retunes q33
or chooses a new feature set.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_gbqr import (
    FROZEN_GB_LOSS,
    conformal_cutoff,
    coverage,
    fit_fixed_booster,
    fit_linear_window,
    load_assigned,
)
from scripts.finalize_review import MARGIN, load_estimates
from scripts.fit_gbqr import categorical_frames
from scripts.fit_iqn import (
    TAU,
    assemble_features,
    collapse_rare_models,
    pinball_loss,
    resolve_data_root,
)
from scripts.predictors import INPUT_SHA256

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "requirements.txt",
    "scripts/fit_gbqr.py",
    "scripts/fit_iqn.py",
    "scripts/fit_catboost.py",
    "scripts/audit_gbqr.py",
    "scripts/finalize_review.py",
    "scripts/feature_experiments.py",
    "scripts/predictors.py",
    "scripts/split_chronological.py",
    "data/chronological_split_assignments.csv",
    "data/expanded_pull_eligible.csv",
]


def encode(train_df, eval_df, predictors):
    x_train = assemble_features(train_df, predictors)
    x_eval = assemble_features(eval_df, predictors)
    collapse_rare_models(x_train, x_eval)
    x_train_cat, x_eval_cat = categorical_frames(x_train, x_eval)
    return x_train, x_eval, x_train_cat, x_eval_cat


def slice_row(label, mask, actual, booster, linear):
    if int(mask.sum()) == 0:
        return None
    return {
        "slice": label,
        "n": int(mask.sum()),
        "booster_pinball": pinball_loss(actual[mask], booster[mask], TAU),
        "linear_pinball": pinball_loss(actual[mask], linear[mask], TAU),
        "booster_at_or_below_q33": float(np.mean(actual[mask] <= booster[mask])),
        "linear_at_or_below_q33": float(np.mean(actual[mask] <= linear[mask])),
    }


def print_frame(frame):
    print(frame.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))


def calibration_2026(frame, predictors):
    print("2026 CALIBRATION BY BRAND, MODEL, AND PREDICTED-PRICE SEGMENT")
    train_df = frame.loc[frame["year"].le(2025)].reset_index(drop=True)
    eval_df = frame.loc[frame["year"].eq(2026)].reset_index(drop=True)
    if train_df["sale_date"].max() >= eval_df["sale_date"].min():
        raise RuntimeError("2026 training sales are not strictly earlier.")
    if set(train_df["record_id"]) & set(eval_df["record_id"]):
        raise RuntimeError("A 2026 sale is also in training.")
    x_train, x_eval, x_train_cat, x_eval_cat = encode(train_df, eval_df, predictors)
    y_train = train_df["usd_price"].to_numpy(dtype=float)
    actual = eval_df["usd_price"].to_numpy(dtype=float)
    booster_model = fit_fixed_booster(x_train_cat, y_train)
    booster = np.exp(booster_model.predict(x_eval_cat))
    train_prediction = np.exp(booster_model.predict(x_train_cat))
    linear, _train_linear = fit_linear_window(x_train, x_eval, y_train)
    print(
        f"Train {len(train_df)} through {train_df['sale_date'].max().date().isoformat()}. "
        f"Evaluate {len(eval_df)} from {eval_df['sale_date'].min().date().isoformat()} "
        f"to {eval_df['sale_date'].max().date().isoformat()}."
    )
    print(f"Booster pinball {pinball_loss(actual, booster, TAU):,.4f}, share {float(np.mean(actual <= booster)):.4f}")
    print(f"Linear pinball {pinball_loss(actual, linear, TAU):,.4f}, share {float(np.mean(actual <= linear)):.4f}")
    rows = [slice_row("All", np.ones(len(actual), dtype=bool), actual, booster, linear)]
    brands = x_eval["brand"].to_numpy()
    for brand in ["Cartier", "Omega", "Rolex"]:
        rows.append(slice_row(brand, brands == brand, actual, booster, linear))
    cuts = np.quantile(train_prediction, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.where(booster <= cuts[0], "Low predicted", np.where(booster <= cuts[1], "Middle predicted", "High predicted"))
    print(f"Predicted-price cuts from the 2021-2025 booster fit: {cuts[0]:,.2f} and {cuts[1]:,.2f}")
    for label in ["Low predicted", "Middle predicted", "High predicted"]:
        rows.append(slice_row(label, labels == label, actual, booster, linear))
    models = x_eval["norm_model"].to_numpy()
    counts = pd.Series(models).value_counts()
    for model_name, count in counts.items():
        if count < 15:
            continue
        rows.append(slice_row(f"Model: {model_name}", models == model_name, actual, booster, linear))
    print_frame(pd.DataFrame(rows))
    print("A share below 33.33% means the predicted q33 sat too low in that slice.")
    print("Model names with fewer than 15 training sales are collapsed to Other. Slices below 15 sales in 2026 are omitted.")


def gap_table(group, mask, price, benchmark, booster, linear):
    excess = price[mask] - benchmark[mask]
    relative = excess / benchmark[mask]
    booster_error = booster[mask] - price[mask]
    linear_error = linear[mask] - price[mask]
    excess_q = np.quantile(excess, [0.1, 0.25, 0.5, 0.75, 0.9])
    relative_q = np.quantile(relative, [0.1, 0.25, 0.5, 0.75, 0.9])
    booster_q = np.quantile(booster_error, [0.25, 0.5, 0.75])
    linear_q = np.quantile(linear_error, [0.25, 0.5, 0.75])
    return {
        "group": group,
        "n": int(mask.sum()),
        "share_above_benchmark": float(np.mean(price[mask] > benchmark[mask])),
        "share_above_10pct": float(np.mean(price[mask] > benchmark[mask] * 1.10)),
        "share_above_25pct": float(np.mean(price[mask] > benchmark[mask] * 1.25)),
        "excess_p10": float(excess_q[0]),
        "excess_p25": float(excess_q[1]),
        "excess_p50": float(excess_q[2]),
        "excess_p75": float(excess_q[3]),
        "excess_p90": float(excess_q[4]),
        "relative_p50": float(relative_q[2]),
        "booster_error_p25": float(booster_q[0]),
        "booster_error_p50": float(booster_q[1]),
        "booster_error_p75": float(booster_q[2]),
        "linear_error_p50": float(linear_q[1]),
    }


def screening_gaps(frame, predictors, eligible_path):
    print("\nSCREENING GAPS AGAINST THE CATALOGUE RANGE")
    print("A flag is predicted q33 at least 10% above the benchmark. The margin is illustrative.")
    print("Recorded sale price minus benchmark is computed after the flag. It is not a purchasing profit.")
    training = frame.loc[frame["partition"].eq("train")].reset_index(drop=True)
    validation = frame.loc[frame["partition"].eq("validation")].reset_index(drop=True)
    estimates = load_estimates(eligible_path)
    validation = validation.merge(estimates, on="record_id", how="left")
    usable = validation["low_usd"].gt(0) & validation["high_usd"].gt(0) & validation["low_usd"].le(validation["high_usd"])
    held = validation.loc[usable].reset_index(drop=True)
    x_train, x_eval, x_train_cat, x_eval_cat = encode(training, held, predictors)
    y_train = training["usd_price"].to_numpy(dtype=float)
    booster = np.exp(fit_fixed_booster(x_train_cat, y_train).predict(x_eval_cat))
    linear, _train_linear = fit_linear_window(x_train, x_eval, y_train)
    price = held["usd_price"].to_numpy(dtype=float)
    benchmarks = {
        "midpoint": ((held["low_usd"] + held["high_usd"]) / 2.0).to_numpy(dtype=float),
        "low estimate": held["low_usd"].to_numpy(dtype=float),
    }
    for name, benchmark in benchmarks.items():
        linear_flag = linear >= benchmark * (1.0 + MARGIN)
        booster_flag = booster >= benchmark * (1.0 + MARGIN)
        groups = {
            "both flag": linear_flag & booster_flag,
            "only linear": linear_flag & ~booster_flag,
            "only booster": ~linear_flag & booster_flag,
            "neither": ~linear_flag & ~booster_flag,
        }
        print(
            f"\n{name}: linear flags {int(linear_flag.sum())}, "
            f"booster flags {int(booster_flag.sum())}, "
            f"changed {int((linear_flag != booster_flag).sum())} of {len(price)}"
        )
        rows = [gap_table(label, mask, price, benchmark, booster, linear) for label, mask in groups.items()]
        print_frame(pd.DataFrame(rows))
    print("booster_error is predicted q33 minus the recorded sale price. Positive means the q33 was above that price.")
    print("A recorded sale price above the catalogue estimate is not evidence that buying the watch would have been profitable.")


def chosen_interval(frame, predictors):
    print("\nCHOSEN INTERVAL PROCEDURE")
    print("q10 and q90 are fit on sales before 2023. The cutoff is read from 2023 only.")
    print("The cutoff is then applied to 2024 and, as a later check, to 2026. It is not refit on either year.")
    fit_df = frame.loc[frame["year"].lt(2023)].reset_index(drop=True)
    cal_df = frame.loc[frame["year"].eq(2023)].reset_index(drop=True)
    eval_2024 = frame.loc[frame["year"].eq(2024)].reset_index(drop=True)
    eval_2026 = frame.loc[frame["year"].eq(2026)].reset_index(drop=True)
    if fit_df["sale_date"].max() >= cal_df["sale_date"].min():
        raise RuntimeError("Calibration sales are not after the interval fit.")
    y_fit = fit_df["usd_price"].to_numpy(dtype=float)
    _x_fit, _x_cal, x_fit_cat, x_cal_cat = encode(fit_df, cal_df, predictors)
    low_model = fit_fixed_booster(x_fit_cat, y_fit, quantile=0.1)
    high_model = fit_fixed_booster(x_fit_cat, y_fit, quantile=0.9)
    y_cal = cal_df["usd_price"].to_numpy(dtype=float)
    cal_low = np.exp(low_model.predict(x_cal_cat))
    cal_high = np.exp(high_model.predict(x_cal_cat))
    scores = np.maximum(cal_low - y_cal, y_cal - cal_high)
    cutoff, rank, n_cal = conformal_cutoff(scores, alpha=0.2)
    expected_rank = int(np.ceil((n_cal + 1) * 0.8))
    print(f"Fit sales: {len(fit_df)}, last sale {fit_df['sale_date'].max().date().isoformat()}.")
    print(f"Calibration sales: {n_cal}, {cal_df['sale_date'].min().date().isoformat()} to {cal_df['sale_date'].max().date().isoformat()}.")
    print(f"Rank ceil((n+1)*0.8) = {expected_rank}. Cutoff rank {rank}: {cutoff:,.2f} dollars on each end.")
    if rank != expected_rank:
        raise RuntimeError("Cutoff rank does not match the 80% order-statistic rule.")
    for label, eval_df in (("2024", eval_2024), ("2026", eval_2026)):
        _x_fit, _x_eval, _x_fit_cat, x_eval_cat = encode(fit_df, eval_df, predictors)
        raw_low = np.exp(low_model.predict(x_eval_cat))
        raw_high = np.exp(high_model.predict(x_eval_cat))
        actual = eval_df["usd_price"].to_numpy(dtype=float)
        brands = eval_df["brand"].to_numpy()
        print(
            f"{label}: raw coverage {coverage(actual, raw_low, raw_high):.4f}, "
            f"after cutoff {coverage(actual, raw_low - cutoff, raw_high + cutoff):.4f}, "
            f"median width {float(np.median((raw_high + cutoff) - (raw_low - cutoff))):,.2f}"
        )
        for brand in ["Cartier", "Omega", "Rolex"]:
            mask = brands == brand
            print(
                f"  {brand} n={int(mask.sum())}: "
                f"coverage {coverage(actual[mask], raw_low[mask] - cutoff, raw_high[mask] + cutoff):.4f}, "
                f"median width {float(np.median((raw_high[mask] + cutoff) - (raw_low[mask] - cutoff))):,.2f}"
            )
    print("This cutoff does not change the q33 prediction. The 70.07% and 67.96% bands are different fits.")


def reproduce_and_list(frame, predictors):
    print("\nREPRODUCIBILITY")
    missing = [path for path in REQUIRED if not (REPO_ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"Missing project files: {missing}")
    print("Required scripts, requirements.txt, the assignment file, and the eligible file are present.")
    print(f"Eligible checksum expected by the scripts: {INPUT_SHA256}")
    train_df = frame.loc[frame["year"].le(2023)].reset_index(drop=True)
    eval_df = frame.loc[frame["year"].eq(2024)].reset_index(drop=True)
    _x_train, _x_eval, x_train_cat, x_eval_cat = encode(train_df, eval_df, predictors)
    prediction = np.exp(fit_fixed_booster(x_train_cat, train_df["usd_price"].to_numpy(dtype=float)).predict(x_eval_cat))
    loss = pinball_loss(eval_df["usd_price"].to_numpy(dtype=float), prediction, TAU)
    print(f"Refit 2024 booster pinball: {loss:,.4f}. Recorded: {FROZEN_GB_LOSS:,.4f}")
    if abs(loss - FROZEN_GB_LOSS) > 0.02:
        raise RuntimeError("The frozen booster was not reproduced.")


def main():
    _root, eligible_path, assignment_path = resolve_data_root()
    frame, predictors = load_assigned(eligible_path, assignment_path)
    reproduce_and_list(frame, predictors)
    calibration_2026(frame, predictors)
    screening_gaps(frame, predictors, eligible_path)
    chosen_interval(frame, predictors)


if __name__ == "__main__":
    main()
