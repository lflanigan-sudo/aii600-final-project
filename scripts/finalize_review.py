"""Checks for the frozen comparison: window dates, screening flags, calibrated widths.

The 2024 catalogue estimate is the pre-auction benchmark. The realized sale
price is not used to build that benchmark. The 10% margin is the illustrative
rule already in the decision log. It is not re-optimized here.
"""

import numpy as np
import pandas as pd

from scripts.audit_gbqr import (
    conformal_cutoff,
    coverage,
    design,
    fit_fixed_booster,
    fit_linear_window,
    load_assigned,
)
from scripts.fit_iqn import TAU, assemble_features, collapse_rare_models, pinball_loss, resolve_data_root
from scripts.fit_gbqr import categorical_frames

MARGIN = 0.10


def verify_windows(frame):
    print("CHRONOLOGICAL WINDOW CHECK")
    print("Each evaluation year is fit only on sales from earlier years. No row is in both roles.")
    windows = [
        ("2022", frame["year"].eq(2021), frame["year"].eq(2022)),
        ("2023", frame["year"].le(2022), frame["year"].eq(2023)),
        ("2024", frame["year"].le(2023), frame["year"].eq(2024)),
        ("2025", frame["year"].le(2024), frame["year"].eq(2025)),
        ("2026", frame["year"].le(2025), frame["year"].eq(2026)),
    ]
    rows = []
    for year, train_mask, eval_mask in windows:
        train_df = frame.loc[train_mask]
        eval_df = frame.loc[eval_mask]
        overlap = set(train_df["record_id"]) & set(eval_df["record_id"])
        if overlap:
            raise RuntimeError(f"{year} reuses {len(overlap)} record ids.")
        if train_df["sale_date"].max() >= eval_df["sale_date"].min():
            raise RuntimeError(f"{year} training dates are not strictly earlier.")
        rows.append(
            {
                "eval_year": year,
                "n_train": len(train_df),
                "train_last_sale": train_df["sale_date"].max().date().isoformat(),
                "n_eval": len(eval_df),
                "eval_first_sale": eval_df["sale_date"].min().date().isoformat(),
                "eval_last_sale": eval_df["sale_date"].max().date().isoformat(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("2025 and 2026 are inside the development file. They are not an untouched test set.")
    return windows


def yearly_scores(frame, predictors, windows):
    print("\nYEARLY PINBALL AND Q33 CALIBRATION, FIXED SETTINGS")
    print("The booster uses the frozen 112-tree log-price settings. The linear model uses alpha 0.01.")
    rows = []
    for year, train_mask, eval_mask in windows:
        train_df = frame.loc[train_mask].reset_index(drop=True)
        eval_df = frame.loc[eval_mask].reset_index(drop=True)
        x_train, x_eval, x_train_cat, x_eval_cat = design(train_df, eval_df, predictors)
        y_train = train_df["usd_price"].to_numpy(dtype=float)
        y_eval = eval_df["usd_price"].to_numpy(dtype=float)
        booster = fit_fixed_booster(x_train_cat, y_train)
        gb_prediction = np.exp(booster.predict(x_eval_cat))
        linear_prediction, _ = fit_linear_window(x_train, x_eval, y_train)
        rows.append(
            {
                "eval_year": year,
                "n": len(eval_df),
                "booster_pinball": pinball_loss(y_eval, gb_prediction, TAU),
                "linear_pinball": pinball_loss(y_eval, linear_prediction, TAU),
                "booster_below_q33": float(np.mean(y_eval <= gb_prediction)),
                "linear_below_q33": float(np.mean(y_eval <= linear_prediction)),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    return rows


def load_estimates(eligible_path):
    estimates = pd.read_csv(
        eligible_path,
        usecols=["record_id", "sale_estimates_low_usd_num", "sale_estimates_high_usd_num"],
        low_memory=False,
    )
    estimates["low_usd"] = pd.to_numeric(estimates["sale_estimates_low_usd_num"], errors="coerce")
    estimates["high_usd"] = pd.to_numeric(estimates["sale_estimates_high_usd_num"], errors="coerce")
    return estimates[["record_id", "low_usd", "high_usd"]]


def flag_table(name, benchmark, linear_prediction, gb_prediction, price):
    linear_flag = linear_prediction >= benchmark * (1.0 + MARGIN)
    gb_flag = gb_prediction >= benchmark * (1.0 + MARGIN)
    groups = {
        "both flag": linear_flag & gb_flag,
        "only linear": linear_flag & ~gb_flag,
        "only booster": ~linear_flag & gb_flag,
        "neither": ~linear_flag & ~gb_flag,
    }
    rows = []
    for label, mask in groups.items():
        if int(mask.sum()) == 0:
            rows.append({"benchmark": name, "group": label, "n": 0, "sold_above_benchmark": np.nan, "median_price_minus_benchmark": np.nan})
            continue
        gap = price[mask] - benchmark[mask]
        rows.append(
            {
                "benchmark": name,
                "group": label,
                "n": int(mask.sum()),
                "sold_above_benchmark": float(np.mean(price[mask] > benchmark[mask])),
                "median_price_minus_benchmark": float(np.median(gap)),
            }
        )
    changed = int((linear_flag != gb_flag).sum())
    print(f"{name}: flags change on {changed} of {len(price)} sales with this benchmark.")
    print(f"{name}: linear flags {int(linear_flag.sum())}, booster flags {int(gb_flag.sum())}.")
    return pd.DataFrame(rows)


def screening(frame, predictors, eligible_path):
    print("\nSCREENING AGAINST THE CATALOGUE ESTIMATE")
    print("The benchmark is the published USD estimate range. It is not the realized sale price.")
    print("A flag means predicted q33 is at least 10% above the benchmark. The 10% margin is illustrative.")
    validation = frame.loc[frame["partition"].eq("validation")].reset_index(drop=True)
    training = frame.loc[frame["partition"].eq("train")].reset_index(drop=True)
    if training["sale_date"].max() >= validation["sale_date"].min():
        raise RuntimeError("Screening training dates are not strictly earlier than validation.")
    estimates = load_estimates(eligible_path)
    validation = validation.merge(estimates, on="record_id", how="left")
    usable = validation["low_usd"].gt(0) & validation["high_usd"].gt(0) & validation["low_usd"].le(validation["high_usd"])
    print(f"2024 validation sales: {len(validation)}. Sales with a positive low and high estimate: {int(usable.sum())}.")
    held = validation.loc[usable].reset_index(drop=True)
    x_train, x_eval, x_train_cat, x_eval_cat = design(training, held, predictors)
    y_train = training["usd_price"].to_numpy(dtype=float)
    booster = fit_fixed_booster(x_train_cat, y_train)
    gb_prediction = np.exp(booster.predict(x_eval_cat))
    linear_prediction, _ = fit_linear_window(x_train, x_eval, y_train)
    price = held["usd_price"].to_numpy(dtype=float)
    midpoint = ((held["low_usd"] + held["high_usd"]) / 2.0).to_numpy(dtype=float)
    low = held["low_usd"].to_numpy(dtype=float)
    tables = [
        flag_table("estimate midpoint", midpoint, linear_prediction, gb_prediction, price),
        flag_table("low estimate", low, linear_prediction, gb_prediction, price),
    ]
    print(pd.concat(tables, ignore_index=True).to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("sold_above_benchmark uses the hammer price only after the flag is set. It does not define the flag.")
    print("Sales without a published estimate are not given a flag.")


def calibrated_by_brand(frame, predictors):
    print("\nTIME-ORDERED INTERVAL, COVERAGE AND WIDTH BY BRAND")
    print("q10 and q90 are fit on sales before 2023. The dollar cutoff is read from 2023 and applied to 2024.")
    print("This is not the original booster interval. The original 2024 coverage of that interval is 70.07%.")
    fit_df = frame.loc[frame["year"].lt(2023)].reset_index(drop=True)
    cal_df = frame.loc[frame["year"].eq(2023)].reset_index(drop=True)
    eval_df = frame.loc[frame["year"].eq(2024)].reset_index(drop=True)
    if fit_df["sale_date"].max() >= cal_df["sale_date"].min() or cal_df["sale_date"].max() >= eval_df["sale_date"].min():
        raise RuntimeError("Interval periods are not in calendar order.")
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
    cutoff, rank, n_cal = conformal_cutoff(np.maximum(cal_low - y_cal, y_cal - cal_high), alpha=0.2)
    raw_low = np.exp(low_model.predict(x_eval_cat))
    raw_high = np.exp(high_model.predict(x_eval_cat))
    adjusted_low = raw_low - cutoff
    adjusted_high = raw_high + cutoff
    print(f"Cutoff rank {rank} of {n_cal}: {cutoff:,.2f} dollars added to each end.")
    rows = []
    for brand in ["All", "Cartier", "Omega", "Rolex"]:
        mask = np.ones(len(eval_df), dtype=bool) if brand == "All" else eval_df["brand"].eq(brand).to_numpy()
        rows.append(
            {
                "brand": brand,
                "n": int(mask.sum()),
                "raw_coverage": coverage(y_eval[mask], raw_low[mask], raw_high[mask]),
                "calibrated_coverage": coverage(y_eval[mask], adjusted_low[mask], adjusted_high[mask]),
                "raw_median_width": float(np.median(raw_high[mask] - raw_low[mask])),
                "calibrated_median_width": float(np.median(adjusted_high[mask] - adjusted_low[mask])),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("q33 pinball is unchanged by this additive cutoff. The cutoff was not chosen to raise q33 accuracy.")


def main():
    _root, eligible_path, assignment_path = resolve_data_root()
    frame, predictors = load_assigned(eligible_path, assignment_path)
    windows = verify_windows(frame)
    yearly_scores(frame, predictors, windows)
    screening(frame, predictors, eligible_path)
    calibrated_by_brand(frame, predictors)


if __name__ == "__main__":
    main()
