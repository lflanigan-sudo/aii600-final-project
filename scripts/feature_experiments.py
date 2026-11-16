"""Feature-group sensitivity for the frozen gradient-boosted q33 model.

The booster settings stay fixed: log price, learning rate 0.1, max depth 6,
minimum leaf 20, no l2 penalty, 112 trees, seed 20261024. Nothing here retunes
those settings or rebuilds the chronological split.

The 2024 window and the 2025-2026 windows are exploratory. Those sales were
already in the development file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.audit_gbqr import (
    FROZEN_GB_LOSS,
    conformal_cutoff,
    coverage,
    date_cluster_mean,
    design,
    fit_fixed_booster,
    load_assigned,
)
from scripts.fit_iqn import (
    SEED,
    TAU,
    assemble_features,
    pinball_loss,
    resolve_data_root,
    row_pinball,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "fig"
MIN_SUPPORT = 15
GROUPS = {
    "A": {
        "label": "Baseline",
        "columns": ["norm_model", "case_body", "movement", "age_at_sale", "brand"],
        "categorical": ["norm_model", "case_body", "movement", "brand"],
        "numeric": ["age_at_sale"],
        "collapse": ["norm_model"],
    },
    "B": {
        "label": "Documentation",
        "columns": ["norm_model", "case_body", "movement", "age_at_sale", "brand", "papers", "box"],
        "categorical": ["norm_model", "case_body", "movement", "brand", "papers", "box"],
        "numeric": ["age_at_sale"],
        "collapse": ["norm_model"],
    },
    "C": {
        "label": "Physical specifications",
        "columns": [
            "norm_model",
            "cleaned_case_material",
            "movement",
            "age_at_sale",
            "brand",
            "diameter_mm",
            "diameter_missing",
            "gold_karat",
            "gold_karat_missing",
        ],
        "categorical": ["norm_model", "cleaned_case_material", "movement", "brand"],
        "numeric": ["age_at_sale", "diameter_mm", "diameter_missing", "gold_karat", "gold_karat_missing"],
        "collapse": ["norm_model"],
    },
    "D": {
        "label": "Collectibility",
        "columns": [
            "norm_model",
            "case_body",
            "movement",
            "age_at_sale",
            "brand",
            "gemstones",
            "production_year",
            "production_year_missing",
        ],
        "categorical": ["norm_model", "case_body", "movement", "brand", "gemstones"],
        "numeric": ["age_at_sale", "production_year", "production_year_missing"],
        "collapse": ["norm_model"],
    },
    "E": {
        "label": "Cleaned model and movement",
        "columns": ["cleaned_model", "case_body", "cleaned_movement", "age_at_sale", "brand"],
        "categorical": ["cleaned_model", "case_body", "cleaned_movement", "brand"],
        "numeric": ["age_at_sale"],
        "collapse": ["cleaned_model", "cleaned_movement"],
    },
    "F": {
        "label": "Combined eligible features",
        "columns": [
            "cleaned_model",
            "cleaned_case_material",
            "cleaned_movement",
            "age_at_sale",
            "brand",
            "papers",
            "box",
            "gemstones",
            "diameter_mm",
            "diameter_missing",
            "gold_karat",
            "gold_karat_missing",
            "production_year",
            "production_year_missing",
        ],
        "categorical": [
            "cleaned_model",
            "cleaned_case_material",
            "cleaned_movement",
            "brand",
            "papers",
            "box",
            "gemstones",
        ],
        "numeric": [
            "age_at_sale",
            "diameter_mm",
            "diameter_missing",
            "gold_karat",
            "gold_karat_missing",
            "production_year",
            "production_year_missing",
        ],
        "collapse": ["cleaned_model", "cleaned_movement"],
    },
}


def text_level(series, blank_label):
    text = series.astype("string").str.strip()
    missing = text.isna() | text.eq("") | text.str.lower().isin(["nan", "none", "null", "-", "unknown"])
    return text.mask(missing, blank_label).astype(str)


def parse_millimeters(series):
    extracted = series.astype("string").str.extract(r"(\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def load_feature_frame(eligible_path, assignment_path):
    assigned, predictors = load_assigned(eligible_path, assignment_path)
    root = eligible_path.parent
    cleaned = pd.read_csv(root / "expanded_pull_eligible_cleaned.csv", low_memory=False)
    cleaned["record_id"] = cleaned["record_id"].astype(str)
    v2 = pd.read_csv(
        root / "expanded_pull_eligible_feature_candidates_v2.csv",
        usecols=["record_id", "papers", "box", "gemstones", "precious_stones_candidate"],
        low_memory=False,
    )
    v2["record_id"] = v2["record_id"].astype(str)
    base = assemble_features(assigned, predictors)
    frame = assigned.reset_index(drop=True).copy()
    for column in base.columns:
        frame[column] = base[column].to_numpy()
    extra = cleaned.merge(v2, on="record_id", how="left")
    extra = frame[["record_id"]].merge(extra, on="record_id", how="left", validate="one_to_one")
    if len(extra) != len(frame):
        raise RuntimeError("Feature join changed the assigned population.")
    frame["cleaned_model"] = text_level(extra["norm_model_cleaned"], "Missing")
    frame["cleaned_movement"] = text_level(extra["norm_movement_cleaned"], "Missing")
    frame["cleaned_case_material"] = text_level(extra["norm_case_material_cleaned"], "Missing")
    frame["papers"] = text_level(extra["papers"], "Not stated")
    frame["box"] = text_level(extra["box"], "Not stated")
    frame["gemstones"] = text_level(extra["gemstones"], "Not stated")
    frame["diameter_mm"] = parse_millimeters(extra["norm_case_diameter_cleaned"])
    frame["diameter_missing"] = frame["diameter_mm"].isna().astype(float)
    frame["gold_karat"] = pd.to_numeric(extra["norm_gold_karat_cleaned"], errors="coerce")
    frame["gold_karat_missing"] = frame["gold_karat"].isna().astype(float)
    frame["production_year"] = pd.to_numeric(extra["production_year_parsed"], errors="coerce")
    frame["production_year_missing"] = frame["production_year"].isna().astype(float)
    stones = text_level(extra["precious_stones_candidate"], "Not stated")
    if not frame["gemstones"].equals(stones):
        raise RuntimeError("precious_stones_candidate is not a duplicate of gemstones.")
    price_gap = (frame["usd_price"] - pd.to_numeric(extra["usd_price_decimal_num"], errors="coerce")).abs()
    if float(price_gap.max()) > 1e-6:
        raise RuntimeError("Cleaned sale price does not match the eligible target.")
    return frame, predictors


def prepare(train_df, eval_df, spec):
    columns = spec["columns"]
    x_train = train_df[columns].copy()
    x_eval = eval_df[columns].copy()
    for column in spec["collapse"]:
        counts = x_train[column].value_counts()
        frequent = set(counts[counts >= MIN_SUPPORT].index)
        x_train[column] = x_train[column].where(x_train[column].isin(frequent), "Other")
        x_eval[column] = x_eval[column].where(x_eval[column].isin(frequent), "Other")
    for column in spec["categorical"]:
        levels = sorted(x_train[column].unique())
        x_train[column] = pd.Categorical(x_train[column], categories=levels)
        known = x_eval[column].isin(levels)
        x_eval[column] = pd.Categorical(x_eval[column].where(known), categories=levels)
    for column in spec["numeric"]:
        x_train[column] = pd.to_numeric(x_train[column], errors="coerce")
        x_eval[column] = pd.to_numeric(x_eval[column], errors="coerce")
    return x_train, x_eval


def fit_group(train_df, eval_df, spec, quantile=TAU):
    y_train = train_df["usd_price"].to_numpy(dtype=float)
    x_train, x_eval = prepare(train_df, eval_df, spec)
    model = fit_fixed_booster(x_train, y_train, quantile=quantile)
    prediction = np.exp(model.predict(x_eval))
    return prediction, model, x_train, x_eval


def reproduce_benchmark(frame, predictors):
    train = frame.loc[frame["partition"].eq("train")].reset_index(drop=True)
    validation = frame.loc[frame["partition"].eq("validation")].reset_index(drop=True)
    _x_train, _x_eval, x_train_cat, x_eval_cat = design(train, validation, predictors)
    model = fit_fixed_booster(x_train_cat, train["usd_price"].to_numpy(dtype=float))
    prediction = np.exp(model.predict(x_eval_cat))
    loss = pinball_loss(validation["usd_price"].to_numpy(dtype=float), prediction, TAU)
    print("BENCHMARK REPRODUCTION")
    print(f"Existing design and frozen booster, 2024 dollar pinball: {loss:,.4f}")
    print(f"Recorded benchmark: {FROZEN_GB_LOSS:,.4f}")
    if abs(loss - FROZEN_GB_LOSS) > 0.02:
        raise RuntimeError("Benchmark was not reproduced. No feature experiment was fit.")
    group_prediction, _model, _train_x, _eval_x = fit_group(train, validation, GROUPS["A"])
    group_loss = pinball_loss(validation["usd_price"].to_numpy(dtype=float), group_prediction, TAU)
    print(f"Group A builder, 2024 dollar pinball: {group_loss:,.4f}")
    if abs(group_loss - loss) > 0.02:
        raise RuntimeError("Group A does not match the existing five-feature booster.")
    return loss


def brand_rows(eval_df, actual, baseline, candidate):
    rows = []
    brands = eval_df["brand"].to_numpy()
    for brand in ["All", "Cartier", "Omega", "Rolex"]:
        mask = np.ones(len(eval_df), dtype=bool) if brand == "All" else brands == brand
        rows.append(
            {
                "brand": brand,
                "n": int(mask.sum()),
                "pinball": pinball_loss(actual[mask], candidate[mask], TAU),
                "baseline_pinball": pinball_loss(actual[mask], baseline[mask], TAU),
                "share_at_or_below_q33": float(np.mean(actual[mask] <= candidate[mask])),
                "baseline_share_at_or_below_q33": float(np.mean(actual[mask] <= baseline[mask])),
            }
        )
    return rows


def segment_rows(actual, baseline, candidate, train_prediction):
    cuts = np.quantile(train_prediction, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.where(baseline <= cuts[0], "Low", np.where(baseline <= cuts[1], "Middle", "High"))
    rows = []
    for label in ["Low", "Middle", "High"]:
        mask = labels == label
        rows.append(
            {
                "segment": label,
                "n": int(mask.sum()),
                "cut_low": float(cuts[0]),
                "cut_high": float(cuts[1]),
                "pinball": pinball_loss(actual[mask], candidate[mask], TAU),
                "baseline_pinball": pinball_loss(actual[mask], baseline[mask], TAU),
            }
        )
    return rows


def interval_report(train_df, eval_df, spec):
    low, _model, _x_train, _x_eval = fit_group(train_df, eval_df, spec, quantile=0.1)
    mid, _model, _x_train, _x_eval = fit_group(train_df, eval_df, spec, quantile=TAU)
    high, _model, _x_train, _x_eval = fit_group(train_df, eval_df, spec, quantile=0.9)
    actual = eval_df["usd_price"].to_numpy(dtype=float)
    brands = eval_df["brand"].to_numpy()
    order = low <= high
    crossing = ~((low <= mid) & (mid <= high))
    rows = []
    for brand in ["All", "Cartier", "Omega", "Rolex"]:
        mask = np.ones(len(eval_df), dtype=bool) if brand == "All" else brands == brand
        rows.append(
            {
                "brand": brand,
                "n": int(mask.sum()),
                "coverage": coverage(actual[mask], low[mask], high[mask]),
                "median_width": float(np.median(high[mask] - low[mask])),
            }
        )
    return {
        "rows": rows,
        "crossing_rate": float(np.mean(crossing)),
        "unordered_rate": float(np.mean(~order)),
        "low": low,
        "high": high,
    }


def conformal_report(frame, spec):
    fit_df = frame.loc[frame["year"].lt(2023)].reset_index(drop=True)
    cal_df = frame.loc[frame["year"].eq(2023)].reset_index(drop=True)
    eval_df = frame.loc[frame["year"].eq(2024)].reset_index(drop=True)
    if fit_df["sale_date"].max() >= cal_df["sale_date"].min() or cal_df["sale_date"].max() >= eval_df["sale_date"].min():
        raise RuntimeError("Interval periods are not in calendar order.")
    cal_low, _model, _x_fit, _x_cal = fit_group(fit_df, cal_df, spec, quantile=0.1)
    cal_high, _model, _x_fit, _x_cal = fit_group(fit_df, cal_df, spec, quantile=0.9)
    y_cal = cal_df["usd_price"].to_numpy(dtype=float)
    cutoff, rank, n_cal = conformal_cutoff(np.maximum(cal_low - y_cal, y_cal - cal_high), alpha=0.2)
    raw_low, _model, _x_fit, _x_eval = fit_group(fit_df, eval_df, spec, quantile=0.1)
    raw_high, _model, _x_fit, _x_eval = fit_group(fit_df, eval_df, spec, quantile=0.9)
    y_eval = eval_df["usd_price"].to_numpy(dtype=float)
    brands = eval_df["brand"].to_numpy()
    rows = []
    for brand in ["All", "Cartier", "Omega", "Rolex"]:
        mask = np.ones(len(eval_df), dtype=bool) if brand == "All" else brands == brand
        rows.append(
            {
                "brand": brand,
                "n": int(mask.sum()),
                "raw_coverage": coverage(y_eval[mask], raw_low[mask], raw_high[mask]),
                "calibrated_coverage": coverage(y_eval[mask], raw_low[mask] - cutoff, raw_high[mask] + cutoff),
                "raw_median_width": float(np.median(raw_high[mask] - raw_low[mask])),
                "calibrated_median_width": float(np.median((raw_high[mask] + cutoff) - (raw_low[mask] - cutoff))),
            }
        )
    return {"cutoff": cutoff, "rank": rank, "n_cal": n_cal, "rows": rows}


def score_window(frame, train_mask, eval_mask):
    train_df = frame.loc[train_mask].reset_index(drop=True)
    eval_df = frame.loc[eval_mask].reset_index(drop=True)
    if train_df["sale_date"].max() >= eval_df["sale_date"].min():
        raise RuntimeError("Training sales are not strictly earlier than the evaluation sales.")
    if set(train_df["record_id"]) & set(eval_df["record_id"]):
        raise RuntimeError("A sale appears in training and evaluation.")
    actual = eval_df["usd_price"].to_numpy(dtype=float)
    fitted = {}
    for key, spec in GROUPS.items():
        prediction, model, x_train, _x_eval = fit_group(train_df, eval_df, spec)
        train_prediction = np.exp(model.predict(x_train))
        fitted[key] = {
            "prediction": prediction,
            "train_prediction": train_prediction,
            "loss": pinball_loss(actual, prediction, TAU),
        }
    return train_df, eval_df, actual, fitted


def comparison_table(actual, dates, brands, fitted):
    baseline = fitted["A"]["prediction"]
    baseline_loss = fitted["A"]["loss"]
    rows = []
    details = []
    for key, spec in GROUPS.items():
        prediction = fitted[key]["prediction"]
        loss = fitted[key]["loss"]
        paired = row_pinball(actual, baseline, TAU) - row_pinball(actual, prediction, TAU)
        mean_gain, low, high, n_dates = date_cluster_mean(paired, dates)
        rows.append(
            {
                "group": key,
                "features": spec["label"],
                "pinball": loss,
                "gain_vs_baseline": baseline_loss - loss,
                "percent_vs_baseline": 100.0 * (baseline_loss - loss) / baseline_loss,
                "share_at_or_below_q33": float(np.mean(actual <= prediction)),
                "lower_loss_than_baseline": int(np.sum(paired > 0)),
                "higher_loss_than_baseline": int(np.sum(paired < 0)),
                "bootstrap_low": low,
                "bootstrap_high": high,
                "auction_dates": n_dates,
                "mean_paired_gain": mean_gain,
            }
        )
        for brand_row in brand_rows(pd.DataFrame({"brand": brands}), actual, baseline, prediction):
            brand_row["group"] = key
            details.append(brand_row)
    return pd.DataFrame(rows), pd.DataFrame(details)


def save_charts(comparison, brand_detail):
    FIG_DIR.mkdir(exist_ok=True)
    ordered = comparison.iloc[::-1]
    labels = ordered["group"] + "  " + ordered["features"]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.barh(labels, ordered["pinball"], color="#4C78A8")
    for position, value in enumerate(ordered["pinball"]):
        axis.text(value + 30, position, f"{value:,.0f}", va="center", fontsize=8)
    axis.set_xlim(0, 4200)
    axis.set_xlabel("2024 mean dollar pinball at q33")
    axis.set_title("Feature groups, frozen gradient booster")
    figure.tight_layout()
    pinball_path = FIG_DIR / "feature_experiment_pinball.png"
    figure.savefig(pinball_path, dpi=150)
    plt.close(figure)

    brands = ["Cartier", "Omega", "Rolex"]
    keys = list(GROUPS)
    x = np.arange(len(brands))
    width = 0.12
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for index, key in enumerate(keys):
        part = brand_detail.loc[brand_detail["group"].eq(key)].set_index("brand")
        values = [100.0 * part.loc[brand, "share_at_or_below_q33"] for brand in brands]
        axis.bar(x + (index - 2.5) * width, values, width, label=key)
    axis.axhline(100.0 / 3.0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(x)
    axis.set_xticklabels(brands)
    axis.set_ylabel("Share of 2024 prices at or below q33 (%)")
    axis.set_title("q33 calibration by brand")
    axis.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    figure.tight_layout()
    calibration_path = FIG_DIR / "feature_experiment_calibration.png"
    figure.savefig(calibration_path, dpi=150)
    plt.close(figure)
    return pinball_path, calibration_path


def main():
    _root, eligible_path, assignment_path = resolve_data_root()
    frame, predictors = load_feature_frame(eligible_path, assignment_path)
    print("FEATURE EXPERIMENTS FOR THE FROZEN GRADIENT BOOSTER")
    print(f"Assigned sales: {len(frame)}. Seed {SEED}. Tau {TAU:.6f}.")
    print("Settings: log price, learning rate 0.1, max depth 6, minimum leaf 20, l2 0, 112 trees.")
    print("Cleaned file name on disk: expanded_pull_eligible_cleaned.csv, 4,664 rows, 278 columns.")
    print("Papers and box are read from expanded_pull_eligible_feature_candidates_v2.csv.")
    reproduce_benchmark(frame, predictors)

    validation = frame.loc[frame["partition"].eq("validation")].reset_index(drop=True)
    training = frame.loc[frame["partition"].eq("train")].reset_index(drop=True)
    _train_df, eval_df, actual, fitted = score_window(
        frame, frame["partition"].eq("train"), frame["partition"].eq("validation")
    )
    comparison, brand_detail = comparison_table(
        actual, eval_df["sale_date"], eval_df["brand"].to_numpy(), fitted
    )
    print("\n2024 VALIDATION, SAME 802 SALES")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nPINBALL AND CALIBRATION BY BRAND")
    print(brand_detail.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    segment_tables = []
    for key in GROUPS:
        for row in segment_rows(actual, fitted["A"]["prediction"], fitted[key]["prediction"], fitted["A"]["train_prediction"]):
            row["group"] = key
            segment_tables.append(row)
    segments = pd.DataFrame(segment_tables)
    print("\nPINBALL BY BASELINE PREDICTED-PRICE TERTILE")
    print("Cuts are the training tertiles of the baseline q33 prediction.")
    print(segments.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    print("\nCHRONOLOGICAL WINDOWS, FIXED SETTINGS, NO RETUNING")
    windows = [
        ("2022", frame["year"].eq(2021), frame["year"].eq(2022)),
        ("2023", frame["year"].le(2022), frame["year"].eq(2023)),
        ("2024", frame["year"].le(2023), frame["year"].eq(2024)),
        ("2025", frame["year"].le(2024), frame["year"].eq(2025)),
        ("2026", frame["year"].le(2025), frame["year"].eq(2026)),
    ]
    temporal_rows = []
    for year, train_mask, eval_mask in windows:
        train_df, year_eval, year_actual, year_fitted = score_window(frame, train_mask, eval_mask)
        print(
            f"{year}: train {len(train_df)} through {train_df['sale_date'].max().date().isoformat()}, "
            f"evaluate {len(year_eval)} from {year_eval['sale_date'].min().date().isoformat()}"
        )
        for key in GROUPS:
            temporal_rows.append(
                {
                    "eval_year": year,
                    "train_through": train_df["sale_date"].max().date().isoformat(),
                    "n_train": len(train_df),
                    "n_eval": len(year_eval),
                    "group": key,
                    "pinball": year_fitted[key]["loss"],
                    "share_at_or_below_q33": float(np.mean(year_actual <= year_fitted[key]["prediction"])),
                }
            )
    temporal = pd.DataFrame(temporal_rows)
    print(temporal.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    promising = comparison.loc[comparison["pinball"] < comparison.loc[comparison["group"].eq("A"), "pinball"].iloc[0] - 1.0, "group"]
    interval_keys = ["A", *promising.tolist()]
    print("\nORIGINAL q10-q90 INTERVALS ON THE 2024 FIT")
    print("These bands use the full training period. They are not conformalized.")
    interval_rows = []
    for key in dict.fromkeys(interval_keys):
        report = interval_report(training, validation, GROUPS[key])
        print(f"{key}: crossing rate {report['crossing_rate']:.4f}, unordered q10 above q90 {report['unordered_rate']:.4f}")
        for row in report["rows"]:
            row["group"] = key
            interval_rows.append(row)
    print(pd.DataFrame(interval_rows).to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    best_key = comparison.sort_values("pinball").iloc[0]["group"]
    print("\nSEPARATE TIME-ORDERED INTERVAL")
    print("q10 and q90 are fit before 2023. The cutoff is read from 2023 and applied to 2024.")
    for key in dict.fromkeys(["A", best_key]):
        report = conformal_report(frame, GROUPS[key])
        print(f"{key}: cutoff rank {report['rank']} of {report['n_cal']}, {report['cutoff']:,.2f} dollars")
        table = pd.DataFrame(report["rows"])
        table.insert(0, "group", key)
        print(table.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    pinball_path, calibration_path = save_charts(comparison, brand_detail)
    print(f"\nSaved {pinball_path.name} and {calibration_path.name}.")
    print("2024, 2025, and 2026 are sensitivity windows. They are not untouched test sets.")


if __name__ == "__main__":
    main()
