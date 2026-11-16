"""Audit candidate columns before any new gradient-booster experiment.

Reads the cleaned 278-column file and the existing feature-candidate table.
Prints aggregate availability only. Does not fit a model.
"""

from collections import Counter

import pandas as pd

from scripts.audit_gbqr import load_assigned
from scripts.fit_iqn import resolve_data_root
from scripts.predictors import INPUT_SHA256, construct_record

CLEANED_CANDIDATES = [
    "norm_model",
    "norm_model_cleaned",
    "designer",
    "norm_designer",
    "norm_designer_cleaned",
    "case_material",
    "norm_case_material",
    "norm_case_material_cleaned",
    "modeling_case_body",
    "movement",
    "norm_movement",
    "norm_movement_cleaned",
    "modeling_movement",
    "case_diameter",
    "norm_case_diameter",
    "norm_case_diameter_cleaned",
    "case_measurement_mm",
    "gold_karat",
    "norm_gold_karat",
    "norm_gold_karat_cleaned",
    "modeling_case_body_karat",
    "production_year",
    "production_year_parsed",
    "norm_production_year",
    "norm_production_year_cleaned",
    "gem_type",
    "norm_gem_type",
    "norm_gem_type_cleaned",
    "primary_gem",
    "norm_primary_gem",
    "norm_primary_gem_cleaned",
    "norm_gemstone_clarity_cleaned",
    "norm_gemstone_cut_cleaned",
    "carats",
    "norm_carats_cleaned",
    "strap_bracelet",
    "norm_strap_bracelet_cleaned",
    "bracelet_material",
    "norm_bracelet_material_cleaned",
    "buckle_clasp",
    "norm_buckle_clasp_cleaned",
    "certification",
    "norm_certification_cleaned",
    "signed",
    "modeling_eligibility",
    "norm_usd_price_decimal_cleaned",
]

V2_CANDIDATES = [
    "papers",
    "papers_review",
    "box",
    "box_review",
    "gemstones",
    "gemstones_review",
    "attachment_type",
    "attachment_type_review",
    "case_diameter_mm",
    "precious_stones_candidate",
]


def blank_mask(series):
    text = series.fillna("").astype(str).str.strip()
    return text.eq("") | text.str.lower().isin(["nan", "none", "null", "unknown", "-"])


def summarize(frame, column, split):
    series = frame[column]
    missing = blank_mask(series)
    values = series.loc[~missing].astype(str).str.strip()
    counts = Counter(values)
    top = ", ".join(f"{name} ({count})" for name, count in counts.most_common(6))
    return {
        "column": column,
        "split": split,
        "n": len(frame),
        "missing": int(missing.sum()),
        "missing_rate": float(missing.mean()),
        "n_distinct": int(values.nunique()),
        "top_values": top,
    }


def main():
    root, eligible_path, assignment_path = resolve_data_root()
    assigned, predictors = load_assigned(eligible_path, assignment_path)
    cleaned_path = root / "expanded_pull_eligible_cleaned.csv"
    v2_path = root / "expanded_pull_eligible_feature_candidates_v2.csv"
    print(f"Eligible checksum file: {eligible_path.name} expected {INPUT_SHA256}")
    print(f"Cleaned file: {cleaned_path.name} exists={cleaned_path.is_file()}")
    print(f"Feature-candidate v2: {v2_path.name} exists={v2_path.is_file()}")
    cleaned = pd.read_csv(cleaned_path, low_memory=False)
    print(f"Cleaned rows: {len(cleaned)}. Columns: {cleaned.shape[1]}.")
    print(f"Cleaned duplicate record ids: {int(cleaned['record_id'].duplicated().sum())}")
    eligible_ids = set(assigned["record_id"])
    cleaned_ids = set(cleaned["record_id"].astype(str))
    print(f"Assigned ids missing from cleaned: {len(eligible_ids - cleaned_ids)}")
    print(f"Cleaned ids absent from assignments: {len(cleaned_ids - eligible_ids)}")
    price = pd.to_numeric(cleaned["usd_price_decimal_num"], errors="coerce")
    print(f"Cleaned nonpositive usd_price_decimal_num: {int((price.fillna(0) <= 0).sum())}")
    joined = assigned.merge(
        cleaned.assign(record_id=cleaned["record_id"].astype(str)),
        on="record_id",
        how="left",
        suffixes=("", "_cleaned_file"),
    )
    mismatch = (joined["usd_price"] - pd.to_numeric(joined["usd_price_decimal_num"], errors="coerce")).abs()
    print(f"Max absolute price difference vs eligible target: {float(mismatch.max()):.6f}")
    print(f"Designer equals assignment brand: {int((joined['designer'] == joined['brand']).sum())} of {len(joined)}")

    rows = []
    for split_name, mask in (
        ("train", joined["partition"].eq("train")),
        ("validation", joined["partition"].eq("validation")),
        ("test", joined["partition"].eq("test")),
    ):
        part = joined.loc[mask]
        for column in CLEANED_CANDIDATES:
            if column not in part.columns:
                rows.append({"column": column, "split": split_name, "n": len(part), "missing": None, "note": "ABSENT"})
                continue
            rows.append(summarize(part, column, split_name))
    audit = pd.DataFrame(rows)
    print("\nCLEANED FILE AVAILABILITY")
    print(audit.to_string(index=False))

    if v2_path.is_file():
        v2 = pd.read_csv(v2_path, low_memory=False)
        print(f"\nV2 rows: {len(v2)}. Columns: {v2.shape[1]}.")
        v2["record_id"] = v2["record_id"].astype(str)
        print(f"Assigned ids missing from v2: {len(eligible_ids - set(v2['record_id']))}")
        present = [column for column in V2_CANDIDATES if column in v2.columns]
        absent = [column for column in V2_CANDIDATES if column not in v2.columns]
        print(f"V2 candidate columns absent: {absent}")
        merged = assigned.merge(v2[["record_id", *present]], on="record_id", how="left")
        v2_rows = []
        for split_name, mask in (
            ("train", merged["partition"].eq("train")),
            ("validation", merged["partition"].eq("validation")),
        ):
            part = merged.loc[mask]
            for column in present:
                v2_rows.append(summarize(part, column, split_name))
        print(pd.DataFrame(v2_rows).to_string(index=False))

    # Overlap between the frozen predictors and the cleaned replacements.
    comparison = joined.loc[joined["partition"].isin(["train", "validation"])].copy()
    constructed = comparison["record_id"].map(lambda record_id: predictors[record_id]["norm_model"])
    print("\nOVERLAP WITH THE FROZEN PREDICTORS, TRAIN PLUS VALIDATION")
    print(f"construct_record norm_model equals norm_model: {int((constructed.fillna('') == comparison['norm_model'].fillna('')).sum())} of {len(comparison)}")
    cleaned_model = comparison["norm_model_cleaned"].fillna("").astype(str).str.strip()
    current_model = comparison["norm_model"].fillna("").astype(str).str.strip()
    print(f"norm_model equals norm_model_cleaned: {int((current_model == cleaned_model).sum())} of {len(comparison)}")
    constructed_movement = comparison["record_id"].map(lambda record_id: predictors[record_id]["movement"] or "")
    cleaned_movement = comparison["norm_movement_cleaned"].fillna("").astype(str).str.strip()
    print(f"construct_record movement equals norm_movement_cleaned: {int((constructed_movement.astype(str) == cleaned_movement).sum())} of {len(comparison)}")
    constructed_body = comparison["record_id"].map(lambda record_id: predictors[record_id]["case_body"] or "")
    cleaned_material = comparison["norm_case_material_cleaned"].fillna("").astype(str).str.strip()
    print(f"construct_record case_body equals norm_case_material_cleaned: {int((constructed_body.astype(str) == cleaned_material).sum())} of {len(comparison)}")
    age = comparison["record_id"].map(lambda record_id: predictors[record_id]["age_at_sale"])
    parsed_year = pd.to_numeric(comparison["production_year_parsed"], errors="coerce")
    sale_year = comparison["sale_date"].dt.year
    reconstructed_age = sale_year - parsed_year
    age_number = pd.to_numeric(age, errors="coerce")
    both = age_number.notna() & reconstructed_age.notna()
    print(f"age_at_sale equals sale year minus production_year_parsed: {int((age_number[both] == reconstructed_age[both]).sum())} of {int(both.sum())}")
    print(f"production_year_parsed missing on train+validation: {int(parsed_year.isna().sum())} of {len(comparison)}")


if __name__ == "__main__":
    main()
