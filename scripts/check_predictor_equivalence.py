"""Compare constructed predictors with the saved modeling rows.

The check reads two files under the shared data root and writes nothing.
Set WATCH_DATA_DIR, or pass --data-root. The eligible file must match
scripts.predictors.INPUT_SHA256 before it is parsed.

    python scripts/check_predictor_equivalence.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.predictors import ELIGIBLE_ROWS, INPUT_SHA256, construct_record

COMPARED_FIELDS = (
    "norm_model",
    "age_at_sale",
    "production_year_approximate",
    "case_body",
    "case_body_source",
    "case_finish",
    "case_finish_source",
    "movement",
    "movement_source",
    "case_diameter_mm",
    "case_width_mm",
    "case_length_mm",
)
NUMERIC_FIELDS = {"age_at_sale", "case_diameter_mm", "case_width_mm", "case_length_mm"}
HISTORICAL_POPULATED = {
    "case_body": 4194,
    "movement": 3241,
    "age_at_sale": 4318,
}
PROTECTED_SHA256 = {
    "processed/expanded_pull_eligible.csv": INPUT_SHA256,
    "modeling/expanded_pull_modeling_rows.csv": "d2ee0efa168114f5e88a381943cf9969afb0032816ff3b1cf9b299e79777fad7",
    "modeling/expanded_pull_validation_report.json": "aa51f17f2840af4e1070448a695703f383ccc4a668332ec230a803f6bfcd5c1d",
    "modeling/expanded_pull_validation_predictions.csv": "74be90e62b395dffdb63dfd43305fcdcf4047ca1000829f52c8fdf4267b355f2",
    "modeling/expanded_pull_ridge.joblib": "58a4dcdf4e403c78427ff841101af4f9ae2d6a20947a290d012b2bf6b3ce6a25",
    "modeling/expanded_pull_split_assignments.csv": "7aeb7c1fcd14edd386000c63cded88fe439f4da7c0df708d8e4f79e06f20e2b6",
    "modeling/expanded_pull_split.json": "1fffd8067dfab6ede998a18dbcda205a997324bc946dd84659a04a0a378c127b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strings(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def missing(value: object) -> bool:
    return value is None or str(value).strip() == ""


def same(field: str, constructed: object, saved: str) -> bool:
    if field in NUMERIC_FIELDS:
        if missing(constructed) or missing(saved):
            return missing(constructed) and missing(saved)
        return float(constructed) == float(saved)
    left = "" if constructed is None else str(constructed)
    return left == saved


def artifact_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in PROTECTED_SHA256}


def data_root_from(argument: str | None) -> Path:
    raw = argument or os.environ.get("WATCH_DATA_DIR", "")
    if not raw:
        raise SystemExit("Set WATCH_DATA_DIR or pass --data-root. The path is not stored in the repository.")
    return Path(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    root = data_root_from(args.data_root)
    before = artifact_hashes(root)
    if before != PROTECTED_SHA256:
        raise SystemExit("A data or modeling artifact hash differs from the audited file. Nothing was constructed.")
    eligible_path = root / "processed" / "expanded_pull_eligible.csv"
    modeling_path = root / "modeling" / "expanded_pull_modeling_rows.csv"
    eligible_hash = before["processed/expanded_pull_eligible.csv"]
    if eligible_hash != INPUT_SHA256:
        raise SystemExit("Eligible file SHA-256 does not match the authoritative checksum. Nothing was constructed.")

    eligible = load_strings(eligible_path)
    modeling = load_strings(modeling_path)
    if len(eligible) != ELIGIBLE_ROWS:
        raise SystemExit(f"Eligible row count is {len(eligible)}, expected {ELIGIBLE_ROWS}. Rules were not changed.")
    eligible_ids = [row["record_id"] for row in eligible]
    modeling_ids = [row["record_id"] for row in modeling]
    if len(set(eligible_ids)) != len(eligible_ids) or len(set(modeling_ids)) != len(modeling_ids):
        raise SystemExit("record_id values are not unique.")
    if set(eligible_ids) != set(modeling_ids):
        raise SystemExit("Eligible record IDs and modeling-row record IDs differ. Rules were not changed.")

    saved = {row["record_id"]: row for row in modeling}
    constructed = [construct_record(row) for row in eligible]
    by_id = {row["record_id"]: row for row in constructed}
    mismatches: dict[str, int] = {field: 0 for field in COMPARED_FIELDS}
    examples: list[str] = []
    for record_id, built in by_id.items():
        reference = saved[record_id]
        for field in COMPARED_FIELDS:
            if same(field, built[field], reference[field]):
                continue
            mismatches[field] += 1
            if len(examples) < 8:
                examples.append(f"{field} {record_id}")

    year_mismatches = 0
    age_identity_mismatches = 0
    for source, built in zip(eligible, constructed, strict=True):
        parsed = source["production_year_parsed"].strip()
        expected_year = int(parsed) if parsed else None
        if built["production_year"] != expected_year:
            year_mismatches += 1
        if built["age_at_sale"] is not None:
            if expected_year is None or int(source["sale_year"]) - expected_year != built["age_at_sale"]:
                age_identity_mismatches += 1
        elif parsed and int(parsed) <= int(source["sale_year"]):
            age_identity_mismatches += 1

    populated = {
        field: sum(not missing(row[field]) for row in constructed)
        for field in (
            "norm_model",
            "production_year",
            "age_at_sale",
            "production_year_approximate",
            "case_body",
            "case_finish",
            "movement",
            "case_diameter_mm",
            "case_width_mm",
            "case_length_mm",
        )
    }
    historical_gaps = {
        field: (populated[field], expected)
        for field, expected in HISTORICAL_POPULATED.items()
        if populated[field] != expected
    }
    total_mismatches = sum(mismatches.values()) + year_mismatches + age_identity_mismatches
    finish_values = {}
    finish_sources = {}
    not_stated_by_source = {}
    for row in by_id.values():
        finish = row["case_finish"] or "MISSING"
        source = str(row["case_finish_source"])
        finish_values[finish] = finish_values.get(finish, 0) + 1
        finish_sources[source] = finish_sources.get(source, 0) + 1
        if finish == "not_stated":
            not_stated_by_source[source] = not_stated_by_source.get(source, 0) + 1
    after = artifact_hashes(root)
    print(f"eligible_rows {len(eligible)}")
    print(f"input_sha256 {eligible_hash}")
    print(f"modeling_rows_sha256 {before['modeling/expanded_pull_modeling_rows.csv']}")
    for field, count in populated.items():
        print(f"populated_{field} {count}")
    print(f"finish_value_not_stated {finish_values.get('not_stated', 0)}")
    print(f"finish_source_no_finish_stated {finish_sources.get('no_finish_stated', 0)}")
    print(f"not_stated_from_no_finish_stated {not_stated_by_source.get('no_finish_stated', 0)}")
    print(f"not_stated_from_approved_correction {not_stated_by_source.get('approved_correction', 0)}")
    for field, count in mismatches.items():
        print(f"mismatches_{field} {count}")
    print(f"production_year_identity_mismatches {year_mismatches}")
    print(f"age_identity_mismatches {age_identity_mismatches}")
    if examples:
        print("mismatch_fields " + ", ".join(examples))
    finish_reconciled = (
        finish_values.get("not_stated", 0) == 4113
        and finish_sources.get("no_finish_stated", 0) == 4065
        and not_stated_by_source.get("no_finish_stated", 0) == 4065
        and not_stated_by_source.get("approved_correction", 0) == 48
        and populated["case_finish"] == 4249
    )
    if after != before:
        raise SystemExit("A data or modeling artifact changed during the check.")
    if historical_gaps or total_mismatches or not finish_reconciled:
        raise SystemExit("Constructed predictors do not match the audited modeling rows. Rules were not changed.")
    print("artifacts unchanged")
    print("equivalence verified")


if __name__ == "__main__":
    main()
