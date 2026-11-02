"""Assign the chronological experiment split.

Training sales are dated before 2024-01-01, validation sales are dated
2024-01-01 through 2024-12-31, and test sales are dated 2025-01-01 through
the last sale in the eligible file. Two earlier sales of confirmed physical
watches are omitted from this experiment only. The eligible file, the random
split, and the saved model outputs are not rewritten.

Set WATCH_DATA_DIR to the absolute shared data root, or pass --data-root.
Every path below is relative to that root. Do not commit these files.

The assignment reads:

- processed/expanded_pull_eligible.csv
- diagnostics/chronological_repeat_watch_review.csv
- diagnostics/chronological_proposed_omissions.csv
- diagnostics/chronological_split_counts.csv
- diagnostics/chronological_prior_exposure_counts.csv
- diagnostics/chronological_split_feasibility.md
- modeling/expanded_pull_split_assignments.csv
- modeling/expanded_pull_split.json

These modeling files must also be present. The script hashes them and leaves
them unchanged:

- modeling/expanded_pull_modeling_rows.csv
- modeling/expanded_pull_validation_predictions.csv
- modeling/expanded_pull_validation_report.json
- modeling/expanded_pull_ridge.joblib

It writes modeling/chronological/chronological_split_assignments.csv,
chronological_omission_ledger.csv, chronological_split.json, and
chronological_split_sha256.txt. When those outputs already exist, they are
compared with this assignment and left unchanged.

Run from the repository root:

    python scripts/split_chronological.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

SCRIPT_VERSION = "1"
INPUT_NAME = "expanded_pull_eligible.csv"
INPUT_SHA256 = "cf3b41d890af1feba9348fa9a73d954fcab193ed890a39044bd0355198fc5a8a"
VALIDATION_START = date(2024, 1, 1)
VALIDATION_END = date(2024, 12, 31)
TEST_START = date(2025, 1, 1)
PARTITIONS = ("train", "validation", "test")
OMISSION_IDS = (
    "cartier_sothebys_2024_o000050_l050.json#15",
    "omega_sothebys_2022_o000000_l050.json#30",
)
OMISSION_REASON = (
    "An earlier sale of a confirmed physical watch is omitted. "
    "The latest sale stays in its own period."
)
NEEDED_COLUMNS = (
    "record_id",
    "lot_url",
    "designer",
    "vendor",
    "sale_date",
    "sale_year",
)
ASSIGNMENT_COLUMNS = (
    "record_id",
    "lot_url",
    "sale_date",
    "partition",
    "brand",
    "auction_house",
    "group_id",
)
LEDGER_COLUMNS = (
    "record_id",
    "lot_url",
    "sale_date",
    "brand",
    "auction_house",
    "group_id",
    "reference",
    "case_serial",
    "date_partition",
    "retained_record_id",
    "retained_lot_url",
    "retained_sale_date",
    "retained_partition",
    "reason",
    "identifying_evidence",
)
OUTPUT_NAMES = (
    "chronological_split_assignments.csv",
    "chronological_omission_ledger.csv",
    "chronological_split.json",
    "chronological_split_sha256.txt",
)
PERIOD_TO_FILE = {"train": "training", "validation": "validation", "test": "test"}
FILE_TO_PERIOD = {label: period for period, label in PERIOD_TO_FILE.items()}
PROTECTED = (
    "processed/expanded_pull_eligible.csv",
    "modeling/expanded_pull_split_assignments.csv",
    "modeling/expanded_pull_split.json",
    "modeling/expanded_pull_modeling_rows.csv",
    "modeling/expanded_pull_validation_predictions.csv",
    "modeling/expanded_pull_validation_report.json",
    "modeling/expanded_pull_ridge.joblib",
    "diagnostics/chronological_split_feasibility.md",
    "diagnostics/chronological_split_counts.csv",
    "diagnostics/chronological_repeat_watch_review.csv",
    "diagnostics/chronological_proposed_omissions.csv",
    "diagnostics/chronological_prior_exposure_counts.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def data_root_from(args: argparse.Namespace) -> Path:
    if args.data_root is not None:
        root = args.data_root
    else:
        raw = os.environ.get("WATCH_DATA_DIR", "").strip().strip('"')
        require(bool(raw), "Set WATCH_DATA_DIR or pass --data-root.")
        root = Path(raw)
    require(root.is_absolute(), f"Data root must be absolute, got {root}")
    root = root.resolve()
    require(root.is_dir(), f"Data root does not exist: {root}")
    return root


def parse_sale_date(record_id: str, value: str) -> date:
    text = value.strip()
    require(len(text) == 10, f"{record_id} sale_date is not YYYY-MM-DD: {value!r}")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit(f"{record_id} sale_date is not YYYY-MM-DD: {value!r}") from exc


def year_matches(sale_day: date, raw_year: str) -> bool:
    text = raw_year.strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text == str(sale_day.year)


def assign_partition(sale_day: date) -> str:
    if sale_day < VALIDATION_START:
        return "train"
    if VALIDATION_START <= sale_day <= VALIDATION_END:
        return "validation"
    if sale_day >= TEST_START:
        return "test"
    raise SystemExit(f"Sale date {sale_day.isoformat()} falls outside the date rules")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{path.name} has no header")
        return list(reader)


def load_eligible(path: Path) -> tuple[list[dict[str, str]], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in NEEDED_COLUMNS if column not in fieldnames]
        require(not missing, f"Eligible file is missing columns: {', '.join(missing)}")
        rows = []
        for raw in reader:
            rows.append({column: raw[column].strip() for column in NEEDED_COLUMNS})
    return rows, len(fieldnames)


def render_csv(columns: tuple[str, ...], rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[column] for column in columns])
    return buffer.getvalue()


def render_checksums(digests: list[tuple[str, str]]) -> str:
    return "".join(f"{digest}  {name}\n" for digest, name in digests)


def snapshot(root: Path) -> dict[str, str]:
    found = {}
    for relative in PROTECTED:
        path = root / relative
        require(path.is_file(), f"Protected file is missing: {relative}")
        found[relative] = sha256_file(path)
    return found


def check_boundaries() -> None:
    require(
        TEST_START == VALIDATION_END + timedelta(days=1),
        "Validation end and test start leave a gap or an overlap",
    )
    require(VALIDATION_START <= VALIDATION_END, "Validation dates are reversed")


def index_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    urls: dict[str, str] = {}
    for row in rows:
        record_id = row["record_id"]
        lot_url = row["lot_url"]
        require(record_id, "A row has no record ID")
        require(lot_url, f"{record_id} has no lot URL")
        require(record_id not in by_id, f"Duplicate record ID: {record_id}")
        require(lot_url not in urls, f"Duplicate lot URL: {lot_url}")
        sale_day = parse_sale_date(record_id, row["sale_date"])
        require(
            year_matches(sale_day, row["sale_year"]),
            f"{record_id} sale_year {row['sale_year']!r} does not match {sale_day.isoformat()}",
        )
        row["sale_day"] = sale_day
        row["date_partition"] = assign_partition(sale_day)
        by_id[record_id] = row
        urls[lot_url] = record_id
    return by_id


def confirmed_groups(
    review_rows: list[dict[str, str]],
    by_id: dict[str, dict[str, str]],
    random_partitions: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for row in review_rows:
        record_id = row["record_id"]
        require(record_id not in seen, f"Repeat evidence repeats {record_id}")
        seen.add(record_id)
        require(record_id in by_id, f"Repeat evidence is not in the eligible file: {record_id}")
        eligible = by_id[record_id]
        require(row["lot_url"] == eligible["lot_url"], f"Repeat evidence URL does not match {record_id}")
        require(row["sale_date"] == eligible["sale_date"], f"Repeat evidence date does not match {record_id}")
        require(row["brand"] == eligible["designer"], f"Repeat evidence brand does not match {record_id}")
        require(row["group_id"] == row["saved_group_id"], f"Saved group ID differs for {record_id}")
        require(
            row["random_partition"] == random_partitions[record_id],
            f"Repeat evidence random partition differs for {record_id}",
        )
        require(row["proposed_action"] in {"omit_from_chronological_experiment", "retain"}, f"Unknown repeat action for {record_id}")
        require(row["crosses_proposed_boundary"] in {"yes", "no"}, f"Unknown crossing flag for {record_id}")
        require(row["case_serial"].strip(), f"Confirmed repeat has no case serial: {record_id}")
        require(row["identifying_evidence"].strip(), f"Confirmed repeat has no saved evidence text: {record_id}")
        groups[row["group_id"]].append(row)
    require(groups, "Repeat-watch evidence has no groups")
    return groups


def select_omissions(
    groups: dict[str, list[dict[str, str]]],
    omissions: list[dict[str, str]],
    by_id: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Return the earlier crossing sales named in the saved evidence.

    A shared reference is not a group. Only rows in the saved serial-confirmed
    review can be omitted, and only the agreed earlier sales.
    """
    expected_ids = set(OMISSION_IDS)
    saved_ids = {row["record_id"] for row in omissions}
    require(saved_ids == expected_ids, "Proposed omissions are not the two agreed earlier sales")
    saved_by_id = {row["record_id"]: row for row in omissions}
    require(len(saved_by_id) == len(omissions), "Proposed omissions repeat a record ID")

    selected: dict[str, dict[str, str]] = {}
    for group_id, members in groups.items():
        periods = {by_id[member["record_id"]]["date_partition"] for member in members}
        latest = max(by_id[member["record_id"]]["sale_day"] for member in members)
        crossing = len(periods) > 1
        for member in members:
            eligible = by_id[member["record_id"]]
            flagged = member["crosses_proposed_boundary"] == "yes"
            require(flagged == crossing, f"Crossing flag disagrees with sale dates for {group_id}")
            is_earlier = crossing and eligible["sale_day"] < latest
            if is_earlier:
                require(
                    member["proposed_action"] == "omit_from_chronological_experiment",
                    f"Earlier crossing sale is not marked for omission: {member['record_id']}",
                )
                require(member["record_id"] in expected_ids, f"Evidence asks to omit a sale outside the agreed pair: {member['record_id']}")
            else:
                require(member["proposed_action"] == "retain", f"A sale that stays in its period is marked for omission: {member['record_id']}")
                require(member["record_id"] not in expected_ids, f"Agreed omission is not an earlier crossing sale: {member['record_id']}")
        if not crossing:
            continue
        earlier = [member for member in members if by_id[member["record_id"]]["sale_day"] < latest]
        retained = [member for member in members if by_id[member["record_id"]]["sale_day"] == latest]
        require(len(retained) == 1, f"{group_id} does not have one latest sale to retain")
        retained_member = retained[0]
        for member in earlier:
            saved = saved_by_id[member["record_id"]]
            eligible = by_id[member["record_id"]]
            for field in ("group_id", "lot_url", "sale_date", "brand", "reference", "case_serial", "identifying_evidence"):
                require(saved[field] == member[field], f"Omission ledger source disagrees with repeat review for {member['record_id']} {field}")
            require(saved["reason"] == OMISSION_REASON, f"Omission reason changed for {member['record_id']}")
            require(saved["proposed_period"] == PERIOD_TO_FILE[eligible["date_partition"]], f"Omission period disagrees with the sale date for {member['record_id']}")
            require(saved["retained_record_id"] == retained_member["record_id"], f"Retained sale disagrees for {member['record_id']}")
            selected[member["record_id"]] = {
                "group_id": group_id,
                "reference": member["reference"],
                "case_serial": member["case_serial"],
                "identifying_evidence": member["identifying_evidence"],
                "retained_record_id": retained_member["record_id"],
            }
    require(set(selected) == expected_ids, "The two agreed sales were not both confirmed as earlier crossing sales")
    for record_id in selected:
        require(by_id[record_id]["date_partition"] != "test", f"An omission would remove a test sale: {record_id}")
    return selected


def group_for(record_id: str, groups: dict[str, list[dict[str, str]]]) -> str:
    for group_id, members in groups.items():
        if any(member["record_id"] == record_id for member in members):
            return group_id
    return ""


def build_outputs(
    root: Path,
    by_id: dict[str, dict[str, str]],
    groups: dict[str, list[dict[str, str]]],
    selected: dict[str, dict[str, str]],
    evidence_hashes: dict[str, str],
    random_assignment_hash: str,
    exposure_counts: dict[tuple[str, str], int],
    column_count: int,
) -> dict[str, str]:
    assignments = []
    ledger = []
    for record_id in sorted(by_id):
        row = by_id[record_id]
        confirmed = group_for(record_id, groups)
        if record_id in selected:
            choice = selected[record_id]
            retained = by_id[choice["retained_record_id"]]
            ledger.append(
                {
                    "record_id": record_id,
                    "lot_url": row["lot_url"],
                    "sale_date": row["sale_date"],
                    "brand": row["designer"],
                    "auction_house": row["vendor"],
                    "group_id": choice["group_id"],
                    "reference": choice["reference"],
                    "case_serial": choice["case_serial"],
                    "date_partition": row["date_partition"],
                    "retained_record_id": choice["retained_record_id"],
                    "retained_lot_url": retained["lot_url"],
                    "retained_sale_date": retained["sale_date"],
                    "retained_partition": retained["date_partition"],
                    "reason": OMISSION_REASON,
                    "identifying_evidence": choice["identifying_evidence"],
                }
            )
            continue
        assignments.append(
            {
                "record_id": record_id,
                "lot_url": row["lot_url"],
                "sale_date": row["sale_date"],
                "partition": row["date_partition"],
                "brand": row["designer"],
                "auction_house": row["vendor"],
                "group_id": confirmed,
            }
        )
    ledger.sort(key=lambda item: item["record_id"])
    verify_assignment(by_id, groups, assignments, ledger)
    check_feasibility_counts(root, assignments, by_id)

    assignment_text = render_csv(ASSIGNMENT_COLUMNS, assignments)
    ledger_text = render_csv(LEDGER_COLUMNS, ledger)
    metadata = build_metadata(
        by_id,
        groups,
        assignments,
        ledger,
        evidence_hashes,
        random_assignment_hash,
        exposure_counts,
        column_count,
        sha256_text(assignment_text),
        sha256_text(ledger_text),
    )
    metadata_text = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    checksum_text = render_checksums(
        [
            (sha256_text(assignment_text), OUTPUT_NAMES[0]),
            (sha256_text(ledger_text), OUTPUT_NAMES[1]),
            (sha256_text(metadata_text), OUTPUT_NAMES[2]),
        ]
    )
    return {
        OUTPUT_NAMES[0]: assignment_text,
        OUTPUT_NAMES[1]: ledger_text,
        OUTPUT_NAMES[2]: metadata_text,
        OUTPUT_NAMES[3]: checksum_text,
    }


def verify_assignment(
    by_id: dict[str, dict[str, str]],
    groups: dict[str, list[dict[str, str]]],
    assignments: list[dict[str, str]],
    ledger: list[dict[str, str]],
) -> None:
    assigned_ids = [row["record_id"] for row in assignments]
    omitted_ids = [row["record_id"] for row in ledger]
    assigned_urls = [row["lot_url"] for row in assignments]
    omitted_urls = [row["lot_url"] for row in ledger]
    require(len(assigned_ids) == len(set(assigned_ids)), "Assignment record IDs are not unique")
    require(len(assigned_urls) == len(set(assigned_urls)), "Assignment lot URLs are not unique")
    require(len(omitted_ids) == len(set(omitted_ids)), "Omission record IDs are not unique")
    require(not (set(assigned_ids) & set(omitted_ids)), "A record is both assigned and omitted")
    require(not (set(assigned_urls) & set(omitted_urls)), "A lot URL is both assigned and omitted")
    require(set(assigned_ids) | set(omitted_ids) == set(by_id), "Not every input record is assigned or omitted")
    require(
        set(assigned_urls) | set(omitted_urls) == {row["lot_url"] for row in by_id.values()},
        "Not every input lot URL is assigned or omitted",
    )
    for row in assignments:
        source = by_id[row["record_id"]]
        require(row["partition"] in PARTITIONS, f"Unknown partition for {row['record_id']}")
        require(row["partition"] == source["date_partition"], f"Partition disagrees with sale date for {row['record_id']}")
        require(row["sale_date"] == source["sale_date"], f"Assignment date changed for {row['record_id']}")
        require(row["lot_url"] == source["lot_url"], f"Assignment URL changed for {row['record_id']}")
        check_boundary(row["record_id"], source["sale_day"], row["partition"])
    for row in ledger:
        source = by_id[row["record_id"]]
        require(row["record_id"] in OMISSION_IDS, f"Unexpected omission: {row['record_id']}")
        require(row["lot_url"] == source["lot_url"], f"Omission URL changed for {row['record_id']}")
        check_boundary(row["record_id"], source["sale_day"], row["date_partition"])
    for group_id, members in groups.items():
        retained = [member["record_id"] for member in members if member["record_id"] not in set(omitted_ids)]
        require(retained, f"{group_id} has no retained sale")
        partitions = {by_id[record_id]["date_partition"] for record_id in retained}
        require(len(partitions) == 1, f"{group_id} crosses partitions after the omissions: {sorted(partitions)}")


def check_boundary(record_id: str, sale_day: date, partition: str) -> None:
    if partition == "train":
        require(sale_day < VALIDATION_START, f"{record_id} is in train but dated {sale_day.isoformat()}")
    elif partition == "validation":
        require(
            VALIDATION_START <= sale_day <= VALIDATION_END,
            f"{record_id} is in validation but dated {sale_day.isoformat()}",
        )
    elif partition == "test":
        require(sale_day >= TEST_START, f"{record_id} is in test but dated {sale_day.isoformat()}")
    else:
        raise SystemExit(f"Unknown partition {partition} for {record_id}")


def count_table(rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str, str], tuple[int, date, date]]:
    found: dict[tuple[str, str, str, str, str], tuple[int, date, date]] = {}

    def observe(kind: str, period: str, brand: str, house: str, year: str, sale_day: date) -> None:
        key = (kind, period, brand, house, year)
        lots, first, last = found.get(key, (0, sale_day, sale_day))
        found[key] = (lots + 1, min(first, sale_day), max(last, sale_day))

    for row in rows:
        period = str(row["partition"])
        sale_day = row["sale_day"]
        require(isinstance(sale_day, date), "Sale date was not parsed")
        brand = str(row["brand"])
        house = str(row["auction_house"])
        year = str(sale_day.year)
        observe("partition", period, "", "", "", sale_day)
        observe("brand", period, brand, "", "", sale_day)
        observe("auction_house", period, "", house, "", sale_day)
        observe("brand_house", period, brand, house, "", sale_day)
        observe("sale_year", "", "", "", year, sale_day)
    return found


def feasibility_key(record: dict[str, str]) -> tuple[str, str, str, str, str] | None:
    kind = record["table"]
    if kind == "sale_year":
        return (kind, "", "", "", record["sale_year"])
    if kind not in {"partition", "brand", "auction_house", "brand_house"}:
        return None
    period = FILE_TO_PERIOD[record["proposed_period"]]
    if kind == "partition":
        return (kind, period, "", "", "")
    if kind == "brand":
        return (kind, period, record["brand"], "", "")
    if kind == "auction_house":
        return (kind, period, "", record["auction_house"], "")
    return (kind, period, record["brand"], record["auction_house"], "")


def check_feasibility_counts(root: Path, assignments: list[dict[str, str]], by_id: dict[str, dict[str, str]]) -> None:
    rows: list[dict[str, object]] = []
    for row in assignments:
        source = by_id[row["record_id"]]
        rows.append({**row, "sale_day": source["sale_day"]})
    actual = count_table(rows)
    expected: dict[tuple[str, str, str, str, str], tuple[int, str, str]] = {}
    for record in load_csv(root / "diagnostics/chronological_split_counts.csv"):
        if record["stage"] != "after_proposed_omission":
            continue
        key = feasibility_key(record)
        if key is None:
            continue
        require(key not in expected, f"Feasibility counts repeat {key}")
        expected[key] = (int(record["lots"]), record["date_min"], record["date_max"])
    problems = []
    if set(expected) != set(actual):
        problems.append(f"count keys differ: extra {sorted(set(actual) - set(expected))}, missing {sorted(set(expected) - set(actual))}")
    for key, (lots, first, last) in actual.items():
        saved = expected.get(key)
        if saved is None:
            continue
        saved_lots, saved_first, saved_last = saved
        if lots != saved_lots or first.isoformat() != saved_first or last.isoformat() != saved_last:
            problems.append(
                f"{key}: actual {lots} {first.isoformat()} {last.isoformat()}, "
                f"feasibility {saved_lots} {saved_first} {saved_last}"
            )
    require(
        not problems,
        "Feasibility counts do not match the assignment. Counts were not adjusted.\n" + "\n".join(problems),
    )


def exposure_from_rows(by_id: dict[str, dict[str, str]], random_partitions: dict[str, str]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    require(set(random_partitions) == set(by_id), "Random split IDs do not match the eligible file")
    for record_id, row in by_id.items():
        partition = random_partitions[record_id]
        require(partition in PARTITIONS, f"Unexpected random partition for {record_id}: {partition}")
        counts[(PERIOD_TO_FILE[row["date_partition"]], partition)] += 1
    return dict(counts)


def check_exposure(root: Path, actual: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    saved: dict[tuple[str, str], int] = {}
    for record in load_csv(root / "diagnostics/chronological_prior_exposure_counts.csv"):
        key = (record["proposed_period"], record["random_partition"])
        require(key not in saved, f"Prior exposure repeats {key}")
        saved[key] = int(record["lots"])
    require(saved == actual, "Prior exposure file does not match the eligible dates and random split")
    return saved


def count_records(rows: list[dict[str, str]], partition_field: str) -> list[dict[str, object]]:
    grouped: dict[str, list[date]] = defaultdict(list)
    for row in rows:
        grouped[str(row[partition_field])].append(row["sale_day"])
    records = []
    for partition in (*PARTITIONS, "omitted"):
        days = grouped.get(partition, [])
        records.append(
            {
                "partition": partition,
                "lots": len(days),
                "date_min": "" if not days else min(days).isoformat(),
                "date_max": "" if not days else max(days).isoformat(),
            }
        )
    return records


def grouped_counts(rows: list[dict[str, str]], partition_field: str, key: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        grouped[(str(row[partition_field]), str(row[key]))] += 1
    records = []
    for partition in (*PARTITIONS, "omitted"):
        labels = sorted({label for period, label in grouped if period == partition})
        for label in labels:
            item = {"partition": partition, key: label, "lots": grouped[(partition, label)]}
            records.append(item)
    return records


def build_metadata(
    by_id: dict[str, dict[str, str]],
    groups: dict[str, list[dict[str, str]]],
    assignments: list[dict[str, str]],
    ledger: list[dict[str, str]],
    evidence_hashes: dict[str, str],
    random_assignment_hash: str,
    exposure_counts: dict[tuple[str, str], int],
    column_count: int,
    assignment_sha256: str,
    ledger_sha256: str,
) -> dict[str, object]:
    dated_assignments = []
    for row in assignments:
        dated_assignments.append({**row, "sale_day": by_id[row["record_id"]]["sale_day"]})
    dated_ledger = []
    for row in ledger:
        dated_ledger.append({**row, "sale_day": by_id[row["record_id"]]["sale_day"], "partition": "omitted"})
    combined = dated_assignments + dated_ledger
    partition_counts = count_records(combined, "partition")
    counts = {str(record["partition"]): int(record["lots"]) for record in partition_counts}
    require(sum(counts.values()) == len(by_id), "Partition counts do not account for every input row")
    brand_counts = grouped_counts(combined, "partition", "brand")
    house_counts = grouped_counts(combined, "partition", "auction_house")
    for record in partition_counts:
        partition = str(record["partition"])
        brand_total = sum(int(item["lots"]) for item in brand_counts if item["partition"] == partition)
        house_total = sum(int(item["lots"]) for item in house_counts if item["partition"] == partition)
        require(brand_total == record["lots"], f"Brand counts do not sum to the {partition} total")
        require(house_total == record["lots"], f"Auction-house counts do not sum to the {partition} total")
    previously_used = exposure_counts[("test", "train")] + exposure_counts[("test", "validation")]
    group_report = []
    omitted_ids = {row["record_id"] for row in ledger}
    for group_id in sorted(groups):
        members = sorted(groups[group_id], key=lambda item: item["record_id"])
        retained = [member["record_id"] for member in members if member["record_id"] not in omitted_ids]
        group_report.append(
            {
                "group_id": group_id,
                "record_ids": [member["record_id"] for member in members],
                "omitted_record_ids": [member["record_id"] for member in members if member["record_id"] in omitted_ids],
                "partitions_after_omission": sorted({by_id[record_id]["date_partition"] for record_id in retained}),
            }
        )
    observed = [row["sale_day"] for row in by_id.values()]
    return {
        "script": "scripts/split_chronological.py",
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_path": "processed/expanded_pull_eligible.csv",
        "input_sha256": INPUT_SHA256,
        "input_rows": len(by_id),
        "input_columns": column_count,
        "experiment_rows": len(assignments),
        "omitted_rows": len(ledger),
        "reconciliation": (
            f"{len(by_id)} = {counts['train']} train + {counts['validation']} validation + "
            f"{counts['test']} test + {counts['omitted']} omitted"
        ),
        "date_boundaries": {
            "train": "sale_date < 2024-01-01",
            "validation": "2024-01-01 <= sale_date <= 2024-12-31",
            "test": "sale_date >= 2025-01-01, through the last sale in this file",
        },
        "observed_input_dates": {"min": min(observed).isoformat(), "max": max(observed).isoformat()},
        "repeat_watch_policy": (
            "Omit only an earlier sale of a physical watch confirmed by brand, reference, and case serial "
            "in the saved diagnostics when that watch also sells in a later period. Keep the latest sale in "
            "its own period. Keep every sale when the confirmed sales fall in one period. Do not omit a lot "
            "because it shares a reference."
        ),
        "omitted_record_ids": [row["record_id"] for row in ledger],
        "partition_counts": partition_counts,
        "brand_counts": brand_counts,
        "auction_house_counts": house_counts,
        "confirmed_groups_after_omission": group_report,
        "prior_exposure": {
            "definition": (
                "Eligible lots dated 2025-01-01 or later whose saved random partition was train or validation. "
                "The random test partition was not scored. Omissions do not remove any test-period lot."
            ),
            "random_assignments": "modeling/expanded_pull_split_assignments.csv",
            "random_assignments_sha256": random_assignment_hash,
            "chronological_test_lots": exposure_counts[("test", "test")] + previously_used,
            "chronological_test_in_random_train": exposure_counts[("test", "train")],
            "chronological_test_in_random_validation": exposure_counts[("test", "validation")],
            "chronological_test_in_random_test": exposure_counts[("test", "test")],
            "chronological_test_previously_used": previously_used,
            "cells": [
                {"proposed_period": period, "random_partition": partition, "lots": exposure_counts[(period, partition)]}
                for period in ("training", "validation", "test")
                for partition in PARTITIONS
            ],
        },
        "evidence_sha256": evidence_hashes,
        "output_sha256": {
            OUTPUT_NAMES[0]: assignment_sha256,
            OUTPUT_NAMES[1]: ledger_sha256,
        },
        "checks": {
            "unique_record_ids": True,
            "unique_lot_urls": True,
            "disjoint_partitions": True,
            "date_boundaries": True,
            "complete_accounting": True,
            "omissions_match_saved_evidence": True,
            "no_confirmed_group_crosses_partitions": True,
            "no_reference_only_exclusions": True,
            "feasibility_counts_match": True,
            "prior_exposure_matches_random_split": True,
        },
    }


def publish(root: Path, outputs: dict[str, str]) -> str:
    directory = root / "modeling" / "chronological"
    paths = [directory / name for name in OUTPUT_NAMES]
    existing = [path for path in paths if path.exists()]
    encoded = {name: outputs[name].encode("utf-8") for name in OUTPUT_NAMES}
    if existing and len(existing) != len(paths):
        found = ", ".join(path.name for path in existing)
        raise SystemExit(f"Chronological outputs are incomplete ({found}). Nothing was written.")
    if len(existing) == len(paths):
        for path in paths:
            actual = path.read_bytes()
            require(actual == encoded[path.name], f"{path.name} already exists and does not match this assignment. It was not overwritten.")
        return "verified"
    directory.mkdir(parents=True, exist_ok=True)
    partials = []
    for name in OUTPUT_NAMES:
        partial = directory / f".{name}.partial"
        partial.write_bytes(encoded[name])
        partials.append((partial, directory / name))
    for partial, final in partials:
        require(not final.exists(), f"{final.name} appeared while writing. Nothing was replaced.")
        partial.replace(final)
    for path in paths:
        require(path.read_bytes() == encoded[path.name], f"Written {path.name} did not match the generated bytes")
    return "wrote"


def main() -> None:
    check_boundaries()
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)
    parser = argparse.ArgumentParser(description="Assign the chronological watch-sale split.")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    root = data_root_from(args)
    print(f"data root {root}")
    before = snapshot(root)
    eligible_path = root / "processed" / INPUT_NAME
    input_hash = sha256_file(eligible_path)
    require(input_hash == INPUT_SHA256, f"Input SHA-256 mismatch: {input_hash}")
    eligible, column_count = load_eligible(eligible_path)
    by_id = index_rows(eligible)
    evidence_names = (
        "diagnostics/chronological_repeat_watch_review.csv",
        "diagnostics/chronological_proposed_omissions.csv",
        "diagnostics/chronological_split_counts.csv",
        "diagnostics/chronological_prior_exposure_counts.csv",
        "diagnostics/chronological_split_feasibility.md",
    )
    evidence_hashes = {name: before[name] for name in evidence_names}
    random_path = root / "modeling" / "expanded_pull_split_assignments.csv"
    random_meta = json.loads((root / "modeling" / "expanded_pull_split.json").read_text(encoding="utf-8"))
    require(random_meta.get("input_sha256") == INPUT_SHA256, "Frozen random split belongs to a different eligible file")
    random_rows = load_csv(random_path)
    random_partitions = {}
    for row in random_rows:
        require(row["record_id"] not in random_partitions, f"Random split repeats {row['record_id']}")
        random_partitions[row["record_id"]] = row["partition"]
    groups = confirmed_groups(load_csv(root / evidence_names[0]), by_id, random_partitions)
    selected = select_omissions(groups, load_csv(root / evidence_names[1]), by_id)
    exposure = exposure_from_rows(by_id, random_partitions)
    require(sum(exposure.values()) == len(by_id), "Prior exposure does not account for every eligible lot")
    check_exposure(root, exposure)
    outputs = build_outputs(
        root,
        by_id,
        groups,
        selected,
        evidence_hashes,
        before["modeling/expanded_pull_split_assignments.csv"],
        exposure,
        column_count,
    )
    mode = publish(root, outputs)
    after = snapshot(root)
    require(after == before, "A protected data file changed")
    counts = json.loads(outputs[OUTPUT_NAMES[2]])
    print(counts["reconciliation"])
    for row in csv.DictReader(io.StringIO(outputs[OUTPUT_NAMES[1]])):
        print(f"omitted {row['record_id']} {row['lot_url']}")
    print(f"previously used chronological test lots {counts['prior_exposure']['chronological_test_previously_used']}")
    print(f"script sha256 {counts['script_sha256']}")
    print(f"{mode} {root / 'modeling' / 'chronological'}")


if __name__ == "__main__":
    main()
