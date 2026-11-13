"""Second-pass feature table.

Reads the completed first-pass feature-candidate CSV and writes a new version.
Does not rerun the first-pass enrichment and does not modify that file, the
approved source, the predictor code, the split, or any model scores.

Run from the repository root:

    python scripts/build_feature_candidates_v2.py
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_NAME = "expanded_pull_eligible_feature_candidates.csv"
OUTPUT_NAME = "expanded_pull_eligible_feature_candidates_v2.csv"
EXPECTED_ROWS = 4664

NEW_COLUMNS = [
    "case_diameter_mm",
    "case_diameter_mm_review",
    "gemstones",
    "gemstones_review",
    "box",
    "box_review",
    "papers",
    "papers_review",
    "attachment_type",
    "attachment_type_review",
    "fitted_bracelet_material",
    "fitted_bracelet_material_audit",
    "production_year_normalization_review",
]

CIRCA_YEAR_RE = re.compile(r"(?i)^(?:circa|ca\.?|c\.?)\s*(\d{4})$")
DECADE_RE = re.compile(r"(?i)\b(?:circa|ca\.?|c\.?)?\s*\d{4}\s*s\b|\b\d{4}s\b")
RANGE_RE = re.compile(r"\b\d{4}\s*[-/–—]\s*\d{4}\b")
PLAIN_YEAR_RE = re.compile(r"^\d{4}$")

DIAMETER_REJECT_NOTES = (
    "unlabeled_rectangular_pair",
    "labeled_length_and_width",
    "labeled_width_only",
    "reviewed_diameter_removed",
)

BOX_YES_RE = re.compile(
    r"(?i)\bbox\s*:\s*yes\b|"
    r"\b(?:presentation|fitted|original)\s+box\b|"
    r"\b(?:rolex|cartier|omega|maker's|makers)\s+(?:presentation\s+)?box\b|"
    r"\bwatch\s+box\b"
)
BOX_NO_RE = re.compile(
    r"(?i)\bbox\s*:\s*no\b|"
    r"\b(?:no|without)\s+(?:a\s+|the\s+)?(?:presentation\s+|original\s+)?box(?:es)?\b|"
    r"\bbox(?:es)?\s+not\s+included\b"
)
PAPERS_YES_RE = re.compile(
    r"(?i)\bpapers?\s*:\s*yes\b|"
    r"\bwarranty\s+cards?\b|"
    r"\binternational\s+warrant(?:y|ies)\b|"
    r"\bwarranty\s+papers?\b|"
    r"\boriginal\s+papers?\b|"
    r"\bwith\s+papers\b|"
    r"\bguarantee\s+cards?\b|"
    r"\binternational\s+guarantee\b|"
    r"\bcertificate\s+of\s+guarantee\b|"
    r"\bguarantee\s+certificate\b"
)
PAPERS_NO_RE = re.compile(
    r"(?i)\bpapers?\s*:\s*no\b|"
    r"\b(?:no|without)\s+(?:any\s+|the\s+|original\s+)?papers?\b|"
    r"\bpapers?\s+not\s+included\b"
)
SERVICE_DOC_RE = re.compile(r"(?i)\bservice\s+(?:receipts?|papers?|certificates?)\b")
BARE_CERTIFICATE_RE = re.compile(r"(?i)\bcertificates?\b")
BRACELET_SECTION_RE = re.compile(r"(?i)\bbracelets?\s*:\s*(.{0,180})")
METAL_SPECS = (
    ("stainless steel", re.compile(r"(?i)\b((?:9|14|18|22)\s?k\s+)?stainless(?:\s*|-)steel\b")),
    ("yellow gold", re.compile(r"(?i)\b((?:9|14|18|22)\s?k\s+)?yellow\s+gold\b")),
    ("white gold", re.compile(r"(?i)\b((?:9|14|18|22)\s?k\s+)?white\s+gold\b")),
    ("rose gold", re.compile(r"(?i)\b((?:9|14|18|22)\s?k\s+)?rose\s+gold\b")),
    ("pink gold", re.compile(r"(?i)\b((?:9|14|18|22)\s?k\s+)?pink\s+gold\b")),
    ("red gold", re.compile(r"(?i)\b((?:9|14|18|22)\s?k\s+)?red\s+gold\b")),
    ("platinum", re.compile(r"(?i)\b((?:9|14|18|22)\s?k\s+)?platinum\b")),
    ("titanium", re.compile(r"(?i)\b((?:9|14|18|22)\s?k\s+)?titanium\b")),
    ("steel", re.compile(r"(?i)\b((?:9|14|18|22)\s?k\s+)?steel\b")),
    ("gold", re.compile(r"(?i)\b((?:9|14|18|22)\s?k\s+)?gold\b")),
)
STATED_MATERIALS = {
    "stainless steel",
    "steel",
    "yellow gold",
    "white gold",
    "rose gold",
    "pink gold",
    "red gold",
    "gold",
    "platinum",
    "titanium",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ").replace("\u202f", " ")).strip()


def format_mm(value: str) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, "f").rstrip("0").rstrip(".")


def diameter_is_verified(note: str, status: str) -> bool:
    if status == "diameter_removed" or "reviewed_diameter_removed" in note:
        return False
    if any(token in note for token in DIAMETER_REJECT_NOTES):
        return False
    return (
        note == "single_measurement_from_case_diameter"
        or "labeled_diameter" in note
        or "reviewed_diameter_approved" in note
    )


def case_diameter(row: dict[str, str]) -> tuple[str, str]:
    note = row["case_measurement_note"]
    status = row["case_measurement_correction_status"].strip()
    millimeters = row["case_measurement_mm"].strip()
    if millimeters and not diameter_is_verified(note, status):
        return "", "not_a_case_diameter"
    if not millimeters:
        return "", ""
    if not diameter_is_verified(note, status):
        return "", ""
    return format_mm(millimeters), ""


def yes_no(text: str, yes_re: re.Pattern[str], no_re: re.Pattern[str]) -> tuple[str, str]:
    found_yes = yes_re.search(text) is not None
    found_no = no_re.search(text) is not None
    if found_yes and found_no:
        return "", "conflict"
    if found_yes:
        return "yes", ""
    if found_no:
        return "no", ""
    return "", ""


def box_and_papers(row: dict[str, str]) -> dict[str, str]:
    text = squash(f"{row['item_title']} {row['item_description']}")
    box, box_review = yes_no(text, BOX_YES_RE, BOX_NO_RE)
    papers, papers_review = yes_no(text, PAPERS_YES_RE, PAPERS_NO_RE)
    if not papers and not papers_review and SERVICE_DOC_RE.search(text):
        papers_review = "service_receipt_not_papers"
    elif not papers and not papers_review and BARE_CERTIFICATE_RE.search(text):
        papers_review = "certificate_not_treated_as_warranty"
    return {
        "box": box,
        "box_review": box_review,
        "papers": papers,
        "papers_review": papers_review,
    }


def gemstones(row: dict[str, str]) -> tuple[str, str]:
    value = row["precious_stones_candidate"].strip()
    review = row["precious_stones_candidate_review"].strip()
    if review:
        return "", review
    if value == "yes":
        return "yes", ""
    if value == "no":
        return "no", ""
    return "", ""


def metals_in(text: str) -> list[str]:
    found: list[str] = []
    occupied = [False] * (len(text) + 1)
    for label, pattern in METAL_SPECS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(occupied[start:end]):
                continue
            for position in range(start, end):
                occupied[position] = True
            karat = re.sub(r"\s+", "", (match.group(1) or "").lower())
            phrase = f"{karat} {label}".strip() if karat else label
            if phrase not in found:
                found.append(phrase)
    return found


def bracelet_section_material(description: str) -> list[str]:
    materials: list[str] = []
    for match in BRACELET_SECTION_RE.finditer(description or ""):
        section = match.group(1)
        section = re.split(r"(?i)\baccompaniments?\s*:", section)[0]
        for phrase in metals_in(section):
            if phrase not in materials:
                materials.append(phrase)
    return materials


def normalize_stated_material(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def attachment_and_material(row: dict[str, str]) -> dict[str, str]:
    candidate = row["attachment_type_candidate"].strip()
    review = row["attachment_type_candidate_review"]
    material = normalize_stated_material(row["bracelet_material_candidate"])
    result = {
        "attachment_type": "",
        "attachment_type_review": "",
        "fitted_bracelet_material": "",
        "fitted_bracelet_material_audit": "",
    }
    if "both_strap_and_bracelet" in review or candidate == "unknown":
        reason = "both_mentioned" if "both_strap_and_bracelet" in review else "unstated"
        if "associated_attachment" in review and reason == "unstated":
            reason = "associated_attachment"
        elif "associated_attachment" in review and reason == "both_mentioned":
            reason = "both_mentioned;associated_attachment"
        result["attachment_type_review"] = reason
        result["fitted_bracelet_material_audit"] = "attachment_unresolved"
        return result
    if "associated_attachment" in review:
        result["attachment_type_review"] = "associated_attachment"
        result["fitted_bracelet_material_audit"] = "attachment_unresolved"
        return result
    if candidate not in {"strap", "bracelet"}:
        result["attachment_type_review"] = "unstated"
        result["fitted_bracelet_material_audit"] = "attachment_unresolved"
        return result
    result["attachment_type"] = candidate
    if candidate == "strap":
        result["fitted_bracelet_material_audit"] = "strap"
        return result
    material_review = row["bracelet_material_candidate_review"]
    if "multiple_bracelet_materials" in material_review:
        phrases = bracelet_section_material(row["item_description"])
        if not phrases:
            field = normalize_stated_material(row["bracelet_material"])
            if field in STATED_MATERIALS:
                phrases = [field]
        if len(phrases) > 1:
            result["fitted_bracelet_material"] = " and ".join(phrases)
            result["fitted_bracelet_material_audit"] = "combination"
            return result
        if len(phrases) == 1:
            result["fitted_bracelet_material"] = phrases[0]
            result["fitted_bracelet_material_audit"] = "stated"
            return result
        result["fitted_bracelet_material_audit"] = "bracelet_material_unstated"
        return result
    if material in STATED_MATERIALS:
        result["fitted_bracelet_material"] = material
        result["fitted_bracelet_material_audit"] = "stated"
        return result
    result["fitted_bracelet_material_audit"] = "bracelet_material_unstated"
    return result


def normalize_year(row: dict[str, str]) -> tuple[str, str]:
    text = row["production_year"].strip()
    if not text:
        return "", ""
    if DECADE_RE.search(text) or RANGE_RE.search(text):
        return "", "unresolved_date"
    circa = CIRCA_YEAR_RE.fullmatch(text)
    if circa:
        year = circa.group(1)
        parsed = row["production_year_parsed"].strip()
        if parsed and parsed != year:
            return text, "normalized_year_disagrees_with_parsed"
        return year, ""
    if PLAIN_YEAR_RE.fullmatch(text):
        return text, ""
    return text, "unresolved_date"


def update_row(row: dict[str, str]) -> dict[str, str]:
    updated = dict(row)
    diameter, diameter_review = case_diameter(row)
    stones, stones_review = gemstones(row)
    year, year_review = normalize_year(row)
    updated["production_year"] = year
    updated.update(
        {
            "case_diameter_mm": diameter,
            "case_diameter_mm_review": diameter_review,
            "gemstones": stones,
            "gemstones_review": stones_review,
            "production_year_normalization_review": year_review,
        }
    )
    updated.update(box_and_papers(row))
    updated.update(attachment_and_material(row))
    return updated


def self_check() -> None:
    assert format_mm("37.5") == "37.5"
    assert format_mm("40") == "40"
    assert format_mm("40.0") == "40"
    assert CIRCA_YEAR_RE.fullmatch("c. 1969").group(1) == "1969"
    assert CIRCA_YEAR_RE.fullmatch("c1969").group(1) == "1969"
    assert CIRCA_YEAR_RE.fullmatch("circa 1969").group(1) == "1969"
    assert CIRCA_YEAR_RE.fullmatch("c. 2008").group(1) == "2008"
    assert DECADE_RE.search("c. 2000s")
    assert RANGE_RE.search("1968-1970")
    assert not diameter_is_verified("single_measurement_from_case_diameter|unlabeled_rectangular_pair", "")
    assert diameter_is_verified("single_measurement_from_case_diameter", "")
    box, review = yes_no("Box: Yes; Papers: No", BOX_YES_RE, BOX_NO_RE)
    papers, papers_review = yes_no("Box: Yes; Papers: No", PAPERS_YES_RE, PAPERS_NO_RE)
    assert (box, review, papers, papers_review) == ("yes", "", "no", "")
    assert metals_in("Stainless steel and 18k pink gold") == ["stainless steel", "18k pink gold"]
    assert metals_in("stainless steel bracelet") == ["stainless steel"]


def verify(original_fields: list[str], original: list[dict[str, str]], written: list[dict[str, str]]) -> None:
    if [row["record_id"] for row in original] != [row["record_id"] for row in written]:
        raise SystemExit("record_id values or row order changed.")
    allowed = set(NEW_COLUMNS) | {"production_year"}
    for before, after in zip(original, written):
        for name in original_fields:
            if name in allowed:
                continue
            if before[name] != after[name]:
                raise SystemExit(f"Changed field outside this update: {name} on {before['record_id']}")
        if before["production_year_original"] != after["production_year_original"]:
            raise SystemExit("production_year_original changed.")


def count_report(rows: list[dict[str, str]]) -> None:
    print("\nCounts")
    fields = [
        "case_diameter_mm",
        "gemstones",
        "box",
        "papers",
        "attachment_type",
        "fitted_bracelet_material",
        "production_year",
    ]
    for name in fields:
        populated = sum(bool(row[name]) for row in rows)
        print(f"  {name}: populated {populated}, blank {len(rows) - populated}")
    print("  gemstones values:", end=" ")
    print({value: sum(row["gemstones"] == value for row in rows) for value in ("yes", "no", "")})
    print("  box values:", end=" ")
    print({value: sum(row["box"] == value for row in rows) for value in ("yes", "no", "")})
    print("  papers values:", end=" ")
    print({value: sum(row["papers"] == value for row in rows) for value in ("yes", "no", "")})
    print("  attachment_type values:", end=" ")
    print({value: sum(row["attachment_type"] == value for row in rows) for value in ("strap", "bracelet", "")})
    print("  material audit:")
    audits: dict[str, int] = {}
    for row in rows:
        key = row["fitted_bracelet_material_audit"] or "(blank audit)"
        audits[key] = audits.get(key, 0) + 1
    for key, count in sorted(audits.items(), key=lambda item: -item[1]):
        print(f"    {key}: {count}")
    circa = sum(
        bool(CIRCA_YEAR_RE.fullmatch(row["production_year_original"].strip()))
        and row["production_year"] == CIRCA_YEAR_RE.fullmatch(row["production_year_original"].strip()).group(1)
        for row in rows
    )
    print(f"  production years normalized from circa text: {circa}")
    print("  flags:")
    for name in (
        "case_diameter_mm_review",
        "gemstones_review",
        "box_review",
        "papers_review",
        "attachment_type_review",
        "production_year_normalization_review",
    ):
        flagged = sum(bool(row[name]) for row in rows)
        print(f"    {name}: {flagged}")


def main() -> None:
    self_check()
    folder = REPO_ROOT / "data"
    source = folder / INPUT_NAME
    destination = folder / OUTPUT_NAME
    if destination.exists():
        raise SystemExit(f"{destination.name} already exists. No file written.")
    print(f"Input path: {source}")
    print(f"Input SHA-256: {sha256(source)}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != EXPECTED_ROWS or len({row["record_id"] for row in rows}) != EXPECTED_ROWS:
        raise SystemExit("Input does not have 4,664 unique record IDs.")
    if any(row["production_year"] != row["production_year_original"] for row in rows):
        raise SystemExit("production_year and production_year_original differ. No file written.")
    if "case_diameter_mm" in fields:
        raise SystemExit("case_diameter_mm already exists. No file written.")
    updated = [update_row(row) for row in rows]
    output_fields = fields + NEW_COLUMNS
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(updated)
    with destination.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    verify(fields, rows, written)
    print(f"Output path: {destination}")
    print(f"Output SHA-256: {sha256(destination)}")
    print(f"Output rows: {len(written)}")
    print(f"Output columns: {len(written[0])}")
    count_report(written)


if __name__ == "__main__":
    main()
