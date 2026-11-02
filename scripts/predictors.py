"""Deterministic predictor construction for the expanded-pull eligible file.

Importing this module does not read data, write files, fit a model, impute,
scale, learn categories, select features, or score a partition.

The functions below are the audited construction rules. Reviewed overrides take
precedence. An approved blank stays blank. An unresolved source conflict stays
missing, with no fallback to the raw field. Component distinctions and the
gold-versus-plating safeguards are unchanged. ``not_stated`` means no finish
was stated. It does not mean plating was verified absent.

Usage
-----
From the repository root, with ``WATCH_DATA_DIR`` set to the shared data root::

    import csv
    import os
    from pathlib import Path

    from scripts.predictors import INPUT_SHA256, construct_record

    path = Path(os.environ["WATCH_DATA_DIR"]) / "processed" / "expanded_pull_eligible.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    predictors = [construct_record(row) for row in rows]

Load the authoritative CSV so source strings stay strings. A spreadsheet resave
is not a substitute for that file. The required SHA-256 is ``INPUT_SHA256``.

Field mapping
-------------
``modeling_case_body`` contains the reviewed case-body overrides. It stays blank
until a material review writes one. When ``material_review_status`` is
``approved``, that override becomes ``case_body``. An approved blank stays
missing, and its blank finish becomes ``not_stated``. When the status is
``unresolved_source_conflict``, the body and finish stay missing. On every other
row, ``case_body`` comes from structured ``case_material`` and
``extracted_case_base_materials``, including the narrow compatibility rules.

``modeling_movement`` is an intermediate column. Cleaning stores a
non-conflicting extracted movement there before any review. The final
``movement`` predictor uses that column only when ``movement_review_status`` is
``approved``. An approved blank stays missing. An unresolved movement conflict
stays missing. On every other row, ``movement`` comes from the structured
movement field and ``extracted_movement``. Generic mechanical wording does not
override a more specific automatic or manual reading.

``production_year`` is ``production_year_parsed`` when that field is present.
``age_at_sale`` is the sale year minus that year when both exist and the
production year is not after the sale year. ``production_year_approximate`` is
``true`` or ``false`` only when a parsed year exists. A stored ``false`` with
no parsed year stays missing. The saved modeling table records the age and the
approximate indicator; it does not have a separate production-year column.

``case_diameter_mm`` is ``case_measurement_mm``. ``case_width_mm`` and
``case_length_mm`` are the labeled side fields. Bracelet circumference,
unlabeled rectangular sides, and a raw diameter removed by the cleaner are not
copied into the diameter.

Missing values
--------------
An empty constructed value is missing. ``not_stated`` is a populated finish
label. ``source_disagreement`` and ``unresolved_source_conflict`` are source
labels, not filled predictor values.

Verified completeness
---------------------
Against ``processed/expanded_pull_eligible.csv`` (SHA-256 ``INPUT_SHA256``,
4,664 rows) and ``modeling/expanded_pull_modeling_rows.csv`` (SHA-256
``d2ee0efa168114f5e88a381943cf9969afb0032816ff3b1cf9b299e79777fad7``), every
compared field matched by record ID. Populated values: normalized model 4,103;
production year 4,318; age at sale 4,318; approximate-year indicator 4,318;
case body 4,194; case finish 4,249; movement 3,241; diameter 2,598; width 70;
length 55. The 4,249 populated finishes contain 4,113 values of ``not_stated``,
131 ``gold-plated``, 4 ``vermeil``, and 1 ``rose-gold-plated``. The earlier
audit count of 4,065 is the source label ``no_finish_stated``. All 4,065 of
those rows have the value ``not_stated``. The other 48 ``not_stated`` values
are approved corrections whose reviewed finish was blank. The modeling table
has no production-year column. The constructed year matched
``production_year_parsed``, and every populated age was the sale year minus
that year.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

INPUT_SHA256 = "cf3b41d890af1feba9348fa9a73d954fcab193ed890a39044bd0355198fc5a8a"
ELIGIBLE_ROWS = 4664


PLACEHOLDER_TOKENS = {"-", "none", "unknown"}

SKIP_MATERIAL_TOKENS = {
    "a", "an", "the", "and", "or", "of", "in", "en", "de", "with", "for",
    "chronograph", "chronographe", "automatic", "calendar", "quartz", "manual",
    "mechanical", "wind", "winding", "limited", "edition", "fine", "rare",
    "lady", "ladys", "ladies", "mens", "men", "reversible", "fluted",
    "rectangular", "professional", "oyster", "centre", "center", "seconds",
    "date", "day", "diamond", "diamond-set", "set", "polished", "brushed",
    "small", "ultra-thin", "cushion", "tonneau", "oval", "integrated",
    "fitted", "smooth", "woven", "link", "form", "octagonal", "black",
    "blue", "green", "grey", "gray", "white", "red", "pink", "mother",
    "pearl", "new", "old", "stock", "officially", "certified", "snap-on",
    "screw", "down", "back", "secured", "by",
}
STRUCTURED_MATERIALS = {
    "stainless steel": {"steel"},
    "yellow gold": {"yellow-gold"},
    "steel & gold": {"steel", "gold"},
    "silver": {"silver"},
    "white gold": {"white-gold"},
    "rose gold": {"rose-gold"},
    "pink gold": {"pink-gold"},
    "gold": {"gold"},
    "18k yellow gold": {"yellow-gold"},
    "platinum": {"platinum"},
    "gold-plated": {"gold-plated"},
    "18k gold": {"gold"},
    "9k gold": {"gold"},
    "ceramic": {"ceramic"},
}
MATERIAL_GROUPS = {
    "steel": "steel",
    "steel-and-gold": "steel-and-gold",
    "titanium-and-gold": "titanium-and-gold",
    "yellow-gold": "gold",
    "white-gold": "gold",
    "rose-gold": "gold",
    "pink-gold": "gold",
    "red-gold": "gold",
    "three-colour-gold": "gold",
    "karat-gold": "gold",
    "gold": "gold",
    "gold-plated": "plated",
    "rose-gold-plated": "plated",
    "vermeil": "plated",
    "gilt": "gilt",
    "silver": "silver",
    "platinum": "platinum",
    "titanium": "titanium",
    "ceramic": "ceramic",
    "gilt-brass": "gilt",
    "gilt-metal": "gilt",
    "honey-gold": "honey-gold",
    "bi-color-gold": "bi-color-gold",
}

FINISH_LABELS = {"gold-plated", "rose-gold-plated", "vermeil", "gilt"}

def is_placeholder(field: str, value: object) -> bool:
    if not isinstance(value, str):
        return False
    token = value.strip().casefold()
    if token == "" or token in PLACEHOLDER_TOKENS:
        return True
    return field == "condition_report" and token == "report:"

def quoted_model(text: str, start: int, end: int) -> bool:
    before = text[:start]
    opening = max(before.rfind("'"), before.rfind("‘"))
    if opening < 0:
        return False
    closing = text.find("'", end)
    if closing < 0:
        closing = text.find("’", end)
    if closing < 0:
        return False
    inside = text[opening:closing]
    return re.search(r"\b(?:wristwatch|watch|montre|case|bo[iî]tier)\b", inside, flags=re.I) is None


def component_after(text: str, end: int) -> str | None:
    window = text[end:end + 140]
    hard = re.search(r"[|.\n]", window)
    if hard:
        window = window[:hard.start()]
    saw_model_name = False
    skips = 0
    for segment_index, segment in enumerate(window.split(",")):
        tokens = [token.strip("'’") for token in re.findall(r"[A-Za-zÀ-ÿ'’\-]+", segment)]
        tokens = [token for token in tokens if token]
        index = 0
        while index < len(tokens) and skips <= 12:
            token = tokens[index]
            if index + 1 < len(tokens):
                pair = f"{token} {tokens[index + 1]}".casefold()
                if pair in {
                    "bracelet watch", "bracelet wristwatch", "wrist watch", "pocket watch",
                    "pendant watch", "montre bracelet", "montres bracelet", "bangle watch",
                }:
                    return "case"
            lowered = token.casefold().replace("'", "")
            if lowered in {
                "wristwatch", "wristwatches", "wrstwatch", "montre", "montres", "timepiece",
                "watch", "watches", "clock", "boitier", "boîtier", "pocket-watch", "wrist-watch",
            }:
                return "case"
            if lowered == "case":
                previous = tokens[index - 1].casefold() if index else ""
                if previous in {
                    "cigarette", "presentation", "jewelry", "jewellery", "leatherette",
                    "display", "travel", "outer", "card", "carrying", "wooden", "storage",
                    "gift", "original",
                }:
                    skips += 1
                    index += 1
                    continue
                return "case"
            if lowered in {"bezel", "lunette"}:
                return "model" if saw_model_name else "bezel"
            if lowered in {"dial", "cadran"}:
                return "model" if saw_model_name else "dial"
            if lowered in {"buckle", "clasp"}:
                return "model" if saw_model_name else "buckle"
            if lowered in {"bracelet", "strap"}:
                return "model" if saw_model_name else "bracelet"
            if lowered == "award":
                return "model"
            if lowered in {"crown", "pusher", "lighter", "pen", "pens", "cigarette"}:
                return "other" if segment_index == 0 else None
            if token[:1].isupper() and not token.isupper() and any(character.islower() for character in token):
                saw_model_name = True
                skips += 1
                index += 1
                continue
            if lowered in SKIP_MATERIAL_TOKENS or token.isupper() or token.islower() or "-" in token:
                skips += 1
                index += 1
                continue
            return "ambiguous" if segment_index == 0 else None
        # A comma in "yellow gold, diamond-set wristwatch" still belongs to the watch.
    if saw_model_name:
        return "model"
    return None


def component_before(text: str, start: int) -> str | None:
    window = text[max(0, start - 80):start]
    cut = max(window.rfind("|"), window.rfind("."), window.rfind("\n"), window.rfind(","))
    if cut >= 0:
        window = window[cut + 1:]
    window = re.sub(r"\bmontres?\s+bracelet\b", "montre", window, flags=re.I)
    window = re.sub(r"\bchronographs?\s+bracelet\b", "chronograph", window, flags=re.I)
    window = re.sub(r"\bchronographes?\s+bracelet\b", "chronographe", window, flags=re.I)
    window = re.sub(r"\bbracelet\s+watch(?:es)?\b", "watch", window, flags=re.I)
    heads = [
        ("other", r"\b(?:lighter|pens?|cigarette)\b"),
        ("bezel", r"\b(?:bezel|lunette)\b"),
        ("dial", r"\b(?:dial|cadran)\b"),
        ("buckle", r"\b(?:buckle|clasp)\b"),
        ("bracelet", r"\b(?:bracelet|strap)\b"),
        ("case", r"\b(?:montres?|chronographes?|chronographs?|wristwatches?|wrstwatch|watches?|timepiece|case|bo[iî]tier|pocket-watch|wrist-watch)\b"),
    ]
    best_name = None
    best_at = -1
    for name, pattern in heads:
        found = None
        for found in re.finditer(pattern, window, flags=re.I):
            pass
        if found and found.start() > best_at:
            best_name = name
            best_at = found.start()
    return best_name


def classify_material(text: str, start: int, end: int, source: str) -> str:
    if quoted_model(text, start, end):
        return "model"
    following = component_after(text, end)
    if following in {"case", "bezel", "dial", "buckle", "bracelet", "model", "other"}:
        return following
    preceding = component_before(text, start)
    if preceding:
        return preceding
    if source == "item_description_case" and following is None:
        return "case"
    return "ambiguous"

def structured_material_tokens(value: object) -> set[str]:
    if not isinstance(value, str) or is_placeholder("case_material", value):
        return set()
    return set(STRUCTURED_MATERIALS.get(value.strip().casefold(), set()))

def expand_base_groups(labels: list[str]) -> set[str]:
    groups = set()
    for label in labels:
        group = MATERIAL_GROUPS.get(label, label)
        if group == "steel-and-gold":
            groups.update({"steel", "gold"})
        elif group == "titanium-and-gold":
            groups.update({"titanium", "gold"})
        elif group:
            groups.add(group)
    return groups


def materials_conflict(structured: set[str], base_labels: list[str], finish_labels: list[str]) -> bool:
    if not structured:
        return False
    structured_groups = {MATERIAL_GROUPS.get(token, token) for token in structured}
    base_groups = expand_base_groups(base_labels)
    finish_groups = {MATERIAL_GROUPS.get(label, label) for label in finish_labels if label}
    if not base_groups and not finish_groups:
        return False
    if structured_groups <= {"plated", "gilt"} and finish_groups & structured_groups:
        return False
    if not base_groups:
        return structured_groups.isdisjoint(finish_groups)
    if base_groups.isdisjoint(structured_groups):
        if structured_groups <= {"plated", "gilt"} and finish_groups & structured_groups:
            return False
        return True
    if "plated" in structured_groups and "gold" in base_groups and not finish_groups & {"plated", "gilt"}:
        return True
    return False

def movement_family(text: str) -> str:
    lowered = text.casefold()
    has_quartz = re.search(r"\bquartz\b", lowered) is not None
    has_automatic = re.search(r"\b(?:automatic|automatique)\b", lowered) is not None
    has_manual = re.search(
        r"\b(?:manual(?:ly)?\s+wound|manual\s+wind(?:ing)?|hand[-\s]?wound|hand[-\s]?wind|remontage\s+manuel)\b",
        lowered,
    ) is not None
    has_mechanical = re.search(r"\b(?:mechanical|m[ée]canique|mecanique)\b", lowered) is not None
    families = []
    if has_quartz:
        families.append("quartz")
    if has_automatic:
        families.append("automatic")
    if has_manual:
        families.append("manual")
    if has_mechanical:
        families.append("mechanical")
    if not families:
        return ""
    if "quartz" in families and any(family in families for family in ("automatic", "manual", "mechanical")):
        return "conflict:" + "+".join(families)
    if "automatic" in families and "manual" in families:
        return "conflict:" + "+".join(family for family in families if family != "mechanical")
    if "mechanical" in families and len(families) > 1:
        families = [family for family in families if family != "mechanical"]
    if len(families) == 1:
        return families[0]
    return "conflict:" + "+".join(families)

def movements_compatible(left: str, right: str) -> bool:
    if not left or not right or left.startswith("conflict:") or right.startswith("conflict:"):
        return False
    if left == right:
        return True
    pair = {left, right}
    if "quartz" in pair:
        return False
    return pair <= {"mechanical", "automatic"} or pair <= {"mechanical", "manual"}


def choose_movement(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left.startswith("conflict:"):
        return left
    if right.startswith("conflict:"):
        return right
    if movements_compatible(left, right):
        if left == "mechanical" and right != "mechanical":
            return right
        if right == "mechanical" and left != "mechanical":
            return left
        return left
    return f"conflict:{left}+{right}"


def structured_movement_family(value: object) -> str:
    if not isinstance(value, str) or is_placeholder("movement", value):
        return ""
    return movement_family(value)


def movements_conflict(structured: str, extracted: str) -> bool:
    if not structured or not extracted:
        return False
    if structured.startswith("conflict:") or extracted.startswith("conflict:"):
        return True
    return not movements_compatible(structured, extracted)

GENERIC_GOLD = {"gold", "karat-gold"}
COLOR_LABELS = {"yellow-gold", "white-gold", "rose-gold", "pink-gold", "red-gold"}
OTHER_BODY_METALS = {"steel", "platinum", "silver", "titanium"}
COMPONENT_SECTIONS = {
    "dial",
    "strap/bracelet",
    "buckle/clasp",
    "closure",
    "movement",
    "caliber",
    "calibre",
    "model",
    "model/reference",
    "reference",
    "date",
    "signed",
    "size",
    "weight",
    "accompaniments",
    "box",
    "papers",
    "accessories",
    "case number",
}
SECTION_RE = re.compile(
    r"(?i)\b(strap/bracelet|buckle/clasp|case number|case|dial|caliber|calibre|movement|"
    r"model/reference|model|reference|date|closure|signed|size|weight|accompaniments|"
    r"box|papers|accessories)\s*:"
)
PATTERNS = (
    ("three-colour-gold", re.compile(r"\b(?:three[\s-]?colou?r|tri[\s-]?colou?r)\b(?:\W+\w+){0,5}\W+\bgold\b|\btrois\s+ors\b", re.I)),
    ("bi-color-gold", re.compile(r"\b(?:bi[\s-]?colou?r)\b(?:\W+\w+){0,5}\W+\bgold\b", re.I)),
    ("gold-plated", re.compile(r"\b(?:gold[\s-]plated|gold[\s-]filled|rolled gold|gold plating|plaqu[eé]\s+or|vermeil|gilt)\b", re.I)),
    ("yellow-gold", re.compile(r"\b(?:yellow gold|or jaune)\b", re.I)),
    ("white-gold", re.compile(r"\b(?:white gold|or blanc)\b", re.I)),
    ("rose-gold", re.compile(r"\b(?:rose gold|pink gold|or rose)\b", re.I)),
    ("red-gold", re.compile(r"\b(?:red gold|or rouge)\b", re.I)),
    ("karat-gold", re.compile(r"\b(?:9|14|18)\s?(?:k|ct)\s+gold\b", re.I)),
    ("steel", re.compile(r"\b(?:stainless steel|en acier|acier|steel)\b", re.I)),
    ("platinum", re.compile(r"\bplatinum\b", re.I)),
    ("silver", re.compile(r"\bsilver\b", re.I)),
    ("titanium", re.compile(r"\btitanium\b", re.I)),
    ("generic-gold", re.compile(r"\ben\s+or\b(?!\s+(?:jaune|blanc|rose|rouge))|\bgold\b", re.I)),
)


@dataclass(frozen=True)
class CompatibilityDecision:
    applied: bool
    body: str
    source: str
    rule: str
    provenance: str
    evidence: str
    withheld_reason: str


def normalize_body_token(label: str) -> str:
    if label == "pink-gold":
        return "rose-gold"
    return label


def prepare_body_labels(labels: list[str], finish_labels: set[str]) -> list[str]:
    tokens = []
    for label in labels:
        if label in finish_labels or not label:
            continue
        label = normalize_body_token(label)
        if label not in tokens:
            tokens.append(label)
    if set(tokens) == {"steel", "gold"}:
        return ["steel-and-gold"]
    if "steel-and-gold" in tokens and "steel" in tokens:
        tokens = [token for token in tokens if token != "steel"]
    return tokens


def _section_name(text: str, index: int) -> str:
    previous = ""
    for match in SECTION_RE.finditer(text):
        if match.start() > index:
            break
        previous = match.group(1).casefold()
    return previous or "preamble"


def _case_body_mention(text: str, start: int, end: int, label: str, section: str) -> bool:
    if section in COMPONENT_SECTIONS:
        return False
    if section not in {"title", "case", "preamble"}:
        return False
    following = text[end:end + 48]
    if label == "steel" and re.search(r"\b(?:hands?|backs?)\b", following, re.I):
        return False
    role = classify_material(text, start, end, "item_title")
    if role in {"bezel", "dial", "bracelet", "buckle", "model", "other"}:
        return False
    return True


def case_body_mentions(title: str, description: str) -> list[tuple[str, str, str]]:
    """Material mentions that describe the auctioned watch's case body."""
    description_text = description or ""
    found = []
    chunks = [("title", title or ""), ("description", description_text)]
    for region, text in chunks:
        occupied = [False] * (len(text) + 1)
        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span()
                if any(occupied[start:end]):
                    continue
                section = "title" if region == "title" else _section_name(text, start)
                if not _case_body_mention(text, start, end, label, section):
                    continue
                for position in range(start, end):
                    occupied[position] = True
                snippet = re.sub(r"\s+", " ", text[max(0, start - 40):min(len(text), end + 40)]).strip()
                found.append((label, section, snippet))
    return found


def _evidence_text(mentions: list[tuple[str, str, str]]) -> str:
    seen = []
    for label, section, snippet in mentions:
        item = f"{section}:{label}:{snippet}"
        if item not in seen:
            seen.append(item)
    return " || ".join(seen)


def assess_case_compatibility(
    structured_labels: list[str],
    extracted_labels: list[str],
    title: str,
    description: str,
    finish_labels: set[str],
) -> CompatibilityDecision:
    """Return a narrow compatibility decision, or applied=False when the row is out of scope."""
    empty = CompatibilityDecision(False, "", "", "", "", "", "")
    structured_raw = [label for label in structured_labels if label and label not in finish_labels]
    extracted_raw = [label for label in extracted_labels if label and label not in finish_labels]
    structured = prepare_body_labels(structured_raw, finish_labels)
    extracted = prepare_body_labels(extracted_raw, finish_labels)
    rose_scope = set(structured_raw) <= {"rose-gold", "pink-gold"} and set(extracted_raw) <= {"rose-gold", "pink-gold"} and bool(structured_raw and extracted_raw)
    steel_scope = "steel-and-gold" in extracted_raw and "steel" in extracted_raw and structured == ["steel-and-gold"]
    gold_scope = len(structured) == 1 and structured[0] in {"yellow-gold", "rose-gold"} and bool(extracted) and set(extracted) <= GENERIC_GOLD
    if not (rose_scope or steel_scope or gold_scope):
        return empty
    mentions = case_body_mentions(title, description)
    labels = {label for label, _section, _snippet in mentions}
    evidence = _evidence_text(mentions)

    if rose_scope:
        if set(structured_raw) != set(extracted_raw) or "pink-gold" in structured_raw or "pink-gold" in extracted_raw:
            blockers = _body_blockers(labels, "rose-gold")
            if blockers:
                return CompatibilityDecision(
                    True, "", "source_disagreement", "rose_pink_synonym", "", evidence, "; ".join(blockers)
                )
            return CompatibilityDecision(
                True,
                "rose-gold",
                "rose_pink_synonym",
                "rose_pink_synonym",
                "Rose gold and pink gold are one modeling color. The structured field supplies the label when it says rose gold; a catalogue pink-gold wording is the same color, not a second metal.",
                evidence,
                "",
            )

    if steel_scope:
        blockers = _body_blockers(labels, "", allowed_metals={"steel"})
        if blockers:
            return CompatibilityDecision(
                True, "", "source_disagreement", "redundant_steel_and_gold", "", evidence, "; ".join(blockers)
            )
        return CompatibilityDecision(
            True,
            "steel-and-gold",
            "redundant_steel_and_gold",
            "redundant_steel_and_gold",
            "The extracted steel token repeats steel-and-gold. It is not a third case metal.",
            evidence,
            "",
        )

    if gold_scope:
        color = structured[0]
        blockers = _body_blockers(labels, color)
        if blockers:
            return CompatibilityDecision(
                True,
                "",
                "source_disagreement",
                "structured_color_with_generic_gold",
                "",
                evidence,
                "; ".join(blockers),
            )
        if labels & (GENERIC_GOLD | {color}):
            return CompatibilityDecision(
                True,
                color,
                "structured_color_with_generic_gold",
                "structured_color_with_generic_gold",
                f"The {color} color comes from the structured case_material field. Generic karat gold in the catalogue names the metal and the fineness; it does not independently name the color.",
                evidence,
                "",
            )
        return CompatibilityDecision(
            True,
            "",
            "source_disagreement",
            "structured_color_with_generic_gold",
            "",
            evidence,
            "The title and case section do not state generic gold or the structured color for the case body.",
        )
    return empty


def _body_blockers(labels: set[str], retained_color: str, allowed_metals: set[str] | None = None) -> list[str]:
    blockers = []
    if "three-colour-gold" in labels:
        blockers.append("The title or case section gives three-colour gold, so a single structured color is not restored.")
    if "bi-color-gold" in labels:
        blockers.append("The title or case section gives bi-colour gold, so a single structured color is not restored.")
    if "gold-plated" in labels:
        blockers.append("Plating, gold-filling, or vermeil describes the case, so it is not treated as solid gold.")
    conflicting = (labels & COLOR_LABELS) - {retained_color}
    if conflicting:
        blockers.append(
            "The title or case section names a different case color: " + ", ".join(sorted(conflicting)) + "."
        )
    extra = labels & OTHER_BODY_METALS
    if allowed_metals:
        extra -= allowed_metals
    if extra:
        blockers.append(
            "The title or case section names another case-body metal: " + ", ".join(sorted(extra)) + "."
        )
    return blockers

def split_labels(value: object) -> list[str]:
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    return [part for part in text.split("|") if part]


def canonical_body(labels: list[str]) -> str:
    tokens = prepare_body_labels(labels, FINISH_LABELS)
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return "|".join(sorted(tokens))


def body_groups(labels: list[str]) -> set[str]:
    return expand_base_groups([label for label in labels if label not in FINISH_LABELS])


def specificity(labels: list[str]) -> int:
    score = 0
    for label in labels:
        if label in FINISH_LABELS:
            continue
        if label != MATERIAL_GROUPS.get(label, label):
            score += 1
    return score


def agree_body(structured_labels: list[str], extracted_labels: list[str]) -> tuple[str, str]:
    structured_groups = body_groups(structured_labels)
    extracted_groups = body_groups(extracted_labels)
    structured_label = canonical_body(structured_labels)
    extracted_label = canonical_body(extracted_labels)
    if structured_groups and extracted_groups and structured_groups != extracted_groups:
        return "", "source_disagreement"
    if structured_label and extracted_label:
        if structured_label == extracted_label:
            return structured_label, "structured_and_extracted"
        structured_score = specificity(structured_labels)
        extracted_score = specificity(extracted_labels)
        if extracted_score > structured_score:
            return extracted_label, "structured_and_extracted"
        if structured_score > extracted_score:
            return structured_label, "structured_and_extracted"
        return "", "source_disagreement"
    if structured_label:
        return structured_label, "structured_case_material"
    if extracted_label:
        return extracted_label, "extracted_case_base"
    return "", "missing"


def choose_finish(
    structured_tokens: set[str],
    extracted_finish: list[str],
    body_selected: bool,
) -> tuple[str, str]:
    structured_finish = [token for token in sorted(structured_tokens) if token in FINISH_LABELS]
    if len(set(extracted_finish)) > 1:
        return "", "source_disagreement"
    if structured_finish and extracted_finish and set(structured_finish) != set(extracted_finish):
        return "", "source_disagreement"
    if extracted_finish and structured_finish:
        return extracted_finish[0], "structured_and_extracted"
    if extracted_finish:
        return extracted_finish[0], "extracted_case_finish"
    if structured_finish:
        return structured_finish[0], "structured_case_material"
    if body_selected:
        return "not_stated", "no_finish_stated"
    return "", "missing"


def choose_case_features(row: dict) -> tuple[str, str, str, str]:
    status = row["material_review_status"]
    if status == "unresolved_source_conflict":
        return "", "unresolved_source_conflict", "", "unresolved_source_conflict"
    if status == "approved":
        body = canonical_body(split_labels(row["modeling_case_body"]))
        finish = str(row["modeling_case_finish"]).strip()
        if finish in {"", "none"}:
            finish = "not_stated"
        return body, "approved_correction", finish, "approved_correction"
    structured = structured_material_tokens(row["case_material"])
    extracted_base = split_labels(row["extracted_case_base_materials"])
    extracted_finish = split_labels(row["extracted_case_finish"])
    if materials_conflict(structured, extracted_base, extracted_finish):
        return "", "source_disagreement", "", "source_disagreement"
    decision = assess_case_compatibility(
        sorted(structured),
        extracted_base,
        str(row.get("item_title", "")),
        str(row.get("item_description", "")),
        FINISH_LABELS,
    )
    if decision.applied:
        finish, finish_source = choose_finish(structured, extracted_finish, bool(decision.body))
        return decision.body, decision.source if decision.body else "source_disagreement", finish, finish_source
    body, body_source = agree_body(sorted(structured), extracted_base)
    raw_material = str(row["case_material"]).strip()
    if not body and raw_material and not structured and not extracted_base:
        body_source = "unsupported_structured_material"
    finish, finish_source = choose_finish(structured, extracted_finish, bool(body))
    return body, body_source, finish, finish_source

def choose_movement_feature(row: dict) -> tuple[str, str]:
    status = row["movement_review_status"]
    if status == "unresolved_source_conflict":
        return "", "unresolved_source_conflict"
    if status == "approved":
        return str(row["modeling_movement"]).strip(), "approved_correction"
    structured = structured_movement_family(row["movement"])
    extracted = str(row["extracted_movement"]).strip()
    if (
        structured.startswith("conflict:")
        or extracted.startswith("conflict:")
        or movements_conflict(structured, extracted)
    ):
        return "", "source_disagreement"
    chosen = choose_movement(structured, extracted)
    if not chosen:
        return "", "missing"
    if chosen.startswith("conflict:"):
        return "", "source_disagreement"
    if structured and extracted:
        source = "structured_and_extracted"
    elif structured:
        source = "structured_movement"
    else:
        source = "extracted_movement"
    return chosen, source


def choose_age(row: dict) -> tuple[float | None, str]:
    parsed = str(row["production_year_parsed"]).strip()
    sale_year = str(row["sale_year"]).strip()
    if not parsed:
        return None, ""
    approximate = "true" if str(row["production_year_approximate"]).strip() == "true" else "false"
    if not sale_year or int(parsed) > int(sale_year):
        return None, approximate
    return float(int(sale_year) - int(parsed)), approximate


def optional_float(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    return float(text)

def _missing(value: object) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def construct_record(row: Mapping[str, object]) -> dict[str, object]:
    """Return one row of audited predictors. Does not read or return price."""
    record = dict(row)
    body, body_source, finish, finish_source = choose_case_features(record)
    movement, movement_source = choose_movement_feature(record)
    age, approximate = choose_age(record)
    parsed = str(record["production_year_parsed"]).strip()
    return {
        "record_id": record["record_id"],
        "norm_model": _missing(record["norm_model"]),
        "production_year": int(parsed) if parsed else None,
        "age_at_sale": age,
        "production_year_approximate": _missing(approximate),
        "case_body": _missing(body),
        "case_body_source": body_source,
        "case_finish": _missing(finish),
        "case_finish_source": finish_source,
        "movement": _missing(movement),
        "movement_source": movement_source,
        "case_diameter_mm": optional_float(record["case_measurement_mm"]),
        "case_width_mm": optional_float(record["case_width_mm"]),
        "case_length_mm": optional_float(record["case_length_mm"]),
    }


PREDICTOR_FIELDS = (
    "norm_model",
    "production_year",
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
