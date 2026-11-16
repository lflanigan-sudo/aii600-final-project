"""Build a local feature-candidate CSV for team review.

Reads the approved eligible file and the cleaned candidate. Writes a new CSV
beside them. Does not modify either input, the predictor module, or model
results, and does not fit or score a model.

Run from the repository root:

    python scripts/build_feature_candidates.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predictors import (  # noqa: E402
    STRUCTURED_MATERIALS,
    case_body_mentions,
    construct_record,
)

APPROVED_SHA256 = "cf3b41d890af1feba9348fa9a73d954fcab193ed890a39044bd0355198fc5a8a"
CLEANED_SHA256 = "74bf8efd5279afc4b59013618063c78292c166c491b922bdaf3db6e347ac9824"
EXPECTED_ROWS = 4664

ACCENT_MAP = str.maketrans(
    {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "á": "a",
        "â": "a",
        "ä": "a",
        "ì": "i",
        "í": "i",
        "ï": "i",
        "î": "i",
        "ò": "o",
        "ó": "o",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "ú": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "’": "'",
        "‘": "'",
        "–": "-",
        "—": "-",
    }
)

YEAR_NUMBER = r"(?:18[5-9]\d|19\d\d|20[0-2]\d)"
CIRCA_YEAR_RE = re.compile(
    rf"(?i)\b(?:circa|c\.|ca\.|vers|environ)\s*({YEAR_NUMBER})(s)?\b"
)
EXPLICIT_YEAR_RE = re.compile(
    rf"(?i)\b(?:dated|manufactured|produced|made|production(?:\s+year)?|année)\s+"
    rf"(?:in\s+)?({YEAR_NUMBER})\b"
)
YEAR_CONTEXT_RE = re.compile(
    r"(?i)\b(?:ref(?:erence)?|no\.|number|serial|purchased|papers|paper|box|"
    r"certificate|document|guarantee|extract|booklet|warranty|sold|sale|"
    r"edition|service)\b"
)
NON_WATCH_TITLE_RE = re.compile(
    r"(?i)travel alarm|boudoir|desk clock|carriage clock|century of\b|first edition"
)
WATCH_TITLE_RE = re.compile(
    r"(?i)\b(?:wristwatch(?:es)?|watch(?:es)?|montres?|chronograph(?:s)?|chronographes?)\b"
)
RETAILED_BY_RE = re.compile(
    r"(?i)\b(?:retailed|issued|distributed)\s+by\s+(?:rolex|cartier|omega)\b"
)
BRAND_RE = re.compile(r"(?i)\b(rolex|cartier|omega)\b")
QUOTE_RE = re.compile(r"['‘’\"]([A-Za-z][A-Za-z][A-Za-z .'\-]{1,40})['‘’\"]")
QUOTE_STOP = {
    "fine",
    "rare",
    "lady",
    "ladies",
    "gold",
    "diamond",
    "manual",
    "the",
    "new",
    "old",
    "stock",
    "very",
    "good",
    "early",
    "steel",
    "silver",
    "platinum",
}

# Catalogue model lines only. Collector nicknames and bare reference numbers
# are intentionally absent.
MODEL_STEMS = {
    "Rolex": [
        "oyster perpetual datejust",
        "oyster perpetual day-date",
        "oyster perpetual explorer",
        "oysterquartz datejust",
        "oysterquartz day-date",
        "lady-datejust",
        "lady datejust",
        "gmt-master ii",
        "gmt master ii",
        "gmt-master",
        "gmt master",
        "yacht-master ii",
        "yacht-master",
        "sky-dweller",
        "sea-dweller",
        "submariner",
        "cosmograph daytona",
        "daytona",
        "cosmograph",
        "datejust",
        "day-date",
        "day date",
        "oyster perpetual",
        "explorer ii",
        "explorer",
        "milgauss",
        "air-king",
        "air king",
        "turn-o-graph",
        "pearlmaster",
        "cellini",
        "deepsea",
    ],
    "Cartier": [
        "must de cartier",
        "tank a vis",
        "tank americaine",
        "tank américaine",
        "tank francaise",
        "tank française",
        "tank louis cartier",
        "tank louis",
        "tank anglaise",
        "tank basculante",
        "tank chinoise",
        "tank cintree",
        "tank cintrée",
        "tank allongee",
        "tank allongée",
        "tank mc",
        "tank solo",
        "tank normale",
        "tank divan",
        "tank must",
        "tank mini",
        "tank xl",
        "tank",
        "ballon bleu",
        "santos dumont",
        "santos-dumont",
        "santos demoiselle",
        "santos galbee",
        "santos galbée",
        "santos octogonale",
        "santos octagonale",
        "santos 100",
        "santos carree",
        "santos carrée",
        "santos",
        "panthere vendome",
        "panthère vendôme",
        "panthere de cartier",
        "panthere",
        "panthère",
        "pasha",
        "roadster",
        "baignoire",
        "tortue",
        "crash",
        "tonneau",
        "cloche",
        "cadenas",
        "ceinture",
        "diabolo",
        "colisee vendome",
        "colisée vendôme",
        "colisee",
        "colisée",
        "tankissime",
        "vendome",
        "vendôme",
        "ronde solo",
        "ronde louis",
        "drive de cartier",
        "calibre de cartier",
        "cle de cartier",
        "clé de cartier",
        "bamboo",
    ],
    "Omega": [
        "speedmaster professional",
        "speedmaster",
        "seamaster aqua terra",
        "seamaster planet ocean",
        "seamaster de ville",
        "seamaster",
        "constellation",
        "de ville",
        "deville",
        "railmaster",
        "planet ocean",
        "aqua terra",
        "dynamic",
        "geneve",
        "genève",
        "speedsonic",
        "flightmaster",
        "ladymatic",
    ],
}
GENERIC_MODEL_LINES = {"must de cartier"}
SUFFIX_MODEL_LINES = {"vendome", "vendôme"}

MOVEMENT_ALLOWED = {
    "quartz": "quartz",
    "automatic": "automatic",
    "manual": "manual winding",
}
SELF_WIND_RE = re.compile(r"(?i)\bself[\s-]?wind(?:ing|er)?\b")
AUTO_RE = re.compile(r"(?i)\b(?:automatic|automatique)\b")
MANUAL_RE = re.compile(
    r"(?i)\b(?:manual(?:ly)?\s+wound|manual\s+wind(?:ing)?|hand[\s-]?wound|"
    r"hand[\s-]?wind(?:ing)?|remontage\s+manuel)\b"
)
QUARTZ_RE = re.compile(r"(?i)\bquartz\b")
QUARTZ_DIAL_RE = re.compile(r"(?i)\bquartz\s+dial\b")
MECH_RE = re.compile(r"(?i)\b(?:mechanical|mecanique|mécanique)\b")

STONE_RE = re.compile(
    r"(?i)\b(diamonds?|rubies|ruby|sapphires?|emeralds?|cultured\s+pearls?|pearls?)\b"
)
DECORATIVE_DIAL_RE = re.compile(
    r"(?i)\b(?:mother[\s-]of[\s-]pearl|onyx|lapis|jade|coral|turquoise|chalcedony|"
    r"agate|hardstone|meteorite|malachite|sodalite|aventurine|tiger'?s\s+eye)\s+dial\b"
)
JEWEL_COUNT_RE = re.compile(r"(?i)\b\d{1,3}\s*-?\s*jewels?\b|\bjewell?ed\b")
STONE_ABSENCE_RE = re.compile(
    r"(?i)\b(?:no|without|not)\s+(?:diamonds?|gemstones?|gem-?set|precious\s+stones|"
    r"stone\s+setting|gems)\b|\bungemmed\b"
)
COMPONENT_RE = re.compile(
    r"(?i)\b(bezel|lunette|bracelet|strap|lugs?|crown|case|hour\s+markers?|"
    r"indices|indexes?|dial)\b"
)
GEM_FIELD_WORDS = {
    "diamond": "diamond",
    "ruby": "ruby",
    "sapphire": "sapphire",
    "emerald": "emerald",
    "pearl": "pearl",
}

BRACELET_WATCH_RE = re.compile(
    r"(?i)\b(?:montres?|chronographes?|chronographs?)[\s\-]+bracelets?\b|"
    r"\bbracelets?[\s\-/]+watch(?:es)?\b"
)
STRAP_MATERIAL_RE = re.compile(
    r"(?i)\b(?:leather|alligator|crocodile|lizard|ostrich|rubber|caoutchouc|"
    r"satin|textile|fabric|nato|calfskin|calf)\b"
)
STRAP_WORD_RE = re.compile(r"(?i)\bstraps?\b")
BRACELET_WORD_RE = re.compile(r"(?i)\bbracelets?\b")
METAL_RE = re.compile(
    r"(?i)\b(?:stainless\s+steel|yellow\s+gold|white\s+gold|rose\s+gold|"
    r"pink\s+gold|red\s+gold|platinum|titanium|steel|gold)\b"
)
ASSOCIATED_RE = re.compile(r"(?i)\b(?:associated|later|replacement|aftermarket)\b")
STRAP_FIELD_VALUES = {
    "leather",
    "alligator",
    "rubber",
    "fabric",
    "lizard",
    "crocodile",
    "satin",
    "textile",
    "nato",
}

CASE_GROUP = {
    "steel": "steel",
    "generic-gold": "gold",
    "karat-gold": "gold",
    "yellow-gold": "gold",
    "white-gold": "gold",
    "rose-gold": "gold",
    "pink-gold": "gold",
    "red-gold": "gold",
    "gold-plated": "plated",
    "three-colour-gold": "tricolor-gold",
    "bi-color-gold": "bicolor-gold",
    "platinum": "platinum",
    "silver": "silver",
    "titanium": "titanium",
}
KARAT_RE = re.compile(r"(?i)\b(9|14|18|22|24)\s?k\b")
COLOR_RE = re.compile(r"(?i)\b(yellow|white|rose|pink|red)\b")

NEW_COLUMNS = [
    "brand_candidate",
    "brand_candidate_source",
    "brand_candidate_evidence",
    "brand_candidate_review",
    "model_candidate",
    "model_candidate_source",
    "model_candidate_evidence",
    "model_candidate_review",
    "production_year_candidate",
    "production_year_candidate_approximate",
    "production_year_candidate_source",
    "production_year_candidate_evidence",
    "production_year_candidate_review",
    "case_material_candidate",
    "case_material_candidate_karat",
    "case_material_candidate_color",
    "case_material_candidate_construction",
    "case_material_candidate_source",
    "case_material_candidate_evidence",
    "case_material_candidate_review",
    "movement_type_candidate",
    "movement_type_candidate_source",
    "movement_type_candidate_evidence",
    "movement_type_candidate_review",
    "precious_stones_candidate",
    "precious_stones_type_candidate",
    "precious_stones_component_candidate",
    "precious_stones_candidate_source",
    "precious_stones_candidate_evidence",
    "precious_stones_candidate_review",
    "attachment_type_candidate",
    "attachment_type_candidate_source",
    "attachment_type_candidate_evidence",
    "attachment_type_candidate_review",
    "bracelet_material_candidate",
    "bracelet_material_candidate_source",
    "bracelet_material_candidate_evidence",
    "bracelet_material_candidate_review",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def normalize(text: str) -> str:
    return text.lower().translate(ACCENT_MAP)


def excerpt(text: str, start: int = 0, end: int | None = None, limit: int = 160) -> str:
    if end is None:
        snippet = text
    else:
        snippet = text[max(0, start): min(len(text), end)]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) > limit:
        return snippet[: limit - 3] + "..."
    return snippet


def stem_pattern(stem: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in re.split(r"[\s\-]+", normalize(stem)) if part]
    body = r"[\s\-]*".join(parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


COMPILED_STEMS = {
    brand: [(stem, stem_pattern(stem)) for stem in stems]
    for brand, stems in MODEL_STEMS.items()
}


def model_family(stem: str) -> str:
    token = normalize(stem).split()[0]
    if token in {"gmt", "gmt-master"}:
        return "gmt-master"
    if token in {"sea", "sea-dweller"}:
        return "sea-dweller"
    if token in {"sky", "sky-dweller"}:
        return "sky-dweller"
    if token in {"yacht", "yacht-master"}:
        return "yacht-master"
    if token in {"day", "day-date"}:
        return "day-date"
    if token in {"oyster", "oysterquartz"}:
        return "oyster-perpetual"
    if token in {"air", "air-king"}:
        return "air-king"
    if token in {"lady", "lady-datejust"}:
        return "datejust"
    if token == "cosmograph":
        return "daytona"
    if token in {"de", "deville"}:
        return "de-ville"
    if token in {"panthere", "panthère"}:
        return "panthere"
    return token


def find_models(title: str, brand: str) -> tuple[list[tuple[int, int, str]], str]:
    folded = normalize(title)
    found: list[tuple[int, int, str]] = []
    for stem, pattern in COMPILED_STEMS.get(brand, []):
        for match in pattern.finditer(folded):
            found.append((match.start(), match.end(), stem))
    found.sort(key=lambda item: (item[1] - item[0], -item[0]), reverse=True)
    kept: list[tuple[int, int, str]] = []
    for start, end, stem in found:
        if any(start >= kept_start and end <= kept_end for kept_start, kept_end, _ in kept):
            continue
        if any(not (end <= kept_start or start >= kept_end) for kept_start, kept_end, _ in kept):
            continue
        kept.append((start, end, stem))
    unique: list[tuple[int, int, str]] = []
    seen_stems: set[str] = set()
    for start, end, stem in kept:
        key = normalize(stem)
        if key in seen_stems:
            continue
        seen_stems.add(key)
        unique.append((start, end, stem))
    kept = unique
    if len(kept) > 1:
        specific = [item for item in kept if normalize(item[2]) not in GENERIC_MODEL_LINES]
        if specific:
            kept = specific
    if len(kept) > 1:
        primary = [item for item in kept if normalize(item[2]) not in SUFFIX_MODEL_LINES]
        if primary:
            kept = primary
    if len(kept) > 1:
        return [], "multiple_models"
    return kept, ""


def quoted_unlisted(title: str, brand: str) -> list[str]:
    names = []
    matched, _ = find_models(title, brand)
    matched_spans = {(start, end) for start, end, _ in matched}
    for match in QUOTE_RE.finditer(title):
        inside = re.sub(r"\s+", " ", match.group(1)).strip(" '")
        if normalize(inside) in QUOTE_STOP or len(normalize(inside)) < 4:
            continue
        if any(start <= match.start(1) and match.end(1) <= end for start, end in matched_spans):
            continue
        if find_models(inside, brand)[0]:
            continue
        names.append(inside)
    return names


def production_year_from_title(title: str) -> tuple[str, str, str, str]:
    """Return year, approximate flag, evidence, review reason."""
    if NON_WATCH_TITLE_RE.search(title):
        return "", "", "", ""
    years: list[tuple[str, bool, str]] = []
    decades = False
    for pattern, forced_approx, use_context in (
        (CIRCA_YEAR_RE, True, False),
        (EXPLICIT_YEAR_RE, False, True),
    ):
        for match in pattern.finditer(title):
            window = title[max(0, match.start() - 48): match.start()]
            if use_context and YEAR_CONTEXT_RE.search(window):
                continue
            year = match.group(1)
            is_decade = bool(match.lastindex and match.lastindex >= 2 and match.group(2))
            if is_decade:
                decades = True
                continue
            years.append((year, forced_approx or bool(re.search(r"(?i)circa|c\.|ca\.|vers|environ", match.group(0))), excerpt(title, match.start(), match.end())))
    unique = []
    for item in years:
        if item[0] not in {year for year, _, _ in unique}:
            unique.append(item)
    if len(unique) > 1:
        return "", "", unique[0][2], "multiple_production_years"
    if decades and not unique:
        return "", "", "", "decade_not_a_year"
    if not unique:
        return "", "", "", ""
    year, approximate, evidence = unique[0]
    return year, "true" if approximate else "false", evidence, ""


def body_groups_from_value(value: str) -> set[str]:
    groups = set()
    for part in value.split("|"):
        part = part.strip()
        if part == "steel-and-gold":
            groups.update({"steel", "gold"})
        elif part == "titanium-and-gold":
            groups.update({"titanium", "gold"})
        elif part:
            groups.add(CASE_GROUP.get(part, part))
    return groups


def phrase_for_case(labels: list[str], snippets: list[str]) -> tuple[str, str, str, str]:
    text = " ".join(snippets)
    karats = []
    for match in KARAT_RE.finditer(text):
        token = match.group(1).lower() + "k"
        if token not in karats:
            karats.append(token)
    karat = karats[0] if len(karats) == 1 else ""
    karat_review = "multiple_karat" if len(karats) > 1 else ""
    groups = []
    for label in labels:
        group = CASE_GROUP.get(label, label)
        if group not in groups:
            groups.append(group)
    base = [group for group in groups if group != "plated"]
    plated = "plated" in groups
    construction = ""
    lowered = text.lower()
    if "vermeil" in lowered:
        construction = "vermeil"
    elif "gold-filled" in lowered or "gold filled" in lowered:
        construction = "gold-filled"
    elif "gilt" in lowered:
        construction = "gilt"
    elif plated or "plated" in lowered:
        construction = "plated"
    elif re.search(r"\bsolid\b", lowered):
        construction = "solid"
    color_match = COLOR_RE.search(text)
    color = color_match.group(1).lower() if color_match and "gold" in " ".join(groups) else ""
    if plated and len(base) == 1:
        metal = "stainless steel" if base[0] == "steel" and "stainless" in lowered else base[0].replace("-", " ")
        if base[0] == "gold":
            metal = " ".join(part for part in (karat, color, "gold") if part)
        phrase = f"{construction} {metal}".strip() if construction else metal
        return phrase, karat, color, ";".join(item for item in (construction, karat_review) if item).replace("plated;", "")
    if plated and not base:
        if construction == "vermeil":
            phrase = "vermeil"
        elif construction == "gilt":
            phrase = "gilt"
        elif construction == "gold-filled":
            phrase = "gold-filled"
        else:
            phrase = "gold-plated"
        return phrase, karat, color, karat_review
    if len(base) != 1:
        return "", karat, color, "multiple_case_materials"
    metal_name = base[0]
    if metal_name == "steel":
        phrase = "stainless steel" if "stainless" in lowered else "steel"
        return phrase, "", "", karat_review
    if metal_name == "gold":
        phrase = " ".join(part for part in (karat, color, "gold") if part)
        return phrase, karat, color, karat_review
    if metal_name == "tricolor-gold":
        return "three-colour gold", karat, "", karat_review
    if metal_name == "bicolor-gold":
        return "bi-color gold", karat, "", karat_review
    return metal_name.replace("-", " "), karat, color, karat_review


def case_from_mentions(mentions: list[tuple[str, str, str]]) -> tuple[str, str, str, str, str, str]:
    if not mentions:
        return "", "", "", "", "", ""
    labels = []
    snippets = []
    for label, _section, snippet in mentions:
        if label not in labels:
            labels.append(label)
        if snippet not in snippets:
            snippets.append(snippet)
    phrase, karat, color, review = phrase_for_case(labels, snippets)
    evidence = excerpt(" | ".join(snippets))
    construction = ""
    for token in ("vermeil", "gold-filled", "gilt", "plated", "solid"):
        if token in review or token in phrase:
            construction = token
            break
    if review in {"multiple_case_materials", "multiple_karat"} and not phrase:
        return "", karat, color, "", evidence, review
    extra = ""
    if "multiple_karat" in review:
        extra = "multiple_karat"
        review = ""
    return phrase, karat, color, construction, evidence, extra or review


def propose_case(row: dict[str, str], predictor: dict[str, object]) -> dict[str, str]:
    result = {
        "case_material_candidate": "",
        "case_material_candidate_karat": "",
        "case_material_candidate_color": "",
        "case_material_candidate_construction": "",
        "case_material_candidate_source": "",
        "case_material_candidate_evidence": "",
        "case_material_candidate_review": "",
    }
    status = row["material_review_status"].strip()
    source = str(predictor["case_body_source"] or "")
    existing = predictor["case_body"]
    protected = []
    if status == "unresolved_source_conflict" or source == "unresolved_source_conflict":
        protected.append("unresolved_source_conflict")
    elif source == "source_disagreement":
        protected.append("source_disagreement")
    elif status == "approved" and existing is None:
        protected.append("approved_blank")
    title = row["item_title"]
    title_is_watch = WATCH_TITLE_RE.search(title)
    title_mentions = [] if NON_WATCH_TITLE_RE.search(title) or not title_is_watch else case_body_mentions(title, "")
    description_mentions = [
        item for item in case_body_mentions("", row["item_description"]) if item[1] == "case"
    ]
    chosen_source = ""
    mentions = []
    if title_mentions:
        chosen_source = "item_title"
        mentions = title_mentions
    elif description_mentions:
        chosen_source = "item_description"
        mentions = description_mentions
    elif not protected and existing is None:
        structured = STRUCTURED_MATERIALS.get(row["case_material"].strip().casefold(), set())
        if len(structured) == 1:
            label = next(iter(structured))
            mentions = [(label, "case_material", row["case_material"].strip())]
            chosen_source = "case_material"
    phrase, karat, color, construction, evidence, review = case_from_mentions(mentions)
    proposal_groups = set()
    if phrase:
        if "steel" in phrase and "gold" in phrase:
            proposal_groups.update({"steel", "gold"})
        elif "steel" in phrase:
            proposal_groups.add("steel")
        elif "gold" in phrase or construction in {"plated", "vermeil", "gilt", "gold-filled"}:
            proposal_groups.add("plated" if construction in {"plated", "vermeil", "gilt", "gold-filled"} and "steel" not in phrase and not karat and not color else "gold")
        elif "platinum" in phrase:
            proposal_groups.add("platinum")
        elif "silver" in phrase:
            proposal_groups.add("silver")
        elif "titanium" in phrase:
            proposal_groups.add("titanium")
    if existing is not None and proposal_groups:
        existing_groups = body_groups_from_value(str(existing))
        solid_existing = existing_groups - {"plated"}
        solid_proposal = proposal_groups - {"plated"}
        if solid_existing and solid_proposal and solid_existing.isdisjoint(solid_proposal):
            result["case_material_candidate_review"] = "conflict_with_existing_case"
            result["case_material_candidate_evidence"] = evidence
            return result
        if "plated" in proposal_groups and "gold" in existing_groups and "steel" not in phrase:
            result["case_material_candidate_review"] = "conflict_plated_versus_solid"
            result["case_material_candidate_evidence"] = evidence
            return result
    if protected:
        result["case_material_candidate_review"] = ";".join(protected)
        return result
    if existing is not None:
        return result
    if not phrase:
        if review:
            result["case_material_candidate_review"] = review
            result["case_material_candidate_evidence"] = evidence
        return result
    result.update(
        {
            "case_material_candidate": phrase,
            "case_material_candidate_karat": karat,
            "case_material_candidate_color": color,
            "case_material_candidate_construction": construction,
            "case_material_candidate_source": chosen_source,
            "case_material_candidate_evidence": evidence,
            "case_material_candidate_review": review,
        }
    )
    return result


def movement_from_text(title: str, description: str) -> tuple[str, str, str, str]:
    chunks = [("item_title", title), ("item_description", description)]
    found: list[tuple[str, str, str]] = []
    for source, text in chunks:
        if not text:
            continue
        cleaned = JEWEL_COUNT_RE.sub(" ", text)
        if QUARTZ_DIAL_RE.search(cleaned) and len(QUARTZ_RE.findall(cleaned)) == QUARTZ_DIAL_RE.findall(cleaned).__len__():
            quartz = False
        else:
            quartz = QUARTZ_RE.search(cleaned) is not None and QUARTZ_DIAL_RE.search(cleaned) is None
        automatic = AUTO_RE.search(cleaned) is not None or SELF_WIND_RE.search(cleaned) is not None
        manual = MANUAL_RE.search(cleaned) is not None
        mechanical = MECH_RE.search(cleaned) is not None
        label = ""
        reason = ""
        if quartz and (automatic or manual or mechanical):
            reason = "movement_conflict"
        elif automatic and manual:
            label = "automatic"
        elif quartz:
            label = "quartz"
        elif automatic:
            label = "automatic"
        elif manual:
            label = "manual winding"
        elif mechanical:
            reason = "mechanical_only"
        if label or reason:
            match = None
            for pattern in (SELF_WIND_RE, AUTO_RE, MANUAL_RE, QUARTZ_RE, MECH_RE):
                match = pattern.search(cleaned)
                if match:
                    break
            evidence = excerpt(cleaned, match.start() - 30, match.end() + 30) if match else ""
            found.append((label, source, reason or evidence))
            if reason:
                found[-1] = ("", source, reason)
            else:
                found[-1] = (label, source, evidence)
    labels = {item[0] for item in found if item[0]}
    reasons = {item[2] for item in found if not item[0]}
    if "movement_conflict" in reasons or len(labels) > 1:
        return "", "", "", "movement_conflict"
    if len(labels) == 1:
        label = next(iter(labels))
        source, evidence = next((item[1], item[2]) for item in found if item[0] == label)
        return label, source, evidence, ""
    if "mechanical_only" in reasons:
        return "", "", "", "mechanical_only"
    return "", "", "", ""


def stone_name(token: str) -> str:
    lowered = token.lower()
    if "diamond" in lowered:
        return "diamond"
    if "rub" in lowered:
        return "ruby"
    if "sapphire" in lowered:
        return "sapphire"
    if "emerald" in lowered:
        return "emerald"
    if "pearl" in lowered:
        return "pearl"
    return lowered


def precious_stones(row: dict[str, str]) -> dict[str, str]:
    text = f"{row['item_title']} {row['item_description']}"
    searchable = JEWEL_COUNT_RE.sub(" ", text)
    types: list[str] = []
    components: list[str] = []
    evidence = ""
    for match in STONE_RE.finditer(searchable):
        name = stone_name(match.group(1))
        window_start = max(0, match.start() - 40)
        window = searchable[window_start: match.end() + 40]
        if name == "pearl" and re.search(r"(?i)mother[\s-]of[\s-]pearl", window):
            continue
        if DECORATIVE_DIAL_RE.search(window) and name not in {"diamond", "ruby", "sapphire", "emerald"}:
            continue
        if name not in types:
            types.append(name)
        nearest_component = ""
        nearest_distance = None
        stone_at = match.start() - window_start
        for component_match in COMPONENT_RE.finditer(window):
            component = component_match.group(1).lower()
            following = window[component_match.end(): component_match.end() + 12]
            if component == "case" and re.match(r"(?i)\s*(?:no\.?|number|#)\b", following):
                continue
            distance = abs(component_match.start() - stone_at)
            if nearest_distance is None or distance < nearest_distance:
                nearest_component = component
                nearest_distance = distance
        if nearest_component and nearest_component not in components:
            components.append(nearest_component)
        if not evidence:
            evidence = excerpt(searchable, match.start() - 35, match.end() + 35)
    field_text = " ".join(part for part in (row["gem_type"], row["primary_gem"]) if part)
    for token, name in GEM_FIELD_WORDS.items():
        if re.search(rf"(?i)\b{token}s?\b", field_text) and name not in types:
            if name == "pearl" and re.search(r"(?i)mother[\s-]of[\s-]pearl", text):
                continue
            types.append(name)
            if not evidence:
                evidence = excerpt(field_text)
    absence = STONE_ABSENCE_RE.search(searchable)
    review = ""
    if types and absence:
        review = "stone_presence_conflict"
        value = "unknown"
    elif types:
        value = "yes"
    elif absence:
        value = "no"
        evidence = excerpt(searchable, absence.start() - 20, absence.end() + 20)
    else:
        value = "unknown"
    source = ""
    if value == "yes" and field_text and not STONE_RE.search(searchable):
        source = "gem_type"
    elif value in {"yes", "no"}:
        source = "item_title" if STONE_RE.search(row["item_title"]) or STONE_ABSENCE_RE.search(row["item_title"]) else "item_description"
        if value == "yes" and field_text:
            source = source + "+gem_type"
    return {
        "precious_stones_candidate": value,
        "precious_stones_type_candidate": ";".join(types),
        "precious_stones_component_candidate": ";".join(components),
        "precious_stones_candidate_source": source,
        "precious_stones_candidate_evidence": evidence if value != "unknown" or review else "",
        "precious_stones_candidate_review": review,
    }


def strip_bracelet_watch(text: str) -> str:
    return BRACELET_WATCH_RE.sub(" watch ", text)


def attachment_signals(text: str) -> tuple[set[str], list[str], bool]:
    cleaned = strip_bracelet_watch(text)
    kinds = set()
    materials = []
    if STRAP_WORD_RE.search(cleaned) or STRAP_MATERIAL_RE.search(cleaned):
        kinds.add("strap")
    for match in BRACELET_WORD_RE.finditer(cleaned):
        kinds.add("bracelet")
        window = cleaned[max(0, match.start() - 40): match.end() + 20]
        metal = METAL_RE.search(window)
        if metal:
            material = re.sub(r"\s+", " ", metal.group(0)).strip().lower()
            if material not in materials:
                materials.append(material)
    associated = ASSOCIATED_RE.search(cleaned) is not None and bool(kinds)
    return kinds, materials, associated


def attachment_fields(row: dict[str, str]) -> dict[str, str]:
    sources = []
    kinds: set[str] = set()
    materials: list[str] = []
    associated = False
    evidence = ""
    evidence_source = ""
    field_value = row["bracelet_material"].strip()
    if field_value.casefold() in STRAP_FIELD_VALUES:
        kinds.add("strap")
        sources.append("bracelet_material")
        evidence = field_value
        evidence_source = "bracelet_material"
    elif field_value and METAL_RE.fullmatch(field_value):
        kinds.add("bracelet")
        material = field_value.lower()
        if material not in materials:
            materials.append(material)
        sources.append("bracelet_material")
        evidence = field_value
        evidence_source = "bracelet_material"
    for source, text in (
        ("strap_bracelet", row["strap_bracelet"]),
        ("item_title", row["item_title"]),
        ("item_description", row["item_description"]),
    ):
        if not text:
            continue
        found_kinds, found_materials, found_associated = attachment_signals(text)
        if not found_kinds:
            continue
        kinds.update(found_kinds)
        for material in found_materials:
            if material not in materials:
                materials.append(material)
        associated = associated or found_associated
        sources.append(source)
        if not evidence:
            evidence = excerpt(strip_bracelet_watch(text))
            evidence_source = source
    review = []
    if associated:
        review.append("associated_attachment")
    if kinds == {"strap", "bracelet"}:
        return {
            "attachment_type_candidate": "unknown",
            "attachment_type_candidate_source": "+".join(sources),
            "attachment_type_candidate_evidence": evidence,
            "attachment_type_candidate_review": ";".join(review + ["both_strap_and_bracelet"]),
            "bracelet_material_candidate": "unknown",
            "bracelet_material_candidate_source": "",
            "bracelet_material_candidate_evidence": "",
            "bracelet_material_candidate_review": "both_strap_and_bracelet",
        }
    if kinds == {"strap"}:
        return {
            "attachment_type_candidate": "strap",
            "attachment_type_candidate_source": evidence_source,
            "attachment_type_candidate_evidence": evidence,
            "attachment_type_candidate_review": ";".join(review),
            "bracelet_material_candidate": "not_applicable",
            "bracelet_material_candidate_source": evidence_source,
            "bracelet_material_candidate_evidence": evidence,
            "bracelet_material_candidate_review": "",
        }
    if kinds == {"bracelet"}:
        material = materials[0] if len(materials) == 1 else ""
        material_review = ""
        if len(materials) > 1:
            material = ""
            material_review = "multiple_bracelet_materials"
        return {
            "attachment_type_candidate": "bracelet",
            "attachment_type_candidate_source": evidence_source,
            "attachment_type_candidate_evidence": evidence,
            "attachment_type_candidate_review": ";".join(review),
            "bracelet_material_candidate": material or "unknown",
            "bracelet_material_candidate_source": evidence_source if material else "",
            "bracelet_material_candidate_evidence": evidence if material else "",
            "bracelet_material_candidate_review": material_review,
        }
    return {
        "attachment_type_candidate": "unknown",
        "attachment_type_candidate_source": "",
        "attachment_type_candidate_evidence": "",
        "attachment_type_candidate_review": "",
        "bracelet_material_candidate": "unknown",
        "bracelet_material_candidate_source": "",
        "bracelet_material_candidate_evidence": "",
        "bracelet_material_candidate_review": "",
    }


def propose_brand(row: dict[str, str]) -> dict[str, str]:
    title = RETAILED_BY_RE.sub(" ", row["item_title"])
    found = []
    for match in BRAND_RE.finditer(title):
        name = match.group(1).lower()
        label = {"rolex": "Rolex", "cartier": "Cartier", "omega": "Omega"}[name]
        if label not in found:
            found.append(label)
    review = ""
    evidence = ""
    if any(label != row["designer"] for label in found):
        review = "conflict_with_existing_brand"
        evidence = excerpt(row["item_title"])
    # Designer is complete on the approved file. Never overwrite it.
    return {
        "brand_candidate": "",
        "brand_candidate_source": "",
        "brand_candidate_evidence": evidence,
        "brand_candidate_review": review,
    }


def propose_model(row: dict[str, str]) -> dict[str, str]:
    existing = row["norm_model"].strip()
    matches, reason = find_models(row["item_title"], row["designer"])
    quoted = quoted_unlisted(row["item_title"], row["designer"])
    result = {
        "model_candidate": "",
        "model_candidate_source": "",
        "model_candidate_evidence": "",
        "model_candidate_review": "",
    }
    reviews = []
    if reason:
        reviews.append(reason)
    if quoted:
        reviews.append("quoted_unlisted_model")
    if matches:
        start, end, stem = matches[0]
        evidence = excerpt(row["item_title"], max(0, start - 30), min(len(row["item_title"]), end + 30))
        proposed = re.sub(r"\s+", " ", row["item_title"][start:end]).strip()
        existing_matches, _ = find_models(existing, row["designer"]) if existing else ([], "")
        if existing:
            if existing_matches and model_family(existing_matches[0][2]) != model_family(stem):
                reviews.append("conflict_with_existing_model")
                result["model_candidate_evidence"] = evidence
            result["model_candidate_review"] = ";".join(reviews)
            return result
        result.update(
            {
                "model_candidate": proposed,
                "model_candidate_source": "item_title",
                "model_candidate_evidence": evidence,
            }
        )
    result["model_candidate_review"] = ";".join(reviews)
    return result


def propose_year(row: dict[str, str]) -> dict[str, str]:
    year, approximate, evidence, review = production_year_from_title(row["item_title"])
    parsed = row["production_year_parsed"].strip()
    result = {
        "production_year_candidate": "",
        "production_year_candidate_approximate": "",
        "production_year_candidate_source": "",
        "production_year_candidate_evidence": "",
        "production_year_candidate_review": review,
    }
    if parsed and year and parsed != year:
        result["production_year_candidate_review"] = ";".join(
            part for part in (review, "conflict_with_existing_year") if part
        )
        result["production_year_candidate_evidence"] = evidence
        return result
    if parsed or row["production_year_correction_status"].strip() == "approved_approximate":
        return result
    if year:
        result.update(
            {
                "production_year_candidate": year,
                "production_year_candidate_approximate": approximate,
                "production_year_candidate_source": "item_title",
                "production_year_candidate_evidence": evidence,
            }
        )
    return result


def propose_movement(row: dict[str, str], predictor: dict[str, object]) -> dict[str, str]:
    result = {
        "movement_type_candidate": "",
        "movement_type_candidate_source": "",
        "movement_type_candidate_evidence": "",
        "movement_type_candidate_review": "",
    }
    status = row["movement_review_status"].strip()
    source = str(predictor["movement_source"] or "")
    movement = predictor["movement"]
    if status == "unresolved_source_conflict" or source in {"unresolved_source_conflict", "source_disagreement"}:
        result["movement_type_candidate_review"] = source or status
        return result
    if status == "approved" and movement is None:
        result["movement_type_candidate_review"] = "approved_blank"
        return result
    if movement in MOVEMENT_ALLOWED:
        evidence_field = row["movement"].strip() or row["extracted_movement"].strip() or row["modeling_movement"].strip()
        result.update(
            {
                "movement_type_candidate": MOVEMENT_ALLOWED[str(movement)],
                "movement_type_candidate_source": f"construct_record:{source}",
                "movement_type_candidate_evidence": excerpt(evidence_field),
            }
        )
        return result
    if movement == "mechanical":
        result["movement_type_candidate_review"] = "mechanical_only"
        return result
    if movement is not None:
        result["movement_type_candidate_review"] = "unmapped_movement"
        return result
    label, text_source, evidence, review = movement_from_text(row["item_title"], row["item_description"])
    if label:
        result.update(
            {
                "movement_type_candidate": label,
                "movement_type_candidate_source": text_source,
                "movement_type_candidate_evidence": evidence,
            }
        )
    elif review:
        result["movement_type_candidate_review"] = review
    return result


def enrich(row: dict[str, str], predictor: dict[str, object]) -> dict[str, str]:
    enriched = dict(row)
    enriched.update(propose_brand(row))
    enriched.update(propose_model(row))
    enriched.update(propose_year(row))
    enriched.update(propose_case(row, predictor))
    enriched.update(propose_movement(row, predictor))
    enriched.update(precious_stones(row))
    enriched.update(attachment_fields(row))
    return enriched


def output_path(folder: Path) -> Path:
    candidate = folder / "expanded_pull_eligible_feature_candidates.csv"
    if not candidate.exists():
        return candidate
    version = 2
    while True:
        versioned = folder / f"expanded_pull_eligible_feature_candidates_v{version}.csv"
        if not versioned.exists():
            return versioned
        version += 1


def verify_inputs(approved_fields: list[str], approved: list[dict[str, str]], cleaned_fields: list[str], cleaned: list[dict[str, str]]) -> None:
    if len(approved) != EXPECTED_ROWS or len(cleaned) != EXPECTED_ROWS:
        raise SystemExit(f"Row count is not {EXPECTED_ROWS}.")
    approved_ids = [row["record_id"] for row in approved]
    cleaned_ids = [row["record_id"] for row in cleaned]
    if len(set(approved_ids)) != EXPECTED_ROWS or len(set(cleaned_ids)) != EXPECTED_ROWS:
        raise SystemExit("record_id values are not unique.")
    if set(approved_ids) != set(cleaned_ids):
        raise SystemExit("Approved and cleaned record_id values do not match.")
    missing = [name for name in approved_fields if name not in cleaned_fields]
    if missing:
        raise SystemExit(f"Cleaned candidate is missing source columns: {missing}")
    approved_by_id = {row["record_id"]: row for row in approved}
    mismatches = 0
    for row in cleaned:
        source = approved_by_id[row["record_id"]]
        for name in approved_fields:
            if row[name] != source[name]:
                mismatches += 1
                if mismatches <= 5:
                    print(f"Mismatch {row['record_id']} {name!r}: {source[name]!r} != {row[name]!r}")
    if mismatches:
        raise SystemExit(f"Cleaned candidate changes {mismatches} source values. No output written.")


def verify_output(cleaned: list[dict[str, str]], written_fields: list[str], written: list[dict[str, str]]) -> None:
    if len(written) != EXPECTED_ROWS or len({row["record_id"] for row in written}) != EXPECTED_ROWS:
        raise SystemExit("Output record_id check failed.")
    cleaned_by_id = {row["record_id"]: row for row in cleaned}
    for row in written:
        source = cleaned_by_id[row["record_id"]]
        for name in written_fields:
            if name in NEW_COLUMNS:
                continue
            if row[name] != source[name]:
                raise SystemExit(f"Output changed pre-existing column {name} for {row['record_id']}.")


def reason_tokens(value: str) -> list[str]:
    return [part for part in value.split(";") if part]


def summarize(rows: list[dict[str, str]]) -> None:
    print("\nFill counts")
    fill_fields = {
        "brand": "brand_candidate",
        "model": "model_candidate",
        "production_year": "production_year_candidate",
        "case_material": "case_material_candidate",
        "movement_type": "movement_type_candidate",
        "precious_stones_yes": "precious_stones_candidate",
        "attachment_strap_or_bracelet": "attachment_type_candidate",
        "bracelet_material_stated": "bracelet_material_candidate",
    }
    for label, column in fill_fields.items():
        if column == "precious_stones_candidate":
            count = sum(row[column] == "yes" for row in rows)
        elif column == "attachment_type_candidate":
            count = sum(row[column] in {"strap", "bracelet"} for row in rows)
        elif column == "bracelet_material_candidate":
            count = sum(row[column] not in {"", "unknown", "not_applicable"} for row in rows)
        else:
            count = sum(bool(row[column]) for row in rows)
        print(f"  {label}: {count}")

    print("\nTitle-derived fills for the four checked fields")
    for label, column, source_column in (
        ("brand", "brand_candidate", "brand_candidate_source"),
        ("model", "model_candidate", "model_candidate_source"),
        ("production_year", "production_year_candidate", "production_year_candidate_source"),
        ("case_material", "case_material_candidate", "case_material_candidate_source"),
    ):
        count = sum(bool(row[column]) and row[source_column] == "item_title" for row in rows)
        print(f"  {label}: {count}")

    print("\nFills by source")
    for column, source_column in (
        ("model_candidate", "model_candidate_source"),
        ("production_year_candidate", "production_year_candidate_source"),
        ("case_material_candidate", "case_material_candidate_source"),
        ("movement_type_candidate", "movement_type_candidate_source"),
    ):
        counter = Counter(row[source_column] for row in rows if row[column])
        print(f"  {column}: {dict(counter)}")

    print("\nFills by brand")
    for column in (
        "model_candidate",
        "production_year_candidate",
        "case_material_candidate",
        "movement_type_candidate",
    ):
        counter = Counter(row["designer"] for row in rows if row[column])
        print(f"  {column}: {dict(counter)}")

    print("\nMovement labels")
    print(dict(Counter(row["movement_type_candidate"] or "(blank)" for row in rows)))
    print("\nPrecious stones")
    print(dict(Counter(row["precious_stones_candidate"] for row in rows)))
    print("stone types", dict(Counter(
        stone
        for row in rows
        for stone in row["precious_stones_type_candidate"].split(";")
        if stone
    )))
    print("\nAttachment")
    print(dict(Counter(row["attachment_type_candidate"] for row in rows)))
    print("bracelet material", dict(Counter(row["bracelet_material_candidate"] or "(blank)" for row in rows)))

    print("\nReview flags")
    review_columns = [name for name in NEW_COLUMNS if name.endswith("_review")]
    counter: Counter[str] = Counter()
    flagged_rows = 0
    for row in rows:
        tokens = set()
        for column in review_columns:
            tokens.update(reason_tokens(row[column]))
        if tokens:
            flagged_rows += 1
        counter.update(tokens)
    print(f"  rows with any review flag: {flagged_rows}")
    for reason, count in counter.most_common():
        print(f"  {reason}: {count}")


def spot_check(rows: list[dict[str, str]]) -> None:
    print("\nSpot checks")

    def show(title: str, selected: list[dict[str, str]], field: str, limit: int = 6) -> None:
        print(f"\n{title}")
        for row in selected[:limit]:
            print(f"  {row['record_id']}")
            print(f"    value: {row[field]!r} review: {row[field.replace('_candidate', '_candidate_review')] if False else ''}")
            print("    title:", excerpt(row["item_title"], limit=180))

    show("Model fills", [row for row in rows if row["model_candidate"]], "model_candidate")
    for row in [row for row in rows if row["model_candidate"]][:6]:
        print(f"  FILL {row['model_candidate']!r} | {excerpt(row['item_title'], limit=160)}")
        print(f"    evidence: {row['model_candidate_evidence']}")
    print("Model conflicts")
    for row in [row for row in rows if "conflict_with_existing_model" in row["model_candidate_review"]][:5]:
        print(f"  existing {row['norm_model']!r} | {excerpt(row['item_title'], limit=160)}")
    print("Year fills")
    for row in [row for row in rows if row["production_year_candidate"]][:6]:
        print(
            f"  {row['production_year_candidate']} approx={row['production_year_candidate_approximate']} "
            f"| {excerpt(row['item_title'], limit=160)}"
        )
    print("Case fills")
    for row in [row for row in rows if row["case_material_candidate"]][:8]:
        print(
            f"  {row['case_material_candidate']!r} karat={row['case_material_candidate_karat']} "
            f"color={row['case_material_candidate_color']} construction={row['case_material_candidate_construction']} "
            f"source={row['case_material_candidate_source']}"
        )
        print(f"    {excerpt(row['item_title'], limit=160)}")
    print("Movement text fills")
    for row in [row for row in rows if row["movement_type_candidate_source"] in {"item_title", "item_description"}][:8]:
        print(f"  {row['movement_type_candidate']} via {row['movement_type_candidate_source']}: {row['movement_type_candidate_evidence']}")
    print("Stone yes")
    for row in [row for row in rows if row["precious_stones_candidate"] == "yes"][:5]:
        print(f"  {row['precious_stones_type_candidate']} @ {row['precious_stones_component_candidate']} | {row['precious_stones_candidate_evidence']}")
    print("Stone no")
    for row in [row for row in rows if row["precious_stones_candidate"] == "no"][:5]:
        print(f"  {row['precious_stones_candidate_evidence']} | {excerpt(row['item_title'], limit=120)}")
    print("Attachment both")
    for row in [row for row in rows if "both_strap_and_bracelet" in row["attachment_type_candidate_review"]][:5]:
        print(f"  {excerpt(row['attachment_type_candidate_evidence'], limit=160)}")


def self_check() -> None:
    year, approximate, _evidence, review = production_year_from_title("Cartier Tank, Circa 1970")
    assert (year, approximate, review) == ("1970", "true", ""), (year, approximate, review)
    year, approximate, _evidence, review = production_year_from_title("Tank, c. 2000s")
    assert year == "" and review == "decade_not_a_year", (year, review)
    year, _approximate, _evidence, review = production_year_from_title("Ref: 1675, circa 1968")
    assert year == "1968" and review == "", (year, review)
    year, _approximate, _evidence, review = production_year_from_title("Purchased 21st October, 1986")
    assert year == "" and review == "", (year, review)
    label, _source, _evidence, reason = movement_from_text("automatic wristwatch with manual winding", "")
    assert label == "automatic" and reason == "", (label, reason)
    label, _source, _evidence, reason = movement_from_text("manual wind wristwatch", "")
    assert label == "manual winding", label
    label, _source, _evidence, reason = movement_from_text("self-winding wristwatch", "")
    assert label == "automatic", label
    label, _source, _evidence, reason = movement_from_text("mechanical wristwatch", "")
    assert label == "" and reason == "mechanical_only", (label, reason)
    label, _source, _evidence, reason = movement_from_text("quartz and automatic wristwatch", "")
    assert label == "" and reason == "movement_conflict", (label, reason)
    mentions = case_body_mentions("A stainless steel wristwatch with an 18k gold bezel", "")
    labels = {label for label, _section, _snippet in mentions}
    assert labels == {"steel"}, labels
    assert "bracelet" not in attachment_signals("A stainless steel bracelet watch")[0]
    assert WATCH_TITLE_RE.search("A GOLD PLATED MANUAL WIND WRISTWATCH")
    mentions = case_body_mentions("A gold plated manual wind wristwatch", "")
    labels = {label for label, _section, _snippet in mentions}
    assert labels == {"gold-plated"}, labels
    matches, reason = find_models("CARTIER | REF. 2825, TANKISSIME, A WHITE GOLD WATCH", "Cartier")
    assert reason == "" and matches and matches[0][2] == "tankissime", (matches, reason)
    matches, reason = find_models("Rolex Submariner and Daytona", "Rolex")
    assert reason == "multiple_models" and not matches, (matches, reason)


def main() -> None:
    self_check()
    data_root = REPO_ROOT / "data"
    approved_path = data_root / "expanded_pull_eligible.csv"
    cleaned_path = data_root / "expanded_pull_eligible_cleaned.csv"
    approved_hash = sha256(approved_path)
    cleaned_hash = sha256(cleaned_path)
    print(f"Approved path: {approved_path}")
    print(f"Approved SHA-256: {approved_hash}")
    print(f"Cleaned path: {cleaned_path}")
    print(f"Cleaned SHA-256: {cleaned_hash}")
    if approved_hash != APPROVED_SHA256:
        raise SystemExit("Approved checksum does not match the documented SHA-256. No output written.")
    if cleaned_hash != CLEANED_SHA256:
        raise SystemExit("Cleaned candidate checksum does not match the documented SHA-256. No output written.")
    approved_fields, approved = load_csv(approved_path)
    cleaned_fields, cleaned = load_csv(cleaned_path)
    verify_inputs(approved_fields, approved, cleaned_fields, cleaned)
    print(f"Integrity verified: {EXPECTED_ROWS} matching record_id values; source columns unchanged.")
    approved_by_id = {row["record_id"]: row for row in approved}
    output_rows = []
    for row in cleaned:
        predictor = construct_record(approved_by_id[row["record_id"]])
        output_rows.append(enrich(row, predictor))
    destination = output_path(data_root)
    fieldnames = cleaned_fields + NEW_COLUMNS
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(output_rows)
    written_fields, written = load_csv(destination)
    verify_output(cleaned, written_fields, written)
    digest = sha256(destination)
    print(f"Output path: {destination}")
    print(f"Output SHA-256: {digest}")
    print(f"Output rows: {len(written)}")
    print(f"Output columns: {len(written_fields)}")
    summarize(written)
    spot_check(written)


if __name__ == "__main__":
    main()
