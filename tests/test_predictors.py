"""Synthetic checks for the audited predictor rules.

These examples are constructed phrases. They are not auction records.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.predictors import case_body_mentions, construct_record


def row(**overrides: str) -> dict[str, str]:
    base = {
        "record_id": "synthetic",
        "norm_model": "",
        "material_review_status": "",
        "modeling_case_body": "",
        "modeling_case_finish": "",
        "case_material": "",
        "extracted_case_base_materials": "",
        "extracted_case_finish": "",
        "item_title": "",
        "item_description": "",
        "movement_review_status": "",
        "modeling_movement": "",
        "movement": "",
        "extracted_movement": "",
        "production_year_parsed": "",
        "production_year_approximate": "false",
        "sale_year": "2024",
        "case_measurement_mm": "",
        "case_width_mm": "",
        "case_length_mm": "",
    }
    base.update(overrides)
    return base


def mention_labels(title: str, description: str = "") -> set[str]:
    return {label for label, _section, _snippet in case_body_mentions(title, description)}


class PredictorRuleTests(unittest.TestCase):
    def test_silver_snoopy_name_does_not_replace_steel_case(self) -> None:
        title = "Speedmaster Apollo XII 'Silver Snoopy Award' Stainless steel chronograph wristwatch"
        built = construct_record(
            row(
                case_material="stainless steel",
                extracted_case_base_materials="steel",
                item_title=title,
            )
        )
        self.assertEqual(built["case_body"], "steel")
        self.assertEqual(built["case_finish"], "not_stated")
        self.assertNotIn("silver", mention_labels(title))
        self.assertIn("steel", mention_labels(title))

    def test_white_gold_bezel_stays_off_the_steel_case(self) -> None:
        title = "A stainless steel wristwatch with a white gold bezel"
        built = construct_record(
            row(
                case_material="stainless steel",
                extracted_case_base_materials="steel",
                item_title=title,
            )
        )
        self.assertEqual(built["case_body"], "steel")
        self.assertNotIn("white-gold", mention_labels(title))
        self.assertIn("steel", mention_labels(title))

    def test_plating_is_finish_and_does_not_become_solid_gold(self) -> None:
        plated = construct_record(
            row(
                case_material="gold-plated",
                extracted_case_base_materials="steel",
                extracted_case_finish="gold-plated",
                item_title="A gold plated stainless steel wristwatch",
            )
        )
        self.assertEqual(plated["case_body"], "steel")
        self.assertEqual(plated["case_finish"], "gold-plated")

        blocked = construct_record(
            row(
                case_material="yellow gold",
                extracted_case_base_materials="karat-gold",
                item_title="A gold plated wristwatch",
                item_description="Case: gold plated",
            )
        )
        self.assertIsNone(blocked["case_body"])
        self.assertEqual(blocked["case_body_source"], "source_disagreement")

    def test_compatible_generic_gold_keeps_structured_color(self) -> None:
        built = construct_record(
            row(
                case_material="yellow gold",
                extracted_case_base_materials="karat-gold",
                item_title="A lady's 18K gold manual wind wristwatch",
                item_description=(
                    "Dial: white, blued steel hands\n"
                    "Case: polished rectangular\n"
                    "Buckle/Clasp: 18K gold folding clasp"
                ),
            )
        )
        self.assertEqual(built["case_body"], "yellow-gold")
        self.assertEqual(built["case_body_source"], "structured_color_with_generic_gold")

    def test_unresolved_conflicts_stay_missing(self) -> None:
        built = construct_record(
            row(
                material_review_status="unresolved_source_conflict",
                modeling_case_body="steel",
                modeling_case_finish="not_stated",
                case_material="stainless steel",
                extracted_case_base_materials="steel",
                movement_review_status="unresolved_source_conflict",
                modeling_movement="automatic",
                movement="automatic",
                extracted_movement="automatic",
            )
        )
        self.assertIsNone(built["case_body"])
        self.assertIsNone(built["case_finish"])
        self.assertEqual(built["case_body_source"], "unresolved_source_conflict")
        self.assertIsNone(built["movement"])
        self.assertEqual(built["movement_source"], "unresolved_source_conflict")

        disagreed = construct_record(row(movement="automatic", extracted_movement="manual"))
        self.assertIsNone(disagreed["movement"])
        self.assertEqual(disagreed["movement_source"], "source_disagreement")

    def test_approved_values_win_and_approved_blanks_stay_blank(self) -> None:
        approved = construct_record(
            row(
                material_review_status="approved",
                modeling_case_body="steel",
                modeling_case_finish="none",
                case_material="yellow gold",
                extracted_case_base_materials="yellow-gold",
                movement_review_status="approved",
                modeling_movement="manual",
                movement="quartz",
                extracted_movement="quartz",
            )
        )
        self.assertEqual(approved["case_body"], "steel")
        self.assertEqual(approved["case_finish"], "not_stated")
        self.assertEqual(approved["case_body_source"], "approved_correction")
        self.assertEqual(approved["movement"], "manual")
        self.assertEqual(approved["movement_source"], "approved_correction")

        blank = construct_record(
            row(
                material_review_status="approved",
                modeling_case_body="",
                modeling_case_finish="",
                case_material="stainless steel",
                extracted_case_base_materials="steel",
                movement_review_status="approved",
                modeling_movement="",
                movement="automatic",
                extracted_movement="automatic",
            )
        )
        self.assertIsNone(blank["case_body"])
        self.assertEqual(blank["case_finish"], "not_stated")
        self.assertIsNone(blank["movement"])
        self.assertEqual(blank["movement_source"], "approved_correction")

    def test_rectangular_sides_and_bracelet_are_not_diameter(self) -> None:
        labeled = construct_record(row(case_width_mm="20", case_length_mm="28"))
        self.assertIsNone(labeled["case_diameter_mm"])
        self.assertEqual(labeled["case_width_mm"], 20.0)
        self.assertEqual(labeled["case_length_mm"], 28.0)

        round_case = construct_record(row(case_measurement_mm="36"))
        self.assertEqual(round_case["case_diameter_mm"], 36.0)
        self.assertIsNone(round_case["case_width_mm"])
        self.assertIsNone(round_case["case_length_mm"])

        unused = row()
        unused["bracelet_circumference_mm"] = "155"
        unused["case_dimension_1_mm"] = "20"
        unused["case_dimension_2_mm"] = "28"
        unused["case_diameter"] = "40"
        blocked = construct_record(unused)
        self.assertIsNone(blocked["case_diameter_mm"])
        self.assertIsNone(blocked["case_width_mm"])
        self.assertIsNone(blocked["case_length_mm"])

    def test_exact_and_approximate_production_years(self) -> None:
        exact = construct_record(
            row(production_year_parsed="1969", production_year_approximate="false", sale_year="2024")
        )
        self.assertEqual(exact["production_year"], 1969)
        self.assertEqual(exact["age_at_sale"], 55.0)
        self.assertEqual(exact["production_year_approximate"], "false")

        approximate = construct_record(
            row(production_year_parsed="1970", production_year_approximate="true", sale_year="2024")
        )
        self.assertEqual(approximate["production_year"], 1970)
        self.assertEqual(approximate["age_at_sale"], 54.0)
        self.assertEqual(approximate["production_year_approximate"], "true")

        unparsed = construct_record(
            row(production_year_parsed="", production_year_approximate="false", sale_year="2024")
        )
        self.assertIsNone(unparsed["production_year"])
        self.assertIsNone(unparsed["age_at_sale"])
        self.assertIsNone(unparsed["production_year_approximate"])

        after_sale = construct_record(
            row(production_year_parsed="2025", production_year_approximate="false", sale_year="2020")
        )
        self.assertEqual(after_sale["production_year"], 2025)
        self.assertIsNone(after_sale["age_at_sale"])
        self.assertEqual(after_sale["production_year_approximate"], "false")

    def test_mechanical_wording_does_not_override_a_specific_family(self) -> None:
        built = construct_record(row(movement="mechanical", extracted_movement="automatic"))
        self.assertEqual(built["movement"], "automatic")
        self.assertEqual(built["movement_source"], "structured_and_extracted")

    def test_import_does_not_require_the_data_root(self) -> None:
        import scripts.predictors as predictors

        self.assertFalse(hasattr(predictors, "ELIGIBLE_PATH"))
        self.assertIn("construct_record", predictors.__dict__)


if __name__ == "__main__":
    unittest.main()
