from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import d012_technical_audit as audit  # noqa: E402


def test_validation_population_equality_is_exact_for_150_cells() -> None:
    section, rows = audit.validation_population_equality()

    assert section["status"] == "PASS"
    assert len(rows) == 150
    assert all(row["population_match"] for row in rows)
    assert all(row["train_validation_overlap_count"] == 0 for row in rows)
    assert {row["validation_count"] for row in rows} == {416, 417, 418}


def test_scoring_records_prediction_recompute_limitation() -> None:
    section = audit.scoring_equivalence()

    assert section["status"] == "UNRESOLVED"
    assert section["prediction_artifacts"]["independent_score_recompute_possible"] is False
    assert section["prediction_artifacts"]["saved_prediction_artifact_count"] == 0


def test_arm_b_equivalence_is_not_claimed_due_to_target_training_differences() -> None:
    section = audit.arm_b_vs_t121()
    classifications = {row["item"]: row["classification"] for row in section["line_item_classifications"]}

    assert section["status"] == "UNRESOLVED"
    assert section["equivalent_except_initialization"] is False
    assert classifications["batch_order_semantics"] == "POTENTIAL_DEFECT"
    assert classifications["lr_schedule"] == "POTENTIAL_DEFECT"


def test_arm_c_detects_all_scalar_head_parameter_adaptation_defect() -> None:
    section = audit.arm_c_anil()

    assert section["status"] == "DEFECT"
    assert section["matches_frozen_anil_semantics"] is False
    defects = [row for row in section["findings"] if row["classification"] == "CONFIRMED_DEFECT"]
    assert defects
    assert "all parameters of D012ScalarHead" in defects[0]["claim"]


def test_full_audit_executive_summary_is_consistent() -> None:
    summary, population_rows, examples = audit.build_audit()

    assert len(population_rows) == 150
    assert len(examples) == 9
    assert summary["executive"]["stored_predictions_allowed_independent_recompute"] is False
    assert summary["executive"]["b_target_training_equivalent_to_t121_except_initialization"] is False
    assert summary["executive"]["c_matches_frozen_anil_semantics"] is False
    assert summary["executive"]["section_status"]["arm_c_anil_implementation"] == "DEFECT"
