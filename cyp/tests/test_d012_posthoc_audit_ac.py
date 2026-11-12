from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import d012_posthoc_audit_ac as audit  # noqa: E402


def test_a_identifies_p1_cells_mislabeled_as_t121_and_corrects_from_saved_t121() -> None:
    result = audit.audit_a()
    correction = audit.build_correction(result)

    assert result["defect_exists"] is True
    assert result["semantic_identity"]["labeled_t121_equals_p1_cells_exactly"] is True
    assert result["semantic_identity"]["labeled_t121_equals_saved_t121_cells_exactly"] is False
    assert result["cell_population"] == {
        "d012_labeled_t121_cells": 150,
        "p1_cells": 150,
        "saved_t121_cells": 150,
        "all_key_sets_equal": True,
        "key_schema": "repetition + fold + training_seed",
    }
    assert abs(result["semantic_identity"]["corrected_t121_o150_from_saved_cells"] - 0.701212576796) < 1e-12
    assert correction["original_d012_outputs_preserved"] is True
    assert correction["unchanged_quantities"] == ["B", "C", "C_minus_B"]


def test_b_reports_endpoint_limitations_without_regenerating_predictions() -> None:
    summary, rows = audit.audit_b()

    assert summary["cell_count"] == 150
    assert "Descriptive only" in summary["independence_boundary"]
    assert summary["limitations"]["B_endpoint_values_available"] is False
    assert summary["limitations"]["unavailable_endpoint_comparisons"] == ["endpoint:B_minus_T121", "endpoint:C_minus_B"]
    all_rows = [row for row in rows if row["dimension"] == "all_150_cells"]
    assert {row["n"] for row in all_rows} == {150}
    assert "macro:B_minus_T121" in {row["metric"] for row in all_rows}
    assert "CYP1A2_pIC50_direct_inhibition:C_minus_T121" in {row["metric"] for row in all_rows}


def test_c_reconciles_frozen_49_overlap_instances_train_only() -> None:
    summary, observations, molecule_rows = audit.audit_c()
    counts = summary["population_counts"]

    assert counts["overlap_openadmet_train_auxiliary_task_identity_instances"] == 49
    assert counts["assay_identity_observations"] == 49
    assert counts["matched_nonmissing_observations"] + counts["missing_train_endpoint_observations_counted_not_paired"] == 49
    assert set(counts["counts_by_cyp"]) <= {"CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"}
    assert all(row["paired"] or row["train_pic50"] is None for row in observations)
    assert molecule_rows
    assert summary["rules"]["train_labels_only"] is True


def test_no_protected_test_or_training_regeneration_terms_in_audit_script() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")

    assert "TEST-BLINDED" not in source
    assert "cyp-challenge-TEST" not in source
    assert "fit(" not in source
