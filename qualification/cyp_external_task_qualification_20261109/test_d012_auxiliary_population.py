from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("materialize_d012_auxiliary_population.py")
SPEC = importlib.util.spec_from_file_location(
    "materialize_d012_auxiliary_population",
    SCRIPT,
)
d012 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(d012)


def test_only_demonstrated_duplicate_pair_is_consolidated() -> None:
    outputs = d012.build_outputs()
    duplicate = outputs["summary"]["demonstrated_duplicate_consolidation"]

    assert duplicate["left_task"] == {
        "isoform": "CYP3A4",
        "assay_chembl_id": "CHEMBL2448771",
        "available_identity_count": 36,
    }
    assert duplicate["right_task"] == {
        "isoform": "CYP3A4",
        "assay_chembl_id": "CHEMBL3137501",
        "available_identity_count": 27,
    }
    assert duplicate["relationship"]["shared_assay_src_id"] == "14"
    assert duplicate["relationship"]["shared_assay_src_assay_id"] == "1034082"


def test_d012_task_and_identity_counts() -> None:
    outputs = d012.build_outputs()
    summary = outputs["summary"]

    assert summary["method_contract"]["source_task_count"] == 176
    assert summary["method_contract"]["final_task_unit_count"] == 175
    assert len(outputs["task_units"]) == 175
    assert summary["demonstrated_duplicate_consolidation"]["identity_intersection_count"] == 0
    assert summary["demonstrated_duplicate_consolidation"]["identity_union_count"] == 63
    assert summary["final_population_counts"]["task_identity_instances"] == 4664


def test_d012_observation_and_routing_counts_are_available_only() -> None:
    outputs = d012.build_outputs()
    counts = outputs["summary"]["final_population_counts"]

    assert counts["source_exact_observations_represented_by_instances"] == 4795
    assert counts["external_only_identity_instances"] == 4615
    assert counts["overlap_openadmet_train_identity_instances"] == 49
    assert counts["mixed_available_overlap_class_identity_instances"] == 0
    assert all(
        row["routing_class"] in {"external_only", "overlap_openadmet_train"}
        for row in outputs["instances"]
    )


def test_task_size_distribution_and_no_minimum_filter() -> None:
    outputs = d012.build_outputs()
    distribution = outputs["summary"]["task_size_distribution"]
    contract = outputs["summary"]["method_contract"]

    assert contract["minimum_task_size_filter_applied"] is False
    assert contract["old_16_molecule_rule_applied"] is False
    assert distribution["overall"]["min"] == 15
    assert distribution["overall"]["max"] == 160
    assert distribution["overall"]["zero_identity_task_count"] == 0
    assert distribution["zero_identity_tasks"] == []
    assert distribution["overall"]["small_size_counts_1_to_20"]["15"] == 1


def test_consolidated_task_preserves_source_assays_and_activities() -> None:
    outputs = d012.build_outputs()
    consolidated = [
        row for row in outputs["task_units"]
        if row["task_unit_id"] == d012.D012_CONSOLIDATED_ASSAY_ID
    ]
    assert len(consolidated) == 1
    assert consolidated[0]["source_assay_chembl_ids"] == "CHEMBL2448771;CHEMBL3137501"
    assert consolidated[0]["available_identity_count"] == "63"
    assert consolidated[0]["source_exact_record_count"] == "66"
    assert consolidated[0]["source_represented_record_count"] == "66"


def test_deterministic_outputs() -> None:
    first = d012.write_outputs(d012.build_outputs())
    second = d012.write_outputs(d012.build_outputs())

    assert first == second
