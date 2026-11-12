from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_successor_identity_routed_data.py")
SPEC = importlib.util.spec_from_file_location(
    "prepare_successor_identity_routed_data",
    SCRIPT,
)
prep = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prep)


def synthetic_record(activity_id: str, value: str) -> dict[str, str]:
    return {
        "source_family": "chembl",
        "source_partition": "external",
        "source_id": activity_id,
        "activity_id": activity_id,
        "isoform": "CYPX",
        "assay_chembl_id": "CHEMBLASSAY",
        "molecule_chembl_id": f"CHEMBLMOL{activity_id}",
        "source_smiles": "CCO",
        "identity_state": "represented",
        "failure_reason": "",
        "canonical_isomeric_smiles_witness": "CCO",
        "standard_inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        "full_standard_inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "standard_relation": "=",
        "standard_value": value,
        "standard_units": "nM",
        "pchembl_value": "",
        "record_id": activity_id,
        "document_chembl_id": "CHEMBLDOC",
        "canonical_smiles_frozen": "CCO",
    }


def test_pic50_unit_conversion() -> None:
    assert prep.pic50_from_ic50("1000", "nM") == 6.0
    assert prep.pic50_from_ic50("1", "uM") == 6.0
    assert prep.pic50_from_ic50("0.001", "mM") == 6.0


def test_median_replicate_aggregation_is_derived_from_pic50_values() -> None:
    rows = [
        synthetic_record("1", "100"),
        synthetic_record("2", "1000"),
        synthetic_record("3", "10000"),
    ]
    aggregated = prep.aggregate_records(rows)

    assert len(aggregated) == 1
    assert aggregated[0]["derived_pic50_values"] == "7;6;5"
    assert aggregated[0]["median_derived_pic50"] == "6"
    assert aggregated[0]["replicate_count"] == "3"


def test_build_outputs_accounts_for_all_represented_records_once() -> None:
    outputs = prep.build_outputs()
    represented_total = outputs["summary"]["measurement_aggregation"]["represented_exact_records"]
    expanded = []
    for row in outputs["aggregated"]:
        expanded.extend(row["activity_ids"].split(";"))

    assert represented_total == 4821
    assert len(expanded) == represented_total
    assert len(expanded) == len(set(expanded))


def test_unresolved_7817060_remains_unavailable_and_unaggregated() -> None:
    outputs = prep.build_outputs()
    summary = outputs["summary"]

    assert summary["measurement_aggregation"]["source_structure_unavailable_records"] == 1
    all_aggregated_ids = ";".join(row["activity_ids"] for row in outputs["aggregated"])
    assert "7817060" not in all_aggregated_ids


def test_protected_partitions_unavailable_and_train_overlap_available() -> None:
    outputs = prep.build_outputs()
    routed = outputs["routed"]

    protected = [row for row in routed if row["protected_overlap"] == "true"]
    train_only = [row for row in routed if row["routing_class"] == "overlap_openadmet_train"]

    assert protected
    assert all(row["available_for_successor_modeling"] == "false" for row in protected)
    assert train_only
    assert all(row["available_for_successor_modeling"] == "true" for row in train_only)
    assert all(row["overlap_openadmet_train"] == "true" for row in train_only)


def test_no_challenge_labels_no_minimum_task_size_filter_and_frozen_counts() -> None:
    outputs = prep.build_outputs()
    summary = outputs["summary"]

    assert summary["safeguards"]["public_test_labels_accessed"] is False
    assert summary["safeguards"]["minimum_task_size_filter_applied"] is False
    assert "unblinded" not in SCRIPT.read_text(encoding="utf-8").lower()
    assert "public test" not in SCRIPT.read_text(encoding="utf-8").lower()
    assert len(outputs["census"]) == 176
    assert summary["task_level_demonstrated_duplicate_status"]["before_consolidation_task_count"] == 176
    assert summary["task_level_demonstrated_duplicate_status"]["after_consolidation_task_count"] == 175


def test_deterministic_outputs() -> None:
    first = prep.write_outputs(prep.build_outputs())
    second = prep.write_outputs(prep.build_outputs())

    assert first == second
