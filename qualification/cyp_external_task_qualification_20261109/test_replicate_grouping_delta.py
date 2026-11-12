from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("characterize_replicate_grouping_delta.py")
SPEC = importlib.util.spec_from_file_location(
    "characterize_replicate_grouping_delta",
    SCRIPT,
)
delta = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(delta)


def test_characterization_uses_only_represented_chembl_records() -> None:
    result = delta.characterize()
    summary = result["summary"]

    assert summary["represented_exact_records_analyzed"] == 4821
    unresolved = summary[
        "unresolved_records_kept_outside_represented_key_comparison"
    ]
    assert unresolved == [{
        "activity_id": "7817060",
        "isoform": "CYP2D6",
        "assay_chembl_id": "CHEMBL1909136",
        "molecule_chembl_id": "CHEMBL508338",
        "identity_state": "source_structure_unavailable",
        "standard_value": "3000.0",
        "standard_units": "nM",
        "pchembl_value": "5.52",
    }]
    assert "7817060" not in summary["affected_activity_ids"]


def test_grouping_delta_invariants_are_nonaggregating() -> None:
    result = delta.characterize()
    summary = result["summary"]

    assert summary["source_molecule_id_groups"] >= summary["full_inchikey_groups"]
    assert summary["same_key_different_standard_inchi_count"] == 0
    assert summary["non_aggregating_contract"] == {
        "median_used_as_modeling_label": False,
        "measurement_aggregation_performed": False,
        "routing_or_exclusion_performed": False,
        "task_removal_performed": False,
        "partition_changes_performed": False,
        "training_performed": False,
        "public_test_label_access": False,
    }
