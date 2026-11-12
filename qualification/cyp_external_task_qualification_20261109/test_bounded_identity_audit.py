from __future__ import annotations

import csv
import importlib.util
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_bounded_identity_audit.py")
SPEC = importlib.util.spec_from_file_location(
    "run_bounded_identity_audit",
    SCRIPT,
)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_selected_task_counts_are_frozen() -> None:
    selected = audit.selected_task_ids()

    assert {iso: len(ids) for iso, ids in selected.items()} == {
        "CYP1A2": 22,
        "CYP2C9": 33,
        "CYP2D6": 40,
        "CYP3A4": 81,
    }
    assert sum(len(ids) for ids in selected.values()) == 176


def test_chembl_selection_uses_frozen_record_eligibility() -> None:
    selected = audit.selected_task_ids()
    records = audit.read_chembl_universe(selected)

    with audit.TASKS.open(newline="") as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))

    expected_record_count = sum(
        int(row["exact_candidate_record_count"])
        for row in tasks
    )

    assert len(records) == expected_record_count

    source_ids = [row["source_id"] for row in records]
    assert "7614549" not in source_ids

    row_7603640 = [row for row in records if row["source_id"] == "7603640"]
    assert row_7603640 == [{
        "source_family": "chembl",
        "source_partition": "external",
        "source_id": "7603640",
        "isoform": "CYP1A2",
        "assay_chembl_id": "CHEMBL1909132",
        "molecule_chembl_id": "CHEMBL31184",
        "source_smiles": "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    }]


def test_openadmet_universe_counts_are_frozen() -> None:
    records = audit.read_openadmet_universe()

    assert Counter(row["source_partition"] for row in records) == Counter({
        "train": 3433,
        "validation": 736,
        "internal_test": 736,
        "challenge_blinded_test": 750,
    })
    assert len(records) == 5655


def test_empty_source_smiles_is_explicitly_unavailable() -> None:
    represented = audit.represent("")

    assert represented["identity_state"] == "source_structure_unavailable"
    assert represented["full_standard_inchikey"] == ""
