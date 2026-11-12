from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from d012_prepare_remaining_arrays import (  # noqa: E402
    COMPLETED_FIRST_CELL,
    OUT,
    frozen_matrix,
    remaining_rows,
    storage_projection,
    summary,
    write_outputs,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = OUT / "d012_remaining_149_cells.tsv"
WRAPPER_B = OUT / "corehpc/d012_remaining_arm_b_array.slurm"
WRAPPER_C = OUT / "corehpc/d012_remaining_arm_c_array.slurm"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_frozen_matrix_completed_cell_and_remaining_union() -> None:
    matrix = frozen_matrix()
    remaining = remaining_rows()
    completed = [
        row for row in matrix
        if (
            int(row["repetition"]),
            int(row["fold"]),
            row["manifest_id"],
            int(row["training_seed"]),
        ) == COMPLETED_FIRST_CELL
    ]

    assert len(matrix) == 150
    assert len({(row["manifest_id"], row["training_seed"]) for row in matrix}) == 150
    assert len(completed) == 1
    assert len(remaining) == 149
    assert len({(row["manifest_id"], row["training_seed"]) for row in remaining}) == 149

    matrix_keys = {(row["manifest_id"], row["training_seed"]) for row in matrix}
    union = {(row["manifest_id"], row["training_seed"]) for row in remaining + completed}
    assert union == matrix_keys


def test_written_manifest_indices_and_manifest_resolution() -> None:
    write_outputs()
    rows = read_rows(MANIFEST_PATH)

    assert len(rows) == 149
    assert [int(row["array_index"]) for row in rows] == list(range(1, 150))
    for row in rows:
        assert (ROOT / row["manifest_path"]).exists()
        assert int(row["training_seed"]) in {20261101, 20261102, 20261103}
    assert rows[0]["array_index"] == "1"
    assert rows[0]["manifest_id"] == "rep_01_fold_01"
    assert rows[0]["training_seed"] == "20261102"


def test_summary_restart_contract_and_boundaries() -> None:
    info = summary(remaining_rows())

    assert info["frozen_matrix_cell_count"] == 150
    assert info["completed_first_cell_count"] == 1
    assert info["remaining_cell_count"] == 149
    assert info["union_completed_and_remaining_equals_frozen_matrix"] is True
    assert info["b_c_mappings_identical"] is True
    assert info["restart_contract"]["completed_valid_completion_and_hashes"] == "skip"
    assert info["restart_contract"]["failed_or_incomplete"] == "never_count_complete_preserve_partial_artifacts"
    assert info["restart_contract"]["rerun"] == "requires_explicit_array_task_selection"
    assert info["boundaries"]["array_submission_performed"] is False


def test_wrappers_resources_mapping_and_no_submission() -> None:
    b_text = WRAPPER_B.read_text(encoding="utf-8")
    c_text = WRAPPER_C.read_text(encoding="utf-8")

    assert "#SBATCH --array=1-149" in b_text
    assert "#SBATCH --array=1-149" in c_text
    assert "#SBATCH --time=01:00:00" in b_text
    assert "#SBATCH --time=00:30:00" in c_text
    for text, arm in ((b_text, "B"), (c_text, "C")):
        assert "#SBATCH --partition=gpu" in text
        assert "#SBATCH --gres=gpu:nvidia_l40s:1" in text
        assert "#SBATCH --cpus-per-task=4" in text
        assert "#SBATCH --mem=32G" in text
        assert "/scratch/user/mhuang8/envs/tripwire_cyp002" in text
        assert "/scratch/user/mhuang8/d012_runtime_qualification_20261110" in text
        assert "awk -F" in text
        assert "completion.json" in text and "SHA256SUMS.txt" in text
        assert f"--arm {arm}" in text
        assert "sbatch" not in text
        assert "/Users/" not in text


def test_storage_projection_marks_missing_actual_first_cell_sizes_as_unresolved() -> None:
    projection = storage_projection()

    assert projection["canonical_storage_statement"]["scratch_is_execution_and_staging_storage"] is True
    assert projection["canonical_storage_statement"]["canonical_persistent_filesystem_path_established"] is False
    assert projection["actual_size_basis_available"] in ({"B": False, "C": False}, {"B": True, "C": False}, {"B": False, "C": True}, {"B": True, "C": True})
    if not projection["actual_size_basis_available"]["C"]:
        assert projection["projections"]["C"] is None
