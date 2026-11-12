from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from d012_arms import ROOT


QUAL = ROOT / "qualification/cyp_external_task_qualification_20261109"
OUT = QUAL / "d012_arm_qualification"
MANIFEST_DIR = ROOT / "cyp/artifacts/cyp_007/split_manifests"
COMPLETED_FIRST_CELL = (1, 1, "rep_01_fold_01", 20261101)
TRAINING_SEEDS = (20261101, 20261102, 20261103)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_matrix() -> list[dict[str, str]]:
    rows = []
    for repetition in range(1, 6):
        for fold in range(1, 11):
            manifest_id = f"rep_{repetition:02d}_fold_{fold:02d}"
            path = MANIFEST_DIR / f"{manifest_id}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            for seed in TRAINING_SEEDS:
                rows.append({
                    "repetition": str(repetition),
                    "fold": str(fold),
                    "manifest_id": manifest_id,
                    "training_seed": str(seed),
                    "manifest_path": str(path.relative_to(ROOT)),
                    "manifest_sha256": sha256_file(path),
                })
    if len(rows) != 150:
        raise RuntimeError(f"expected 150 frozen cells, got {len(rows)}")
    return rows


def remaining_rows() -> list[dict[str, str]]:
    rows = []
    for row in frozen_matrix():
        key = (
            int(row["repetition"]),
            int(row["fold"]),
            row["manifest_id"],
            int(row["training_seed"]),
        )
        if key == COMPLETED_FIRST_CELL:
            continue
        rows.append({"array_index": str(len(rows) + 1), **row})
    if len(rows) != 149:
        raise RuntimeError(f"expected 149 remaining cells, got {len(rows)}")
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "array_index",
        "repetition",
        "fold",
        "manifest_id",
        "training_seed",
        "manifest_path",
        "manifest_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def local_first_cell_dirs() -> dict[str, Path]:
    base = OUT
    return {
        "B": base / "d012_first_cell_arm_b_rep01_fold01_seed20261101",
        "C": base / "d012_first_cell_arm_c_rep01_fold01_seed20261101",
    }


def dir_size(path: Path) -> int | None:
    if not path.exists() or not path.is_dir():
        return None
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def storage_projection() -> dict[str, object]:
    dirs = local_first_cell_dirs()
    actual = {arm: dir_size(path) for arm, path in dirs.items()}
    basis_available = {arm: size is not None for arm, size in actual.items()}
    projections = {}
    for arm, size in actual.items():
        projections[arm] = None if size is None else {
            "first_cell_bytes": size,
            "remaining_149_cells_projected_bytes": size * 149,
            "full_150_cells_projected_bytes": size * 150,
        }
    return {
        "artifact_type": "D012_remaining_arrays_storage_projection_v1",
        "basis": "actual local first-cell artifact directories if present; otherwise unresolved input",
        "local_first_cell_paths_checked": {arm: str(path.relative_to(ROOT)) for arm, path in dirs.items()},
        "actual_size_basis_available": basis_available,
        "projections": projections,
        "checkpoint_type_separation": {
            "B": "requires actual Arm B first-cell directory to separate auxiliary final checkpoint, best target checkpoint, and lightweight artifacts",
            "C": "requires actual Arm C first-cell directory to separate auxiliary final checkpoint, final transfer state, and lightweight artifacts",
        },
        "canonical_storage_statement": {
            "scratch_is_execution_and_staging_storage": True,
            "canonical_persistent_filesystem_path_established": False,
            "canonical_persistent_path": None,
            "unresolved_input": "No completed first-cell artifact directories are present in the local repository checkout; exact checkpoint/file size basis must be collected from gpu-dev1/CoreHPC preserved artifacts before storage can be projected authoritatively.",
        },
    }


def summary(rows: list[dict[str, str]]) -> dict[str, object]:
    matrix = frozen_matrix()
    completed = [
        row for row in matrix
        if (
            int(row["repetition"]),
            int(row["fold"]),
            row["manifest_id"],
            int(row["training_seed"]),
        ) == COMPLETED_FIRST_CELL
    ]
    remaining_keys = {(row["manifest_id"], row["training_seed"]) for row in rows}
    matrix_keys = {(row["manifest_id"], row["training_seed"]) for row in matrix}
    completed_keys = {(row["manifest_id"], row["training_seed"]) for row in completed}
    return {
        "artifact_type": "D012_remaining_production_array_manifest_summary_v1",
        "frozen_matrix_cell_count": len(matrix),
        "completed_first_cell_count": len(completed),
        "remaining_cell_count": len(rows),
        "union_completed_and_remaining_equals_frozen_matrix": (remaining_keys | completed_keys) == matrix_keys,
        "remaining_unique_cell_count": len(remaining_keys),
        "completed_first_cell": {
            "repetition": COMPLETED_FIRST_CELL[0],
            "fold": COMPLETED_FIRST_CELL[1],
            "manifest_id": COMPLETED_FIRST_CELL[2],
            "training_seed": COMPLETED_FIRST_CELL[3],
        },
        "array_index_range": [1, len(rows)],
        "b_c_mappings_identical": True,
        "restart_contract": {
            "completed_valid_completion_and_hashes": "skip",
            "failed_or_incomplete": "never_count_complete_preserve_partial_artifacts",
            "rerun": "requires_explicit_array_task_selection",
            "automatic_scientific_retry_or_change": False,
        },
        "boundaries": {
            "minimum_task_size_filtering": False,
            "tuning": False,
            "t121_retraining": False,
            "d011_changes": False,
            "public_challenge_label_access": False,
            "array_submission_performed": False,
        },
    }


def write_outputs() -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = remaining_rows()
    manifest_path = OUT / "d012_remaining_149_cells.tsv"
    summary_path = OUT / "d012_remaining_arrays_summary.json"
    storage_path = OUT / "d012_remaining_storage_projection.json"
    write_tsv(manifest_path, rows)
    summary_path.write_text(json.dumps(summary(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    storage_path.write_text(json.dumps(storage_projection(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "d012_remaining_149_cells.tsv": sha256_file(manifest_path),
        "d012_remaining_arrays_summary.json": sha256_file(summary_path),
        "d012_remaining_storage_projection.json": sha256_file(storage_path),
    }


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, sort_keys=True))
