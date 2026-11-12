from __future__ import annotations

import hashlib
import json
from pathlib import Path

from d012_arms import ROOT, bounded_smoke


OUT = ROOT / "qualification/cyp_external_task_qualification_20261109/d012_arm_qualification"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = bounded_smoke("rep_01_fold_01", 20261101)
    smoke_path = OUT / "d012_bounded_correctness_smoke_rep01_fold01_seed20261101.json"
    summary_path = OUT / "d012_arm_qualification_summary.json"
    smoke_path.write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "artifact_type": "D012_arm_b_c_bounded_qualification_v1",
        "authority": "bounded_correctness_only_no_production_training",
        "smoke_cell": {
            "manifest_id": "rep_01_fold_01",
            "training_seed": 20261101,
        },
        "qualified_semantics": {
            "arm_b_target_update_changes_encoder": smoke["arm_b"]["target_update_encoder_changed"],
            "arm_c_outer_gradient_reaches_encoder": smoke["arm_c"]["encoder_grad_l1"] > 0,
            "arm_c_target_adaptation_changes_only_head": (
                smoke["arm_c"]["target_adaptation"]["head_changed"]
                and smoke["arm_c"]["target_adaptation"]["encoder_unchanged"]
            ),
            "arm_b_validation_used_for_checkpoint_selection": smoke["arm_b"]["validation_used_for_checkpoint_selection"],
            "arm_c_validation_used_for_checkpoint_selection": smoke["arm_c"]["validation_used_for_checkpoint_selection"],
            "training_only_target_scaling": smoke["target_scaling"]["fit_on_training_names_only"],
            "qualified_plans_consumed": True,
        },
        "boundaries": {
            "production_training_run": False,
            "matrix_150_cell_run": False,
            "slurm_submitted": False,
            "tuning_performed": False,
            "d011_modified": False,
            "t121_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_files = [
        ("d012_bounded_correctness_smoke_rep01_fold01_seed20261101.json", smoke_path),
        ("d012_arm_qualification_summary.json", summary_path),
        ("README.md", OUT / "README.md"),
        ("corehpc/d012_bounded_runtime_benchmark.slurm", OUT / "corehpc/d012_bounded_runtime_benchmark.slurm"),
        ("corehpc/d012_first_cell_arm_b.slurm", OUT / "corehpc/d012_first_cell_arm_b.slurm"),
        ("corehpc/d012_first_cell_arm_c.slurm", OUT / "corehpc/d012_first_cell_arm_c.slurm"),
        ("corehpc/d012_remaining_arm_b_array.slurm", OUT / "corehpc/d012_remaining_arm_b_array.slurm"),
        ("corehpc/d012_remaining_arm_c_array.slurm", OUT / "corehpc/d012_remaining_arm_c_array.slurm"),
        ("d012_remaining_149_cells.tsv", OUT / "d012_remaining_149_cells.tsv"),
        ("d012_remaining_arrays_summary.json", OUT / "d012_remaining_arrays_summary.json"),
        ("d012_remaining_storage_projection.json", OUT / "d012_remaining_storage_projection.json"),
        ("d012_isolated_acceptance_rep01_fold01_seed20261102.json", OUT / "d012_isolated_acceptance_rep01_fold01_seed20261102.json"),
        ("cyp/scripts/d012_bounded_runtime_benchmark.py", ROOT / "cyp/scripts/d012_bounded_runtime_benchmark.py"),
        ("cyp/scripts/d012_complete_model_runtime_benchmark.py", ROOT / "cyp/scripts/d012_complete_model_runtime_benchmark.py"),
        ("cyp/scripts/d012_first_cell_runner.py", ROOT / "cyp/scripts/d012_first_cell_runner.py"),
        ("cyp/scripts/d012_prepare_remaining_arrays.py", ROOT / "cyp/scripts/d012_prepare_remaining_arrays.py"),
        ("cyp/scripts/d012_isolated_acceptance.py", ROOT / "cyp/scripts/d012_isolated_acceptance.py"),
    ]
    hash_path = OUT / "SHA256SUMS.txt"
    with hash_path.open("w", encoding="utf-8") as handle:
        for label, path in output_files:
            handle.write(f"{sha256_file(path)}  {label}\n")
    print(json.dumps({label: sha256_file(path) for label, path in output_files + [("SHA256SUMS.txt", hash_path)]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
