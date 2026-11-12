from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from d012_first_cell_runner import first_cell_workload_assertions  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "cyp/scripts/d012_first_cell_runner.py"
WRAPPER_B = ROOT / "qualification/cyp_external_task_qualification_20261109/d012_arm_qualification/corehpc/d012_first_cell_arm_b.slurm"
WRAPPER_C = ROOT / "qualification/cyp_external_task_qualification_20261109/d012_arm_qualification/corehpc/d012_first_cell_arm_c.slurm"


def test_first_cell_workload_assertions() -> None:
    result = first_cell_workload_assertions()

    assert result["retained_auxiliary_instances"] == 4660
    assert result["surviving_task_units"] == 175
    assert result["b_updates_per_aux_epoch"] == 388
    assert result["b_total_aux_updates_50_epochs"] == 19400
    assert result["c_updates_per_aux_epoch"] == 857
    assert result["c_total_aux_updates_50_epochs"] == 42850
    assert result["b_c_same_masked_auxiliary_population"] is True
    assert result["validation_labels_enter_auxiliary_training"] is False
    assert result["protected_test_or_challenge_labels_accessed"] is False


def test_preflight_writes_startup_and_completion_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "preflight"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--arm",
            "B",
            "--output-dir",
            str(out),
            "--device",
            "cpu",
            "--allow-cpu",
            "--preflight-only",
        ],
        cwd=ROOT,
        check=True,
    )

    startup = json.loads((out / "startup_config.json").read_text(encoding="utf-8"))
    complete = json.loads((out / "completion.json").read_text(encoding="utf-8"))
    assert startup["workload_assertions"]["retained_auxiliary_instances"] == 4660
    assert complete["status"] == "completed"
    assert (out / "progress.jsonl").exists()
    assert (out / "SHA256SUMS.txt").exists()


def test_failure_preserves_progress_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "failure"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--arm",
            "C",
            "--output-dir",
            str(out),
            "--device",
            "cpu",
            "--allow-cpu",
            "--fail-after-startup",
        ],
        cwd=ROOT,
        check=False,
    )

    assert result.returncode != 0
    assert (out / "startup_config.json").exists()
    assert (out / "progress.jsonl").exists()
    failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert (out / "SHA256SUMS.txt").exists()


def test_corehpc_wrappers_resources_and_paths() -> None:
    b_text = WRAPPER_B.read_text(encoding="utf-8")
    c_text = WRAPPER_C.read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu" in b_text
    assert "#SBATCH --gres=gpu:nvidia_l40s:1" in b_text
    assert "#SBATCH --cpus-per-task=4" in b_text
    assert "#SBATCH --mem=32G" in b_text
    assert "#SBATCH --time=01:00:00" in b_text
    assert "--arm B" in b_text

    assert "#SBATCH --partition=gpu" in c_text
    assert "#SBATCH --gres=gpu:nvidia_l40s:1" in c_text
    assert "#SBATCH --cpus-per-task=4" in c_text
    assert "#SBATCH --mem=32G" in c_text
    assert "#SBATCH --time=00:30:00" in c_text
    assert "--arm C" in c_text

    for text in (b_text, c_text):
        assert "/scratch/user/mhuang8/envs/tripwire_cyp002" in text
        assert "/scratch/user/mhuang8/d012_runtime_qualification_20261110" in text
        assert "/Users/" not in text
