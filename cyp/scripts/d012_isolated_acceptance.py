from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PATHS = [
    "cyp/scripts",
    "cyp/data/cyp-challenge-TRAIN_inhibition.csv",
    "cyp/artifacts/cyp_007/split_manifests",
    "qualification/cyp_external_task_qualification_20261109/d012_auxiliary_population",
    "qualification/cyp_external_task_qualification_20261109/d012_implementation_qualification",
    "references/cyp_upstream/custom_scoring_functions.py",
]


def copy_required_tree(bundle_root: Path) -> None:
    for rel in REQUIRED_PATHS:
        source = ROOT / rel
        dest = bundle_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copy2(source, dest)
        else:
            raise FileNotFoundError(source)


def run_command(command: list[str], *, cwd: Path) -> dict[str, object]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = ""
    env.pop("TRIPWIRE_PROJECT_ROOT", None)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def acceptance_probe_code() -> str:
    return r'''
import importlib
import json
import sys

modules = [
    "d012_first_cell_runner",
    "d012_arms",
    "d012_complete_model_runtime_benchmark",
    "cyp002_validation",
    "st_rae",
]
loaded = {name: importlib.import_module(name).__file__ for name in modules}
print(json.dumps({"sys_path": sys.path, "module_files": loaded}, indent=2, sort_keys=True))
'''


def run_acceptance(bundle_root: Path) -> dict[str, object]:
    scripts_dir = bundle_root / "cyp/scripts"
    proof = run_command([sys.executable, "-c", acceptance_probe_code()], cwd=scripts_dir)
    arm_b = run_command(
        [
            sys.executable,
            "d012_first_cell_runner.py",
            "--arm",
            "B",
            "--manifest-id",
            "rep_01_fold_01",
            "--training-seed",
            "20261102",
            "--device",
            "cpu",
            "--allow-cpu",
            "--preflight-only",
            "--output-dir",
            str(bundle_root / "acceptance_outputs/arm_b_rep01_fold01_seed20261102"),
        ],
        cwd=scripts_dir,
    )
    arm_c = run_command(
        [
            sys.executable,
            "d012_first_cell_runner.py",
            "--arm",
            "C",
            "--manifest-id",
            "rep_01_fold_01",
            "--training-seed",
            "20261102",
            "--device",
            "cpu",
            "--allow-cpu",
            "--preflight-only",
            "--output-dir",
            str(bundle_root / "acceptance_outputs/arm_c_rep01_fold01_seed20261102"),
        ],
        cwd=scripts_dir,
    )
    return {
        "bundle_root": str(bundle_root),
        "python_executable": sys.executable,
        "pythonpath_cleared": True,
        "import_probe": proof,
        "arm_b_preflight": arm_b,
        "arm_c_preflight": arm_c,
    }


def validate_result(result: dict[str, object]) -> None:
    failures = []
    for key in ("import_probe", "arm_b_preflight", "arm_c_preflight"):
        if result[key]["returncode"] != 0:
            failures.append(key)
    if failures:
        raise RuntimeError(f"isolated acceptance failures: {failures}")

    bundle_root = Path(result["bundle_root"]).resolve()
    proof = json.loads(result["import_probe"]["stdout"])
    module_files = proof["module_files"]
    for name, path_text in module_files.items():
        path = Path(path_text).resolve()
        if bundle_root not in path.parents:
            raise RuntimeError(f"{name} resolved outside extracted bundle: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D012 isolated extracted-bundle acceptance test.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--keep-bundle", action="store_true")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="d012_acceptance_"))
    bundle_root = tmp / "tripwire-project"
    try:
        copy_required_tree(bundle_root)
        result = run_acceptance(bundle_root)
        validate_result(result)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if not args.keep_bundle:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
