"""D024 launch layout: two penalty-arm 150-cell arrays from the existing r1 screening checkpoint.

Controller side only; runs no model code. No screening run is created: both
arms read the saved r1 selected screening record, and the unchanged arm is the
saved r1 cell array. Nothing has a default: every output directory must be new.

Run roots (CoreHPC): t121_d024_{q3,rank}_cells_run.

Subcommands:
    pins          build d024_pins.json from a local package tree (run by the builder)
    environments  write each arm's variables for t121_cells.slurm and the cell planner
"""
from pathlib import Path
import argparse
import json
import re
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t121_d021_layout as base  # noqa: E402

STAGING = base.STAGING
R_PACKAGE = base.R_PACKAGE
P_PACKAGE = f"{STAGING}/t121_d024_package_20261124"
R1_SELECTION = f"{STAGING}/t121_production_chain_20261117_screen_run/selected_screen_record.json"
ARMS = ("q3", "rank")
CONFIG_ROLES = {"q3": "penalty_config_q3", "rank": "penalty_config_rank"}

# Roles whose files differ from the reference package or are new in it.
P_EDITED_ROLES = {
    "screen_stage": "cyp/scripts/t121_screen_stage.py",
    "weighted_training": "cyp/scripts/t121_weighted_training.py",
    "cyp002_config": "cyp/scripts/cyp002_config.py",
    "cyp002_model": "cyp/scripts/cyp002_model.py",
    "cyp002_train": "cyp/scripts/cyp002_train.py",
    "production_cell": "cyp/scripts/t121_production_cell.py",
    "plan_cells": "cyp/scripts/t121_plan_cells.py",
    "cells_slurm": "cyp/scripts/t121_cells.slurm",
    "descriptor_cache_module": "cyp/scripts/t121_descriptor_cache.py",
    "chemeleon_module": "cyp/scripts/t121_chemeleon.py",
    "penalty_module": "cyp/scripts/t121_screening_penalty.py",
    "penalty_config_q3": "cyp/artifacts/d024_penalty/penalty_q3.json",
    "penalty_config_rank": "cyp/artifacts/d024_penalty/penalty_rank.json",
    "evaluation_config": "cyp/artifacts/d024_penalty/evaluation.json",
    "coverage_config": "cyp/artifacts/d024_penalty/coverage.json",
    "validation_inference": "cyp/scripts/t121_validation_inference.py",
    "validation_inference_slurm": "cyp/scripts/t121_validation_inference_timing.slurm",
    "d021_aggregate": "cyp/scripts/t121_d021_aggregate.py",
    "d024_aggregate": "cyp/scripts/t121_d024_aggregate.py",
    # Reference bytes, pinned here because the cell planner checks its hash.
    "executor_helpers": "executor_source/cyp007_production_helpers.py",
}
sha256_file = base.sha256_file
write_new_json = base.write_new_json
reference_manifest_checked = base.reference_manifest_checked


def cells_run_dir(arm):
    if arm not in ARMS:
        raise ValueError(f"D024 creates the {ARMS} runs only.")
    return f"{STAGING}/t121_d024_{arm}_cells_run"


def build_pins(package_dir, reference_manifest):
    """d024_pins.json: the reference manifest's pins with this package's paths
    and hashes, plus the penalty module and configs. Paths in the output are
    CoreHPC paths; `package_dir` is the local tree."""
    package_dir = Path(package_dir)
    pinned = {}
    for role, entry in reference_manifest["pinned_files"].items():
        path = entry["path"]
        if path.startswith(R_PACKAGE + "/"):
            relative = path[len(R_PACKAGE) + 1:]
            local = package_dir / relative
            if not local.is_file():
                raise FileNotFoundError(f"Reference-pinned file missing from the package: {relative}")
            pinned[role] = {"path": f"{P_PACKAGE}/{relative}", "sha256": sha256_file(local)}
        else:
            pinned[role] = dict(entry)  # external input (screening data), same file
    for role, relative in P_EDITED_ROLES.items():
        local = package_dir / relative
        if not local.is_file():
            raise FileNotFoundError(f"Package file missing: {relative}")
        pinned[role] = {"path": f"{P_PACKAGE}/{relative}", "sha256": sha256_file(local)}
    return {
        "package_root": P_PACKAGE,
        "pinned_files": dict(sorted(pinned.items())),
        "screen_inputs": dict(reference_manifest["screen_inputs"]),
        "proposal_path": pinned["proposal"]["path"],
        "executor_source_sha256": pinned["executor_cell_fit"]["sha256"],
    }


def cell_environment(arm, pins, screen_selection_sha256):
    """Variables for t121_cells.slurm plus the two hashes the cell planner needs."""
    if not re.fullmatch(r"[0-9a-f]{64}", screen_selection_sha256 or ""):
        raise ValueError("screen_selection_sha256 must be a lowercase SHA-256 hex digest.")
    files = pins["pinned_files"]
    screen = pins["screen_inputs"]
    return {
        "PACKAGE_ROOT": P_PACKAGE,
        "T121_ENV": base.T121_ENV,
        "CELL_RUN_ROOT": cells_run_dir(arm),
        "SCREEN_SELECTION": R1_SELECTION,
        "SCREEN_SELECTION_SHA256": screen_selection_sha256,
        "EXECUTOR_SOURCE_SHA256": base.EXECUTOR_CELL_FIT_SHA256,
        "DATASET": f"{base.PROJECT_ROOT}/cyp/data/cyp-challenge-TRAIN_inhibition.csv",
        "ORIGINAL_SPLIT": f"{base.PROJECT_ROOT}/cyp/artifacts/cyp_baseline/molecule_split_assignments.csv",
        "SPLIT_MANIFEST_ROOT": f"{base.PROJECT_ROOT}/cyp/artifacts/cyp_007/split_manifests",
        "SCREEN_PROPOSAL": files["proposal"]["path"],
        "SCREEN_CSV": screen["screen_path"],
        "SCREEN_IDENTITY": screen["identity_path"],
        "SCREEN_SIMILARITY": screen["similarity_path"],
        "SCREEN_CONTRACT": screen["contract_path"],
        "PENALTY_CONFIG": files[CONFIG_ROLES[arm]]["path"],
        "PENALTY_CONFIG_SHA256": files[CONFIG_ROLES[arm]]["sha256"],
        "PROPOSAL_SHA256": files["proposal"]["sha256"],
        "HELPERS_SHA256": files["executor_helpers"]["sha256"],
    }


def shell_exports(environment):
    return "".join(f"export {key}={shlex.quote(value)}\n" for key, value in sorted(environment.items()))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pins = sub.add_parser("pins")
    pins.add_argument("--package-dir", required=True, type=Path)
    pins.add_argument("--reference-manifest", required=True, type=Path)
    pins.add_argument("--output", required=True, type=Path)
    environments = sub.add_parser("environments")
    environments.add_argument("--pins", required=True, type=Path)
    environments.add_argument("--screen-selection-sha256", required=True)
    environments.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "pins":
        reference = reference_manifest_checked(args.reference_manifest)
        write_new_json(args.output, build_pins(args.package_dir, reference))
        return 0
    pins_data = json.loads(args.pins.read_text())
    built = {arm: cell_environment(arm, pins_data, args.screen_selection_sha256) for arm in ARMS}
    args.output_dir.mkdir(parents=True)
    for arm, environment in built.items():
        name = f"t121_d024_{arm}"
        write_new_json(args.output_dir / f"cells_env_{name}.json", environment)
        with (args.output_dir / f"cells_env_{name}.sh").open("x") as handle:
            handle.write(shell_exports(environment))
        print("wrote", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
