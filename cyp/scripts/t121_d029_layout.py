"""D029 launch layout: two further penalty-form 150-cell arrays from the existing r1
screening checkpoint (twosided: two-sided squared band; huber: Huber upper penalty).

Controller side only; runs no model code. Same layout as D024: no screening run
is created, both arms read the saved r1 selected screening record, and the
unchanged arm is the saved r1 cell array. Nothing has a default: every output
directory must be new.

Run roots (CoreHPC): t121_d029_{twosided,huber}_cells_run.

Subcommands:
    pins          build d029_pins.json from a local package tree (run by the builder)
    environments  write each arm's variables for t121_cells.slurm and the cell planner
"""
from pathlib import Path
import argparse
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t121_d024_layout as d024  # noqa: E402

base = d024.base
STAGING = d024.STAGING
R_PACKAGE = d024.R_PACKAGE
P_PACKAGE = f"{STAGING}/t121_d029_package_20261126"
R1_SELECTION = d024.R1_SELECTION
ARMS = ("twosided", "huber")
CONFIG_ROLES = {"twosided": "penalty_config_twosided", "huber": "penalty_config_huber"}

# The D024 roles with this comparison's configs in place of the D024 arm and
# evaluation configs. The coverage grouping is D024's, unchanged.
P_EDITED_ROLES = {
    **{
        role: path for role, path in d024.P_EDITED_ROLES.items()
        if role not in (*d024.CONFIG_ROLES.values(), "evaluation_config")
    },
    "penalty_config_twosided": "cyp/artifacts/d029_penalty/penalty_rank_twosided.json",
    "penalty_config_huber": "cyp/artifacts/d029_penalty/penalty_rank_huber.json",
    "evaluation_config": "cyp/artifacts/d029_penalty/evaluation.json",
}
sha256_file = base.sha256_file
write_new_json = base.write_new_json
reference_manifest_checked = base.reference_manifest_checked
shell_exports = d024.shell_exports


def cells_run_dir(arm):
    if arm not in ARMS:
        raise ValueError(f"D029 creates the {ARMS} runs only.")
    return f"{STAGING}/t121_d029_{arm}_cells_run"


def build_pins(package_dir, reference_manifest):
    """d029_pins.json: the reference manifest's pins with this package's paths
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
        name = f"t121_d029_{arm}"
        write_new_json(args.output_dir / f"cells_env_{name}.json", environment)
        with (args.output_dir / f"cells_env_{name}.sh").open("x") as handle:
            handle.write(shell_exports(environment))
        print("wrote", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
