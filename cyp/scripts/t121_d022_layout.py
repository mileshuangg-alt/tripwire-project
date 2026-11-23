"""D022 launch layout: three CheMeleon screening runs and three 150-cell arrays.

Controller side only; runs no model code. Arm R comparator runs (r1..r3) already
exist from D021 and are read, never regenerated. Screening seeds are the same
three (20261101-20261103). Reuses the D021 layout helpers and the same
hash-checked reference screening manifest. Nothing has a default: every output
directory must be new.

Run roots (CoreHPC): t121_d022_c_s{N}_{screen,cells}_run, N = 1..3.

Subcommands:
    pins      build d022_pins.json from a local package tree (run by the builder)
    manifests write the three screening manifests and cell-array environments;
              evidence is the CheMeleon smoke result's SHA-256.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t121_d021_layout as base  # noqa: E402

STAGING = base.STAGING
R_PACKAGE = base.R_PACKAGE
C_PACKAGE = f"{STAGING}/t121_d022_package_20261121"
SEEDS = base.SEEDS
NEW_RUNS = (("c", 1), ("c", 2), ("c", 3))

# Roles whose files differ from the reference package (D021 and D022 edits;
# the descriptor module is imported by the cell runner and is shipped but unused).
C_EDITED_ROLES = {
    "screen_stage": "cyp/scripts/t121_screen_stage.py",
    "weighted_training": "cyp/scripts/t121_weighted_training.py",
    "cyp002_config": "cyp/scripts/cyp002_config.py",
    "cyp002_model": "cyp/scripts/cyp002_model.py",
    "cyp002_train": "cyp/scripts/cyp002_train.py",
    "production_cell": "cyp/scripts/t121_production_cell.py",
    "screen_chain_run": "cyp/scripts/t121_screen_chain_run.py",
    "descriptor_cache_module": "cyp/scripts/t121_descriptor_cache.py",
    "chemeleon_module": "cyp/scripts/t121_chemeleon.py",
    "chemeleon_config": "cyp/artifacts/d022_chemeleon/chemeleon_config.json",
    "chemeleon_weights": "cyp/artifacts/d022_chemeleon/chemeleon_mp.pt",
}
SMOKE_CHECK_REL = "qualification/t121/d022_chemeleon_smoke_20261121/check.py"
sha256_file = base.sha256_file
write_new_json = base.write_new_json
reference_manifest_checked = base.reference_manifest_checked


def seed_for(replicate):
    return SEEDS[replicate - 1]


def run_name(arm, replicate):
    if arm != "c":
        raise ValueError("D022 creates CheMeleon (c) runs only.")
    return f"t121_d022_c_s{replicate}"


def screen_run_dir(arm, replicate):
    return f"{STAGING}/{run_name(arm, replicate)}_screen_run"


def cells_run_dir(arm, replicate):
    return f"{STAGING}/{run_name(arm, replicate)}_cells_run"


def build_pins(package_dir, reference_manifest):
    """d022_pins.json: the reference manifest's pins with the C package's paths
    and hashes, plus the CheMeleon module, frozen config and weights. Paths in
    the output are CoreHPC paths; `package_dir` is the local tree."""
    package_dir = Path(package_dir)
    pinned = {}
    for role, entry in reference_manifest["pinned_files"].items():
        path = entry["path"]
        if path.startswith(R_PACKAGE + "/"):
            relative = path[len(R_PACKAGE) + 1:]
            local = package_dir / relative
            if not local.is_file():
                raise FileNotFoundError(f"Reference-pinned file missing from the C package: {relative}")
            pinned[role] = {"path": f"{C_PACKAGE}/{relative}", "sha256": sha256_file(local)}
        else:
            pinned[role] = dict(entry)  # external input (screening data), same file
    for role, relative in C_EDITED_ROLES.items():
        local = package_dir / relative
        if not local.is_file():
            raise FileNotFoundError(f"C package file missing: {relative}")
        pinned[role] = {"path": f"{C_PACKAGE}/{relative}", "sha256": sha256_file(local)}
    config = json.loads((package_dir / C_EDITED_ROLES["chemeleon_config"]).read_text())
    if pinned["chemeleon_weights"]["sha256"] != config["weights"]["sha256"]:
        raise ValueError("Packaged CheMeleon weights do not match the frozen config's weights hash.")
    return {
        "package_root": C_PACKAGE,
        "pinned_files": dict(sorted(pinned.items())),
        "screen_inputs": dict(reference_manifest["screen_inputs"]),
        "proposal_path": pinned["proposal"]["path"],
        "chemeleon": {
            "config_path": pinned["chemeleon_config"]["path"],
            "config_sha256": pinned["chemeleon_config"]["sha256"],
            "weights_path": pinned["chemeleon_weights"]["path"],
        },
        "executor_source_sha256": pinned["executor_cell_fit"]["sha256"],
        "smoke_check": {
            "result_path": f"{C_PACKAGE}/results/d022_chemeleon_smoke/result.json",
            "check_py_sha256": sha256_file(package_dir / SMOKE_CHECK_REL),
        },
    }


def smoke_evidence(pins, result_sha256):
    """Qualification evidence: the CheMeleon smoke result, bound to the check
    script, the frozen config and the exact deployed modules it names."""
    if not (isinstance(result_sha256, str) and len(result_sha256) == 64):
        raise ValueError("result_sha256 must be a SHA-256 hex digest.")
    files = pins["pinned_files"]
    return [{
        "path": pins["smoke_check"]["result_path"],
        "sha256": result_sha256,
        "bound_fields": {
            "check_py_sha256": pins["smoke_check"]["check_py_sha256"],
            "chemeleon_config_sha256": pins["chemeleon"]["config_sha256"],
            "chemeleon_weights_sha256": files["chemeleon_weights"]["sha256"],
            "chemeleon_module_sha256": files["chemeleon_module"]["sha256"],
            "screen_stage_sha256": files["screen_stage"]["sha256"],
            "weighted_training_sha256": files["weighted_training"]["sha256"],
        },
    }]


def arm_c_manifest(pins, evidence, replicate):
    if replicate not in (1, 2, 3):
        raise ValueError("CheMeleon has screening replicates 1 to 3.")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("CheMeleon evidence must be a non-empty list of PASS checks.")
    return {
        "output_dir": screen_run_dir("c", replicate),
        "training_seed": seed_for(replicate),
        "proposal_path": pins["proposal_path"],
        "pinned_files": json.loads(json.dumps(pins["pinned_files"])),
        "screen_inputs": dict(pins["screen_inputs"]),
        "chemeleon": dict(pins["chemeleon"]),
        "evidence": evidence,
    }


def cell_environment(replicate, pins):
    """Variables for t121_cells.slurm, except SCREEN_SELECTION_SHA256 (known
    only after the screening run; the commands compute it from the record)."""
    environment = {
        "PACKAGE_ROOT": C_PACKAGE,
        "T121_ENV": base.T121_ENV,
        "CELL_RUN_ROOT": cells_run_dir("c", replicate),
        "SCREEN_SELECTION": f"{screen_run_dir('c', replicate)}/selected_screen_record.json",
        "EXECUTOR_SOURCE_SHA256": base.EXECUTOR_CELL_FIT_SHA256,
        "DATASET": f"{base.PROJECT_ROOT}/cyp/data/cyp-challenge-TRAIN_inhibition.csv",
        "ORIGINAL_SPLIT": f"{base.PROJECT_ROOT}/cyp/artifacts/cyp_baseline/molecule_split_assignments.csv",
        "SPLIT_MANIFEST_ROOT": f"{base.PROJECT_ROOT}/cyp/artifacts/cyp_007/split_manifests",
        "SCREEN_PROPOSAL": f"{C_PACKAGE}/inputs/trial_121_proposal.json",
        "SCREEN_CSV": pins["screen_inputs"]["screen_path"],
        "SCREEN_IDENTITY": pins["screen_inputs"]["identity_path"],
        "SCREEN_SIMILARITY": pins["screen_inputs"]["similarity_path"],
        "SCREEN_CONTRACT": pins["screen_inputs"]["contract_path"],
        "CHEMELEON_CONFIG": pins["chemeleon"]["config_path"],
        "CHEMELEON_CONFIG_SHA256": pins["chemeleon"]["config_sha256"],
        "CHEMELEON_WEIGHTS": pins["chemeleon"]["weights_path"],
    }
    return environment


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pins = sub.add_parser("pins")
    pins.add_argument("--package-dir", required=True, type=Path)
    pins.add_argument("--reference-manifest", required=True, type=Path)
    pins.add_argument("--output", required=True, type=Path)
    manifests = sub.add_parser("manifests")
    manifests.add_argument("--pins", required=True, type=Path)
    manifests.add_argument("--smoke-result-sha256", required=True)
    manifests.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "pins":
        reference = reference_manifest_checked(args.reference_manifest)
        write_new_json(args.output, build_pins(args.package_dir, reference))
        return 0
    pins_data = json.loads(args.pins.read_text())
    evidence = smoke_evidence(pins_data, args.smoke_result_sha256)
    args.output_dir.mkdir(parents=True)
    for arm, replicate in NEW_RUNS:
        name = run_name(arm, replicate)
        write_new_json(args.output_dir / f"screen_manifest_{name}.json",
                       arm_c_manifest(pins_data, evidence, replicate))
        write_new_json(args.output_dir / f"cells_env_{name}.json",
                       cell_environment(replicate, pins_data))
        print("wrote", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
