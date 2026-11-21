"""D021 launch layout: five screening runs, five 150-cell arrays, one frozen spec.

Controller side only; runs no model code. Everything here is derived from the
D021 record (screening seeds 20261101-20261103 in arm D; two new seeds in arm
R) and from the existing reference manifest (screen_manifest_v2.json, SHA-256
pinned below). Nothing has a default: every output directory must be new.

Run roots (CoreHPC), one per (arm, screening seed):
    R replicate 1  existing, untouched: ..._production_chain_20261117_{screen,cells}_run
    R replicate 2/3, D replicate 1/2/3:  t121_d021_{r,d}_s{N}_{screen,cells}_run

Subcommands:
    pins      build d021_pins.json from a local package tree (run by the builder)
    manifests write the screening manifests (and the cell-array environments)
              for the new runs; arm D evidence is the smoke result's SHA-256.
"""
from pathlib import Path
import argparse
import hashlib
import json

STAGING = "/scratch/user/mhuang8/tripwire_staging"
R_PACKAGE = f"{STAGING}/t121_production_chain_20261117"
D_PACKAGE = f"{STAGING}/t121_d021_package_20261119"
T121_ENV = "/scratch/user/mhuang8/envs/tripwire_cyp002"
PROJECT_ROOT = "/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project"
R_MANIFEST_V2_SHA256 = "ee43737dcf5185be3bd8a369f80fab4e5951471daadcc7caed72d1534ec4d293"
EXECUTOR_CELL_FIT_SHA256 = "625a64fae191573b5a860cdba9a6ae9456d17bc6cd5b0337cc0b7cff1df1a00b"
SEEDS = (20261101, 20261102, 20261103)
# (arm, replicate number) of the runs this layout creates; R replicate 1 exists.
NEW_RUNS = (("r", 2), ("r", 3), ("d", 1), ("d", 2), ("d", 3))

# Roles whose files differ between arm D and the reference package (D021 edits)
# or are new in it; every other pinned role keeps the reference package's file.
D_EDITED_ROLES = {
    "screen_stage": "cyp/scripts/t121_screen_stage.py",
    "weighted_training": "cyp/scripts/t121_weighted_training.py",
    "cyp002_config": "cyp/scripts/cyp002_config.py",
    "cyp002_model": "cyp/scripts/cyp002_model.py",
    "cyp002_train": "cyp/scripts/cyp002_train.py",
    "production_cell": "cyp/scripts/t121_production_cell.py",
    "screen_chain_run": "cyp/scripts/t121_screen_chain_run.py",
    "descriptor_cache_module": "cyp/scripts/t121_descriptor_cache.py",
}
DESCRIPTOR_CACHE_REL = "cyp/artifacts/d021_descriptors/descriptor_cache_v1"
SMOKE_CHECK_REL = "qualification/t121/d021_descriptor_smoke_20261119/check.py"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(replicate):
    return SEEDS[replicate - 1]


def run_name(arm, replicate):
    return f"t121_d021_{arm}_s{replicate}"


def screen_run_dir(arm, replicate):
    return f"{STAGING}/{run_name(arm, replicate)}_screen_run"


def cells_run_dir(arm, replicate):
    return f"{STAGING}/{run_name(arm, replicate)}_cells_run"


def build_pins(package_dir, reference_manifest):
    """d021_pins.json for arm D: the reference manifest's pins with the D
    package's paths and hashes for the edited roles, plus the descriptor cache.
    `package_dir` is the local tree; paths in the output are CoreHPC paths."""
    package_dir = Path(package_dir)
    pinned = {}
    for role, entry in reference_manifest["pinned_files"].items():
        path = entry["path"]
        if path.startswith(R_PACKAGE + "/"):
            relative = path[len(R_PACKAGE) + 1:]
            local = package_dir / relative
            if not local.is_file():
                raise FileNotFoundError(f"Reference-pinned file missing from the D package: {relative}")
            pinned[role] = {"path": f"{D_PACKAGE}/{relative}", "sha256": sha256_file(local)}
        else:
            pinned[role] = dict(entry)  # external input (screening data), same file
    for role, relative in D_EDITED_ROLES.items():
        local = package_dir / relative
        if not local.is_file():
            raise FileNotFoundError(f"D package file missing: {relative}")
        pinned[role] = {"path": f"{D_PACKAGE}/{relative}", "sha256": sha256_file(local)}
    cache_dir = f"{D_PACKAGE}/{DESCRIPTOR_CACHE_REL}"
    for role, name in (("descriptor_manifest", "descriptor_manifest.json"), ("descriptor_npz", "descriptors.npz")):
        local = package_dir / DESCRIPTOR_CACHE_REL / name
        pinned[role] = {"path": f"{cache_dir}/{name}", "sha256": sha256_file(local)}
    proposal = pinned["proposal"]["path"]
    return {
        "package_root": D_PACKAGE,
        "pinned_files": dict(sorted(pinned.items())),
        "screen_inputs": dict(reference_manifest["screen_inputs"]),
        "proposal_path": proposal,
        "descriptor_cache": {
            "directory": cache_dir,
            "manifest_sha256": pinned["descriptor_manifest"]["sha256"],
        },
        "executor_source_sha256": pinned["executor_cell_fit"]["sha256"],
        "smoke_check": {
            "result_path": f"{D_PACKAGE}/results/d021_descriptor_smoke/result.json",
            "check_py_sha256": sha256_file(package_dir / SMOKE_CHECK_REL),
        },
    }


def reference_manifest_checked(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != R_MANIFEST_V2_SHA256:
        raise ValueError("Reference screening manifest does not match its pinned SHA-256.")
    return json.loads(data)


def arm_r_manifest(reference, replicate):
    """Arm R replicate 2/3: the reference manifest unchanged except the output
    directory and the screening seed."""
    if replicate not in (2, 3):
        raise ValueError("R replicate 1 already exists and is never regenerated.")
    manifest = json.loads(json.dumps(reference))
    manifest["output_dir"] = screen_run_dir("r", replicate)
    manifest["training_seed"] = seed_for(replicate)
    return manifest


def smoke_evidence(pins, result_sha256):
    """The D qualification evidence: the descriptor smoke result, bound to the
    check script and to the exact deployed modules and cache it names. Same
    standard as the reference package (PASS file, pinned hash, bound fields)."""
    if not (isinstance(result_sha256, str) and len(result_sha256) == 64):
        raise ValueError("result_sha256 must be a SHA-256 hex digest.")
    files = pins["pinned_files"]
    return [{
        "path": pins["smoke_check"]["result_path"],
        "sha256": result_sha256,
        "bound_fields": {
            "check_py_sha256": pins["smoke_check"]["check_py_sha256"],
            "descriptor_manifest_sha256": pins["descriptor_cache"]["manifest_sha256"],
            "screen_stage_sha256": files["screen_stage"]["sha256"],
            "weighted_training_sha256": files["weighted_training"]["sha256"],
        },
    }]


def arm_d_manifest(pins, evidence, replicate):
    if replicate not in (1, 2, 3):
        raise ValueError("Arm D has screening replicates 1 to 3.")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Arm D evidence must be a non-empty list of PASS checks.")
    return {
        "output_dir": screen_run_dir("d", replicate),
        "training_seed": seed_for(replicate),
        "proposal_path": pins["proposal_path"],
        "pinned_files": json.loads(json.dumps(pins["pinned_files"])),
        "screen_inputs": dict(pins["screen_inputs"]),
        "descriptor_cache": dict(pins["descriptor_cache"]),
        "evidence": evidence,
    }


def cell_environment(arm, replicate, pins=None, reference=None):
    """Variables for t121_cells.slurm, except SCREEN_SELECTION_SHA256 (known
    only after the screening run; the commands compute it from the record)."""
    package = D_PACKAGE if arm == "d" else R_PACKAGE
    source = pins if arm == "d" else None
    screen_inputs = (pins or {}).get("screen_inputs") or reference["screen_inputs"]
    environment = {
        "PACKAGE_ROOT": package,
        "T121_ENV": T121_ENV,
        "CELL_RUN_ROOT": cells_run_dir(arm, replicate),
        "SCREEN_SELECTION": f"{screen_run_dir(arm, replicate)}/selected_screen_record.json",
        "EXECUTOR_SOURCE_SHA256": EXECUTOR_CELL_FIT_SHA256,
        "DATASET": f"{PROJECT_ROOT}/cyp/data/cyp-challenge-TRAIN_inhibition.csv",
        "ORIGINAL_SPLIT": f"{PROJECT_ROOT}/cyp/artifacts/cyp_baseline/molecule_split_assignments.csv",
        "SPLIT_MANIFEST_ROOT": f"{PROJECT_ROOT}/cyp/artifacts/cyp_007/split_manifests",
        "SCREEN_PROPOSAL": f"{package}/inputs/trial_121_proposal.json",
        "SCREEN_CSV": screen_inputs["screen_path"],
        "SCREEN_IDENTITY": screen_inputs["identity_path"],
        "SCREEN_SIMILARITY": screen_inputs["similarity_path"],
        "SCREEN_CONTRACT": screen_inputs["contract_path"],
    }
    if source is not None:
        environment["DESCRIPTOR_CACHE_DIR"] = source["descriptor_cache"]["directory"]
        environment["DESCRIPTOR_MANIFEST_SHA256"] = source["descriptor_cache"]["manifest_sha256"]
    return environment


def write_new_json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pins = sub.add_parser("pins")
    pins.add_argument("--package-dir", required=True, type=Path)
    pins.add_argument("--reference-manifest", required=True, type=Path)
    pins.add_argument("--output", required=True, type=Path)
    manifests = sub.add_parser("manifests")
    manifests.add_argument("--pins", required=True, type=Path)
    manifests.add_argument("--reference-manifest", required=True, type=Path)
    manifests.add_argument("--smoke-result-sha256", required=True)
    manifests.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    reference = reference_manifest_checked(args.reference_manifest)
    if args.command == "pins":
        write_new_json(args.output, build_pins(args.package_dir, reference))
        return 0
    pins_data = json.loads(args.pins.read_text())
    evidence = smoke_evidence(pins_data, args.smoke_result_sha256)
    args.output_dir.mkdir(parents=True)
    for arm, replicate in NEW_RUNS:
        name = run_name(arm, replicate)
        manifest = (
            arm_r_manifest(reference, replicate) if arm == "r"
            else arm_d_manifest(pins_data, evidence, replicate)
        )
        write_new_json(args.output_dir / f"screen_manifest_{name}.json", manifest)
        environment = cell_environment(arm, replicate, pins_data if arm == "d" else None, reference)
        write_new_json(args.output_dir / f"cells_env_{name}.json", environment)
        print("wrote", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
