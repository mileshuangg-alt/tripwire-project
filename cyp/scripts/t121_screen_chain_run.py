"""Launcher for the updated-T121 screening stage (D015 item 4: 50 epochs).

One reviewed manifest, no defaults. Every pinned file is hashed before any
scientific import. The stage itself is the existing t121_screen_stage code:
prepare_screen_stage -> fit_screen_stage (per-epoch checkpoints, minimum
native screening score, no early stopping, no full-screen refit).

The selected epoch's record is written once to selected_screen_record.json;
each pIC50 cell (t121_production_cell.py) pins that file's SHA-256.

Usage:
    python -I t121_screen_chain_run.py --manifest M.json [--authorize-training]
Without --authorize-training the manifest is validated and the pinned files
are verified, and nothing else happens.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` omits the script dir
from actfound_stage_chain_run import require_keys, sha256_file, verify_pinned_files  # noqa: E402
from actfound_stage_controller import verify_bound_evidence  # noqa: E402

MANIFEST_KEYS = {
    "output_dir", "training_seed", "proposal_path", "pinned_files",
    "screen_inputs", "evidence",
}
SCREEN_INPUT_KEYS = {"screen_path", "identity_path", "similarity_path", "contract_path"}
REQUIRED_PINNED_ROLES = {"screen_stage", "weighted_training", "proposal"}
# D021 arm D: optional section; when present the module and both cache files
# must also be pinned, and the section must point at the pinned manifest.
DESCRIPTOR_SECTION_KEYS = {"directory", "manifest_sha256"}
DESCRIPTOR_PINNED_ROLES = {"descriptor_cache_module", "descriptor_manifest", "descriptor_npz"}


def validate_manifest(manifest):
    original = manifest
    has_descriptors = isinstance(manifest, dict) and "descriptor_cache" in manifest
    if has_descriptors:
        manifest = {k: v for k, v in manifest.items() if k != "descriptor_cache"}
    require_keys("manifest", manifest, MANIFEST_KEYS)
    require_keys("screen_inputs", manifest["screen_inputs"], SCREEN_INPUT_KEYS)
    if not isinstance(manifest["training_seed"], int) or isinstance(manifest["training_seed"], bool):
        raise ValueError("training_seed must be an explicit integer.")
    pinned = manifest["pinned_files"]
    if not isinstance(pinned, dict) or not REQUIRED_PINNED_ROLES <= set(pinned):
        raise ValueError(f"pinned_files must include roles {sorted(REQUIRED_PINNED_ROLES)}.")
    for role, entry in pinned.items():
        require_keys(f"pinned_files.{role}", entry, {"path", "sha256"})
    evidence = manifest["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must be a non-empty list of qualification checks.")
    for index, entry in enumerate(evidence):
        require_keys(f"evidence[{index}]", entry, {"path", "sha256", "bound_fields"})
    if Path(manifest["proposal_path"]).resolve() != Path(pinned["proposal"]["path"]).resolve():
        raise ValueError("proposal_path must be the pinned proposal file.")
    if has_descriptors:
        spec = original["descriptor_cache"]
        require_keys("descriptor_cache", spec, DESCRIPTOR_SECTION_KEYS)
        if not DESCRIPTOR_PINNED_ROLES <= set(pinned):
            raise ValueError(
                f"descriptor_cache requires pinned roles {sorted(DESCRIPTOR_PINNED_ROLES)}."
            )
        if pinned["descriptor_manifest"]["sha256"] != spec["manifest_sha256"]:
            raise ValueError("descriptor_cache.manifest_sha256 must equal the pinned descriptor_manifest hash.")
        if Path(spec["directory"]).resolve() != Path(pinned["descriptor_manifest"]["path"]).resolve().parent:
            raise ValueError("descriptor_cache.directory must be the pinned descriptor manifest's directory.")
        if Path(pinned["descriptor_npz"]["path"]).resolve().parent != Path(spec["directory"]).resolve():
            raise ValueError("The pinned descriptor_npz must be in descriptor_cache.directory.")


def verify_evidence(entries):
    """Every listed check must be PASS, be the pinned file, and name the
    deployed sources through its bound fields."""
    verified = []
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_file():
            raise ValueError(f"Qualification evidence missing: {path}")
        evidence = json.loads(path.read_text())
        if evidence.get("status") != "PASS":
            raise ValueError(f"Qualification evidence did not pass: {path}")
        verified.append(verify_bound_evidence(path, evidence, {
            "evidence_sha256": entry["sha256"],
            "bound_fields": dict(entry["bound_fields"]),
        }))
    return verified


def write_new_json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run_screen_chain(manifest, stage_module):
    """Prepare and fit the screening stage; record the selected checkpoint."""
    root = Path(manifest["output_dir"])
    if root.exists():
        raise FileExistsError(f"Refusing to reuse screening output: {root}")
    root.mkdir(parents=True)

    extra = {}
    if manifest.get("descriptor_cache"):
        # D021 arm D: frozen descriptors on every screening molecule. The
        # manifest must pin the cache by its manifest hash.
        from t121_descriptor_cache import DescriptorCache

        spec = manifest["descriptor_cache"]
        require_keys("descriptor_cache", spec, {"directory", "manifest_sha256"})
        extra["descriptor_cache"] = DescriptorCache.load(
            spec["directory"], expected_manifest_sha256=spec["manifest_sha256"]
        )
    prepared = stage_module.prepare_screen_stage(
        **dict(manifest["screen_inputs"]),
        proposal_path=manifest["proposal_path"],
        output_dir=root / "preparation",
        training_seed=manifest["training_seed"],
        **extra,
    )
    selected = stage_module.fit_screen_stage(prepared, authorize_training=True)

    checkpoint = Path(selected["checkpoint"])
    if sha256_file(checkpoint) != selected["sha256"]:
        raise ValueError("Selected screening checkpoint hash mismatch.")
    record_path = root / "selected_screen_record.json"
    write_new_json(record_path, selected)
    completion = {
        "status": "completed",
        "epochs_recorded": len(
            json.loads((checkpoint.parent / "selection.json").read_text())["records"]
        ),
        "selected_epoch": selected["epoch"],
        "selected_record_sha256": sha256_file(record_path),
        "training_seed": manifest["training_seed"],
        "full_screen_refit": False,
    }
    write_new_json(root / "completion.json", completion)
    return completion


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args(argv)

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    verified = verify_pinned_files(manifest["pinned_files"])
    evidence = verify_evidence(manifest["evidence"])
    print(json.dumps({
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "verified_files": verified,
        "verified_evidence": evidence,
    }, indent=2, sort_keys=True), flush=True)
    if not args.authorize_training:
        print("Validated only: --authorize-training not given.", flush=True)
        return 0

    directory = str(Path(verified["screen_stage"]["path"]).parent)
    if directory != str(Path(verified["weighted_training"]["path"]).parent):
        raise ValueError("t121_screen_stage and t121_weighted_training must share a directory.")
    for entry in (directory,) + tuple(
        str(Path(verified[role]["path"]).parent)
        for role in verified if role.startswith("executor_")
    ):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    stage_module = importlib.import_module(Path(verified["screen_stage"]["path"]).stem)
    if Path(stage_module.__file__).resolve() != Path(verified["screen_stage"]["path"]).resolve():
        raise ValueError("Imported screening stage is not the verified file.")
    print(json.dumps(run_screen_chain(manifest, stage_module), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
