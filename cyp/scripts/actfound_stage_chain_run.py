"""Launcher for the ActFound foundation -> screening -> pIC50 stage chain.

Everything comes from one reviewed manifest; nothing falls back to a default
path or value. Before any import of scientific code, every file the manifest
pins (runner, model-side, adapters, controller, configs, inputs) is hashed and
must match. The qualification evidence is then bound to the same sources by
the controller (verify_bound_evidence).

Usage:
    python -I actfound_stage_chain_run.py --manifest M.json [--authorize-training]
Without --authorize-training the manifest is validated and every pinned file
is verified, and nothing else happens.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys

MANIFEST_KEYS = {
    "output_dir", "pinned_files", "evidence",
    "foundation", "screening", "pic50",
}
EVIDENCE_KEYS = {"path", "sha256", "bound_fields"}
FOUNDATION_KEYS = {"frozen_config_path", "repo_root", "upstream_dir"}
SCREENING_KEYS = {"frozen_config_path", "upstream_dir", "screen_inputs"}
SCREEN_INPUT_KEYS = {
    "screen_path", "identity_path", "similarity_path", "contract_path",
    "split_assignments_path",
}
PIC50_KEYS = {
    "actfound_python", "scoring_python", "evaluation_dir", "evaluation_config",
    "qualification_config", "counts", "frozen_train_config", "manifest_dir",
    "freeze_record", "label_table", "repo_root", "upstream_dir",
}
REQUIRED_PINNED_ROLES = {
    "runner", "model_side", "stage_adapters", "stage_controller",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_keys(name, mapping, expected):
    if not isinstance(mapping, dict) or set(mapping) != expected:
        actual = sorted(mapping) if isinstance(mapping, dict) else type(mapping)
        raise ValueError(
            f"Manifest section {name} must have exactly {sorted(expected)}; "
            f"got {actual}."
        )


def validate_manifest(manifest):
    require_keys("manifest", manifest, MANIFEST_KEYS)
    require_keys("evidence", manifest["evidence"], EVIDENCE_KEYS)
    require_keys("foundation", manifest["foundation"], FOUNDATION_KEYS)
    require_keys("screening", manifest["screening"], SCREENING_KEYS)
    require_keys(
        "screening.screen_inputs",
        manifest["screening"]["screen_inputs"],
        SCREEN_INPUT_KEYS,
    )
    require_keys("pic50", manifest["pic50"], PIC50_KEYS)
    pinned = manifest["pinned_files"]
    if not isinstance(pinned, dict) or not REQUIRED_PINNED_ROLES <= set(pinned):
        raise ValueError(
            f"pinned_files must include roles {sorted(REQUIRED_PINNED_ROLES)}."
        )
    for role, entry in pinned.items():
        require_keys(f"pinned_files.{role}", entry, {"path", "sha256"})


def verify_pinned_files(pinned):
    """Hash every pinned file; return the verified {role: {path, sha256}}."""
    verified = {}
    for role, entry in sorted(pinned.items()):
        path = Path(entry["path"])
        if not path.is_file():
            raise ValueError(f"Pinned file missing: {role}: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"Pinned file changed: {role}: {path} "
                f"(actual {actual}, pinned {entry['sha256']})"
            )
        verified[role] = {"path": str(path.resolve()), "sha256": actual}
    return verified


def evidence_pins(manifest):
    evidence = manifest["evidence"]
    return {
        "evidence_sha256": evidence["sha256"],
        "bound_fields": dict(evidence["bound_fields"]),
    }


def import_pinned_modules(verified):
    """Import the runner, adapters and controller from their verified files."""
    directories = {Path(verified[role]["path"]).parent for role in (
        "runner", "model_side", "stage_adapters", "stage_controller",
    )}
    if len(directories) != 1:
        raise ValueError("Runner, model-side, adapters and controller must share a directory.")
    directory = str(directories.pop())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    runner = importlib.import_module(Path(verified["runner"]["path"]).stem)
    adapters = importlib.import_module(Path(verified["stage_adapters"]["path"]).stem)
    controller = importlib.import_module(Path(verified["stage_controller"]["path"]).stem)
    for module, role in ((runner, "runner"), (adapters, "stage_adapters"), (controller, "stage_controller")):
        if Path(module.__file__).resolve() != Path(verified[role]["path"]).resolve():
            raise ValueError(f"Imported {role} is not the verified file: {module.__file__}")
    return runner, adapters, controller


def build_executors(manifest, runner, adapters):
    foundation = manifest["foundation"]
    screening = manifest["screening"]
    pic50 = manifest["pic50"]
    return {
        "foundation": adapters.foundation_adapter(
            runner=runner,
            frozen_config_path=foundation["frozen_config_path"],
            repo_root=foundation["repo_root"],
            upstream_dir=foundation["upstream_dir"],
        ),
        "screening": adapters.screening_adapter(
            runner=runner,
            frozen_config_path=screening["frozen_config_path"],
            upstream_dir=screening["upstream_dir"],
            screen_inputs=dict(screening["screen_inputs"]),
        ),
        "pic50": adapters.pic50_adapter(
            actfound_python=pic50["actfound_python"],
            scoring_python=pic50["scoring_python"],
            evaluation_dir=pic50["evaluation_dir"],
            evaluation_config=pic50["evaluation_config"],
            qualification_config=pic50["qualification_config"],
            counts=pic50["counts"],
            frozen_train_config=pic50["frozen_train_config"],
            manifest_dir=pic50["manifest_dir"],
            freeze_record=pic50["freeze_record"],
            label_table=pic50["label_table"],
            repo_root=pic50["repo_root"],
            upstream_dir=pic50["upstream_dir"],
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args(argv)

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    verified = verify_pinned_files(manifest["pinned_files"])
    print(json.dumps({
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "verified_files": verified,
    }, indent=2, sort_keys=True), flush=True)
    if not args.authorize_training:
        print("Validated only: --authorize-training not given.", flush=True)
        return 0

    runner, adapters, controller = import_pinned_modules(verified)
    result = controller.run_actfound_stage_chain(
        output_dir=manifest["output_dir"],
        executors=build_executors(manifest, runner, adapters),
        qualification_result_path=manifest["evidence"]["path"],
        evidence_pins=evidence_pins(manifest),
        authorize_training=True,
    )
    print(json.dumps({"status": result["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
