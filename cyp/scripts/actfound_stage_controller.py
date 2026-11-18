
"""Artifact-first routing for the approved ActFound staged design.

Stage executors own the established scientific implementations.
This module does not implement training, checkpoint selection, or resume.
"""
from pathlib import Path
import hashlib
import json


STAGES = ("foundation", "screening", "pic50")


def checkpoint_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Selected checkpoint does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result = {"path": str(path), "sha256": digest.hexdigest()}
    return result


def verify_checkpoint(record):
    actual = checkpoint_record(record["path"])
    if actual != record:
        raise ValueError("Selected checkpoint changed after it was recorded.")
    result = actual
    return result


def write_new_json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_bound_evidence(evidence_path, evidence, pins):
    """Require qualification evidence to be the reviewed file and to name the
    exact deployed sources.

    pins = {"evidence_sha256": str, "bound_fields": {field: sha256, ...}}
    The evidence file must hash to evidence_sha256, and every bound field must
    be present in the evidence and equal its pinned value. No field is bound
    by default; an empty bound_fields is rejected.
    """
    if not isinstance(pins, dict) or set(pins) != {"evidence_sha256", "bound_fields"}:
        raise ValueError(
            "evidence_pins must contain exactly evidence_sha256 and bound_fields."
        )
    bound = pins["bound_fields"]
    if not isinstance(bound, dict) or not bound:
        raise ValueError("evidence_pins.bound_fields must be a non-empty mapping.")
    actual_file = checkpoint_record(evidence_path)["sha256"]
    if actual_file != pins["evidence_sha256"]:
        raise ValueError(
            "Qualification evidence file does not match its pinned SHA-256."
        )
    for field, expected in bound.items():
        if field not in evidence:
            raise ValueError(f"Qualification evidence lacks bound field: {field}")
        if evidence[field] != expected:
            raise ValueError(
                f"Qualification evidence {field} does not match the deployed "
                f"value (evidence {evidence[field]}, pinned {expected})."
            )
    result = {
        "evidence_sha256": actual_file,
        "bound_fields": dict(bound),
    }
    return result


def run_actfound_stage_chain(
    *,
    output_dir,
    executors,
    qualification_result_path,
    evidence_pins,
    authorize_training=False,
):
    """Route foundation -> screening -> pIC50 with explicit artifact lineage.

    Executor contract:
        executor(output_dir=Path, initialization=dict) -> dict

    Foundation/screening return selected_checkpoint.
    pIC50 returns evaluation_manifest, listing stage-relative artifacts.
    Manifest validation here is structural, not proof of O150 completeness.

    The pIC50 executor must implement the existing full evaluation protocol;
    it must not substitute one fit for the approved evaluation.
    """
    if authorize_training is not True:
        raise RuntimeError("ActFound training is not authorized.")
    if set(executors) != set(STAGES):
        raise ValueError("Exactly foundation, screening and pic50 executors required.")
    if not all(callable(executors[name]) for name in STAGES):
        raise ValueError("All stage executors must be callable.")

    evidence_path = Path(qualification_result_path).resolve()
    evidence = json.loads(evidence_path.read_text())
    if (
        evidence.get("status") != "PASS"
        or evidence.get("scope")
        != "GPU fixed-episode validation and checkpoint correctness"
        or evidence.get("optimizer_steps") != 0
    ):
        raise ValueError("Required GPU validation qualification has not passed.")
    binding = verify_bound_evidence(evidence_path, evidence, evidence_pins)

    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(
            f"Refusing implicit resume or overwrite of stage chain: {root}"
        )

    root.mkdir(parents=True)
    write_new_json(root / "qualification_reference.json", {
        "path": str(evidence_path),
        "sha256": binding["evidence_sha256"],
        "bound_fields": binding["bound_fields"],
    })

    completed = {}
    previous = None

    for name in STAGES:
        stage_dir = root / name
        stage_dir.mkdir()

        if name == "foundation":
            initialization = {
                "mode": "original_runner_defined_initialization",
                "source_checkpoint": None,
                "import_optimizer_state": False,
                "import_scheduler_state": False,
                "import_epoch_state": False,
            }
        else:
            verify_checkpoint(previous)
            initialization = {
                "mode": "stage_transfer_not_resume",
                "source_checkpoint": dict(previous),
                "import_optimizer_state": False,
                "import_scheduler_state": False,
                "import_epoch_state": False,
            }

        write_new_json(stage_dir / "initialization.json", initialization)
        try:
            returned = executors[name](
                output_dir=stage_dir,
                initialization=initialization,
            )
            if not isinstance(returned, dict):
                raise ValueError("Stage executor must return a dictionary.")
            if previous is not None:
                verify_checkpoint(previous)

            completion = {
                "stage": name,
                "status": "completed",
                "initialization": initialization,
            }

            if name == "pic50":
                if "evaluation_manifest" not in returned:
                    raise ValueError("pIC50 executor omitted evaluation_manifest.")
                manifest_path = Path(returned["evaluation_manifest"]).resolve()
                if stage_dir not in manifest_path.parents:
                    raise ValueError("Evaluation manifest is outside stage output.")
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("status") != "completed":
                    raise ValueError("pIC50 evaluation manifest is not completed.")

                # Transport contract only: the real evaluator remains
                # responsible for O150 completeness and D007 semantics.
                artifacts = manifest.get("artifacts")
                if not isinstance(artifacts, list) or not artifacts:
                    raise ValueError("Evaluation manifest has no constituent artifacts.")

                recorded = []
                seen = set()
                for relative in artifacts:
                    if not isinstance(relative, str):
                        raise ValueError("Evaluation artifact paths must be strings.")
                    relative_path = Path(relative)
                    if relative_path.is_absolute() or ".." in relative_path.parts:
                        raise ValueError("Expected stage-relative evaluation artifact.")
                    artifact = (stage_dir / relative_path).resolve()
                    if stage_dir not in artifact.parents:
                        raise ValueError("Evaluation artifact escaped stage output.")
                    if artifact == manifest_path or artifact in seen:
                        raise ValueError("Duplicate or self-referencing evaluation artifact.")
                    seen.add(artifact)
                    recorded.append(checkpoint_record(artifact))

                completion["evaluation_manifest"] = checkpoint_record(manifest_path)
                completion["evaluation_artifacts"] = recorded
            else:
                if "selected_checkpoint" not in returned:
                    raise ValueError("Stage executor omitted selected_checkpoint.")
                selected = checkpoint_record(returned["selected_checkpoint"])
                if stage_dir not in Path(selected["path"]).parents:
                    raise ValueError("Selected checkpoint is outside this stage output.")
                completion["selected_checkpoint"] = selected
                previous = selected

            write_new_json(stage_dir / "completion.json", completion)
            completed[name] = completion
        except Exception as error:
            write_new_json(stage_dir / "failure.json", {
                "stage": name,
                "status": "failed",
                "exception_type": type(error).__name__,
                "message": str(error),
            })
            raise

    result = {"status": "completed", "stages": completed}
    write_new_json(root / "completion.json", result)
    return result
