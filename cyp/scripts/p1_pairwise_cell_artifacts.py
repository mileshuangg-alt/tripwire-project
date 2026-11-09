from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from cyp002_manifest import canonical_json_bytes, finalize_manifest, sha256_bytes, sha256_file
from p1_pairwise_chemprop import (
    build_p1_pair_head,
    build_p1_shared_message_passing,
    count_parameters,
)
from p1_pairwise_model import reconstruct_absolute_from_references
from p1_pairwise_training import P1ValidationController


P1_CELL_CHECKPOINT_SCHEMA = "D009-P1-cell-checkpoint-v1"
P1_CELL_MANIFEST_SCHEMA = "D009-P1-cell-manifest-v1"
MANIFEST_NAME = "manifest.json"
BEST_CHECKPOINT_NAME = "best.ckpt"
METRICS_NAME = "metrics.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"


@dataclass(frozen=True)
class P1CellIdentity:
    repetition: int
    fold: int
    training_seed: int
    split_manifest_path: str
    split_manifest_sha256: str
    dataset_path: str
    dataset_sha256: str
    t121_config_path: str
    t121_config_sha256: str


def p1_config_sha256(config_object: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(config_object)))


def create_p1_cell_artifact_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(f"P1 cell artifact directory is not empty: {path}")
    else:
        path.mkdir(parents=True)
    return path


def _required_manifest_paths(cell_dir: Path) -> dict[str, Path]:
    return {
        "manifest": cell_dir / MANIFEST_NAME,
        "checkpoint": cell_dir / BEST_CHECKPOINT_NAME,
        "metrics": cell_dir / METRICS_NAME,
        "sha256sums": cell_dir / SHA256SUMS_NAME,
    }


def build_p1_cell_manifest(
    *,
    cell_identity: P1CellIdentity,
    p1_config: Mapping[str, Any],
    output_dir: str | Path,
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    paths = _required_manifest_paths(output_dir)
    config_hash = p1_config_sha256(p1_config)
    return {
        "manifest_schema_version": P1_CELL_MANIFEST_SCHEMA,
        "cell_identity": asdict(cell_identity),
        "p1_config": dict(p1_config),
        "p1_config_sha256": config_hash,
        "execution": {
            "status": "running",
            "start_utc": None,
            "end_utc": None,
        },
        "outputs": {
            "best_checkpoint_path": str(paths["checkpoint"]),
            "best_checkpoint_sha256": None,
            "metrics_path": str(paths["metrics"]),
            "metrics_sha256": None,
        },
        "environment": dict(environment),
        "provenance": dict(provenance),
    }


def write_planned_cell_manifest(manifest: Mapping[str, Any], manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f"P1 cell manifest already exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def build_checkpoint_state(
    *,
    shared_encoder: torch.nn.Module,
    pair_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    validation_controller: P1ValidationController,
    cell_identity: P1CellIdentity,
    p1_config: Mapping[str, Any],
    reference_metadata: Mapping[str, Any],
    update_state: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
) -> dict[str, Any]:
    best_score = validation_controller.best_score
    best_update = validation_controller.best_update
    if best_score is None or best_update is None:
        raise ValueError("A selected P1 checkpoint requires a best validation score and update.")

    config_hash = p1_config_sha256(p1_config)
    return {
        "checkpoint_schema_version": P1_CELL_CHECKPOINT_SCHEMA,
        "cell_identity": asdict(cell_identity),
        "p1_config": dict(p1_config),
        "p1_config_sha256": config_hash,
        "shared_encoder_state_dict": _clone_state_dict(shared_encoder),
        "pair_head_state_dict": _clone_state_dict(pair_head),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "validation_state": {
            "best_score": float(best_score),
            "best_update": int(best_update),
            "bad_checks": int(validation_controller.bad_checks),
            "history": list(validation_controller.history),
        },
        "reference_metadata": dict(reference_metadata),
        "update_state": dict(update_state),
        "sampler_state": dict(sampler_state),
    }


def save_selected_checkpoint(state: Mapping[str, Any], checkpoint_path: str | Path) -> str:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"P1 checkpoint already exists: {checkpoint_path}")
    torch.save(dict(state), checkpoint_path)
    return sha256_file(checkpoint_path)


def write_metrics(metrics: Mapping[str, Any], metrics_path: str | Path) -> str:
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        raise FileExistsError(f"P1 metrics artifact already exists: {metrics_path}")
    metrics_path.write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(metrics_path)


def write_sha256sums(cell_dir: str | Path) -> str:
    cell_dir = Path(cell_dir)
    paths = _required_manifest_paths(cell_dir)
    required = [paths["manifest"], paths["checkpoint"], paths["metrics"]]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot write SHA256SUMS; missing artifacts: {missing}")

    lines = []
    for path in required:
        lines.append(f"{sha256_file(path)}  {path.name}\n")
    paths["sha256sums"].write_text("".join(lines), encoding="utf-8")
    return sha256_file(paths["sha256sums"])


def finalize_p1_cell_manifest(cell_dir: str | Path, *, status: str = "completed") -> dict[str, Any]:
    cell_dir = Path(cell_dir)
    paths = _required_manifest_paths(cell_dir)
    finalize_manifest(
        manifest_path=paths["manifest"],
        checkpoint_path=paths["checkpoint"],
        metrics_path=paths["metrics"],
        status=status,
    )
    write_sha256sums(cell_dir)
    return json.loads(paths["manifest"].read_text(encoding="utf-8"))


def verify_p1_cell_inventory(cell_dir: str | Path) -> dict[str, str]:
    cell_dir = Path(cell_dir)
    paths = _required_manifest_paths(cell_dir)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"P1 cell artifact inventory is incomplete: {missing}")

    observed = {}
    for line in paths["sha256sums"].read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        observed[name] = digest
    for name in (MANIFEST_NAME, BEST_CHECKPOINT_NAME, METRICS_NAME):
        expected = observed.get(name)
        if expected is None:
            raise ValueError(f"SHA256SUMS is missing required artifact {name}.")
        if sha256_file(cell_dir / name) != expected:
            raise ValueError(f"Artifact hash mismatch for {name}.")
    return observed


def load_checkpoint_state(
    checkpoint_path: str | Path,
    *,
    expected_cell_identity: P1CellIdentity,
    expected_p1_config: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        state = torch.load(checkpoint_path, map_location=map_location)
    except Exception as exc:
        raise RuntimeError(f"Could not load P1 checkpoint: {checkpoint_path}") from exc


    if not isinstance(state, dict):
        raise ValueError("P1 checkpoint is not a dictionary.")
    if state.get("checkpoint_schema_version") != P1_CELL_CHECKPOINT_SCHEMA:
        raise ValueError("P1 checkpoint schema mismatch.")
    if state.get("cell_identity") != asdict(expected_cell_identity):
        raise ValueError("P1 checkpoint cell identity mismatch.")
    expected_config_hash = p1_config_sha256(expected_p1_config)
    if state.get("p1_config_sha256") != expected_config_hash:
        raise ValueError("P1 checkpoint configuration hash mismatch.")
    for key in (
        "shared_encoder_state_dict",
        "pair_head_state_dict",
        "reference_metadata",
        "validation_state",
        "update_state",
        "sampler_state",
    ):
        if key not in state:
            raise ValueError(f"P1 checkpoint is missing required key {key}.")
    return state


def reload_p1_model_from_checkpoint(
    *,
    checkpoint_path: str | Path,
    config,
    expected_cell_identity: P1CellIdentity,
    expected_p1_config: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    state = load_checkpoint_state(
        checkpoint_path,
        expected_cell_identity=expected_cell_identity,
        expected_p1_config=expected_p1_config,
        map_location=map_location,
    )
    shared_encoder = build_p1_shared_message_passing(config)
    pair_head = build_p1_pair_head(config)
    shared_encoder.load_state_dict(state["shared_encoder_state_dict"])
    pair_head.load_state_dict(state["pair_head_state_dict"])
    shared_encoder.eval()
    pair_head.eval()
    return shared_encoder, pair_head, state


def verify_resume_state(
    state: Mapping[str, Any],
    *,
    expected_cell_identity: P1CellIdentity,
    expected_p1_config: Mapping[str, Any],
    expected_update: int,
) -> None:
    if state.get("cell_identity") != asdict(expected_cell_identity):
        raise ValueError("Resume cell identity mismatch.")
    if state.get("p1_config_sha256") != p1_config_sha256(expected_p1_config):
        raise ValueError("Resume P1 configuration mismatch.")
    update_state = state.get("update_state", {})
    if int(update_state.get("update_count", -1)) != int(expected_update):
        raise ValueError("Resume update count mismatch.")
    for key in ("optimizer_state_dict", "scheduler_state_dict", "sampler_state", "validation_state"):
        if state.get(key) is None:
            raise ValueError(f"Resume state is missing {key}.")


def p1_checkpoint_inference(
    *,
    pair_head: torch.nn.Module,
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    reference_y: torch.Tensor,
    reference_mask: torch.Tensor,
    scaler_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from p1_pairwise_benchmark import _all_reference_pair_head

    pair_head.eval()
    with torch.no_grad():
        predicted_delta_z, _ranges = _all_reference_pair_head(
            pair_head,
            query_embeddings,
            reference_embeddings,
            None,
        )
        return reconstruct_absolute_from_references(
            predicted_delta_z,
            reference_y,
            reference_mask,
            scaler_scales,
        )
