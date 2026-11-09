from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from cyp002_manifest import canonical_json_bytes, finalize_manifest, sha256_bytes, sha256_file
from p2_anil_model import (
    P2MetaHead,
    build_p2_aggregation,
    build_p2_message_passing,
    build_p2_meta_head,
)
from p2_anil_training import P2ValidationController


P2_CELL_CHECKPOINT_SCHEMA = "D010-P2-cell-checkpoint-v1"
P2_CELL_MANIFEST_SCHEMA = "D010-P2-cell-manifest-v1"
MANIFEST_NAME = "manifest.json"
BEST_CHECKPOINT_NAME = "best.ckpt"
METRICS_NAME = "metrics.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"


@dataclass(frozen=True)
class P2CellIdentity:
    repetition: int
    fold: int
    training_seed: int
    split_manifest_path: str
    split_manifest_sha256: str
    dataset_path: str
    dataset_sha256: str
    d010_decision_path: str = "notes/Decisions.md"


def p2_config_sha256(config_object: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(config_object)))


def create_p2_cell_artifact_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"P2 cell artifact directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _paths(cell_dir: Path) -> dict[str, Path]:
    return {
        "manifest": cell_dir / MANIFEST_NAME,
        "checkpoint": cell_dir / BEST_CHECKPOINT_NAME,
        "metrics": cell_dir / METRICS_NAME,
        "sha256sums": cell_dir / SHA256SUMS_NAME,
    }


def build_p2_cell_manifest(
    *,
    cell_identity: P2CellIdentity,
    p2_config: Mapping[str, Any],
    output_dir: str | Path,
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _paths(Path(output_dir))
    return {
        "manifest_schema_version": P2_CELL_MANIFEST_SCHEMA,
        "cell_identity": asdict(cell_identity),
        "p2_config": dict(p2_config),
        "p2_config_sha256": p2_config_sha256(p2_config),
        "execution": {"status": "running", "start_utc": None, "end_utc": None},
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
        raise FileExistsError(f"P2 cell manifest already exists: {manifest_path}")
    manifest_path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def build_checkpoint_state(
    *,
    message_passing: torch.nn.Module,
    aggregation: torch.nn.Module,
    meta_head: P2MetaHead,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    validation_controller: P2ValidationController,
    cell_identity: P2CellIdentity,
    p2_config: Mapping[str, Any],
    scaler_metadata: Mapping[str, Any],
    update_state: Mapping[str, Any],
    episode_state: Mapping[str, Any],
) -> dict[str, Any]:
    if validation_controller.best_score is None or validation_controller.best_update is None:
        raise ValueError("A selected P2 checkpoint requires a best validation score and update.")
    return {
        "checkpoint_schema_version": P2_CELL_CHECKPOINT_SCHEMA,
        "cell_identity": asdict(cell_identity),
        "p2_config": dict(p2_config),
        "p2_config_sha256": p2_config_sha256(p2_config),
        "message_passing_state_dict": _clone_state_dict(message_passing),
        "aggregation_state_dict": _clone_state_dict(aggregation),
        "meta_head_state_dict": _clone_state_dict(meta_head),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "validation_state": {
            "best_score": float(validation_controller.best_score),
            "best_update": int(validation_controller.best_update),
            "bad_checks": int(validation_controller.bad_checks),
            "history": list(validation_controller.history),
        },
        "scaler_metadata": dict(scaler_metadata),
        "update_state": dict(update_state),
        "episode_state": dict(episode_state),
        "interpretation_state": "reload unadapted encoder, aggregation, and meta-head, then perform D010 evaluation-time one-step support adaptation independently per isoform",
    }


def save_selected_checkpoint(state: Mapping[str, Any], checkpoint_path: str | Path) -> str:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"P2 checkpoint already exists: {checkpoint_path}")
    torch.save(dict(state), checkpoint_path)
    return sha256_file(checkpoint_path)


def write_metrics(metrics: Mapping[str, Any], metrics_path: str | Path) -> str:
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        raise FileExistsError(f"P2 metrics artifact already exists: {metrics_path}")
    metrics_path.write_text(json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(metrics_path)


def write_sha256sums(cell_dir: str | Path) -> str:
    paths = _paths(Path(cell_dir))
    required = [paths["manifest"], paths["checkpoint"], paths["metrics"]]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot write P2 SHA256SUMS; missing artifacts: {missing}")
    paths["sha256sums"].write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in required), encoding="utf-8")
    return sha256_file(paths["sha256sums"])


def finalize_p2_cell_manifest(cell_dir: str | Path, *, status: str = "completed") -> dict[str, Any]:
    paths = _paths(Path(cell_dir))
    finalize_manifest(
        manifest_path=paths["manifest"],
        checkpoint_path=paths["checkpoint"],
        metrics_path=paths["metrics"],
        status=status,
    )
    write_sha256sums(cell_dir)
    return json.loads(paths["manifest"].read_text(encoding="utf-8"))


def verify_p2_cell_inventory(cell_dir: str | Path) -> dict[str, str]:
    paths = _paths(Path(cell_dir))
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"P2 cell artifact inventory is incomplete: {missing}")
    observed = {}
    for line in paths["sha256sums"].read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        observed[name] = digest
    for name in (MANIFEST_NAME, BEST_CHECKPOINT_NAME, METRICS_NAME):
        if name not in observed:
            raise ValueError(f"P2 SHA256SUMS is missing {name}.")
        if sha256_file(Path(cell_dir) / name) != observed[name]:
            raise ValueError(f"P2 artifact hash mismatch for {name}.")
    return observed


def load_checkpoint_state(
    checkpoint_path: str | Path,
    *,
    expected_cell_identity: P2CellIdentity,
    expected_p2_config: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        state = torch.load(checkpoint_path, map_location=map_location)
    except Exception as exc:
        raise RuntimeError(f"Could not load P2 checkpoint: {checkpoint_path}") from exc
    if not isinstance(state, dict):
        raise ValueError("P2 checkpoint is not a dictionary.")
    if state.get("checkpoint_schema_version") != P2_CELL_CHECKPOINT_SCHEMA:
        raise ValueError("P2 checkpoint schema mismatch.")
    if state.get("cell_identity") != asdict(expected_cell_identity):
        raise ValueError("P2 checkpoint cell identity mismatch.")
    if state.get("p2_config_sha256") != p2_config_sha256(expected_p2_config):
        raise ValueError("P2 checkpoint configuration hash mismatch.")
    required = {
        "message_passing_state_dict",
        "aggregation_state_dict",
        "meta_head_state_dict",
        "validation_state",
        "scaler_metadata",
        "update_state",
        "episode_state",
    }
    missing = required - set(state)
    if missing:
        raise ValueError(f"P2 checkpoint missing required state: {sorted(missing)}")
    return state


def reload_p2_meta_head_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_cell_identity: P2CellIdentity,
    expected_p2_config: Mapping[str, Any],
    config,
    map_location: str | torch.device = "cpu",
) -> P2MetaHead:
    state = load_checkpoint_state(
        checkpoint_path,
        expected_cell_identity=expected_cell_identity,
        expected_p2_config=expected_p2_config,
        map_location=map_location,
    )
    meta_head = build_p2_meta_head(config)
    meta_head.load_state_dict(state["meta_head_state_dict"])
    return meta_head


def reload_p2_complete_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_cell_identity: P2CellIdentity,
    expected_p2_config: Mapping[str, Any],
    config,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, torch.nn.Module, P2MetaHead]:
    state = load_checkpoint_state(
        checkpoint_path,
        expected_cell_identity=expected_cell_identity,
        expected_p2_config=expected_p2_config,
        map_location=map_location,
    )
    message_passing = build_p2_message_passing(config)
    aggregation = build_p2_aggregation(config)
    meta_head = build_p2_meta_head(config)
    message_passing.load_state_dict(state["message_passing_state_dict"])
    aggregation.load_state_dict(state["aggregation_state_dict"])
    meta_head.load_state_dict(state["meta_head_state_dict"])
    return message_passing, aggregation, meta_head
