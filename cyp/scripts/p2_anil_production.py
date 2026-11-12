from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from chemprop import data
import numpy as np
import torch

from cyp002_config import CYP002_TASK_NAMES
from cyp002_validation import compute_validation_macro_st_rae
from cyp007_infrastructure import D007_FOLDS, D007_REPETITIONS, D007_TRAINING_SEEDS, sha256_file
from multitask_targets import build_cyp_target_matrix
from p1_pairwise_real_workload_benchmark import (
    build_training_reference_dataset,
    build_validation_query_dataset,
    load_and_validate_manifest,
    load_t121_config,
    route_d007_manifest_data,
)
from p2_anil_benchmark import (
    COMPLETE_MODEL_ARTIFACT_TYPE,
    EXPECTED_COMPLETE_MODEL_PARAMETERS,
    _complete_model_parameter_counts,
    _encode_dataset,
    _subset_dataset,
    p2_complete_model_outer_update_step,
)
from p2_anil_cell_artifacts import (
    BEST_CHECKPOINT_NAME,
    MANIFEST_NAME,
    METRICS_NAME,
    SHA256SUMS_NAME,
    P2CellIdentity,
    build_checkpoint_state,
    build_p2_cell_manifest,
    create_p2_cell_artifact_dir,
    finalize_p2_cell_manifest,
    save_selected_checkpoint,
    verify_p2_cell_inventory,
    write_metrics,
    write_planned_cell_manifest,
)
from p2_anil_model import build_p2_aggregation, build_p2_message_passing, build_p2_meta_head
from p2_anil_training import (
    P2_EARLY_STOPPING_PATIENCE,
    P2_MAX_UPDATES,
    P2_VALIDATION_INTERVAL_UPDATES,
    P2ValidationController,
    build_p2_optimizer_and_scheduler,
    draw_p2_episodes,
    fit_training_only_scaler,
    full_support_task_adaptation_prediction,
    p2_strict_o150_advancement,
    D011Candidate,
)


P2_PRODUCTION_SCHEMA = "D010-P2-production-orchestration-v1"
EXPECTED_CELL_COUNT = D007_REPETITIONS * D007_FOLDS * len(D007_TRAINING_SEEDS)
REQUESTED_WALLTIME_MINUTES = 20
PROSPECTIVE_GPU = "NVIDIA L40S"
PROSPECTIVE_GPU_COUNT = 1
BENCHMARK_SHA256 = "b44457583ab37b14a0f36b38b458cf4874f6fe2da7f742c5f81af4f80bf7b9f4"
MEASURED_COMPONENT_SECONDS_PER_CELL = 672.524620135373
T121_O150 = 0.701212576796


@dataclass(frozen=True, order=True)
class P2CellKey:
    repetition: int
    fold: int
    training_seed: int


@dataclass(frozen=True)
class P2CellState:
    key: P2CellKey
    path: str
    state: str
    reason: str | None = None


def validate_cell_key(key: P2CellKey) -> None:
    if key.repetition < 1 or key.repetition > D007_REPETITIONS:
        raise ValueError("P2 cell repetition is outside the frozen D007 range.")
    if key.fold < 1 or key.fold > D007_FOLDS:
        raise ValueError("P2 cell fold is outside the frozen D007 range.")
    if key.training_seed not in D007_TRAINING_SEEDS:
        raise ValueError("P2 cell training seed is not frozen for D007.")


def enumerate_p2_cells() -> list[P2CellKey]:
    cells = [
        P2CellKey(repetition, fold, seed)
        for repetition in range(1, D007_REPETITIONS + 1)
        for fold in range(1, D007_FOLDS + 1)
        for seed in D007_TRAINING_SEEDS
    ]
    if len(cells) != EXPECTED_CELL_COUNT or len(set(cells)) != EXPECTED_CELL_COUNT:
        raise RuntimeError("P2 cell enumeration did not produce exactly 150 unique cells.")
    return cells


def p2_cell_id(key: P2CellKey) -> str:
    validate_cell_key(key)
    return f"rep_{key.repetition:02d}/fold_{key.fold:02d}/seed_{key.training_seed}"


def p2_cell_dir(root: str | Path, key: P2CellKey) -> Path:
    return Path(root) / p2_cell_id(key)


def split_manifest_name(key: P2CellKey) -> str:
    validate_cell_key(key)
    return f"rep_{key.repetition:02d}_fold_{key.fold:02d}.json"


def split_manifest_for_cell(split_manifest_dir: str | Path, key: P2CellKey) -> Path:
    return Path(split_manifest_dir) / split_manifest_name(key)


def load_split_manifests(split_manifest_dir: str | Path) -> dict[tuple[int, int], Path]:
    manifests = {}
    for path in sorted(Path(split_manifest_dir).glob("rep_*_fold_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "D007-split-manifest-v1":
            raise ValueError(f"Unexpected split manifest schema: {path}")
        key = (int(payload["repetition"]), int(payload["fold"]))
        if key in manifests:
            raise ValueError(f"Duplicate split manifest for {key}.")
        manifests[key] = path
    expected = {(r, f) for r in range(1, D007_REPETITIONS + 1) for f in range(1, D007_FOLDS + 1)}
    if set(manifests) != expected:
        raise ValueError("Split manifest directory is not the exact frozen 50-cell grid.")
    return manifests


def classify_cell(cell_root: str | Path, key: P2CellKey) -> P2CellState:
    cell_path = p2_cell_dir(cell_root, key)
    if not cell_path.exists():
        return P2CellState(key=key, path=str(cell_path), state="absent")
    required = [MANIFEST_NAME, BEST_CHECKPOINT_NAME, METRICS_NAME, SHA256SUMS_NAME]
    missing = [name for name in required if not (cell_path / name).exists()]
    if missing:
        return P2CellState(key=key, path=str(cell_path), state="incomplete/failed", reason=f"missing required artifacts: {missing}")
    try:
        verify_p2_cell_inventory(cell_path)
        manifest = json.loads((cell_path / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("execution", {}).get("status") != "completed":
            return P2CellState(key=key, path=str(cell_path), state="incomplete/failed", reason="manifest status is not completed")
    except Exception as exc:
        return P2CellState(key=key, path=str(cell_path), state="incomplete/failed", reason=str(exc))
    return P2CellState(key=key, path=str(cell_path), state="complete")


def _validation_arrays(validation_df) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    targets, mask = build_cyp_target_matrix(validation_df, CYP002_TASK_NAMES)
    y_true = {}
    conf_low = {}
    conf_high = {}
    observed_mask = {}
    for task_index, task in enumerate(CYP002_TASK_NAMES):
        keep = mask[:, task_index]
        observed_mask[task] = keep
        y_true[task] = targets[keep, task_index]
        conf_low[task] = validation_df.loc[keep, f"{task}_conf_low"].to_numpy(dtype=np.float64)
        conf_high[task] = validation_df.loc[keep, f"{task}_conf_high"].to_numpy(dtype=np.float64)
    return y_true, conf_low, conf_high, observed_mask


def evaluate_p2_validation(
    *,
    message_passing,
    aggregation,
    meta_head,
    train_dataset,
    validation_dataset,
    train_df,
    validation_df,
    targets,
    mask,
    scaler_means,
    scaler_scales,
    device: torch.device,
    batch_size: int,
    inner_lr: float = 0.01,
    inner_steps: int = 1,
) -> dict[str, Any]:
    message_passing.eval()
    aggregation.eval()
    meta_head.eval()
    molecule_ids = train_df["Molecule_Name"].tolist()
    name_to_index = {name: index for index, name in enumerate(molecule_ids)}
    train_labels = torch.tensor(targets, dtype=torch.float32, device=device)
    train_mask = torch.tensor(mask, dtype=torch.bool, device=device)
    means_t = torch.tensor(scaler_means, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scaler_scales, dtype=torch.float32, device=device)
    y_pred_all: dict[str, np.ndarray] = {}
    for task_index, task in enumerate(CYP002_TASK_NAMES):
        eligible_names = [name for name, keep in zip(molecule_ids, mask[:, task_index], strict=True) if keep]
        support_dataset = _subset_dataset(train_dataset, eligible_names, name_to_index)
        support_embeddings = _encode_dataset(
            message_passing=message_passing,
            aggregation=aggregation,
            dataset=support_dataset,
            device=device,
            batch_size=batch_size,
        )
        full_train_embeddings = torch.zeros((len(molecule_ids), support_embeddings.shape[1]), dtype=support_embeddings.dtype, device=device)
        full_train_embeddings[[name_to_index[name] for name in eligible_names]] = support_embeddings
        validation_embeddings = _encode_dataset(
            message_passing=message_passing,
            aggregation=aggregation,
            dataset=validation_dataset,
            device=device,
            batch_size=batch_size,
        )
        pred = full_support_task_adaptation_prediction(
            meta_head=meta_head,
            train_embeddings=full_train_embeddings,
            train_labels=train_labels,
            train_mask=train_mask,
            validation_embeddings=validation_embeddings,
            scaler_means=means_t,
            scaler_scales=scales_t,
            task_index=task_index,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
        )
        y_pred_all[task] = pred.detach().cpu().numpy()
    y_true, conf_low, conf_high, observed_mask = _validation_arrays(validation_df)
    y_pred = {task: y_pred_all[task][observed_mask[task]] for task in CYP002_TASK_NAMES}
    return compute_validation_macro_st_rae(
        y_true=y_true,
        y_pred=y_pred,
        conf_low=conf_low,
        conf_high=conf_high,
        task_names=CYP002_TASK_NAMES,
    )


def _load_cell_inputs(*, dataset_path, repository_root, split_manifest_path, t121_config_path, training_seed, device):
    manifest = load_and_validate_manifest(split_manifest_path, repository_root=repository_root, training_seed=training_seed)
    train_df, validation_query_df, provenance = route_d007_manifest_data(dataset_path=dataset_path, manifest=manifest)
    train_dataset, _scaler, _reference_y, _reference_mask, _scaler_scales, reference_metadata = build_training_reference_dataset(train_df)
    validation_dataset = build_validation_query_dataset(validation_query_df)
    validation_df = train_df.iloc[0:0].copy()
    # route_d007_manifest_data intentionally strips validation labels for benchmark use;
    # production validation reloads the labeled rows directly after the firewall split.
    import pandas as pd
    dataset = pd.read_csv(dataset_path).set_index("Molecule_Name", drop=False)
    validation_df = dataset.loc[manifest["validation_names"]].copy()
    config, config_provenance = load_t121_config(t121_config_path=t121_config_path, training_seed=training_seed)
    message_passing = build_p2_message_passing(config).to(device)
    aggregation = build_p2_aggregation(config).to(device)
    meta_head = build_p2_meta_head(config).to(device)
    targets, mask = build_cyp_target_matrix(train_df, CYP002_TASK_NAMES)
    scaler_means, scaler_scales = fit_training_only_scaler(targets, mask)
    return {
        "manifest": manifest,
        "train_df": train_df,
        "validation_df": validation_df,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "dataset_provenance": provenance,
        "reference_metadata": reference_metadata,
        "config": config,
        "config_provenance": config_provenance,
        "message_passing": message_passing,
        "aggregation": aggregation,
        "meta_head": meta_head,
        "targets": targets,
        "mask": mask,
        "scaler_means": scaler_means,
        "scaler_scales": scaler_scales,
    }


def run_p2_cell(
    *,
    output_dir: str | Path,
    repository_root: str | Path,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    t121_config_path: str | Path,
    training_seed: int,
    device: str | torch.device = "cuda:0",
    max_updates: int = P2_MAX_UPDATES,
    validation_interval_updates: int = P2_VALIDATION_INTERVAL_UPDATES,
    inner_lr: float = 0.01,
    inner_steps: int = 1,
) -> dict[str, Any]:
    if max_updates <= 0 or validation_interval_updates <= 0:
        raise ValueError("max_updates and validation_interval_updates must be positive.")
    device = torch.device(device)
    candidate = D011Candidate(float(inner_lr), int(inner_steps))
    loaded = _load_cell_inputs(
        dataset_path=dataset_path,
        repository_root=repository_root,
        split_manifest_path=split_manifest_path,
        t121_config_path=t121_config_path,
        training_seed=training_seed,
        device=device,
    )
    manifest = loaded["manifest"]
    key = P2CellKey(int(manifest["repetition"]), int(manifest["fold"]), int(training_seed))
    output_dir = create_p2_cell_artifact_dir(output_dir)
    identity = P2CellIdentity(
        repetition=key.repetition,
        fold=key.fold,
        training_seed=key.training_seed,
        split_manifest_path=str(split_manifest_path),
        split_manifest_sha256=sha256_file(split_manifest_path),
        dataset_path=str(dataset_path),
        dataset_sha256=sha256_file(dataset_path),
    )
    parameter_counts = _complete_model_parameter_counts(loaded["message_passing"], loaded["aggregation"], loaded["meta_head"])
    p2_config = {
        "model": "D010-P2",
        "artifact_type": "D010-P2-production-cell-v1",
        "inner_lr": 0.01,
        "candidate_inner_lr": candidate.inner_lr,
        "candidate_inner_steps": candidate.inner_steps,
        "max_updates": P2_MAX_UPDATES,
        "validation_interval_updates": P2_VALIDATION_INTERVAL_UPDATES,
        "early_stopping_patience": P2_EARLY_STOPPING_PATIENCE,
        "parameter_counts": parameter_counts,
    }
    planned = build_p2_cell_manifest(
        cell_identity=identity,
        p2_config=p2_config,
        output_dir=output_dir,
        environment={"device": str(device)},
        provenance={
            "schema": P2_PRODUCTION_SCHEMA,
            "protected_internal_test_excluded": True,
            "d008_evidence_excluded": True,
            "t121_config": loaded["config_provenance"],
        },
    )
    write_planned_cell_manifest(planned, output_dir / MANIFEST_NAME)
    optimizer, scheduler = build_p2_optimizer_and_scheduler(
        list(loaded["message_passing"].parameters()) + list(loaded["meta_head"].parameters())
    )
    controller = P2ValidationController(patience=P2_EARLY_STOPPING_PATIENCE)
    rng = np.random.default_rng(training_seed)
    molecule_ids = loaded["train_df"]["Molecule_Name"].tolist()
    name_to_index = {name: index for index, name in enumerate(molecule_ids)}
    best_states = None
    update = 0
    while update < max_updates and not controller.stopped:
        update += 1
        episodes = draw_p2_episodes(
            molecule_ids=molecule_ids,
            labels=loaded["targets"],
            mask=loaded["mask"],
            scaler_means=loaded["scaler_means"],
            scaler_scales=loaded["scaler_scales"],
            rng=rng,
        )
        p2_complete_model_outer_update_step(
            message_passing=loaded["message_passing"],
            aggregation=loaded["aggregation"],
            meta_head=loaded["meta_head"],
            optimizer=optimizer,
            scheduler=scheduler,
            episodes=episodes,
            train_dataset=loaded["train_dataset"],
            name_to_index=name_to_index,
            device=device,
            batch_size=loaded["config"].batch_size,
            inner_lr=candidate.inner_lr,
            inner_steps=candidate.inner_steps,
        )
        if update % validation_interval_updates == 0:
            validation_result = evaluate_p2_validation(
                message_passing=loaded["message_passing"],
                aggregation=loaded["aggregation"],
                meta_head=loaded["meta_head"],
                train_df=loaded["train_df"],
                validation_df=loaded["validation_df"],
                train_dataset=loaded["train_dataset"],
                validation_dataset=loaded["validation_dataset"],
                targets=loaded["targets"],
                mask=loaded["mask"],
                scaler_means=loaded["scaler_means"],
                scaler_scales=loaded["scaler_scales"],
                device=device,
                batch_size=loaded["config"].batch_size,
                inner_lr=candidate.inner_lr,
                inner_steps=candidate.inner_steps,
            )
            improved = controller.observe(update=update, validation_result=validation_result, state_dict={"dummy": torch.tensor([0.0])})
            if improved:
                best_states = {
                    "message_passing": {k: v.detach().cpu().clone() for k, v in loaded["message_passing"].state_dict().items()},
                    "aggregation": {k: v.detach().cpu().clone() for k, v in loaded["aggregation"].state_dict().items()},
                    "meta_head": {k: v.detach().cpu().clone() for k, v in loaded["meta_head"].state_dict().items()},
                }
    if best_states is None:
        raise RuntimeError("P2 cell completed no validation checkpoint selection.")
    loaded["message_passing"].load_state_dict(best_states["message_passing"])
    loaded["aggregation"].load_state_dict(best_states["aggregation"])
    loaded["meta_head"].load_state_dict(best_states["meta_head"])
    checkpoint = build_checkpoint_state(
        message_passing=loaded["message_passing"],
        aggregation=loaded["aggregation"],
        meta_head=loaded["meta_head"],
        optimizer=optimizer,
        scheduler=scheduler,
        validation_controller=controller,
        cell_identity=identity,
        p2_config=p2_config,
        scaler_metadata={
            "endpoint_order": list(CYP002_TASK_NAMES),
            "scaler_means": loaded["scaler_means"].tolist(),
            "scaler_scales": loaded["scaler_scales"].tolist(),
            "endpoint_eligible_training_counts": loaded["mask"].sum(axis=0).astype(int).tolist(),
        },
        update_state={"update_count": update, "max_updates": max_updates, "scheduler_state": scheduler.state_dict()},
        episode_state={"episodes_per_outer_update": 4, "support_per_task": 13, "query_per_task": 3},
    )
    save_selected_checkpoint(checkpoint, output_dir / BEST_CHECKPOINT_NAME)
    metrics = {
        "best_validation": controller.history[[entry["improved"] for entry in controller.history].index(True)] if controller.history else None,
        "validation_history": controller.history,
        "accounting": {"updates_completed": update, "stopped_early": controller.stopped},
    }
    # The first improved record is not necessarily the final best; recover it from score.
    best_score = controller.best_score
    best_record = next(entry for entry in controller.history if float(entry["macro_st_rae"]) == float(best_score))
    metrics["best_validation"] = {k: v for k, v in best_record.items() if k != "improved"}
    write_metrics(metrics, output_dir / METRICS_NAME)
    finalized = finalize_p2_cell_manifest(output_dir)
    return {"cell": asdict(key), "state": "complete", "manifest": finalized, "metrics": metrics}


def preflight_p2_production(
    *,
    production_root: str | Path,
    repository_root: str | Path,
    split_manifest_dir: str | Path,
    dataset_path: str | Path,
    t121_config_path: str | Path,
    benchmark_artifact_path: str | Path,
) -> dict[str, Any]:
    manifests = load_split_manifests(split_manifest_dir)
    cells = enumerate_p2_cells()
    states = [classify_cell(production_root, key) for key in cells]
    counts = {
        "absent": sum(state.state == "absent" for state in states),
        "complete": sum(state.state == "complete" for state in states),
        "incomplete/failed": sum(state.state == "incomplete/failed" for state in states),
    }
    benchmark = json.loads(Path(benchmark_artifact_path).read_text(encoding="utf-8"))
    if benchmark.get("artifact_type") != COMPLETE_MODEL_ARTIFACT_TYPE:
        raise ValueError("D010 benchmark artifact type mismatch.")
    if sha256_file(benchmark_artifact_path) != BENCHMARK_SHA256:
        raise ValueError("D010 benchmark artifact SHA256 mismatch.")
    return {
        "schema": P2_PRODUCTION_SCHEMA,
        "cell_count": len(cells),
        "split_manifest_count": len(manifests),
        "training_seeds": list(D007_TRAINING_SEEDS),
        "expected_unique_cell_identities": EXPECTED_CELL_COUNT,
        "repository_root": str(Path(repository_root).resolve()),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "t121_config_path": str(t121_config_path),
        "t121_config_sha256": sha256_file(t121_config_path),
        "benchmark_artifact_path": str(benchmark_artifact_path),
        "benchmark_artifact_sha256": sha256_file(benchmark_artifact_path),
        "protected_internal_test_excluded": True,
        "d008_evidence_excluded": True,
        "complete_model_architecture": "D010 P2 BondMessagePassing + NormAggregation + P2MetaHead",
        "complete_model_parameters": EXPECTED_COMPLETE_MODEL_PARAMETERS,
        "frozen_schedule": {"max_updates": P2_MAX_UPDATES, "validation_interval_updates": P2_VALIDATION_INTERVAL_UPDATES, "patience": P2_EARLY_STOPPING_PATIENCE},
        "execution_topology": {"gpu_model": PROSPECTIVE_GPU, "gpu_count": PROSPECTIVE_GPU_COUNT, "requested_walltime_minutes": REQUESTED_WALLTIME_MINUTES, "measured_component_seconds_per_cell": MEASURED_COMPONENT_SECONDS_PER_CELL, "measured_component_is_not_complete_runtime_or_upper_bound": True},
        "performance_driven_scheduling": False,
        "cell_artifact_contract": [MANIFEST_NAME, BEST_CHECKPOINT_NAME, METRICS_NAME, SHA256SUMS_NAME],
        "cell_state_counts": counts,
        "cell_states": [asdict(state) for state in states],
        "execution_plan_absent_cells_only": [asdict(state.key) for state in states if state.state == "absent"],
    }


def aggregate_completed_p2_cells(production_root: str | Path) -> dict[str, Any]:
    rows = []
    for key in enumerate_p2_cells():
        state = classify_cell(production_root, key)
        if state.state != "complete":
            raise RuntimeError(f"Cannot aggregate P2 cells; {key} is {state.state}: {state.reason}")
        metrics = json.loads((p2_cell_dir(production_root, key) / METRICS_NAME).read_text(encoding="utf-8"))
        best = metrics.get("best_validation")
        if not isinstance(best, dict) or "macro_st_rae" not in best:
            raise ValueError(f"P2 metrics missing best validation macro for {key}.")
        macro = float(best["macro_st_rae"])
        if not np.isfinite(macro):
            raise ValueError(f"P2 best validation macro is non-finite for {key}.")
        rows.append({"cell": asdict(key), "macro_st_rae": macro, "per_endpoint_st_rae": best.get("per_task")})
    result = p2_strict_o150_advancement([row["macro_st_rae"] for row in rows])
    result["cells"] = rows
    result["endpoint_veto"] = False
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D010 P2 production wrapper utilities.")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--production-root", required=True, type=Path)
    preflight.add_argument("--repository-root", required=True, type=Path)
    preflight.add_argument("--split-manifest-dir", required=True, type=Path)
    preflight.add_argument("--dataset-path", required=True, type=Path)
    preflight.add_argument("--t121-config-artifact-path", required=True, type=Path)
    preflight.add_argument("--benchmark-artifact-path", required=True, type=Path)
    preflight.add_argument("--output-json", required=True, type=Path)
    run = sub.add_parser("run-cell")
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--repository-root", required=True, type=Path)
    run.add_argument("--dataset-path", required=True, type=Path)
    run.add_argument("--split-manifest-path", required=True, type=Path)
    run.add_argument("--t121-config-artifact-path", required=True, type=Path)
    run.add_argument("--training-seed", required=True, type=int)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--inner-lr", default=0.01, type=float)
    run.add_argument("--inner-steps", default=1, type=int)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--production-root", required=True, type=Path)
    aggregate.add_argument("--output-json", required=True, type=Path)
    return parser


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight_p2_production(
            production_root=args.production_root,
            repository_root=args.repository_root,
            split_manifest_dir=args.split_manifest_dir,
            dataset_path=args.dataset_path,
            t121_config_path=args.t121_config_artifact_path,
            benchmark_artifact_path=args.benchmark_artifact_path,
        )
        _write_json(args.output_json, report)
    elif args.command == "run-cell":
        report = run_p2_cell(
            output_dir=args.output_dir,
            repository_root=args.repository_root,
            dataset_path=args.dataset_path,
            split_manifest_path=args.split_manifest_path,
            t121_config_path=args.t121_config_artifact_path,
            training_seed=args.training_seed,
            device=args.device,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
        )
        _write_json(args.output_dir / "run_cell.json", report)
    elif args.command == "aggregate":
        _write_json(args.output_json, aggregate_completed_p2_cells(args.production_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
