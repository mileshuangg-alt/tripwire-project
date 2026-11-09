from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from cyp002_config import CYP002_TASK_NAMES
from cyp002_validation import compute_validation_macro_st_rae
from cyp007_infrastructure import (
    D007_FOLDS,
    D007_REPETITIONS,
    D007_TRAINING_SEEDS,
    D007_TOTAL_SPLITS,
    sha256_file,
)
from p1_pairwise_cell_artifacts import (
    BEST_CHECKPOINT_NAME,
    MANIFEST_NAME,
    METRICS_NAME,
    SHA256SUMS_NAME,
    P1CellIdentity,
    build_checkpoint_state,
    build_p1_cell_manifest,
    create_p1_cell_artifact_dir,
    finalize_p1_cell_manifest,
    save_selected_checkpoint,
    verify_p1_cell_inventory,
    write_metrics,
    write_planned_cell_manifest,
)
from p1_pairwise_real_workload_benchmark import (
    build_p1_encoder_and_head,
    build_training_reference_dataset,
    build_validation_query_dataset,
    load_and_validate_manifest,
    load_t121_config,
    route_d007_manifest_data,
)
from p1_pairwise_training import (
    P1_EARLY_STOPPING_PATIENCE,
    P1_MAX_UPDATES,
    P1_VALIDATION_INTERVAL_UPDATES,
    P1ValidationController,
    build_p1_optimizer_and_scheduler,
    build_pairwise_contract_from_training_labels,
    draw_p1_training_pair_batch,
    endpoint_weights_tensor,
)
from p1_pairwise_training_runtime_benchmark import RealP1TrainingUpdate
from p1_pairwise_benchmark import _all_reference_pair_head


P1_PRODUCTION_SCHEMA = "D009-P1-production-orchestration-v1"
T121_O150 = 0.701212576796
PROSPECTIVE_WALLTIME_MINUTES = 30
PROSPECTIVE_GPU = "NVIDIA L40S"
PROSPECTIVE_GPU_COUNT = 1
MEASURED_COMPONENT_SUBTOTAL_GPU_HOURS = 30.22
EXPECTED_CELL_COUNT = D007_TOTAL_SPLITS * len(D007_TRAINING_SEEDS)


@dataclass(frozen=True, order=True)
class P1CellKey:
    repetition: int
    fold: int
    training_seed: int


@dataclass(frozen=True)
class P1CellState:
    key: P1CellKey
    path: str
    state: str
    reason: str | None = None


def p1_cell_id(key: P1CellKey) -> str:
    validate_cell_key(key)
    return f"rep_{key.repetition:02d}/fold_{key.fold:02d}/seed_{key.training_seed}"


def p1_cell_dir(root: str | Path, key: P1CellKey) -> Path:
    return Path(root) / p1_cell_id(key)


def validate_cell_key(key: P1CellKey) -> None:
    if key.repetition < 1 or key.repetition > D007_REPETITIONS:
        raise ValueError("P1 cell repetition is outside the frozen D007 range.")
    if key.fold < 1 or key.fold > D007_FOLDS:
        raise ValueError("P1 cell fold is outside the frozen D007 range.")
    if key.training_seed not in D007_TRAINING_SEEDS:
        raise ValueError("P1 cell training seed is not frozen for D007.")


def enumerate_p1_cells() -> list[P1CellKey]:
    cells = [
        P1CellKey(repetition, fold, seed)
        for repetition in range(1, D007_REPETITIONS + 1)
        for fold in range(1, D007_FOLDS + 1)
        for seed in D007_TRAINING_SEEDS
    ]
    if len(cells) != EXPECTED_CELL_COUNT:
        raise RuntimeError("P1 cell enumeration did not produce 150 cells.")
    if len(set(cells)) != EXPECTED_CELL_COUNT:
        raise RuntimeError("P1 cell enumeration contains duplicates.")
    return cells


def split_manifest_name(key: P1CellKey) -> str:
    validate_cell_key(key)
    return f"rep_{key.repetition:02d}_fold_{key.fold:02d}.json"


def split_manifest_for_cell(split_manifest_dir: str | Path, key: P1CellKey) -> Path:
    return Path(split_manifest_dir) / split_manifest_name(key)


def load_split_manifests(split_manifest_dir: str | Path) -> dict[tuple[int, int], Path]:
    split_manifest_dir = Path(split_manifest_dir)
    manifests = {}
    for path in sorted(split_manifest_dir.glob("rep_*_fold_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "D007-split-manifest-v1":
            raise ValueError(f"Unexpected split manifest schema: {path}")
        key = (int(data["repetition"]), int(data["fold"]))
        if key in manifests:
            raise ValueError(f"Duplicate split manifest for repetition/fold: {key}")
        manifests[key] = path
    expected = {
        (repetition, fold)
        for repetition in range(1, D007_REPETITIONS + 1)
        for fold in range(1, D007_FOLDS + 1)
    }
    if set(manifests) != expected:
        raise ValueError("Split manifest directory is not the exact frozen 50-cell grid.")
    return manifests


def classify_cell(cell_root: str | Path, key: P1CellKey) -> P1CellState:
    cell_path = p1_cell_dir(cell_root, key)
    if not cell_path.exists():
        return P1CellState(key=key, path=str(cell_path), state="absent")
    required = [MANIFEST_NAME, BEST_CHECKPOINT_NAME, METRICS_NAME, SHA256SUMS_NAME]
    missing = [name for name in required if not (cell_path / name).exists()]
    if missing:
        return P1CellState(
            key=key,
            path=str(cell_path),
            state="incomplete/failed",
            reason=f"missing required artifacts: {missing}",
        )
    try:
        verify_p1_cell_inventory(cell_path)
        manifest = json.loads((cell_path / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("execution", {}).get("status") != "completed":
            return P1CellState(
                key=key,
                path=str(cell_path),
                state="incomplete/failed",
                reason="manifest status is not completed",
            )
    except Exception as exc:
        return P1CellState(
            key=key,
            path=str(cell_path),
            state="incomplete/failed",
            reason=str(exc),
        )
    return P1CellState(key=key, path=str(cell_path), state="complete")


def preflight_p1_production(
    *,
    production_root: str | Path,
    repository_root: str | Path,
    split_manifest_dir: str | Path,
    dataset_path: str | Path,
    t121_config_path: str | Path,
) -> dict[str, Any]:
    production_root = Path(production_root)
    repository_root = Path(repository_root).resolve()
    if not repository_root.exists() or not repository_root.is_dir():
        raise FileNotFoundError(f"Authoritative repository root is invalid: {repository_root}")
    manifests = load_split_manifests(split_manifest_dir)
    cells = enumerate_p1_cells()
    states = [classify_cell(production_root, key) for key in cells]
    counts = {
        "absent": sum(state.state == "absent" for state in states),
        "complete": sum(state.state == "complete" for state in states),
        "incomplete/failed": sum(state.state == "incomplete/failed" for state in states),
    }
    execution_plan = [asdict(state.key) for state in states if state.state == "absent"]
    return {
        "schema": P1_PRODUCTION_SCHEMA,
        "cell_count": len(cells),
        "split_manifest_count": len(manifests),
        "training_seeds": list(D007_TRAINING_SEEDS),
        "expected_unique_cell_identities": EXPECTED_CELL_COUNT,
        "repository_root": str(repository_root),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "t121_config_path": str(t121_config_path),
        "t121_config_sha256": sha256_file(t121_config_path),
        "protected_internal_test_excluded": True,
        "d008_evidence_excluded": True,
        "p1_scientific_configuration": "qualified D009 P1 components and schedule",
        "cell_artifact_contract": [MANIFEST_NAME, BEST_CHECKPOINT_NAME, METRICS_NAME, SHA256SUMS_NAME],
        "execution_topology": {
            "gpu_model": PROSPECTIVE_GPU,
            "gpu_count": PROSPECTIVE_GPU_COUNT,
            "walltime_minutes": PROSPECTIVE_WALLTIME_MINUTES,
            "measured_component_subtotal_gpu_hours": MEASURED_COMPONENT_SUBTOTAL_GPU_HOURS,
            "subtotal_is_not_complete_runtime_or_upper_bound": True,
        },
        "cell_state_counts": counts,
        "cell_states": [asdict(state) for state in states],
        "execution_plan_absent_cells_only": execution_plan,
        "performance_driven_scheduling": False,
    }


def run_one_cell_wrapper(
    *,
    production_root: str | Path,
    split_manifest_dir: str | Path,
    key: P1CellKey,
    cell_runner: Callable[[P1CellKey, Path, Path], None],
) -> Path:
    validate_cell_key(key)
    cell_path = p1_cell_dir(production_root, key)
    state = classify_cell(production_root, key)
    if state.state == "complete":
        raise FileExistsError("Completed P1 cell must not be overwritten or rerun.")
    if state.state == "incomplete/failed":
        raise RuntimeError("Incomplete P1 cell requires adjudication; mid-cell resume is not authorized.")
    manifest_path = split_manifest_for_cell(split_manifest_dir, key)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Frozen split manifest does not exist: {manifest_path}")
    cell_runner(key, manifest_path, cell_path)
    final_state = classify_cell(production_root, key)
    if final_state.state != "complete":
        raise RuntimeError("P1 cell runner did not produce a complete qualified artifact.")
    return cell_path


def execute_qualified_p1_cell(
    *,
    key: P1CellKey,
    split_manifest_path: Path,
    cell_path: Path,
    repository_root: str | Path,
    dataset_path: str | Path,
    t121_config_path: str | Path,
    max_updates: int = P1_MAX_UPDATES,
    validation_interval_updates: int = P1_VALIDATION_INTERVAL_UPDATES,
    patience: int = P1_EARLY_STOPPING_PATIENCE,
) -> None:
    """
    Production cell execution binding point.

    This function is intentionally a thin orchestration seam over the already
    qualified P1 training, validation, checkpoint, and cell-artifact helpers.
    The CLI invokes this binding through `run_one_cell_wrapper` after all
    frozen cell identity and overwrite checks have passed.
    """
    run_qualified_p1_cell(
        key=key,
        split_manifest_path=split_manifest_path,
        cell_path=cell_path,
        repository_root=repository_root,
        dataset_path=dataset_path,
        t121_config_path=t121_config_path,
        max_updates=max_updates,
        validation_interval_updates=validation_interval_updates,
        patience=patience,
    )


def _validation_arrays(validation_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    from multitask_targets import build_cyp_target_matrix

    targets, mask = build_cyp_target_matrix(validation_df, CYP002_TASK_NAMES)
    y_true = {}
    conf_low = {}
    conf_high = {}
    for task_index, task in enumerate(CYP002_TASK_NAMES):
        low_col = f"{task}_conf_low"
        high_col = f"{task}_conf_high"
        if low_col not in validation_df.columns or high_col not in validation_df.columns:
            raise KeyError(f"Validation confidence interval columns are missing for {task}.")
        present = mask[:, task_index]
        y_true[task] = targets[present, task_index]
        conf_low[task] = validation_df[low_col].to_numpy(dtype=np.float64)[present]
        conf_high[task] = validation_df[high_col].to_numpy(dtype=np.float64)[present]
    return y_true, conf_low, conf_high


def _score_validation(
    *,
    encoder,
    pair_head,
    validation_dataset,
    train_dataset,
    reference_y: torch.Tensor,
    reference_mask: torch.Tensor,
    scaler_scales: torch.Tensor,
    validation_df: pd.DataFrame,
) -> dict[str, Any]:
    with torch.no_grad():
        query_embeddings = encoder(validation_dataset)
        reference_embeddings = encoder(train_dataset)
        predicted_delta_z, _ranges = _all_reference_pair_head(
            pair_head,
            query_embeddings,
            reference_embeddings,
            None,
        )
        from p1_pairwise_model import reconstruct_absolute_from_references

        absolute_mean, reference_dispersion = reconstruct_absolute_from_references(
            predicted_delta_z,
            reference_y.to(predicted_delta_z.device),
            reference_mask.to(predicted_delta_z.device),
            scaler_scales.to(predicted_delta_z.device),
        )
    y_true, conf_low, conf_high = _validation_arrays(validation_df)
    predictions_np = absolute_mean.detach().cpu().numpy()
    y_pred = {
        task: predictions_np[:, task_index][~np.isnan(validation_df[task].to_numpy(dtype=np.float64))]
        for task_index, task in enumerate(CYP002_TASK_NAMES)
    }
    result = compute_validation_macro_st_rae(
        y_true=y_true,
        y_pred=y_pred,
        conf_low=conf_low,
        conf_high=conf_high,
        task_names=CYP002_TASK_NAMES,
    )
    result["reference_dispersion_shape"] = tuple(reference_dispersion.shape)
    return result


def _copy_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def run_qualified_p1_cell(
    *,
    key: P1CellKey,
    split_manifest_path: Path,
    cell_path: Path,
    repository_root: str | Path,
    dataset_path: str | Path,
    t121_config_path: str | Path,
    max_updates: int = P1_MAX_UPDATES,
    validation_interval_updates: int = P1_VALIDATION_INTERVAL_UPDATES,
    patience: int = P1_EARLY_STOPPING_PATIENCE,
) -> Path:
    validate_cell_key(key)
    if max_updates <= 0 or validation_interval_updates <= 0:
        raise ValueError("max_updates and validation_interval_updates must be positive.")

    cell_path = create_p1_cell_artifact_dir(cell_path)
    manifest = load_and_validate_manifest(
        split_manifest_path,
        repository_root=repository_root,
        training_seed=key.training_seed,
    )
    if int(manifest["repetition"]) != key.repetition or int(manifest["fold"]) != key.fold:
        raise ValueError("Resolved split manifest does not match requested P1 cell identity.")

    train_df, validation_query_df, _dataset_provenance = route_d007_manifest_data(
        dataset_path=dataset_path,
        manifest=manifest,
    )
    full_dataset = pd.read_csv(dataset_path)
    by_name = full_dataset.set_index("Molecule_Name", drop=False)
    validation_df = by_name.loc[list(manifest["validation_names"])].copy()

    train_dataset, _scaler, reference_y, reference_mask, scaler_scales, reference_metadata = (
        build_training_reference_dataset(train_df)
    )
    validation_dataset = build_validation_query_dataset(validation_query_df)
    config, config_provenance = load_t121_config(
        t121_config_path=t121_config_path,
        training_seed=key.training_seed,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shared_encoder, aggregation, pair_head, encoder = build_p1_encoder_and_head(
        config,
        device=device,
    )
    optimizer, scheduler = build_p1_optimizer_and_scheduler(
        list(shared_encoder.parameters()) + list(pair_head.parameters())
    )

    targets, _mask = __import__("multitask_targets", fromlist=["build_cyp_target_matrix"]).build_cyp_target_matrix(
        train_df,
        CYP002_TASK_NAMES,
    )
    molecule_ids = train_df["Molecule_Name"].tolist()
    labels = targets.tolist()
    scaler_means = tuple(reference_metadata["scaler_means"])
    scaler_scales_tuple = tuple(reference_metadata["scaler_scales"])
    contract = build_pairwise_contract_from_training_labels(
        molecule_ids=molecule_ids,
        labels=labels,
        scaler_means=scaler_means,
        scaler_scales=scaler_scales_tuple,
    )
    weights = endpoint_weights_tensor(contract)
    molecule_smiles = dict(zip(molecule_ids, train_df["SMILES"].tolist(), strict=True))
    update_step = RealP1TrainingUpdate(
        shared_message_passing=shared_encoder,
        aggregation=aggregation,
        pair_head=pair_head,
        optimizer=optimizer,
        scheduler=scheduler,
        weights=weights,
        molecule_smiles=molecule_smiles,
        device=device,
    )
    controller = P1ValidationController(patience=patience)
    rng = np.random.default_rng(key.training_seed)
    pair_attempts = 0
    accepted_pair_rows = 0
    no_supervision_rejections = 0
    eligible_target_count = [0, 0, 0, 0]
    best_shared_state = None
    best_pair_state = None

    p1_config = {
        "model": "D009-P1",
        "t121_reconstructed_config": config_provenance["reconstructed_config"],
        "selection": "strictly lowest validation macro ST-RAE",
        "validation_interval_updates": P1_VALIDATION_INTERVAL_UPDATES,
        "max_updates": P1_MAX_UPDATES,
        "early_stopping_patience": P1_EARLY_STOPPING_PATIENCE,
    }
    identity = P1CellIdentity(
        repetition=key.repetition,
        fold=key.fold,
        training_seed=key.training_seed,
        split_manifest_path=str(split_manifest_path),
        split_manifest_sha256=sha256_file(split_manifest_path),
        dataset_path=str(dataset_path),
        dataset_sha256=sha256_file(dataset_path),
        t121_config_path=str(t121_config_path),
        t121_config_sha256=sha256_file(t121_config_path),
    )
    planned_manifest = build_p1_cell_manifest(
        cell_identity=identity,
        p1_config=p1_config,
        output_dir=cell_path,
        environment={"device": str(device)},
        provenance={"split_manifest_sha256": sha256_file(split_manifest_path)},
    )
    write_planned_cell_manifest(planned_manifest, cell_path / MANIFEST_NAME)

    update_count = 0
    validation_checks = 0
    while update_count < max_updates:
        batch = draw_p1_training_pair_batch(
            contract=contract,
            molecule_ids=molecule_ids,
            labels=labels,
            scaler_means=scaler_means,
            scaler_scales=scaler_scales_tuple,
            rng=rng,
        )
        metrics = update_step(batch)
        loss = float(metrics["loss"])
        if not np.isfinite(loss):
            raise FloatingPointError("P1 training loss is non-finite.")
        update_count += 1
        pair_attempts += int(batch.attempts)
        accepted_pair_rows += int(batch.accepted_count)
        no_supervision_rejections += int(batch.no_supervision_rejections)
        eligible_target_count = [
            old + int(new)
            for old, new in zip(eligible_target_count, batch.eligible_target_count, strict=True)
        ]

        if update_count % validation_interval_updates == 0:
            validation_checks += 1
            validation_result = _score_validation(
                encoder=encoder,
                pair_head=pair_head,
                validation_dataset=validation_dataset,
                train_dataset=train_dataset,
                reference_y=reference_y,
                reference_mask=reference_mask,
                scaler_scales=scaler_scales,
                validation_df=validation_df,
            )
            improved = controller.observe(
                update=update_count,
                validation_result=validation_result,
                state_dict=pair_head.state_dict(),
            )
            if improved:
                best_shared_state = _copy_module_state(shared_encoder)
                best_pair_state = _copy_module_state(pair_head)
            if controller.stopped:
                break

    if controller.best_score is None or best_shared_state is None or best_pair_state is None:
        raise RuntimeError("P1 cell completed no validation checkpoint selection.")
    shared_encoder.load_state_dict(best_shared_state)
    pair_head.load_state_dict(best_pair_state)
    state = build_checkpoint_state(
        shared_encoder=shared_encoder,
        pair_head=pair_head,
        optimizer=optimizer,
        scheduler=scheduler,
        validation_controller=controller,
        cell_identity=identity,
        p1_config=p1_config,
        reference_metadata={
            **reference_metadata,
            "endpoint_n_t": [endpoint.n_labeled for endpoint in contract.endpoints],
            "endpoint_q_t": [endpoint.q_pair_eligible for endpoint in contract.endpoints],
            "endpoint_w_t": [endpoint.weight for endpoint in contract.endpoints],
        },
        update_state={
            "update_count": update_count,
            "validation_checks": validation_checks,
            "scheduler_last_epoch": scheduler.last_epoch,
        },
        sampler_state={"numpy_bit_generator_state": rng.bit_generator.state},
    )
    save_selected_checkpoint(state, cell_path / BEST_CHECKPOINT_NAME)
    write_metrics(
        {
            "best_validation": controller.history[[record["update"] for record in controller.history].index(controller.best_update)],
            "validation_history": controller.history,
            "pair_accounting": {
                "pair_attempts": pair_attempts,
                "accepted_pair_rows": accepted_pair_rows,
                "no_supervision_rejections": no_supervision_rejections,
                "eligible_target_count": eligible_target_count,
            },
            "test_evaluation_performed": False,
        },
        cell_path / METRICS_NAME,
    )
    finalize_p1_cell_manifest(cell_path)
    if classify_cell(cell_path.parent.parent.parent, key).state != "complete":
        raise RuntimeError("Finalized P1 cell does not classify as complete.")
    return cell_path


def _cell_identity_from_manifest(cell_manifest: Mapping[str, Any]) -> P1CellKey:
    identity = cell_manifest.get("cell_identity", {})
    return P1CellKey(
        repetition=int(identity["repetition"]),
        fold=int(identity["fold"]),
        training_seed=int(identity["training_seed"]),
    )


def _metrics_macro(metrics: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    best = metrics.get("best_validation", metrics)
    per_task = best.get("per_task")
    macro = best.get("macro_st_rae")
    if not isinstance(per_task, dict):
        raise ValueError("Cell metrics are missing per-endpoint ST-RAE values.")
    if macro is None:
        raise ValueError("Cell metrics are missing macro ST-RAE.")
    macro = float(macro)
    if not np.isfinite(macro):
        raise ValueError("Cell macro ST-RAE is non-finite.")
    per_task = {key: float(value) for key, value in per_task.items()}
    if any(not np.isfinite(value) for value in per_task.values()):
        raise ValueError("Cell per-endpoint ST-RAE contains non-finite values.")
    return per_task, macro


def aggregate_completed_p1_cells(cell_root: str | Path) -> dict[str, Any]:
    cell_root = Path(cell_root)
    expected = set(enumerate_p1_cells())
    observed: dict[P1CellKey, dict[str, Any]] = {}
    for key in enumerate_p1_cells():
        state = classify_cell(cell_root, key)
        if state.state != "complete":
            raise ValueError(f"P1 O150 is ineligible; cell is not complete: {key}")
        cell_path = p1_cell_dir(cell_root, key)
        manifest = json.loads((cell_path / MANIFEST_NAME).read_text(encoding="utf-8"))
        manifest_key = _cell_identity_from_manifest(manifest)
        if manifest_key != key:
            raise ValueError("P1 cell manifest identity does not match its path.")
        if manifest_key in observed:
            raise ValueError("Duplicate P1 cell identity encountered.")
        metrics = json.loads((cell_path / METRICS_NAME).read_text(encoding="utf-8"))
        per_task, macro = _metrics_macro(metrics)
        observed[manifest_key] = {
            "cell": asdict(manifest_key),
            "per_endpoint_st_rae": per_task,
            "macro_st_rae": macro,
        }
    if set(observed) != expected or len(observed) != EXPECTED_CELL_COUNT:
        raise ValueError("P1 O150 requires exactly one complete artifact for each frozen cell.")
    macros = [observed[key]["macro_st_rae"] for key in sorted(observed)]
    o150 = float(np.mean(np.asarray(macros, dtype=np.float64)))
    return {
        "cell_count": len(observed),
        "cells": [observed[key] for key in sorted(observed)],
        "o150": o150,
        "t121_o150": T121_O150,
        "strict_advancement_rule": "P1 advances iff O150(P1) < 0.701212576796",
        "p1_advances": bool(o150 < T121_O150),
        "endpoint_veto": False,
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D009 P1 production orchestration CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--production-root", required=True, type=Path)
    preflight.add_argument("--repository-root", required=True, type=Path)
    preflight.add_argument("--dataset-path", required=True, type=Path)
    preflight.add_argument("--split-manifest-dir", required=True, type=Path)
    preflight.add_argument("--t121-config-artifact", required=True, type=Path)
    preflight.add_argument("--output-json", required=True, type=Path)

    run_cell = subparsers.add_parser("run-cell")
    run_cell.add_argument("--repetition", required=True, type=int)
    run_cell.add_argument("--fold", required=True, type=int)
    run_cell.add_argument("--training-seed", required=True, type=int)
    run_cell.add_argument("--production-root", required=True, type=Path)
    run_cell.add_argument("--repository-root", required=True, type=Path)
    run_cell.add_argument("--dataset-path", required=True, type=Path)
    run_cell.add_argument("--split-manifest-dir", required=True, type=Path)
    run_cell.add_argument("--t121-config-artifact", required=True, type=Path)
    run_cell.add_argument("--output-json", required=True, type=Path)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--production-root", required=True, type=Path)
    aggregate.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "preflight":
        report = preflight_p1_production(
            production_root=args.production_root,
            repository_root=args.repository_root,
            split_manifest_dir=args.split_manifest_dir,
            dataset_path=args.dataset_path,
            t121_config_path=args.t121_config_artifact,
        )
        _write_json(args.output_json, report)
        return 0

    if args.command == "run-cell":
        key = P1CellKey(args.repetition, args.fold, args.training_seed)

        def runner(cell_key: P1CellKey, split_manifest_path: Path, cell_path: Path) -> None:
            execute_qualified_p1_cell(
                key=cell_key,
                split_manifest_path=split_manifest_path,
                cell_path=cell_path,
                repository_root=args.repository_root,
                dataset_path=args.dataset_path,
                t121_config_path=args.t121_config_artifact,
            )

        cell_path = run_one_cell_wrapper(
            production_root=args.production_root,
            split_manifest_dir=args.split_manifest_dir,
            key=key,
            cell_runner=runner,
        )
        report = {
            "schema": P1_PRODUCTION_SCHEMA,
            "command": "run-cell",
            "cell": asdict(key),
            "cell_path": str(cell_path),
            "state": asdict(classify_cell(args.production_root, key)),
        }
        _write_json(args.output_json, report)
        return 0

    if args.command == "aggregate":
        report = aggregate_completed_p1_cells(args.production_root)
        _write_json(args.output_json, report)
        return 0

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
