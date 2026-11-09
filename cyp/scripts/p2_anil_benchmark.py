from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import socket
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

from cyp002_config import CYP002_TASK_NAMES
from multitask_targets import build_cyp_target_matrix
from p1_pairwise_real_workload_benchmark import (
    build_training_reference_dataset,
    build_validation_query_dataset,
    load_and_validate_manifest,
    load_t121_config,
    route_d007_manifest_data,
)
from p2_anil_model import (
    build_p2_aggregation,
    build_p2_message_passing,
    build_p2_meta_head,
    count_parameters,
    query_loss_after_exact_adaptation,
)
from p2_anil_training import (
    P2_MAX_UPDATES,
    P2_MAX_VALIDATION_OPPORTUNITIES,
    build_p2_optimizer_and_scheduler,
    draw_p2_episodes,
    fit_training_only_scaler,
    full_support_task_adaptation_prediction,
)


ARTIFACT_TYPE = "D010_P2_exact_anil_bounded_real_workload_benchmark_v1"
COMPLETE_MODEL_ARTIFACT_TYPE = "D010_P2_exact_anil_complete_model_bounded_benchmark_v2"
EXPECTED_COMPLETE_MODEL_PARAMETERS = 2025404
NO_PERFORMANCE_STATEMENT = (
    "Benchmark may use validation molecule inputs only for prediction-cost timing. "
    "It does not access validation labels, calculate ST-RAE, select checkpoints, "
    "early-stop on performance, access protected test, or access D008 evidence."
)


def _environment(device: torch.device) -> dict[str, Any]:
    result = {
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "visible_gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = 0 if device.index is None else device.index
        result["device_index"] = index
        result["gpu_model"] = torch.cuda.get_device_name(index)
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _subset_dataset(source_dataset, names: list[str], name_to_index: dict[str, int]):
    from chemprop import data

    return data.MoleculeDataset([source_dataset.data[name_to_index[name]] for name in names])


def _encode_dataset(*, message_passing, aggregation, dataset, device: torch.device, batch_size: int) -> torch.Tensor:
    from chemprop import data

    loader = data.build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    pieces = []
    for batch in loader:
        batch_mol_graph = batch[0]
        moved = batch_mol_graph.to(device)
        if moved is not None:
            batch_mol_graph = moved
        atom_embeddings = message_passing(batch_mol_graph)
        pieces.append(aggregation(atom_embeddings, batch_mol_graph.batch))
    return torch.cat(pieces, dim=0)


def _complete_model_parameter_counts(message_passing, aggregation, meta_head) -> dict[str, dict[str, int]]:
    encoder = count_parameters(message_passing).__dict__
    aggregation_counts = count_parameters(aggregation).__dict__
    shared_hidden = count_parameters(meta_head.shared_hidden).__dict__
    output = count_parameters(meta_head.output).__dict__
    meta_head_counts = count_parameters(meta_head).__dict__
    complete_total = encoder["total"] + aggregation_counts["total"] + meta_head_counts["total"]
    complete_trainable = encoder["trainable"] + aggregation_counts["trainable"] + meta_head_counts["trainable"]
    if complete_total != EXPECTED_COMPLETE_MODEL_PARAMETERS:
        raise ValueError(
            "D010 complete model parameter count mismatch: "
            f"{complete_total} != {EXPECTED_COMPLETE_MODEL_PARAMETERS}."
        )
    if complete_trainable != EXPECTED_COMPLETE_MODEL_PARAMETERS:
        raise ValueError(
            "D010 complete model trainable parameter count mismatch: "
            f"{complete_trainable} != {EXPECTED_COMPLETE_MODEL_PARAMETERS}."
        )
    return {
        "encoder": encoder,
        "aggregation": aggregation_counts,
        "shared_hidden": shared_hidden,
        "output_layer": output,
        "meta_head": meta_head_counts,
        "complete_p2_model": {"total": complete_total, "trainable": complete_trainable},
    }


def p2_complete_model_outer_update_step(
    *,
    message_passing,
    aggregation,
    meta_head,
    optimizer: torch.optim.Optimizer,
    scheduler,
    episodes,
    train_dataset,
    name_to_index: dict[str, int],
    device: torch.device,
    batch_size: int,
):
    message_passing.train()
    aggregation.train()
    meta_head.train()
    optimizer.zero_grad(set_to_none=True)
    query_losses = []
    support_losses = []
    for episode in episodes:
        support_dataset = _subset_dataset(train_dataset, list(episode.support_molecule_ids), name_to_index)
        query_dataset = _subset_dataset(train_dataset, list(episode.query_molecule_ids), name_to_index)
        support_embeddings = _encode_dataset(
            message_passing=message_passing,
            aggregation=aggregation,
            dataset=support_dataset,
            device=device,
            batch_size=batch_size,
        )
        query_embeddings = _encode_dataset(
            message_passing=message_passing,
            aggregation=aggregation,
            dataset=query_dataset,
            device=device,
            batch_size=batch_size,
        )
        support_targets = torch.tensor(
            episode.support_targets_z,
            dtype=support_embeddings.dtype,
            device=device,
        )
        query_targets = torch.tensor(
            episode.query_targets_z,
            dtype=query_embeddings.dtype,
            device=device,
        )
        query_loss, support_loss = query_loss_after_exact_adaptation(
            meta_head,
            support_embeddings,
            support_targets,
            query_embeddings,
            query_targets,
            task_index=episode.task_index,
        )
        query_losses.append(query_loss)
        support_losses.append(support_loss)
    outer_loss = sum(query_losses) / 4
    if not torch.isfinite(outer_loss):
        raise FloatingPointError("D010 complete-model outer loss is non-finite.")
    outer_loss.backward()
    for label, module in (("encoder", message_passing), ("meta_head", meta_head)):
        for parameter in module.parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    raise RuntimeError(f"D010 {label} parameter is missing exact meta-gradient.")
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"D010 {label} exact meta-gradient is non-finite.")
    optimizer.step()
    scheduler.step()
    return {
        "outer_loss": outer_loss,
        "query_losses": tuple(query_losses),
        "support_losses": tuple(support_losses),
    }


def _validation_cost_complete_model(
    *,
    message_passing,
    aggregation,
    meta_head,
    train_dataset,
    validation_dataset,
    name_to_index: dict[str, int],
    molecule_ids: list[str],
    targets,
    mask,
    scaler_means,
    scaler_scales,
    device: torch.device,
    batch_size: int,
) -> tuple[float, list[int]]:
    message_passing.eval()
    aggregation.eval()
    meta_head.eval()
    train_labels = torch.tensor(targets, dtype=torch.float32, device=device)
    train_mask = torch.tensor(mask, dtype=torch.bool, device=device)
    means_t = torch.tensor(scaler_means, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scaler_scales, dtype=torch.float32, device=device)
    validation_counts = []
    _sync(device)
    start = time.perf_counter()
    for task_index in range(4):
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
        eligible_indices = [name_to_index[name] for name in eligible_names]
        full_train_embeddings[eligible_indices] = support_embeddings
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
        )
        validation_counts.append(int(pred.numel()))
    _sync(device)
    return time.perf_counter() - start, validation_counts


def run_p2_exact_anil_benchmark(
    *,
    repository_root: str | Path,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    t121_config_path: str | Path,
    training_seed: int,
    warmup_update_count: int,
    measured_update_count: int,
    output_json: str | Path | None = None,
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    if warmup_update_count < 0 or measured_update_count <= 0:
        raise ValueError("warmup_update_count must be >=0 and measured_update_count must be >0.")
    benchmark_device = torch.device(device)
    if benchmark_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is not available.")
    manifest = load_and_validate_manifest(split_manifest_path, repository_root=repository_root, training_seed=training_seed)
    train_df, validation_query_df, dataset_provenance = route_d007_manifest_data(dataset_path=dataset_path, manifest=manifest)
    train_dataset, _scaler, _reference_y, _reference_mask, _scaler_scales, reference_metadata = build_training_reference_dataset(train_df)
    validation_dataset = build_validation_query_dataset(validation_query_df)
    config, config_provenance = load_t121_config(t121_config_path=t121_config_path, training_seed=training_seed)
    message_passing = build_p2_message_passing(config).to(benchmark_device)
    aggregation = build_p2_aggregation(config).to(benchmark_device)
    meta_head = build_p2_meta_head(config).to(benchmark_device)
    optimizer, scheduler = build_p2_optimizer_and_scheduler(
        list(message_passing.parameters()) + list(meta_head.parameters())
    )
    parameter_counts = _complete_model_parameter_counts(message_passing, aggregation, meta_head)
    targets, mask = build_cyp_target_matrix(train_df, CYP002_TASK_NAMES)
    scaler_means, scaler_scales = fit_training_only_scaler(targets, mask)
    molecule_ids = train_df["Molecule_Name"].tolist()
    name_to_index = {name: index for index, name in enumerate(molecule_ids)}
    rng = np.random.default_rng(training_seed)
    if benchmark_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(benchmark_device)
    timings = []
    losses = []
    total = warmup_update_count + measured_update_count
    for update_index in range(total):
        _sync(benchmark_device)
        start = time.perf_counter()
        episodes = draw_p2_episodes(
            molecule_ids=molecule_ids,
            labels=targets,
            mask=mask,
            scaler_means=scaler_means,
            scaler_scales=scaler_scales,
            rng=rng,
        )
        result = p2_complete_model_outer_update_step(
            message_passing=message_passing,
            aggregation=aggregation,
            meta_head=meta_head,
            optimizer=optimizer,
            scheduler=scheduler,
            episodes=episodes,
            train_dataset=train_dataset,
            name_to_index=name_to_index,
            device=benchmark_device,
            batch_size=config.batch_size,
        )
        _sync(benchmark_device)
        elapsed = time.perf_counter() - start
        if update_index >= warmup_update_count:
            timings.append(elapsed)
            losses.append(float(result["outer_loss"].detach().cpu()))
    validation_seconds, validation_counts = _validation_cost_complete_model(
        message_passing=message_passing,
        aggregation=aggregation,
        meta_head=meta_head,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        name_to_index=name_to_index,
        molecule_ids=molecule_ids,
        targets=targets,
        mask=mask,
        scaler_means=scaler_means,
        scaler_scales=scaler_scales,
        device=benchmark_device,
        batch_size=config.batch_size,
    )
    measured_total = sum(timings)
    seconds_per_update = measured_total / measured_update_count
    training_per_cell = seconds_per_update * P2_MAX_UPDATES
    validation_per_cell = validation_seconds * P2_MAX_VALIDATION_OPPORTUNITIES
    artifact = {
        "artifact_type": COMPLETE_MODEL_ARTIFACT_TYPE,
        "no_validation_label_or_performance_access_statement": NO_PERFORMANCE_STATEMENT,
        "input_paths": {
            "repository_root": str(Path(repository_root).resolve()),
            "authoritative_cyp_dataset": str(dataset_path),
            "d007_split_manifest": str(split_manifest_path),
            "t121_config_artifact": str(t121_config_path),
        },
        "input_hashes": {
            "authoritative_cyp_dataset_sha256": dataset_provenance["dataset_sha256"],
            "d007_split_manifest_sha256": manifest["_manifest_sha256"],
            "d007_source_split_assignments_sha256": dataset_provenance["source_split_assignments_sha256"],
            "t121_config_artifact_sha256": config_provenance["sha256"],
        },
        "split": {"repetition": manifest["repetition"], "fold": manifest["fold"], "training_seed": training_seed},
        "environment": _environment(benchmark_device),
        "workload": {
            "training_molecule_count": len(train_df),
            "validation_input_molecule_count": len(validation_query_df),
            "endpoint_order": list(CYP002_TASK_NAMES),
            "episode_accounting_per_outer_update": {"task_episodes": 4, "support_per_task": 13, "query_per_task": 3},
            "training_only_scaler_means": scaler_means.tolist(),
            "training_only_scaler_scales": scaler_scales.tolist(),
            "endpoint_eligible_training_reference_counts": reference_metadata["endpoint_eligible_reference_counts"],
        },
        "configuration": {"t121_config_provenance": config_provenance, "inner_lr": 0.01},
        "parameter_counts": parameter_counts,
        "training_timing": {
            "warmup_update_count": warmup_update_count,
            "measured_update_count": measured_update_count,
            "total_optimizer_updates": total,
            "aggregate_measured_duration_seconds": measured_total,
            "seconds_per_complete_four_task_outer_update": seconds_per_update,
            "per_update_synchronized_seconds": timings,
            "summary": _summarize(timings),
            "early_window_seconds": _summarize(timings[: max(1, len(timings) // 3)]),
            "late_window_seconds": _summarize(timings[-max(1, len(timings) // 3):]),
            "descriptive_drift_seconds_late_minus_early_mean": (_summarize(timings[-max(1, len(timings) // 3):])["mean"] - _summarize(timings[: max(1, len(timings) // 3)])["mean"]),
            "outer_losses": losses,
            "current_scheduler_lr": optimizer.param_groups[0]["lr"],
            "timing_includes": ["episode sampling/index selection", "graph dataset subset construction", "graph batching/collation", "GPU transfer", "support encoder/aggregation/head forward", "differentiable support gradient", "one-step task-output adaptation", "query encoder/aggregation/head forward", "exact through-adaptation meta-backward", "four-task equal accumulation", "Adam step", "NoamLR step"],
        },
        "validation_cost_timing": {
            "seconds_for_four_isoform_full_support_adaptation_and_validation_input_prediction": validation_seconds,
            "prediction_counts_by_isoform": validation_counts,
            "timing_includes": ["fresh unadapted state per isoform", "all eligible training-support molecular graphs through encoder/aggregation/head", "one full-support inner step", "validation molecule graphs through encoder/aggregation/adapted head"],
        },
        "memory": {
            "peak_allocated_gpu_memory_bytes": torch.cuda.max_memory_allocated(benchmark_device) if benchmark_device.type == "cuda" else None,
            "peak_reserved_gpu_memory_bytes": torch.cuda.max_memory_reserved(benchmark_device) if benchmark_device.type == "cuda" else None,
        },
        "projections": {
            "training_seconds_per_cell_11750_updates": training_per_cell,
            "validation_seconds_per_cell_up_to_50_checks": validation_per_cell,
            "combined_measured_component_seconds_per_cell": training_per_cell + validation_per_cell,
            "combined_measured_component_seconds_150_cells": (training_per_cell + validation_per_cell) * 150,
            "subtotal_is_not_complete_experiment_runtime_or_upper_bound": True,
        },
        "measured_costs": ["complete exact-ANIL outer updates with real molecular graph encoder", "four-isoform full-support adaptation plus validation-input graph prediction"],
        "excluded_costs": ["one-time dataset loading", "SMILES parsing", "static Chemprop graph featurization/cache construction", "checkpoint loading", "checkpoint serialization", "artifact writing", "validation scoring", "ST-RAE", "checkpoint selection", "early stopping", "caller-side orchestration overhead"],
    }
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run D010 P2 bounded exact-ANIL real-workload benchmark.")
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--split-manifest-path", required=True, type=Path)
    parser.add_argument("--t121-config-artifact-path", required=True, type=Path)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--warmup-update-count", required=True, type=int)
    parser.add_argument("--measured-update-count", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_p2_exact_anil_benchmark(
        repository_root=args.repository_root,
        dataset_path=args.dataset_path,
        split_manifest_path=args.split_manifest_path,
        t121_config_path=args.t121_config_artifact_path,
        training_seed=args.training_seed,
        warmup_update_count=args.warmup_update_count,
        measured_update_count=args.measured_update_count,
        output_json=args.output_json,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
