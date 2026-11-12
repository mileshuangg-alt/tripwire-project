from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, Callable

from chemprop import data
import numpy as np
import torch

from cyp002_config import CYP002_TASK_NAMES
from cyp007_infrastructure import sha256_file
from multitask_targets import build_cyp_target_matrix
from p1_pairwise_training import (
    P1_ACCEPTED_PAIR_BATCH_SIZE,
    P1_MAX_UPDATES,
    build_p1_optimizer_and_scheduler,
    build_pairwise_contract_from_training_labels,
    draw_p1_training_pair_batch,
    endpoint_weights_tensor,
    p1_forward_loss_backward_step,
)
from p1_pairwise_chemprop import count_parameters
from p1_pairwise_real_workload_benchmark import (
    build_p1_encoder_and_head,
    build_training_reference_dataset,
    load_and_validate_manifest,
    load_t121_config,
)


ARTIFACT_TYPE = "D009_P1_training_runtime_benchmark_v1"
STATEMENT = (
    "Representative compute-cost extrapolation only: bounded real-training-"
    "partition optimizer updates on the D009 P1 training path. This artifact "
    "has no validation-performance, checkpoint-selection, model-selection, "
    "stopping, calibration, or scientific advancement authority."
)
NO_VALIDATION_STATEMENT = (
    "This runtime benchmark does not construct validation queries, access "
    "validation labels, compute ST-RAE, select checkpoints, apply early "
    "stopping, or access protected-test, D008, or challenge evidence."
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _require_unique_names(frame, *, label: str) -> None:
    if "Molecule_Name" not in frame.columns:
        raise ValueError(f"{label} is missing Molecule_Name.")
    if frame["Molecule_Name"].duplicated().any():
        raise ValueError(f"{label} contains duplicate Molecule_Name values.")


def _route_training_partition_only(*, dataset_path: str | Path, manifest: dict[str, Any]):
    import pandas as pd

    dataset_path = Path(dataset_path)
    dataset = pd.read_csv(dataset_path)
    _require_unique_names(dataset, label="authoritative CYP dataset")
    if "SMILES" not in dataset.columns:
        raise ValueError("authoritative CYP dataset is missing SMILES.")

    train_names = list(manifest["training_names"])
    if len(train_names) != len(set(train_names)):
        raise ValueError("D007 manifest contains duplicate training names.")

    dataset_names = set(dataset["Molecule_Name"])
    missing = sorted(set(train_names) - dataset_names)
    if missing:
        raise ValueError(f"D007 training names are missing from dataset: {missing[:5]}")

    by_name = dataset.set_index("Molecule_Name", drop=False)
    train_df = by_name.loc[train_names].copy()
    return train_df


def _move_batch_mol_graph(batch_mol_graph, device: torch.device):
    moved = batch_mol_graph.to(device)
    return batch_mol_graph if moved is None else moved


def _molecule_dataset_from_smiles(smiles: list[str]) -> data.MoleculeDataset:
    return data.MoleculeDataset([
        data.MoleculeDatapoint.from_smi(smiles_value)
        for smiles_value in smiles
    ])


def _single_batch_mol_graph(dataset: data.MoleculeDataset, *, device: torch.device):
    loader = data.build_dataloader(
        dataset,
        batch_size=len(dataset),
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    return _move_batch_mol_graph(batch[0], device)


class RealP1TrainingUpdate:
    def __init__(
        self,
        *,
        shared_message_passing,
        aggregation,
        pair_head,
        optimizer,
        scheduler,
        weights,
        molecule_smiles: dict[str, str],
        device: torch.device,
    ) -> None:
        self.shared_message_passing = shared_message_passing
        self.aggregation = aggregation
        self.pair_head = pair_head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weights = weights
        self.molecule_smiles = molecule_smiles
        self.device = device

    def __call__(self, batch) -> dict[str, float]:
        smiles_a = [self.molecule_smiles[name] for name in batch.molecule_a_ids]
        smiles_b = [self.molecule_smiles[name] for name in batch.molecule_b_ids]
        dataset_a = _molecule_dataset_from_smiles(smiles_a)
        dataset_b = _molecule_dataset_from_smiles(smiles_b)
        bmg_a = _single_batch_mol_graph(dataset_a, device=self.device)
        bmg_b = _single_batch_mol_graph(dataset_b, device=self.device)

        self.shared_message_passing.train()
        self.aggregation.train()
        atom_embeddings = self.shared_message_passing([bmg_a, bmg_b])
        h_a = self.aggregation(atom_embeddings[0], bmg_a.batch)
        h_b = self.aggregation(atom_embeddings[1], bmg_b.batch)
        targets = torch.tensor(
            batch.delta_targets,
            dtype=h_a.dtype,
            device=self.device,
        )
        mask = torch.tensor(
            batch.eligibility_masks,
            dtype=torch.bool,
            device=self.device,
        )
        loss, _predictions = p1_forward_loss_backward_step(
            pair_head=self.pair_head,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            h_a=h_a,
            h_b=h_b,
            targets=targets,
            mask=mask,
            weights=self.weights.to(device=self.device, dtype=h_a.dtype),
        )
        return {"loss": float(loss.item())}


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


def _sync_if_cuda(device: torch.device, cuda_backend) -> None:
    if device.type == "cuda":
        cuda_backend.synchronize(device)


def _reset_peak_if_cuda(device: torch.device, cuda_backend) -> None:
    if device.type == "cuda":
        cuda_backend.reset_peak_memory_stats(device)


def _peak_allocated(device: torch.device, cuda_backend) -> int | None:
    if device.type != "cuda":
        return None
    return int(cuda_backend.max_memory_allocated(device))


def _peak_reserved(device: torch.device, cuda_backend) -> int | None:
    if device.type != "cuda":
        return None
    return int(cuda_backend.max_memory_reserved(device))


def _accumulate_accounting(totals: dict[str, Any], batch) -> None:
    totals["updates"] += 1
    totals["pair_attempts"] += int(batch.attempts)
    totals["accepted_pair_rows"] += int(batch.accepted_count)
    totals["no_supervision_rejections"] += int(batch.no_supervision_rejections)
    totals["eligible_target_count"] = [
        old + int(new)
        for old, new in zip(totals["eligible_target_count"], batch.eligible_target_count, strict=True)
    ]


def _run_updates(
    *,
    update_count: int,
    contract,
    molecule_ids,
    labels,
    scaler_means,
    scaler_scales,
    rng,
    update_step: Callable[[Any], Mapping[str, float]],
    device: torch.device | None = None,
    cuda_backend=None,
    record_update_durations: bool = False,
) -> dict[str, Any]:
    totals = {
        "updates": 0,
        "pair_attempts": 0,
        "accepted_pair_rows": 0,
        "no_supervision_rejections": 0,
        "eligible_target_count": [0, 0, 0, 0],
        "losses": [],
        "update_durations_seconds": [],
    }
    for _ in range(update_count):
        batch = draw_p1_training_pair_batch(
            contract=contract,
            molecule_ids=molecule_ids,
            labels=labels,
            scaler_means=scaler_means,
            scaler_scales=scaler_scales,
            rng=rng,
            accepted_pair_count=P1_ACCEPTED_PAIR_BATCH_SIZE,
        )
        if record_update_durations:
            if device is None or cuda_backend is None:
                raise ValueError(
                    "device and cuda_backend are required when recording update durations."
                )
            _sync_if_cuda(device, cuda_backend)
            update_start = time.perf_counter()
        metrics = update_step(batch)
        if record_update_durations:
            _sync_if_cuda(device, cuda_backend)
            update_elapsed = time.perf_counter() - update_start
            if update_elapsed <= 0.0 or not np.isfinite(update_elapsed):
                raise FloatingPointError(
                    "per-update elapsed time must be finite and positive."
                )
            totals["update_durations_seconds"].append(float(update_elapsed))
        loss = float(metrics.get("loss", 0.0))
        if not np.isfinite(loss):
            raise FloatingPointError("training update loss is non-finite.")
        totals["losses"].append(loss)
        _accumulate_accounting(totals, batch)
    return totals


def _drift_statistics(update_durations: Sequence[float]) -> dict[str, float | None]:
    values = np.asarray(update_durations, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("update_durations must contain at least one value.")
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("update_durations must be finite and positive.")

    first_count = min(25, values.size)
    last_count = min(25, values.size)
    first_mean = float(values[:first_count].mean())
    last_mean = float(values[-last_count:].mean())
    absolute = last_mean - first_mean
    relative = absolute / first_mean if first_mean != 0.0 else None
    return {
        "mean_seconds_per_update": float(values.mean()),
        "median_seconds_per_update": float(np.median(values)),
        "first_25_mean_seconds_per_update": first_mean,
        "last_25_mean_seconds_per_update": last_mean,
        "last_minus_first_25_mean_seconds_per_update": float(absolute),
        "relative_last_vs_first_25_mean_difference": (
            None if relative is None else float(relative)
        ),
    }


def run_training_runtime_benchmark(
    *,
    repository_root: str | Path,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    t121_config_path: str | Path,
    training_seed: int,
    measured_update_count: int,
    warmup_update_count: int = 1,
    output_json: str | Path | None = None,
    device: str | torch.device = "cuda:0",
    require_cuda: bool = True,
    cuda_backend=torch.cuda,
    update_step_factory: Callable[..., Callable[[Any], Mapping[str, float]]] | None = None,
) -> dict[str, Any]:
    if measured_update_count <= 0:
        raise ValueError("measured_update_count must be positive.")
    if warmup_update_count < 0:
        raise ValueError("warmup_update_count must be nonnegative.")

    benchmark_device = torch.device(device)
    if require_cuda:
        if not cuda_backend.is_available():
            raise RuntimeError("CUDA is required for the P1 training runtime benchmark.")
        if int(cuda_backend.device_count()) != 1:
            raise RuntimeError("Exactly one visible GPU is required.")
        cuda_backend.set_device(benchmark_device)

    manifest = load_and_validate_manifest(
        split_manifest_path,
        repository_root=repository_root,
        training_seed=training_seed,
    )
    train_df = _route_training_partition_only(
        dataset_path=dataset_path,
        manifest=manifest,
    )
    train_dataset, _scaler, _reference_y, _reference_mask, _scaler_scales_tensor, reference_metadata = (
        build_training_reference_dataset(train_df)
    )
    del train_dataset

    config, config_provenance = load_t121_config(
        t121_config_path=t121_config_path,
        training_seed=training_seed,
    )
    shared_message_passing, aggregation, pair_head, _encoder = build_p1_encoder_and_head(
        config,
        device=benchmark_device,
    )
    optimizer, scheduler = build_p1_optimizer_and_scheduler(
        list(shared_message_passing.parameters()) + list(pair_head.parameters())
    )

    targets, _mask = build_cyp_target_matrix(train_df, CYP002_TASK_NAMES)
    molecule_ids = train_df["Molecule_Name"].tolist()
    labels = targets.tolist()
    scaler_means = tuple(reference_metadata["scaler_means"])
    scaler_scales = tuple(reference_metadata["scaler_scales"])
    contract = build_pairwise_contract_from_training_labels(
        molecule_ids=molecule_ids,
        labels=labels,
        scaler_means=scaler_means,
        scaler_scales=scaler_scales,
    )
    weights = endpoint_weights_tensor(contract)
    molecule_smiles = dict(zip(molecule_ids, train_df["SMILES"].tolist(), strict=True))
    rng = np.random.default_rng(training_seed)

    if update_step_factory is None:
        update_step = RealP1TrainingUpdate(
            shared_message_passing=shared_message_passing,
            aggregation=aggregation,
            pair_head=pair_head,
            optimizer=optimizer,
            scheduler=scheduler,
            weights=weights,
            molecule_smiles=molecule_smiles,
            device=benchmark_device,
        )
    else:
        update_step = update_step_factory(
            shared_message_passing=shared_message_passing,
            aggregation=aggregation,
            pair_head=pair_head,
            optimizer=optimizer,
            scheduler=scheduler,
            weights=weights,
            molecule_smiles=molecule_smiles,
            device=benchmark_device,
        )

    warmup_accounting = _run_updates(
        update_count=warmup_update_count,
        contract=contract,
        molecule_ids=molecule_ids,
        labels=labels,
        scaler_means=scaler_means,
        scaler_scales=scaler_scales,
        rng=rng,
        update_step=update_step,
        record_update_durations=False,
    ) if warmup_update_count else {
        "updates": 0,
        "pair_attempts": 0,
        "accepted_pair_rows": 0,
        "no_supervision_rejections": 0,
        "eligible_target_count": [0, 0, 0, 0],
        "losses": [],
        "update_durations_seconds": [],
    }

    start_lr = float(optimizer.param_groups[0]["lr"])
    start_scheduler_position = int(getattr(scheduler, "last_epoch", 0))

    _reset_peak_if_cuda(benchmark_device, cuda_backend)
    _sync_if_cuda(benchmark_device, cuda_backend)
    start = time.perf_counter()
    measured_accounting = _run_updates(
        update_count=measured_update_count,
        contract=contract,
        molecule_ids=molecule_ids,
        labels=labels,
        scaler_means=scaler_means,
        scaler_scales=scaler_scales,
        rng=rng,
        update_step=update_step,
        device=benchmark_device,
        cuda_backend=cuda_backend,
        record_update_durations=True,
    )
    _sync_if_cuda(benchmark_device, cuda_backend)
    elapsed_seconds = time.perf_counter() - start
    if elapsed_seconds <= 0.0 or not np.isfinite(elapsed_seconds):
        raise FloatingPointError("measured elapsed time must be finite and positive.")

    end_lr = float(optimizer.param_groups[0]["lr"])
    end_scheduler_position = int(getattr(scheduler, "last_epoch", 0))
    seconds_per_update = elapsed_seconds / measured_update_count
    projected_seconds = seconds_per_update * P1_MAX_UPDATES * 150
    drift_statistics = _drift_statistics(
        measured_accounting["update_durations_seconds"]
    )

    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "benchmark_authority_statement": STATEMENT,
        "no_validation_or_performance_statement": NO_VALIDATION_STATEMENT,
        "input_paths": {
            "repository_root": str(Path(repository_root).resolve()),
            "dataset_path": str(dataset_path),
            "split_manifest_path": str(split_manifest_path),
            "t121_config_path": str(t121_config_path),
        },
        "input_hashes": {
            "dataset_sha256": sha256_file(dataset_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "source_split_assignments_sha256": manifest["_observed_source_split_assignments_sha256"],
            "t121_config_sha256": config_provenance["sha256"],
        },
        "environment": _environment(benchmark_device),
        "split": {
            "repetition": manifest["repetition"],
            "fold": manifest["fold"],
            "training_seed": training_seed,
        },
        "configuration": {
            "t121_sampled_config": config_provenance["sampled_config"],
            "dtype": "float32",
            "accepted_pair_batch_size": P1_ACCEPTED_PAIR_BATCH_SIZE,
            "measured_update_count": measured_update_count,
            "warmup_update_count": warmup_update_count,
        },
        "training_partition": {
            "training_molecule_count": len(train_df),
            "endpoint_order": list(CYP002_TASK_NAMES),
            "scaler_means": list(scaler_means),
            "scaler_scales": list(scaler_scales),
            "n_t": [endpoint.n_labeled for endpoint in contract.endpoints],
            "q_t": [endpoint.q_pair_eligible for endpoint in contract.endpoints],
            "w_t": [endpoint.weight for endpoint in contract.endpoints],
        },
        "parameter_counts": {
            "shared_encoder": count_parameters(shared_message_passing).__dict__,
            "pair_head": count_parameters(pair_head).__dict__,
        },
        "timing": {
            "warmup_updates_excluded": warmup_update_count,
            "measured_updates": measured_update_count,
            "total_measured_seconds": elapsed_seconds,
            "per_update_durations_seconds": measured_accounting[
                "update_durations_seconds"
            ],
            "seconds_per_update": seconds_per_update,
            "updates_per_second": measured_update_count / elapsed_seconds,
            "pair_rows_per_second": (
                measured_update_count * P1_ACCEPTED_PAIR_BATCH_SIZE / elapsed_seconds
            ),
            "peak_allocated_gpu_memory_bytes": _peak_allocated(benchmark_device, cuda_backend),
            "peak_reserved_gpu_memory_bytes": _peak_reserved(benchmark_device, cuda_backend),
            "drift_statistics": drift_statistics,
        },
        "sampler_accounting": {
            "warmup": warmup_accounting,
            "measured": measured_accounting,
        },
        "scheduler": {
            "start_lr": start_lr,
            "end_lr": end_lr,
            "start_scheduler_position": start_scheduler_position,
            "end_scheduler_position": end_scheduler_position,
        },
        "projection": {
            "label": "representative-cost extrapolation, not a measured full experiment or proven runtime upper bound",
            "seconds_per_update": seconds_per_update,
            "updates_per_cell": P1_MAX_UPDATES,
            "cell_count": 150,
            "projected_training_seconds": projected_seconds,
            "projected_training_gpu_hours": projected_seconds / 3600.0,
        },
    }

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded D009 P1 real-data training-runtime benchmark."
    )
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--split-manifest-path", required=True, type=Path)
    parser.add_argument("--t121-config-artifact-path", required=True, type=Path)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--measured-update-count", required=True, type=_positive_int)
    parser.add_argument("--warmup-update-count", default=1, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training_runtime_benchmark(
        repository_root=args.repository_root,
        dataset_path=args.dataset_path,
        split_manifest_path=args.split_manifest_path,
        t121_config_path=args.t121_config_artifact_path,
        training_seed=args.training_seed,
        measured_update_count=args.measured_update_count,
        warmup_update_count=args.warmup_update_count,
        output_json=args.output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
