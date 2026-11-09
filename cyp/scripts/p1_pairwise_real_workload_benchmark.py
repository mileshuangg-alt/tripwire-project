from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import socket
import sys
from typing import Any

from chemprop import data, nn
import numpy as np
import pandas as pd
import torch

from cyp002_config import CYP002_TASK_NAMES
from cyp007_infrastructure import (
    D007_TRAINING_SEEDS,
    config_from_trial,
    sha256_file,
)
from multitask_targets import build_cyp_target_matrix
from p1_pairwise_benchmark import run_p1_all_reference_benchmark
from p1_pairwise_chemprop import (
    build_p1_pair_head,
    build_p1_shared_message_passing,
    count_parameters,
)


ARTIFACT_TYPE = "D009_P1_real_workload_validation_cost_benchmark_v1"
MAX_PROSPECTIVE_VALIDATION_EXECUTIONS = 150 * 50
NO_VALIDATION_LABEL_STATEMENT = (
    "No validation labels, confidence intervals, ST-RAE inputs, or performance "
    "metrics are accessed by the benchmark path. Validation rows are reduced "
    "to molecule identity and SMILES before query featurization."
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_unique_names(frame: pd.DataFrame, *, label: str) -> None:
    if "Molecule_Name" not in frame.columns:
        raise ValueError(f"{label} is missing Molecule_Name.")
    if frame["Molecule_Name"].isna().any():
        raise ValueError(f"{label} contains missing Molecule_Name values.")
    if frame["Molecule_Name"].duplicated().any():
        raise ValueError(f"{label} contains duplicate Molecule_Name values.")


def _read_split_assignments(path: str | Path) -> pd.DataFrame:
    split = pd.read_csv(path)
    required = {"Molecule_Name", "cluster_id", "split"}
    if set(split.columns) != required:
        raise ValueError(
            "D007 source split assignments must have exactly Molecule_Name, "
            "cluster_id, split columns."
        )
    _require_unique_names(split, label="D007 source split assignments")
    if set(split["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("D007 source split assignments have unexpected split labels.")
    return split


def load_and_validate_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path,
    training_seed: int,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    repository_root = Path(repository_root).resolve()
    if not repository_root.exists():
        raise FileNotFoundError(
            f"authoritative repository root does not exist: {repository_root}"
        )
    if not repository_root.is_dir():
        raise NotADirectoryError(
            f"authoritative repository root is not a directory: {repository_root}"
        )
    manifest = _read_json(manifest_path)

    required = {
        "schema_version",
        "repetition",
        "fold",
        "training_seeds",
        "source_split_assignments",
        "source_split_assignments_sha256",
        "training_names",
        "validation_names",
        "training_count",
        "validation_count",
        "protected_test_count_in_partitions",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"D007 split manifest is missing required fields: {sorted(missing)}")

    if manifest["schema_version"] != "D007-split-manifest-v1":
        raise ValueError("D007 split manifest has unexpected schema_version.")

    if int(training_seed) not in set(manifest["training_seeds"]):
        raise ValueError("Explicit training seed is not present in the D007 manifest.")

    if tuple(manifest["training_seeds"]) != D007_TRAINING_SEEDS:
        raise ValueError("D007 split manifest has unexpected training seeds.")

    train_names = list(manifest["training_names"])
    validation_names = list(manifest["validation_names"])

    if len(train_names) != len(set(train_names)):
        raise ValueError("D007 split manifest contains duplicate training names.")
    if len(validation_names) != len(set(validation_names)):
        raise ValueError("D007 split manifest contains duplicate validation names.")
    if set(train_names) & set(validation_names):
        raise ValueError("D007 train/validation partitions overlap.")
    if len(train_names) != int(manifest["training_count"]):
        raise ValueError("D007 manifest training_count does not match training_names.")
    if len(validation_names) != int(manifest["validation_count"]):
        raise ValueError("D007 manifest validation_count does not match validation_names.")
    if int(manifest["protected_test_count_in_partitions"]) != 0:
        raise ValueError("D007 manifest reports protected-test contamination.")

    declared_source_path = Path(manifest["source_split_assignments"])
    if declared_source_path.is_absolute():
        source_path = declared_source_path
    else:
        source_path = repository_root / declared_source_path

    if not source_path.exists():
        raise FileNotFoundError(
            "D007 source split assignments do not exist: "
            f"declared={manifest['source_split_assignments']} "
            f"resolved={source_path}"
        )
    source_path = source_path.resolve()
    observed_source_sha = sha256_file(source_path)
    if observed_source_sha != manifest["source_split_assignments_sha256"]:
        raise ValueError("D007 manifest/source split-assignment identity mismatch.")

    manifest = dict(manifest)
    manifest["_repository_root"] = str(repository_root)
    manifest["_declared_source_split_assignments"] = manifest["source_split_assignments"]
    manifest["_resolved_source_split_assignments"] = str(source_path)
    manifest["_observed_source_split_assignments_sha256"] = observed_source_sha
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_manifest_sha256"] = sha256_file(manifest_path)
    manifest["_training_seed"] = int(training_seed)
    return manifest


def route_d007_manifest_data(
    *,
    dataset_path: str | Path,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset_path = Path(dataset_path)
    dataset = pd.read_csv(dataset_path)
    _require_unique_names(dataset, label="authoritative CYP dataset")
    if "SMILES" not in dataset.columns:
        raise ValueError("authoritative CYP dataset is missing SMILES.")

    split = _read_split_assignments(manifest["_resolved_source_split_assignments"])
    dataset_names = set(dataset["Molecule_Name"])
    source_names = set(split["Molecule_Name"])
    if dataset_names != source_names:
        missing = sorted(source_names - dataset_names)
        unexpected = sorted(dataset_names - source_names)
        raise ValueError(
            "authoritative CYP dataset and D007 source split universe differ; "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    train_names = set(manifest["training_names"])
    validation_names = set(manifest["validation_names"])
    protected_test = set(split.loc[split["split"] == "test", "Molecule_Name"])
    development = set(split.loc[split["split"].isin(["train", "validation"]), "Molecule_Name"])

    if train_names & validation_names:
        raise ValueError("D007 train/validation partitions overlap.")
    if (train_names | validation_names) != development:
        raise ValueError("D007 manifest partitions do not equal the development universe.")
    if (train_names | validation_names) & protected_test:
        raise ValueError("Protected internal-test molecule entered D007 workload.")

    missing_from_dataset = sorted((train_names | validation_names) - dataset_names)
    if missing_from_dataset:
        raise ValueError(f"D007 manifest names are missing from dataset: {missing_from_dataset[:5]}")

    by_name = dataset.set_index("Molecule_Name", drop=False)
    train_df = by_name.loc[list(manifest["training_names"])].copy()
    validation_df = by_name.loc[list(manifest["validation_names"])].copy()

    validation_query_df = validation_df.loc[:, ["Molecule_Name", "SMILES"]].copy()
    provenance = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "repository_root": manifest["_repository_root"],
        "source_split_assignments_declared_path": manifest[
            "_declared_source_split_assignments"
        ],
        "source_split_assignments_resolved_path": manifest[
            "_resolved_source_split_assignments"
        ],
        "source_split_assignments_sha256": manifest[
            "_observed_source_split_assignments_sha256"
        ],
        "protected_internal_test_count": len(protected_test),
        "protected_internal_test_count_in_workload": 0,
    }
    return train_df, validation_query_df, provenance


def build_training_reference_dataset(
    train_df: pd.DataFrame,
) -> tuple[data.MoleculeDataset, Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    targets, mask = build_cyp_target_matrix(train_df, CYP002_TASK_NAMES)
    datapoints = [
        data.MoleculeDatapoint.from_smi(
            smiles,
            target_row,
            name=str(name),
        )
        for smiles, name, target_row in zip(
            train_df["SMILES"].tolist(),
            train_df["Molecule_Name"].tolist(),
            targets,
            strict=True,
        )
    ]
    dataset = data.MoleculeDataset(datapoints)
    scaler = dataset.normalize_targets()

    means = np.asarray(scaler.mean_, dtype=np.float64)
    scales = np.asarray(scaler.scale_, dtype=np.float64)
    if means.shape != (len(CYP002_TASK_NAMES),):
        raise ValueError("training target scaler means do not align with CYP endpoints.")
    if scales.shape != (len(CYP002_TASK_NAMES),):
        raise ValueError("training target scaler scales do not align with CYP endpoints.")
    if not np.isfinite(means).all():
        raise ValueError("training target scaler means must be finite.")
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("training target scaler scales must be finite and positive.")

    reference_y = torch.tensor(targets, dtype=torch.float32)
    reference_mask = torch.tensor(mask, dtype=torch.bool)
    scaler_scales = torch.tensor(scales, dtype=torch.float32)
    metadata = {
        "endpoint_order": list(CYP002_TASK_NAMES),
        "scaler_means": means.tolist(),
        "scaler_scales": scales.tolist(),
        "endpoint_eligible_reference_counts": mask.sum(axis=0).astype(int).tolist(),
    }
    return dataset, scaler, reference_y, reference_mask, scaler_scales, metadata


def build_validation_query_dataset(
    validation_query_df: pd.DataFrame,
) -> data.MoleculeDataset:
    if set(validation_query_df.columns) != {"Molecule_Name", "SMILES"}:
        raise ValueError("validation query frame must contain only Molecule_Name and SMILES.")
    datapoints = [
        data.MoleculeDatapoint.from_smi(
            smiles,
            name=str(name),
        )
        for smiles, name in zip(
            validation_query_df["SMILES"].tolist(),
            validation_query_df["Molecule_Name"].tolist(),
            strict=True,
        )
    ]
    return data.MoleculeDataset(datapoints)


def _extract_sampled_config(artifact: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for key in ("sampled", "sampled_config", "sampled_configuration"):
        value = artifact.get(key)
        if isinstance(value, dict):
            return key, value
    if isinstance(artifact.get("best_trial"), dict):
        best_trial = artifact["best_trial"]
        for key in ("sampled", "sampled_config", "params", "parameters"):
            value = best_trial.get(key)
            if isinstance(value, dict):
                return f"best_trial.{key}", value
    raise ValueError("T121 proposal/config artifact does not contain a sampled config.")


def load_t121_config(
    *,
    t121_config_path: str | Path,
    training_seed: int,
) -> tuple[Any, dict[str, Any]]:
    t121_config_path = Path(t121_config_path)
    artifact = _read_json(t121_config_path)
    sampled_key, sampled = _extract_sampled_config(artifact)
    config = config_from_trial(
        name="D007-T121-P1-real-workload-benchmark",
        sampled=sampled,
        seed=training_seed,
    )
    provenance = {
        "path": str(t121_config_path),
        "sha256": sha256_file(t121_config_path),
        "sampled_config_key": sampled_key,
        "sampled_config": sampled,
        "reconstructed_config": asdict(config),
    }
    return config, provenance


class ChempropP1EmbeddingEncoder:
    def __init__(
        self,
        *,
        message_passing,
        aggregation,
        device: torch.device,
        batch_size: int,
    ) -> None:
        self.message_passing = message_passing
        self.aggregation = aggregation
        self.device = device
        self.batch_size = batch_size

    def __call__(self, dataset: data.MoleculeDataset) -> torch.Tensor:
        loader = data.build_dataloader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        )
        embeddings = []
        self.message_passing.eval()
        self.aggregation.eval()
        with torch.no_grad():
            for batch in loader:
                batch_mol_graph = batch[0]
                moved = batch_mol_graph.to(self.device)
                if moved is not None:
                    batch_mol_graph = moved
                atom_embeddings = self.message_passing(batch_mol_graph)
                molecule_embeddings = self.aggregation(
                    atom_embeddings,
                    batch_mol_graph.batch,
                )
                embeddings.append(molecule_embeddings)
        return torch.cat(embeddings, dim=0)


def build_p1_encoder_and_head(config, *, device: torch.device):
    shared_message_passing = build_p1_shared_message_passing(config).to(device)
    aggregation = nn.NormAggregation(norm=config.aggregation_norm).to(device)
    pair_head = build_p1_pair_head(config).to(device)
    encoder = ChempropP1EmbeddingEncoder(
        message_passing=shared_message_passing.blocks[0],
        aggregation=aggregation,
        device=device,
        batch_size=config.batch_size,
    )
    return shared_message_passing, aggregation, pair_head, encoder


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


def run_real_workload_benchmark(
    *,
    dataset_path: str | Path,
    repository_root: str | Path,
    split_manifest_path: str | Path,
    t121_config_path: str | Path,
    training_seed: int,
    chunk_size: int,
    device: str | torch.device = "cuda:0",
    benchmark_runner=run_p1_all_reference_benchmark,
) -> dict[str, Any]:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    benchmark_device = torch.device(device)
    manifest = load_and_validate_manifest(
        split_manifest_path,
        repository_root=repository_root,
        training_seed=training_seed,
    )
    train_df, validation_query_df, dataset_provenance = route_d007_manifest_data(
        dataset_path=dataset_path,
        manifest=manifest,
    )
    train_dataset, _scaler, reference_y, reference_mask, scaler_scales, reference_metadata = (
        build_training_reference_dataset(train_df)
    )
    validation_dataset = build_validation_query_dataset(validation_query_df)
    config, config_provenance = load_t121_config(
        t121_config_path=t121_config_path,
        training_seed=training_seed,
    )
    shared_encoder, _aggregation, pair_head, encoder = build_p1_encoder_and_head(
        config,
        device=benchmark_device,
    )

    result = benchmark_runner(
        encoder=encoder,
        pair_head=pair_head,
        query_inputs=validation_dataset,
        reference_inputs=train_dataset,
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
        chunk_size=chunk_size,
        validation_evaluations_per_cell=50,
        device=benchmark_device,
    )

    stage_d = result.timings.stage_d_end_to_end_seconds
    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "no_validation_label_or_performance_access_statement": NO_VALIDATION_LABEL_STATEMENT,
        "input_paths": {
            "authoritative_cyp_dataset": str(dataset_path),
            "authoritative_repository_root": str(Path(repository_root).resolve()),
            "d007_split_manifest": str(split_manifest_path),
            "t121_config_artifact": str(t121_config_path),
        },
        "input_hashes": {
            "authoritative_cyp_dataset_sha256": dataset_provenance["dataset_sha256"],
            "d007_split_manifest_sha256": manifest["_manifest_sha256"],
            "d007_source_split_assignments_sha256": dataset_provenance[
                "source_split_assignments_sha256"
            ],
            "t121_config_artifact_sha256": config_provenance["sha256"],
        },
        "d007_source_split_assignments": {
            "manifest_declared_path": dataset_provenance[
                "source_split_assignments_declared_path"
            ],
            "resolved_absolute_path": dataset_provenance[
                "source_split_assignments_resolved_path"
            ],
            "sha256": dataset_provenance["source_split_assignments_sha256"],
        },
        "split": {
            "repetition": manifest["repetition"],
            "fold": manifest["fold"],
            "training_seed": training_seed,
            "training_seeds_declared_by_manifest": manifest["training_seeds"],
        },
        "workload": {
            "training_molecule_count": len(train_df),
            "validation_query_count": len(validation_query_df),
            "endpoint_order": list(CYP002_TASK_NAMES),
            "endpoint_specific_eligible_training_reference_counts": result.accounting.endpoint_eligible_reference_counts,
            "scaler_means": reference_metadata["scaler_means"],
            "scaler_scales": reference_metadata["scaler_scales"],
        },
        "t121_config_provenance": config_provenance,
        "environment": _environment(benchmark_device),
        "p1_parameter_counts": {
            "shared_encoder": asdict(count_parameters(shared_encoder)),
            "pair_head": asdict(count_parameters(pair_head)),
        },
        "benchmark": {
            "stage_a_encode_cache_seconds": result.timings.stage_a_encode_cache_seconds,
            "stage_b_pair_head_seconds": result.timings.stage_b_pair_head_seconds,
            "stage_c_reconstruction_seconds": result.timings.stage_c_reconstruction_seconds,
            "stage_d_end_to_end_seconds": stage_d,
            "unique_query_reference_pairs": result.accounting.unique_query_reference_pairs,
            "directional_g_evaluations": result.accounting.directional_g_evaluations,
            "chunk_size": result.accounting.chunk_size,
            "peak_gpu_memory_bytes": result.peak_accelerator_memory_bytes,
            "timing_excludes": result.timing_excludes,
        },
        "projection": {
            "measured_stage_d_seconds": stage_d,
            "prospective_max_validation_executions": MAX_PROSPECTIVE_VALIDATION_EXECUTIONS,
            "projected_max_validation_seconds": stage_d * MAX_PROSPECTIVE_VALIDATION_EXECUTIONS,
        },
    }
    return artifact


def write_json_artifact(artifact: dict[str, Any], output_json: str | Path) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the D009 P1 real-workload validation-cost benchmark adapter."
    )
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--split-manifest-path", required=True, type=Path)
    parser.add_argument("--t121-config-artifact-path", required=True, type=Path)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--chunk-size", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifact = run_real_workload_benchmark(
        dataset_path=args.dataset_path,
        repository_root=args.repository_root,
        split_manifest_path=args.split_manifest_path,
        t121_config_path=args.t121_config_artifact_path,
        training_seed=args.training_seed,
        chunk_size=args.chunk_size,
    )
    write_json_artifact(artifact, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
