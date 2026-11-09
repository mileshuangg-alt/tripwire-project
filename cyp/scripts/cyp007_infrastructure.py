from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from cyp002_config import CYP002_C0, CYP002ModelConfig


D007_REPETITIONS = 5
D007_FOLDS = 10
D007_TOTAL_SPLITS = D007_REPETITIONS * D007_FOLDS
D007_TRAINING_SEEDS = (20261101, 20261102, 20261103)
D007_SPLIT_SEED = 20261100
D007_CONTROLLER_RNG_SEED = 20261107
D007_DEVELOPMENT_ROWS = 4169
D007_PROTECTED_TEST_ROWS = 736

D007_SEARCH_SPACE = {
    "message_passing_depth": {
        "kind": "int",
        "low": 2,
        "high": 6,
    },
    "message_hidden_dim": {
        "kind": "int_step",
        "low": 300,
        "high": 2400,
        "step": 100,
    },
    "ffn_hidden_dim": {
        "kind": "int_step",
        "low": 300,
        "high": 2400,
        "step": 100,
    },
    "ffn_num_layers": {
        "kind": "int",
        "low": 1,
        "high": 2,
    },
    "dropout": {
        "kind": "categorical",
        "values": [0.0] * 8
        + [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "batch_size": {
        "kind": "categorical",
        "values": [16, 32, 64, 128, 256],
    },
    "max_lr": {
        "kind": "loguniform",
        "low": 1e-4,
        "high": 1e-2,
    },
    "init_lr_ratio": {
        "kind": "loguniform",
        "low": 1e-2,
        "high": 1.0,
    },
    "final_lr_ratio": {
        "kind": "loguniform",
        "low": 1e-2,
        "high": 1.0,
    },
    "warmup_epochs": {
        "kind": "int",
        "low": 1,
        "high": 25,
    },
}


@dataclass(frozen=True)
class D007ControllerConfig:
    total_trials: int = 128
    random_startup_trials: int = 32
    tpe_trials: int = 96
    controller_rng_seed: int = D007_CONTROLLER_RNG_SEED
    pruning: str = "none"
    imported_observations: int = 0
    objective: str = "mean_validation_macro_st_rae_50_folds_x_3_seeds"


D007_CONTROLLER_CONFIG = D007ControllerConfig()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def d007_search_space() -> dict:
    return json.loads(json.dumps(D007_SEARCH_SPACE))


def d007_controller_config() -> dict:
    return asdict(D007_CONTROLLER_CONFIG)


def validate_search_space(space: dict) -> None:
    expected_keys = set(D007_SEARCH_SPACE)
    if set(space) != expected_keys:
        raise ValueError(
            "D007 search space must contain exactly the intended ten "
            f"dimensions: {sorted(expected_keys)}"
        )

    for name, expected in D007_SEARCH_SPACE.items():
        if space[name] != expected:
            raise ValueError(f"D007 search-space mismatch for {name!r}.")


def config_from_trial(
    *,
    name: str,
    sampled: dict,
    seed: int,
) -> CYP002ModelConfig:
    validate_search_space(D007_SEARCH_SPACE)
    missing = set(D007_SEARCH_SPACE) - set(sampled)
    extra = set(sampled) - set(D007_SEARCH_SPACE)
    if missing or extra:
        raise ValueError(
            "Sampled D007 configuration keys do not match the search "
            f"space. Missing={sorted(missing)} extra={sorted(extra)}"
        )

    max_lr = float(sampled["max_lr"])
    init_lr_ratio = float(sampled["init_lr_ratio"])
    final_lr_ratio = float(sampled["final_lr_ratio"])
    warmup_epochs = sampled["warmup_epochs"]

    if not isinstance(warmup_epochs, int) or isinstance(warmup_epochs, bool):
        raise ValueError("warmup_epochs must be an integer epoch count.")

    return CYP002ModelConfig(
        name=name,
        message_passing_depth=int(sampled["message_passing_depth"]),
        message_hidden_dim=int(sampled["message_hidden_dim"]),
        ffn_hidden_dim=int(sampled["ffn_hidden_dim"]),
        ffn_num_layers=int(sampled["ffn_num_layers"]),
        dropout=float(sampled["dropout"]),
        batch_size=int(sampled["batch_size"]),
        max_lr=max_lr,
        initial_lr=init_lr_ratio * max_lr,
        final_lr=final_lr_ratio * max_lr,
        warmup_epochs=warmup_epochs,
        seed=seed,
    )


def c0_for_d007(seed: int) -> CYP002ModelConfig:
    return CYP002ModelConfig(
        name="C0",
        message_passing_depth=CYP002_C0.message_passing_depth,
        message_hidden_dim=CYP002_C0.message_hidden_dim,
        ffn_hidden_dim=CYP002_C0.ffn_hidden_dim,
        ffn_num_layers=CYP002_C0.ffn_num_layers,
        dropout=CYP002_C0.dropout,
        max_lr=CYP002_C0.max_lr,
        batch_size=CYP002_C0.batch_size,
        initial_lr=CYP002_C0.initial_lr,
        final_lr=CYP002_C0.final_lr,
        warmup_epochs=CYP002_C0.warmup_epochs,
        seed=seed,
    )


def _read_split_assignments(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Molecule_Name", "cluster_id", "split"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(
                "Split assignments must have exactly Molecule_Name, "
                "cluster_id, split columns."
            )
        rows = list(reader)

    names = [row["Molecule_Name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Split assignments contain duplicate molecules.")
    return rows


def _development_and_test_rows(rows: list[dict]) -> tuple[list[dict], set[str]]:
    split_values = {row["split"] for row in rows}
    if split_values != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected split labels: {sorted(split_values)}")

    development = [
        row for row in rows if row["split"] in {"train", "validation"}
    ]
    protected_test = {
        row["Molecule_Name"] for row in rows if row["split"] == "test"
    }
    if len(development) != D007_DEVELOPMENT_ROWS:
        raise ValueError(
            f"Expected {D007_DEVELOPMENT_ROWS} development molecules, "
            f"got {len(development)}."
        )
    if len(protected_test) != D007_PROTECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {D007_PROTECTED_TEST_ROWS} protected test molecules, "
            f"got {len(protected_test)}."
        )
    return development, protected_test


def generate_d007_split_manifests(
    split_assignment_path: str | Path,
) -> list[dict]:
    rows = _read_split_assignments(split_assignment_path)
    development, protected_test = _development_and_test_rows(rows)

    by_cluster: dict[int, list[str]] = {}
    for row in development:
        cluster_id = int(row["cluster_id"])
        by_cluster.setdefault(cluster_id, []).append(row["Molecule_Name"])

    manifests = []
    all_development = {row["Molecule_Name"] for row in development}

    for repetition in range(1, D007_REPETITIONS + 1):
        rng = random.Random(D007_SPLIT_SEED + repetition)
        clusters = sorted(by_cluster)
        rng.shuffle(clusters)

        fold_clusters: list[list[int]] = [[] for _ in range(D007_FOLDS)]
        fold_sizes = [0 for _ in range(D007_FOLDS)]
        for cluster_id in clusters:
            fold_index = min(range(D007_FOLDS), key=lambda idx: (fold_sizes[idx], idx))
            fold_clusters[fold_index].append(cluster_id)
            fold_sizes[fold_index] += len(by_cluster[cluster_id])

        for fold_index in range(D007_FOLDS):
            validation_clusters = set(fold_clusters[fold_index])
            validation_names = sorted(
                name
                for cluster_id in validation_clusters
                for name in by_cluster[cluster_id]
            )
            training_names = sorted(all_development - set(validation_names))
            if set(validation_names) & protected_test:
                raise RuntimeError("Protected test molecule entered D007 validation.")
            if set(training_names) & protected_test:
                raise RuntimeError("Protected test molecule entered D007 training.")
            manifests.append(
                {
                    "schema_version": "D007-split-manifest-v1",
                    "repetition": repetition,
                    "fold": fold_index + 1,
                    "split_seed": D007_SPLIT_SEED + repetition,
                    "training_seeds": list(D007_TRAINING_SEEDS),
                    "source_split_assignments": str(split_assignment_path),
                    "source_split_assignments_sha256": sha256_file(
                        split_assignment_path
                    ),
                    "development_universe": "original_train_plus_validation_only",
                    "protected_internal_test_rows": D007_PROTECTED_TEST_ROWS,
                    "training_names": training_names,
                    "validation_names": validation_names,
                    "validation_cluster_ids": sorted(validation_clusters),
                    "training_count": len(training_names),
                    "validation_count": len(validation_names),
                    "protected_test_count_in_partitions": 0,
                }
            )

    if len(manifests) != D007_TOTAL_SPLITS:
        raise RuntimeError(f"Expected 50 D007 split manifests, got {len(manifests)}.")
    return manifests


def validate_d007_split_manifests(
    manifests: Iterable[dict],
    split_assignment_path: str | Path,
) -> dict:
    manifests = list(manifests)
    rows = _read_split_assignments(split_assignment_path)
    development, protected_test = _development_and_test_rows(rows)
    cluster_by_name = {
        row["Molecule_Name"]: int(row["cluster_id"])
        for row in development
    }
    development_names = set(cluster_by_name)
    seen = set()

    for manifest in manifests:
        key = (manifest["repetition"], manifest["fold"])
        if key in seen:
            raise ValueError(f"Duplicate D007 split manifest: {key}")
        seen.add(key)

        if tuple(manifest["training_seeds"]) != D007_TRAINING_SEEDS:
            raise ValueError("D007 split manifest has unexpected training seeds.")

        train = set(manifest["training_names"])
        validation = set(manifest["validation_names"])
        if train & validation:
            raise ValueError("D007 train/validation partitions overlap.")
        if train | validation != development_names:
            raise ValueError("D007 split does not equal the development universe.")
        if (train | validation) & protected_test:
            raise RuntimeError("Protected internal-test molecule entered D007 split.")

        train_clusters = {cluster_by_name[name] for name in train}
        validation_clusters = {cluster_by_name[name] for name in validation}
        if train_clusters & validation_clusters:
            raise ValueError("A cluster crosses train/validation in a D007 fold.")

    expected_keys = {
        (repetition, fold)
        for repetition in range(1, D007_REPETITIONS + 1)
        for fold in range(1, D007_FOLDS + 1)
    }
    if seen != expected_keys:
        raise ValueError("D007 split manifests are not the complete 5 x 10 grid.")

    return {
        "manifest_count": len(manifests),
        "development_rows": len(development_names),
        "protected_test_rows": len(protected_test),
        "protected_test_rows_in_d007_partitions": 0,
        "training_seeds": list(D007_TRAINING_SEEDS),
    }


def write_split_manifests(manifests: Iterable[dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"D007 split-manifest directory already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    for manifest in manifests:
        path = output_dir / (
            f"rep_{manifest['repetition']:02d}_fold_{manifest['fold']:02d}.json"
        )
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    freeze_record = {
        "schema_version": "D007-split-freeze-v1",
        "manifest_count": D007_TOTAL_SPLITS,
        "training_seeds": list(D007_TRAINING_SEEDS),
        "split_seed_base": D007_SPLIT_SEED,
        "files": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.glob("rep_*_fold_*.json"))
        },
    }
    (output_dir / "FREEZE_RECORD.json").write_text(
        json.dumps(freeze_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def aggregate_d007_objective(results: Iterable[dict]) -> float:
    values: dict[tuple[int, int, int], float] = {}
    expected = {
        (repetition, fold, seed)
        for repetition in range(1, D007_REPETITIONS + 1)
        for fold in range(1, D007_FOLDS + 1)
        for seed in D007_TRAINING_SEEDS
    }

    for result in results:
        key = (
            result.get("repetition"),
            result.get("fold"),
            result.get("seed"),
        )
        if key not in expected:
            raise ValueError(f"Unexpected D007 fit-result cell: {key!r}")
        if key in values:
            raise ValueError(f"Duplicate D007 fit-result cell: {key!r}")
        value = float(result.get("macro_st_rae"))
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite D007 macro ST-RAE for cell {key!r}")
        values[key] = value

    missing = expected - set(values)
    if missing:
        raise ValueError(f"Missing D007 fit-result cells: {sorted(missing)[:5]}")
    if len(values) != 150:
        raise ValueError(f"Expected exactly 150 D007 fit results, got {len(values)}")

    fold_means = []
    for repetition in range(1, D007_REPETITIONS + 1):
        for fold in range(1, D007_FOLDS + 1):
            fold_means.append(
                sum(values[(repetition, fold, seed)] for seed in D007_TRAINING_SEEDS)
                / len(D007_TRAINING_SEEDS)
            )
    return sum(fold_means) / len(fold_means)


def controller_observation(trial_index: int, objective: float) -> dict:
    if not isinstance(trial_index, int) or isinstance(trial_index, bool):
        raise ValueError("trial_index must be an integer.")
    if not math.isfinite(float(objective)):
        raise ValueError("D007 controller objective must be finite.")
    return {
        "trial_index": trial_index,
        "objective": float(objective),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-assignments",
        default="cyp/artifacts/cyp_baseline/molecule_split_assignments.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="cyp/artifacts/cyp_007/split_manifests",
    )
    args = parser.parse_args()

    manifests = generate_d007_split_manifests(args.split_assignments)
    summary = validate_d007_split_manifests(manifests, args.split_assignments)
    write_split_manifests(manifests, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
