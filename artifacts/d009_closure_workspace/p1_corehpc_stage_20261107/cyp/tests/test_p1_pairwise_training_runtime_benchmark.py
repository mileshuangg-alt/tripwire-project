from __future__ import annotations

import json

import pandas as pd
import pytest

from cyp002_config import CYP002_TASK_NAMES
from cyp007_infrastructure import D007_TRAINING_SEEDS, sha256_file
from p1_pairwise_training import P1_ACCEPTED_PAIR_BATCH_SIZE, P1_MAX_UPDATES
from p1_pairwise_training_runtime_benchmark import (
    ARTIFACT_TYPE,
    build_parser,
    run_training_runtime_benchmark,
)
import p1_pairwise_training_runtime_benchmark as runtime_benchmark


T121_SAMPLED = {
    "batch_size": 16,
    "dropout": 0.4,
    "ffn_hidden_dim": 2000,
    "ffn_num_layers": 1,
    "final_lr_ratio": 0.24427727346003839,
    "init_lr_ratio": 0.017323283256612185,
    "max_lr": 0.0008245400010426866,
    "message_hidden_dim": 600,
    "message_passing_depth": 4,
    "warmup_epochs": 10,
}


class FakeCudaBackend:
    def is_available(self):
        return True

    def device_count(self):
        return 1

    def set_device(self, device):
        self.device = device

    def synchronize(self, device):
        pass

    def reset_peak_memory_stats(self, device):
        self.reset_device = device

    def max_memory_allocated(self, device):
        return 111

    def max_memory_reserved(self, device):
        return 222


def _write_inputs(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    split_path = repo_root / "cyp" / "artifacts" / "cyp_baseline" / "molecule_split_assignments.csv"
    split_path.parent.mkdir(parents=True)
    split = pd.DataFrame(
        {
            "Molecule_Name": ["a", "b", "c", "v", "t"],
            "cluster_id": [1, 2, 3, 4, 5],
            "split": ["train", "train", "validation", "validation", "test"],
        }
    )
    split.to_csv(split_path, index=False)

    dataset_path = tmp_path / "dataset.csv"
    rows = [
        ["a", "CCO", 1.0, 10.0, 100.0, 1000.0],
        ["b", "CCN", 2.0, 20.0, 200.0, 2000.0],
        ["c", "CCC", 3.0, 30.0, 300.0, 3000.0],
        ["v", "CCCl", 4.0, 40.0, 400.0, 4000.0],
        ["t", "CCBr", 5.0, 50.0, 500.0, 5000.0],
    ]
    pd.DataFrame(rows, columns=["Molecule_Name", "SMILES", *CYP002_TASK_NAMES]).to_csv(
        dataset_path,
        index=False,
    )

    manifest = {
        "schema_version": "D007-split-manifest-v1",
        "repetition": 1,
        "fold": 2,
        "split_seed": 20261101,
        "training_seeds": list(D007_TRAINING_SEEDS),
        "source_split_assignments": "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv",
        "source_split_assignments_sha256": sha256_file(split_path),
        "development_universe": "original_train_plus_validation_only",
        "protected_internal_test_rows": 1,
        "training_names": ["a", "b", "c"],
        "validation_names": ["v"],
        "validation_cluster_ids": [4],
        "training_count": 3,
        "validation_count": 1,
        "protected_test_count_in_partitions": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config_path = tmp_path / "t121.json"
    config_path.write_text(json.dumps({"sampled": T121_SAMPLED}), encoding="utf-8")
    return repo_root, dataset_path, split_path, manifest_path, config_path


def _stepping_update_factory(call_log):
    def factory(**kwargs):
        optimizer = kwargs["optimizer"]
        scheduler = kwargs["scheduler"]

        def update(batch):
            call_log.append(batch.accepted_count)
            optimizer.step()
            scheduler.step()
            return {"loss": 1.0}

        return update

    return factory


def test_runtime_benchmark_timing_accounting_scheduler_and_projection(tmp_path):
    repo_root, dataset_path, _split_path, manifest_path, config_path = _write_inputs(tmp_path)
    output_json = tmp_path / "runtime.json"
    call_log = []

    artifact = run_training_runtime_benchmark(
        repository_root=repo_root,
        dataset_path=dataset_path,
        split_manifest_path=manifest_path,
        t121_config_path=config_path,
        training_seed=D007_TRAINING_SEEDS[0],
        measured_update_count=3,
        warmup_update_count=2,
        output_json=output_json,
        device="cpu",
        require_cuda=False,
        cuda_backend=FakeCudaBackend(),
        update_step_factory=_stepping_update_factory(call_log),
    )

    assert output_json.exists()
    assert artifact["artifact_type"] == ARTIFACT_TYPE
    assert call_log == [16, 16, 16, 16, 16]
    assert artifact["timing"]["warmup_updates_excluded"] == 2
    assert artifact["timing"]["measured_updates"] == 3
    assert len(artifact["timing"]["per_update_durations_seconds"]) == 3
    assert artifact["sampler_accounting"]["warmup"]["update_durations_seconds"] == []
    assert len(artifact["sampler_accounting"]["measured"]["update_durations_seconds"]) == 3
    assert artifact["sampler_accounting"]["warmup"]["updates"] == 2
    assert artifact["sampler_accounting"]["measured"]["updates"] == 3
    assert artifact["sampler_accounting"]["measured"]["accepted_pair_rows"] == (
        3 * P1_ACCEPTED_PAIR_BATCH_SIZE
    )
    assert artifact["scheduler"]["start_scheduler_position"] == 2
    assert artifact["scheduler"]["end_scheduler_position"] == 5
    assert artifact["scheduler"]["end_lr"] != artifact["scheduler"]["start_lr"]
    assert artifact["projection"]["projected_training_seconds"] == pytest.approx(
        artifact["timing"]["seconds_per_update"] * P1_MAX_UPDATES * 150
    )
    assert artifact["projection"]["projected_training_gpu_hours"] == pytest.approx(
        artifact["projection"]["projected_training_seconds"] / 3600.0
    )
    assert "representative-cost extrapolation" in artifact["projection"]["label"]


def test_runtime_benchmark_drift_statistic_arithmetic(tmp_path, monkeypatch):
    repo_root, dataset_path, _split_path, manifest_path, config_path = _write_inputs(tmp_path)
    clock_values = [0.0]
    current = 0.0
    durations = [float(index + 1) for index in range(100)]
    for duration in durations:
        clock_values.append(current)
        current += duration
        clock_values.append(current)
    clock_values.append(current)
    clock_iter = iter(clock_values)
    monkeypatch.setattr(
        runtime_benchmark.time,
        "perf_counter",
        lambda: next(clock_iter),
    )

    artifact = run_training_runtime_benchmark(
        repository_root=repo_root,
        dataset_path=dataset_path,
        split_manifest_path=manifest_path,
        t121_config_path=config_path,
        training_seed=D007_TRAINING_SEEDS[0],
        measured_update_count=100,
        warmup_update_count=0,
        device="cpu",
        require_cuda=False,
        cuda_backend=FakeCudaBackend(),
        update_step_factory=_stepping_update_factory([]),
    )

    recorded = artifact["timing"]["per_update_durations_seconds"]
    assert recorded == durations
    drift = artifact["timing"]["drift_statistics"]
    first_25_mean = sum(durations[:25]) / 25
    last_25_mean = sum(durations[-25:]) / 25
    assert drift["mean_seconds_per_update"] == pytest.approx(sum(durations) / 100)
    assert drift["median_seconds_per_update"] == pytest.approx(50.5)
    assert drift["first_25_mean_seconds_per_update"] == pytest.approx(first_25_mean)
    assert drift["last_25_mean_seconds_per_update"] == pytest.approx(last_25_mean)
    assert drift["last_minus_first_25_mean_seconds_per_update"] == pytest.approx(
        last_25_mean - first_25_mean
    )
    assert drift["relative_last_vs_first_25_mean_difference"] == pytest.approx(
        (last_25_mean - first_25_mean) / first_25_mean
    )
    assert artifact["timing"]["seconds_per_update"] == pytest.approx(
        artifact["timing"]["total_measured_seconds"] / 100
    )
    assert artifact["projection"]["seconds_per_update"] == pytest.approx(
        artifact["timing"]["seconds_per_update"]
    )


def test_runtime_benchmark_provenance_and_hashes(tmp_path):
    repo_root, dataset_path, split_path, manifest_path, config_path = _write_inputs(tmp_path)
    artifact = run_training_runtime_benchmark(
        repository_root=repo_root,
        dataset_path=dataset_path,
        split_manifest_path=manifest_path,
        t121_config_path=config_path,
        training_seed=D007_TRAINING_SEEDS[0],
        measured_update_count=1,
        warmup_update_count=0,
        device="cpu",
        require_cuda=False,
        cuda_backend=FakeCudaBackend(),
        update_step_factory=_stepping_update_factory([]),
    )

    assert artifact["input_hashes"]["dataset_sha256"] == sha256_file(dataset_path)
    assert artifact["input_hashes"]["split_manifest_sha256"] == sha256_file(manifest_path)
    assert artifact["input_hashes"]["source_split_assignments_sha256"] == sha256_file(split_path)
    assert artifact["input_hashes"]["t121_config_sha256"] == sha256_file(config_path)
    assert artifact["split"] == {
        "repetition": 1,
        "fold": 2,
        "training_seed": D007_TRAINING_SEEDS[0],
    }
    assert artifact["training_partition"]["training_molecule_count"] == 3
    assert artifact["training_partition"]["n_t"] == [3, 3, 3, 3]
    assert artifact["training_partition"]["w_t"] == [1.0, 1.0, 1.0, 1.0]
    assert artifact["parameter_counts"]["shared_encoder"]["total"] > 0
    assert artifact["parameter_counts"]["pair_head"]["total"] > 0


def test_runtime_benchmark_forbidden_validation_performance_api_surface():
    parser = build_parser()
    option_dests = {action.dest for action in parser._actions}
    forbidden = {
        "validation_labels",
        "validation_targets",
        "validation_queries",
        "st_rae",
        "metric",
        "checkpoint_selection",
        "early_stopping",
        "d008_labels",
        "protected_test_labels",
    }
    assert option_dests.isdisjoint(forbidden)
    assert {
        "repository_root",
        "dataset_path",
        "split_manifest_path",
        "t121_config_artifact_path",
        "training_seed",
        "measured_update_count",
        "output_json",
    }.issubset(option_dests)


def test_runtime_benchmark_fails_closed_on_bad_inputs_and_nonfinite_loss(tmp_path):
    repo_root, dataset_path, _split_path, manifest_path, config_path = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="measured_update_count"):
        run_training_runtime_benchmark(
            repository_root=repo_root,
            dataset_path=dataset_path,
            split_manifest_path=manifest_path,
            t121_config_path=config_path,
            training_seed=D007_TRAINING_SEEDS[0],
            measured_update_count=0,
            device="cpu",
            require_cuda=False,
            cuda_backend=FakeCudaBackend(),
            update_step_factory=_stepping_update_factory([]),
        )

    def bad_factory(**kwargs):
        def update(_batch):
            kwargs["optimizer"].step()
            kwargs["scheduler"].step()
            return {"loss": float("nan")}
        return update

    with pytest.raises(FloatingPointError, match="non-finite"):
        run_training_runtime_benchmark(
            repository_root=repo_root,
            dataset_path=dataset_path,
            split_manifest_path=manifest_path,
            t121_config_path=config_path,
            training_seed=D007_TRAINING_SEEDS[0],
            measured_update_count=1,
            warmup_update_count=0,
            device="cpu",
            require_cuda=False,
            cuda_backend=FakeCudaBackend(),
            update_step_factory=bad_factory,
        )


def test_runtime_benchmark_requires_cuda_for_real_execution(tmp_path):
    repo_root, dataset_path, _split_path, manifest_path, config_path = _write_inputs(tmp_path)

    class NoCuda(FakeCudaBackend):
        def is_available(self):
            return False

    with pytest.raises(RuntimeError, match="CUDA is required"):
        run_training_runtime_benchmark(
            repository_root=repo_root,
            dataset_path=dataset_path,
            split_manifest_path=manifest_path,
            t121_config_path=config_path,
            training_seed=D007_TRAINING_SEEDS[0],
            measured_update_count=1,
            cuda_backend=NoCuda(),
            update_step_factory=_stepping_update_factory([]),
        )
