from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from chemprop import nn

from cyp007_infrastructure import D007_TRAINING_SEEDS, config_from_trial, sha256_file
from cyp002_config import CYP002_TASK_NAMES
from p1_pairwise_chemprop import build_p1_shared_message_passing
from p1_pairwise_real_workload_benchmark import (
    ARTIFACT_TYPE,
    ChempropP1EmbeddingEncoder,
    MAX_PROSPECTIVE_VALIDATION_EXECUTIONS,
    build_parser,
    build_training_reference_dataset,
    build_validation_query_dataset,
    load_and_validate_manifest,
    route_d007_manifest_data,
    run_real_workload_benchmark,
    write_json_artifact,
)


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


def _dataset_frame():
    rows = [
        ("train_a", "CCO", 1.0, 10.0, 100.0, 1000.0),
        ("train_b", "CCN", 3.0, np.nan, 300.0, 3000.0),
        ("train_c", "CCC", 5.0, 50.0, np.nan, 5000.0),
        ("val_a", "CCCl", 7.0, 70.0, 700.0, 7000.0),
        ("val_b", "CCBr", 9.0, 90.0, 900.0, 9000.0),
        ("test_a", "CCI", 11.0, 110.0, 1100.0, 11000.0),
    ]
    return pd.DataFrame(
        rows,
        columns=("Molecule_Name", "SMILES", *CYP002_TASK_NAMES),
    )


def _write_inputs(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    dataset_path = tmp_path / "cyp.csv"
    _dataset_frame().to_csv(dataset_path, index=False)

    split_path = repo_root / "cyp" / "artifacts" / "cyp_baseline" / "molecule_split_assignments.csv"
    split_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "Molecule_Name": [
                "train_a",
                "train_b",
                "train_c",
                "val_a",
                "val_b",
                "test_a",
            ],
            "cluster_id": [1, 2, 3, 4, 5, 6],
            "split": [
                "train",
                "train",
                "validation",
                "validation",
                "validation",
                "test",
            ],
        }
    ).to_csv(split_path, index=False)

    manifest = {
        "schema_version": "D007-split-manifest-v1",
        "repetition": 2,
        "fold": 7,
        "split_seed": 20261102,
        "training_seeds": list(D007_TRAINING_SEEDS),
        "source_split_assignments": "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv",
        "source_split_assignments_sha256": sha256_file(split_path),
        "development_universe": "original_train_plus_validation_only",
        "protected_internal_test_rows": 1,
        "training_names": ["train_a", "train_b", "train_c"],
        "validation_names": ["val_a", "val_b"],
        "validation_cluster_ids": [4, 5],
        "training_count": 3,
        "validation_count": 2,
        "protected_test_count_in_partitions": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    t121_path = tmp_path / "t121.json"
    t121_path.write_text(
        json.dumps({"sampled": T121_SAMPLED}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset_path, repo_root, split_path, manifest_path, t121_path


def test_exact_d007_manifest_identity_routing_and_disjointness(tmp_path):
    dataset_path, repo_root, split_path, manifest_path, _t121_path = _write_inputs(tmp_path)
    manifest = load_and_validate_manifest(
        manifest_path,
        repository_root=repo_root,
        training_seed=D007_TRAINING_SEEDS[0],
    )
    train_df, validation_query_df, provenance = route_d007_manifest_data(
        dataset_path=dataset_path,
        manifest=manifest,
    )

    assert train_df["Molecule_Name"].tolist() == ["train_a", "train_b", "train_c"]
    assert validation_query_df["Molecule_Name"].tolist() == ["val_a", "val_b"]
    assert set(train_df["Molecule_Name"]).isdisjoint(validation_query_df["Molecule_Name"])
    assert provenance["protected_internal_test_count_in_workload"] == 0
    assert provenance["dataset_sha256"] == sha256_file(dataset_path)
    assert provenance["repository_root"] == str(repo_root.resolve())
    assert provenance["source_split_assignments_declared_path"] == (
        "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv"
    )
    assert provenance["source_split_assignments_resolved_path"] == str(split_path.resolve())


def test_protected_test_exclusion_and_malformed_manifest_rejection(tmp_path):
    _dataset_path, repo_root, _split_path, manifest_path, _t121_path = _write_inputs(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_names"] = ["train_a", "train_b", "test_a"]
    manifest["protected_test_count_in_partitions"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="protected-test contamination"):
        load_and_validate_manifest(
            manifest_path,
            repository_root=repo_root,
            training_seed=D007_TRAINING_SEEDS[0],
        )


def test_manifest_source_identity_mismatch_rejected(tmp_path):
    _dataset_path, repo_root, split_path, manifest_path, _t121_path = _write_inputs(tmp_path)
    split = pd.read_csv(split_path)
    split.loc[0, "cluster_id"] = 999
    split.to_csv(split_path, index=False)

    with pytest.raises(ValueError, match="identity mismatch"):
        load_and_validate_manifest(
            manifest_path,
            repository_root=repo_root,
            training_seed=D007_TRAINING_SEEDS[0],
        )


def test_repo_relative_source_resolution_is_independent_of_cwd(tmp_path, monkeypatch):
    dataset_path, repo_root, _split_path, manifest_path, _t121_path = _write_inputs(tmp_path)
    other_cwd = tmp_path / "separate_staging_tree"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    manifest = load_and_validate_manifest(
        manifest_path,
        repository_root=repo_root,
        training_seed=D007_TRAINING_SEEDS[0],
    )
    train_df, validation_query_df, provenance = route_d007_manifest_data(
        dataset_path=dataset_path,
        manifest=manifest,
    )

    assert train_df["Molecule_Name"].tolist() == ["train_a", "train_b", "train_c"]
    assert validation_query_df["Molecule_Name"].tolist() == ["val_a", "val_b"]
    assert provenance["source_split_assignments_resolved_path"].startswith(
        str(repo_root.resolve())
    )


def test_wrong_repo_root_fails_closed(tmp_path):
    _dataset_path, _repo_root, _split_path, manifest_path, _t121_path = _write_inputs(tmp_path)
    wrong_root = tmp_path / "wrong_repo"
    wrong_root.mkdir()

    with pytest.raises(FileNotFoundError, match="resolved="):
        load_and_validate_manifest(
            manifest_path,
            repository_root=wrong_root,
            training_seed=D007_TRAINING_SEEDS[0],
        )


def test_training_only_scaler_fitting_and_endpoint_missingness_alignment(tmp_path):
    dataset_path, repo_root, _split_path, manifest_path, _t121_path = _write_inputs(tmp_path)
    manifest = load_and_validate_manifest(
        manifest_path,
        repository_root=repo_root,
        training_seed=D007_TRAINING_SEEDS[0],
    )
    train_df, _validation_query_df, _provenance = route_d007_manifest_data(
        dataset_path=dataset_path,
        manifest=manifest,
    )

    _dataset, _scaler, reference_y, reference_mask, _scales, metadata = (
        build_training_reference_dataset(train_df)
    )

    assert metadata["endpoint_order"] == list(CYP002_TASK_NAMES)
    assert metadata["endpoint_eligible_reference_counts"] == [3, 2, 2, 3]
    assert reference_mask.tolist() == [
        [True, True, True, True],
        [True, False, True, True],
        [True, True, False, True],
    ]
    assert reference_y[0, 0].item() == pytest.approx(1.0)
    assert metadata["scaler_means"][0] == pytest.approx(3.0)
    assert metadata["scaler_means"][1] == pytest.approx(30.0)


def test_validation_query_construction_has_no_validation_targets(tmp_path):
    dataset_path, repo_root, _split_path, manifest_path, _t121_path = _write_inputs(tmp_path)
    manifest = load_and_validate_manifest(
        manifest_path,
        repository_root=repo_root,
        training_seed=D007_TRAINING_SEEDS[0],
    )
    _train_df, validation_query_df, _provenance = route_d007_manifest_data(
        dataset_path=dataset_path,
        manifest=manifest,
    )

    assert list(validation_query_df.columns) == ["Molecule_Name", "SMILES"]
    assert set(validation_query_df.columns).isdisjoint(CYP002_TASK_NAMES)
    dataset = build_validation_query_dataset(validation_query_df)
    assert len(dataset) == 2


def test_real_workload_encoder_uses_real_batch_mol_graph_and_preserves_order():
    class RecordingMessagePassing(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.seen_batch_mol_graphs = []

        def forward(self, batch_mol_graph):
            self.seen_batch_mol_graphs.append(batch_mol_graph)
            assert batch_mol_graph is not None
            assert batch_mol_graph.__class__.__name__ == "BatchMolGraph"
            assert hasattr(batch_mol_graph, "V")
            return self.wrapped(batch_mol_graph)

    config = config_from_trial(
        name="D007-T121-P1-real-workload-encoder-test",
        sampled=T121_SAMPLED,
        seed=D007_TRAINING_SEEDS[0],
    )
    shared_message_passing = build_p1_shared_message_passing(config)
    recording_message_passing = RecordingMessagePassing(
        shared_message_passing.blocks[0]
    )
    aggregation = nn.NormAggregation(norm=config.aggregation_norm)
    encoder = ChempropP1EmbeddingEncoder(
        message_passing=recording_message_passing,
        aggregation=aggregation,
        device=torch.device("cpu"),
        batch_size=2,
    )

    query_df = pd.DataFrame(
        {
            "Molecule_Name": ["q0", "q1", "q2"],
            "SMILES": ["CCO", "CCN", "c1ccccc1"],
        }
    )
    reference_df = pd.DataFrame(
        {
            "Molecule_Name": ["r0", "r1", "r2"],
            "SMILES": ["CCO", "CCN", "c1ccccc1"],
            CYP002_TASK_NAMES[0]: [1.0, 2.0, 3.0],
            CYP002_TASK_NAMES[1]: [10.0, 20.0, 30.0],
            CYP002_TASK_NAMES[2]: [100.0, 200.0, 300.0],
            CYP002_TASK_NAMES[3]: [1000.0, 2000.0, 3000.0],
        }
    )

    query_dataset = build_validation_query_dataset(query_df)
    reference_dataset, *_ = build_training_reference_dataset(reference_df)

    query_embeddings = encoder(query_dataset)
    reference_embeddings = encoder(reference_dataset)

    assert query_embeddings.shape == (3, 600)
    assert reference_embeddings.shape == (3, 600)
    assert torch.isfinite(query_embeddings).all()
    assert torch.isfinite(reference_embeddings).all()
    assert all(
        batch_mol_graph is not None
        and batch_mol_graph.__class__.__name__ == "BatchMolGraph"
        for batch_mol_graph in recording_message_passing.seen_batch_mol_graphs
    )

    single_query_embeddings = []
    for row_index in range(len(query_df)):
        single_query_embeddings.append(
            encoder(build_validation_query_dataset(query_df.iloc[[row_index]].copy()))
        )
    single_query_embeddings = torch.cat(single_query_embeddings, dim=0)
    torch.testing.assert_close(
        query_embeddings,
        single_query_embeddings,
        rtol=1e-6,
        atol=1e-7,
    )

    reversed_reference_df = reference_df.iloc[::-1].reset_index(drop=True)
    reversed_reference_dataset, *_ = build_training_reference_dataset(
        reversed_reference_df
    )
    reversed_reference_embeddings = encoder(reversed_reference_dataset)
    torch.testing.assert_close(
        reversed_reference_embeddings,
        reference_embeddings[[2, 1, 0]],
        rtol=1e-6,
        atol=1e-7,
    )


def test_no_validation_label_or_performance_api_surface():
    parser = build_parser()
    option_dests = {action.dest for action in parser._actions}
    forbidden = {
        "validation_labels",
        "validation_targets",
        "confidence_intervals",
        "st_rae",
        "metric",
        "performance",
        "d008_labels",
        "protected_test_labels",
    }
    assert option_dests.isdisjoint(forbidden)
    assert "repository_root" in option_dests


def test_projection_arithmetic_and_provenance_hash_recording(tmp_path):
    dataset_path, repo_root, split_path, manifest_path, t121_path = _write_inputs(tmp_path)

    def fake_runner(**kwargs):
        assert kwargs["validation_evaluations_per_cell"] == 50
        assert kwargs["chunk_size"] == 4
        assert kwargs["query_inputs"] is not None
        assert kwargs["reference_inputs"] is not None
        return SimpleNamespace(
            timings=SimpleNamespace(
                stage_a_encode_cache_seconds=1.0,
                stage_b_pair_head_seconds=2.0,
                stage_c_reconstruction_seconds=3.0,
                stage_d_end_to_end_seconds=4.0,
            ),
            accounting=SimpleNamespace(
                unique_query_reference_pairs=6,
                directional_g_evaluations=12,
                chunk_size=4,
                endpoint_eligible_reference_counts=(3, 2, 2, 3),
            ),
            peak_accelerator_memory_bytes=1234,
            timing_excludes=("warmup executions",),
        )

    artifact = run_real_workload_benchmark(
        dataset_path=dataset_path,
        repository_root=repo_root,
        split_manifest_path=manifest_path,
        t121_config_path=t121_path,
        training_seed=D007_TRAINING_SEEDS[0],
        chunk_size=4,
        device="cpu",
        benchmark_runner=fake_runner,
    )

    assert artifact["artifact_type"] == ARTIFACT_TYPE
    assert artifact["input_hashes"]["authoritative_cyp_dataset_sha256"] == sha256_file(dataset_path)
    assert artifact["input_hashes"]["d007_split_manifest_sha256"] == sha256_file(manifest_path)
    assert artifact["input_hashes"]["t121_config_artifact_sha256"] == sha256_file(t121_path)
    assert artifact["input_paths"]["authoritative_repository_root"] == str(repo_root.resolve())
    assert artifact["d007_source_split_assignments"] == {
        "manifest_declared_path": "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv",
        "resolved_absolute_path": str(split_path.resolve()),
        "sha256": sha256_file(split_path),
    }
    assert artifact["split"] == {
        "repetition": 2,
        "fold": 7,
        "training_seed": D007_TRAINING_SEEDS[0],
        "training_seeds_declared_by_manifest": list(D007_TRAINING_SEEDS),
    }
    assert artifact["projection"]["measured_stage_d_seconds"] == 4.0
    assert artifact["projection"]["prospective_max_validation_executions"] == (
        MAX_PROSPECTIVE_VALIDATION_EXECUTIONS
    )
    assert artifact["projection"]["projected_max_validation_seconds"] == 30000.0
    assert "No validation labels" in artifact[
        "no_validation_label_or_performance_access_statement"
    ]

    output_json = tmp_path / "artifact.json"
    write_json_artifact(artifact, output_json)
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["artifact_type"] == ARTIFACT_TYPE


def test_mismatched_dataset_universe_rejected(tmp_path):
    dataset_path, repo_root, _split_path, manifest_path, _t121_path = _write_inputs(tmp_path)
    dataset = pd.read_csv(dataset_path)
    dataset.loc[0, "Molecule_Name"] = "unexpected"
    dataset.to_csv(dataset_path, index=False)
    manifest = load_and_validate_manifest(
        manifest_path,
        repository_root=repo_root,
        training_seed=D007_TRAINING_SEEDS[0],
    )

    with pytest.raises(ValueError, match="universe differ"):
        route_d007_manifest_data(
            dataset_path=dataset_path,
            manifest=manifest,
        )
