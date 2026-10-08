from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from cyp002_config import CYP002_C0, CYP002_GRID
from cyp002_manifest import (
    canonical_json_bytes,
    canonical_experiment_config,
    experiment_config_sha256,
    finalize_manifest,
    build_planned_manifest,
    sha256_file,
)


def test_canonical_json_serialization_is_frozen():
    obj = {
        "z": 2,
        "a": 1,
        "nested": {
            "b": True,
            "a": 0.001
        }
    }

    expected = (
        '{"a":1,"nested":{"a":0.001,"b":true},"z":2}'
    ).encode("utf-8")

    assert canonical_json_bytes(obj) == expected


def test_experiment_config_contains_load_bearing_decisions():
    config = canonical_experiment_config(CYP002_C0, "stock")

    assert config["configuration"] == "C0"
    assert config["variant"] == "stock"
    assert config["seed"] == 20261001

    architecture = config["architecture"]
    assert architecture["aggregation"] == "norm"
    assert architecture["aggregation_norm"] == 100
    assert architecture["depth"] == 3
    assert architecture["message_hidden_dim"] == 300
    assert architecture["ffn_hidden_dim"] == 300
    assert architecture["activation"] == "relu"

    training = config["training"]
    assert training["batch_size"] == 64
    assert training["max_epochs"] == 50
    assert training["lr_schedule"] == "NoamLR"
    assert training["initial_lr"] == 1e-4
    assert training["max_lr"] == 1e-3
    assert training["final_lr"] == 1e-4

    loss = config["loss"]
    assert loss["type_per_task"]
    assert loss["weighting_strategy"] == "observed_label_count"
    assert loss["renormalize_over_present_tasks"] is False


def test_default_seed_reproduces_explicit_20261001_config():
    default_config = canonical_experiment_config(CYP002_C0, "stock")
    explicit_config = canonical_experiment_config(
        CYP002_C0,
        "stock",
        seed=20261001,
    )

    assert explicit_config == default_config
    assert experiment_config_sha256(explicit_config) == experiment_config_sha256(
        default_config
    )


def test_different_seed_changes_only_seed_fields():
    default_config = canonical_experiment_config(CYP002_C0, "stock")
    alternate_config = canonical_experiment_config(
        CYP002_C0,
        "stock",
        seed=12345,
    )

    expected = deepcopy(default_config)
    expected["seed"] = 12345
    expected["data_loading"]["shuffle_seed"] = 12345

    assert alternate_config == expected
    assert experiment_config_sha256(alternate_config) != experiment_config_sha256(
        default_config
    )


def test_stock_and_task_balanced_configs_have_different_hashes():
    stock_hash = experiment_config_sha256(
        canonical_experiment_config(CYP002_C0, "stock")
    )

    balanced_hash = experiment_config_sha256(
        canonical_experiment_config(CYP002_C0, "task_balanced")
    )

    assert stock_hash != balanced_hash


def test_sha256_file_matches_standard_sha256(tmp_path: Path):
    path = tmp_path / "example.txt"
    path.write_bytes(b"tripwire")

    expected = hashlib.sha256(b"tripwire").hexdigest()

    assert sha256_file(path) == expected


def test_finalize_manifest_preserves_experiment_config_hash(tmp_path: Path):
    checkpoint = tmp_path / "best.ckpt"
    metrics = tmp_path / "metrics.json"

    checkpoint.write_bytes(b"checkpoint")
    metrics.write_text(
        json.dumps({"val_macro_st_rae": 0.5}),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.json"

    manifest = {
        "manifest_schema_version": "1.0",
        "experiment_id": "CYP002-C0-stock-seed20261001",
        "configuration": "C0",
        "variant": "stock",
        "seed": 20261001,
        "experiment_config_sha256": "CONFIG_HASH",
        "execution": {
            "status": "running",
            "start_utc": "2026-10-02T00:00:00+00:00",
            "end_utc": None,
            "invocation_command": "python train.py"
        },
        "outputs": {
            "best_checkpoint_path": str(checkpoint),
            "best_checkpoint_sha256": None,
            "metrics_path": str(metrics),
            "metrics_sha256": None
        }
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    finalize_manifest(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        end_utc="2026-10-02T01:00:00+00:00",
        status="completed",
    )

    finalized = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert finalized["experiment_config_sha256"] == "CONFIG_HASH"
    assert finalized["outputs"]["best_checkpoint_sha256"] == hashlib.sha256(
        b"checkpoint"
    ).hexdigest()
    assert finalized["outputs"]["metrics_sha256"] == hashlib.sha256(
        b'{"val_macro_st_rae": 0.5}'
    ).hexdigest()
    assert finalized["execution"]["status"] == "completed"
    assert finalized["execution"]["end_utc"] == (
        "2026-10-02T01:00:00+00:00"
    )


def _write_manifest_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = tmp_path / "dataset.csv"
    split_path = tmp_path / "split.csv"
    dataset_path.write_text(
        "Molecule_Name,SMILES\n"
        + "\n".join(f"mol_{index},CC" for index in range(4905))
        + "\n",
        encoding="utf-8",
    )
    split_path.write_text(
        "Molecule_Name,cluster_id,split\n"
        + "\n".join(
            f"mol_{index},{index},train" for index in range(4905)
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_path, split_path


def _build_test_manifest(tmp_path: Path, **kwargs):
    dataset_path, split_path = _write_manifest_inputs(tmp_path)
    return build_planned_manifest(
        repo_root=Path.cwd(),
        git_commit_sha_value="fbb27f0a65a1d536154f4e43616bb5208eb62895",
        repo_dirty_value=False,
        dataset_path=dataset_path,
        split_path=split_path,
        dataset_source="unit synthetic dataset",
        dataset_version_or_export_date="unit-test",
        split_method="unit synthetic split",
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        variant="stock",
        experiment_id="unit-manifest",
        invocation_command="unit invocation",
        output_checkpoint_path=tmp_path / "best.ckpt",
        output_metrics_path=tmp_path / "metrics.json",
        **kwargs,
    )


def test_build_planned_manifest_defaults_to_c0_and_preserves_explicit_git_sha(tmp_path: Path):
    manifest = _build_test_manifest(tmp_path)
    expected_config = canonical_experiment_config(CYP002_C0, "stock")

    assert manifest["configuration"] == "C0"
    assert manifest["canonical_experiment_config"] == expected_config
    assert manifest["experiment_config_sha256"] == experiment_config_sha256(
        expected_config
    )
    assert manifest["code_identity"]["git_commit_sha"] == (
        "fbb27f0a65a1d536154f4e43616bb5208eb62895"
    )
    assert manifest["code_identity"]["repo_dirty"] is False


def test_build_planned_manifest_selects_c1_from_existing_grid(tmp_path: Path):
    manifest = _build_test_manifest(tmp_path, config_name="C1")
    c1_config = next(config for config in CYP002_GRID if config.name == "C1")
    expected_config = canonical_experiment_config(c1_config, "stock")

    assert manifest["configuration"] == "C1"
    assert manifest["canonical_experiment_config"] == expected_config
    assert manifest["experiment_config_sha256"] == experiment_config_sha256(
        expected_config
    )
    assert manifest["canonical_experiment_config"]["architecture"]["depth"] == 2
    assert manifest["canonical_experiment_config"]["architecture"]["message_hidden_dim"] == 300
    assert manifest["canonical_experiment_config"]["architecture"]["ffn_hidden_dim"] == 300
    assert manifest["canonical_experiment_config"]["architecture"]["ffn_num_layers"] == 1
    assert manifest["canonical_experiment_config"]["architecture"]["dropout"] == 0.0
    assert manifest["canonical_experiment_config"]["training"]["max_lr"] == 1e-3
    assert manifest["code_identity"]["git_commit_sha"] == (
        "fbb27f0a65a1d536154f4e43616bb5208eb62895"
    )


def run_tests() -> None:
    tests = [
        test_canonical_json_serialization_is_frozen,
        test_experiment_config_contains_load_bearing_decisions,
        test_default_seed_reproduces_explicit_20261001_config,
        test_different_seed_changes_only_seed_fields,
        test_stock_and_task_balanced_configs_have_different_hashes,
        test_sha256_file_matches_standard_sha256,
        test_finalize_manifest_preserves_experiment_config_hash,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"PASS: all {len(tests)} manifest tests")


if __name__ == "__main__":
    run_tests()
