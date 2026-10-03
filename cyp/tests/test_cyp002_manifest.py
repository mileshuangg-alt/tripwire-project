from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cyp002_config import CYP002_C0
from cyp002_manifest import (
    canonical_json_bytes,
    canonical_experiment_config,
    experiment_config_sha256,
    finalize_manifest,
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


def run_tests() -> None:
    tests = [
        test_canonical_json_serialization_is_frozen,
        test_experiment_config_contains_load_bearing_decisions,
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
