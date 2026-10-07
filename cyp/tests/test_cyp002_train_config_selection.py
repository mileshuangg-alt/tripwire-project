from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import cyp002_train as train
from cyp002_config import (
    CYP002_GRID,
    CYP002_TASK_NAMES,
    CYP002_VARIANT_STOCK,
)


CONFIG_BY_NAME = {config.name: config for config in CYP002_GRID}


def _exercise_config_selection(monkeypatch, tmp_path: Path, config_name: str):
    selected_config = CONFIG_BY_NAME[config_name]
    captured = {
        "contract_config_names": [],
        "dataloader_batch_sizes": [],
        "model_configs": [],
        "callback_kwargs": [],
        "trainer_kwargs": [],
        "finalize_kwargs": [],
    }

    manifest_path = tmp_path / config_name / "run_manifest.json"
    manifest_path.parent.mkdir()

    def fake_assert_manifest_contract(manifest_path_arg, variant, config_name_arg):
        captured["contract_config_names"].append(config_name_arg)
        assert manifest_path_arg == manifest_path
        assert variant == CYP002_VARIANT_STOCK
        return {
            "experiment_id": f"unit-{config_name}",
            "configuration": config_name_arg,
            "variant": variant,
            "execution": {"status": "planned"},
            "outputs": {
                "best_checkpoint_sha256": None,
                "metrics_sha256": None,
            },
            "environment": {"hostname": "n4pu04"},
        }

    validation_rows = {
        f"{task}_conf_low": [0.0]
        for task in CYP002_TASK_NAMES
    }
    validation_rows.update(
        {f"{task}_conf_high": [1.0] for task in CYP002_TASK_NAMES}
    )
    validation_df = pd.DataFrame(validation_rows)
    train_df = pd.DataFrame({"placeholder": [1]})

    def fake_build_dataframes(dataset, split_dataframe):
        return train_df, validation_df

    def fake_build_datasets(train_dataframe, validation_dataframe):
        assert train_dataframe is train_df
        assert validation_dataframe is validation_df
        return "train_dataset", "validation_dataset", "scaler"

    def fake_build_dataloader(dataset, batch_size, **kwargs):
        captured["dataloader_batch_sizes"].append(batch_size)
        return {"dataset": dataset, "batch_size": batch_size}

    class FakeModel:
        validation_history = [
            {"epoch": 2, "macro_st_rae": 0.25},
            {"epoch": 1, "macro_st_rae": 0.5},
        ]

    def fake_build_model(config, variant, **kwargs):
        captured["model_configs"].append(config)
        assert config is selected_config
        assert variant == CYP002_VARIANT_STOCK
        assert kwargs["output_scaler"] == "scaler"
        for task in CYP002_TASK_NAMES:
            assert kwargs["validation_conf_low"][task].tolist() == [0.0]
            assert kwargs["validation_conf_high"][task].tolist() == [1.0]
        return FakeModel()

    def fake_build_callbacks(checkpoint_dir, **kwargs):
        captured["callback_kwargs"].append(
            {"checkpoint_dir": checkpoint_dir, **kwargs}
        )
        return ["callbacks"]

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"].append(kwargs)

        def fit(self, model, train_dataloaders, val_dataloaders):
            assert train_dataloaders["batch_size"] == selected_config.batch_size
            assert val_dataloaders["batch_size"] == selected_config.batch_size
            (manifest_path.parent / "best.ckpt").write_text("checkpoint")

    def fake_finalize_manifest(**kwargs):
        captured["finalize_kwargs"].append(kwargs)

    monkeypatch.setattr(train, "_configure_determinism", lambda: None)
    monkeypatch.setattr(train, "_assert_manifest_contract", fake_assert_manifest_contract)
    monkeypatch.setattr(train.pd, "read_csv", lambda path: object())
    monkeypatch.setattr(train, "build_cyp002_dataframes", fake_build_dataframes)
    monkeypatch.setattr(train, "build_cyp002_datasets", fake_build_datasets)
    monkeypatch.setattr(train.data, "build_dataloader", fake_build_dataloader)
    monkeypatch.setattr(train, "build_cyp002_model", fake_build_model)
    monkeypatch.setattr(train, "build_validation_callbacks", fake_build_callbacks)
    monkeypatch.setattr(train.pl, "Trainer", FakeTrainer)
    monkeypatch.setattr(train, "finalize_manifest", fake_finalize_manifest)

    train.run_training(
        variant=CYP002_VARIANT_STOCK,
        config_name=config_name,
        manifest_path=manifest_path,
        dataset_path=tmp_path / "dataset.csv",
        split_path=tmp_path / "split.csv",
    )

    metrics = json.loads((manifest_path.parent / "metrics.json").read_text())
    return selected_config, captured, metrics


def test_c0_selection_preserves_existing_training_plumbing(monkeypatch, tmp_path):
    config, captured, metrics = _exercise_config_selection(
        monkeypatch,
        tmp_path,
        "C0",
    )

    assert captured["contract_config_names"] == ["C0"]
    assert captured["model_configs"] == [config]
    assert captured["dataloader_batch_sizes"] == [config.batch_size, config.batch_size]
    assert captured["callback_kwargs"] == [
        {
            "checkpoint_dir": tmp_path / "C0",
            "patience": config.early_stopping_patience,
            "min_delta": config.early_stopping_min_delta,
        }
    ]
    assert captured["trainer_kwargs"][0]["max_epochs"] == config.max_epochs
    assert metrics["configuration"] == "C0"
    assert metrics["best_validation"] == {"epoch": 2, "macro_st_rae": 0.25}
    assert metrics["test_evaluation_performed"] is False


def test_c1_selection_propagates_frozen_grid_config(monkeypatch, tmp_path):
    config, captured, metrics = _exercise_config_selection(
        monkeypatch,
        tmp_path,
        "C1",
    )

    assert config.message_passing_depth == 2
    assert config.message_hidden_dim == 300
    assert config.ffn_hidden_dim == 300
    assert config.ffn_num_layers == 1
    assert config.dropout == 0.0
    assert config.max_lr == 1e-3
    assert captured["contract_config_names"] == ["C1"]
    assert captured["model_configs"] == [config]
    assert captured["dataloader_batch_sizes"] == [config.batch_size, config.batch_size]
    assert captured["callback_kwargs"] == [
        {
            "checkpoint_dir": tmp_path / "C1",
            "patience": config.early_stopping_patience,
            "min_delta": config.early_stopping_min_delta,
        }
    ]
    assert captured["trainer_kwargs"][0]["max_epochs"] == config.max_epochs
    assert metrics["configuration"] == "C1"
    assert metrics["test_evaluation_performed"] is False
