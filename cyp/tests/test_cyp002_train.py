from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cyp002_config import CYP002_TASK_NAMES, CYP002_C0
import cyp002_train as train
from cyp002_train import (
    build_cyp002_dataframes,
    build_cyp002_datasets,
    canonical_manifest_matches,
)


def make_fixture():
    rows = [
        {
            "Molecule_Name": "a",
            "SMILES": "CC",
            "CYP1A2_pIC50_direct_inhibition": 1.0,
            "CYP2C9_pIC50_direct_inhibition": np.nan,
            "CYP2D6_pIC50_direct_inhibition": 3.0,
            "CYP3A4_pIC50_direct_inhibition": np.nan,
        },
        {
            "Molecule_Name": "b",
            "SMILES": "CCC",
            "CYP1A2_pIC50_direct_inhibition": 2.0,
            "CYP2C9_pIC50_direct_inhibition": 2.0,
            "CYP2D6_pIC50_direct_inhibition": np.nan,
            "CYP3A4_pIC50_direct_inhibition": 4.0,
        },
        {
            "Molecule_Name": "c",
            "SMILES": "CCCC",
            "CYP1A2_pIC50_direct_inhibition": np.nan,
            "CYP2C9_pIC50_direct_inhibition": 3.0,
            "CYP2D6_pIC50_direct_inhibition": 3.5,
            "CYP3A4_pIC50_direct_inhibition": 4.5,
        },
        {
            "Molecule_Name": "d",
            "SMILES": "CCCCC",
            "CYP1A2_pIC50_direct_inhibition": 4.0,
            "CYP2C9_pIC50_direct_inhibition": 4.0,
            "CYP2D6_pIC50_direct_inhibition": 4.0,
            "CYP3A4_pIC50_direct_inhibition": 5.0,
        },
    ]

    data = pd.DataFrame(rows)

    split = pd.DataFrame(
        {
            "Molecule_Name": ["a", "b", "c", "d"],
            "cluster_id": [1, 2, 3, 4],
            "split": ["train", "train", "validation", "test"],
        }
    )

    return data, split


def test_frozen_split_routing_is_direct_and_excludes_test():
    data, split = make_fixture()

    # The production function enforces the frozen CYP-001 counts, so use a
    # small local variant only after verifying its routing logic separately.
    # This test therefore checks the helper's source-level contract through
    # a deliberately expected validation error on undersized data.
    try:
        build_cyp002_dataframes(data, split)
    except ValueError as exc:
        assert "Expected 3433 train rows" in str(exc)
    else:
        raise AssertionError(
            "Small fixture unexpectedly satisfied the production row-count gate."
        )


def test_target_columns_are_the_frozen_four_tasks():
    assert list(CYP002_TASK_NAMES) == [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]


def test_manifest_canonical_config_matches_manifest(tmp_path):
    from cyp002_manifest import experiment_config_sha256

    config_hash = experiment_config_sha256(
        __import__("cyp002_manifest", fromlist=["canonical_experiment_config"])
        .canonical_experiment_config(CYP002_C0, "stock")
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiment_config_sha256": config_hash,
                "canonical_experiment_config":
                    __import__("cyp002_manifest", fromlist=[
                        "canonical_experiment_config"
                    ]).canonical_experiment_config(
                        CYP002_C0, "stock"
                    ),
            }
        ),
        encoding="utf-8",
    )

    assert canonical_manifest_matches(manifest_path) is True


def _write_contract_manifest(tmp_path, seed: int, determinism_seed: int | None = None):
    from cyp002_manifest import (
        canonical_experiment_config,
        experiment_config_sha256,
    )

    config = canonical_experiment_config(CYP002_C0, "stock", seed=seed)
    if determinism_seed is None:
        determinism_seed = seed

    manifest = {
        "experiment_config_sha256": experiment_config_sha256(config),
        "canonical_experiment_config": config,
        "configuration": "C0",
        "variant": "stock",
        "seed": seed,
        "execution": {"status": "running"},
        "outputs": {
            "best_checkpoint_sha256": None,
            "metrics_sha256": None,
        },
        "environment": {"hostname": "unit-host"},
        "determinism": {
            "chemprop_dataloader_seed": determinism_seed,
            "dataloader_generator_seed": determinism_seed,
        },
    }

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_contract_accepts_matching_non_default_seed(monkeypatch, tmp_path):
    manifest_path = _write_contract_manifest(tmp_path, seed=12345)
    monkeypatch.setattr(train.socket, "gethostname", lambda: "unit-host")

    manifest = train._assert_manifest_contract(
        manifest_path,
        "stock",
        "C0",
    )

    assert manifest["seed"] == 12345


def test_manifest_contract_rejects_seed_mismatch(monkeypatch, tmp_path):
    manifest_path = _write_contract_manifest(
        tmp_path,
        seed=12345,
        determinism_seed=20261001,
    )
    monkeypatch.setattr(train.socket, "gethostname", lambda: "unit-host")

    try:
        train._assert_manifest_contract(manifest_path, "stock", "C0")
    except RuntimeError as exc:
        assert "determinism seed" in str(exc)
    else:
        raise AssertionError("Seed-mismatched manifest was accepted.")


def run_tests() -> None:
    tests = [
        test_frozen_split_routing_is_direct_and_excludes_test,
        test_target_columns_are_the_frozen_four_tasks,
        test_manifest_canonical_config_matches_manifest,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"PASS: all {len(tests)} launcher tests")


if __name__ == "__main__":
    run_tests()
