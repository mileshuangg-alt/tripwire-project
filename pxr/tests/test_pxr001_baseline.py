"""Tests for PXR-001 classical baseline data preparation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts.pxr001_baseline import prepare_train_validation_data


def synthetic_inputs():
    data = pd.DataFrame(
        {
            "Molecule Name": ["A", "B", "C", "D"],
            "SMILES": ["CCO", "CCN", "CCC", "c1ccccc1"],
            "pEC50": [4.0, 4.5, 5.0, 7.0],
        }
    )

    assignments = pd.DataFrame(
        {
            "Molecule Name": ["A", "B", "C", "D"],
            "split": ["train", "train", "validation", "holdout"],
        }
    )

    return data, assignments


def test_returns_only_train_and_validation():
    data, assignments = synthetic_inputs()

    train, validation = prepare_train_validation_data(
        data,
        assignments,
    )

    assert set(train["Molecule Name"]) == {"A", "B"}
    assert set(validation["Molecule Name"]) == {"C"}
    assert "D" not in set(train["Molecule Name"])
    assert "D" not in set(validation["Molecule Name"])


def test_does_not_mutate_frozen_inputs():
    data, assignments = synthetic_inputs()
    original_data = data.copy(deep=True)
    original_assignments = assignments.copy(deep=True)

    prepare_train_validation_data(data, assignments)

    pd.testing.assert_frame_equal(data, original_data)
    pd.testing.assert_frame_equal(assignments, original_assignments)


def test_rejects_identity_set_mismatch():
    data, assignments = synthetic_inputs()
    assignments = assignments.iloc[:-1].copy()

    with pytest.raises(ValueError, match="identity sets differ"):
        prepare_train_validation_data(data, assignments)


def test_rejects_duplicate_train_identity():
    data, assignments = synthetic_inputs()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate molecule identities"):
        prepare_train_validation_data(data, assignments)


def test_rejects_duplicate_assignment_identity():
    data, assignments = synthetic_inputs()
    assignments = pd.concat(
        [assignments, assignments.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate molecule identities"):
        prepare_train_validation_data(data, assignments)


def test_rejects_unknown_partition():
    data, assignments = synthetic_inputs()
    assignments.loc[3, "split"] = "test"

    with pytest.raises(ValueError, match="Expected partitions"):
        prepare_train_validation_data(data, assignments)


def test_rejects_missing_training_target():
    data, assignments = synthetic_inputs()
    data.loc[0, "pEC50"] = float("nan")

    with pytest.raises(ValueError, match="missing SMILES or pEC50"):
        prepare_train_validation_data(data, assignments)


def test_real_pxr_artifacts_join_and_partition_correctly():
    """Verify the adapter against the preserved PXR TRAIN and D014 split."""
    train_path = (
        REPO_ROOT
        / "pxr/data/openadmet_pxr_release_2026-11-15/pxr-challenge_TRAIN.csv"
    )
    assignments_path = (
        REPO_ROOT
        / "pxr/artifacts/pxr001_split/pxr001_split_assignments.csv"
    )

    assert train_path.is_file()
    assert assignments_path.is_file()

    data = pd.read_csv(train_path)
    assignments = pd.read_csv(assignments_path)

    assert len(data) == 4139
    assert len(assignments) == 4139

    train, validation = prepare_train_validation_data(
        data,
        assignments,
    )

    assert len(train) == 2897
    assert len(validation) == 621

    train_ids = set(train["Molecule Name"])
    validation_ids = set(validation["Molecule Name"])
    holdout_ids = set(
        assignments.loc[
            assignments["split"].eq("holdout"),
            "Molecule Name",
        ]
    )

    assert len(holdout_ids) == 621
    assert len(train_ids & validation_ids) == 0
    assert len(train_ids & holdout_ids) == 0
    assert len(validation_ids & holdout_ids) == 0

    assert train["pEC50"].notna().all()
    assert validation["pEC50"].notna().all()
    assert train["SMILES"].notna().all()
    assert validation["SMILES"].notna().all()

    assert train_ids == set(
        assignments.loc[
            assignments["split"].eq("train"),
            "Molecule Name",
        ]
    )
    assert validation_ids == set(
        assignments.loc[
            assignments["split"].eq("validation"),
            "Molecule Name",
        ]
    )





def test_real_pxr_ecfp4_feature_matrices():
    """Verify actual PXR train/validation feature construction."""
    from pxr.scripts.pxr001_baseline import build_ecfp4_features

    train_path = (
        REPO_ROOT
        / "pxr/data/openadmet_pxr_release_2026-11-15/pxr-challenge_TRAIN.csv"
    )
    assignments_path = (
        REPO_ROOT
        / "pxr/artifacts/pxr001_split/pxr001_split_assignments.csv"
    )

    data = pd.read_csv(train_path)
    assignments = pd.read_csv(assignments_path)

    train, validation = prepare_train_validation_data(
        data,
        assignments,
    )

    features = build_ecfp4_features(train, validation)

    assert features["x_train"].shape == (2897, 2048)
    assert features["x_validation"].shape == (621, 2048)

    assert features["y_train"].shape == (2897,)
    assert features["y_validation"].shape == (621,)

    assert features["x_train"].dtype.name == "uint8"
    assert features["x_validation"].dtype.name == "uint8"

    assert set(features) == {
        "x_train",
        "x_validation",
        "y_train",
        "y_validation",
    }
