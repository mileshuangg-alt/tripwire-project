"""PXR-001 classical baseline data preparation."""

from __future__ import annotations

import pandas as pd


ID_COLUMN = "Molecule Name"
SMILES_COLUMN = "SMILES"
TARGET_COLUMN = "pEC50"
SPLIT_COLUMN = "split"

EXPECTED_SPLITS = {"train", "validation", "holdout"}


def prepare_train_validation_data(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate frozen PXR identities and return train/validation only.

    The final holdout target is masked before constructing the returned
    modeling datasets. No observations are silently excluded.
    """
    required_data = {ID_COLUMN, SMILES_COLUMN, TARGET_COLUMN}
    required_assignments = {ID_COLUMN, SPLIT_COLUMN}

    missing_data = required_data - set(data.columns)
    missing_assignments = required_assignments - set(assignments.columns)

    if missing_data:
        raise ValueError(
            f"TRAIN data missing required columns: {sorted(missing_data)}"
        )

    if missing_assignments:
        raise ValueError(
            "Split assignments missing required columns: "
            f"{sorted(missing_assignments)}"
        )

    if data[ID_COLUMN].isna().any():
        raise ValueError("TRAIN data contains null molecule identities.")

    if assignments[ID_COLUMN].isna().any():
        raise ValueError("Assignments contain null molecule identities.")

    if data[ID_COLUMN].duplicated().any():
        raise ValueError("TRAIN data contains duplicate molecule identities.")

    if assignments[ID_COLUMN].duplicated().any():
        raise ValueError("Assignments contain duplicate molecule identities.")

    if assignments[SPLIT_COLUMN].isna().any():
        raise ValueError("Assignments contain unassigned molecules.")

    data_ids = set(data[ID_COLUMN])
    assignment_ids = set(assignments[ID_COLUMN])

    if data_ids != assignment_ids:
        missing = sorted(data_ids - assignment_ids)[:5]
        extra = sorted(assignment_ids - data_ids)[:5]
        raise ValueError(
            "TRAIN/assignment identity sets differ: "
            f"missing assignments={missing}, extra assignments={extra}"
        )

    observed_splits = set(assignments[SPLIT_COLUMN].unique())
    if observed_splits != EXPECTED_SPLITS:
        raise ValueError(
            f"Expected partitions {sorted(EXPECTED_SPLITS)}, "
            f"found {sorted(observed_splits)}"
        )

    merged = data.merge(
        assignments[[ID_COLUMN, SPLIT_COLUMN]],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != len(data):
        raise AssertionError(
            f"Join changed row count: {len(data)} -> {len(merged)}"
        )

    # The final holdout is not exposed to the iterative modeling datasets.
    holdout_mask = merged[SPLIT_COLUMN].eq("holdout")
    merged[TARGET_COLUMN] = merged[TARGET_COLUMN].where(~holdout_mask)

    train = merged.loc[merged[SPLIT_COLUMN].eq("train")].copy()
    validation = merged.loc[
        merged[SPLIT_COLUMN].eq("validation")
    ].copy()

    if train.empty or validation.empty:
        raise ValueError("Training or validation partition is empty.")

    for name, frame in (
        ("train", train),
        ("validation", validation),
    ):
        if frame[[SMILES_COLUMN, TARGET_COLUMN]].isna().any().any():
            raise ValueError(
                f"{name} partition contains missing SMILES or pEC50."
            )

    overlap = set(train[ID_COLUMN]) & set(validation[ID_COLUMN])
    if overlap:
        raise AssertionError(
            f"Train/validation identity overlap: {sorted(overlap)[:5]}"
        )

    result = train, validation
    return result


def build_ecfp4_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict:
    """Build ECFP4 matrices and aligned pEC50 arrays for train/validation."""
    import numpy as np

    from cyp.scripts.ecfp4 import N_BITS, ecfp4_matrix

    for name, frame in (
        ("train", train),
        ("validation", validation),
    ):
        if frame.empty:
            raise ValueError(f"{name} partition is empty.")

        required = {SMILES_COLUMN, TARGET_COLUMN}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} partition missing columns: {sorted(missing)}"
            )

        if frame[[SMILES_COLUMN, TARGET_COLUMN]].isna().any().any():
            raise ValueError(
                f"{name} partition contains missing SMILES or pEC50."
            )

    x_train = ecfp4_matrix(train[SMILES_COLUMN].tolist())
    x_validation = ecfp4_matrix(validation[SMILES_COLUMN].tolist())

    y_train = pd.to_numeric(
        train[TARGET_COLUMN], errors="raise"
    ).to_numpy(dtype=np.float64)

    y_validation = pd.to_numeric(
        validation[TARGET_COLUMN], errors="raise"
    ).to_numpy(dtype=np.float64)

    for name, matrix, expected_rows in (
        ("train", x_train, len(train)),
        ("validation", x_validation, len(validation)),
    ):
        if matrix.shape != (expected_rows, N_BITS):
            raise AssertionError(
                f"{name} fingerprint shape is {matrix.shape}; "
                f"expected {(expected_rows, N_BITS)}."
            )

        if matrix.dtype != np.uint8:
            raise AssertionError(
                f"{name} fingerprint dtype is {matrix.dtype}; "
                "expected uint8."
            )

        if not np.isin(matrix, [0, 1]).all():
            raise AssertionError(
                f"{name} fingerprint matrix contains non-binary values."
            )

    if not np.isfinite(y_train).all():
        raise ValueError("Training targets contain non-finite values.")

    if not np.isfinite(y_validation).all():
        raise ValueError("Validation targets contain non-finite values.")

    result = {
        "x_train": x_train,
        "x_validation": x_validation,
        "y_train": y_train,
        "y_validation": y_validation,
    }
    return result
