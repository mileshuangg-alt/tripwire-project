from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


CYP_TARGET_COLUMNS: tuple[str, ...] = (
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
)


def build_cyp_target_matrix(
    dataframe: pd.DataFrame,
    target_columns: Sequence[str] = CYP_TARGET_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert CYP multitask labels into a numeric target matrix and observation mask.

    Parameters
    ----------
    dataframe:
        Input dataframe containing the four CYP endpoint columns.
    target_columns:
        Ordered endpoint names. The default order is frozen for Tripwire-CYP.

    Returns
    -------
    targets:
        Float matrix of shape (n_compounds, n_tasks). Observed labels retain
        their numeric values. Missing labels are represented as NaN.
    mask:
        Boolean matrix of the same shape. True means the corresponding label
        is observed; False means it is missing.

    Raises
    ------
    KeyError
        If any requested target column is absent.
    ValueError
        If an observed target value cannot be converted to float.
    """
    missing_columns = [
        column for column in target_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Missing required CYP target columns: {missing_columns}"
        )

    labels = dataframe.loc[:, list(target_columns)].copy()

    for column in target_columns:
        labels[column] = pd.to_numeric(labels[column], errors="coerce")

    original_missing = dataframe.loc[:, list(target_columns)].isna()
    coerced_missing = labels.isna()

    newly_coerced = coerced_missing & ~original_missing
    if newly_coerced.any().any():
        bad_columns = [
            column
            for column in target_columns
            if newly_coerced[column].any()
        ]
        raise ValueError(
            "Non-numeric observed CYP target values encountered in columns: "
            f"{bad_columns}"
        )

    targets = labels.to_numpy(dtype=np.float64)
    mask = ~np.isnan(targets)

    result = targets, mask
    return result
