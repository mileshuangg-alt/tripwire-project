from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multitask_targets import build_cyp_target_matrix


def test_build_cyp_target_matrix_preserves_values_and_masks_missing():
    dataframe = pd.DataFrame(
        {
            "CYP1A2_pIC50_direct_inhibition": [7.1, np.nan, 6.5],
            "CYP2C9_pIC50_direct_inhibition": [np.nan, 5.2, 6.0],
            "CYP2D6_pIC50_direct_inhibition": [6.8, 5.9, np.nan],
            "CYP3A4_pIC50_direct_inhibition": [8.0, np.nan, 7.2],
        }
    )

    targets, mask = build_cyp_target_matrix(dataframe)

    assert targets.shape == (3, 4)
    assert mask.shape == (3, 4)

    expected_mask = np.array(
        [
            [True, False, True, True],
            [False, True, True, False],
            [True, True, False, True],
        ],
        dtype=bool,
    )

    np.testing.assert_array_equal(mask, expected_mask)

    assert targets[0, 0] == 7.1
    assert targets[1, 1] == 5.2
    assert targets[2, 3] == 7.2

    assert np.isnan(targets[0, 1])
    assert np.isnan(targets[1, 0])
    assert np.isnan(targets[2, 2])

    # Missing is not encoded as zero.
    assert targets[0, 1] != 0.0


def test_build_cyp_target_matrix_uses_frozen_task_order():
    dataframe = pd.DataFrame(
        {
            "CYP3A4_pIC50_direct_inhibition": [8.0],
            "CYP2D6_pIC50_direct_inhibition": [7.0],
            "CYP2C9_pIC50_direct_inhibition": [6.0],
            "CYP1A2_pIC50_direct_inhibition": [5.0],
        }
    )

    targets, mask = build_cyp_target_matrix(dataframe)

    np.testing.assert_array_equal(
        targets[0],
        np.array([5.0, 6.0, 7.0, 8.0]),
    )
    np.testing.assert_array_equal(
        mask[0],
        np.array([True, True, True, True]),
    )


def test_build_cyp_target_matrix_rejects_missing_columns():
    dataframe = pd.DataFrame(
        {
            "CYP1A2_pIC50_direct_inhibition": [7.1],
            "CYP2C9_pIC50_direct_inhibition": [6.2],
            "CYP2D6_pIC50_direct_inhibition": [6.8],
        }
    )

    with pytest.raises(KeyError):
        build_cyp_target_matrix(dataframe)


def test_build_cyp_target_matrix_rejects_non_numeric_observed_values():
    dataframe = pd.DataFrame(
        {
            "CYP1A2_pIC50_direct_inhibition": [7.1],
            "CYP2C9_pIC50_direct_inhibition": ["not_a_measurement"],
            "CYP2D6_pIC50_direct_inhibition": [6.8],
            "CYP3A4_pIC50_direct_inhibition": [8.0],
        }
    )

    with pytest.raises(ValueError):
        build_cyp_target_matrix(dataframe)
