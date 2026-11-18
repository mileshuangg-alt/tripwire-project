import numpy as np
import pytest

from training_uncertainty_weights import (
    InvalidObservationUncertainty,
    curve_fit_training_weights,
    screen_training_weights,
)


def test_curve_fit_uses_bounded_inverse_width():
    low = np.array([4.0, 5.0])
    high = np.array([5.0, 8.0])

    weights = curve_fit_training_weights(
        low,
        high,
        [True, True],
        ["global", "global"],
        ["A", "B"],
    )

    raw = np.array([1.0 / 1.25, 1.0 / 3.25])
    expected = raw / raw.mean()

    np.testing.assert_allclose(weights, expected)
    assert weights.mean() == pytest.approx(1.0)


def test_curve_fit_normalizes_independently_by_reduction_group():
    weights = curve_fit_training_weights(
        [4.0, 5.0, 3.0, 4.0],
        [5.0, 8.0, 7.0, 9.0],
        [True] * 4,
        ["task_A", "task_A", "task_B", "task_B"],
        ["A1", "A2", "B1", "B2"],
    )

    assert weights[:2].mean() == pytest.approx(1.0)
    assert weights[2:].mean() == pytest.approx(1.0)


def test_screen_uses_inverse_variance():
    weights = screen_training_weights(
        [0.5, 1.0],
        [True, True],
        ["global", "global"],
        ["A", "B"],
    )

    np.testing.assert_allclose(weights, [1.6, 0.4])
    assert weights.mean() == pytest.approx(1.0)


def test_screen_normalizes_within_groups():
    weights = screen_training_weights(
        [0.5, 1.0, 0.25, 0.5],
        [True] * 4,
        ["enzyme_A", "enzyme_A", "enzyme_B", "enzyme_B"],
        ["A1", "A2", "B1", "B2"],
    )

    assert weights[:2].mean() == pytest.approx(1.0)
    assert weights[2:].mean() == pytest.approx(1.0)


def test_existing_missing_label_mask_is_preserved():
    weights = screen_training_weights(
        [0.5, np.nan, 1.0],
        [True, False, True],
        ["global"] * 3,
        ["A", "MISSING_LABEL", "C"],
    )

    assert weights[1] == 0.0
    assert weights[0] > 0
    assert weights[2] > 0


def test_observed_screen_nan_fails_with_row_identity():
    with pytest.raises(InvalidObservationUncertainty, match="COMPOUND_7"):
        screen_training_weights(
            [np.nan],
            [True],
            ["global"],
            ["COMPOUND_7"],
        )


def test_observed_screen_nonpositive_se_fails_with_row_identity():
    with pytest.raises(InvalidObservationUncertainty, match="COMPOUND_9"):
        screen_training_weights(
            [0.0],
            [True],
            ["global"],
            ["COMPOUND_9"],
        )


def test_observed_curve_fit_missing_bound_fails_with_row_identity():
    with pytest.raises(InvalidObservationUncertainty, match="COMPOUND_11"):
        curve_fit_training_weights(
            [np.nan],
            [5.0],
            [True],
            ["global"],
            ["COMPOUND_11"],
        )


def test_invalid_uncertainty_in_masked_row_is_not_an_error():
    weights = curve_fit_training_weights(
        [np.nan, 4.0],
        [np.nan, 5.0],
        [False, True],
        ["global", "global"],
        ["MISSING", "OBSERVED"],
    )

    assert weights[0] == 0.0
    assert weights[1] == pytest.approx(1.0)


def test_shape_mismatch_fails():
    with pytest.raises(ValueError, match="shape mismatch"):
        curve_fit_training_weights(
            [4.0, 5.0],
            [5.0],
            [True, True],
            ["global", "global"],
            ["A", "B"],
        )
