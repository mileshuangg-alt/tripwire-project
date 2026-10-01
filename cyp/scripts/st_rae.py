from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from references.cyp_upstream.custom_scoring_functions import (
    rae_soft_threshold_absolute_error,
)


def st_rae_metric(
    y_true,
    y_pred,
    conf_low,
    conf_high,
) -> float:
    """Compute the official OpenADMET ST-RAE for one isoform."""
    result = rae_soft_threshold_absolute_error(
        y_true=y_true,
        y_pred=y_pred,
        y_true_lower=conf_low,
        y_true_upper=conf_high,
    )
    return result


def make_st_rae_evaluator(y_true, conf_low, conf_high):
    """Create a LightGBM sklearn custom evaluator for official ST-RAE."""

    def evaluate(y_true_unused, y_pred):
        result = st_rae_metric(
            y_true=y_true,
            y_pred=y_pred,
            conf_low=conf_low,
            conf_high=conf_high,
        )
        return "ST-RAE", result, False

    return evaluate


def check_st_rae_invariants() -> None:
    # Non-degenerate case: the mean predictor must have nonzero
    # soft-thresholded error so the official RAE denominator is finite.
    y_true = np.array([4.0, 5.0, 6.0])
    conf_low = np.array([3.5, 4.5, 5.5])
    conf_high = np.array([4.5, 5.5, 6.5])

    inside = np.array([4.0, 5.0, 6.0])
    outside = np.array([6.0, 5.0, 4.0])

    inside_score = st_rae_metric(
        y_true,
        inside,
        conf_low,
        conf_high,
    )

    outside_score = st_rae_metric(
        y_true,
        outside,
        conf_low,
        conf_high,
    )

    if not np.isfinite(inside_score):
        raise AssertionError("Inside-band ST-RAE is not finite.")

    if not np.isfinite(outside_score):
        raise AssertionError("Outside-band ST-RAE is not finite.")

    if not np.isclose(inside_score, 0.0):
        raise AssertionError(
            f"Expected zero ST-RAE for entirely in-band predictions; "
            f"got {inside_score}"
        )

    expected_outside_numerator = 3.0

    observed_outside_numerator = (
        np.clip(outside - conf_high, a_min=0.0, a_max=None)
        + np.clip(conf_low - outside, a_min=0.0, a_max=None)
    ).sum()

    if not np.isclose(
        observed_outside_numerator,
        expected_outside_numerator,
    ):
        raise AssertionError(
            "Soft-threshold numerator invariant failed: "
            f"expected {expected_outside_numerator}, "
            f"got {observed_outside_numerator}"
        )

    evaluator = make_st_rae_evaluator(
        y_true=y_true,
        conf_low=conf_low,
        conf_high=conf_high,
    )

    callback_name, callback_value, lower_is_better = evaluator(
        y_true,
        outside,
    )

    direct_value = st_rae_metric(
        y_true=y_true,
        y_pred=outside,
        conf_low=conf_low,
        conf_high=conf_high,
    )

    if callback_name != "ST-RAE":
        raise AssertionError(
            f"Expected callback metric name ST-RAE; got {callback_name}"
        )

    if lower_is_better is not False:
        raise AssertionError(
            "ST-RAE must be configured as a minimization metric."
        )

    if not np.isclose(callback_value, direct_value):
        raise AssertionError(
            "LightGBM callback does not match direct ST-RAE calculation: "
            f"{callback_value} != {direct_value}"
        )

    print("PASS: official ST-RAE wrapper and LightGBM callback.")


if __name__ == "__main__":
    check_st_rae_invariants()
