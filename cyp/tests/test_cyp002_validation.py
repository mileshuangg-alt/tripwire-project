from __future__ import annotations

import numpy as np

from cyp002_config import CYP002_TASK_NAMES
from cyp002_validation import compute_validation_macro_st_rae
from st_rae import st_rae_metric


def test_validation_macro_matches_direct_qualified_scorer():
    y_true = {}
    y_pred = {}
    conf_low = {}
    conf_high = {}

    for task_index, task_name in enumerate(CYP002_TASK_NAMES):
        y_true[task_name] = np.array([5.0, 6.0, 7.0, 8.0])
        y_pred[task_name] = np.array(
            [5.0 + 0.1 * (task_index + 1),
             6.0 + 0.1 * (task_index + 1),
             7.0 + 0.1 * (task_index + 1),
             8.0 + 0.1 * (task_index + 1)]
        )
        conf_low[task_name] = np.array([4.9, 5.9, 6.9, 7.9])
        conf_high[task_name] = np.array([5.1, 6.1, 7.1, 8.1])

    result = compute_validation_macro_st_rae(
        y_true,
        y_pred,
        conf_low,
        conf_high,
        CYP002_TASK_NAMES,
    )

    expected_per_task = {
        task_name: st_rae_metric(
            y_true[task_name],
            y_pred[task_name],
            conf_low[task_name],
            conf_high[task_name],
        )
        for task_name in CYP002_TASK_NAMES
    }

    expected_macro = np.mean(
        [expected_per_task[name] for name in CYP002_TASK_NAMES]
    )

    assert result["task_order"] == list(CYP002_TASK_NAMES)

    for task_name in CYP002_TASK_NAMES:
        np.testing.assert_allclose(
            result["per_task"][task_name],
            expected_per_task[task_name],
            atol=1e-12,
            rtol=0.0,
        )

    np.testing.assert_allclose(
        result["macro_st_rae"],
        expected_macro,
        atol=1e-12,
        rtol=0.0,
    )


def test_task_order_is_preserved():
    values = {}
    predictions = {}
    low = {}
    high = {}

    for task_name in CYP002_TASK_NAMES:
        values[task_name] = np.array([5.0, 6.0])
        predictions[task_name] = np.array([5.1, 6.1])
        low[task_name] = np.array([4.9, 5.9])
        high[task_name] = np.array([5.1, 6.1])

    result = compute_validation_macro_st_rae(
        values,
        predictions,
        low,
        high,
        CYP002_TASK_NAMES,
    )

    assert result["task_order"] == list(CYP002_TASK_NAMES)
    assert list(result["per_task"]) == list(CYP002_TASK_NAMES)


def run_tests() -> None:
    tests = [
        test_validation_macro_matches_direct_qualified_scorer,
        test_task_order_is_preserved,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"PASS: all {len(tests)} CYP-002 validation tests")


if __name__ == "__main__":
    run_tests()
