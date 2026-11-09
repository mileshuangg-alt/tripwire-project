from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import d008_external_evaluation
from d008_external_evaluation import (
    D008_BOOTSTRAP_ITERATIONS,
    D008_BOOTSTRAP_SEED,
    aggregate_macro_st_rae,
    evaluate_d008_from_paths,
    organizer_compatible_bootstrap,
    route_endpoint_rows,
    score_endpoint_st_rae,
    serialize_d008_result,
    sha256_file,
    validate_and_align_external_rows,
)
from st_rae import st_rae_metric


def _endpoint_labels():
    return pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-b", "mol-c", "mol-d"],
            "target": [5.0, pd.NA, 7.0, 8.0],
            "conf_low": [4.8, pd.NA, 6.8, 7.8],
            "conf_high": [5.2, pd.NA, 7.2, 8.2],
        }
    )


def _endpoint_predictions():
    return pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-b", "mol-c", "mol-d"],
            "prediction": [5.1, pd.NA, 7.1, 8.1],
        }
    )


def _route(labels=None, predictions=None):
    return route_endpoint_rows(
        _endpoint_labels() if labels is None else labels,
        _endpoint_predictions() if predictions is None else predictions,
        target_column="target",
        prediction_column="prediction",
        conf_low_column="conf_low",
        conf_high_column="conf_high",
    )


def _endpoint_score(endpoint, score, n_scored=10):
    return {
        "st_rae": score,
        "n_total_aligned": 100,
        "n_target_missing": 100 - n_scored,
        "n_scored": n_scored,
        "molecule_ids": [f"{endpoint}-mol-{i}" for i in range(n_scored)],
        "target_column": endpoint,
        "prediction_column": f"{endpoint}_prediction",
        "conf_low_column": f"{endpoint}_conf_low",
        "conf_high_column": f"{endpoint}_conf_high",
        "molecule_id_column": "Molecule_Name",
    }


def _four_endpoint_scores():
    return [
        _endpoint_score("endpoint_a", 0.2, n_scored=5),
        _endpoint_score("endpoint_b", 0.4, n_scored=500),
        _endpoint_score("endpoint_c", 0.6, n_scored=50),
        _endpoint_score("endpoint_d", 0.8, n_scored=1),
    ]


def _bootstrap_endpoint(endpoint, offset):
    y_true = np.linspace(4.0 + offset, 8.75 + offset, 20)
    return {
        "n_total_aligned": 20,
        "n_target_missing": 0,
        "n_scored": 20,
        "molecule_ids": [f"{endpoint}-mol-{i}" for i in range(20)],
        "y_true": y_true,
        "y_pred": y_true + np.sin(np.arange(20) + offset) * 0.35,
        "conf_low": y_true - 0.1,
        "conf_high": y_true + 0.1,
        "target_column": endpoint,
        "prediction_column": f"{endpoint}_prediction",
        "conf_low_column": f"{endpoint}_conf_low",
        "conf_high_column": f"{endpoint}_conf_high",
        "molecule_id_column": "Molecule_Name",
    }


def _four_bootstrap_endpoints():
    return [
        _bootstrap_endpoint("endpoint_a", 0.0),
        _bootstrap_endpoint("endpoint_b", 0.5),
        _bootstrap_endpoint("endpoint_c", 1.0),
        _bootstrap_endpoint("endpoint_d", 1.5),
    ]


def _production_io_fixture(tmp_path):
    names = [f"mol-{i:02d}" for i in range(20)]
    label_columns = ["Molecule_Name"]
    prediction_columns = ["Molecule_Name"]
    label_data = {"Molecule_Name": names}
    prediction_data = {"Molecule_Name": names}
    mappings = []

    for endpoint_index, endpoint in enumerate(
        ["endpoint_a", "endpoint_b", "endpoint_c", "endpoint_d"]
    ):
        target_column = f"{endpoint}_target"
        prediction_column = f"{endpoint}_prediction"
        conf_low_column = f"{endpoint}_conf_low"
        conf_high_column = f"{endpoint}_conf_high"
        offset = endpoint_index * 0.5
        y_true = np.linspace(4.0 + offset, 8.75 + offset, 20)

        label_columns.extend(
            [target_column, conf_low_column, conf_high_column]
        )
        prediction_columns.append(prediction_column)
        label_data[target_column] = y_true
        label_data[conf_low_column] = y_true - 0.1
        label_data[conf_high_column] = y_true + 0.1
        prediction_data[prediction_column] = (
            y_true + np.sin(np.arange(20) + offset) * 0.35
        )
        mappings.append(
            {
                "endpoint": endpoint,
                "target_column": target_column,
                "prediction_column": prediction_column,
                "conf_low_column": conf_low_column,
                "conf_high_column": conf_high_column,
            }
        )

    labels = pd.DataFrame(label_data, columns=label_columns)
    predictions = pd.DataFrame(prediction_data, columns=prediction_columns)
    labels_path = tmp_path / "synthetic_labels.csv"
    predictions_path = tmp_path / "synthetic_predictions.csv"
    labels.to_csv(labels_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    return {
        "labels_path": labels_path,
        "predictions_path": predictions_path,
        "labels_sha256": sha256_file(labels_path),
        "predictions_sha256": sha256_file(predictions_path),
        "label_columns": label_columns,
        "prediction_columns": prediction_columns,
        "mappings": mappings,
    }


def test_shuffled_prediction_rows_align_by_molecule_name():
    labels = pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-b", "mol-c"],
            "endpoint_label": [5.0, 6.0, 7.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "Molecule_Name": ["mol-c", "mol-a", "mol-b"],
            "endpoint_prediction": [70.0, 50.0, 60.0],
        }
    )

    result = validate_and_align_external_rows(labels, predictions)

    assert result["molecule_id_column"] == "Molecule_Name"
    assert result["n_rows"] == 3
    pdt.assert_frame_equal(result["labels"], labels)
    assert result["predictions"]["Molecule_Name"].tolist() == [
        "mol-a",
        "mol-b",
        "mol-c",
    ]
    assert result["predictions"]["endpoint_prediction"].tolist() == [
        50.0,
        60.0,
        70.0,
    ]


def test_missing_endpoint_labels_remain_missing():
    labels = pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-b", "mol-c"],
            "endpoint_label": [5.0, pd.NA, 7.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "Molecule_Name": ["mol-b", "mol-c", "mol-a"],
            "endpoint_prediction": [6.1, 7.1, 5.1],
        }
    )

    result = validate_and_align_external_rows(labels, predictions)

    pdt.assert_frame_equal(result["labels"], labels)
    assert pd.isna(result["labels"].loc[1, "endpoint_label"])
    assert result["predictions"]["Molecule_Name"].tolist() == [
        "mol-a",
        "mol-b",
        "mol-c",
    ]


def test_duplicate_label_identity_fails_closed():
    labels = pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-a"],
            "endpoint_label": [5.0, 6.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-b"],
            "endpoint_prediction": [5.1, 6.1],
        }
    )

    with pytest.raises(ValueError, match="duplicate molecule identifiers"):
        validate_and_align_external_rows(labels, predictions)


def test_duplicate_prediction_identity_fails_closed():
    labels = pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-b"],
            "endpoint_label": [5.0, 6.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "Molecule_Name": ["mol-b", "mol-b"],
            "endpoint_prediction": [6.1, 6.2],
        }
    )

    with pytest.raises(ValueError, match="duplicate molecule identifiers"):
        validate_and_align_external_rows(labels, predictions)


def test_missing_prediction_fails_closed():
    labels = pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-b", "mol-c"],
            "endpoint_label": [5.0, 6.0, 7.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-c"],
            "endpoint_prediction": [5.1, 7.1],
        }
    )

    with pytest.raises(ValueError, match="Missing predictions"):
        validate_and_align_external_rows(labels, predictions)


def test_extra_prediction_fails_closed():
    labels = pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-b"],
            "endpoint_label": [5.0, 6.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "Molecule_Name": ["mol-a", "mol-b", "mol-c"],
            "endpoint_prediction": [5.1, 6.1, 7.1],
        }
    )

    with pytest.raises(ValueError, match="extra predictions"):
        validate_and_align_external_rows(labels, predictions)


def test_missing_identity_column_fails_closed():
    labels = pd.DataFrame({"endpoint_label": [5.0]})
    predictions = pd.DataFrame(
        {"Molecule_Name": ["mol-a"], "endpoint_prediction": [5.1]}
    )

    with pytest.raises(ValueError, match="missing molecule identity column"):
        validate_and_align_external_rows(labels, predictions)

    labels = pd.DataFrame(
        {"Molecule_Name": ["mol-a"], "endpoint_label": [5.0]}
    )
    predictions = pd.DataFrame({"endpoint_prediction": [5.1]})

    with pytest.raises(ValueError, match="missing molecule identity column"):
        validate_and_align_external_rows(labels, predictions)


def test_endpoint_missing_target_rows_are_excluded_and_counts_are_correct():
    result = _route()

    assert result["n_total_aligned"] == 4
    assert result["n_target_missing"] == 1
    assert result["n_scored"] == 3
    assert result["target_column"] == "target"
    assert result["prediction_column"] == "prediction"
    assert result["conf_low_column"] == "conf_low"
    assert result["conf_high_column"] == "conf_high"


def test_endpoint_eligible_rows_preserve_identity_and_order():
    result = _route()

    assert result["molecule_ids"] == ["mol-a", "mol-c", "mol-d"]
    np.testing.assert_allclose(result["y_true"], [5.0, 7.0, 8.0])
    np.testing.assert_allclose(result["y_pred"], [5.1, 7.1, 8.1])
    np.testing.assert_allclose(result["conf_low"], [4.8, 6.8, 7.8])
    np.testing.assert_allclose(result["conf_high"], [5.2, 7.2, 8.2])


def test_endpoint_missing_prediction_on_eligible_row_fails_closed():
    predictions = _endpoint_predictions()
    predictions.loc[2, "prediction"] = pd.NA

    with pytest.raises(ValueError, match="missing prediction values"):
        _route(predictions=predictions)


def test_endpoint_missing_confidence_lower_bound_on_eligible_row_fails_closed():
    labels = _endpoint_labels()
    labels.loc[2, "conf_low"] = pd.NA

    with pytest.raises(ValueError, match="missing conf_low values"):
        _route(labels=labels)


def test_endpoint_missing_confidence_upper_bound_on_eligible_row_fails_closed():
    labels = _endpoint_labels()
    labels.loc[2, "conf_high"] = pd.NA

    with pytest.raises(ValueError, match="missing conf_high values"):
        _route(labels=labels)


def test_endpoint_missing_data_on_target_missing_row_is_not_scored_or_imputed():
    labels = _endpoint_labels()
    predictions = _endpoint_predictions()
    labels.loc[1, "conf_low"] = pd.NA
    labels.loc[1, "conf_high"] = pd.NA
    predictions.loc[1, "prediction"] = pd.NA

    result = _route(labels=labels, predictions=predictions)

    assert result["molecule_ids"] == ["mol-a", "mol-c", "mol-d"]
    assert result["n_target_missing"] == 1
    assert result["n_scored"] == 3


@pytest.mark.parametrize(
    ("frame_name", "column", "match"),
    [
        ("labels", "target", "nonnumeric target values"),
        ("predictions", "prediction", "nonnumeric prediction values"),
        ("labels", "conf_low", "nonnumeric conf_low values"),
        ("labels", "conf_high", "nonnumeric conf_high values"),
    ],
)
def test_endpoint_nonnumeric_eligible_values_fail_closed(
    frame_name,
    column,
    match,
):
    labels = _endpoint_labels()
    predictions = _endpoint_predictions()
    frame = labels if frame_name == "labels" else predictions
    frame.loc[2, column] = "not-numeric"

    with pytest.raises(ValueError, match=match):
        _route(labels=labels, predictions=predictions)


@pytest.mark.parametrize(
    ("frame_name", "column", "match"),
    [
        ("labels", "target", "nonfinite target values"),
        ("predictions", "prediction", "nonfinite prediction values"),
        ("labels", "conf_low", "nonfinite conf_low values"),
        ("labels", "conf_high", "nonfinite conf_high values"),
    ],
)
def test_endpoint_nonfinite_eligible_values_fail_closed(
    frame_name,
    column,
    match,
):
    labels = _endpoint_labels()
    predictions = _endpoint_predictions()
    frame = labels if frame_name == "labels" else predictions
    frame.loc[2, column] = np.inf

    with pytest.raises(ValueError, match=match):
        _route(labels=labels, predictions=predictions)


def test_endpoint_confidence_lower_greater_than_upper_fails_closed():
    labels = _endpoint_labels()
    labels.loc[2, "conf_low"] = 7.3
    labels.loc[2, "conf_high"] = 7.2

    with pytest.raises(ValueError, match="lower bounds greater than upper"):
        _route(labels=labels)


def test_endpoint_zero_eligible_target_labels_fails_closed():
    labels = _endpoint_labels()
    labels["target"] = pd.NA

    with pytest.raises(ValueError, match="zero eligible labels"):
        _route(labels=labels)


@pytest.mark.parametrize(
    ("frame_name", "column", "match"),
    [
        ("labels", "Molecule_Name", "Aligned labels are missing column"),
        ("labels", "target", "Aligned labels are missing column"),
        ("labels", "conf_low", "Aligned labels are missing column"),
        ("labels", "conf_high", "Aligned labels are missing column"),
        (
            "predictions",
            "Molecule_Name",
            "Aligned predictions are missing column",
        ),
        ("predictions", "prediction", "Aligned predictions are missing column"),
    ],
)
def test_endpoint_missing_required_named_columns_fail_closed(
    frame_name,
    column,
    match,
):
    labels = _endpoint_labels()
    predictions = _endpoint_predictions()
    if frame_name == "labels":
        labels = labels.drop(columns=[column])
    else:
        predictions = predictions.drop(columns=[column])

    with pytest.raises(ValueError, match=match):
        _route(labels=labels, predictions=predictions)


def test_endpoint_st_rae_matches_existing_project_metric():
    routed = _route()

    result = score_endpoint_st_rae(routed)
    expected = st_rae_metric(
        y_true=routed["y_true"],
        y_pred=routed["y_pred"],
        conf_low=routed["conf_low"],
        conf_high=routed["conf_high"],
    )

    assert result["st_rae"] == pytest.approx(expected)
    assert result["n_total_aligned"] == routed["n_total_aligned"]
    assert result["n_target_missing"] == routed["n_target_missing"]
    assert result["n_scored"] == routed["n_scored"]
    assert result["molecule_ids"] == routed["molecule_ids"]
    assert result["target_column"] == "target"
    assert result["prediction_column"] == "prediction"
    assert result["conf_low_column"] == "conf_low"
    assert result["conf_high_column"] == "conf_high"


def test_endpoint_st_rae_delegates_to_existing_metric(monkeypatch):
    routed = _route()
    calls = []

    def fake_st_rae_metric(y_true, y_pred, conf_low, conf_high):
        calls.append(
            {
                "y_true": y_true.copy(),
                "y_pred": y_pred.copy(),
                "conf_low": conf_low.copy(),
                "conf_high": conf_high.copy(),
            }
        )
        return 0.123

    monkeypatch.setattr(
        d008_external_evaluation,
        "st_rae_metric",
        fake_st_rae_metric,
    )

    result = score_endpoint_st_rae(routed)

    assert result["st_rae"] == pytest.approx(0.123)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["y_true"], routed["y_true"])
    np.testing.assert_allclose(calls[0]["y_pred"], routed["y_pred"])
    np.testing.assert_allclose(calls[0]["conf_low"], routed["conf_low"])
    np.testing.assert_allclose(calls[0]["conf_high"], routed["conf_high"])


def test_endpoint_st_rae_missing_routed_key_fails_closed():
    routed = _route()
    del routed["conf_high"]

    with pytest.raises(ValueError, match="missing key: conf_high"):
        score_endpoint_st_rae(routed)


def test_endpoint_st_rae_array_length_mismatch_fails_closed():
    routed = _route()
    routed["y_pred"] = routed["y_pred"][:-1]

    with pytest.raises(ValueError, match="length does not match n_scored"):
        score_endpoint_st_rae(routed)


def test_endpoint_st_rae_nonfinite_metric_result_fails_closed(monkeypatch):
    routed = _route()

    monkeypatch.setattr(
        d008_external_evaluation,
        "st_rae_metric",
        lambda **kwargs: np.inf,
    )

    with pytest.raises(ValueError, match="ST-RAE result is nonfinite"):
        score_endpoint_st_rae(routed)


def test_macro_st_rae_is_unweighted_arithmetic_mean():
    endpoint_results = _four_endpoint_scores()

    result = aggregate_macro_st_rae(endpoint_results)

    assert result["macro_st_rae"] == pytest.approx(0.5)
    assert result["per_endpoint_st_rae"] == {
        "endpoint_a": 0.2,
        "endpoint_b": 0.4,
        "endpoint_c": 0.6,
        "endpoint_d": 0.8,
    }


def test_macro_st_rae_preserves_endpoint_order_and_results():
    endpoint_results = _four_endpoint_scores()

    result = aggregate_macro_st_rae(endpoint_results)

    assert result["endpoint_order"] == [
        "endpoint_a",
        "endpoint_b",
        "endpoint_c",
        "endpoint_d",
    ]
    assert result["per_endpoint_results"] == endpoint_results


def test_macro_st_rae_missing_endpoint_score_fails_closed():
    with pytest.raises(ValueError, match="exactly four endpoint scoring results"):
        aggregate_macro_st_rae(_four_endpoint_scores()[:3])


def test_macro_st_rae_extra_endpoint_score_fails_closed():
    endpoint_results = _four_endpoint_scores() + [
        _endpoint_score("endpoint_e", 1.0)
    ]

    with pytest.raises(ValueError, match="exactly four endpoint scoring results"):
        aggregate_macro_st_rae(endpoint_results)


def test_macro_st_rae_duplicate_endpoint_score_fails_closed():
    endpoint_results = _four_endpoint_scores()
    endpoint_results[3] = _endpoint_score("endpoint_c", 0.8)

    with pytest.raises(ValueError, match="Duplicate endpoint scoring result"):
        aggregate_macro_st_rae(endpoint_results)


def test_macro_st_rae_missing_score_key_fails_closed():
    endpoint_results = _four_endpoint_scores()
    del endpoint_results[2]["st_rae"]

    with pytest.raises(ValueError, match="missing st_rae"):
        aggregate_macro_st_rae(endpoint_results)


def test_macro_st_rae_missing_endpoint_identity_fails_closed():
    endpoint_results = _four_endpoint_scores()
    del endpoint_results[2]["target_column"]

    with pytest.raises(ValueError, match="missing endpoint identity"):
        aggregate_macro_st_rae(endpoint_results)


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_macro_st_rae_nonfinite_endpoint_score_fails_closed(bad_score):
    endpoint_results = _four_endpoint_scores()
    endpoint_results[2]["st_rae"] = bad_score

    with pytest.raises(ValueError, match="nonfinite ST-RAE"):
        aggregate_macro_st_rae(endpoint_results)


def test_organizer_bootstrap_is_deterministic_and_matches_manual_replay():
    routed_endpoints = _four_bootstrap_endpoints()

    result = organizer_compatible_bootstrap(routed_endpoints)

    rng = np.random.default_rng(seed=D008_BOOTSTRAP_SEED)
    expected_endpoint_scores = {
        endpoint["target_column"]: np.empty(D008_BOOTSTRAP_ITERATIONS)
        for endpoint in routed_endpoints
    }
    expected_macro_scores = np.empty(D008_BOOTSTRAP_ITERATIONS)

    for iteration in range(D008_BOOTSTRAP_ITERATIONS):
        iteration_scores = []
        for endpoint in routed_endpoints:
            n_scored = endpoint["n_scored"]
            sample_indices = rng.integers(0, n_scored, size=n_scored)
            score = st_rae_metric(
                y_true=endpoint["y_true"][sample_indices],
                y_pred=endpoint["y_pred"][sample_indices],
                conf_low=endpoint["conf_low"][sample_indices],
                conf_high=endpoint["conf_high"][sample_indices],
            )
            expected_endpoint_scores[endpoint["target_column"]][iteration] = score
            iteration_scores.append(score)
        expected_macro_scores[iteration] = np.mean(iteration_scores)

    assert result["bootstrap_iterations"] == 1000
    assert result["bootstrap_seed"] == 0
    assert result["endpoint_order"] == [
        "endpoint_a",
        "endpoint_b",
        "endpoint_c",
        "endpoint_d",
    ]

    for endpoint in result["endpoint_order"]:
        np.testing.assert_allclose(
            result["endpoint_bootstrap_st_rae"][endpoint],
            expected_endpoint_scores[endpoint],
        )
        assert result["per_endpoint"][endpoint]["mean"] == pytest.approx(
            expected_endpoint_scores[endpoint].mean()
        )
        assert result["per_endpoint"][endpoint]["std"] == pytest.approx(
            expected_endpoint_scores[endpoint].std()
        )

    np.testing.assert_allclose(
        result["macro_bootstrap_st_rae"],
        expected_macro_scores,
    )
    assert result["macro"]["mean"] == pytest.approx(
        expected_macro_scores.mean()
    )
    assert result["macro"]["std"] == pytest.approx(expected_macro_scores.std())


def test_organizer_bootstrap_delegates_to_existing_metric(monkeypatch):
    routed_endpoints = _four_bootstrap_endpoints()
    calls = []

    def fake_st_rae_metric(y_true, y_pred, conf_low, conf_high):
        calls.append((y_true.copy(), y_pred.copy(), conf_low.copy(), conf_high.copy()))
        return float(y_true[0])

    monkeypatch.setattr(
        d008_external_evaluation,
        "st_rae_metric",
        fake_st_rae_metric,
    )

    result = organizer_compatible_bootstrap(routed_endpoints)

    assert len(calls) == 4 * D008_BOOTSTRAP_ITERATIONS
    assert result["bootstrap_iterations"] == D008_BOOTSTRAP_ITERATIONS
    assert np.isfinite(result["macro"]["mean"])


def test_organizer_bootstrap_requires_exactly_four_endpoints():
    with pytest.raises(ValueError, match="exactly four routed endpoint results"):
        organizer_compatible_bootstrap(_four_bootstrap_endpoints()[:3])

    with pytest.raises(ValueError, match="exactly four routed endpoint results"):
        organizer_compatible_bootstrap(
            _four_bootstrap_endpoints()
            + [_bootstrap_endpoint("endpoint_e", 2.0)]
        )


def test_organizer_bootstrap_duplicate_endpoint_fails_closed():
    routed_endpoints = _four_bootstrap_endpoints()
    routed_endpoints[3] = _bootstrap_endpoint("endpoint_c", 1.5)

    with pytest.raises(ValueError, match="Duplicate routed endpoint result"):
        organizer_compatible_bootstrap(routed_endpoints)


def test_organizer_bootstrap_missing_required_key_fails_closed():
    routed_endpoints = _four_bootstrap_endpoints()
    del routed_endpoints[2]["conf_high"]

    with pytest.raises(ValueError, match="missing key: conf_high"):
        organizer_compatible_bootstrap(routed_endpoints)


def test_organizer_bootstrap_nonfinite_input_fails_closed():
    routed_endpoints = _four_bootstrap_endpoints()
    routed_endpoints[2]["y_pred"] = routed_endpoints[2]["y_pred"].copy()
    routed_endpoints[2]["y_pred"][0] = np.inf

    with pytest.raises(ValueError, match="contains nonfinite values"):
        organizer_compatible_bootstrap(routed_endpoints)


def test_organizer_bootstrap_nonfinite_metric_result_fails_closed(monkeypatch):
    routed_endpoints = _four_bootstrap_endpoints()

    monkeypatch.setattr(
        d008_external_evaluation,
        "st_rae_metric",
        lambda **kwargs: np.nan,
    )

    with pytest.raises(ValueError, match="bootstrap ST-RAE is nonfinite"):
        organizer_compatible_bootstrap(routed_endpoints)


def test_production_io_runs_from_explicit_temp_paths(tmp_path):
    fixture = _production_io_fixture(tmp_path)

    result = evaluate_d008_from_paths(
        fixture["labels_path"],
        fixture["predictions_path"],
        expected_labels_sha256=fixture["labels_sha256"],
        expected_predictions_sha256=fixture["predictions_sha256"],
        expected_label_columns=fixture["label_columns"],
        expected_prediction_columns=fixture["prediction_columns"],
        endpoint_mappings=fixture["mappings"],
    )

    assert result["labels_path"] == str(fixture["labels_path"])
    assert result["predictions_path"] == str(fixture["predictions_path"])
    assert result["labels_sha256"] == fixture["labels_sha256"]
    assert result["predictions_sha256"] == fixture["predictions_sha256"]
    assert result["alignment"]["n_rows"] == 20
    assert len(result["routed_endpoints"]) == 4
    assert len(result["endpoint_scores"]) == 4
    assert result["primary"]["endpoint_order"] == [
        mapping["target_column"] for mapping in fixture["mappings"]
    ]
    assert np.isfinite(result["primary"]["macro_st_rae"])
    assert result["bootstrap"]["bootstrap_iterations"] == 1000
    assert result["bootstrap"]["bootstrap_seed"] == 0
    assert np.isfinite(result["bootstrap"]["macro"]["mean"])


def test_d008_result_serializer_is_json_safe_without_changing_scalars(
    tmp_path,
    monkeypatch,
):
    fixture = _production_io_fixture(tmp_path)
    result = evaluate_d008_from_paths(
        fixture["labels_path"],
        fixture["predictions_path"],
        expected_labels_sha256=fixture["labels_sha256"],
        expected_predictions_sha256=fixture["predictions_sha256"],
        expected_label_columns=fixture["label_columns"],
        expected_prediction_columns=fixture["prediction_columns"],
        endpoint_mappings=fixture["mappings"],
    )

    monkeypatch.setattr(
        d008_external_evaluation,
        "st_rae_metric",
        lambda **kwargs: pytest.fail("serializer must not recalculate ST-RAE"),
    )

    artifact = serialize_d008_result(result)
    json.dumps(artifact, allow_nan=False)

    assert "alignment" not in artifact
    assert "routed_endpoints" not in artifact
    assert "endpoint_scores" not in artifact
    assert "endpoint_bootstrap_st_rae" not in artifact["bootstrap"]
    assert "macro_bootstrap_st_rae" not in artifact["bootstrap"]

    assert artifact["inputs"]["labels_sha256"] == result["labels_sha256"]
    assert artifact["inputs"]["predictions_sha256"] == result["predictions_sha256"]
    assert artifact["integrity"]["n_aligned_rows"] == result["alignment"]["n_rows"]
    assert artifact["primary"]["macro_st_rae"] == result["primary"]["macro_st_rae"]
    assert artifact["primary"]["per_endpoint_st_rae"] == result["primary"][
        "per_endpoint_st_rae"
    ]
    assert artifact["bootstrap"]["macro"] == result["bootstrap"]["macro"]

    for serialized_endpoint, endpoint_score in zip(
        artifact["endpoints"],
        result["endpoint_scores"],
    ):
        assert serialized_endpoint["st_rae"] == endpoint_score["st_rae"]
        assert serialized_endpoint["n_total_aligned"] == endpoint_score[
            "n_total_aligned"
        ]
        assert serialized_endpoint["n_target_missing"] == endpoint_score[
            "n_target_missing"
        ]
        assert serialized_endpoint["n_scored"] == endpoint_score["n_scored"]
        assert serialized_endpoint["molecule_ids"] == endpoint_score["molecule_ids"]
        assert serialized_endpoint["target_column"] == endpoint_score[
            "target_column"
        ]
        assert serialized_endpoint["prediction_column"] == endpoint_score[
            "prediction_column"
        ]
        assert serialized_endpoint["conf_low_column"] == endpoint_score[
            "conf_low_column"
        ]
        assert serialized_endpoint["conf_high_column"] == endpoint_score[
            "conf_high_column"
        ]
        endpoint = endpoint_score["target_column"]
        assert artifact["bootstrap"]["per_endpoint"][endpoint] == result[
            "bootstrap"
        ]["per_endpoint"][endpoint]


def test_production_io_sha_mismatch_fails_before_loading(
    tmp_path,
    monkeypatch,
):
    fixture = _production_io_fixture(tmp_path)

    def fail_if_loaded(path):
        raise AssertionError("table loading must not occur after SHA mismatch")

    monkeypatch.setattr(
        d008_external_evaluation,
        "_read_explicit_table",
        fail_if_loaded,
    )

    with pytest.raises(ValueError, match="label artifact SHA-256 mismatch"):
        evaluate_d008_from_paths(
            fixture["labels_path"],
            fixture["predictions_path"],
            expected_labels_sha256="0" * 64,
            expected_predictions_sha256=fixture["predictions_sha256"],
            expected_label_columns=fixture["label_columns"],
            expected_prediction_columns=fixture["prediction_columns"],
            endpoint_mappings=fixture["mappings"],
        )


def test_production_io_prediction_sha_mismatch_fails_closed(tmp_path):
    fixture = _production_io_fixture(tmp_path)

    with pytest.raises(ValueError, match="prediction artifact SHA-256 mismatch"):
        evaluate_d008_from_paths(
            fixture["labels_path"],
            fixture["predictions_path"],
            expected_labels_sha256=fixture["labels_sha256"],
            expected_predictions_sha256="0" * 64,
            expected_label_columns=fixture["label_columns"],
            expected_prediction_columns=fixture["prediction_columns"],
            endpoint_mappings=fixture["mappings"],
        )


def test_production_io_schema_mismatch_fails_closed(tmp_path):
    fixture = _production_io_fixture(tmp_path)
    expected_label_columns = list(fixture["label_columns"])
    expected_label_columns[1], expected_label_columns[2] = (
        expected_label_columns[2],
        expected_label_columns[1],
    )

    with pytest.raises(ValueError, match="label artifact schema mismatch"):
        evaluate_d008_from_paths(
            fixture["labels_path"],
            fixture["predictions_path"],
            expected_labels_sha256=fixture["labels_sha256"],
            expected_predictions_sha256=fixture["predictions_sha256"],
            expected_label_columns=expected_label_columns,
            expected_prediction_columns=fixture["prediction_columns"],
            endpoint_mappings=fixture["mappings"],
        )


def test_production_io_mapping_with_extra_key_fails_closed(tmp_path):
    fixture = _production_io_fixture(tmp_path)
    mappings = [dict(mapping) for mapping in fixture["mappings"]]
    mappings[0]["extra"] = "not allowed"

    with pytest.raises(ValueError, match="Endpoint mapping must contain exactly"):
        evaluate_d008_from_paths(
            fixture["labels_path"],
            fixture["predictions_path"],
            expected_labels_sha256=fixture["labels_sha256"],
            expected_predictions_sha256=fixture["predictions_sha256"],
            expected_label_columns=fixture["label_columns"],
            expected_prediction_columns=fixture["prediction_columns"],
            endpoint_mappings=mappings,
        )


def test_production_io_mapping_referencing_absent_column_fails_closed(tmp_path):
    fixture = _production_io_fixture(tmp_path)
    mappings = [dict(mapping) for mapping in fixture["mappings"]]
    mappings[0]["prediction_column"] = "absent_prediction"

    with pytest.raises(ValueError, match="prediction column not in expected schema"):
        evaluate_d008_from_paths(
            fixture["labels_path"],
            fixture["predictions_path"],
            expected_labels_sha256=fixture["labels_sha256"],
            expected_predictions_sha256=fixture["predictions_sha256"],
            expected_label_columns=fixture["label_columns"],
            expected_prediction_columns=fixture["prediction_columns"],
            endpoint_mappings=mappings,
        )


def test_production_io_identity_mismatch_fails_closed(tmp_path):
    fixture = _production_io_fixture(tmp_path)
    predictions = pd.read_csv(fixture["predictions_path"])
    predictions.loc[0, "Molecule_Name"] = "substituted-molecule"
    predictions.to_csv(fixture["predictions_path"], index=False)
    prediction_sha256 = sha256_file(fixture["predictions_path"])

    with pytest.raises(ValueError, match="identity sets differ"):
        evaluate_d008_from_paths(
            fixture["labels_path"],
            fixture["predictions_path"],
            expected_labels_sha256=fixture["labels_sha256"],
            expected_predictions_sha256=prediction_sha256,
            expected_label_columns=fixture["label_columns"],
            expected_prediction_columns=fixture["prediction_columns"],
            endpoint_mappings=fixture["mappings"],
        )
