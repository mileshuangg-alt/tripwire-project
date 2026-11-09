from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from st_rae import st_rae_metric


D008_BOOTSTRAP_ITERATIONS = 1000
D008_BOOTSTRAP_SEED = 0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    return result


def _read_explicit_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        result = pd.read_csv(path)
        return result
    if suffix == ".parquet":
        result = pd.read_parquet(path)
        return result
    raise ValueError(f"Unsupported D008 input file type: {path.suffix}")


def _validate_expected_columns(
    dataframe: pd.DataFrame,
    expected_columns: Sequence[str],
    *,
    artifact_name: str,
) -> None:
    actual_columns = list(dataframe.columns)
    expected_column_list = list(expected_columns)
    if actual_columns != expected_column_list:
        raise ValueError(
            f"{artifact_name} schema mismatch. Expected columns "
            f"{expected_column_list}; observed {actual_columns}."
        )
    if len(expected_column_list) != len(set(expected_column_list)):
        raise ValueError(f"{artifact_name} expected schema contains duplicates.")


def _validate_endpoint_mappings(
    endpoint_mappings: Sequence[dict[str, str]],
    *,
    expected_label_columns: Sequence[str],
    expected_prediction_columns: Sequence[str],
    molecule_id_column: str,
) -> list[dict[str, str]]:
    if len(endpoint_mappings) != 4:
        raise ValueError("D008 production evaluation requires four endpoints.")

    required_keys = {
        "endpoint",
        "target_column",
        "prediction_column",
        "conf_low_column",
        "conf_high_column",
    }
    label_columns = set(expected_label_columns)
    prediction_columns = set(expected_prediction_columns)

    if molecule_id_column not in label_columns:
        raise ValueError(
            f"Expected label schema is missing molecule identity column: "
            f"{molecule_id_column}"
        )
    if molecule_id_column not in prediction_columns:
        raise ValueError(
            f"Expected prediction schema is missing molecule identity column: "
            f"{molecule_id_column}"
        )

    normalized_mappings = []
    seen_endpoints = set()
    for mapping in endpoint_mappings:
        if set(mapping) != required_keys:
            raise ValueError(
                "Endpoint mapping must contain exactly endpoint, target_column, "
                "prediction_column, conf_low_column, conf_high_column."
            )

        normalized = {}
        for key in required_keys:
            value = mapping[key]
            if not isinstance(value, str) or value == "":
                raise ValueError(f"Endpoint mapping field {key} must be a string.")
            normalized[key] = value

        endpoint = normalized["endpoint"]
        if endpoint in seen_endpoints:
            raise ValueError(f"Duplicate endpoint mapping: {endpoint}")
        seen_endpoints.add(endpoint)

        for key in ["target_column", "conf_low_column", "conf_high_column"]:
            if normalized[key] not in label_columns:
                raise ValueError(
                    f"Endpoint {endpoint} mapping references label column not "
                    f"in expected schema: {normalized[key]}"
                )
        if normalized["prediction_column"] not in prediction_columns:
            raise ValueError(
                f"Endpoint {endpoint} mapping references prediction column not "
                f"in expected schema: {normalized['prediction_column']}"
            )

        normalized_mappings.append(normalized)

    result = normalized_mappings
    return result


def validate_and_align_external_rows(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    molecule_id_column: str = "Molecule_Name",
) -> dict[str, object]:
    """
    Validate one-to-one molecular identity and align predictions to labels.

    This is D008 identity/integrity infrastructure only. It does not impute
    labels, route endpoints, or calculate any performance metric.
    """
    if molecule_id_column not in labels.columns:
        raise ValueError(
            f"Labels are missing molecule identity column: "
            f"{molecule_id_column}"
        )

    if molecule_id_column not in predictions.columns:
        raise ValueError(
            f"Predictions are missing molecule identity column: "
            f"{molecule_id_column}"
        )

    label_ids = labels[molecule_id_column]
    prediction_ids = predictions[molecule_id_column]

    if label_ids.isna().any():
        raise ValueError("Labels contain missing molecule identifiers.")

    if prediction_ids.isna().any():
        raise ValueError("Predictions contain missing molecule identifiers.")

    duplicate_label_ids = label_ids[label_ids.duplicated()].unique().tolist()
    if duplicate_label_ids:
        raise ValueError(
            "Labels contain duplicate molecule identifiers: "
            f"{duplicate_label_ids}"
        )

    duplicate_prediction_ids = prediction_ids[
        prediction_ids.duplicated()
    ].unique().tolist()
    if duplicate_prediction_ids:
        raise ValueError(
            "Predictions contain duplicate molecule identifiers: "
            f"{duplicate_prediction_ids}"
        )

    label_id_set = set(label_ids)
    prediction_id_set = set(prediction_ids)
    missing_predictions = [
        molecule_id
        for molecule_id in label_ids
        if molecule_id not in prediction_id_set
    ]
    extra_predictions = [
        molecule_id
        for molecule_id in prediction_ids
        if molecule_id not in label_id_set
    ]
    if missing_predictions or extra_predictions:
        raise ValueError(
            "Label and prediction molecule identity sets differ. "
            f"Missing predictions={missing_predictions}; "
            f"extra predictions={extra_predictions}."
        )

    aligned_predictions = (
        predictions.set_index(molecule_id_column, drop=False)
        .loc[label_ids]
        .reset_index(drop=True)
    )

    result = {
        "labels": labels,
        "predictions": aligned_predictions,
        "molecule_id_column": molecule_id_column,
        "n_rows": len(labels),
    }
    return result


def route_endpoint_rows(
    aligned_labels: pd.DataFrame,
    aligned_predictions: pd.DataFrame,
    *,
    target_column: str,
    prediction_column: str,
    conf_low_column: str,
    conf_high_column: str,
    molecule_id_column: str = "Molecule_Name",
) -> dict[str, object]:
    """
    Build one endpoint's scoring population from aligned D008 rows.

    Endpoint eligibility is determined only by nonmissing target labels. This
    function does not score, impute, or repair malformed eligible rows.
    """
    required_label_columns = [
        molecule_id_column,
        target_column,
        conf_low_column,
        conf_high_column,
    ]
    required_prediction_columns = [molecule_id_column, prediction_column]

    for column in required_label_columns:
        if column not in aligned_labels.columns:
            raise ValueError(f"Aligned labels are missing column: {column}")

    for column in required_prediction_columns:
        if column not in aligned_predictions.columns:
            raise ValueError(
                f"Aligned predictions are missing column: {column}"
            )

    if len(aligned_labels) != len(aligned_predictions):
        raise ValueError("Aligned label and prediction row counts differ.")

    label_ids = aligned_labels[molecule_id_column].tolist()
    prediction_ids = aligned_predictions[molecule_id_column].tolist()
    if label_ids != prediction_ids:
        raise ValueError("Aligned label and prediction molecule order differs.")

    eligible_mask = aligned_labels[target_column].notna()
    n_total_aligned = len(aligned_labels)
    n_target_missing = int((~eligible_mask).sum())
    n_scored = int(eligible_mask.sum())

    if n_scored == 0:
        raise ValueError(f"Endpoint {target_column} has zero eligible labels.")

    eligible_labels = aligned_labels.loc[eligible_mask]
    eligible_predictions = aligned_predictions.loc[eligible_mask]

    endpoint_columns = {
        "target": eligible_labels[target_column],
        "prediction": eligible_predictions[prediction_column],
        "conf_low": eligible_labels[conf_low_column],
        "conf_high": eligible_labels[conf_high_column],
    }

    for name, values in endpoint_columns.items():
        if values.isna().any():
            raise ValueError(
                f"Eligible endpoint rows contain missing {name} values."
            )

    numeric_arrays: dict[str, np.ndarray] = {}
    for name, values in endpoint_columns.items():
        try:
            numeric_values = pd.to_numeric(values, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Eligible endpoint rows contain nonnumeric {name} values."
            ) from error

        array = numeric_values.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(array).all():
            raise ValueError(
                f"Eligible endpoint rows contain nonfinite {name} values."
            )
        numeric_arrays[name] = array

    if (numeric_arrays["conf_low"] > numeric_arrays["conf_high"]).any():
        raise ValueError(
            "Eligible endpoint rows contain confidence lower bounds greater "
            "than upper bounds."
        )

    result = {
        "n_total_aligned": n_total_aligned,
        "n_target_missing": n_target_missing,
        "n_scored": n_scored,
        "molecule_ids": eligible_labels[molecule_id_column].tolist(),
        "y_true": numeric_arrays["target"],
        "y_pred": numeric_arrays["prediction"],
        "conf_low": numeric_arrays["conf_low"],
        "conf_high": numeric_arrays["conf_high"],
        "target_column": target_column,
        "prediction_column": prediction_column,
        "conf_low_column": conf_low_column,
        "conf_high_column": conf_high_column,
        "molecule_id_column": molecule_id_column,
    }
    return result


def score_endpoint_st_rae(routed_endpoint: dict[str, object]) -> dict[str, object]:
    """
    Calculate one endpoint's D008 ST-RAE from pre-routed eligible rows.

    This function delegates scoring to the existing project ST-RAE wrapper. It
    does not calculate macro aggregation, bootstrap values, or other metrics.
    """
    required_keys = [
        "n_total_aligned",
        "n_target_missing",
        "n_scored",
        "molecule_ids",
        "y_true",
        "y_pred",
        "conf_low",
        "conf_high",
        "target_column",
        "prediction_column",
        "conf_low_column",
        "conf_high_column",
        "molecule_id_column",
    ]
    for key in required_keys:
        if key not in routed_endpoint:
            raise ValueError(f"Routed endpoint result is missing key: {key}")

    n_scored = int(routed_endpoint["n_scored"])
    if n_scored <= 0:
        raise ValueError("Endpoint ST-RAE requires at least one scored row.")

    y_true = np.asarray(routed_endpoint["y_true"], dtype=np.float64)
    y_pred = np.asarray(routed_endpoint["y_pred"], dtype=np.float64)
    conf_low = np.asarray(routed_endpoint["conf_low"], dtype=np.float64)
    conf_high = np.asarray(routed_endpoint["conf_high"], dtype=np.float64)

    for name, values in {
        "y_true": y_true,
        "y_pred": y_pred,
        "conf_low": conf_low,
        "conf_high": conf_high,
    }.items():
        if values.shape != (n_scored,):
            raise ValueError(
                f"Routed endpoint {name} length does not match n_scored."
            )

    molecule_ids = list(routed_endpoint["molecule_ids"])
    if len(molecule_ids) != n_scored:
        raise ValueError(
            "Routed endpoint molecule identifier count does not match n_scored."
        )

    st_rae = float(
        st_rae_metric(
            y_true=y_true,
            y_pred=y_pred,
            conf_low=conf_low,
            conf_high=conf_high,
        )
    )
    if not np.isfinite(st_rae):
        raise ValueError("Endpoint ST-RAE result is nonfinite.")

    result = {
        "st_rae": st_rae,
        "n_total_aligned": routed_endpoint["n_total_aligned"],
        "n_target_missing": routed_endpoint["n_target_missing"],
        "n_scored": n_scored,
        "molecule_ids": molecule_ids,
        "target_column": routed_endpoint["target_column"],
        "prediction_column": routed_endpoint["prediction_column"],
        "conf_low_column": routed_endpoint["conf_low_column"],
        "conf_high_column": routed_endpoint["conf_high_column"],
        "molecule_id_column": routed_endpoint["molecule_id_column"],
    }
    return result


def aggregate_macro_st_rae(
    endpoint_results: Sequence[dict[str, object]],
) -> dict[str, object]:
    """
    Aggregate the four D008 endpoint ST-RAE scores by arithmetic mean.

    This function consumes already-scored endpoint results. It does not weight
    by endpoint sample size and does not calculate bootstrap values.
    """
    if len(endpoint_results) != 4:
        raise ValueError(
            "D008 macro ST-RAE requires exactly four endpoint scoring results."
        )

    endpoint_order = []
    scores = []
    seen_endpoints = set()
    per_endpoint_results = []
    for endpoint_result in endpoint_results:
        if "target_column" not in endpoint_result:
            raise ValueError(
                "Endpoint scoring result is missing endpoint identity: "
                "target_column"
            )
        if "st_rae" not in endpoint_result:
            raise ValueError("Endpoint scoring result is missing st_rae.")

        endpoint = endpoint_result["target_column"]
        if endpoint in seen_endpoints:
            raise ValueError(f"Duplicate endpoint scoring result: {endpoint}")
        seen_endpoints.add(endpoint)

        try:
            score = float(endpoint_result["st_rae"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Endpoint {endpoint} has nonnumeric ST-RAE."
            ) from error

        if not np.isfinite(score):
            raise ValueError(f"Endpoint {endpoint} has nonfinite ST-RAE.")

        endpoint_order.append(endpoint)
        scores.append(score)
        per_endpoint_results.append(endpoint_result)

    result = {
        "macro_st_rae": float(np.mean(np.asarray(scores, dtype=np.float64))),
        "endpoint_order": endpoint_order,
        "per_endpoint_st_rae": dict(zip(endpoint_order, scores)),
        "per_endpoint_results": per_endpoint_results,
    }
    return result


def organizer_compatible_bootstrap(
    routed_endpoints: Sequence[dict[str, object]],
) -> dict[str, object]:
    """
    Run the D008 organizer-compatible 1,000-iteration bootstrap.

    Each iteration resamples each endpoint's eligible scoring observations with
    replacement, delegates endpoint scoring to the existing ST-RAE wrapper, and
    computes the unweighted four-endpoint macro for that iteration.
    """
    if len(routed_endpoints) != 4:
        raise ValueError(
            "D008 bootstrap requires exactly four routed endpoint results."
        )

    endpoint_order = []
    seen_endpoints = set()
    prepared_endpoints = []
    required_keys = [
        "n_scored",
        "y_true",
        "y_pred",
        "conf_low",
        "conf_high",
        "target_column",
    ]

    for routed_endpoint in routed_endpoints:
        for key in required_keys:
            if key not in routed_endpoint:
                raise ValueError(f"Routed endpoint result is missing key: {key}")

        endpoint = routed_endpoint["target_column"]
        if endpoint in seen_endpoints:
            raise ValueError(f"Duplicate routed endpoint result: {endpoint}")
        seen_endpoints.add(endpoint)

        n_scored = int(routed_endpoint["n_scored"])
        if n_scored <= 0:
            raise ValueError(
                f"Endpoint {endpoint} bootstrap requires at least one scored row."
            )

        arrays = {
            "y_true": np.asarray(routed_endpoint["y_true"], dtype=np.float64),
            "y_pred": np.asarray(routed_endpoint["y_pred"], dtype=np.float64),
            "conf_low": np.asarray(routed_endpoint["conf_low"], dtype=np.float64),
            "conf_high": np.asarray(routed_endpoint["conf_high"], dtype=np.float64),
        }
        for name, values in arrays.items():
            if values.shape != (n_scored,):
                raise ValueError(
                    f"Routed endpoint {endpoint} {name} length does not match "
                    "n_scored."
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Routed endpoint {endpoint} {name} contains nonfinite "
                    "values."
                )

        endpoint_order.append(endpoint)
        prepared_endpoints.append((endpoint, n_scored, arrays))

    rng = np.random.default_rng(seed=D008_BOOTSTRAP_SEED)
    endpoint_bootstrap_scores = {
        endpoint: np.empty(D008_BOOTSTRAP_ITERATIONS, dtype=np.float64)
        for endpoint in endpoint_order
    }
    macro_bootstrap_scores = np.empty(
        D008_BOOTSTRAP_ITERATIONS,
        dtype=np.float64,
    )

    for iteration in range(D008_BOOTSTRAP_ITERATIONS):
        iteration_scores = []
        for endpoint, n_scored, arrays in prepared_endpoints:
            sample_indices = rng.integers(0, n_scored, size=n_scored)
            score = float(
                st_rae_metric(
                    y_true=arrays["y_true"][sample_indices],
                    y_pred=arrays["y_pred"][sample_indices],
                    conf_low=arrays["conf_low"][sample_indices],
                    conf_high=arrays["conf_high"][sample_indices],
                )
            )
            if not np.isfinite(score):
                raise ValueError(
                    f"Endpoint {endpoint} bootstrap ST-RAE is nonfinite at "
                    f"iteration {iteration}."
                )
            endpoint_bootstrap_scores[endpoint][iteration] = score
            iteration_scores.append(score)

        macro_bootstrap_scores[iteration] = float(
            np.mean(np.asarray(iteration_scores, dtype=np.float64))
        )

    per_endpoint = {
        endpoint: {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
        for endpoint, values in endpoint_bootstrap_scores.items()
    }
    result = {
        "bootstrap_iterations": D008_BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": D008_BOOTSTRAP_SEED,
        "endpoint_order": endpoint_order,
        "per_endpoint": per_endpoint,
        "macro": {
            "mean": float(macro_bootstrap_scores.mean()),
            "std": float(macro_bootstrap_scores.std()),
        },
        "endpoint_bootstrap_st_rae": endpoint_bootstrap_scores,
        "macro_bootstrap_st_rae": macro_bootstrap_scores,
    }
    return result


def _json_safe_scalar(value: object, *, field_name: str) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"D008 report field {field_name} is nonfinite.")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"D008 report field {field_name} is not JSON-safe.")


def _json_safe_mapping(
    values: Mapping[str, object],
    *,
    field_name: str,
) -> dict[str, object]:
    return {
        str(key): _json_safe_container(value, field_name=f"{field_name}.{key}")
        for key, value in values.items()
    }


def _json_safe_container(value: object, *, field_name: str) -> object:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value, field_name=field_name)
    if isinstance(value, (list, tuple)):
        return [
            _json_safe_container(item, field_name=f"{field_name}[]")
            for item in value
        ]
    return _json_safe_scalar(value, field_name=field_name)


def serialize_d008_result(evaluation_result: Mapping[str, object]) -> dict[str, object]:
    """
    Build the JSON-serializable D008 result artifact from an evaluation result.

    This reporting layer only selects and normalizes already-computed D008
    outputs. It does not route rows, call ST-RAE, aggregate macro scores, or run
    bootstrap calculations.
    """
    required_top_level_keys = [
        "labels_path",
        "predictions_path",
        "labels_sha256",
        "predictions_sha256",
        "molecule_id_column",
        "endpoint_mappings",
        "alignment",
        "endpoint_scores",
        "primary",
        "bootstrap",
    ]
    for key in required_top_level_keys:
        if key not in evaluation_result:
            raise ValueError(f"D008 evaluation result is missing key: {key}")

    alignment = evaluation_result["alignment"]
    primary = evaluation_result["primary"]
    bootstrap = evaluation_result["bootstrap"]
    if not isinstance(alignment, Mapping):
        raise ValueError("D008 evaluation alignment result is not a mapping.")
    if not isinstance(primary, Mapping):
        raise ValueError("D008 primary result is not a mapping.")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("D008 bootstrap result is not a mapping.")

    for key in ["macro_st_rae", "endpoint_order", "per_endpoint_st_rae"]:
        if key not in primary:
            raise ValueError(f"D008 primary result is missing key: {key}")
    if "n_rows" not in alignment:
        raise ValueError("D008 alignment result is missing key: n_rows")
    for key in [
        "bootstrap_iterations",
        "bootstrap_seed",
        "endpoint_order",
        "per_endpoint",
        "macro",
    ]:
        if key not in bootstrap:
            raise ValueError(f"D008 bootstrap result is missing key: {key}")

    endpoint_scores = list(evaluation_result["endpoint_scores"])
    endpoint_order = list(primary["endpoint_order"])
    if len(endpoint_order) != 4:
        raise ValueError("D008 serialized result requires exactly four endpoints.")
    if list(bootstrap["endpoint_order"]) != endpoint_order:
        raise ValueError("D008 bootstrap endpoint order does not match primary order.")
    if len(endpoint_scores) != len(endpoint_order):
        raise ValueError("D008 endpoint score count does not match endpoint order.")

    endpoint_artifacts = []
    for endpoint, endpoint_score in zip(endpoint_order, endpoint_scores):
        if not isinstance(endpoint_score, Mapping):
            raise ValueError("D008 endpoint score result is not a mapping.")
        required_endpoint_keys = [
            "st_rae",
            "n_total_aligned",
            "n_target_missing",
            "n_scored",
            "molecule_ids",
            "target_column",
            "prediction_column",
            "conf_low_column",
            "conf_high_column",
            "molecule_id_column",
        ]
        for key in required_endpoint_keys:
            if key not in endpoint_score:
                raise ValueError(f"D008 endpoint result is missing key: {key}")
        if endpoint_score["target_column"] != endpoint:
            raise ValueError("D008 endpoint score order does not match primary order.")

        endpoint_artifacts.append(
            {
                "endpoint": _json_safe_scalar(endpoint, field_name="endpoint"),
                "st_rae": _json_safe_scalar(
                    endpoint_score["st_rae"],
                    field_name=f"endpoint.{endpoint}.st_rae",
                ),
                "n_total_aligned": _json_safe_scalar(
                    endpoint_score["n_total_aligned"],
                    field_name=f"endpoint.{endpoint}.n_total_aligned",
                ),
                "n_target_missing": _json_safe_scalar(
                    endpoint_score["n_target_missing"],
                    field_name=f"endpoint.{endpoint}.n_target_missing",
                ),
                "n_scored": _json_safe_scalar(
                    endpoint_score["n_scored"],
                    field_name=f"endpoint.{endpoint}.n_scored",
                ),
                "molecule_ids": _json_safe_container(
                    endpoint_score["molecule_ids"],
                    field_name=f"endpoint.{endpoint}.molecule_ids",
                ),
                "target_column": _json_safe_scalar(
                    endpoint_score["target_column"],
                    field_name=f"endpoint.{endpoint}.target_column",
                ),
                "prediction_column": _json_safe_scalar(
                    endpoint_score["prediction_column"],
                    field_name=f"endpoint.{endpoint}.prediction_column",
                ),
                "conf_low_column": _json_safe_scalar(
                    endpoint_score["conf_low_column"],
                    field_name=f"endpoint.{endpoint}.conf_low_column",
                ),
                "conf_high_column": _json_safe_scalar(
                    endpoint_score["conf_high_column"],
                    field_name=f"endpoint.{endpoint}.conf_high_column",
                ),
                "molecule_id_column": _json_safe_scalar(
                    endpoint_score["molecule_id_column"],
                    field_name=f"endpoint.{endpoint}.molecule_id_column",
                ),
            }
        )

    artifact = {
        "inputs": {
            "labels_path": _json_safe_scalar(
                evaluation_result["labels_path"],
                field_name="labels_path",
            ),
            "predictions_path": _json_safe_scalar(
                evaluation_result["predictions_path"],
                field_name="predictions_path",
            ),
            "labels_sha256": _json_safe_scalar(
                evaluation_result["labels_sha256"],
                field_name="labels_sha256",
            ),
            "predictions_sha256": _json_safe_scalar(
                evaluation_result["predictions_sha256"],
                field_name="predictions_sha256",
            ),
            "molecule_id_column": _json_safe_scalar(
                evaluation_result["molecule_id_column"],
                field_name="molecule_id_column",
            ),
            "endpoint_mappings": _json_safe_container(
                evaluation_result["endpoint_mappings"],
                field_name="endpoint_mappings",
            ),
        },
        "integrity": {
            "n_aligned_rows": _json_safe_scalar(
                alignment["n_rows"],
                field_name="alignment.n_rows",
            ),
        },
        "primary": {
            "macro_st_rae": _json_safe_scalar(
                primary["macro_st_rae"],
                field_name="primary.macro_st_rae",
            ),
            "endpoint_order": _json_safe_container(
                endpoint_order,
                field_name="primary.endpoint_order",
            ),
            "per_endpoint_st_rae": _json_safe_container(
                primary["per_endpoint_st_rae"],
                field_name="primary.per_endpoint_st_rae",
            ),
        },
        "bootstrap": {
            "bootstrap_iterations": _json_safe_scalar(
                bootstrap["bootstrap_iterations"],
                field_name="bootstrap.bootstrap_iterations",
            ),
            "bootstrap_seed": _json_safe_scalar(
                bootstrap["bootstrap_seed"],
                field_name="bootstrap.bootstrap_seed",
            ),
            "endpoint_order": _json_safe_container(
                bootstrap["endpoint_order"],
                field_name="bootstrap.endpoint_order",
            ),
            "per_endpoint": _json_safe_container(
                bootstrap["per_endpoint"],
                field_name="bootstrap.per_endpoint",
            ),
            "macro": _json_safe_container(
                bootstrap["macro"],
                field_name="bootstrap.macro",
            ),
        },
        "endpoints": endpoint_artifacts,
    }
    return artifact


def evaluate_d008_from_paths(
    labels_path: str | Path,
    predictions_path: str | Path,
    *,
    expected_labels_sha256: str,
    expected_predictions_sha256: str,
    expected_label_columns: Sequence[str],
    expected_prediction_columns: Sequence[str],
    endpoint_mappings: Sequence[dict[str, str]],
    molecule_id_column: str = "Molecule_Name",
) -> dict[str, object]:
    """
    Execute the pre-qualified D008 evaluator from explicit frozen artifacts.

    The supplied SHA-256 identities are verified before parsing or scoring. All
    scientific operations are delegated to the qualified D008 primitives above.
    """
    label_path = Path(labels_path)
    prediction_path = Path(predictions_path)
    mappings = _validate_endpoint_mappings(
        endpoint_mappings,
        expected_label_columns=expected_label_columns,
        expected_prediction_columns=expected_prediction_columns,
        molecule_id_column=molecule_id_column,
    )

    observed_labels_sha256 = sha256_file(label_path)
    if observed_labels_sha256 != expected_labels_sha256:
        raise ValueError(
            "D008 label artifact SHA-256 mismatch. "
            f"Expected {expected_labels_sha256}; observed "
            f"{observed_labels_sha256}."
        )

    observed_predictions_sha256 = sha256_file(prediction_path)
    if observed_predictions_sha256 != expected_predictions_sha256:
        raise ValueError(
            "D008 prediction artifact SHA-256 mismatch. "
            f"Expected {expected_predictions_sha256}; observed "
            f"{observed_predictions_sha256}."
        )

    labels = _read_explicit_table(label_path)
    predictions = _read_explicit_table(prediction_path)
    _validate_expected_columns(
        labels,
        expected_label_columns,
        artifact_name="D008 label artifact",
    )
    _validate_expected_columns(
        predictions,
        expected_prediction_columns,
        artifact_name="D008 prediction artifact",
    )

    alignment = validate_and_align_external_rows(
        labels,
        predictions,
        molecule_id_column=molecule_id_column,
    )

    routed_endpoints = []
    endpoint_scores = []
    for mapping in mappings:
        routed = route_endpoint_rows(
            alignment["labels"],
            alignment["predictions"],
            target_column=mapping["target_column"],
            prediction_column=mapping["prediction_column"],
            conf_low_column=mapping["conf_low_column"],
            conf_high_column=mapping["conf_high_column"],
            molecule_id_column=molecule_id_column,
        )
        score = score_endpoint_st_rae(routed)
        routed_endpoints.append(routed)
        endpoint_scores.append(score)

    primary = aggregate_macro_st_rae(endpoint_scores)
    bootstrap = organizer_compatible_bootstrap(routed_endpoints)

    result = {
        "labels_path": str(label_path),
        "predictions_path": str(prediction_path),
        "labels_sha256": observed_labels_sha256,
        "predictions_sha256": observed_predictions_sha256,
        "molecule_id_column": molecule_id_column,
        "endpoint_mappings": mappings,
        "alignment": alignment,
        "routed_endpoints": routed_endpoints,
        "endpoint_scores": endpoint_scores,
        "primary": primary,
        "bootstrap": bootstrap,
    }
    return result
