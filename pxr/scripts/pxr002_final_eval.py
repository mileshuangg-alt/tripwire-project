"""Tripwire PXR-002 single holdout evaluation (approved protocol, 2026-11-18). Run locally.

The ONLY place holdout labels are read. It requires the explicit
--authorize-holdout-evaluation flag, the SHA-256 of the prediction file (so the labels are
scored against exactly the predictions the final models produced), and a new output
directory (one evaluation; nothing is overwritten). Metrics are descriptive:
  - primary result: the multitask five-seed ensemble (mean of the five predictions)
  - secondary, reported alongside: the aux-off ensemble
  - per-model MAE for context, and the constant training-mean predictor as a floor
    (Rev 3 line 129 satisfied descriptively; no numeric threshold, no statistical test)
No selection between variants is made on these numbers. The official test is not used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "Molecule Name"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float))))


def spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(pd.Series(np.asarray(pred, dtype=float)).corr(pd.Series(np.asarray(truth, dtype=float)), method="spearman"))


def constant_floor(train_frame: pd.DataFrame) -> float:
    """Mean primary-train pEC50 of the v2 training cohort."""
    labels = train_frame.loc[train_frame["role"].eq("primary_train"), "pEC50"].dropna()
    if labels.empty:
        raise ValueError("no primary-train labels for the constant predictor")
    return float(labels.mean())


def evaluate(predictions: pd.DataFrame, labels: pd.Series, constant: float) -> dict:
    """Metrics from long-format predictions (row_id, variant, seed, prediction) and labels."""
    if set(predictions["row_id"]) != set(labels.index):
        raise ValueError("predictions do not cover exactly the holdout molecules")
    ids = sorted(labels.index)
    truth = labels.loc[ids].to_numpy(dtype=float)
    result = {"n_holdout": len(ids), "constant_training_mean": constant,
              "constant_predictor_mae": mae(np.full(len(ids), constant), truth), "variants": {}}
    for variant, group in predictions.groupby("variant"):
        wide = group.pivot(index="row_id", columns="seed", values="prediction").loc[ids]
        if wide.isna().any().any():
            raise ValueError(f"{variant}: missing predictions")
        ensemble = wide.mean(axis=1).to_numpy()
        per_model = {int(s): mae(wide[s].to_numpy(), truth) for s in wide.columns}
        result["variants"][variant] = {
            "n_models": int(wide.shape[1]),
            "ensemble_mae": mae(ensemble, truth),
            "ensemble_spearman": spearman(ensemble, truth),
            "per_model_mae": per_model,
            "mean_per_model_mae": float(np.mean(list(per_model.values()))),
            "ensemble_mae_minus_constant_floor": mae(ensemble, truth) - result["constant_predictor_mae"],
        }
    if {"multitask", "aux_off"} <= set(result["variants"]):
        result["aux_off_minus_multitask_ensemble_mae"] = (
            result["variants"]["aux_off"]["ensemble_mae"] - result["variants"]["multitask"]["ensemble_mae"])
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--predictions-sha256", required=True)
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--assignments", required=True, type=Path)
    p.add_argument("--artifact-dir", required=True, type=Path, help="v2 artifact (for the constant predictor)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--authorize-holdout-evaluation", action="store_true")
    a = p.parse_args()

    if not a.authorize_holdout_evaluation:
        raise SystemExit("refusing to read holdout labels without --authorize-holdout-evaluation")
    if sha256_file(a.predictions) != a.predictions_sha256:
        raise SystemExit("predictions file does not match --predictions-sha256")
    a.out_dir.mkdir(parents=True, exist_ok=False)  # one evaluation; refuses to overwrite

    predictions = pd.read_csv(a.predictions)
    assignments = pd.read_csv(a.assignments, usecols=[ID_COLUMN, "split"])
    holdout_ids = set(assignments.loc[assignments["split"].eq("holdout"), ID_COLUMN])
    truth = pd.read_csv(a.train_csv, usecols=[ID_COLUMN, "pEC50"])  # the only read of holdout labels
    labels = truth.loc[truth[ID_COLUMN].isin(holdout_ids)].set_index(ID_COLUMN)["pEC50"]
    constant = constant_floor(pd.read_csv(a.artifact_dir / "pxr002_train_frame.csv"))
    result = evaluate(predictions, labels, constant)
    result["predictions_sha256"] = a.predictions_sha256
    result["evidence_class"] = "internal-retrospective (Rev 3 line 53); official test not used"
    (a.out_dir / "final_evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
