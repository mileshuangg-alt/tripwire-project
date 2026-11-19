"""PXR-003 single reporting-only holdout report. The ONLY place holdout labels are read.

Scores the frozen LightGBM and random forest ensembles and, from the existing PXR-002 holdout
predictions, the neural ensembles (adds RMSE). Requires --authorize-holdout-evaluation, the
SHA-256 of the frozen config and of both prediction files, and a new output directory.
Re-derives the neural MAE and Spearman and asserts they equal PXR-002 final_evaluation.json.
Evidence class: internal-retrospective. No selection or configuration change may follow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pxr.scripts import pxr003_common as common
from pxr.scripts.pxr001_baseline import ID_COLUMN, TARGET_COLUMN

AUR = "OADMET-0003727"


def ensemble(predictions: pd.DataFrame, key: str, value: str, ids: list) -> np.ndarray:
    sub = predictions.loc[predictions[key].eq(value)]
    return sub.groupby("row_id")["prediction"].mean().loc[ids].to_numpy()


def per_model(predictions: pd.DataFrame, key: str, value: str, ids: list, truth: np.ndarray) -> dict:
    sub = predictions.loc[predictions[key].eq(value)]
    rows = {int(seed): common.metrics(g.set_index("row_id")["prediction"].loc[ids].to_numpy(), truth)
            for seed, g in sub.groupby("seed")}
    return {"per_model": rows,
            "mean_per_model": {k: float(np.mean([r[k] for r in rows.values()])) for k in ("mae", "spearman", "rmse")}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-predictions", required=True, type=Path)
    p.add_argument("--baseline-predictions-sha256", required=True)
    p.add_argument("--frozen-config", required=True, type=Path)
    p.add_argument("--frozen-sha256", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--authorize-holdout-evaluation", action="store_true")
    a = p.parse_args()

    if not a.authorize_holdout_evaluation:
        raise SystemExit("refusing to read holdout labels without --authorize-holdout-evaluation")
    if common.sha256_file(a.frozen_config) != a.frozen_sha256:
        raise SystemExit("frozen config does not match --frozen-sha256")
    if common.sha256_file(a.baseline_predictions) != a.baseline_predictions_sha256:
        raise SystemExit("baseline predictions do not match --baseline-predictions-sha256")
    neural_path = common.NEURAL_DIR / "holdout_predictions.csv"
    if common.sha256_file(neural_path) != common.NEURAL_PREDICTIONS_SHA256:
        raise SystemExit("neural holdout predictions do not match the recorded SHA-256")
    common.verify_inputs()
    a.out_dir.mkdir(parents=True, exist_ok=False)  # one evaluation; refuses to overwrite

    assignments = pd.read_csv(common.ASSIGNMENTS_CSV, usecols=[ID_COLUMN, "split"])
    holdout_ids = assignments.loc[assignments["split"].eq("holdout"), ID_COLUMN].tolist()
    labels = pd.read_csv(common.TRAIN_CSV, usecols=[ID_COLUMN, TARGET_COLUMN]).set_index(ID_COLUMN)[TARGET_COLUMN]
    truth = labels.loc[holdout_ids].to_numpy()
    train_ids = assignments.loc[assignments["split"].eq("train"), ID_COLUMN]
    mean_2897 = float(labels.loc[train_ids].mean())
    mean_2896 = float(labels.loc[train_ids[train_ids != AUR]].mean())

    base = pd.read_csv(a.baseline_predictions)
    base = base.loc[base["split"].eq("holdout")]
    neural = pd.read_csv(neural_path)
    for name, frame in (("baseline", base), ("neural", neural)):
        if set(frame["row_id"]) != set(holdout_ids):
            raise AssertionError(f"{name} predictions do not cover exactly the holdout identities")
        if not np.isfinite(frame["prediction"]).all():
            raise AssertionError(f"{name} predictions contain non-finite values")

    results = {"evidence_class": common.EVIDENCE_CLASS, "disclosure": common.DISCLOSURE, "n_holdout": len(truth),
               "frozen_config_sha256": a.frozen_sha256, "baseline_predictions_sha256": a.baseline_predictions_sha256,
               "neural_predictions_sha256": common.NEURAL_PREDICTIONS_SHA256, "systems": {}}
    for model, label in (("lgbm", "ECFP4 + LightGBM"), ("rf", "ECFP4 + random forest")):
        results["systems"][label] = {
            "ensemble": common.metrics(ensemble(base, "model", model, holdout_ids), truth),
            **per_model(base, "model", model, holdout_ids, truth)}
    for variant, label in (("multitask", "Chemprop multitask (primary)"), ("aux_off", "Chemprop aux-off (secondary)")):
        results["systems"][label] = {
            "ensemble": common.metrics(ensemble(neural, "variant", variant, holdout_ids), truth),
            **per_model(neural, "variant", variant, holdout_ids, truth)}
    results["constant_baseline"] = {
        "training_mean_2897_baseline_cohort": {"mean": mean_2897, **common.constant_metrics(mean_2897, truth)},
        "training_mean_2896_neural_cohort": {"mean": mean_2896, **common.constant_metrics(mean_2896, truth)}}

    # Consistency with the recorded PXR-002 evaluation (MAE and Spearman must reproduce exactly).
    recorded = json.loads((common.NEURAL_DIR / "evaluation/final_evaluation.json").read_text())
    for variant, label in (("multitask", "Chemprop multitask (primary)"), ("aux_off", "Chemprop aux-off (secondary)")):
        got = results["systems"][label]["ensemble"]
        want = recorded["variants"][variant]
        assert abs(got["mae"] - want["ensemble_mae"]) < 1e-9, (variant, "MAE")
        assert abs(got["spearman"] - want["ensemble_spearman"]) < 1e-9, (variant, "Spearman")
    assert abs(results["constant_baseline"]["training_mean_2896_neural_cohort"]["mae"]
               - recorded["constant_predictor_mae"]) < 1e-9, "neural constant baseline MAE"

    common.write_json(a.out_dir / "holdout_report.json", results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
