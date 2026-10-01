from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ecfp4 import ecfp4_matrix
from scripts.st_rae import st_rae_metric


TRAIN_PATH = REPO_ROOT / "data/cyp_challenge/cyp-challenge-TRAIN_inhibition.csv"
SPLIT_PATH = REPO_ROOT / "artifacts/cyp_baseline/molecule_split_assignments.csv"
CONFIG_PATH = REPO_ROOT / "artifacts/cyp_baseline/frozen_baseline_config.json"

OUTDIR = REPO_ROOT / "artifacts/cyp_baseline/test_evaluation"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]


def fit_and_score(data, target, config):
    train = data[
        (data["split"] == "train")
        & data[target].notna()
    ].copy()

    test = data[
        (data["split"] == "test")
        & data[target].notna()
    ].copy()

    x_train = ecfp4_matrix(train["SMILES"])
    x_test = ecfp4_matrix(test["SMILES"])

    y_train = train[target].to_numpy()
    y_test = test[target].to_numpy()

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=config["best_iteration"],
        learning_rate=config["learning_rate"],
        num_leaves=config["num_leaves"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        bagging_freq=1,
        random_state=20261001,
        verbosity=-1,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    lower = test[f"{target}_conf_low"].to_numpy()
    upper = test[f"{target}_conf_high"].to_numpy()

    st_rae = st_rae_metric(
        y_true=y_test,
        y_pred=predictions,
        conf_low=lower,
        conf_high=upper,
    )

    result = {
        "n_train": len(train),
        "n_test": len(test),
        "ST-RAE": float(st_rae),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "R2": float(r2_score(y_test, predictions)),
        "Spearman_R": float(spearmanr(y_test, predictions).statistic),
        "Kendall_Tau": float(kendalltau(y_test, predictions).statistic),
    }

    predictions_out = test[
        ["Molecule_Name", "SMILES"]
    ].copy()
    predictions_out["y_true"] = y_test
    predictions_out["y_pred"] = predictions
    predictions_out["conf_low"] = lower
    predictions_out["conf_high"] = upper

    prediction_path = (
        OUTDIR / f"{target}_predictions.csv"
    )
    predictions_out.to_csv(prediction_path, index=False)

    result["prediction_artifact"] = str(prediction_path)

    return result


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    raw = pd.read_csv(TRAIN_PATH)
    split = pd.read_csv(SPLIT_PATH)

    data = raw.merge(
        split,
        on="Molecule_Name",
        how="inner",
        validate="one_to_one",
    )

    test_ids = set(
        data.loc[data["split"] == "test", "Molecule_Name"]
    )

    if len(test_ids) != 736:
        raise SystemExit(
            f"Expected 736 held-out compounds, found {len(test_ids)}"
        )

    results = {}

    for target in TARGETS:
        result = fit_and_score(
            data,
            target,
            config["isoforms"][target],
        )
        results[target] = result

        print(f"\n=== {target} ===")
        for key, value in result.items():
            if key != "prediction_artifact":
                print(f"{key}: {value}")
        print("Prediction artifact:", result["prediction_artifact"])

    metric_values = {
        metric: [
            results[target][metric]
            for target in TARGETS
        ]
        for metric in [
            "ST-RAE",
            "MAE",
            "R2",
            "Spearman_R",
            "Kendall_Tau",
        ]
    }

    macro = {
        metric: float(np.mean(values))
        for metric, values in metric_values.items()
    }

    print("\n=== MACRO-AVERAGED TEST RESULTS ===")
    for metric, value in macro.items():
        print(f"{metric}: {value}")

    final = {
        "status": "FINAL_INTERNAL_TEST_EVALUATION",
        "test_compounds": 736,
        "isoform_results": results,
        "macro_results": macro,
    }

    output = OUTDIR / "baseline_test_results.json"
    with open(output, "w") as f:
        json.dump(final, f, indent=2)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
