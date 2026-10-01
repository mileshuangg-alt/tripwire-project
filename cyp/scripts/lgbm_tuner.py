from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ecfp4 import ecfp4_matrix
from scripts.st_rae import make_st_rae_evaluator


TRAIN_PATH = (
    REPO_ROOT
    / "data"
    / "cyp_challenge"
    / "cyp-challenge-TRAIN_inhibition.csv"
)
SPLIT_PATH = (
    REPO_ROOT
    / "artifacts"
    / "cyp_baseline"
    / "molecule_split_assignments.csv"
)

OUTDIR = REPO_ROOT / "artifacts" / "cyp_baseline" / "tuning"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 30
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50
BASE_SEED = 20261001

ISOFORM_SEEDS = {
    "CYP1A2_pIC50_direct_inhibition": 20261101,
    "CYP2C9_pIC50_direct_inhibition": 20261102,
    "CYP2D6_pIC50_direct_inhibition": 20261103,
    "CYP3A4_pIC50_direct_inhibition": 20261104,
}


def sample_log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
) -> float:
    result = float(
        math.exp(
            rng.uniform(
                math.log(low),
                math.log(high),
            )
        )
    )
    return result


def sample_parameters(
    rng: np.random.Generator,
) -> dict:
    result = {
        "learning_rate": sample_log_uniform(
            rng,
            0.005,
            0.2,
        ),
        "num_leaves": int(
            rng.integers(
                7,
                64,
            )
        ),
        "min_child_weight": sample_log_uniform(
            rng,
            0.001,
            10.0,
        ),
        "subsample": float(
            rng.uniform(
                0.5,
                1.0,
            )
        ),
    }
    return result


def load_train_validation(
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(TRAIN_PATH)
    assignments = pd.read_csv(SPLIT_PATH)

    merged = data.merge(
        assignments,
        on="Molecule_Name",
        how="inner",
        validate="one_to_one",
    )

    train = merged[
        (merged["split"] == "train")
        & merged[target].notna()
    ].copy()

    validation = merged[
        (merged["split"] == "validation")
        & merged[target].notna()
    ].copy()

    if train.empty or validation.empty:
        raise ValueError(
            f"Empty train/validation set for target {target}"
        )

    overlap = set(train["Molecule_Name"]) & set(
        validation["Molecule_Name"]
    )
    if overlap:
        raise AssertionError(
            f"Train/validation overlap for {target}: "
            f"{len(overlap)} molecules"
        )

    return train, validation


def fit_trial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    conf_low: np.ndarray,
    conf_high: np.ndarray,
    parameters: dict,
    trial_seed: int,
) -> tuple[lgb.LGBMRegressor, float, int]:
    evaluator = make_st_rae_evaluator(
        y_true=y_validation,
        conf_low=conf_low,
        conf_high=conf_high,
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=N_ESTIMATORS,
        learning_rate=parameters["learning_rate"],
        num_leaves=parameters["num_leaves"],
        min_child_weight=parameters["min_child_weight"],
        subsample=parameters["subsample"],
        bagging_freq=1,
        random_state=trial_seed,
        verbosity=-1,
    )

    model.fit(
        x_train,
        y_train,
        eval_X=x_validation,
        eval_y=y_validation,
        eval_names=["validation"],
        eval_metric=evaluator,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )

    best_iteration = model.best_iteration_

    if best_iteration is None:
        raise RuntimeError("LightGBM did not report best_iteration_.")

    metric_history = model.evals_result_["validation"]["ST-RAE"]
    best_score = float(metric_history[best_iteration - 1])

    result = model, best_score, int(best_iteration)
    return result


def tune_isoform(target: str) -> dict:
    train, validation = load_train_validation(target)

    print(f"\n=== {target} ===")
    print("Train rows:", len(train))
    print("Validation rows:", len(validation))

    x_train = ecfp4_matrix(train["SMILES"])
    x_validation = ecfp4_matrix(validation["SMILES"])

    y_train = train[target].to_numpy()
    y_validation = validation[target].to_numpy()

    conf_low = validation[f"{target}_conf_low"].to_numpy()
    conf_high = validation[f"{target}_conf_high"].to_numpy()

    if target not in ISOFORM_SEEDS:
        raise ValueError(f"No deterministic seed registered for {target}")

    rng = np.random.default_rng(
        ISOFORM_SEEDS[target]
    )

    trial_records = []
    best_model = None
    best_record = None

    for trial_number in range(1, N_TRIALS + 1):
        parameters = sample_parameters(rng)

        trial_seed = BASE_SEED + trial_number

        model, score, best_iteration = fit_trial(
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            conf_low=conf_low,
            conf_high=conf_high,
            parameters=parameters,
            trial_seed=trial_seed,
        )

        record = {
            "trial": trial_number,
            "validation_ST_RAE": score,
            "best_iteration": best_iteration,
            **parameters,
        }

        trial_records.append(record)

        print(
            f"trial={trial_number:02d} "
            f"ST-RAE={score:.6f} "
            f"iteration={best_iteration:4d} "
            f"lr={parameters['learning_rate']:.6f} "
            f"leaves={parameters['num_leaves']:2d} "
            f"min_child_weight={parameters['min_child_weight']:.6f} "
            f"subsample={parameters['subsample']:.4f}"
        )

        if (
            best_record is None
            or score < best_record["validation_ST_RAE"]
        ):
            best_record = record
            best_model = model

    results = pd.DataFrame(trial_records).sort_values(
        "validation_ST_RAE"
    )

    results_path = (
        OUTDIR
        / f"{target}_trials.csv"
    )
    results.to_csv(
        results_path,
        index=False,
    )

    best_path = (
        OUTDIR
        / f"{target}_best.json"
    )

    summary = {
        "target": target,
        "n_trials": N_TRIALS,
        "selection_metric": "validation_ST-RAE",
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "bagging_freq": 1,
        "max_depth": "default",
        "best_trial": best_record,
        "trial_results": str(results_path),
    }

    with open(best_path, "w") as f:
        json.dump(summary, f, indent=2)

    if best_model is None:
        raise RuntimeError("No best model selected.")

    print("\n=== WINNER ===")
    print(json.dumps(best_record, indent=2))
    print("Saved:", results_path)
    print("Saved:", best_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        required=True,
        choices=sorted(ISOFORM_SEEDS),
    )
    args = parser.parse_args()

    tune_isoform(args.target)


if __name__ == "__main__":
    main()
