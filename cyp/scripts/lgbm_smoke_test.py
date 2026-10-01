from __future__ import annotations

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

TARGET = "CYP3A4_pIC50_direct_inhibition"
CONF_LOW = TARGET + "_conf_low"
CONF_HIGH = TARGET + "_conf_high"


def load_train_validation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(TRAIN_PATH)
    assignments = pd.read_csv(SPLIT_PATH)

    merged = data.merge(
        assignments,
        on="Molecule_Name",
        how="inner",
        validate="one_to_one",
    )

    if merged["split"].isin(["test"]).any():
        # Test rows may exist in the merged dataframe, but this function
        # must explicitly remove them before returning anything.
        pass

    train = merged[
        (merged["split"] == "train")
        & merged[TARGET].notna()
    ].copy()

    validation = merged[
        (merged["split"] == "validation")
        & merged[TARGET].notna()
    ].copy()

    if len(train) == 0 or len(validation) == 0:
        raise ValueError("Train or validation set is empty.")

    if train["Molecule_Name"].isin(validation["Molecule_Name"]).any():
        raise AssertionError("Train/validation identity overlap detected.")

    return train, validation


def train_smoke_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> lgb.LGBMRegressor:
    x_train = ecfp4_matrix(train["SMILES"])
    x_validation = ecfp4_matrix(validation["SMILES"])

    y_train = train[TARGET].to_numpy()
    y_validation = validation[TARGET].to_numpy()

    conf_low = validation[CONF_LOW].to_numpy()
    conf_high = validation[CONF_HIGH].to_numpy()

    evaluator = make_st_rae_evaluator(
        y_true=y_validation,
        conf_low=conf_low,
        conf_high=conf_high,
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_weight=1.0,
        subsample=0.8,
        bagging_freq=1,
        random_state=20261001,
        verbosity=-1,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        eval_names=["validation"],
        eval_metric=evaluator,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )

    return model


def main() -> None:
    train, validation = load_train_validation_data()

    print("Train rows:", len(train))
    print("Validation rows:", len(validation))

    print(
        "Train/validation overlap:",
        len(set(train["Molecule_Name"]) & set(validation["Molecule_Name"])),
    )

    model = train_smoke_model(train, validation)

    if model.best_iteration_ is None:
        raise AssertionError("LightGBM did not report a best iteration.")

    if not (1 <= model.best_iteration_ <= 2000):
        raise AssertionError(
            f"Unexpected best iteration: {model.best_iteration_}"
        )

    predictions = model.predict(
        ecfp4_matrix(validation["SMILES"]),
        num_iteration=model.best_iteration_,
    )

    if len(predictions) != len(validation):
        raise AssertionError("Prediction count does not match validation count.")

    if not np.isfinite(predictions).all():
        raise AssertionError("Validation predictions contain non-finite values.")

    print("Best iteration:", model.best_iteration_)
    print("Validation predictions:", len(predictions))
    print("PASS: CYP3A4 LightGBM smoke test.")


if __name__ == "__main__":
    main()
