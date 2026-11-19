"""PXR-003 shared helpers: frozen inputs, cohort loading, metrics, hashing.

Reads (via load_cohort): pxr-challenge_TRAIN.csv (Molecule Name, SMILES, pEC50) and the frozen
D014 split assignments (Molecule Name, split). Holdout labels are masked inside
pxr001_baseline.prepare_train_validation_data before any modeling frame exists.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts.pxr001_baseline import (  # noqa: E402
    ID_COLUMN,
    SMILES_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    build_ecfp4_features,
    prepare_train_validation_data,
)

TRAIN_CSV = REPO_ROOT / "pxr/data/openadmet_pxr_release_2026-11-15/pxr-challenge_TRAIN.csv"
ASSIGNMENTS_CSV = REPO_ROOT / "pxr/artifacts/pxr001_split/pxr001_split_assignments.csv"
CONFIG_PATH = REPO_ROOT / "pxr/scripts/pxr003_config.json"

# Recorded in D014 (data) and D014/D017/D018 (assignments).
EXPECTED_TRAIN_SHA256 = "fdc2252bb5763ab5742e78cb1ba692bebe131a972cf83aa1bd45e43d8c5bc223"
EXPECTED_ASSIGNMENTS_SHA256 = "c6028f1b7216ab2ea4bab96616fc02b95ce011be20b1b96f807d1e7d582ece46"
EXPECTED_COUNTS = {"train": 2897, "validation": 621, "holdout": 621}

# PXR-002 neural holdout evidence (SHA-256 recorded in its closed session record).
NEURAL_PREDICTIONS_SHA256 = "45c84c925b9dfc1c5c959e3a95c6b01961667dd19e2338470e8ae26655aa07aa"
NEURAL_DIR = REPO_ROOT / "pxr/artifacts/pxr002_final_20261118"

EVIDENCE_CLASS = "internal-retrospective (Rev 3 line 53); official test not used"
DISCLOSURE = (
    "Internal-retrospective comparison. The neural holdout results were known before the "
    "baselines were configured. No holdout score changed any configuration or chose between "
    "random forest and LightGBM. The baselines train on 2,897 molecules; the neural models on "
    "2,896 (OADMET-0003727 excluded for descriptor eligibility). Validation and holdout sets "
    "are identical across all models."
)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def mae(pred, truth) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float))))


def rmse(pred, truth) -> float:
    diff = np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def spearman(pred, truth) -> float:
    return float(
        pd.Series(np.asarray(pred, dtype=float)).corr(
            pd.Series(np.asarray(truth, dtype=float)), method="spearman"
        )
    )


def metrics(pred, truth) -> dict:
    return {"mae": mae(pred, truth), "spearman": spearman(pred, truth), "rmse": rmse(pred, truth)}


def constant_metrics(train_mean: float, truth) -> dict:
    return metrics(np.full(len(truth), train_mean, dtype=float), truth)


def verify_inputs(train_csv: Path = TRAIN_CSV, assignments_csv: Path = ASSIGNMENTS_CSV) -> dict:
    hashes = {"train_csv": sha256_file(train_csv), "assignments_csv": sha256_file(assignments_csv)}
    if hashes["train_csv"] != EXPECTED_TRAIN_SHA256:
        raise AssertionError(f"TRAIN csv hash {hashes['train_csv']} != recorded {EXPECTED_TRAIN_SHA256}")
    if hashes["assignments_csv"] != EXPECTED_ASSIGNMENTS_SHA256:
        raise AssertionError(
            f"assignments hash {hashes['assignments_csv']} != recorded {EXPECTED_ASSIGNMENTS_SHA256}"
        )
    return hashes


def load_cohort(
    train_csv: Path = TRAIN_CSV,
    assignments_csv: Path = ASSIGNMENTS_CSV,
    verify: bool = True,
    expected_counts: dict | None = None,
):
    """Return (train, validation, holdout_ids, hashes); holdout labels are never returned."""
    hashes = verify_inputs(train_csv, assignments_csv) if verify else {}
    counts = EXPECTED_COUNTS if (verify and expected_counts is None) else expected_counts
    data = pd.read_csv(train_csv, usecols=[ID_COLUMN, SMILES_COLUMN, TARGET_COLUMN])
    assignments = pd.read_csv(assignments_csv, usecols=[ID_COLUMN, SPLIT_COLUMN])
    train, validation = prepare_train_validation_data(data, assignments)
    holdout_ids = set(assignments.loc[assignments[SPLIT_COLUMN].eq("holdout"), ID_COLUMN])
    train_ids, validation_ids = set(train[ID_COLUMN]), set(validation[ID_COLUMN])
    if train_ids & validation_ids or train_ids & holdout_ids or validation_ids & holdout_ids:
        raise AssertionError("train, validation and holdout identities are not pairwise disjoint")
    if counts is not None:
        observed = {"train": len(train), "validation": len(validation), "holdout": len(holdout_ids)}
        if observed != counts:
            raise AssertionError(f"partition sizes {observed} != expected {counts}")
    return train, validation, holdout_ids, hashes


def load_holdout_structures(
    train_csv: Path = TRAIN_CSV, assignments_csv: Path = ASSIGNMENTS_CSV
) -> pd.DataFrame:
    """Holdout identities and SMILES only. The target column is never read."""
    assignments = pd.read_csv(assignments_csv, usecols=[ID_COLUMN, SPLIT_COLUMN])
    holdout_ids = set(assignments.loc[assignments[SPLIT_COLUMN].eq("holdout"), ID_COLUMN])
    frame = pd.read_csv(train_csv, usecols=[ID_COLUMN, SMILES_COLUMN])
    frame = frame.loc[frame[ID_COLUMN].isin(holdout_ids)].reset_index(drop=True)
    if list(frame.columns) != [ID_COLUMN, SMILES_COLUMN]:
        raise AssertionError("holdout structure frame must contain only identity and SMILES")
    if set(frame[ID_COLUMN]) != holdout_ids:
        raise AssertionError("holdout structure ids do not match the frozen assignments")
    return frame


def build_features(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """ECFP4 (radius 2, 2048 bits) via the existing PXR-001 helper; float32 for the learners."""
    feats = build_ecfp4_features(train, validation)
    feats["x_train"] = feats["x_train"].astype(np.float32)
    feats["x_validation"] = feats["x_validation"].astype(np.float32)
    return feats


def holdout_matrix(structures: pd.DataFrame) -> np.ndarray:
    from cyp.scripts.ecfp4 import ecfp4_matrix

    return ecfp4_matrix(structures[SMILES_COLUMN].tolist()).astype(np.float32)


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def library_versions() -> dict:
    import platform

    import sklearn

    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__}
    try:
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__
    except ImportError:
        versions["lightgbm"] = None
    return versions
