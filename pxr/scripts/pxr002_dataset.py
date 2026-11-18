"""Tripwire PXR-002 Chemprop datasets (D018 and its 2026-11-18 amendment).

Reads a frozen artifact (the v2 cohort written by pxr002_artifact_v2.py; it never
touches holdout or official-test labels). The fixed v1_rdkit_2d_normalized descriptors
are computed once, cached in the artifact before any learned scaling, and verified
finite on load. Learned preprocessing -- per-head target scaling and descriptor
standardization -- is fit on the training frame only here.

Validation targets are left in original pEC50 units on purpose: the model's output
transform un-scales predictions, so primary-head validation MAE is computed directly
in original units. Validation auxiliary labels are masked.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TASK_COLUMNS = ("pEC50", "log2fc_8p25uM", "log2fc_33uM", "log2fc_99uM")
SMILES_COLUMN = "SMILES"
N_DESCRIPTORS = 200

TRAIN_FRAME = "pxr002_train_frame.csv"
VALIDATION_FRAME = "pxr002_validation_frame.csv"
TRAIN_DESCRIPTORS = "pxr002_train_descriptors.npy"
VALIDATION_DESCRIPTORS = "pxr002_validation_descriptors.npy"
AMENDMENT = "pxr002_eligibility_amendment_v1.csv"


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(artifact_dir: str | Path) -> dict[str, str]:
    """Check SHA256SUMS of a frozen artifact; return name -> hash."""
    artifact_dir = Path(artifact_dir)
    recorded = {}
    for line in (artifact_dir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        recorded[name.strip()] = digest
    for name, digest in recorded.items():
        actual = sha256(artifact_dir / name)
        if actual != digest:
            raise RuntimeError(f"{name} does not match SHA256SUMS: {actual} != {digest}")
    return recorded


def load_artifact(artifact_dir: str | Path) -> dict:
    """Hash-verified frames, cached descriptors and the eligibility amendment."""
    hashes = verify_artifact(artifact_dir)
    d = Path(artifact_dir)
    for required in (TRAIN_FRAME, VALIDATION_FRAME, TRAIN_DESCRIPTORS, VALIDATION_DESCRIPTORS, AMENDMENT):
        if required not in hashes:
            raise RuntimeError(f"{required} is not listed in SHA256SUMS")
    train = pd.read_csv(d / TRAIN_FRAME)
    validation = pd.read_csv(d / VALIDATION_FRAME)
    amendment = pd.read_csv(d / AMENDMENT)
    train_x = np.load(d / TRAIN_DESCRIPTORS, allow_pickle=False)
    val_x = np.load(d / VALIDATION_DESCRIPTORS, allow_pickle=False)

    if set(train["role"]) - {"primary_train", "aux_only"}:
        raise ValueError("training frame contains an unexpected role")
    if set(validation["role"]) != {"primary_validation"}:
        raise ValueError("validation frame must contain primary_validation rows only")
    if validation[list(TASK_COLUMNS[1:])].notna().any().any():
        raise ValueError("validation auxiliary labels must be masked")
    for name, frame, x in (("train", train, train_x), ("validation", validation, val_x)):
        if x.dtype != np.float64 or x.shape != (len(frame), N_DESCRIPTORS):
            raise ValueError(f"{name} descriptor cache has shape {x.shape}/{x.dtype}; expected ({len(frame)}, {N_DESCRIPTORS}) float64")
        if not np.isfinite(x).all():
            raise ValueError(f"{name} descriptor cache contains non-finite values")
    excluded = set(amendment["row_id"])
    if excluded & (set(train["row_id"]) | set(validation["row_id"])):
        raise ValueError("an amended-out molecule is present in the cohort")
    return {
        "train": train,
        "validation": validation,
        "train_x": train_x,
        "val_x": val_x,
        "amendment": amendment,
        "hashes": hashes,
    }


def compute_descriptors(smiles: list[str], allow_nonfinite: bool = False) -> np.ndarray:
    """Fixed v1_rdkit_2d_normalized descriptors (no learned scaling).

    Used by the artifact builder. Non-finite values raise unless explicitly allowed so
    the builder can report them; imputation is not approved.
    """
    from chemprop.featurizers import MoleculeFeaturizerRegistry
    from rdkit import Chem

    featurizer = MoleculeFeaturizerRegistry["v1_rdkit_2d_normalized"]()
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"unparsable SMILES: {smi}")
        rows.append(np.asarray(featurizer(mol), dtype=np.float64))
    matrix = np.vstack(rows)
    bad = ~np.isfinite(matrix)
    if bad.any() and not allow_nonfinite:
        raise ValueError(
            f"{int(bad.sum())} non-finite descriptor values in {int(bad.any(axis=1).sum())} molecules; "
            "imputation is not approved"
        )
    return matrix


def frame_to_dataset(frame: pd.DataFrame, descriptors: np.ndarray):
    from chemprop import data

    if len(frame) != len(descriptors):
        raise ValueError("descriptor rows do not match frame rows")
    points = [
        data.MoleculeDatapoint.from_smi(smi, y, x_d=x)
        for smi, y, x in zip(
            frame[SMILES_COLUMN].tolist(),
            frame[list(TASK_COLUMNS)].to_numpy(dtype=np.float64),
            descriptors,
        )
    ]
    return data.MoleculeDataset(points)


def build_datasets(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_x: np.ndarray,
    val_x: np.ndarray,
) -> dict:
    """Datasets from cached descriptors with train-only target and descriptor scaling."""
    for name, x in (("train", train_x), ("validation", val_x)):
        if not np.isfinite(x).all():
            raise ValueError(f"{name} descriptors contain non-finite values")

    train_ds = frame_to_dataset(train, train_x)
    val_ds = frame_to_dataset(validation, val_x)

    y_scaler = train_ds.normalize_targets()
    xd_scaler = train_ds.normalize_inputs("X_d")
    val_ds.normalize_inputs("X_d", xd_scaler)

    return {
        "train": train_ds,
        "validation": val_ds,
        "y_scaler": y_scaler,
        "xd_scaler": xd_scaler,
        "xd_dim": int(train_x.shape[1]),
    }
