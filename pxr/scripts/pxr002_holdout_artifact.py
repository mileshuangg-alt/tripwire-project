"""Tripwire PXR-002 structure-only holdout artifact (final-evaluation protocol, 2026-11-18).

Builds the inputs a trained model needs to PREDICT the frozen D014 internal holdout, and
nothing else: row ids (Molecule Name), released SMILES and the fixed
v1_rdkit_2d_normalized descriptors computed before any learned scaling. The pEC50 column
of the primary table is never read here (usecols excludes it), so this artifact, and
anything staged to CoreHPC from it, carries no holdout labels. Labels are read only by
pxr002_final_eval.py, locally, at the single approved evaluation step.

A non-finite descriptor raises; no exclusion or imputation is approved for the holdout
(a structure-only check on 2026-11-18 found none among the 621 molecules).

Usage (repository root; Chemprop environment with descriptastorus):
    python -I pxr/scripts/pxr002_holdout_artifact.py \
        pxr/data/openadmet_pxr_release_2026-11-15 \
        pxr/artifacts/pxr001_split/pxr001_split_assignments.csv \
        pxr/artifacts/pxr002_holdout_v1
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr002_dataset as ds  # noqa: E402

HOLDOUT_FRAME = "pxr002_holdout_frame.csv"
HOLDOUT_DESCRIPTORS = "pxr002_holdout_descriptors.npy"
ID_COLUMN = "Molecule Name"


def build_holdout_frame(assignments: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """Holdout identities and SMILES only. ``structures`` must not carry labels."""
    leaked = {c for c in structures.columns if c.lower().startswith(("pec50", "emax"))}
    if leaked:
        raise ValueError(f"structure table carries label columns: {sorted(leaked)}")
    holdout = assignments.loc[assignments["split"].eq("holdout"), [ID_COLUMN]]
    frame = holdout.merge(structures[[ID_COLUMN, "SMILES"]], on=ID_COLUMN, how="left", validate="one_to_one")
    if frame["SMILES"].isna().any() or len(frame) != len(holdout):
        raise ValueError("a holdout molecule has no structure")
    frame = frame.sort_values(ID_COLUMN).reset_index(drop=True)
    return frame.rename(columns={ID_COLUMN: "row_id"})[["row_id", "SMILES"]]


def load_holdout_artifact(artifact_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    hashes = ds.verify_artifact(artifact_dir)
    d = Path(artifact_dir)
    frame = pd.read_csv(d / HOLDOUT_FRAME)
    x = np.load(d / HOLDOUT_DESCRIPTORS, allow_pickle=False)
    if list(frame.columns) != ["row_id", "SMILES"]:
        raise ValueError("holdout frame must contain only row_id and SMILES")
    if x.dtype != np.float64 or x.shape != (len(frame), ds.N_DESCRIPTORS) or not np.isfinite(x).all():
        raise ValueError("holdout descriptor cache has an unexpected shape, dtype or non-finite values")
    return frame, x, hashes


def main(data_dir: str, assignments_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refusing to overwrite a frozen artifact")
    out.mkdir(parents=True, exist_ok=True)
    primary_path = Path(data_dir) / "pxr-challenge_TRAIN.csv"
    structures = pd.read_csv(primary_path, usecols=[ID_COLUMN, "SMILES"])  # labels never read
    assignments = pd.read_csv(assignments_path, usecols=[ID_COLUMN, "split"])
    frame = build_holdout_frame(assignments, structures)
    x = ds.compute_descriptors(frame["SMILES"].tolist())  # raises on any non-finite value
    frame.to_csv(out / HOLDOUT_FRAME, index=False)
    np.save(out / HOLDOUT_DESCRIPTORS, np.ascontiguousarray(x, dtype=np.float64), allow_pickle=False)
    names = [HOLDOUT_FRAME, HOLDOUT_DESCRIPTORS]
    (out / "SHA256SUMS").write_text("".join(f"{ds.sha256(out / n)}  {n}\n" for n in names))
    (out / "PROVENANCE.txt").write_text(
        "PXR-002 structure-only holdout artifact (2026-11-18)\n"
        f"assignments sha256: {ds.sha256(assignments_path)}\n"
        f"primary table sha256: {ds.sha256(primary_path)} (columns read: Molecule Name, SMILES only)\n"
        f"holdout molecules: {len(frame)}; descriptors: v1_rdkit_2d_normalized, cached before learned scaling\n"
    )
    print(f"holdout molecules {len(frame)}; descriptors {x.shape}; no labels read")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
