"""Tripwire PXR-002 cohort v2: one-row descriptor eligibility amendment + descriptor cache.

Context (2026-11-17/18). The runtime-estimate job 3106209 failed because one primary-train
molecule, OADMET-0003727 (auranofin), has non-finite RDKit partial-charge descriptors:
MaxAbsPartialCharge, MaxPartialCharge, MinAbsPartialCharge and MinPartialCharge. The raw
(un-normalized) values are already NaN, so the normalization transform is not at fault.
Miles decided to exclude that molecule from the descriptor-based training cohort rather
than impute.

This builder reads the frozen v1 artifact (pxr/artifacts/pxr002_aux, never modified),
computes the fixed v1_rdkit_2d_normalized descriptors once for every training and
validation row, requires that the non-finite rows and descriptors are exactly the
approved ones, drops only that row, and writes a new versioned artifact: frames,
descriptor caches (before any learned scaling), a one-row amendment CSV, counts,
SHA256SUMS and PROVENANCE. No other molecule moves; the D014 split is untouched. The
auxiliary coefficient is recomputed from this artifact's counts at training time
(pxr002_loss_weights), preserving the nominal 65:35 allocation.

Comparisons intended to isolate auxiliary-data effects must use this same cohort.

Usage (repository root; Chemprop environment with descriptastorus):
    python -I pxr/scripts/pxr002_artifact_v2.py \
        pxr/artifacts/pxr002_aux pxr/artifacts/pxr002_aux_v2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr002_dataset as ds  # noqa: E402
from pxr.scripts.pxr002_aux_data import TASK_COLUMNS, label_counts  # noqa: E402

APPROVED_EXCLUSION_ROW = "OADMET-0003727"
APPROVED_DESCRIPTORS = ("MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge")
AMENDMENT_REASON = (
    "raw RDKit partial-charge descriptors are NaN (not a normalization artifact); "
    "excluded from the descriptor-based training cohort; imputation not approved"
)


def descriptor_names() -> list[str]:
    from descriptastorus.descriptors import rdNormalizedDescriptors

    return [c[0] if isinstance(c, tuple) else c for c in rdNormalizedDescriptors.RDKit2DNormalized().columns]


def apply_descriptor_exclusion(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_x: np.ndarray,
    val_x: np.ndarray,
    names: list[str],
    approved_row: str = APPROVED_EXCLUSION_ROW,
    approved_descriptors: tuple[str, ...] = APPROVED_DESCRIPTORS,
):
    """Drop exactly the approved row; refuse anything else non-finite."""
    if len(train) != len(train_x) or len(validation) != len(val_x):
        raise ValueError("frames and descriptor arrays are misaligned")
    if not np.isfinite(val_x).all():
        raise ValueError("a validation molecule has non-finite descriptors; no exclusion is approved for validation")
    bad_rows = np.where(~np.isfinite(train_x).all(axis=1))[0]
    flagged = train["row_id"].iloc[bad_rows].tolist()
    if flagged != [approved_row]:
        raise ValueError(f"non-finite descriptor rows {flagged} differ from the approved exclusion [{approved_row!r}]")
    bad_cols = sorted(set(np.where(~np.isfinite(train_x[bad_rows]))[1].tolist()))
    flagged_names = [names[j] for j in bad_cols]
    if sorted(flagged_names) != sorted(approved_descriptors):
        raise ValueError(f"non-finite descriptors {flagged_names} differ from the approved set {list(approved_descriptors)}")

    keep = np.ones(len(train), dtype=bool)
    keep[bad_rows] = False
    removed = train.iloc[bad_rows].copy()
    amendment = pd.DataFrame(
        {
            "row_id": removed["row_id"].to_numpy(),
            "role": removed["role"].to_numpy(),
            "SMILES": removed["SMILES"].to_numpy(),
            "action": "exclude_from_descriptor_training_cohort",
            "non_finite_descriptors": ";".join(flagged_names),
            "reason": AMENDMENT_REASON,
        }
    )
    return train.loc[keep].reset_index(drop=True), train_x[keep], amendment, removed


def write_artifact_v2(out: Path, train, validation, train_x, val_x, amendment, summary: dict, provenance: list[str]) -> None:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refusing to overwrite a frozen artifact")
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / ds.TRAIN_FRAME, index=False)
    validation.to_csv(out / ds.VALIDATION_FRAME, index=False)
    np.save(out / ds.TRAIN_DESCRIPTORS, np.ascontiguousarray(train_x, dtype=np.float64), allow_pickle=False)
    np.save(out / ds.VALIDATION_DESCRIPTORS, np.ascontiguousarray(val_x, dtype=np.float64), allow_pickle=False)
    amendment.to_csv(out / ds.AMENDMENT, index=False)
    (out / "pxr002_counts.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    names = [ds.TRAIN_FRAME, ds.VALIDATION_FRAME, ds.TRAIN_DESCRIPTORS, ds.VALIDATION_DESCRIPTORS, ds.AMENDMENT, "pxr002_counts.json"]
    (out / "SHA256SUMS").write_text("".join(f"{ds.sha256(out / n)}  {n}\n" for n in names))
    (out / "PROVENANCE.txt").write_text("\n".join(provenance) + "\n")


def main(v1_dir: str, out_dir: str) -> None:
    import chemprop
    import descriptastorus
    import rdkit

    v1 = Path(v1_dir)
    v1_hashes = ds.verify_artifact(v1)
    train = pd.read_csv(v1 / "pxr002_train_frame.csv")
    validation = pd.read_csv(v1 / "pxr002_validation_frame.csv")

    print(f"computing fixed descriptors for {len(train)} + {len(validation)} molecules (several minutes)")
    train_x = ds.compute_descriptors(train["SMILES"].tolist(), allow_nonfinite=True)
    val_x = ds.compute_descriptors(validation["SMILES"].tolist(), allow_nonfinite=True)

    names = descriptor_names()
    train_v2, train_x_v2, amendment, removed = apply_descriptor_exclusion(train, validation, train_x, val_x, names)

    counts_v1, counts_v2 = label_counts(train), label_counts(train_v2)
    summary = {
        "cohort_version": "v2",
        "parent_artifact": str(v1),
        "parent_SHA256SUMS_sha256": ds.sha256(v1 / "SHA256SUMS"),
        "excluded_rows": amendment["row_id"].tolist(),
        "excluded_row_labels": {t: (None if pd.isna(removed[t].iloc[0]) else float(removed[t].iloc[0])) for t in TASK_COLUMNS},
        "train_molecules": int(len(train_v2)),
        "validation_molecules": int(len(validation)),
        "train_label_counts_v1": counts_v1,
        "train_label_counts": counts_v2,
        "descriptor_set": "v1_rdkit_2d_normalized",
        "descriptor_count": int(train_x_v2.shape[1]),
    }
    provenance = [
        "PXR-002 cohort v2 (D018 amendment, 2026-11-18): one-row descriptor eligibility amendment",
        f"parent v1 SHA256SUMS sha256: {summary['parent_SHA256SUMS_sha256']}",
        f"parent v1 file hashes: {json.dumps(v1_hashes, sort_keys=True)}",
        f"versions: chemprop {chemprop.__version__}, descriptastorus {getattr(descriptastorus, '__version__', 'unknown')}, rdkit {rdkit.__version__}",
        "descriptors cached before any learned scaling; learned scaling is training-only (pxr002_dataset.build_datasets)",
    ]
    write_artifact_v2(Path(out_dir), train_v2, validation, train_x_v2, val_x, amendment, summary, provenance)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
