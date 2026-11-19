"""Tripwire PXR-004 data and stage-1 monitor (D020). Authorizes no run by itself.

Descriptor-free inputs on the frozen v1 frames (pxr/artifacts/pxr002_aux: 2,897 primary-train
plus 7,922 eligible auxiliary-only training molecules, 621 primary-validation molecules).
The holdout and official-test labels are never loaded.

Stage-1 monitor (D020, pilot convention, a new internal rule that does not change D018
item 1): a fixed set of whole D014 groups drawn from the auxiliary-labelled training pool.

  * seed 20261119, one draw, shared by all training seeds, saved as a hashed artifact;
  * group 417 (the large series, 278 molecules) is never eligible and stays in training;
  * groups are accumulated in a seeded random order until about 10% of the
    auxiliary-labelled molecules are held out; no search across draws is performed;
  * monitor molecules' auxiliary labels never enter stage-1 training; their pEC50 labels
    stay in stage-2 training (stage 2 uses every primary-train molecule).

Usage (repository root, tripwire_cyp002 environment, run by the user):
    python -I pxr/scripts/pxr004_data.py build-monitor \
        --artifact-dir pxr/artifacts/pxr002_aux \
        --assignments pxr/artifacts/pxr001_split/pxr001_split_assignments.csv \
        --out-dir pxr/artifacts/pxr004_monitor
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRIMARY_TASK = "pEC50"
AUX_TASKS = ("log2fc_8p25uM", "log2fc_33uM", "log2fc_99uM")
SMILES_COLUMN = "SMILES"

MONITOR_SEED = 20261119
MONITOR_FRACTION = 0.10
EXCLUDED_GROUP = 417
FROZEN_SPLIT_SHA256 = "c6028f1b7216ab2ea4bab96616fc02b95ce011be20b1b96f807d1e7d582ece46"

TRAIN_FRAME = "pxr002_train_frame.csv"
VALIDATION_FRAME = "pxr002_validation_frame.csv"
ELIGIBILITY = "pxr002_eligibility.csv"
MONITOR_FILE = "pxr004_monitor_assignments.csv"
MONITOR_COLUMNS = ["row_id", "OCNT_ID", "group_id", "role", "n_aux", "in_monitor"]


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sums(directory: str | Path, required: tuple[str, ...] = ()) -> dict[str, str]:
    """Check every entry of a directory's SHA256SUMS; return name -> hash."""
    directory = Path(directory)
    recorded = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        recorded[name.strip()] = digest
    for name in required:
        if name not in recorded:
            raise RuntimeError(f"{name} is not listed in {directory}/SHA256SUMS")
    for name, digest in recorded.items():
        actual = sha256(directory / name)
        if actual != digest:
            raise RuntimeError(f"{directory}/{name} does not match SHA256SUMS: {actual} != {digest}")
    return recorded


# ------------------------------------------------------------------ frames and groups
def load_frames(artifact_dir: str | Path) -> dict:
    """Hash-verified v1 frames (no descriptors). Validation auxiliary labels must be masked."""
    artifact_dir = Path(artifact_dir)
    hashes = verify_sums(artifact_dir, required=(TRAIN_FRAME, VALIDATION_FRAME, ELIGIBILITY))
    train = pd.read_csv(artifact_dir / TRAIN_FRAME)
    validation = pd.read_csv(artifact_dir / VALIDATION_FRAME)
    eligibility = pd.read_csv(artifact_dir / ELIGIBILITY)
    if set(train["role"]) - {"primary_train", "aux_only"}:
        raise ValueError("training frame contains an unexpected role")
    if set(validation["role"]) != {"primary_validation"}:
        raise ValueError("validation frame must contain primary_validation rows only")
    if validation[list(AUX_TASKS)].notna().any().any():
        raise ValueError("validation auxiliary labels must be masked")
    if train.loc[train["role"] == "primary_train", PRIMARY_TASK].isna().any():
        raise ValueError("a primary-train molecule has no pEC50")
    if train.loc[train["role"] == "aux_only", PRIMARY_TASK].notna().any():
        raise ValueError("an auxiliary-only molecule carries a pEC50 label")
    return {"train": train, "validation": validation, "eligibility": eligibility, "hashes": hashes}


def attach_groups(train: pd.DataFrame, eligibility: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Add OCNT_ID, D014 group_id and n_aux (count of auxiliary labels) to the training frame."""
    frame = train.copy()
    primary = frame["role"] == "primary_train"
    to_ocnt = assignments.set_index("Molecule Name")["OCNT_ID"]
    frame["OCNT_ID"] = frame["row_id"]
    frame.loc[primary, "OCNT_ID"] = frame.loc[primary, "row_id"].map(to_ocnt)
    if frame["OCNT_ID"].isna().any():
        raise ValueError("a primary-train molecule is missing from the frozen split assignments")
    group_of = eligibility.set_index("OCNT_ID")["group_id"]
    frame["group_id"] = frame["OCNT_ID"].map(group_of)
    if frame["group_id"].isna().any():
        raise ValueError("a training molecule has no D014 group")
    frame["group_id"] = frame["group_id"].astype(int)
    frame["n_aux"] = frame[list(AUX_TASKS)].notna().sum(axis=1)
    return frame


def draw_monitor_groups(
    frame: pd.DataFrame,
    seed: int = MONITOR_SEED,
    fraction: float = MONITOR_FRACTION,
    excluded_group: int = EXCLUDED_GROUP,
) -> list[int]:
    """Whole D014 groups of auxiliary-labelled molecules, seeded; one draw, no search.

    Groups are listed in ascending id, shuffled with ``default_rng(seed)``, and accumulated
    until the held-out count of auxiliary-labelled molecules reaches ``fraction`` of them.
    """
    labelled = frame[frame["n_aux"] > 0]
    sizes = labelled.groupby("group_id").size()
    order = np.array(sorted(g for g in sizes.index if g != excluded_group), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    target = fraction * len(labelled)
    chosen: list[int] = []
    total = 0
    for gid in order:
        if total >= target:
            break
        chosen.append(int(gid))
        total += int(sizes[gid])
    return chosen


def monitor_table(frame: pd.DataFrame, chosen: list[int]) -> pd.DataFrame:
    table = frame[["row_id", "OCNT_ID", "group_id", "role", "n_aux"]].copy()
    table["in_monitor"] = table["group_id"].isin(set(chosen))
    return table[MONITOR_COLUMNS]


def build_monitor_artifact(artifact_dir, assignments_path, out_dir) -> dict:
    """Write the one monitor draw as a hashed artifact. Refuses to overwrite."""
    out_dir = Path(out_dir)
    if (out_dir / MONITOR_FILE).exists():
        raise RuntimeError(f"{out_dir / MONITOR_FILE} exists; refusing to overwrite")
    if sha256(assignments_path) != FROZEN_SPLIT_SHA256:
        raise RuntimeError("split assignments do not match the frozen D014 hash")
    frames = load_frames(artifact_dir)
    assignments = pd.read_csv(assignments_path)
    frame = attach_groups(frames["train"], frames["eligibility"], assignments)
    chosen = draw_monitor_groups(frame)
    table = monitor_table(frame, chosen)
    monitor = frame[frame["group_id"].isin(set(chosen)) & (frame["n_aux"] > 0)]
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / MONITOR_FILE, index=False)
    counts = {
        "monitor_groups": len(chosen),
        "monitor_molecules_labelled": int(len(monitor)),
        "monitor_labels": {t: int(monitor[t].notna().sum()) for t in AUX_TASKS},
        "monitor_primary_train_molecules": int((monitor["role"] == "primary_train").sum()),
        "labelled_training_molecules": int((frame["n_aux"] > 0).sum()),
        "stage1_train_molecules": int(((frame["n_aux"] > 0) & ~frame["group_id"].isin(set(chosen))).sum()),
    }
    (out_dir / "PROVENANCE.txt").write_text(
        "Tripwire-PXR-004 stage-1 monitor (D020). One draw; no search across draws.\n"
        f"seed={MONITOR_SEED} fraction={MONITOR_FRACTION} excluded_group={EXCLUDED_GROUP}\n"
        f"input artifact: {artifact_dir} (SHA256SUMS verified)\n"
        f"frozen split assignments sha256={FROZEN_SPLIT_SHA256}\n"
        f"script: pxr/scripts/pxr004_data.py sha256={sha256(Path(__file__))}\n"
        + "".join(f"{k}={v}\n" for k, v in counts.items())
    )
    (out_dir / "SHA256SUMS").write_text(f"{sha256(out_dir / MONITOR_FILE)}  {MONITOR_FILE}\n")
    return counts


def load_monitor(monitor_dir: str | Path, frame: pd.DataFrame) -> set[int]:
    """Hash-verified monitor group ids; checks the table matches the frame exactly."""
    monitor_dir = Path(monitor_dir)
    verify_sums(monitor_dir, required=(MONITOR_FILE,))
    table = pd.read_csv(monitor_dir / MONITOR_FILE)
    if list(table.columns) != MONITOR_COLUMNS:
        raise ValueError("monitor table has unexpected columns")
    merged = frame[["row_id", "group_id", "n_aux"]].merge(
        table[["row_id", "group_id", "n_aux", "in_monitor"]], on="row_id", how="outer", suffixes=("", "_m"), indicator=True
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("monitor table and training frame differ in molecules")
    if (merged["group_id"] != merged["group_id_m"]).any() or (merged["n_aux"] != merged["n_aux_m"]).any():
        raise ValueError("monitor table disagrees with the training frame on groups or labels")
    in_monitor = table.loc[table["in_monitor"].astype(bool), "group_id"]
    groups = set(int(g) for g in in_monitor)
    if EXCLUDED_GROUP in groups:
        raise ValueError(f"excluded group {EXCLUDED_GROUP} is in the monitor")
    # whole groups only
    partial = table.groupby("group_id")["in_monitor"].nunique()
    if (partial > 1).any():
        raise ValueError("a group is only partly in the monitor")
    return groups


# --------------------------------------------------------------------------- datasets
def make_featurizer():
    """V2 multi-hot atoms and default bonds: the CheMeleon requirement (d_v 72, d_e 14)."""
    from chemprop.featurizers import MultiHotAtomFeaturizer, SimpleMoleculeMolGraphFeaturizer

    return SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v2())


def frame_to_dataset(frame: pd.DataFrame, columns: tuple[str, ...]):
    """Descriptor-free MoleculeDataset over the given task columns (NaN = missing label)."""
    from chemprop import data

    points = [
        data.MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(frame[SMILES_COLUMN].tolist(), frame[list(columns)].to_numpy(dtype=np.float64))
    ]
    return data.MoleculeDataset(points, featurizer=make_featurizer())


def stage1_frames(frame: pd.DataFrame, monitor_groups: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Auxiliary-labelled molecules split by the fixed monitor groups."""
    labelled = frame[frame["n_aux"] > 0]
    is_monitor = labelled["group_id"].isin(monitor_groups)
    return labelled[~is_monitor].reset_index(drop=True), labelled[is_monitor].reset_index(drop=True)


def build_stage1_datasets(train_frame: pd.DataFrame, monitor_frame: pd.DataFrame) -> dict:
    """Per-head scaler fit on the post-split stage-1 training labels only, then frozen.

    Monitor targets are transformed with that frozen scaler (standardized units), so the
    pooled masked MSE is computed directly on standardized predictions and targets.
    """
    train = frame_to_dataset(train_frame, AUX_TASKS)
    monitor = frame_to_dataset(monitor_frame, AUX_TASKS)
    scaler = train.normalize_targets()
    monitor.normalize_targets(scaler)
    return {"train": train, "monitor": monitor, "y_scaler": scaler}


def build_stage2_datasets(frame: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """pEC50 only. Training is every primary-train molecule (monitor molecules included).

    Validation targets stay in original units; predictions are un-scaled in the validation
    hook, exactly as in PXR-002.
    """
    primary = frame[frame["role"] == "primary_train"].reset_index(drop=True)
    train = frame_to_dataset(primary, (PRIMARY_TASK,))
    val = frame_to_dataset(validation, (PRIMARY_TASK,))
    scaler = train.normalize_targets()
    return {"train": train, "validation": val, "y_scaler": scaler, "n_train": len(primary)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-monitor")
    b.add_argument("--artifact-dir", required=True, type=Path)
    b.add_argument("--assignments", required=True, type=Path)
    b.add_argument("--out-dir", required=True, type=Path)
    a = p.parse_args()
    if a.command == "build-monitor":
        counts = build_monitor_artifact(a.artifact_dir, a.assignments, a.out_dir)
        for k, v in counts.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
