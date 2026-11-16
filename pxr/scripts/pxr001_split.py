"""Tripwire PXR-001 predeclared scaffold holdout generator.

Implements the split frozen by D014 and its three amendments in notes/Decisions.md:

  grouping      Bemis-Murcko scaffold, chirality excluded
  constraint 1  whole scaffold groups never cross a partition boundary
  constraint 2  near-duplicate groups stay intact (Rev 3 line 109); grouping keys
                sharing an InChIKey connectivity block are merged into one unit
  amendment 4   a ringless molecule has an empty Bemis-Murcko scaffold; each one forms
                its own group rather than all of them sharing the empty string, while
                the connectivity-block merge still keeps related ringless molecules
                together
  constraint 3  group assignment stratified on group-median pEC50
  constraint 4  single fixed seed; assignment frozen as a CSV keyed on "Molecule Name"
  structure     train / validation / holdout at 70 / 15 / 15 (Rev 3 line 109 requires
                a validation partition distinct from the holdout)
  seed          20261115

The script reads the preserved primary training table, writes one assignment CSV, and
asserts every invariant before writing. It applies no exclusion and no standardization:
the official SMILES strings are used as released. Realized partition sizes will not land
exactly on the nominal percentages because scaffold groups are assigned indivisibly.

Usage (from the repository root):
    python3 -I pxr/scripts/pxr001_split.py \
        pxr/data/openadmet_pxr_release_2026-11-15 \
        pxr/artifacts/pxr001_split
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SEED = 20261115
PROPORTIONS = {"train": 0.70, "validation": 0.15, "holdout": 0.15}
PARTITIONS = ("train", "validation", "holdout")
ID_COLUMN = "Molecule Name"
TARGET = "pEC50"
N_STRATA = 10


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaffold_and_connectivity(smiles_values):
    """Bemis-Murcko scaffold SMILES (chirality excluded) and InChIKey connectivity block."""
    scaffolds, blocks = [], []
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Unable to parse SMILES: {smiles}")
        scaffolds.append(
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        )
        blocks.append(Chem.MolToInchiKey(mol).split("-")[0])
    return scaffolds, blocks


def group_keys(scaffolds):
    """One grouping key per molecule.

    A ringless molecule has an empty Bemis-Murcko scaffold. Keying on that empty string
    would lump every acyclic molecule into a single group, which is an artifact of the
    standard implementation rather than a shared scaffold, so each ringless molecule gets
    its own key (D014 Amendment 4). The connectivity-block merge below still applies to
    them, so ringless molecules that are genuinely related are still kept together.
    """
    return [f"S:{s}" if s else f"R:{i}" for i, s in enumerate(scaffolds)]


def merge_groups(scaffolds, blocks):
    """Union grouping keys that share an InChIKey connectivity block (constraint 2).

    Returns one integer group id per molecule. Group ids are assigned in order of first
    appearance so they do not depend on dictionary iteration order.
    """
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    keys = group_keys(scaffolds)
    for key in keys:
        find(key)
    for key, block in zip(keys, blocks):
        union(key, f"B:{block}")

    ids: dict[str, int] = {}
    group_ids = []
    for key in keys:
        root = find(key)
        if root not in ids:
            ids[root] = len(ids)
        group_ids.append(ids[root])
    return group_ids


def assign_groups(group_frame: pd.DataFrame, seed: int = SEED) -> dict[int, str]:
    """Assign whole groups to partitions, stratified on group-median pEC50.

    Groups are placed in descending size order within each stratum, each going to the
    partition whose filled share is currently furthest behind its target share, measured
    as placed / proportion. Comparing ratios rather than absolute shortfalls is what makes
    stratification work: an absolute shortfall would send every early stratum to the
    largest partition, so the partitions would differ systematically in pEC50. Size-
    descending placement keeps large groups from overshooting a small partition at the
    end. The seed only breaks ties between equal-sized groups, so the result is
    deterministic.
    """
    placed = {name: 0.0 for name in PARTITIONS}
    assignment: dict[int, str] = {}

    ranks = group_frame["median_target"].rank(method="first")
    strata = np.minimum((ranks - 1) * N_STRATA // len(group_frame), N_STRATA - 1)

    rng = np.random.default_rng(seed)
    frame = group_frame.assign(_stratum=strata.astype(int), _tie=rng.random(len(group_frame)))

    for stratum in range(N_STRATA):
        block = frame[frame["_stratum"] == stratum].sort_values(
            ["size", "_tie"], ascending=[False, True]
        )
        for group_id, size in zip(block["group_id"], block["size"]):
            filled = {name: placed[name] / PROPORTIONS[name] for name in PARTITIONS}
            choice = min(PARTITIONS, key=lambda name: (filled[name], -PROPORTIONS[name]))
            assignment[int(group_id)] = choice
            placed[choice] += float(size)

    return assignment


def build(data_dir: str) -> pd.DataFrame:
    train_path = Path(data_dir) / "pxr-challenge_TRAIN.csv"
    frame = pd.read_csv(train_path)

    scaffolds, blocks = scaffold_and_connectivity(frame["SMILES"])
    frame = frame.assign(scaffold_smiles=scaffolds, connectivity_block=blocks)
    frame["group_id"] = merge_groups(scaffolds, blocks)

    groups = (
        frame.groupby("group_id")
        .agg(size=(ID_COLUMN, "size"), median_target=(TARGET, "median"))
        .reset_index()
    )
    assignment = assign_groups(groups)
    frame["split"] = frame["group_id"].map(assignment)

    check_invariants(frame)
    return frame[[ID_COLUMN, "OCNT_ID", "scaffold_smiles", "connectivity_block",
                  "group_id", "split"]]


def check_invariants(frame: pd.DataFrame) -> None:
    """Assert the D014 invariants. Raises AssertionError rather than writing a bad split."""
    assert frame["split"].notna().all(), "some molecules were not assigned"
    assert set(frame["split"]) <= set(PARTITIONS), "unexpected partition label"
    assert frame[ID_COLUMN].is_unique, "duplicate molecule identity"

    # The empty scaffold is excluded from the scaffold check: it marks a ringless
    # molecule, not a shared scaffold, and under Amendment 4 those molecules are
    # grouped individually, so the empty string is expected to span partitions.
    ringed = frame[frame["scaffold_smiles"].astype(str) != ""]
    for column, label, subject in [("scaffold_smiles", "scaffold", ringed),
                                   ("connectivity_block", "connectivity block", frame),
                                   ("group_id", "group", frame)]:
        spanning = subject.groupby(column)["split"].nunique()
        offenders = spanning[spanning > 1]
        assert offenders.empty, (
            f"{label} spans more than one partition: {list(offenders.index[:5])}"
        )


def write_artifact(frame: pd.DataFrame, out_dir: str, data_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "pxr001_split_assignments.csv"
    frame.to_csv(csv_path, index=False)

    counts = frame["split"].value_counts()
    groups = frame.groupby("split")["group_id"].nunique()
    lines = [
        "Tripwire PXR-001 predeclared scaffold holdout",
        "Produced: 2026-11-15",
        "",
        "Governing record: notes/Decisions.md D014 and Amendments 1-3.",
        f"Generator: pxr/scripts/pxr001_split.py   seed {SEED}",
        f"Input: {data_dir}/pxr-challenge_TRAIN.csv",
        f"  SHA-256 {sha256(Path(data_dir) / 'pxr-challenge_TRAIN.csv')}",
        "",
        "Grouping: Bemis-Murcko scaffold, chirality excluded. Grouping keys sharing an",
        "InChIKey connectivity block are merged into one assignment unit, so near-duplicates",
        "cannot straddle a partition boundary (D014 constraint 2; Rev 3 line 109).",
        "Ringless molecules have an empty Bemis-Murcko scaffold and are grouped individually",
        "rather than lumped together by that empty string (D014 Amendment 4); the",
        "connectivity-block merge still keeps related ringless molecules in one group.",
        "Assignment: whole groups only, stratified on group-median pEC50 across 10 strata.",
        "No exclusion, filtering or molecule standardization is applied; the official SMILES",
        "strings are used as released.",
        "",
        "Realized sizes (nominal 70 / 15 / 15; whole groups are indivisible, so realized",
        "fractions do not land exactly on the nominal percentages):",
    ]
    total = len(frame)
    for name in PARTITIONS:
        lines.append(
            f"  {name:<11} {int(counts.get(name, 0)):>5} molecules "
            f"({100 * counts.get(name, 0) / total:5.2f}%)   "
            f"{int(groups.get(name, 0)):>5} groups"
        )
    lines += [
        "",
        f"Total molecules {total}; total assignment groups {frame['group_id'].nunique()}.",
        "",
        "Invariants asserted before the artifact was written:",
        "  - every molecule assigned to exactly one partition;",
        "  - no scaffold spans more than one partition (the empty scaffold is excluded:",
        "    it marks a ringless molecule, not a shared scaffold, and those are grouped",
        "    individually under Amendment 4);",
        "  - no InChIKey connectivity block spans more than one partition;",
        "  - molecule identity unique across the assignment.",
        "",
        "The holdout is not a tuning target (Rev 3 line 109). Primary metric is MAE of",
        "predicted pEC50 with Spearman for ranking quality (Rev 3 line 117).",
    ]
    (out / "PROVENANCE.txt").write_text("\n".join(lines) + "\n")

    sums = [f"{sha256(csv_path)}  {out_dir}/{csv_path.name}"]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    print("\n".join(lines))


def main() -> None:
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    frame = build(data_dir)
    write_artifact(frame, out_dir, data_dir)


if __name__ == "__main__":
    main()
