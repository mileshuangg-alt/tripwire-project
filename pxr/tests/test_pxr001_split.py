"""Invariant tests for the PXR-001 predeclared scaffold holdout generator.

These exercise the generator on synthetic molecules, so they do not depend on the
preserved release being present. They check the D014 invariants, including the cases
that would silently produce a leaking split: a scaffold straddling a boundary, a
near-duplicate pair separated, and a non-deterministic assignment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr001_split as split  # noqa: E402


def test_merge_groups_unions_shared_connectivity_block():
    """Two different scaffolds sharing a connectivity block must land in one group."""
    scaffolds = ["c1ccccc1", "c1ccncc1", "c1ccccc1"]
    blocks = ["AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"]
    groups = split.merge_groups(scaffolds, blocks)
    assert groups[0] == groups[1] == groups[2], (
        "scaffolds joined through a shared connectivity block must merge transitively"
    )


def test_merge_groups_keeps_unrelated_scaffolds_apart():
    scaffolds = ["c1ccccc1", "c1ccncc1"]
    blocks = ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"]
    groups = split.merge_groups(scaffolds, blocks)
    assert groups[0] != groups[1]


def test_merge_groups_is_order_stable():
    scaffolds = ["c1ccccc1"] * 3 + ["c1ccncc1"] * 2
    blocks = [f"K{i:013d}" for i in range(5)]
    assert split.merge_groups(scaffolds, blocks) == [0, 0, 0, 1, 1]


def test_ringless_molecules_get_their_own_groups():
    """Empty scaffolds must not lump every acyclic molecule together (Amendment 4)."""
    scaffolds = ["", "", "c1ccccc1"]
    blocks = ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"]
    groups = split.merge_groups(scaffolds, blocks)
    assert groups[0] != groups[1], "two unrelated ringless molecules were grouped together"
    assert len({*groups}) == 3


def test_ringless_molecules_sharing_a_connectivity_block_stay_together():
    """Having no rings does not prove they are unrelated: the block merge still applies."""
    scaffolds = ["", ""]
    blocks = ["AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA"]
    groups = split.merge_groups(scaffolds, blocks)
    assert groups[0] == groups[1]


def test_real_acyclic_molecule_has_an_empty_scaffold():
    """Guards the premise of Amendment 4 against an RDKit behaviour change."""
    scaffolds, _ = split.scaffold_and_connectivity(["CCCCO", "CC(C)CN"])
    assert scaffolds == ["", ""]


def test_check_invariants_allows_the_empty_scaffold_to_span_partitions():
    frame = pd.DataFrame(
        {
            split.ID_COLUMN: ["A", "B"],
            "scaffold_smiles": ["", ""],
            "connectivity_block": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
            "group_id": [0, 1],
            "split": ["train", "holdout"],
        }
    )
    split.check_invariants(frame)  # must not raise


def test_check_invariants_still_catches_a_split_ringless_near_duplicate_pair():
    frame = pd.DataFrame(
        {
            split.ID_COLUMN: ["A", "B"],
            "scaffold_smiles": ["", ""],
            "connectivity_block": ["AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA"],
            "group_id": [0, 1],
            "split": ["train", "holdout"],
        }
    )
    with pytest.raises(AssertionError, match="connectivity block spans"):
        split.check_invariants(frame)


def _synthetic_frame(n_groups: int = 60, per_group: int = 5) -> pd.DataFrame:
    rows = []
    for group in range(n_groups):
        for member in range(per_group):
            rows.append(
                {
                    "group_id": group,
                    split.ID_COLUMN: f"OADMET-{group:04d}-{member}",
                    split.TARGET: 3.0 + group * 0.05,
                }
            )
    return pd.DataFrame(rows)


def _assign(frame: pd.DataFrame, seed: int = split.SEED) -> pd.DataFrame:
    groups = (
        frame.groupby("group_id")
        .agg(size=(split.ID_COLUMN, "size"), median_target=(split.TARGET, "median"))
        .reset_index()
    )
    assignment = split.assign_groups(groups, seed=seed)
    return frame.assign(split=frame["group_id"].map(assignment))


def test_whole_groups_never_split():
    assigned = _assign(_synthetic_frame())
    spanning = assigned.groupby("group_id")["split"].nunique()
    assert (spanning == 1).all(), "a group was divided across partitions"


def test_every_molecule_assigned_to_a_known_partition():
    assigned = _assign(_synthetic_frame())
    assert assigned["split"].notna().all()
    assert set(assigned["split"]) <= set(split.PARTITIONS)


def test_realized_proportions_are_close_to_nominal():
    assigned = _assign(_synthetic_frame(n_groups=200, per_group=4))
    shares = assigned["split"].value_counts(normalize=True)
    for name, nominal in split.PROPORTIONS.items():
        assert abs(shares[name] - nominal) < 0.03, (
            f"{name} share {shares[name]:.3f} is far from nominal {nominal}"
        )


def test_assignment_is_deterministic_for_a_fixed_seed():
    frame = _synthetic_frame()
    first = _assign(frame)["split"].tolist()
    second = _assign(frame)["split"].tolist()
    assert first == second


def test_stratification_spreads_the_target_across_partitions():
    """Every partition should see low- and high-pEC50 groups, not one end of the range."""
    assigned = _assign(_synthetic_frame(n_groups=200, per_group=4))
    medians = assigned.groupby("split")[split.TARGET].median()
    assert medians.max() - medians.min() < 1.0, (
        f"partitions have skewed target distributions: {medians.to_dict()}"
    )


def test_check_invariants_rejects_a_scaffold_spanning_partitions():
    frame = pd.DataFrame(
        {
            split.ID_COLUMN: ["A", "B"],
            "scaffold_smiles": ["c1ccccc1", "c1ccccc1"],
            "connectivity_block": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
            "group_id": [0, 1],
            "split": ["train", "holdout"],
        }
    )
    with pytest.raises(AssertionError, match="scaffold spans"):
        split.check_invariants(frame)


def test_check_invariants_rejects_a_near_duplicate_pair_separated():
    frame = pd.DataFrame(
        {
            split.ID_COLUMN: ["A", "B"],
            "scaffold_smiles": ["c1ccccc1", "c1ccncc1"],
            "connectivity_block": ["AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA"],
            "group_id": [0, 1],
            "split": ["train", "holdout"],
        }
    )
    with pytest.raises(AssertionError, match="connectivity block spans"):
        split.check_invariants(frame)


def test_check_invariants_rejects_duplicate_identity():
    frame = pd.DataFrame(
        {
            split.ID_COLUMN: ["A", "A"],
            "scaffold_smiles": ["c1ccccc1", "c1ccccc1"],
            "connectivity_block": ["AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA"],
            "group_id": [0, 0],
            "split": ["train", "train"],
        }
    )
    with pytest.raises(AssertionError, match="duplicate molecule identity"):
        split.check_invariants(frame)


def test_check_invariants_rejects_unassigned_molecules():
    frame = pd.DataFrame(
        {
            split.ID_COLUMN: ["A"],
            "scaffold_smiles": ["c1ccccc1"],
            "connectivity_block": ["AAAAAAAAAAAAAA"],
            "group_id": [0],
            "split": [None],
        }
    )
    with pytest.raises(AssertionError, match="not assigned"):
        split.check_invariants(frame)


def test_scaffold_and_connectivity_rejects_unparseable_smiles():
    with pytest.raises(ValueError, match="Unable to parse"):
        split.scaffold_and_connectivity(["not_a_molecule"])


def test_scaffold_is_chirality_free():
    """Enantiomers must share a scaffold, so they cannot be split apart."""
    scaffolds, _ = split.scaffold_and_connectivity(
        ["C[C@H]1CCCCN1c1ccccc1", "C[C@@H]1CCCCN1c1ccccc1"]
    )
    assert scaffolds[0] == scaffolds[1]
