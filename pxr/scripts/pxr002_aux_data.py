"""Tripwire PXR-002 auxiliary-label eligibility (D017 Amendment 1, D018).

Builds the frozen training/validation frames for the multitask Chemprop recipe:

  tasks         pEC50 (primary) + log2 fold change at 8.25, 33 and 99 uM (auxiliary)
  source        single_concentration_TRAIN.csv, compound_class == "Library" only
                (positive controls are excluded; counter-assay, crude and semi-pure
                data are not used in this recipe)
  eligibility   auxiliary labels on primary validation / holdout molecules never enter
                training. An auxiliary-only molecule (no primary label) is used only if
                its D014 group -- Bemis-Murcko scaffold (chirality excluded), merged
                through InChIKey connectivity blocks, each ringless molecule its own
                group -- contains no validation, holdout or official-test molecule.
  no new rule   no similarity cutoff is introduced; grouping reuses pxr001_split.

The 99 uM readout comes only from experiment OPA_PXR-030, so concentration and
experiment effects cannot be disentangled for that head.

Training frame: every primary-train molecule (released SMILES) plus every eligible
auxiliary-only molecule (its single-concentration SMILES). Validation frame: the 621
primary validation molecules with pEC50 only. The holdout and official-test
molecules appear in neither frame; only their structures are used for exclusion.

Usage (repository root; use the Chemprop environment whose RDKit supports scaffolds):
    python -I pxr/scripts/pxr002_aux_data.py \
        pxr/data/openadmet_pxr_release_2026-11-15 \
        pxr/artifacts/pxr001_split/pxr001_split_assignments.csv \
        pxr/artifacts/pxr002_aux
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

from pxr.scripts.pxr001_split import merge_groups, scaffold_and_connectivity  # noqa: E402

ID_COLUMN = "Molecule Name"
OCNT_COLUMN = "OCNT_ID"
SMILES_COLUMN = "SMILES"
PRIMARY_TASK = "pEC50"

AUX_CONCENTRATION_TO_HEAD = {
    8.251e-06: "log2fc_8p25uM",
    3.3e-05: "log2fc_33uM",
    9.901e-05: "log2fc_99uM",
}
AUX_TASKS = tuple(AUX_CONCENTRATION_TO_HEAD.values())
TASK_COLUMNS = (PRIMARY_TASK, *AUX_TASKS)

LIBRARY_CLASS = "Library"
AUX_VALUE_COLUMN = "log2_fc_estimate"

FRAME_COLUMNS = ["row_id", "role", SMILES_COLUMN, *TASK_COLUMNS]
GROUP_TRAIN_ONLY = {"train"}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_groups(primary: pd.DataFrame, assignments: pd.DataFrame) -> None:
    """Require the working RDKit to reproduce the frozen D014 scaffold/block columns."""
    needed = {"scaffold_smiles", "connectivity_block", ID_COLUMN}
    if needed - set(assignments.columns):
        raise ValueError(f"assignments missing columns: {sorted(needed - set(assignments.columns))}")
    merged = primary[[ID_COLUMN, SMILES_COLUMN]].merge(
        assignments[[ID_COLUMN, "scaffold_smiles", "connectivity_block"]],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(primary):
        raise ValueError("primary table and assignments do not cover the same molecules")
    scaffolds, blocks = scaffold_and_connectivity(merged[SMILES_COLUMN])
    frozen_scaffold = merged["scaffold_smiles"].fillna("").tolist()
    frozen_block = merged["connectivity_block"].tolist()
    bad_scaffold = sum(a != b for a, b in zip(scaffolds, frozen_scaffold))
    bad_block = sum(a != b for a, b in zip(blocks, frozen_block))
    if bad_scaffold or bad_block:
        raise RuntimeError(
            "RDKit does not reproduce the frozen D014 groups: "
            f"{bad_scaffold} scaffold and {bad_block} connectivity-block mismatches. "
            "Use the environment that generated the split."
        )


def _library_wide(single: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Library rows -> one row per OCNT_ID with three log2FC columns, plus SMILES."""
    needed = {OCNT_COLUMN, SMILES_COLUMN, "compound_class", "concentration_M", AUX_VALUE_COLUMN}
    if needed - set(single.columns):
        raise ValueError(f"single-concentration table missing: {sorted(needed - set(single.columns))}")
    library = single.loc[single["compound_class"].eq(LIBRARY_CLASS)].copy()

    known = np.array(list(AUX_CONCENTRATION_TO_HEAD))
    heads = []
    for value in library["concentration_M"].to_numpy(dtype=float):
        hit = np.isclose(known, value, rtol=1e-3, atol=0.0)
        if hit.sum() != 1:
            raise ValueError(f"unexpected library concentration {value!r}")
        heads.append(AUX_CONCENTRATION_TO_HEAD[float(known[hit][0])])
    library["head"] = heads

    if library[AUX_VALUE_COLUMN].isna().any() or not np.isfinite(library[AUX_VALUE_COLUMN]).all():
        raise ValueError("library log2 fold-change values contain non-finite entries")
    if library.duplicated([OCNT_COLUMN, "head"]).any():
        raise ValueError("duplicate (OCNT_ID, concentration) library rows; averaging is not approved")
    if library.groupby(OCNT_COLUMN)[SMILES_COLUMN].nunique().gt(1).any():
        raise ValueError("an OCNT_ID carries more than one SMILES in the single-concentration table")

    wide = library.pivot(index=OCNT_COLUMN, columns="head", values=AUX_VALUE_COLUMN)
    wide = wide.reindex(columns=list(AUX_TASKS))
    smiles = library.drop_duplicates(OCNT_COLUMN).set_index(OCNT_COLUMN)[SMILES_COLUMN]
    return wide, smiles.reindex(wide.index)


def build_eligibility(
    primary: pd.DataFrame,
    assignments: pd.DataFrame,
    test: pd.DataFrame,
    single: pd.DataFrame,
) -> dict:
    """Return the eligibility table, wide auxiliary labels and auxiliary-only SMILES.

    ``primary`` needs Molecule Name, OCNT_ID, SMILES, pEC50; ``assignments`` needs
    Molecule Name and split; ``test`` needs SMILES (blinded structures only).
    """
    if primary[OCNT_COLUMN].isna().any() or primary[OCNT_COLUMN].duplicated().any():
        raise ValueError("primary OCNT_ID must be present and unique")
    part = primary[[ID_COLUMN, OCNT_COLUMN, SMILES_COLUMN]].merge(
        assignments[[ID_COLUMN, "split"]], on=ID_COLUMN, how="inner", validate="one_to_one"
    )
    if len(part) != len(primary):
        raise ValueError("primary table and assignments do not cover the same molecules")

    wide, aux_smiles = _library_wide(single)
    primary_ocnt = set(part[OCNT_COLUMN])
    aux_only_ids = [oc for oc in wide.index if oc not in primary_ocnt]

    # One universe: primary (all partitions) + blinded test + auxiliary-only, grouped
    # with the D014 rule so a bridge through any of them is seen.
    universe_smiles = (
        part[SMILES_COLUMN].tolist()
        + test[SMILES_COLUMN].tolist()
        + aux_smiles.loc[aux_only_ids].tolist()
    )
    n_primary, n_test = len(part), len(test)
    scaffolds, blocks = scaffold_and_connectivity(universe_smiles)
    group_ids = np.array(merge_groups(scaffolds, blocks))

    group_partitions: dict[int, set[str]] = {}
    group_has_test: dict[int, bool] = {}
    for gid, split in zip(group_ids[:n_primary], part["split"]):
        group_partitions.setdefault(int(gid), set()).add(split)
    for gid in group_ids[n_primary : n_primary + n_test]:
        group_has_test[int(gid)] = True

    rows = []
    for gid, oc, split in zip(group_ids[:n_primary], part[OCNT_COLUMN], part["split"]):
        has_aux = oc in wide.index
        if split == "train":
            reason = "primary_train"
            eligible = True
        else:
            reason = f"primary_{split}_aux_excluded" if has_aux else f"primary_{split}"
            eligible = False
        rows.append((oc, f"primary_{split}", eligible, reason, int(gid), has_aux))
    for gid, oc in zip(group_ids[n_primary + n_test :], aux_only_ids):
        gid = int(gid)
        partitions = group_partitions.get(gid, set())
        touches_test = group_has_test.get(gid, False)
        if touches_test:
            eligible, reason = False, "aux_only_group_contains_test_structure"
        elif partitions - GROUP_TRAIN_ONLY:
            eligible, reason = False, "aux_only_group_contains_validation_or_holdout"
        else:
            eligible, reason = True, "aux_only_eligible"
        rows.append((oc, "aux_only", eligible, reason, gid, True))

    table = pd.DataFrame(
        rows, columns=[OCNT_COLUMN, "role", "eligible", "reason", "group_id", "has_aux_labels"]
    )
    return {"table": table, "wide": wide, "aux_smiles": aux_smiles, "primary_partition": part}


def assemble_frames(primary: pd.DataFrame, assignments: pd.DataFrame, elig: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and validation frames. Holdout molecules appear in neither."""
    table, wide, aux_smiles = elig["table"], elig["wide"], elig["aux_smiles"]
    part = elig["primary_partition"].merge(
        primary[[ID_COLUMN, PRIMARY_TASK]], on=ID_COLUMN, how="left", validate="one_to_one"
    )

    train_primary = part.loc[part["split"].eq("train")]
    train_rows = pd.DataFrame(
        {
            "row_id": train_primary[ID_COLUMN].to_numpy(),
            "role": "primary_train",
            SMILES_COLUMN: train_primary[SMILES_COLUMN].to_numpy(),
            PRIMARY_TASK: train_primary[PRIMARY_TASK].to_numpy(dtype=float),
        }
    )
    aux_for_train = wide.reindex(train_primary[OCNT_COLUMN])
    for task in AUX_TASKS:
        train_rows[task] = aux_for_train[task].to_numpy(dtype=float)

    eligible_aux = table.loc[table["role"].eq("aux_only") & table["eligible"], OCNT_COLUMN]
    aux_rows = pd.DataFrame(
        {
            "row_id": eligible_aux.to_numpy(),
            "role": "aux_only",
            SMILES_COLUMN: aux_smiles.loc[eligible_aux].to_numpy(),
            PRIMARY_TASK: np.nan,
        }
    )
    for task in AUX_TASKS:
        aux_rows[task] = wide.loc[eligible_aux, task].to_numpy(dtype=float)

    train = pd.concat([train_rows, aux_rows], ignore_index=True)[FRAME_COLUMNS]

    val_primary = part.loc[part["split"].eq("validation")]
    validation = pd.DataFrame(
        {
            "row_id": val_primary[ID_COLUMN].to_numpy(),
            "role": "primary_validation",
            SMILES_COLUMN: val_primary[SMILES_COLUMN].to_numpy(),
            PRIMARY_TASK: val_primary[PRIMARY_TASK].to_numpy(dtype=float),
        }
    )
    for task in AUX_TASKS:
        validation[task] = np.nan
    validation = validation[FRAME_COLUMNS]

    check_frames(train, validation, part)
    return train, validation


def check_frames(train: pd.DataFrame, validation: pd.DataFrame, part: pd.DataFrame) -> None:
    holdout_ids = set(part.loc[part["split"].eq("holdout"), ID_COLUMN])
    holdout_ocnt = set(part.loc[part["split"].eq("holdout"), OCNT_COLUMN])
    val_ids = set(part.loc[part["split"].eq("validation"), ID_COLUMN])
    val_ocnt = set(part.loc[part["split"].eq("validation"), OCNT_COLUMN])
    train_ids = set(train["row_id"])
    if train_ids & (holdout_ids | holdout_ocnt | val_ids | val_ocnt):
        raise AssertionError("training frame contains validation or holdout molecules")
    if set(validation["row_id"]) & (holdout_ids | train_ids):
        raise AssertionError("validation frame overlaps holdout or training molecules")
    if train["row_id"].duplicated().any() or validation["row_id"].duplicated().any():
        raise AssertionError("duplicate row ids in a frame")
    if train[PRIMARY_TASK].notna().sum() != int(part["split"].eq("train").sum()):
        raise AssertionError("training frame lost primary labels")
    if validation[PRIMARY_TASK].isna().any():
        raise AssertionError("validation frame has missing pEC50")
    if train[list(TASK_COLUMNS)].isna().all(axis=1).any():
        raise AssertionError("training frame has a molecule with no label at all")


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {task: int(frame[task].notna().sum()) for task in TASK_COLUMNS}


def main(data_dir: str, assignments_path: str, out_dir: str) -> None:
    data_dir = Path(data_dir)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refusing to overwrite a frozen artifact")
    out.mkdir(parents=True, exist_ok=True)

    primary_path = data_dir / "pxr-challenge_TRAIN.csv"
    single_path = data_dir / "pxr-challenge_single_concentration_TRAIN.csv"
    test_path = data_dir / "pxr-challenge_TEST_BLINDED.csv"
    primary = pd.read_csv(primary_path)
    assignments = pd.read_csv(assignments_path)
    test = pd.read_csv(test_path)
    single = pd.read_csv(single_path)

    verify_frozen_groups(primary, assignments)
    elig = build_eligibility(primary, assignments, test, single)
    train, validation = assemble_frames(primary, assignments, elig)

    elig["table"].to_csv(out / "pxr002_eligibility.csv", index=False)
    train.to_csv(out / "pxr002_train_frame.csv", index=False)
    validation.to_csv(out / "pxr002_validation_frame.csv", index=False)

    table = elig["table"]
    summary = {
        "train_molecules": int(len(train)),
        "validation_molecules": int(len(validation)),
        "train_label_counts": label_counts(train),
        "aux_only_eligible_molecules": int((table["role"].eq("aux_only") & table["eligible"]).sum()),
        "aux_only_excluded_by_reason": table.loc[table["role"].eq("aux_only") & ~table["eligible"], "reason"]
        .value_counts()
        .to_dict(),
        "library_molecules_total": int(len(elig["wide"])),
    }
    (out / "pxr002_counts.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    inputs = {
        "TRAIN": primary_path,
        "single_concentration": single_path,
        "TEST_BLINDED": test_path,
        "assignments": Path(assignments_path),
    }
    names = ["pxr002_eligibility.csv", "pxr002_train_frame.csv", "pxr002_validation_frame.csv", "pxr002_counts.json"]
    sums = "".join(f"{sha256(out / n)}  {n}\n" for n in names)
    (out / "SHA256SUMS").write_text(sums)
    prov = ["PXR-002 auxiliary eligibility artifact (D017 Amendment 1, D018)", "Input SHA-256:"]
    prov += [f"  {k}: {sha256(p)}  {p}" for k, p in inputs.items()]
    (out / "PROVENANCE.txt").write_text("\n".join(prov) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
