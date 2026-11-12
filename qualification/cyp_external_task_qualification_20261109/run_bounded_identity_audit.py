from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import inchi


ROOT = Path("/Users/mileshuang/Desktop/tripwire-project")
QUAL = ROOT / "qualification/cyp_external_task_qualification_20261109"
OUT = QUAL / "identity_audit"

SPLIT = ROOT / "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv"
TRAIN = ROOT / "cyp/data/cyp-challenge-TRAIN_inhibition.csv"
BLIND = ROOT / "cyp/data/cyp-challenge-TEST-BLINDED.csv"

TASKS = QUAL / "derived/p2_compatible_ic50_task_provenance.tsv"
RECORD_ELIGIBILITY = QUAL / "derived/record_eligibility.tsv"

ACTIVITIES = {
    "CYP1A2": QUAL / "raw_bulk/chembl37_CYP1A2_activities.jsonl",
    "CYP2C9": QUAL / "raw_bulk/chembl37_CYP2C9_activities.jsonl",
    "CYP2D6": QUAL / "raw_bulk/chembl37_CYP2D6_activities.jsonl",
    "CYP3A4": QUAL / "raw_bulk/chembl37_CYP3A4_activities.jsonl",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inchi_runtime_version() -> dict[str, object]:
    return {
        "inchi_available": bool(inchi.INCHI_AVAILABLE),
        "inchi_version": inchi.GetInchiVersion(),
    }


def represent(source_smiles: str) -> dict[str, str]:
    result = {
        "identity_state": "",
        "failure_reason": "",
        "canonical_isomeric_smiles_witness": "",
        "standard_inchi": "",
        "full_standard_inchikey": "",
    }

    if source_smiles == "":
        result["identity_state"] = "source_structure_unavailable"
        result["failure_reason"] = "source SMILES is empty"
        return result

    mol = Chem.MolFromSmiles(source_smiles)

    if mol is None:
        result["identity_state"] = "parse_failed"
        result["failure_reason"] = "Chem.MolFromSmiles returned None"
        return result

    result["canonical_isomeric_smiles_witness"] = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )

    try:
        std_inchi = inchi.MolToInchi(mol)
    except Exception as exc:
        result["identity_state"] = "inchi_failed"
        result["failure_reason"] = repr(exc)
        return result

    if not std_inchi:
        result["identity_state"] = "inchi_failed"
        result["failure_reason"] = "MolToInchi returned empty string"
        return result

    result["standard_inchi"] = std_inchi

    try:
        key = inchi.InchiToInchiKey(std_inchi)
    except Exception as exc:
        result["identity_state"] = "inchikey_failed"
        result["failure_reason"] = repr(exc)
        return result

    if not key or len(key) != 27:
        result["identity_state"] = "inchikey_failed"
        result["failure_reason"] = (
            f"unexpected full Standard InChIKey {key!r}"
        )
        return result

    result["full_standard_inchikey"] = key
    result["identity_state"] = "represented"
    return result


def read_openadmet_universe() -> list[dict[str, str]]:
    """
    Construct the complete bounded OpenADMET structure universe.

    Frozen CYP-002 universe:
      Molecule_Name -> partition from molecule_split_assignments.csv
      structure -> cyp-challenge-TRAIN_inhibition.csv

    Original blinded challenge:
      independent 750-molecule structure set from
      cyp-challenge-TEST-BLINDED.csv

    The challenge is not the frozen CYP-002 internal-test partition.
    No unblinded challenge artifact is read.
    """
    with SPLIT.open(newline="") as handle:
        split_rows = list(csv.DictReader(handle))

    if len(split_rows) != 4905:
        raise RuntimeError(
            f"expected 4905 frozen split rows, got {len(split_rows)}"
        )

    split_by_name = {
        row["Molecule_Name"]: row["split"]
        for row in split_rows
    }

    if len(split_by_name) != len(split_rows):
        raise RuntimeError(
            "duplicate Molecule_Name in frozen split assignments"
        )

    split_counts = Counter(
        row["split"]
        for row in split_rows
    )

    expected_split_counts = Counter({
        "train": 3433,
        "validation": 736,
        "test": 736,
    })

    if split_counts != expected_split_counts:
        raise RuntimeError(
            f"unexpected frozen split counts: "
            f"{dict(split_counts)} != "
            f"{dict(expected_split_counts)}"
        )

    with TRAIN.open(newline="") as handle:
        train_rows = list(csv.DictReader(handle))

    train_names = [
        row["Molecule_Name"]
        for row in train_rows
    ]

    if len(train_names) != 4905:
        raise RuntimeError(
            f"expected 4905 frozen TRAIN structures, "
            f"got {len(train_names)}"
        )

    if len(train_names) != len(set(train_names)):
        raise RuntimeError(
            "duplicate Molecule_Name in frozen TRAIN dataset"
        )

    if set(train_names) != set(split_by_name):
        raise RuntimeError(
            "frozen TRAIN Molecule_Name set does not equal "
            "frozen split-assignment Molecule_Name set"
        )

    records = []

    # Complete frozen CYP-002 molecular universe.
    for row in train_rows:
        name = row["Molecule_Name"]
        split = split_by_name[name]

        # Rename the split's historical "test" label for the audit output
        # so it cannot be confused with the independent blinded challenge.
        audit_partition = (
            "internal_test"
            if split == "test"
            else split
        )

        records.append({
            "source_family": "openadmet",
            "source_partition": audit_partition,
            "source_id": name,
            "isoform": "",
            "assay_chembl_id": "",
            "molecule_chembl_id": "",
            "source_smiles": row["SMILES"],
        })

    # Independent original blinded challenge structure universe.
    with BLIND.open(newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != ["Molecule_Name", "SMILES"]:
            raise RuntimeError(
                "original blinded challenge schema must be exactly "
                "Molecule_Name,SMILES"
            )

        blind_rows = list(reader)

    if len(blind_rows) != 750:
        raise RuntimeError(
            f"expected 750 original blinded challenge structures, "
            f"got {len(blind_rows)}"
        )

    blind_names = [
        row["Molecule_Name"]
        for row in blind_rows
    ]

    if len(blind_names) != len(set(blind_names)):
        raise RuntimeError(
            "duplicate Molecule_Name in original blinded challenge"
        )

    if set(blind_names) & set(split_by_name):
        raise RuntimeError(
            "original blinded challenge Molecule_Name unexpectedly "
            "overlaps the frozen CYP-002 molecular universe"
        )

    for row in blind_rows:
        records.append({
            "source_family": "openadmet",
            "source_partition": "challenge_blinded_test",
            "source_id": row["Molecule_Name"],
            "isoform": "",
            "assay_chembl_id": "",
            "molecule_chembl_id": "",
            "source_smiles": row["SMILES"],
        })

    if len(records) != 5655:
        raise RuntimeError(
            f"expected 5655 total OpenADMET audit records, "
            f"got {len(records)}"
        )

    return records


def selected_task_ids() -> dict[str, set[str]]:
    with TASKS.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    if len(rows) != 176:
        raise RuntimeError(
            f"expected 176 pre-consolidation candidate tasks, got {len(rows)}"
        )

    selected = defaultdict(set)

    for row in rows:
        selected[row["isoform"]].add(row["assay_chembl_id"])

    observed = {iso: len(ids) for iso, ids in selected.items()}
    expected = {
        "CYP1A2": 22,
        "CYP2C9": 33,
        "CYP2D6": 40,
        "CYP3A4": 81,
    }

    if observed != expected:
        raise RuntimeError(
            f"selected task counts differ: {observed} != {expected}"
        )

    return dict(selected)


def read_chembl_universe(
    selected: dict[str, set[str]],
) -> list[dict[str, str]]:
    records = []
    selected_keys = {
        (isoform, assay_chembl_id)
        for isoform, assay_ids in selected.items()
        for assay_chembl_id in assay_ids
    }
    selected_activity_ids = set()

    with RECORD_ELIGIBILITY.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(
            csv.DictReader(handle, delimiter="\t"),
            2,
        ):
            key = (row["isoform"], row["assay_chembl_id"])
            if key not in selected_keys:
                continue

            if row["normalized_endpoint_semantics"] != "potency_candidate:IC50":
                continue

            if row["primary_class"] != "exact_quantitative_potency_candidate":
                continue

            selected_activity_ids.add(row["activity_id"])

            # Preserve every selected frozen eligibility row and its original
            # canonical_smiles spelling; no molecule-level deduplication.
            records.append({
                "source_family": "chembl",
                "source_partition": "external",
                "source_id": row["activity_id"] or (
                    f"{RECORD_ELIGIBILITY.name}:{line_number}"
                ),
                "isoform": row["isoform"],
                "assay_chembl_id": row["assay_chembl_id"],
                "molecule_chembl_id": row["molecule_chembl_id"],
                "source_smiles": row["canonical_smiles"],
            })

    if not records:
        raise RuntimeError("selected ChEMBL audit universe is empty")

    if "7614549" in selected_activity_ids:
        raise RuntimeError(
            "activity_id 7614549 must not be selected for the audit"
        )

    rows_7603640 = [
        row for row in records
        if row["source_id"] == "7603640"
    ]

    if len(rows_7603640) != 1:
        raise RuntimeError(
            "activity_id 7603640 must be selected exactly once"
        )

    expected_smiles_7603640 = "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"
    if rows_7603640[0]["source_smiles"] != expected_smiles_7603640:
        raise RuntimeError(
            "activity_id 7603640 does not retain its frozen "
            "canonical_smiles"
        )

    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    if not inchi.INCHI_AVAILABLE:
        raise RuntimeError("RDKit InChI support is unavailable")

    selected = selected_task_ids()

    openadmet = read_openadmet_universe()
    chembl = read_chembl_universe(selected)

    all_records = openadmet + chembl

    represented = []

    for row in all_records:
        result = dict(row)
        result.update(represent(row["source_smiles"]))
        represented.append(result)

    fields = [
        "source_family",
        "source_partition",
        "source_id",
        "isoform",
        "assay_chembl_id",
        "molecule_chembl_id",
        "source_smiles",
        "identity_state",
        "failure_reason",
        "canonical_isomeric_smiles_witness",
        "standard_inchi",
        "full_standard_inchikey",
    ]

    rep_path = OUT / "bounded_representation_audit.tsv"

    with rep_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(represented)

    # Protected OpenADMET representation completeness.
    openadmet_failures = [
        row for row in represented
        if (
            row["source_family"] == "openadmet"
            and row["identity_state"] != "represented"
        )
    ]

    # Same-key groups and full-InChI collision/inconsistency check.
    key_groups = defaultdict(list)

    for row in represented:
        if row["identity_state"] == "represented":
            key_groups[row["full_standard_inchikey"]].append(row)

    inconsistent_keys = {}

    for key, rows in key_groups.items():
        inchis = sorted({
            row["standard_inchi"]
            for row in rows
        })
        if len(inchis) > 1:
            inconsistent_keys[key] = inchis

    # Cross-source overlap by frozen OpenADMET partition.
    open_by_key = defaultdict(list)
    chembl_by_key = defaultdict(list)

    for row in represented:
        if row["identity_state"] != "represented":
            continue

        if row["source_family"] == "openadmet":
            open_by_key[row["full_standard_inchikey"]].append(row)
        else:
            chembl_by_key[row["full_standard_inchikey"]].append(row)

    cross_matches = []

    for key in sorted(set(open_by_key) & set(chembl_by_key)):
        open_rows = open_by_key[key]
        ext_rows = chembl_by_key[key]

        literal = any(
            a["source_smiles"] == b["source_smiles"]
            for a in open_rows
            for b in ext_rows
        )

        partitions = sorted({
            row["source_partition"]
            for row in open_rows
        })

        cross_matches.append({
            "full_standard_inchikey": key,
            "identity_key_status": (
                "collision_or_inconsistency"
                if key in inconsistent_keys
                else "same_full_key_same_standard_inchi"
            ),
            "standard_inchi": open_rows[0]["standard_inchi"],
            "openadmet_partitions": ";".join(partitions),
            "literal_smiles_cross_source_match": str(literal).lower(),
            "openadmet_molecule_names": ";".join(sorted({
                row["source_id"] for row in open_rows
            })),
            "chembl_molecule_ids": ";".join(sorted({
                row["molecule_chembl_id"]
                for row in ext_rows
                if row["molecule_chembl_id"]
            })),
            "chembl_assays": ";".join(sorted({
                row["assay_chembl_id"]
                for row in ext_rows
            })),
            "openadmet_source_spellings": json.dumps(
                sorted({row["source_smiles"] for row in open_rows})
            ),
            "chembl_source_spellings": json.dumps(
                sorted({row["source_smiles"] for row in ext_rows})
            ),
        })

    match_path = OUT / "cross_source_matches.tsv"

    match_fields = [
        "full_standard_inchikey",
        "identity_key_status",
        "standard_inchi",
        "openadmet_partitions",
        "literal_smiles_cross_source_match",
        "openadmet_molecule_names",
        "chembl_molecule_ids",
        "chembl_assays",
        "openadmet_source_spellings",
        "chembl_source_spellings",
    ]

    with match_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=match_fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(cross_matches)

    extra_matches = [
        row for row in cross_matches
        if row["literal_smiles_cross_source_match"] == "false"
    ]

    # Declared bounded inspection selection:
    # all <=25; otherwise deterministic evenly spaced 25 including ends.
    if len(extra_matches) <= 25:
        inspection = list(extra_matches)
    else:
        n = len(extra_matches)
        indices = [
            round(i * (n - 1) / 24)
            for i in range(25)
        ]
        inspection = [extra_matches[i] for i in indices]

    inspection_path = OUT / "bounded_extra_match_inspection.tsv"

    with inspection_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=match_fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(inspection)

    state_counts = Counter(
        (
            row["source_family"],
            row["source_partition"],
            row["identity_state"],
        )
        for row in represented
    )

    partition_match_counts = Counter()

    for row in cross_matches:
        for partition in row["openadmet_partitions"].split(";"):
            if partition:
                partition_match_counts[partition] += 1

    partition_match_counts_report = {
        partition: partition_match_counts[partition]
        for partition in (
            "train",
            "validation",
            "internal_test",
            "challenge_blinded_test",
        )
    }

    summary = {
        "artifact_type":
            "CYP_bounded_cross_source_identity_representation_audit_v1",
        "authority": "audit_only_no_routing_or_eligibility_authority",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "rdkit_version": rdBase.rdkitVersion,
            "inchi": inchi_runtime_version(),
        },
        "representation_contract_under_audit": {
            "source_smiles_preserved_verbatim": True,
            "project_added_standardization": False,
            "standard_inchi_intrinsic_normalization_still_applies": True,
            "canonical_isomeric_smiles_is_witness_only": True,
            "standard_inchi_is_retained_witness": True,
            "proposed_authority": "full_27_character_standard_inchikey",
            "contract_frozen": False,
        },
        "input_universe": {
            "candidate_tasks_pre_consolidation": 176,
            "candidate_task_counts_by_isoform": {
                iso: len(ids)
                for iso, ids in sorted(selected.items())
            },
            "openadmet_source_records": len(openadmet),
            "chembl_source_records": len(chembl),
            "all_original_source_spellings_retained": True,
            "blinded_test_structures_only": True,
        },
        "representation_state_counts": {
            "|".join(key): value
            for key, value in sorted(state_counts.items())
        },
        "protected_openadmet_representation_failures":
            len(openadmet_failures),
        "same_key_different_standard_inchi_count":
            len(inconsistent_keys),
        "same_key_different_standard_inchi": inconsistent_keys,
        "cross_source_full_key_match_groups":
            len(cross_matches),
        "cross_source_match_groups_by_openadmet_partition":
            partition_match_counts_report,
        "literal_smiles_cross_source_match_groups":
            sum(
                row["literal_smiles_cross_source_match"] == "true"
                for row in cross_matches
            ),
        "extra_full_key_match_groups_beyond_literal_smiles":
            len(extra_matches),
        "bounded_extra_match_inspection_count":
            len(inspection),
        "bounded_inspection_rule": (
            "inspect all extra full-key cross-source match groups if <=25; "
            "otherwise sort by full Standard InChIKey and inspect 25 "
            "evenly spaced groups including first and last"
        ),
        "important_claim_boundaries": [
            "training overlap is reported separately from validation/test overlap",
            "no overlap causes routing or exclusion in this audit",
            "representation failures remain explicit and unresolved",
            "same-key groups are checked against retained full Standard InChI",
            "inspection examples inform the identity decision but do not prove all Standard InChI equivalences appropriate",
            "no replicate grouping or measurement aggregation is performed",
            "no task or partition is modified",
        ],
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                SPLIT,
                TRAIN,
                BLIND,
                TASKS,
                RECORD_ELIGIBILITY,
                *ACTIVITIES.values(),
            ]
        },
    }

    summary_path = OUT / "bounded_identity_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    hashes = OUT / "SHA256SUMS.txt"

    output_files = [
        rep_path,
        match_path,
        inspection_path,
        summary_path,
    ]

    with hashes.open("w", encoding="utf-8") as handle:
        for path in output_files:
            handle.write(
                f"{sha256_file(path)}  {path.name}\n"
            )

    print("=== BOUNDED IDENTITY AUDIT ===")
    print("OpenADMET source records:", len(openadmet))
    print("ChEMBL source records:", len(chembl))
    print(
        "protected OpenADMET representation failures:",
        len(openadmet_failures),
    )
    print(
        "same-key/different-InChI groups:",
        len(inconsistent_keys),
    )
    print(
        "cross-source full-key match groups:",
        len(cross_matches),
    )
    print(
        "cross-source match groups by partition:",
        partition_match_counts_report,
    )
    print(
        "literal-SMILES match groups:",
        summary["literal_smiles_cross_source_match_groups"],
    )
    print(
        "extra full-key groups:",
        len(extra_matches),
    )
    print(
        "bounded inspection rows:",
        len(inspection),
    )
    print("AUDIT COMPLETE — NO ROUTING PERFORMED")


if __name__ == "__main__":
    main()
