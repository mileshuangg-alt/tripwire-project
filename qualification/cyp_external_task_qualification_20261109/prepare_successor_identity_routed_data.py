from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import rdBase
from rdkit.Chem import inchi


ROOT = Path("/Users/mileshuang/Desktop/tripwire-project")
QUAL = ROOT / "qualification/cyp_external_task_qualification_20261109"
AUDIT = QUAL / "identity_audit"
OUT = QUAL / "successor_identity_routed_preparation"

REPRESENTATION_AUDIT = AUDIT / "bounded_representation_audit.tsv"
TASKS = QUAL / "derived/p2_compatible_ic50_task_provenance.tsv"
RECORD_ELIGIBILITY = QUAL / "derived/record_eligibility.tsv"
RELATIONSHIPS = QUAL / "derived/p2_compatible_ic50_relationships.tsv"

PROTECTED_PARTITIONS = (
    "validation",
    "internal_test",
    "challenge_blinded_test",
)
ALL_PARTITIONS = (
    "train",
    "validation",
    "internal_test",
    "challenge_blinded_test",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: float) -> str:
    return f"{value:.12g}"


def ic50_to_molar(value: str, unit: str) -> float:
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"IC50 must be positive, got {value!r}")

    factors = {
        "M": 1.0,
        "mM": 1e-3,
        "uM": 1e-6,
        "µM": 1e-6,
        "nM": 1e-9,
        "pM": 1e-12,
    }
    if unit not in factors:
        raise ValueError(f"unsupported qualified concentration unit {unit!r}")
    return numeric * factors[unit]


def pic50_from_ic50(value: str, unit: str) -> float:
    return -math.log10(ic50_to_molar(value, unit))


def join_sorted(values: set[str] | list[str]) -> str:
    return ";".join(sorted(values))


def load_tasks() -> list[dict[str, str]]:
    tasks = read_tsv(TASKS)
    if len(tasks) != 176:
        raise RuntimeError(f"expected 176 frozen candidate tasks, got {len(tasks)}")
    counts = Counter(row["isoform"] for row in tasks)
    expected = Counter({"CYP1A2": 22, "CYP2C9": 33, "CYP2D6": 40, "CYP3A4": 81})
    if counts != expected:
        raise RuntimeError(f"unexpected task counts: {dict(counts)} != {dict(expected)}")
    return tasks


def selected_task_keys(tasks: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["isoform"], row["assay_chembl_id"]) for row in tasks}


def load_relationships(
    task_keys: set[tuple[str, str]],
) -> tuple[set[tuple[str, str]], list[dict[str, str]]]:
    rows = read_tsv(RELATIONSHIPS)
    duplicate_pairs = [
        row for row in rows
        if row["relationship_category"] == "demonstrated_duplicate"
    ]
    if len(duplicate_pairs) != 1:
        raise RuntimeError(
            f"expected one demonstrated duplicate relationship, got {len(duplicate_pairs)}"
        )

    participating = set()
    for row in duplicate_pairs:
        left = (row["left_isoform"], row["left_assay_chembl_id"])
        right = (row["right_isoform"], row["right_assay_chembl_id"])
        if left not in task_keys or right not in task_keys:
            raise RuntimeError("demonstrated duplicate pair is outside selected tasks")
        participating.update([left, right])

    return participating, duplicate_pairs


def load_record_values(
    task_keys: set[tuple[str, str]],
) -> dict[str, dict[str, str]]:
    values = {}
    with RECORD_ELIGIBILITY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["isoform"], row["assay_chembl_id"])
            if key not in task_keys:
                continue
            if row["normalized_endpoint_semantics"] != "potency_candidate:IC50":
                continue
            if row["primary_class"] != "exact_quantitative_potency_candidate":
                continue
            values[row["activity_id"]] = row
    return values


def load_identity_audit() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = read_tsv(REPRESENTATION_AUDIT)
    chembl = [row for row in rows if row["source_family"] == "chembl"]
    openadmet = [row for row in rows if row["source_family"] == "openadmet"]
    return chembl, openadmet


def openadmet_partitions_by_key(
    openadmet_rows: list[dict[str, str]],
) -> dict[str, set[str]]:
    by_key = defaultdict(set)
    for row in openadmet_rows:
        if row["identity_state"] != "represented":
            raise RuntimeError("qualified OpenADMET record is not represented")
        by_key[row["full_standard_inchikey"]].add(row["source_partition"])
    return dict(by_key)


def selected_chembl_records(
    chembl_rows: list[dict[str, str]],
    values_by_activity: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    represented = []
    unresolved = []
    for row in chembl_rows:
        activity_id = row["source_id"]
        if activity_id not in values_by_activity:
            raise RuntimeError(f"ChEMBL audit row missing eligibility value: {activity_id}")
        value_row = values_by_activity[activity_id]
        merged = {
            **row,
            "activity_id": activity_id,
            "standard_relation": value_row["standard_relation"],
            "standard_value": value_row["standard_value"],
            "standard_units": value_row["standard_units"],
            "pchembl_value": value_row["pchembl_value"],
            "record_id": value_row["record_id"],
            "document_chembl_id": value_row["document_chembl_id"],
            "canonical_smiles_frozen": value_row["canonical_smiles"],
        }
        if row["identity_state"] == "represented":
            represented.append(merged)
        else:
            unresolved.append(merged)

    if len(represented) != 4821:
        raise RuntimeError(f"expected 4821 represented ChEMBL records, got {len(represented)}")
    if len(unresolved) != 1:
        raise RuntimeError(f"expected 1 unresolved ChEMBL record, got {len(unresolved)}")
    unresolved_record = unresolved[0]
    if (
        unresolved_record["activity_id"] != "7817060"
        or unresolved_record["molecule_chembl_id"] != "CHEMBL508338"
        or unresolved_record["identity_state"] != "source_structure_unavailable"
        or unresolved_record["source_smiles"] != ""
    ):
        raise RuntimeError("activity_id 7817060 must remain unresolved and structureless")

    return represented, unresolved


def aggregate_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    groups = defaultdict(list)
    for row in records:
        groups[(row["isoform"], row["assay_chembl_id"], row["full_standard_inchikey"])].append(row)

    aggregated = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: row["activity_id"])
        derived = [pic50_from_ic50(row["standard_value"], row["standard_units"]) for row in rows]
        median_pic50 = statistics.median(derived)
        stored = [row["pchembl_value"] for row in rows]
        discrepancies = []
        for row, pic50 in zip(rows, derived, strict=True):
            if row["pchembl_value"] == "":
                continue
            delta = pic50 - float(row["pchembl_value"])
            if abs(delta) > 1e-6:
                discrepancies.append(delta)

        standard_inchis = sorted({row["standard_inchi"] for row in rows})
        if len(standard_inchis) != 1:
            raise RuntimeError(
                "same full-InChIKey group has differing retained Standard InChI"
            )

        aggregated.append({
            "identity_group_id": "|".join(key),
            "isoform": key[0],
            "assay_chembl_id": key[1],
            "full_standard_inchikey": key[2],
            "standard_inchi": standard_inchis[0],
            "canonical_isomeric_smiles_witnesses": json.dumps(sorted({
                row["canonical_isomeric_smiles_witness"] for row in rows
            })),
            "source_smiles_values": json.dumps([row["source_smiles"] for row in rows]),
            "activity_ids": ";".join(row["activity_id"] for row in rows),
            "record_ids": ";".join(row["record_id"] for row in rows),
            "molecule_chembl_ids": join_sorted({row["molecule_chembl_id"] for row in rows}),
            "document_chembl_ids": join_sorted({row["document_chembl_id"] for row in rows if row["document_chembl_id"]}),
            "replicate_count": str(len(rows)),
            "standard_relation_values": ";".join(row["standard_relation"] for row in rows),
            "ic50_values": ";".join(row["standard_value"] for row in rows),
            "ic50_units": ";".join(row["standard_units"] for row in rows),
            "derived_pic50_values": ";".join(fmt_float(value) for value in derived),
            "median_derived_pic50": fmt_float(median_pic50),
            "derived_pic50_min": fmt_float(min(derived)),
            "derived_pic50_max": fmt_float(max(derived)),
            "derived_pic50_range": fmt_float(max(derived) - min(derived)),
            "stored_pchembl_values": ";".join(stored),
            "pchembl_witness_discrepancy_count": str(len(discrepancies)),
            "pchembl_witness_max_abs_discrepancy": (
                fmt_float(max(abs(value) for value in discrepancies))
                if discrepancies else "0"
            ),
        })

    if len(records) == 4821 and len(aggregated) != 4690:
        raise RuntimeError(f"expected 4690 aggregated identities, got {len(aggregated)}")
    return aggregated


def route_identities(
    aggregated: list[dict[str, str]],
    open_partitions: dict[str, set[str]],
) -> list[dict[str, str]]:
    routed = []
    for row in aggregated:
        partitions = sorted(open_partitions.get(row["full_standard_inchikey"], set()))
        flags = {partition: partition in partitions for partition in ALL_PARTITIONS}
        protected = any(flags[partition] for partition in PROTECTED_PARTITIONS)
        train_overlap = flags["train"]
        if protected:
            available = False
            if flags["challenge_blinded_test"]:
                route = "protected_challenge_blinded_test_overlap"
            elif flags["internal_test"]:
                route = "protected_internal_test_overlap"
            else:
                route = "protected_validation_overlap"
        elif train_overlap:
            available = True
            route = "overlap_openadmet_train"
        else:
            available = True
            route = "external_only"

        routed.append({
            **row,
            "openadmet_partitions": ";".join(partitions),
            "overlap_openadmet_train": str(flags["train"]).lower(),
            "overlap_openadmet_validation": str(flags["validation"]).lower(),
            "overlap_openadmet_internal_test": str(flags["internal_test"]).lower(),
            "overlap_openadmet_challenge_blinded_test": str(flags["challenge_blinded_test"]).lower(),
            "protected_overlap": str(protected).lower(),
            "available_for_successor_modeling": str(available).lower(),
            "routing_class": route,
        })
    return routed


def build_census(
    tasks: list[dict[str, str]],
    represented: list[dict[str, str]],
    unresolved: list[dict[str, str]],
    routed: list[dict[str, str]],
    duplicate_task_keys: set[tuple[str, str]],
) -> list[dict[str, str]]:
    represented_counts = Counter((row["isoform"], row["assay_chembl_id"]) for row in represented)
    unresolved_counts = Counter((row["isoform"], row["assay_chembl_id"]) for row in unresolved)
    routed_by_task = defaultdict(list)
    for row in routed:
        routed_by_task[(row["isoform"], row["assay_chembl_id"])].append(row)

    census = []
    for task in sorted(tasks, key=lambda row: (row["isoform"], row["assay_chembl_id"])):
        key = (task["isoform"], task["assay_chembl_id"])
        rows = routed_by_task.get(key, [])
        final_available = sum(row["available_for_successor_modeling"] == "true" for row in rows)
        census.append({
            "isoform": key[0],
            "assay_chembl_id": key[1],
            "original_exact_records": task["exact_candidate_record_count"],
            "represented_exact_records": str(represented_counts[key]),
            "unresolved_identity_records": str(unresolved_counts[key]),
            "replicate_groups_aggregated_identities": str(len(rows)),
            "train_overlap_identities": str(sum(row["overlap_openadmet_train"] == "true" for row in rows)),
            "validation_overlap_identities": str(sum(row["overlap_openadmet_validation"] == "true" for row in rows)),
            "internal_test_overlap_identities": str(sum(row["overlap_openadmet_internal_test"] == "true" for row in rows)),
            "challenge_blinded_test_overlap_identities": str(sum(row["overlap_openadmet_challenge_blinded_test"] == "true" for row in rows)),
            "external_only_identities": str(sum(row["routing_class"] == "external_only" for row in rows)),
            "final_available_identities_after_protected_overlap_removal": str(final_available),
            "participates_in_demonstrated_duplicate_relationship": str(key in duplicate_task_keys).lower(),
            "task_size_filter_applied": "false",
        })
    return census


def distribution(census: list[dict[str, str]], duplicate_pairs: list[dict[str, str]]) -> dict[str, object]:
    by_isoform = defaultdict(list)
    zero_available = []
    for row in census:
        value = int(row["final_available_identities_after_protected_overlap_removal"])
        by_isoform[row["isoform"]].append(value)
        if value == 0:
            zero_available.append({
                "isoform": row["isoform"],
                "assay_chembl_id": row["assay_chembl_id"],
            })

    def summarize(values: list[int]) -> dict[str, int]:
        counts = Counter(values)
        return {
            "task_count": len(values),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "zero_available_task_count": counts[0],
            "distribution": dict(sorted((str(key), value) for key, value in counts.items())),
        }

    all_values = [
        int(row["final_available_identities_after_protected_overlap_removal"])
        for row in census
    ]
    return {
        "task_counts": {
            "before_demonstrated_duplicate_consolidation": 176,
            "demonstrated_duplicate_relationship_count": len(duplicate_pairs),
            "after_demonstrated_duplicate_consolidation": 175,
            "remaining_tasks_are_not_called_independent": True,
        },
        "overall": summarize(all_values),
        "by_isoform": {isoform: summarize(values) for isoform, values in sorted(by_isoform.items())},
        "tasks_with_zero_available_identities": zero_available,
        "minimum_task_size_filter_applied": False,
    }


def build_outputs() -> dict[str, object]:
    tasks = load_tasks()
    keys = selected_task_keys(tasks)
    duplicate_task_keys, duplicate_pairs = load_relationships(keys)
    values_by_activity = load_record_values(keys)
    if len(values_by_activity) != 4822:
        raise RuntimeError(f"expected 4822 exact records, got {len(values_by_activity)}")

    chembl, openadmet = load_identity_audit()
    represented, unresolved = selected_chembl_records(chembl, values_by_activity)
    aggregated = aggregate_records(represented)
    routed = route_identities(aggregated, openadmet_partitions_by_key(openadmet))
    census = build_census(tasks, represented, unresolved, routed, duplicate_task_keys)
    dist = distribution(census, duplicate_pairs)

    represented_activity_ids = [row["activity_id"] for row in represented]
    aggregated_activity_ids = []
    for row in aggregated:
        aggregated_activity_ids.extend(row["activity_ids"].split(";"))
    if sorted(represented_activity_ids) != sorted(aggregated_activity_ids):
        raise RuntimeError("represented records are not accounted for exactly once")

    protected_unavailable_violations = [
        row["identity_group_id"] for row in routed
        if row["protected_overlap"] == "true" and row["available_for_successor_modeling"] == "true"
    ]
    if protected_unavailable_violations:
        raise RuntimeError("protected identities were marked available")
    train_unavailable_violations = [
        row["identity_group_id"] for row in routed
        if (
            row["routing_class"] == "overlap_openadmet_train"
            and row["available_for_successor_modeling"] != "true"
        )
    ]
    if train_unavailable_violations:
        raise RuntimeError("train-only overlap identities were not retained")

    discrepancy_counts = [int(row["pchembl_witness_discrepancy_count"]) for row in aggregated]
    max_discrepancy = max(float(row["pchembl_witness_max_abs_discrepancy"]) for row in aggregated)
    route_counts = Counter(row["routing_class"] for row in routed)
    partition_counts = {
        partition: sum(row[f"overlap_openadmet_{partition}"] == "true" for row in routed)
        for partition in ALL_PARTITIONS
    }
    summary = {
        "artifact_type": "CYP_successor_identity_routed_exact_IC50_preparation_v1",
        "authority": "preparation_and_characterization_only_no_training_or_episode_design",
        "qualified_identity_contract": {
            "python": sys.version,
            "rdkit_version": rdBase.rdkitVersion,
            "inchi_version": inchi.GetInchiVersion(),
            "cross_source_molecular_identity": "full Standard InChIKey",
            "project_added_molecular_standardization": False,
            "source_smiles_preserved": True,
            "canonical_isomeric_smiles_witness_preserved": True,
            "full_standard_inchi_preserved": True,
            "molecule_name_partitions_unchanged": True,
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [REPRESENTATION_AUDIT, TASKS, RECORD_ELIGIBILITY, RELATIONSHIPS]
        },
        "measurement_aggregation": {
            "original_exact_records": len(values_by_activity),
            "represented_exact_records": len(represented),
            "source_structure_unavailable_records": len(unresolved),
            "aggregated_within_assay_identity_groups": len(aggregated),
            "grouping": ["isoform", "assay_chembl_id", "full_standard_inchikey"],
            "label": "median derived pIC50 from exact IC50 molar conversion",
            "aggregate_across_assays": False,
            "non_exact_records_incorporated": False,
        },
        "pchembl_witness_discrepancies": {
            "groups_with_any_discrepancy": sum(value > 0 for value in discrepancy_counts),
            "records_with_discrepancy": sum(discrepancy_counts),
            "max_abs_discrepancy": fmt_float(max_discrepancy),
            "comparison_tolerance": "1e-6 absolute pIC50 units",
        },
        "molecular_identity_routing": {
            "routing_class_counts": dict(sorted(route_counts.items())),
            "overlap_counts_by_partition": partition_counts,
            "available_identity_count": sum(row["available_for_successor_modeling"] == "true" for row in routed),
            "protected_unavailable_identity_count": sum(row["protected_overlap"] == "true" for row in routed),
            "train_overlap_retained_as_available": True,
        },
        "task_level_demonstrated_duplicate_status": {
            "before_consolidation_task_count": 176,
            "demonstrated_duplicate_relationships": duplicate_pairs,
            "participating_task_count": len(duplicate_task_keys),
            "after_consolidation_task_count": 175,
            "unresolved_provenance_remains_unresolved": True,
            "remaining_tasks_called_independent": False,
        },
        "final_task_population_characterization": dist,
        "safeguards": {
            "public_test_labels_accessed": False,
            "minimum_task_size_filter_applied": False,
            "support_query_sizes_chosen": False,
            "models_trained": False,
            "openadmet_partitions_changed": False,
        },
    }

    return {
        "aggregated": aggregated,
        "routed": routed,
        "census": census,
        "distribution": dist,
        "summary": summary,
    }


AGGREGATED_FIELDS = [
    "identity_group_id",
    "isoform",
    "assay_chembl_id",
    "full_standard_inchikey",
    "standard_inchi",
    "canonical_isomeric_smiles_witnesses",
    "source_smiles_values",
    "activity_ids",
    "record_ids",
    "molecule_chembl_ids",
    "document_chembl_ids",
    "replicate_count",
    "standard_relation_values",
    "ic50_values",
    "ic50_units",
    "derived_pic50_values",
    "median_derived_pic50",
    "derived_pic50_min",
    "derived_pic50_max",
    "derived_pic50_range",
    "stored_pchembl_values",
    "pchembl_witness_discrepancy_count",
    "pchembl_witness_max_abs_discrepancy",
]

ROUTING_FIELDS = AGGREGATED_FIELDS + [
    "openadmet_partitions",
    "overlap_openadmet_train",
    "overlap_openadmet_validation",
    "overlap_openadmet_internal_test",
    "overlap_openadmet_challenge_blinded_test",
    "protected_overlap",
    "available_for_successor_modeling",
    "routing_class",
]

CENSUS_FIELDS = [
    "isoform",
    "assay_chembl_id",
    "original_exact_records",
    "represented_exact_records",
    "unresolved_identity_records",
    "replicate_groups_aggregated_identities",
    "train_overlap_identities",
    "validation_overlap_identities",
    "internal_test_overlap_identities",
    "challenge_blinded_test_overlap_identities",
    "external_only_identities",
    "final_available_identities_after_protected_overlap_removal",
    "participates_in_demonstrated_duplicate_relationship",
    "task_size_filter_applied",
]


def write_outputs(outputs: dict[str, object]) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "aggregated_exact_ic50.tsv": OUT / "aggregated_exact_ic50.tsv",
        "identity_routing.tsv": OUT / "identity_routing.tsv",
        "post_routing_task_census.tsv": OUT / "post_routing_task_census.tsv",
        "post_routing_task_distribution.json": OUT / "post_routing_task_distribution.json",
        "preparation_summary.json": OUT / "preparation_summary.json",
    }
    write_tsv(paths["aggregated_exact_ic50.tsv"], AGGREGATED_FIELDS, outputs["aggregated"])
    write_tsv(paths["identity_routing.tsv"], ROUTING_FIELDS, outputs["routed"])
    write_tsv(paths["post_routing_task_census.tsv"], CENSUS_FIELDS, outputs["census"])
    paths["post_routing_task_distribution.json"].write_text(
        json.dumps(outputs["distribution"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["preparation_summary.json"].write_text(
        json.dumps(outputs["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    hash_path = OUT / "SHA256SUMS.txt"
    with hash_path.open("w", encoding="utf-8") as handle:
        for name in sorted(paths):
            handle.write(f"{sha256_file(paths[name])}  {name}\n")

    return {name: sha256_file(path) for name, path in {**paths, "SHA256SUMS.txt": hash_path}.items()}


def main() -> None:
    outputs = build_outputs()
    hashes = write_outputs(outputs)
    summary = outputs["summary"]
    print("=== SUCCESSOR IDENTITY-ROUTED PREPARATION ===")
    print("represented exact records:", summary["measurement_aggregation"]["represented_exact_records"])
    print("aggregated identities:", summary["measurement_aggregation"]["aggregated_within_assay_identity_groups"])
    print("routing classes:", summary["molecular_identity_routing"]["routing_class_counts"])
    print("partition overlaps:", summary["molecular_identity_routing"]["overlap_counts_by_partition"])
    print("tasks before/after demonstrated duplicate consolidation:", 176, 175)
    print("zero-available tasks:", len(summary["final_task_population_characterization"]["tasks_with_zero_available_identities"]))
    print("output hashes:", hashes)
    print("PREPARATION COMPLETE -- NO TRAINING OR EPISODE DESIGN PERFORMED")


if __name__ == "__main__":
    main()
