from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/Users/mileshuang/Desktop/tripwire-project")
QUAL = ROOT / "qualification/cyp_external_task_qualification_20261109"
PREP = QUAL / "successor_identity_routed_preparation"
OUT = QUAL / "d012_auxiliary_population"

IDENTITY_ROUTING = PREP / "identity_routing.tsv"
TASK_CENSUS = PREP / "post_routing_task_census.tsv"
RELATIONSHIPS = QUAL / "derived/p2_compatible_ic50_relationships.tsv"

DUPLICATE_LEFT = ("CYP3A4", "CHEMBL2448771")
DUPLICATE_RIGHT = ("CYP3A4", "CHEMBL3137501")
DUPLICATE_MEMBERS = {DUPLICATE_LEFT, DUPLICATE_RIGHT}
D012_CONSOLIDATED_ASSAY_ID = "D012_CONSOLIDATED_CYP3A4_CHEMBL2448771_CHEMBL3137501"


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


def split_semicolon(value: str) -> list[str]:
    return [] if value == "" else value.split(";")


def join_sorted(values: set[str] | list[str]) -> str:
    return ";".join(sorted(values))


def ic50_to_molar(value: str, unit: str) -> float:
    factor = {
        "M": 1.0,
        "mM": 1e-3,
        "uM": 1e-6,
        "µM": 1e-6,
        "nM": 1e-9,
        "pM": 1e-12,
    }[unit]
    return float(value) * factor


def pic50_from_ic50(value: str, unit: str) -> float:
    return -math.log10(ic50_to_molar(value, unit))


def source_task_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["isoform"], row["assay_chembl_id"])


def load_duplicate_relationship() -> dict[str, str]:
    duplicate_rows = [
        row for row in read_tsv(RELATIONSHIPS)
        if row["relationship_category"] == "demonstrated_duplicate"
    ]
    if len(duplicate_rows) != 1:
        raise RuntimeError(f"expected one demonstrated duplicate relationship, got {len(duplicate_rows)}")

    row = duplicate_rows[0]
    observed = {
        (row["left_isoform"], row["left_assay_chembl_id"]),
        (row["right_isoform"], row["right_assay_chembl_id"]),
    }
    if observed != DUPLICATE_MEMBERS:
        raise RuntimeError(f"unexpected demonstrated duplicate members: {observed}")
    if row["shared_assay_src_id"] != "14" or row["shared_assay_src_assay_id"] != "1034082":
        raise RuntimeError("demonstrated duplicate evidence does not match D012 contract")
    return row


def available_routed_rows() -> list[dict[str, str]]:
    rows = read_tsv(IDENTITY_ROUTING)
    available = [row for row in rows if row["available_for_successor_modeling"] == "true"]
    if len(available) != 4664:
        raise RuntimeError(f"expected 4664 available routed identities, got {len(available)}")
    if any(row["protected_overlap"] == "true" for row in available):
        raise RuntimeError("protected overlap identity leaked into D012 available population")
    return available


def observation_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    observations = []
    for row in rows:
        activity_ids = split_semicolon(row["activity_ids"])
        record_ids = split_semicolon(row["record_ids"])
        ic50_values = split_semicolon(row["ic50_values"])
        ic50_units = split_semicolon(row["ic50_units"])
        derived_pic50_values = split_semicolon(row["derived_pic50_values"])
        stored_pchembl_values = split_semicolon(row["stored_pchembl_values"])
        source_smiles_values = json.loads(row["source_smiles_values"])
        if not (
            len(activity_ids)
            == len(record_ids)
            == len(ic50_values)
            == len(ic50_units)
            == len(derived_pic50_values)
            == len(stored_pchembl_values)
            == len(source_smiles_values)
        ):
            raise RuntimeError(f"replicate vector lengths differ for {row['identity_group_id']}")
        for index, activity_id in enumerate(activity_ids):
            derived = pic50_from_ic50(ic50_values[index], ic50_units[index])
            observations.append({
                "activity_id": activity_id,
                "record_id": record_ids[index],
                "source_assay_chembl_id": row["assay_chembl_id"],
                "source_smiles": source_smiles_values[index],
                "ic50_value": ic50_values[index],
                "ic50_unit": ic50_units[index],
                "derived_pic50": fmt_float(derived),
                "stored_pchembl_value": stored_pchembl_values[index],
            })
    return observations


def combine_identity_rows(
    task_unit_id: str,
    isoform: str,
    source_task_keys: list[tuple[str, str]],
    rows: list[dict[str, str]],
) -> dict[str, str]:
    rows = sorted(rows, key=lambda row: (row["assay_chembl_id"], row["full_standard_inchikey"], row["identity_group_id"]))
    full_keys = {row["full_standard_inchikey"] for row in rows}
    if len(full_keys) != 1:
        raise RuntimeError("cannot combine different full-InChIKey identities")
    standard_inchis = {row["standard_inchi"] for row in rows}
    if len(standard_inchis) != 1:
        raise RuntimeError("same full-InChIKey has differing retained Standard InChI")

    observations = sorted(observation_records(rows), key=lambda row: (row["source_assay_chembl_id"], row["activity_id"]))
    pic50_values = [float(row["derived_pic50"]) for row in observations]
    activity_ids = [row["activity_id"] for row in observations]
    if len(activity_ids) != len(set(activity_ids)):
        raise RuntimeError(f"duplicate activity_id in D012 identity {full_keys}")

    source_assay_ids = sorted({assay_id for _, assay_id in source_task_keys})
    routing_classes = sorted({row["routing_class"] for row in rows})
    openadmet_partitions = sorted({
        partition
        for row in rows
        for partition in split_semicolon(row["openadmet_partitions"])
    })
    if any(row["routing_class"] not in {"external_only", "overlap_openadmet_train"} for row in rows):
        raise RuntimeError("D012 materialization received unavailable/protected identity")

    return {
        "task_unit_id": task_unit_id,
        "isoform": isoform,
        "source_assay_chembl_ids": ";".join(source_assay_ids),
        "is_demonstrated_duplicate_consolidated_task": str(len(source_assay_ids) > 1).lower(),
        "full_standard_inchikey": next(iter(full_keys)),
        "standard_inchi": next(iter(standard_inchis)),
        "median_derived_pic50": fmt_float(statistics.median(pic50_values)),
        "derived_pic50_values": ";".join(row["derived_pic50"] for row in observations),
        "derived_pic50_min": fmt_float(min(pic50_values)),
        "derived_pic50_max": fmt_float(max(pic50_values)),
        "derived_pic50_range": fmt_float(max(pic50_values) - min(pic50_values)),
        "contributing_observation_count": str(len(observations)),
        "contributing_activity_ids": ";".join(activity_ids),
        "contributing_record_ids": ";".join(row["record_id"] for row in observations),
        "contributing_source_assay_ids_by_observation": ";".join(row["source_assay_chembl_id"] for row in observations),
        "contributing_ic50_values": ";".join(row["ic50_value"] for row in observations),
        "contributing_ic50_units": ";".join(row["ic50_unit"] for row in observations),
        "contributing_source_smiles": json.dumps([row["source_smiles"] for row in observations]),
        "stored_pchembl_values": ";".join(row["stored_pchembl_value"] for row in observations),
        "routing_class": routing_classes[0] if len(routing_classes) == 1 else "mixed_available_overlap_classes",
        "openadmet_partitions": ";".join(openadmet_partitions),
        "overlap_openadmet_train": str(any(row["overlap_openadmet_train"] == "true" for row in rows)).lower(),
        "available_for_d012_auxiliary_population": "true",
    }


def task_unit_id_for(key: tuple[str, str]) -> str:
    if key in DUPLICATE_MEMBERS:
        return D012_CONSOLIDATED_ASSAY_ID
    return f"{key[0]}|{key[1]}"


def build_task_identity_instances(available: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = defaultdict(list)
    for row in available:
        key = source_task_key(row)
        if key in DUPLICATE_MEMBERS:
            grouped[(D012_CONSOLIDATED_ASSAY_ID, row["isoform"], row["full_standard_inchikey"])].append(row)
        else:
            grouped[(task_unit_id_for(key), row["isoform"], row["full_standard_inchikey"])].append(row)

    instances = []
    for (task_unit_id, isoform, _full_key), rows in sorted(grouped.items()):
        source_keys = sorted({source_task_key(row) for row in rows})
        instances.append(combine_identity_rows(task_unit_id, isoform, source_keys, rows))
    return instances


def percentile_nearest_rank(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if percentile == 0:
        return ordered[0]
    rank = math.ceil(percentile / 100 * len(ordered))
    return ordered[min(max(rank - 1, 0), len(ordered) - 1)]


def summarize_sizes(values: list[int]) -> dict[str, object]:
    counts = Counter(values)
    return {
        "task_count": len(values),
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "quantile_method": "nearest_rank",
        "quantiles": {
            "p0": percentile_nearest_rank(values, 0),
            "p25": percentile_nearest_rank(values, 25),
            "p50": percentile_nearest_rank(values, 50),
            "p75": percentile_nearest_rank(values, 75),
            "p100": percentile_nearest_rank(values, 100),
        },
        "distribution": dict(sorted((str(key), value) for key, value in counts.items())),
        "small_size_counts_1_to_20": {str(size): counts[size] for size in range(1, 21)},
        "zero_identity_task_count": counts[0],
    }


def build_task_units(instances: list[dict[str, str]], source_census: list[dict[str, str]]) -> list[dict[str, str]]:
    source_by_key = {(row["isoform"], row["assay_chembl_id"]): row for row in source_census}
    task_units = []
    source_keys_seen = set()
    instances_by_task = defaultdict(list)
    for row in instances:
        instances_by_task[row["task_unit_id"]].append(row)

    for source_key in sorted(source_by_key):
        if source_key in source_keys_seen:
            continue
        if source_key in DUPLICATE_MEMBERS:
            source_keys = sorted(DUPLICATE_MEMBERS)
            source_keys_seen.update(source_keys)
            task_unit_id = D012_CONSOLIDATED_ASSAY_ID
        else:
            source_keys = [source_key]
            source_keys_seen.add(source_key)
            task_unit_id = task_unit_id_for(source_key)

        rows = instances_by_task.get(task_unit_id, [])
        route_counts = Counter(row["routing_class"] for row in rows)
        task_units.append({
            "task_unit_id": task_unit_id,
            "isoform": source_keys[0][0],
            "source_assay_chembl_ids": ";".join(assay for _, assay in source_keys),
            "is_demonstrated_duplicate_consolidated_task": str(len(source_keys) > 1).lower(),
            "source_task_count": str(len(source_keys)),
            "available_identity_count": str(len(rows)),
            "source_exact_record_count": str(sum(int(source_by_key[key]["original_exact_records"]) for key in source_keys)),
            "source_represented_record_count": str(sum(int(source_by_key[key]["represented_exact_records"]) for key in source_keys)),
            "external_only_identity_count": str(route_counts["external_only"]),
            "overlap_openadmet_train_identity_count": str(route_counts["overlap_openadmet_train"]),
            "mixed_available_overlap_class_identity_count": str(route_counts["mixed_available_overlap_classes"]),
        })

    if len(task_units) != 175:
        raise RuntimeError(f"expected 175 D012 task units, got {len(task_units)}")
    return sorted(task_units, key=lambda row: (row["isoform"], row["task_unit_id"]))


def build_outputs() -> dict[str, object]:
    relationship = load_duplicate_relationship()
    available = available_routed_rows()
    source_census = read_tsv(TASK_CENSUS)
    if len(source_census) != 176:
        raise RuntimeError(f"expected 176 source task census rows, got {len(source_census)}")

    left_rows = [row for row in available if source_task_key(row) == DUPLICATE_LEFT]
    right_rows = [row for row in available if source_task_key(row) == DUPLICATE_RIGHT]
    if len(left_rows) != 36 or len(right_rows) != 27:
        raise RuntimeError("duplicate member available identity counts changed")
    left_keys = {row["full_standard_inchikey"] for row in left_rows}
    right_keys = {row["full_standard_inchikey"] for row in right_rows}
    intersection = sorted(left_keys & right_keys)
    union = sorted(left_keys | right_keys)

    instances = build_task_identity_instances(available)
    task_units = build_task_units(instances, source_census)

    instance_task_units = {row["task_unit_id"] for row in instances}
    task_unit_ids = {row["task_unit_id"] for row in task_units}
    if not instance_task_units <= task_unit_ids:
        raise RuntimeError("identity instance references unknown task unit")

    if len(task_units) != 175:
        raise RuntimeError("D012 task unit count changed")
    duplicate_instances = [row for row in instances if row["task_unit_id"] == D012_CONSOLIDATED_ASSAY_ID]
    if len(duplicate_instances) != len(union):
        raise RuntimeError("duplicate union identity count mismatch")

    total_observations = sum(int(row["contributing_observation_count"]) for row in instances)
    route_counts = Counter(row["routing_class"] for row in instances)
    sizes = [int(row["available_identity_count"]) for row in task_units]
    by_isoform = defaultdict(list)
    for row in task_units:
        by_isoform[row["isoform"]].append(int(row["available_identity_count"]))
    zero_tasks = [
        {"task_unit_id": row["task_unit_id"], "isoform": row["isoform"], "source_assay_chembl_ids": row["source_assay_chembl_ids"]}
        for row in task_units
        if int(row["available_identity_count"]) == 0
    ]

    shared_details = []
    for key in intersection:
        source_rows = [row for row in left_rows + right_rows if row["full_standard_inchikey"] == key]
        shared_details.append({
            "full_standard_inchikey": key,
            "contributing_observation_count": sum(int(row["replicate_count"]) for row in source_rows),
            "contributing_activity_ids": ";".join(
                activity_id
                for row in sorted(source_rows, key=lambda item: item["assay_chembl_id"])
                for activity_id in split_semicolon(row["activity_ids"])
            ),
        })

    duplicate_summary = {
        "relationship": relationship,
        "left_task": {"isoform": DUPLICATE_LEFT[0], "assay_chembl_id": DUPLICATE_LEFT[1], "available_identity_count": len(left_rows)},
        "right_task": {"isoform": DUPLICATE_RIGHT[0], "assay_chembl_id": DUPLICATE_RIGHT[1], "available_identity_count": len(right_rows)},
        "identity_intersection_count": len(intersection),
        "identity_union_count": len(union),
        "shared_full_standard_inchikeys": intersection,
        "shared_identity_observations": shared_details,
        "consolidated_task_unit_id": D012_CONSOLIDATED_ASSAY_ID,
    }

    distribution = {
        "overall": summarize_sizes(sizes),
        "by_isoform": {isoform: summarize_sizes(values) for isoform, values in sorted(by_isoform.items())},
        "zero_identity_tasks": zero_tasks,
    }

    summary = {
        "artifact_type": "D012_single_demonstrated_duplicate_auxiliary_population_v1",
        "authority": "materialization_only_no_training_no_episode_design",
        "runtime": {"python": sys.version},
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [IDENTITY_ROUTING, TASK_CENSUS, RELATIONSHIPS]
        },
        "method_contract": {
            "source_task_count": 176,
            "final_task_unit_count": 175,
            "minimum_task_size_filter_applied": False,
            "old_16_molecule_rule_applied": False,
            "only_demonstrated_duplicate_relationship_materialized": True,
            "public_test_labels_accessed": False,
            "d011_modified": False,
            "training_performed": False,
            "unresolved_provenance_changed": False,
        },
        "demonstrated_duplicate_consolidation": duplicate_summary,
        "final_population_counts": {
            "task_identity_instances": len(instances),
            "source_exact_observations_represented_by_instances": total_observations,
            "external_only_identity_instances": route_counts["external_only"],
            "overlap_openadmet_train_identity_instances": route_counts["overlap_openadmet_train"],
            "mixed_available_overlap_class_identity_instances": route_counts["mixed_available_overlap_classes"],
        },
        "task_size_distribution": distribution,
    }
    return {
        "instances": instances,
        "task_units": task_units,
        "duplicate_summary": duplicate_summary,
        "distribution": distribution,
        "summary": summary,
    }


INSTANCE_FIELDS = [
    "task_unit_id",
    "isoform",
    "source_assay_chembl_ids",
    "is_demonstrated_duplicate_consolidated_task",
    "full_standard_inchikey",
    "standard_inchi",
    "median_derived_pic50",
    "derived_pic50_values",
    "derived_pic50_min",
    "derived_pic50_max",
    "derived_pic50_range",
    "contributing_observation_count",
    "contributing_activity_ids",
    "contributing_record_ids",
    "contributing_source_assay_ids_by_observation",
    "contributing_ic50_values",
    "contributing_ic50_units",
    "contributing_source_smiles",
    "stored_pchembl_values",
    "routing_class",
    "openadmet_partitions",
    "overlap_openadmet_train",
    "available_for_d012_auxiliary_population",
]

TASK_UNIT_FIELDS = [
    "task_unit_id",
    "isoform",
    "source_assay_chembl_ids",
    "is_demonstrated_duplicate_consolidated_task",
    "source_task_count",
    "available_identity_count",
    "source_exact_record_count",
    "source_represented_record_count",
    "external_only_identity_count",
    "overlap_openadmet_train_identity_count",
    "mixed_available_overlap_class_identity_count",
]


def write_outputs(outputs: dict[str, object]) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "d012_task_identity_instances.tsv": OUT / "d012_task_identity_instances.tsv",
        "d012_task_units.tsv": OUT / "d012_task_units.tsv",
        "d012_duplicate_consolidation.json": OUT / "d012_duplicate_consolidation.json",
        "d012_task_size_distribution.json": OUT / "d012_task_size_distribution.json",
        "d012_auxiliary_population_summary.json": OUT / "d012_auxiliary_population_summary.json",
    }
    write_tsv(paths["d012_task_identity_instances.tsv"], INSTANCE_FIELDS, outputs["instances"])
    write_tsv(paths["d012_task_units.tsv"], TASK_UNIT_FIELDS, outputs["task_units"])
    paths["d012_duplicate_consolidation.json"].write_text(
        json.dumps(outputs["duplicate_summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["d012_task_size_distribution.json"].write_text(
        json.dumps(outputs["distribution"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["d012_auxiliary_population_summary.json"].write_text(
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
    duplicate = summary["demonstrated_duplicate_consolidation"]
    counts = summary["final_population_counts"]
    print("=== D012 AUXILIARY POPULATION ===")
    print("source tasks -> task units:", 176, "->", 175)
    print("duplicate member available identities:", duplicate["left_task"]["available_identity_count"], duplicate["right_task"]["available_identity_count"])
    print("duplicate identity intersection/union:", duplicate["identity_intersection_count"], duplicate["identity_union_count"])
    print("final task-identity instances:", counts["task_identity_instances"])
    print("source exact observations represented:", counts["source_exact_observations_represented_by_instances"])
    print("available routing counts:", {
        "external_only": counts["external_only_identity_instances"],
        "overlap_openadmet_train": counts["overlap_openadmet_train_identity_instances"],
        "mixed_available_overlap_classes": counts["mixed_available_overlap_class_identity_instances"],
    })
    print("zero-identity tasks:", len(summary["task_size_distribution"]["zero_identity_tasks"]))
    print("output hashes:", hashes)
    print("D012 MATERIALIZATION COMPLETE -- NO TRAINING PERFORMED")


if __name__ == "__main__":
    main()
