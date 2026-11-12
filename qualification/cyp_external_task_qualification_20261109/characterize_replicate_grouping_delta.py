from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/mileshuang/Desktop/tripwire-project")
QUAL = ROOT / "qualification/cyp_external_task_qualification_20261109"
OUT = QUAL / "identity_audit"

REPRESENTATION_AUDIT = OUT / "bounded_representation_audit.tsv"
RECORD_ELIGIBILITY = QUAL / "derived/record_eligibility.tsv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def numeric_or_none(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def range_fields(values: list[str]) -> dict[str, str]:
    numeric = [numeric_or_none(value) for value in values]
    if not numeric or any(value is None for value in numeric):
        return {
            "numeric_value_count": str(sum(value is not None for value in numeric)),
            "value_min": "",
            "value_max": "",
            "value_spread": "",
        }

    typed = [value for value in numeric if value is not None]
    value_min = min(typed)
    value_max = max(typed)
    return {
        "numeric_value_count": str(len(typed)),
        "value_min": f"{value_min:g}",
        "value_max": f"{value_max:g}",
        "value_spread": f"{value_max - value_min:g}",
    }


def join_sorted(values: set[str] | list[str]) -> str:
    return ";".join(sorted(values))


def load_selected_representation_records() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:
    rows = read_tsv(REPRESENTATION_AUDIT)
    chembl_rows = [
        row for row in rows
        if row["source_family"] == "chembl"
    ]
    represented = [
        row for row in chembl_rows
        if row["identity_state"] == "represented"
    ]
    unresolved = [
        row for row in chembl_rows
        if row["identity_state"] != "represented"
    ]

    unresolved_7817060 = [
        row for row in unresolved
        if row["source_id"] == "7817060"
    ]
    if len(unresolved_7817060) != 1:
        raise RuntimeError(
            "activity_id 7817060 must remain the single unresolved "
            "selected ChEMBL record"
        )

    row_7817060 = unresolved_7817060[0]
    if row_7817060["molecule_chembl_id"] != "CHEMBL508338":
        raise RuntimeError("activity_id 7817060 molecule ID changed")
    if row_7817060["identity_state"] != "source_structure_unavailable":
        raise RuntimeError(
            "activity_id 7817060 must remain source_structure_unavailable"
        )
    if row_7817060["source_smiles"] != "":
        raise RuntimeError(
            "activity_id 7817060 must not receive an inferred source structure"
        )

    for row in represented:
        if not row["full_standard_inchikey"] or not row["standard_inchi"]:
            raise RuntimeError(
                f"represented activity {row['source_id']} lacks full key/InChI"
            )

    return represented, unresolved


def load_exact_values(activity_ids: set[str]) -> dict[str, dict[str, str]]:
    wanted = set(activity_ids)
    values = {}
    with RECORD_ELIGIBILITY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            activity_id = row["activity_id"]
            if activity_id not in wanted:
                continue
            if row["normalized_endpoint_semantics"] != "potency_candidate:IC50":
                raise RuntimeError(f"activity {activity_id} is not IC50")
            if row["primary_class"] != "exact_quantitative_potency_candidate":
                raise RuntimeError(f"activity {activity_id} is not exact quantitative")
            values[activity_id] = row

    missing = wanted - set(values)
    if missing:
        raise RuntimeError(
            "missing frozen record_eligibility rows for activities: "
            f"{join_sorted(missing)}"
        )
    return values


def enrich_records(
    represented: list[dict[str, str]],
    values: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    enriched = []
    for row in represented:
        value_row = values[row["source_id"]]
        enriched.append({
            **row,
            "activity_id": row["source_id"],
            "standard_type": value_row["standard_type"],
            "standard_relation": value_row["standard_relation"],
            "standard_value": value_row["standard_value"],
            "standard_units": value_row["standard_units"],
            "pchembl_value": value_row["pchembl_value"],
        })
    return enriched


def group_records(
    records: list[dict[str, str]],
    key_fields: tuple[str, str, str],
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped = defaultdict(list)
    for row in records:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    return dict(grouped)


def activity_signature(rows: list[dict[str, str]]) -> tuple[str, ...]:
    return tuple(sorted(row["activity_id"] for row in rows))


def group_summary_row(
    comparison_kind: str,
    key: tuple[str, str, str],
    rows: list[dict[str, str]],
) -> dict[str, str]:
    standard_ranges = range_fields([row["standard_value"] for row in rows])
    pchembl_ranges = range_fields([row["pchembl_value"] for row in rows])
    return {
        "comparison_kind": comparison_kind,
        "isoform": key[0],
        "assay_chembl_id": key[1],
        "group_key": key[2],
        "record_count": str(len(rows)),
        "activity_ids": join_sorted({row["activity_id"] for row in rows}),
        "molecule_chembl_ids": join_sorted({
            row["molecule_chembl_id"] for row in rows
        }),
        "full_standard_inchikeys": join_sorted({
            row["full_standard_inchikey"] for row in rows
        }),
        "standard_value_nM_min": standard_ranges["value_min"],
        "standard_value_nM_max": standard_ranges["value_max"],
        "standard_value_nM_spread": standard_ranges["value_spread"],
        "pchembl_value_min": pchembl_ranges["value_min"],
        "pchembl_value_max": pchembl_ranges["value_max"],
        "pchembl_value_spread": pchembl_ranges["value_spread"],
    }


def characterize() -> dict[str, object]:
    represented, unresolved = load_selected_representation_records()
    values = load_exact_values({row["source_id"] for row in represented + unresolved})
    records = enrich_records(represented, values)

    source_groups = group_records(
        records,
        ("isoform", "assay_chembl_id", "molecule_chembl_id"),
    )
    key_groups = group_records(
        records,
        ("isoform", "assay_chembl_id", "full_standard_inchikey"),
    )

    source_signatures = {
        (key[0], key[1], activity_signature(rows))
        for key, rows in source_groups.items()
    }
    key_signatures = {
        (key[0], key[1], activity_signature(rows))
        for key, rows in key_groups.items()
    }
    identical_groups = source_signatures & key_signatures

    affected_key_groups = {
        key: rows for key, rows in key_groups.items()
        if len({row["molecule_chembl_id"] for row in rows}) > 1
    }
    affected_source_groups = {
        key: rows for key, rows in source_groups.items()
        if len({row["full_standard_inchikey"] for row in rows}) > 1
    }

    affected_records_by_id = {}
    for rows in list(affected_key_groups.values()) + list(affected_source_groups.values()):
        for row in rows:
            affected_records_by_id[row["activity_id"]] = row

    affected_records = [
        affected_records_by_id[activity_id]
        for activity_id in sorted(affected_records_by_id)
    ]

    same_key_different_inchi = {
        "|".join(key): sorted({row["standard_inchi"] for row in rows})
        for key, rows in key_groups.items()
        if len({row["standard_inchi"] for row in rows}) > 1
    }

    affected_assays = sorted({
        (row["isoform"], row["assay_chembl_id"])
        for row in affected_records
    })

    summary = {
        "artifact_type":
            "CYP_selected_exact_IC50_replicate_grouping_delta_characterization_v1",
        "authority": "characterization_only_no_aggregation_or_routing_authority",
        "inputs": {
            "representation_audit": str(REPRESENTATION_AUDIT.relative_to(ROOT)),
            "record_eligibility": str(RECORD_ELIGIBILITY.relative_to(ROOT)),
            "representation_audit_sha256": sha256_file(REPRESENTATION_AUDIT),
            "record_eligibility_sha256": sha256_file(RECORD_ELIGIBILITY),
        },
        "grouping_definitions": {
            "old_source_grouping": [
                "isoform",
                "assay_chembl_id",
                "molecule_chembl_id",
            ],
            "proposed_qualified_identity_grouping": [
                "isoform",
                "assay_chembl_id",
                "full_standard_inchikey",
            ],
        },
        "non_aggregating_contract": {
            "median_used_as_modeling_label": False,
            "measurement_aggregation_performed": False,
            "routing_or_exclusion_performed": False,
            "task_removal_performed": False,
            "partition_changes_performed": False,
            "training_performed": False,
            "public_test_label_access": False,
        },
        "represented_exact_records_analyzed": len(records),
        "unresolved_records_kept_outside_represented_key_comparison": [
            {
                "activity_id": row["source_id"],
                "isoform": row["isoform"],
                "assay_chembl_id": row["assay_chembl_id"],
                "molecule_chembl_id": row["molecule_chembl_id"],
                "identity_state": row["identity_state"],
                "standard_value": values[row["source_id"]]["standard_value"],
                "standard_units": values[row["source_id"]]["standard_units"],
                "pchembl_value": values[row["source_id"]]["pchembl_value"],
            }
            for row in sorted(unresolved, key=lambda item: item["source_id"])
        ],
        "source_molecule_id_groups": len(source_groups),
        "full_inchikey_groups": len(key_groups),
        "groups_identical_under_both_definitions": len(identical_groups),
        "full_key_groups_containing_multiple_molecule_chembl_ids":
            len(affected_key_groups),
        "molecule_chembl_id_groups_mapping_to_multiple_full_keys":
            len(affected_source_groups),
        "affected_assay_task_count": len(affected_assays),
        "affected_assays": [
            {"isoform": isoform, "assay_chembl_id": assay_id}
            for isoform, assay_id in affected_assays
        ],
        "affected_activity_id_count": len(affected_records),
        "affected_activity_ids": [row["activity_id"] for row in affected_records],
        "same_key_different_standard_inchi_count": len(same_key_different_inchi),
        "same_key_different_standard_inchi": same_key_different_inchi,
    }

    return {
        "summary": summary,
        "records": records,
        "affected_key_groups": affected_key_groups,
        "affected_source_groups": affected_source_groups,
        "affected_records": affected_records,
    }


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


def write_outputs(result: dict[str, object]) -> dict[str, str]:
    summary = result["summary"]
    affected_key_groups = result["affected_key_groups"]
    affected_source_groups = result["affected_source_groups"]
    affected_records = result["affected_records"]

    summary_path = OUT / "replicate_grouping_delta_summary.json"
    key_groups_path = OUT / "replicate_grouping_delta_full_key_groups.tsv"
    source_groups_path = OUT / "replicate_grouping_delta_molecule_id_groups.tsv"
    records_path = OUT / "replicate_grouping_delta_affected_records.tsv"
    hashes_path = OUT / "replicate_grouping_delta_SHA256SUMS.txt"

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    group_fields = [
        "comparison_kind",
        "isoform",
        "assay_chembl_id",
        "group_key",
        "record_count",
        "activity_ids",
        "molecule_chembl_ids",
        "full_standard_inchikeys",
        "standard_value_nM_min",
        "standard_value_nM_max",
        "standard_value_nM_spread",
        "pchembl_value_min",
        "pchembl_value_max",
        "pchembl_value_spread",
    ]
    write_tsv(
        key_groups_path,
        group_fields,
        [
            group_summary_row("full_key_spans_multiple_molecule_ids", key, rows)
            for key, rows in sorted(affected_key_groups.items())
        ],
    )
    write_tsv(
        source_groups_path,
        group_fields,
        [
            group_summary_row("molecule_id_spans_multiple_full_keys", key, rows)
            for key, rows in sorted(affected_source_groups.items())
        ],
    )

    record_fields = [
        "activity_id",
        "isoform",
        "assay_chembl_id",
        "molecule_chembl_id",
        "full_standard_inchikey",
        "standard_inchi",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "source_smiles",
    ]
    write_tsv(
        records_path,
        record_fields,
        [
            {field: row[field] for field in record_fields}
            for row in affected_records
        ],
    )

    output_paths = [
        summary_path,
        key_groups_path,
        source_groups_path,
        records_path,
    ]
    with hashes_path.open("w", encoding="utf-8") as handle:
        for path in output_paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")

    return {
        path.name: sha256_file(path)
        for path in output_paths + [hashes_path]
    }


def main() -> None:
    result = characterize()
    hashes = write_outputs(result)
    summary = result["summary"]

    print("=== REPLICATE GROUPING DELTA CHARACTERIZATION ===")
    print(
        "represented exact records analyzed:",
        summary["represented_exact_records_analyzed"],
    )
    print("source molecule-ID groups:", summary["source_molecule_id_groups"])
    print("full-InChIKey groups:", summary["full_inchikey_groups"])
    print(
        "groups identical under both definitions:",
        summary["groups_identical_under_both_definitions"],
    )
    print(
        "full-key groups with >1 molecule_chembl_id:",
        summary["full_key_groups_containing_multiple_molecule_chembl_ids"],
    )
    print(
        "molecule_chembl_id groups mapping to >1 full key:",
        summary["molecule_chembl_id_groups_mapping_to_multiple_full_keys"],
    )
    print("affected assays/tasks:", summary["affected_assay_task_count"])
    print("affected activity IDs:", summary["affected_activity_id_count"])
    print(
        "same-key/different-InChI groups:",
        summary["same_key_different_standard_inchi_count"],
    )
    print("output hashes:", hashes)
    print("CHARACTERIZATION COMPLETE -- NO AGGREGATION PERFORMED")


if __name__ == "__main__":
    main()
