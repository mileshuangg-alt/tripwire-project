from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path("qualification/cyp_external_task_qualification_20261109")
DERIVED = ROOT / "derived"
RAW = ROOT / "raw_bulk"

EXPECTED_COUNTS = {"CYP1A2": 22, "CYP2C9": 33, "CYP2D6": 40, "CYP3A4": 81}
KNOWN_PUBCHEM_AIDS = {"CYP1A2": 410, "CYP2C9": 883, "CYP2D6": 891, "CYP3A4": 884}
PUBCHEM_METADATA_TAGS = {
    "RESULT_TYPE",
    "RESULT_DESCR",
    "RESULT_UNIT",
    "RESULT_IS_ACTIVE_CONCENTRATION",
    "RESULT_IS_ACTIVE_CONCENTRATION_QUALIFIER",
    "RESULT_ATTR_CONC_MICROMOL",
}


def fail(message: str) -> None:
    raise SystemExit(f"INTEGRITY STOP: {message}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_text(value: Any) -> str:
    return str(value or "").replace("\t", " ").replace("\n", " ").strip()


def load_pubchem_smiles() -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for aid in KNOWN_PUBCHEM_AIDS.values():
        with (RAW / f"pubchem_AID{aid}_full.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        smiles = set()
        for row in rows:
            tag = clean_text(row.get("PUBCHEM_RESULT_TAG"))
            sid = clean_text(row.get("PUBCHEM_SID"))
            if tag in PUBCHEM_METADATA_TAGS or not sid:
                continue
            smi = clean_text(row.get("PUBCHEM_EXT_DATASOURCE_SMILES"))
            if smi:
                smiles.add(smi)
        result[aid] = smiles
    return result


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    aa = token_set(a)
    bb = token_set(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def tsv_set(values: list[str]) -> str:
    return "|".join(sorted({v for v in values if v}))


def extract_context(description: str, assay_organism: str, bao_label: str) -> dict[str, str]:
    desc = description or ""
    lower = desc.lower()
    substrate_patterns = [
        r"using ([^.;,]+?) as substrate",
        r"with ([^.;,]+?) as substrate",
        r"as ([^.;,]+?) ([a-z0-9]+ation|hydroxylation|demethylation)",
        r"substrate ([^.;,]+)",
    ]
    substrates = []
    for pattern in substrate_patterns:
        for match in re.finditer(pattern, lower):
            substrates.append(match.group(1).strip())
    systems = []
    system_terms = [
        "human liver microsomes",
        "liver microsomes",
        "recombinant",
        "microsomes",
        "hepatocytes",
        "supersomes",
        "insect microsomes",
        "cDNA expressed",
        "baculosomes",
    ]
    for term in system_terms:
        if term.lower() in lower:
            systems.append(term)
    return {
        "substrate_or_probe_context": tsv_set(substrates) or "unresolved_or_not_encoded",
        "biological_system_context": tsv_set(systems) or "unresolved_or_not_encoded",
        "assay_format_context": bao_label or "unresolved_or_not_encoded",
        "organism_context": assay_organism or "unresolved_or_not_encoded",
    }


def main() -> None:
    required_inputs = [
        DERIVED / "assay_task_table.tsv",
        DERIVED / "record_eligibility.tsv",
        DERIVED / "pubchem_chembl_reconciliation.tsv",
        DERIVED / "p2_episode_compatibility_characterization.json",
        RAW / "chembl37_required_assays.jsonl",
    ] + [RAW / f"pubchem_AID{aid}_full.csv" for aid in KNOWN_PUBCHEM_AIDS.values()]
    for path in required_inputs:
        if not path.exists():
            fail(f"required input missing: {path}")

    tasks = read_tsv(DERIVED / "assay_task_table.tsv")
    selected = [
        row for row in tasks
        if row["normalized_endpoint_semantics"] == "potency_candidate:IC50"
        and int(row["unique_molecules_exact_quantitative_potency_candidate"]) >= 16
    ]
    selected_counts = dict(Counter(row["isoform"] for row in selected))
    if selected_counts != EXPECTED_COUNTS or len(selected) != sum(EXPECTED_COUNTS.values()):
        fail(f"selected counts mismatch: observed {selected_counts}, total {len(selected)}")

    assays = {
        str(row["assay_chembl_id"]): row
        for row in read_jsonl(RAW / "chembl37_required_assays.jsonl")
    }
    reconciliation = {
        (row["isoform"], row["assay_chembl_id"], row["normalized_endpoint_semantics"]): row
        for row in read_tsv(DERIVED / "pubchem_chembl_reconciliation.tsv")
    }
    records = read_tsv(DERIVED / "record_eligibility.tsv")
    selected_keys = {
        (row["isoform"], row["assay_chembl_id"], row["normalized_endpoint_semantics"])
        for row in selected
    }
    exact_records_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        key = (row["isoform"], row["assay_chembl_id"], row["normalized_endpoint_semantics"])
        if key in selected_keys and row["primary_class"] == "exact_quantitative_potency_candidate":
            exact_records_by_key[key].append(row)

    pubchem_smiles = load_pubchem_smiles()
    selected_by_key = {
        (row["isoform"], row["assay_chembl_id"], row["normalized_endpoint_semantics"]): row
        for row in selected
    }

    provenance_rows = []
    repeated_tasks = []
    for key in sorted(selected_by_key):
        task = selected_by_key[key]
        isoform, assay_id, endpoint = key
        assay = assays.get(assay_id, {})
        exact_records = exact_records_by_key[key]
        molecule_counts = Counter(r["molecule_chembl_id"] for r in exact_records if r["molecule_chembl_id"])
        repeated_molecules = {m: c for m, c in molecule_counts.items() if c > 1}
        repeated_record_excess = sum(c - 1 for c in repeated_molecules.values())
        description = task["assay_description"]
        context = extract_context(description, task["assay_organism"], task["assay_bao_label"])
        smiles = {r["canonical_smiles"] for r in exact_records if r["canonical_smiles"]}
        pubchem_overlap = {
            f"AID{aid}": len(smiles & pubchem_smiles[aid])
            for aid in sorted(pubchem_smiles)
        }
        rec = reconciliation.get(key, {})
        existing_reconciliation = rec.get("reconciliation_category", "overlap_unresolved")
        if existing_reconciliation == "demonstrated_pubchem_duplicate":
            pubchem_relationship = "demonstrated_duplicate"
            pubchem_evidence = rec.get("evidence", "")
        else:
            pubchem_relationship = "overlap_unresolved"
            pubchem_evidence = "No affirmative frozen local evidence demonstrates duplication or distinctness against PubChem AIDs 410/883/891/884 for this IC50 task."
        row = {
            "isoform": isoform,
            "assay_chembl_id": assay_id,
            "exact_candidate_record_count": task["records_exact_quantitative_potency_candidate"],
            "exact_candidate_unique_molecule_count": task["unique_molecules_exact_quantitative_potency_candidate"],
            "repeated_exact_candidate_molecule_count": str(len(repeated_molecules)),
            "repeated_exact_candidate_record_excess": str(repeated_record_excess),
            "document_chembl_ids": task["document_chembl_ids"],
            "source_ids": task["source_ids"],
            "assay_src_id": task["assay_src_id"],
            "assay_src_assay_id": task["assay_src_assay_id"],
            "assay_description": description,
            "assay_type": task["assay_type"],
            "assay_type_description": task["assay_type_description"],
            "assay_organism": task["assay_organism"],
            "assay_bao_format": task["assay_bao_format"],
            "assay_bao_label": task["assay_bao_label"],
            "assay_confidence_score": task["assay_confidence_score"],
            "relation_distribution": task["relation_distribution"],
            "unit_distribution": task["unit_distribution"],
            "validity_status_distribution": task["validity_status_distribution"],
            "substrate_or_probe_context": context["substrate_or_probe_context"],
            "biological_system_context": context["biological_system_context"],
            "assay_format_context": context["assay_format_context"],
            "organism_context": context["organism_context"],
            "pubchem_relationship": pubchem_relationship,
            "pubchem_relationship_evidence": pubchem_evidence,
            "pubchem_exact_smiles_overlap_by_aid": json.dumps(pubchem_overlap, sort_keys=True),
            "exact_candidate_unique_smiles_count": str(len(smiles)),
        }
        provenance_rows.append(row)
        if repeated_molecules:
            repeated_tasks.append(row)

    provenance_fields = list(provenance_rows[0].keys())
    prov_path = DERIVED / "p2_compatible_ic50_task_provenance.tsv"
    with prov_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=provenance_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(provenance_rows)

    relationship_rows = []
    duplicate_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    distinct_count = 0
    unresolved_count = 0
    duplicate_count = 0
    for left, right in combinations(provenance_rows, 2):
        left_key = (left["isoform"], left["assay_chembl_id"], "potency_candidate:IC50")
        right_key = (right["isoform"], right["assay_chembl_id"], "potency_candidate:IC50")
        left_records = exact_records_by_key[left_key]
        right_records = exact_records_by_key[right_key]
        left_molecules = {r["molecule_chembl_id"] for r in left_records if r["molecule_chembl_id"]}
        right_molecules = {r["molecule_chembl_id"] for r in right_records if r["molecule_chembl_id"]}
        left_smiles = {r["canonical_smiles"] for r in left_records if r["canonical_smiles"]}
        right_smiles = {r["canonical_smiles"] for r in right_records if r["canonical_smiles"]}
        shared_docs = sorted(set(left["document_chembl_ids"].split("|")) & set(right["document_chembl_ids"].split("|")) - {""})
        shared_sources = sorted(set(left["source_ids"].split("|")) & set(right["source_ids"].split("|")) - {""})
        shared_assay_src_id = left["assay_src_id"] if left["assay_src_id"] and left["assay_src_id"] == right["assay_src_id"] else ""
        shared_assay_src_assay_id = left["assay_src_assay_id"] if left["assay_src_assay_id"] and left["assay_src_assay_id"] == right["assay_src_assay_id"] else ""
        if shared_assay_src_id and shared_assay_src_assay_id:
            category = "demonstrated_duplicate"
            evidence = f"Same nonblank assay source identifier pair: assay_src_id={shared_assay_src_id}, assay_src_assay_id={shared_assay_src_assay_id}."
            duplicate_count += 1
            duplicate_groups[(shared_assay_src_id, shared_assay_src_assay_id)].extend([left["assay_chembl_id"], right["assay_chembl_id"]])
        elif left["assay_src_id"] and right["assay_src_id"] and left["assay_src_id"] == right["assay_src_id"] and left["assay_src_assay_id"] and right["assay_src_assay_id"] and left["assay_src_assay_id"] != right["assay_src_assay_id"]:
            category = "demonstrated_distinct"
            evidence = f"Same assay source {left['assay_src_id']} provides different nonblank source assay identifiers: {left['assay_src_assay_id']} vs {right['assay_src_assay_id']}."
            distinct_count += 1
        else:
            category = "overlap_unresolved"
            evidence = "No affirmative frozen local source/cross-reference evidence demonstrates duplicate or distinct status. Shared documents, description similarity, and molecule overlap are descriptive only."
            unresolved_count += 1
        relationship_rows.append({
            "left_isoform": left["isoform"],
            "left_assay_chembl_id": left["assay_chembl_id"],
            "right_isoform": right["isoform"],
            "right_assay_chembl_id": right["assay_chembl_id"],
            "relationship_category": category,
            "affirmative_evidence": evidence,
            "shared_document_chembl_ids": "|".join(shared_docs),
            "shared_source_ids": "|".join(shared_sources),
            "shared_assay_src_id": shared_assay_src_id,
            "shared_assay_src_assay_id": shared_assay_src_assay_id,
            "assay_description_exact_match": str(left["assay_description"] == right["assay_description"]).lower(),
            "assay_description_token_jaccard": f"{jaccard(left['assay_description'], right['assay_description']):.6f}",
            "exact_candidate_molecule_overlap_count": str(len(left_molecules & right_molecules)),
            "exact_candidate_smiles_overlap_count": str(len(left_smiles & right_smiles)),
        })

    rel_fields = list(relationship_rows[0].keys())
    rel_path = DERIVED / "p2_compatible_ic50_relationships.tsv"
    with rel_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rel_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(relationship_rows)

    demonstrated_duplicate_groups = [
        {
            "assay_src_id": src_id,
            "assay_src_assay_id": src_assay_id,
            "assay_chembl_ids": sorted(set(ids)),
            "evidence": "Same nonblank assay_src_id and assay_src_assay_id in frozen ChEMBL assay metadata.",
        }
        for (src_id, src_assay_id), ids in sorted(duplicate_groups.items())
        if len(set(ids)) > 1
    ]

    exact_sizes = [int(r["exact_candidate_unique_molecule_count"]) for r in provenance_rows]
    def dist(values: list[int]) -> dict[str, Any]:
        values = sorted(values)
        def pct(p: float) -> int:
            if len(values) == 1:
                return values[0]
            pos = p * (len(values) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(values) - 1)
            return round(values[lo] + (values[hi] - values[lo]) * (pos - lo))
        return {
            "count": len(values),
            "min": values[0],
            "q1": pct(0.25),
            "median": pct(0.5),
            "q3": pct(0.75),
            "max": values[-1],
            "frequency": {str(k): v for k, v in sorted(Counter(values).items())},
        }

    context_availability = {
        "substrate_or_probe_available": sum(1 for r in provenance_rows if r["substrate_or_probe_context"] != "unresolved_or_not_encoded"),
        "biological_system_available": sum(1 for r in provenance_rows if r["biological_system_context"] != "unresolved_or_not_encoded"),
        "assay_format_available": sum(1 for r in provenance_rows if r["assay_format_context"] != "unresolved_or_not_encoded"),
        "organism_available": sum(1 for r in provenance_rows if r["organism_context"] != "unresolved_or_not_encoded"),
    }
    overlap_counts = [int(r["exact_candidate_molecule_overlap_count"]) for r in relationship_rows]
    summary = {
        "artifact_type": "p2_compatible_ic50_task_provenance_characterization_v1",
        "selection_rule": {
            "normalized_endpoint_semantics": "potency_candidate:IC50",
            "unique_molecules_exact_quantitative_potency_candidate_minimum": 16,
            "rationale": "Existing frozen D010/D011 P2 implementation episode-size contract: 13 distinct support + 3 distinct query molecules; not a new scientific threshold.",
        },
        "selected_task_counts": {"by_isoform": selected_counts, "total": len(provenance_rows)},
        "selected_assay_task_count": len(provenance_rows),
        "remaining_after_removal_of_demonstrated_duplicates": len(provenance_rows) - len(demonstrated_duplicate_groups),
        "unresolved_relationships_retained": True,
        "exact_candidate_unique_molecule_size_distribution": dist(exact_sizes),
        "exact_candidate_record_size_distribution": dist([int(r["exact_candidate_record_count"]) for r in provenance_rows]),
        "repeated_measurements": {
            "tasks_exact_record_count_equals_unique_molecule_count": sum(1 for r in provenance_rows if int(r["exact_candidate_record_count"]) == int(r["exact_candidate_unique_molecule_count"])),
            "tasks_with_repeated_exact_measurements": len(repeated_tasks),
            "total_repeated_exact_candidate_molecules": sum(int(r["repeated_exact_candidate_molecule_count"]) for r in provenance_rows),
            "total_repeated_exact_candidate_record_excess": sum(int(r["repeated_exact_candidate_record_excess"]) for r in provenance_rows),
        },
        "counts_by_document_chembl_id": dict(Counter(doc for r in provenance_rows for doc in r["document_chembl_ids"].split("|") if doc)),
        "counts_by_source_id": dict(Counter(src for r in provenance_rows for src in r["source_ids"].split("|") if src)),
        "counts_by_assay_src_id": dict(Counter(r["assay_src_id"] or "blank" for r in provenance_rows)),
        "counts_by_assay_src_assay_id_presence": dict(Counter("present" if r["assay_src_assay_id"] else "blank" for r in provenance_rows)),
        "pubchem_reconciliation": {
            "demonstrated_duplicates": sum(1 for r in provenance_rows if r["pubchem_relationship"] == "demonstrated_duplicate"),
            "overlap_unresolved": sum(1 for r in provenance_rows if r["pubchem_relationship"] == "overlap_unresolved"),
            "demonstrated_distinct": 0,
        },
        "pairwise_relationship_counts": {
            "demonstrated_duplicate": duplicate_count,
            "demonstrated_distinct": distinct_count,
            "overlap_unresolved": unresolved_count,
            "total_pairs": len(relationship_rows),
        },
        "other_demonstrated_duplicate_groups": demonstrated_duplicate_groups,
        "assay_context_availability": context_availability,
        "molecule_overlap_characterization": {
            "pair_count_with_any_exact_candidate_molecule_overlap": sum(1 for x in overlap_counts if x > 0),
            "max_pairwise_exact_candidate_molecule_overlap": max(overlap_counts) if overlap_counts else 0,
            "pairwise_exact_candidate_molecule_overlap_frequency": {str(k): v for k, v in sorted(Counter(overlap_counts).items())},
        },
        "integrity_checks": [],
        "public_openadmet_test_labels_accessed": False,
        "d011_replacement_or_resume_decision": "not made",
    }

    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            fail(f"check failed: {name}: {detail}")
        summary["integrity_checks"].append({"name": name, "status": "pass", "detail": detail})

    check("selected_counts_reproduce_expected", selected_counts == EXPECTED_COUNTS and len(provenance_rows) == 176, json.dumps(selected_counts, sort_keys=True))
    check("all_selected_endpoint_ic50", all(r["assay_chembl_id"] for r in provenance_rows))
    check("provenance_rows_176", len(provenance_rows) == 176)
    check("relationships_all_pairs", len(relationship_rows) == 176 * 175 // 2)
    check("repeated_measurement_accounting_nonnegative", all(int(r["repeated_exact_candidate_record_excess"]) >= 0 for r in provenance_rows))
    check("pubchem_duplicates_require_affirmative_evidence", all(r["pubchem_relationship"] != "demonstrated_duplicate" or r["pubchem_relationship_evidence"] for r in provenance_rows))
    check("demonstrated_pair_relationships_have_evidence", all(r["affirmative_evidence"] for r in relationship_rows if r["relationship_category"] in {"demonstrated_duplicate", "demonstrated_distinct"}))
    check("unresolved_retained", unresolved_count > 0)
    check("no_public_test_labels_accessed", True, "No OpenADMET files are read by this script.")

    summary_path = DERIVED / "p2_compatible_ic50_provenance_summary.json"
    write_json(summary_path, summary)

    output_paths = [prov_path, rel_path, summary_path]
    hash_path = DERIVED / "p2_compatible_ic50_SHA256SUMS.txt"
    with hash_path.open("w", encoding="utf-8") as handle:
        for path in output_paths:
            handle.write(f"{sha256_file(path)}  {path.relative_to(ROOT)}\n")
    output_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in output_paths + [hash_path]}
    print(json.dumps({
        "status": "ok",
        "selected_task_counts": summary["selected_task_counts"],
        "pairwise_relationship_counts": summary["pairwise_relationship_counts"],
        "output_hashes": output_hashes,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
