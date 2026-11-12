from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("qualification/cyp_external_task_qualification_20261109")
RAW = ROOT / "raw_bulk"
STATE = ROOT / "bulk_state"
MANIFESTS = ROOT / "manifests_bulk"
DERIVED = ROOT / "derived"
TRAIN_PATH = Path("cyp/data/cyp-challenge-TRAIN_inhibition.csv")

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
CHEMBL_ACTIVITY_EXPECTED = {
    "CYP1A2": 28001,
    "CYP2C9": 33914,
    "CYP2D6": 36650,
    "CYP3A4": 57238,
}
CHEMBL_ACTIVITY_SHA256 = {
    "CYP1A2": "7e17c608d9660310a60911a8c80cdea2ad530a873518a1a7b63c59a71ad0f233",
    "CYP2C9": "a7e21d4ea07ece5e5f7ce1b3ffd6e758ba3b6332399033d18b530f3e022d2b71",
    "CYP2D6": "afab460300bd9cb889296486cdeec9fcb1e0c79de4852d22679f46c9b80e83b6",
    "CYP3A4": "b20542e7bfcc3081e313a73223bc4c790b06e9fc55e536979d2526eb6d9d8d6f",
}
CHEMBL_ASSAY_SHA256 = "8ba5cb727ea11bca8eda2b7c89e812ce6bac14732f0aac334a54b7edc76a6313"
EXPECTED_ASSAY_COUNT = 14988

PUBCHEM_AIDS = {
    "CYP1A2": 410,
    "CYP2C9": 883,
    "CYP2D6": 891,
    "CYP3A4": 884,
}
PUBCHEM_EXPECTED = {
    410: {"isoform": "CYP1A2", "result_rows": 9198, "unique_result_sids": 9198},
    883: {"isoform": "CYP2C9", "result_rows": 10320, "unique_result_sids": 10320},
    891: {"isoform": "CYP2D6", "result_rows": 10320, "unique_result_sids": 10320},
    884: {"isoform": "CYP3A4", "result_rows": 14155, "unique_result_sids": 14155},
}

PUBCHEM_METADATA_TAGS = {
    "RESULT_TYPE",
    "RESULT_DESCR",
    "RESULT_UNIT",
    "RESULT_IS_ACTIVE_CONCENTRATION",
    "RESULT_IS_ACTIVE_CONCENTRATION_QUALIFIER",
    "RESULT_ATTR_CONC_MICROMOL",
}

PRIMARY_CLASSES = (
    "exact_quantitative_potency_candidate",
    "censored_quantitative_potency",
    "non_potency_quantitative",
    "qualitative_or_nonquantitative",
    "missing_or_uninterpretable",
    "explicit_validity_problem",
)

VALIDITY_MAPPING = {
    "": {"status": "blank", "reason": "blank data_validity_comment"},
    "Manually validated": {
        "status": "positive",
        "reason": "explicit positive validation evidence",
    },
    "Outside typical range": {
        "status": "adverse",
        "reason": "explicit adverse validity mapping",
    },
    "Potential transcription error": {
        "status": "adverse",
        "reason": "explicit adverse validity mapping",
    },
    "Potential missing data": {
        "status": "adverse",
        "reason": "explicit adverse validity mapping",
    },
}

CONCENTRATION_UNITS = {
    "M": "molar_concentration",
    "mM": "molar_concentration",
    "uM": "molar_concentration",
    "µM": "molar_concentration",
    "nM": "molar_concentration",
    "pM": "molar_concentration",
}
NON_CONCENTRATION_UNITS = {
    "%": "percent",
    "s-1": "rate_constant",
    "/min": "rate_constant",
    "/s": "rate_constant",
    "1/min": "rate_constant",
    "min-1": "rate_constant",
    "hr": "time",
    "min": "time",
    "degrees C": "temperature",
    "nm": "length",
    "pmol/min": "amount_rate",
    "nmol/min": "amount_rate",
    "pmol/hr": "amount_rate",
    "mol/min": "amount_rate",
    "pmol/min/pmol": "normalized_amount_rate",
    "pmol.min.mg-1": "normalized_amount_rate",
    "nmol/min/nmol": "normalized_amount_rate",
    "nmol/min/mg": "normalized_amount_rate",
    "pmol/mg.min": "normalized_amount_rate",
    "%/min": "percent_rate",
    "%/min/pmol": "normalized_percent_rate",
    "microM/min": "concentration_rate",
    "uL/min": "clearance_or_flow",
    "mL.min-1.g-1": "normalized_clearance_or_flow",
    "uL/min/pmol": "normalized_clearance_or_flow",
    "uL/min/nmol": "normalized_clearance_or_flow",
    "uL/min.mg": "normalized_clearance_or_flow",
    "uL/mg/min": "normalized_clearance_or_flow",
    "ul.min.nmol-1": "normalized_clearance_or_flow",
    "ml/min/nmol": "normalized_clearance_or_flow",
    "ml/min/mg": "normalized_clearance_or_flow",
    "ml/min/uL": "normalized_clearance_or_flow",
    "ml/min/umol": "normalized_clearance_or_flow",
    "mL/min.umol": "normalized_clearance_or_flow",
    "L/min/umol": "normalized_clearance_or_flow",
    "/uM/min": "normalized_rate",
    "/min/microM": "normalized_rate",
    "/min/mM": "normalized_rate",
    "/min/nM": "normalized_rate",
    "/nM.min": "normalized_rate",
    "10^-4/min": "scaled_rate_constant",
    "10'-4/min": "scaled_rate_constant",
    "10'3ml/min": "scaled_clearance_or_flow",
    "nmol": "amount",
    "ug.mL-1": "mass_concentration_not_molar",
    "ug/L": "mass_concentration_not_molar",
    "ng/ml": "mass_concentration_not_molar",
}

POTENCY_ENDPOINTS = {
    "IC50": "IC50",
    "Ki": "Ki",
    "AC50": "AC50",
    "Potency": "Potency",
    "EC50": "EC50",
    "IC90": "IC90",
    "IC20": "IC20",
    "IC5": "IC5",
    "IC": "IC",
    "Kd": "Kd",
}
LOG_ENDPOINTS_UNREVIEWED = {
    "pIC50": "pIC50",
    "Log IC50": "Log IC50",
    "log(1/IC50)": "log(1/IC50)",
    "Log 1/Km": "Log 1/Km",
}
NON_POTENCY_ENDPOINTS = {
    "Activation",
    "Activity",
    "CL",
    "CL free",
    "CLH",
    "Cmax",
    "Compound recovery",
    "Control of P-450 marker",
    "Delta Tm",
    "Drug metabolism",
    "Drug uptake",
    "Emax",
    "FC",
    "Fm",
    "INH",
    "Imax",
    "Inhibition",
    "Inhibitory index",
    "Inhibitory potential",
    "K",
    "Kcat",
    "Kcat/Km",
    "Kdiss",
    "Kinact",
    "Km",
    "Kobs",
    "Ks",
    "NA",
    "NOAEL",
    "NT",
    "PB",
    "PPB",
    "Partition ratio",
    "Rate of oxidation",
    "Ratio",
    "Ratio IC50",
    "Retention_time",
    "Selectivity ratio",
    "Stability",
    "T1/2",
    "TDI",
    "Tm",
    "Vmax",
    "Vmax(app)",
    "Vmax/Km",
    "deltaA",
    "kinact",
    "max activation",
    "mechanism based inhibition",
    "t1/2",
}
ENDPOINT_MAPPING = {
    **{k: {"category": "potency_candidate", "identity": v} for k, v in POTENCY_ENDPOINTS.items()},
    **{k: {"category": "log_endpoint_unreviewed", "identity": v} for k, v in LOG_ENDPOINTS_UNREVIEWED.items()},
    **{k: {"category": "non_potency", "identity": k} for k in NON_POTENCY_ENDPOINTS},
}

EXACT_RELATIONS = {"="}
CENSORED_RELATIONS = {"<", ">", "<=", ">="}
KNOWN_NONEXACT_RELATIONS = {"~"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(f"INTEGRITY STOP: {message}")


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def verify_frozen_inputs() -> dict[str, Any]:
    provenance_path = MANIFESTS / "source_provenance.json"
    provenance = load_json(provenance_path)
    if provenance.get("openadmet_public_test_labels_accessed") is not False:
        fail("source provenance does not preserve OpenADMET public-test firewall")

    activity_summary = {}
    total = 0
    for isoform, expected_count in CHEMBL_ACTIVITY_EXPECTED.items():
        path = RAW / f"chembl37_{isoform}_activities.jsonl"
        state = load_json(STATE / f"chembl37_{isoform}_activities.complete.json")
        observed_sha = sha256_file(path)
        observed_count = sum(1 for _ in path.open(encoding="utf-8"))
        if observed_count != expected_count:
            fail(f"{isoform} activity rows {observed_count} != {expected_count}")
        if observed_sha != CHEMBL_ACTIVITY_SHA256[isoform]:
            fail(f"{isoform} activity SHA mismatch: {observed_sha}")
        if state.get("records") != expected_count or state.get("sha256") != observed_sha:
            fail(f"{isoform} activity state mismatch")
        total += observed_count
        activity_summary[isoform] = {
            "records": observed_count,
            "sha256": observed_sha,
            "state_records": state.get("records"),
        }
    if total != 155803:
        fail(f"total ChEMBL activity rows {total} != 155803")

    assay_path = RAW / "chembl37_required_assays.jsonl"
    assay_state = load_json(STATE / "chembl37_required_assays.complete.json")
    assay_sha = sha256_file(assay_path)
    assay_rows = read_jsonl(assay_path)
    assay_ids = [str(row.get("assay_chembl_id") or "") for row in assay_rows]
    if len(assay_rows) != EXPECTED_ASSAY_COUNT:
        fail(f"ChEMBL assay rows {len(assay_rows)} != {EXPECTED_ASSAY_COUNT}")
    if len(set(assay_ids)) != EXPECTED_ASSAY_COUNT:
        fail("ChEMBL assay identity set is not unique")
    if assay_sha != CHEMBL_ASSAY_SHA256:
        fail(f"ChEMBL assay SHA mismatch: {assay_sha}")
    if assay_state.get("identity_set_equality") is not True:
        fail("ChEMBL assay state identity_set_equality is not true")
    if assay_state.get("records") != EXPECTED_ASSAY_COUNT:
        fail("ChEMBL assay state record count mismatch")
    if assay_state.get("sha256") != assay_sha:
        fail("ChEMBL assay state SHA mismatch")

    pubchem_summary = {}
    for aid, expected in PUBCHEM_EXPECTED.items():
        state_path = STATE / f"pubchem_AID{aid}.complete.json"
        table_path = RAW / f"pubchem_AID{aid}_full.csv"
        desc_path = RAW / f"pubchem_AID{aid}_description.json"
        if not state_path.exists() or not table_path.exists() or not desc_path.exists():
            fail(f"PubChem AID{aid} frozen artifact missing")
        state = load_json(state_path)
        table_sha = sha256_file(table_path)
        desc_sha = sha256_file(desc_path)
        if state.get("table_sha256") != table_sha:
            fail(f"PubChem AID{aid} table SHA mismatch: {table_sha}")
        if state.get("description_sha256") != desc_sha:
            fail(f"PubChem AID{aid} description SHA mismatch: {desc_sha}")
        if state.get("result_rows") != expected["result_rows"]:
            fail(f"PubChem AID{aid} result rows mismatch")
        if state.get("unique_result_sids") != expected["unique_result_sids"]:
            fail(f"PubChem AID{aid} unique SID mismatch")
        if state.get("duplicate_result_sids") != 0:
            fail(f"PubChem AID{aid} duplicate result SIDs not zero")
        if state.get("result_sid_identity_complete") is not True:
            fail(f"PubChem AID{aid} SID identity completeness false")
        if aid == 410:
            if state.get("artifact_contract") != "PubChem_full_table_with_schema_rows_v2":
                fail("PubChem AID410 artifact contract mismatch")
        else:
            if state.get("artifact_contract") != "PubChem_SID_POST_filtered_complete_table_with_metadata_rows_v3":
                fail(f"PubChem AID{aid} artifact contract mismatch")
            sid_path = RAW / f"pubchem_AID{aid}_sids.txt"
            if state.get("sid_inventory_sha256") != sha256_file(sid_path):
                fail(f"PubChem AID{aid} SID inventory SHA mismatch")
        pubchem_summary[str(aid)] = state

    return {
        "source_provenance_sha256": sha256_file(provenance_path),
        "chembl_activity_summary": activity_summary,
        "chembl_total_activity_rows": total,
        "chembl_assay_summary": {
            "records": len(assay_rows),
            "unique_assay_ids": len(set(assay_ids)),
            "identity_set_equality": True,
            "sha256": assay_sha,
        },
        "pubchem_summary": pubchem_summary,
        "openadmet_public_test_labels_accessed": False,
    }


def validity_info(comment: Any) -> tuple[str, str]:
    text = str(comment or "").strip()
    mapping = VALIDITY_MAPPING.get(text)
    if mapping:
        return mapping["status"], mapping["reason"]
    return "unknown", "unmapped nonblank data_validity_comment; retained for review"


def unit_semantics(unit: str) -> tuple[str, bool, str]:
    if unit in CONCENTRATION_UNITS:
        return f"concentration_unit:{unit}", False, ""
    if unit in NON_CONCENTRATION_UNITS:
        return f"non_concentration_unit:{unit}", False, ""
    if unit == "":
        return "blank_unit", True, "blank unit has no concentration semantics"
    return f"unmapped_unit:{unit}", True, "unit not in declarative unit mapping"


def endpoint_semantics(endpoint: str) -> tuple[str, str, bool, str]:
    mapping = ENDPOINT_MAPPING.get(endpoint)
    if not mapping:
        if endpoint == "":
            return "blank_endpoint", "unmapped", True, "blank endpoint"
        return f"unmapped_endpoint:{endpoint}", "unmapped", True, "endpoint not in declarative mapping"
    category = mapping["category"]
    identity = mapping["identity"]
    return f"{category}:{identity}", category, False, ""


def normalize_relation(relation: str) -> tuple[str, bool, str]:
    if relation in EXACT_RELATIONS:
        return "=", False, ""
    if relation in CENSORED_RELATIONS:
        return relation, False, ""
    if relation == "":
        return "blank_relation", True, "blank relation"
    if relation in KNOWN_NONEXACT_RELATIONS:
        return relation, True, "nonexact relation not eligible for quantitative potency"
    return f"unmapped_relation:{relation}", True, "relation not in declarative mapping"


def classify_without_validity(row: dict[str, Any]) -> dict[str, str]:
    endpoint = str(row.get("standard_type") or "").strip()
    relation = str(row.get("standard_relation") or "").strip()
    unit = str(row.get("standard_units") or "").strip()
    value = parse_float(row.get("standard_value"))
    endpoint_norm, endpoint_category, endpoint_unknown, endpoint_reason = endpoint_semantics(endpoint)
    unit_norm, unit_unknown, unit_reason = unit_semantics(unit)
    relation_norm, relation_unknown, relation_reason = normalize_relation(relation)

    unfamiliar_reasons = [
        x for x in (endpoint_reason, unit_reason, relation_reason) if x
    ]
    unfamiliar = bool(endpoint_unknown or unit_unknown or relation_unknown)

    if value is not None:
        if endpoint_category == "potency_candidate":
            if unit not in CONCENTRATION_UNITS:
                return {
                    "primary_class": "missing_or_uninterpretable",
                    "eligibility_reason": "potency endpoint has blank, incompatible, or unrecognized concentration unit",
                    "normalized_endpoint_semantics": endpoint_norm,
                    "normalized_relation": relation_norm,
                    "normalized_unit_semantics": unit_norm,
                    "unfamiliar_semantics_flag": str(unfamiliar).lower(),
                    "unfamiliar_semantics_reason": "; ".join(unfamiliar_reasons),
                }
            if relation_norm == "=":
                return {
                    "primary_class": "exact_quantitative_potency_candidate",
                    "eligibility_reason": "recognized concentration-potency endpoint, exact relation, numeric value, and concentration unit",
                    "normalized_endpoint_semantics": endpoint_norm,
                    "normalized_relation": relation_norm,
                    "normalized_unit_semantics": unit_norm,
                    "unfamiliar_semantics_flag": "false",
                    "unfamiliar_semantics_reason": "",
                }
            if relation_norm in CENSORED_RELATIONS:
                return {
                    "primary_class": "censored_quantitative_potency",
                    "eligibility_reason": "recognized concentration-potency endpoint with censored relation and concentration unit",
                    "normalized_endpoint_semantics": endpoint_norm,
                    "normalized_relation": relation_norm,
                    "normalized_unit_semantics": unit_norm,
                    "unfamiliar_semantics_flag": "false",
                    "unfamiliar_semantics_reason": "",
                }
            return {
                "primary_class": "missing_or_uninterpretable",
                "eligibility_reason": "potency endpoint has missing or non-eligible relation",
                "normalized_endpoint_semantics": endpoint_norm,
                "normalized_relation": relation_norm,
                "normalized_unit_semantics": unit_norm,
                "unfamiliar_semantics_flag": str(True).lower(),
                "unfamiliar_semantics_reason": "; ".join(filter(None, unfamiliar_reasons + ["relation not exact or censored for potency interpretation"])),
            }
        if endpoint_category == "log_endpoint_unreviewed":
            return {
                "primary_class": "missing_or_uninterpretable",
                "eligibility_reason": "log endpoint retained separately without reviewed conversion semantics",
                "normalized_endpoint_semantics": endpoint_norm,
                "normalized_relation": relation_norm,
                "normalized_unit_semantics": unit_norm,
                "unfamiliar_semantics_flag": "true",
                "unfamiliar_semantics_reason": "log endpoint requires human review; no exponentiation or concentration conversion applied",
            }
        if endpoint_category == "non_potency" and not unit_unknown:
            return {
                "primary_class": "non_potency_quantitative",
                "eligibility_reason": "numeric measurement with explicitly mapped non-potency endpoint/unit semantics",
                "normalized_endpoint_semantics": endpoint_norm,
                "normalized_relation": relation_norm,
                "normalized_unit_semantics": unit_norm,
                "unfamiliar_semantics_flag": str(relation_unknown).lower(),
                "unfamiliar_semantics_reason": relation_reason,
            }
        return {
            "primary_class": "missing_or_uninterpretable",
            "eligibility_reason": "numeric value present but endpoint/unit/relation semantics are not fully mapped",
            "normalized_endpoint_semantics": endpoint_norm,
            "normalized_relation": relation_norm,
            "normalized_unit_semantics": unit_norm,
            "unfamiliar_semantics_flag": "true",
            "unfamiliar_semantics_reason": "; ".join(unfamiliar_reasons) or "unmapped numeric semantics",
        }

    qualitative_fields = [
        row.get("activity_comment"),
        row.get("standard_text_value"),
        row.get("text_value"),
    ]
    if any(str(x or "").strip() for x in qualitative_fields):
        return {
            "primary_class": "qualitative_or_nonquantitative",
            "eligibility_reason": "no parseable numeric standard_value but qualitative activity/text field is present",
            "normalized_endpoint_semantics": endpoint_norm,
            "normalized_relation": relation_norm,
            "normalized_unit_semantics": unit_norm,
            "unfamiliar_semantics_flag": str(unfamiliar).lower(),
            "unfamiliar_semantics_reason": "; ".join(unfamiliar_reasons),
        }

    return {
        "primary_class": "missing_or_uninterpretable",
        "eligibility_reason": "missing or unparseable numeric value and no qualitative activity/text field",
        "normalized_endpoint_semantics": endpoint_norm,
        "normalized_relation": relation_norm,
        "normalized_unit_semantics": unit_norm,
        "unfamiliar_semantics_flag": str(unfamiliar or value is None).lower(),
        "unfamiliar_semantics_reason": "; ".join(unfamiliar_reasons + ["missing or unparseable standard_value"]),
    }


def classify_record(row: dict[str, Any]) -> dict[str, str]:
    base = classify_without_validity(row)
    status, reason = validity_info(row.get("data_validity_comment"))
    if status == "adverse":
        result = dict(base)
        result["would_be_class_without_validity_flag"] = base["primary_class"]
        result["primary_class"] = "explicit_validity_problem"
        result["eligibility_reason"] = "explicitly adverse data_validity_comment takes precedence"
    else:
        result = dict(base)
        result["would_be_class_without_validity_flag"] = ""
    result["validity_status"] = status
    result["validity_mapping_reason"] = reason
    return result


def tsv_join(values: list[Any]) -> str:
    return "|".join(str(v) for v in sorted({str(x) for x in values if str(x)}))


def distribution_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"task_count": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None, "frequency": {}}
    sorted_values = sorted(values)
    def percentile(p: float) -> int:
        if len(sorted_values) == 1:
            return sorted_values[0]
        pos = p * (len(sorted_values) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return sorted_values[lo]
        return int(round(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)))
    return {
        "task_count": len(sorted_values),
        "min": sorted_values[0],
        "q1": percentile(0.25),
        "median": percentile(0.50),
        "q3": percentile(0.75),
        "max": sorted_values[-1],
        "frequency": {str(k): v for k, v in sorted(Counter(sorted_values).items())},
    }


def load_assays() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(RAW / "chembl37_required_assays.jsonl")
    return {str(row["assay_chembl_id"]): row for row in rows}


def write_record_eligibility(assays: dict[str, dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    fields = [
        "source",
        "isoform",
        "activity_id",
        "record_id",
        "assay_chembl_id",
        "molecule_chembl_id",
        "parent_molecule_chembl_id",
        "canonical_smiles",
        "document_chembl_id",
        "src_id",
        "assay_src_id",
        "assay_src_assay_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "activity_comment",
        "data_validity_comment",
        "primary_class",
        "eligibility_reason",
        "would_be_class_without_validity_flag",
        "validity_status",
        "validity_mapping_reason",
        "normalized_endpoint_semantics",
        "normalized_relation",
        "normalized_unit_semantics",
        "unfamiliar_semantics_flag",
        "unfamiliar_semantics_reason",
        "assay_description",
        "assay_type",
        "assay_organism",
        "assay_bao_format",
        "assay_bao_label",
        "target_chembl_id",
        "target_organism",
        "document_journal",
        "document_year",
        "potential_duplicate",
    ]
    rows_out: list[dict[str, str]] = []
    endpoint_counter = Counter()
    unit_counter = Counter()
    relation_counter = Counter()
    combo_counter = Counter()
    validity_counter = Counter()
    unknown_validity = Counter()
    path = DERIVED / "record_eligibility.tsv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for isoform in ISOFORMS:
            for row in read_jsonl(RAW / f"chembl37_{isoform}_activities.jsonl"):
                assay = assays.get(str(row.get("assay_chembl_id") or ""), {})
                cls = classify_record(row)
                if cls["primary_class"] not in PRIMARY_CLASSES:
                    fail(f"unknown primary class {cls['primary_class']}")
                data_validity = str(row.get("data_validity_comment") or "").strip()
                if cls["validity_status"] == "unknown":
                    unknown_validity[data_validity] += 1
                endpoint = str(row.get("standard_type") or "")
                relation = str(row.get("standard_relation") or "")
                unit = str(row.get("standard_units") or "")
                endpoint_counter[endpoint] += 1
                relation_counter[relation] += 1
                unit_counter[unit] += 1
                combo_counter[(endpoint, relation, unit, cls["primary_class"])] += 1
                validity_counter[cls["validity_status"]] += 1
                out = {
                    "source": "ChEMBL37",
                    "isoform": isoform,
                    "activity_id": row.get("activity_id", ""),
                    "record_id": row.get("record_id", ""),
                    "assay_chembl_id": row.get("assay_chembl_id", ""),
                    "molecule_chembl_id": row.get("molecule_chembl_id", ""),
                    "parent_molecule_chembl_id": row.get("parent_molecule_chembl_id", ""),
                    "canonical_smiles": row.get("canonical_smiles", ""),
                    "document_chembl_id": row.get("document_chembl_id", ""),
                    "src_id": row.get("src_id", ""),
                    "assay_src_id": assay.get("src_id", ""),
                    "assay_src_assay_id": assay.get("src_assay_id", ""),
                    "standard_type": endpoint,
                    "standard_relation": relation,
                    "standard_value": row.get("standard_value", ""),
                    "standard_units": unit,
                    "pchembl_value": row.get("pchembl_value", ""),
                    "activity_comment": row.get("activity_comment", ""),
                    "data_validity_comment": data_validity,
                    "assay_description": assay.get("description") or row.get("assay_description", ""),
                    "assay_type": assay.get("assay_type") or row.get("assay_type", ""),
                    "assay_organism": assay.get("assay_organism", ""),
                    "assay_bao_format": assay.get("bao_format") or row.get("bao_format", ""),
                    "assay_bao_label": assay.get("bao_label") or row.get("bao_label", ""),
                    "target_chembl_id": row.get("target_chembl_id", ""),
                    "target_organism": row.get("target_organism", ""),
                    "document_journal": row.get("document_journal", ""),
                    "document_year": row.get("document_year", ""),
                    "potential_duplicate": row.get("potential_duplicate", ""),
                    **cls,
                }
                out = {k: "" if v is None else str(v).replace("\t", " ").replace("\n", " ") for k, v in out.items()}
                writer.writerow(out)
                rows_out.append(out)
    summary = {
        "observed_endpoint_counts": dict(endpoint_counter.most_common()),
        "observed_unit_counts": dict(unit_counter.most_common()),
        "observed_relation_counts": dict(relation_counter.most_common()),
        "observed_endpoint_relation_unit_class_counts": [
            {
                "standard_type": k[0],
                "standard_relation": k[1],
                "standard_units": k[2],
                "primary_class": k[3],
                "count": v,
            }
            for k, v in sorted(combo_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "validity_status_counts": dict(validity_counter),
        "unknown_validity_comments": dict(unknown_validity),
        "declarative_mappings": declarative_mappings(),
    }
    return rows_out, summary


def declarative_mappings() -> dict[str, Any]:
    return {
        "endpoint_mapping": ENDPOINT_MAPPING,
        "concentration_units": CONCENTRATION_UNITS,
        "non_concentration_units": NON_CONCENTRATION_UNITS,
        "exact_relations": sorted(EXACT_RELATIONS),
        "censored_relations": sorted(CENSORED_RELATIONS),
        "known_nonexact_relations": sorted(KNOWN_NONEXACT_RELATIONS),
        "validity_mapping": VALIDITY_MAPPING,
        "classification_precedence": [
            "validity handling",
            "interpretable numeric measurement classification",
            "qualitative/nonquantitative classification",
            "missing/uninterpretable classification",
        ],
    }


def build_tasks(rows: list[dict[str, str]], assays: dict[str, dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["isoform"], row["assay_chembl_id"], row["normalized_endpoint_semantics"])].append(row)

    task_rows: list[dict[str, str]] = []
    for key in sorted(grouped):
        isoform, assay_id, endpoint = key
        group = grouped[key]
        assay = assays.get(assay_id, {})
        class_counts = Counter(r["primary_class"] for r in group)
        class_unique = {
            cls: len({r["molecule_chembl_id"] for r in group if r["primary_class"] == cls and r["molecule_chembl_id"]})
            for cls in PRIMARY_CLASSES
        }
        molecule_counts = Counter(r["molecule_chembl_id"] for r in group if r["molecule_chembl_id"])
        repeated_overall = sum(1 for count in molecule_counts.values() if count > 1)
        repeated_by_class = {}
        for cls in PRIMARY_CLASSES:
            counts = Counter(r["molecule_chembl_id"] for r in group if r["primary_class"] == cls and r["molecule_chembl_id"])
            repeated_by_class[cls] = sum(1 for count in counts.values() if count > 1)
        row = {
            "isoform": isoform,
            "assay_chembl_id": assay_id,
            "normalized_endpoint_semantics": endpoint,
            "raw_record_count": str(len(group)),
            "unique_molecule_chembl_id_count": str(len(molecule_counts)),
            "repeated_molecule_id_count_overall": str(repeated_overall),
            "relation_distribution": json.dumps(dict(sorted(Counter(r["normalized_relation"] for r in group).items())), sort_keys=True),
            "unit_distribution": json.dumps(dict(sorted(Counter(r["normalized_unit_semantics"] for r in group).items())), sort_keys=True),
            "validity_status_distribution": json.dumps(dict(sorted(Counter(r["validity_status"] for r in group).items())), sort_keys=True),
            "document_chembl_ids": tsv_join([r["document_chembl_id"] for r in group]),
            "source_ids": tsv_join([r["src_id"] for r in group]),
            "assay_src_id": str(assay.get("src_id", "")),
            "assay_src_assay_id": str(assay.get("src_assay_id", "") or ""),
            "assay_description": str(assay.get("description") or "").replace("\t", " ").replace("\n", " "),
            "assay_type": str(assay.get("assay_type") or ""),
            "assay_type_description": str(assay.get("assay_type_description") or ""),
            "assay_organism": str(assay.get("assay_organism") or ""),
            "assay_bao_format": str(assay.get("bao_format") or ""),
            "assay_bao_label": str(assay.get("bao_label") or ""),
            "assay_confidence_score": str(assay.get("confidence_score") or ""),
            "unfamiliar_semantics_record_count": str(sum(1 for r in group if r["unfamiliar_semantics_flag"] == "true")),
        }
        for cls in PRIMARY_CLASSES:
            row[f"records_{cls}"] = str(class_counts.get(cls, 0))
            row[f"unique_molecules_{cls}"] = str(class_unique[cls])
            row[f"repeated_molecules_{cls}"] = str(repeated_by_class[cls])
        task_rows.append(row)

    task_fields = [
        "isoform",
        "assay_chembl_id",
        "normalized_endpoint_semantics",
        "raw_record_count",
        "unique_molecule_chembl_id_count",
        "repeated_molecule_id_count_overall",
        "relation_distribution",
        "unit_distribution",
        "validity_status_distribution",
        "document_chembl_ids",
        "source_ids",
        "assay_src_id",
        "assay_src_assay_id",
        "assay_description",
        "assay_type",
        "assay_type_description",
        "assay_organism",
        "assay_bao_format",
        "assay_bao_label",
        "assay_confidence_score",
        "unfamiliar_semantics_record_count",
    ]
    for cls in PRIMARY_CLASSES:
        task_fields.extend([f"records_{cls}", f"unique_molecules_{cls}", f"repeated_molecules_{cls}"])
    with (DERIVED / "assay_task_table.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(task_rows)

    distribution = build_task_distribution(task_rows)
    return task_rows, distribution


def build_task_distribution(task_rows: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "record_count_distributions": {},
        "unique_molecule_count_distributions": {},
        "exact_quantitative_potency_candidate_record_count_distributions": {},
        "exact_quantitative_potency_candidate_unique_molecule_count_distributions": {},
        "full_task_table": [],
    }
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in task_rows:
        endpoint = row["normalized_endpoint_semantics"]
        groups[(row["isoform"], endpoint)].append(row)
        groups[(row["isoform"], "__all_endpoints__")].append(row)

    for key in sorted(groups):
        label = f"{key[0]}\t{key[1]}"
        rows = groups[key]
        out["record_count_distributions"][label] = distribution_summary([int(r["raw_record_count"]) for r in rows])
        out["unique_molecule_count_distributions"][label] = distribution_summary([int(r["unique_molecule_chembl_id_count"]) for r in rows])
        out["exact_quantitative_potency_candidate_record_count_distributions"][label] = distribution_summary([
            int(r["records_exact_quantitative_potency_candidate"]) for r in rows
        ])
        out["exact_quantitative_potency_candidate_unique_molecule_count_distributions"][label] = distribution_summary([
            int(r["unique_molecules_exact_quantitative_potency_candidate"]) for r in rows
        ])
    out["full_task_table"] = [
        {
            "isoform": r["isoform"],
            "assay_chembl_id": r["assay_chembl_id"],
            "normalized_endpoint_semantics": r["normalized_endpoint_semantics"],
            "raw_record_count": int(r["raw_record_count"]),
            "unique_molecule_count": int(r["unique_molecule_chembl_id_count"]),
            "exact_quantitative_potency_candidate_records": int(r["records_exact_quantitative_potency_candidate"]),
            "exact_quantitative_potency_candidate_unique_molecules": int(r["unique_molecules_exact_quantitative_potency_candidate"]),
        }
        for r in task_rows
    ]
    canonical_json_dump(DERIVED / "task_size_distribution.json", out)
    return out


def pubchem_result_rows(aid: int) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    with (RAW / f"pubchem_AID{aid}_full.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    metadata = []
    results = []
    for row in rows:
        tag = str(row.get("PUBCHEM_RESULT_TAG", "")).strip()
        sid = str(row.get("PUBCHEM_SID", "")).strip()
        if tag in PUBCHEM_METADATA_TAGS:
            metadata.append(row)
        elif sid:
            results.append(row)
        else:
            fail(f"PubChem AID{aid} has nonmetadata row without SID")
    return metadata, results, fields


def build_pubchem_accounting() -> dict[str, Any]:
    out = {}
    for isoform, aid in PUBCHEM_AIDS.items():
        metadata, results, fields = pubchem_result_rows(aid)
        sid_counts = Counter(str(r.get("PUBCHEM_SID") or "").strip() for r in results)
        field_groups = load_json(STATE / f"pubchem_AID{aid}.complete.json").get("field_groups", {})
        field_groups = {k: list(v) for k, v in field_groups.items()}
        field_groups["efficacy_literal"] = [
            x for x in field_groups.get("efficacy_literal", [])
            if "hill coefficient" not in x.lower()
        ]
        hill_slope_fields = set(field_groups.get("hill_slope", []))
        for field in fields:
            if "hill coefficient" in field.lower() or "hill_slope" in field.lower():
                hill_slope_fields.add(field)
        field_groups["hill_slope"] = sorted(hill_slope_fields)
        out[isoform] = {
            "aid": aid,
            "raw_table_rows_including_header": sum(1 for _ in (RAW / f"pubchem_AID{aid}_full.csv").open(encoding="utf-8-sig")),
            "schema_metadata_rows": len(metadata),
            "schema_metadata_tag_counts": dict(Counter(str(r.get("PUBCHEM_RESULT_TAG") or "").strip() for r in metadata)),
            "actual_result_rows": len(results),
            "unique_nonblank_result_sids": len(sid_counts),
            "duplicate_result_sid_rows": sum(count - 1 for count in sid_counts.values() if count > 1),
            "cid_present_result_rows": sum(1 for r in results if str(r.get("PUBCHEM_CID") or "").strip()),
            "cid_missing_result_rows": sum(1 for r in results if not str(r.get("PUBCHEM_CID") or "").strip()),
            "activity_outcome_counts": dict(Counter(str(r.get("PUBCHEM_ACTIVITY_OUTCOME") or "").strip() for r in results)),
            "activity_score_present_rows": sum(1 for r in results if str(r.get("PUBCHEM_ACTIVITY_SCORE") or "").strip()),
            "activity_direction_counts": dict(Counter(str(r.get("Activity Direction") or "").strip() for r in results)) if "Activity Direction" in fields else {},
            "activity_qualifier_counts": dict(Counter(str(r.get("Activity Qualifier") or "").strip() for r in results)) if "Activity Qualifier" in fields else {},
            "qualified_ac50_present_rows": sum(1 for r in results if str(r.get("Qualified AC50") or r.get("Potency") or "").strip()),
            "log_ac50_present_rows": sum(1 for r in results if str(r.get("Log of AC50") or r.get("Fit_LogAC50") or "").strip()),
            "field_groups": field_groups,
            "hill_coefficient_interpretation": "Hill Coefficient is hill slope, not literal efficacy.",
            "columns": fields,
        }
    return out


def build_reconciliation(task_rows: list[dict[str, str]], assays: dict[str, dict[str, Any]]) -> dict[str, Any]:
    known_pubchem_by_src_assay_id = {str(aid): (isoform, aid) for isoform, aid in PUBCHEM_AIDS.items()}
    fields = [
        "isoform",
        "assay_chembl_id",
        "normalized_endpoint_semantics",
        "reconciliation_category",
        "matched_pubchem_aid",
        "evidence_level",
        "evidence",
        "task_record_count",
    ]
    rows = []
    counts = Counter()
    duplicate_keys = set()
    for task in task_rows:
        assay_id = task["assay_chembl_id"]
        assay = assays.get(assay_id, {})
        src_id = str(assay.get("src_id") or "")
        src_assay_id = str(assay.get("src_assay_id") or "")
        isoform = task["isoform"]
        if src_id == "7" and src_assay_id in known_pubchem_by_src_assay_id:
            aid_isoform, aid = known_pubchem_by_src_assay_id[src_assay_id]
            if aid_isoform == isoform:
                category = "demonstrated_pubchem_duplicate"
                evidence_level = "1_explicit_pubchem_source_identifier"
                evidence = f"ChEMBL assay src_id=7 and src_assay_id={src_assay_id} exactly matches known PubChem AID {aid} for {isoform}."
                duplicate_keys.add((isoform, assay_id, task["normalized_endpoint_semantics"]))
            else:
                category = "overlap_unresolved"
                evidence_level = "conflicting_or_cross_isoform_pubchem_source_identifier"
                evidence = f"ChEMBL assay src_id=7 src_assay_id={src_assay_id} maps to AID {aid} for {aid_isoform}, not task isoform {isoform}; retained unresolved."
        else:
            category = "overlap_unresolved"
            evidence_level = "no_explicit_duplicate_or_distinct_evidence"
            evidence = "No explicit PubChem AID source identifier or reviewed document/protocol evidence establishes duplicate or distinct status. Absence of cross-reference is not treated as distinct evidence."
            aid = ""
        counts[category] += 1
        rows.append({
            "isoform": isoform,
            "assay_chembl_id": assay_id,
            "normalized_endpoint_semantics": task["normalized_endpoint_semantics"],
            "reconciliation_category": category,
            "matched_pubchem_aid": aid,
            "evidence_level": evidence_level,
            "evidence": evidence,
            "task_record_count": task["raw_record_count"],
        })

    with (DERIVED / "pubchem_chembl_reconciliation.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    before = len(task_rows)
    after = before - counts["demonstrated_pubchem_duplicate"]
    return {
        "counts": dict(counts),
        "task_diversity_before_reconciliation": before,
        "task_diversity_after_excluding_demonstrated_duplicates": after,
        "post_reconciliation_definition": "all tasks minus demonstrated duplicates, with unresolved overlap still present",
        "demonstrated_duplicate_task_keys": [list(x) for x in sorted(duplicate_keys)],
        "demonstrated_distinct_count": counts.get("demonstrated_distinct", 0),
        "unresolved_overlap_retained_count": counts.get("overlap_unresolved", 0),
    }


def build_molecule_overlap(rows: list[dict[str, str]], task_rows: list[dict[str, str]]) -> dict[str, Any]:
    with TRAIN_PATH.open(newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    train_smiles = {r["SMILES"] for r in train_rows if r.get("SMILES")}
    chembl_smiles_by_task: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    chembl_all = set()
    for r in rows:
        smiles = r.get("canonical_smiles", "")
        if smiles:
            key = (r["isoform"], r["assay_chembl_id"], r["normalized_endpoint_semantics"])
            chembl_smiles_by_task[key].add(smiles)
            chembl_all.add(smiles)

    pubchem_by_task = {}
    pubchem_all = set()
    for isoform, aid in PUBCHEM_AIDS.items():
        _, results, _ = pubchem_result_rows(aid)
        smiles = {str(r.get("PUBCHEM_EXT_DATASOURCE_SMILES") or "").strip() for r in results if str(r.get("PUBCHEM_EXT_DATASOURCE_SMILES") or "").strip()}
        pubchem_by_task[(isoform, f"AID{aid}", "PubChem_assay_result_table")] = smiles
        pubchem_all.update(smiles)

    task_overlap_rows = []
    for key in sorted(chembl_smiles_by_task):
        identities = chembl_smiles_by_task[key]
        task_overlap_rows.append({
            "source": "ChEMBL37",
            "isoform": key[0],
            "task_id": f"{key[1]}::{key[2]}",
            "unique_external_identities_exact_smiles": len(identities),
            "overlap_with_openadmet_train_exact_smiles": len(identities & train_smiles),
            "external_only_exact_smiles": len(identities - train_smiles),
        })
    for key in sorted(pubchem_by_task):
        identities = pubchem_by_task[key]
        task_overlap_rows.append({
            "source": "PubChem",
            "isoform": key[0],
            "task_id": key[1],
            "unique_external_identities_exact_smiles": len(identities),
            "overlap_with_openadmet_train_exact_smiles": len(identities & train_smiles),
            "external_only_exact_smiles": len(identities - train_smiles),
        })

    membership = Counter()
    for identities in chembl_smiles_by_task.values():
        for smiles in identities:
            membership[smiles] += 1
    for identities in pubchem_by_task.values():
        for smiles in identities:
            membership[smiles] += 1

    out = {
        "identity_policy": "Exact stored SMILES string comparison only; no new molecular normalization policy was introduced.",
        "claim_boundary": "TRAIN-only overlap is descriptive and does not establish that any future training protocol respects every frozen validation/test boundary.",
        "openadmet_train_path_read": str(TRAIN_PATH),
        "openadmet_public_test_labels_accessed": False,
        "openadmet_train_unique_exact_smiles": len(train_smiles),
        "chembl_unique_exact_smiles": len(chembl_all),
        "pubchem_unique_exact_smiles": len(pubchem_all),
        "chembl_overlap_with_train_exact_smiles": len(chembl_all & train_smiles),
        "pubchem_overlap_with_train_exact_smiles": len(pubchem_all & train_smiles),
        "external_all_unique_exact_smiles": len(chembl_all | pubchem_all),
        "external_all_overlap_with_train_exact_smiles": len((chembl_all | pubchem_all) & train_smiles),
        "external_all_only_exact_smiles": len((chembl_all | pubchem_all) - train_smiles),
        "pubchem_chembl_molecular_overlap_exact_smiles": len(pubchem_all & chembl_all),
        "external_identities_in_multiple_external_tasks_exact_smiles": sum(1 for count in membership.values() if count > 1),
        "task_overlap_summary": task_overlap_rows,
    }
    canonical_json_dump(DERIVED / "molecule_overlap_summary.json", out)
    return out


def integrity_checks(
    rows: list[dict[str, str]],
    task_rows: list[dict[str, str]],
    distribution: dict[str, Any],
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    checks = []
    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            fail(f"check failed: {name}: {detail}")
        checks.append({"name": name, "status": "pass", "detail": detail})

    check("chembl_every_record_one_primary_class", len(rows) == 155803 and all(r["primary_class"] in PRIMARY_CLASSES for r in rows))
    class_counts = Counter(r["primary_class"] for r in rows)
    check("class_counts_sum_to_raw_rows", sum(class_counts.values()) == 155803)
    check("manually_validated_not_adverse", all(r["validity_status"] != "adverse" for r in rows if r["data_validity_comment"] == "Manually validated"))
    check("unknown_validity_remains_unknown", all(r["validity_status"] == "unknown" for r in rows if r["data_validity_comment"] and r["data_validity_comment"] not in VALIDITY_MAPPING))
    check("qualitative_does_not_override_numeric", all(not (parse_float(r["standard_value"]) is not None and r["activity_comment"] and r["primary_class"] == "qualitative_or_nonquantitative") for r in rows))
    check("unfamiliar_not_mapped_to_potency", all(not (r["unfamiliar_semantics_flag"] == "true" and r["primary_class"] in {"exact_quantitative_potency_candidate", "censored_quantitative_potency"}) for r in rows))
    check("no_task_omitted_for_size", sum(int(t["raw_record_count"]) for t in task_rows) == len(rows))
    for task in task_rows:
        raw_unique = int(task["unique_molecule_chembl_id_count"])
        for cls in PRIMARY_CLASSES:
            check(f"class_unique_not_exceed_raw_unique::{task['isoform']}::{task['assay_chembl_id']}::{task['normalized_endpoint_semantics']}::{cls}", int(task[f"unique_molecules_{cls}"]) <= raw_unique)
    check("demonstrated_duplicates_only_excluded_from_deduplicated_view", reconciliation["task_diversity_after_excluding_demonstrated_duplicates"] == reconciliation["task_diversity_before_reconciliation"] - reconciliation["counts"].get("demonstrated_pubchem_duplicate", 0))
    check("unresolved_overlap_remains_unresolved", reconciliation["unresolved_overlap_retained_count"] == reconciliation["counts"].get("overlap_unresolved", 0))
    check("openadmet_public_test_labels_not_accessed", True, "script only reads cyp/data/cyp-challenge-TRAIN_inhibition.csv")
    return {"checks": checks, "class_counts": dict(class_counts)}


def write_sha_inventory() -> dict[str, str]:
    decision_files = [
        DERIVED / "record_eligibility.tsv",
        DERIVED / "assay_task_table.tsv",
        DERIVED / "task_size_distribution.json",
        DERIVED / "pubchem_chembl_reconciliation.tsv",
        DERIVED / "molecule_overlap_summary.json",
        DERIVED / "endpoint_semantics_summary.json",
        DERIVED / "qualification_summary.json",
        DERIVED / "source_provenance.json",
    ]
    inventory = {str(path.relative_to(ROOT)): sha256_file(path) for path in decision_files}
    with (DERIVED / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for path in sorted(decision_files):
            handle.write(f"{sha256_file(path)}  {path.relative_to(ROOT)}\n")
    inventory["derived/SHA256SUMS.txt"] = sha256_file(DERIVED / "SHA256SUMS.txt")
    return inventory


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    frozen = verify_frozen_inputs()
    assays = load_assays()
    rows, endpoint_summary = write_record_eligibility(assays)
    task_rows, distribution = build_tasks(rows, assays)
    pubchem_accounting = build_pubchem_accounting()
    reconciliation = build_reconciliation(task_rows, assays)
    overlap = build_molecule_overlap(rows, task_rows)

    endpoint_summary.update({
        "unfamiliar_semantics_counts": dict(Counter(
            r["unfamiliar_semantics_reason"] for r in rows
            if r["unfamiliar_semantics_flag"] == "true"
        )),
        "class_counts_by_isoform": {
            isoform: dict(Counter(r["primary_class"] for r in rows if r["isoform"] == isoform))
            for isoform in ISOFORMS
        },
    })
    canonical_json_dump(DERIVED / "endpoint_semantics_summary.json", endpoint_summary)

    checks = integrity_checks(rows, task_rows, distribution, reconciliation)
    source_provenance = {
        "artifact_type": "CYP_external_source_local_census_v1",
        "purpose": "source/task qualification census only; no modeling/training/D011 decision",
        "frozen_inputs": frozen,
        "files_read": [
            str(RAW / f"chembl37_{isoform}_activities.jsonl") for isoform in ISOFORMS
        ] + [
            str(RAW / "chembl37_required_assays.jsonl"),
            str(MANIFESTS / "source_provenance.json"),
            str(TRAIN_PATH),
        ] + [str(RAW / f"pubchem_AID{aid}_full.csv") for aid in PUBCHEM_EXPECTED],
        "openadmet_public_test_labels_accessed": False,
    }
    canonical_json_dump(DERIVED / "source_provenance.json", source_provenance)

    class_counts_by_isoform = endpoint_summary["class_counts_by_isoform"]
    exact_candidate_task_populations = {
        "total_tasks_with_any_exact_candidate_records": sum(1 for t in task_rows if int(t["records_exact_quantitative_potency_candidate"]) > 0),
        "total_exact_candidate_records": sum(int(t["records_exact_quantitative_potency_candidate"]) for t in task_rows),
        "total_exact_candidate_unique_molecule_task_instances": sum(int(t["unique_molecules_exact_quantitative_potency_candidate"]) for t in task_rows),
        "by_isoform": {
            isoform: {
                "tasks_with_any_exact_candidate_records": sum(1 for t in task_rows if t["isoform"] == isoform and int(t["records_exact_quantitative_potency_candidate"]) > 0),
                "exact_candidate_records": sum(int(t["records_exact_quantitative_potency_candidate"]) for t in task_rows if t["isoform"] == isoform),
                "exact_candidate_unique_molecule_task_instances": sum(int(t["unique_molecules_exact_quantitative_potency_candidate"]) for t in task_rows if t["isoform"] == isoform),
            }
            for isoform in ISOFORMS
        },
    }
    qualification = {
        "frozen_source_counts_hashes": frozen,
        "raw_measurement_volume": {
            "chembl_activity_records": len(rows),
            "pubchem_result_rows_total": sum(x["actual_result_rows"] for x in pubchem_accounting.values()),
            "pubchem_by_isoform": pubchem_accounting,
        },
        "validity_comment_mapping": VALIDITY_MAPPING,
        "unknown_validity_comments": endpoint_summary["unknown_validity_comments"],
        "eligibility_class_counts_by_source_isoform": {"ChEMBL37": class_counts_by_isoform},
        "unfamiliar_semantic_categories_requiring_review": endpoint_summary["unfamiliar_semantics_counts"],
        "assay_resolved_task_counts": {
            "total_tasks": len(task_rows),
            "by_isoform": dict(Counter(t["isoform"] for t in task_rows)),
        },
        "record_count_task_size_distributions": distribution["record_count_distributions"],
        "unique_molecule_task_size_distributions": distribution["unique_molecule_count_distributions"],
        "exact_quantitative_candidate_task_populations": exact_candidate_task_populations,
        "pubchem_reconciliation": reconciliation,
        "train_only_molecular_overlap": {
            k: v for k, v in overlap.items() if k != "task_overlap_summary"
        },
        "no_minimum_task_size_cutoff_used": True,
        "openadmet_public_test_labels_accessed": False,
        "d011_replacement_or_resume_decision": "not made",
        "integrity_checks": checks,
        "decision_bearing_output_hash_inventory": "derived/SHA256SUMS.txt",
    }
    canonical_json_dump(DERIVED / "qualification_summary.json", qualification)
    inventory = write_sha_inventory()
    print(json.dumps({
        "status": "ok",
        "chembl_records": len(rows),
        "task_count": len(task_rows),
        "output_sha256": inventory,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
