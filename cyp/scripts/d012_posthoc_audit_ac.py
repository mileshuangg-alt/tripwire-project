from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "qualification/cyp_external_task_qualification_20261109/d012_posthoc_audit_ac"
D012_AGG = ROOT / "qualification/cyp_external_task_qualification_20261109/d012_final_aggregation"
D009 = ROOT / "artifacts/d009_closure_workspace"
T121_DIR = D009 / "d009_t121_comparator_20261109"
P1_O150 = D009 / "p1_d009_o150_result_20261108.json"
D009_INVENTORY = D009 / "d009_p1_evidence_inventory_20261109.json"
D009_CHARACTERIZATION = D009 / "d009_p1_closure_characterization_20261109.json"
AUX_INSTANCES = ROOT / "qualification/cyp_external_task_qualification_20261109/d012_auxiliary_population/d012_task_identity_instances.tsv"
CROSS_SOURCE = ROOT / "qualification/cyp_external_task_qualification_20261109/identity_audit/cross_source_matches.tsv"
TRAIN = ROOT / "cyp/data/cyp-challenge-TRAIN_inhibition.csv"

TASKS = (
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
)
ISO_TO_TASK = {task.split("_", 1)[0]: task for task in TASKS}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def cell_key(repetition: int, fold: int, seed: int) -> str:
    return f"rep_{repetition:02d}_fold_{fold:02d}|seed_{seed}"


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sample_sd": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def diff_stats(values: list[float]) -> dict[str, Any]:
    result = stats(values)
    result.update({
        "first_method_wins_lower_is_better": sum(value < 0 for value in values),
        "second_method_wins_lower_is_better": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
    })
    return result


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / denom


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in range(i, j):
            result[ordered[k][0]] = rank
        i = j
    return result


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(ranks(xs), ranks(ys))


def correlation_stats(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    diffs = [aux - train for aux, train in pairs]
    xs = [aux for aux, _ in pairs]
    ys = [train for _, train in pairs]
    result = stats(diffs)
    result.update({
        "mae": sum(abs(value) for value in diffs) / len(diffs) if diffs else None,
        "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs)) if diffs else None,
        "pearson_r": pearson(xs, ys),
        "spearman_rho": spearman(xs, ys),
    })
    return result


def load_p1_cells() -> dict[str, dict[str, Any]]:
    payload = read_json(P1_O150)
    result = {}
    for row in payload["cells"]:
        c = row["cell"]
        key = cell_key(int(c["repetition"]), int(c["fold"]), int(c["training_seed"]))
        result[key] = row
    return result


def load_d012_rows() -> dict[str, dict[str, Any]]:
    rows = read_tsv(D012_AGG / "d012_cell_level_joined_results.tsv")
    return {row["cell_key"]: row for row in rows}


def load_t121_cells() -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((T121_DIR / "cells").glob("cell_*/result.json")):
        row = read_json(path)
        key = cell_key(int(row["repetition"]), int(row["fold"]), int(row["seed"]))
        per = row["best_validation"].get("per_task")
        if set(per or {}) != set(TASKS):
            raise ValueError(f"T121 per-task keys differ for {path}")
        macro = float(row["macro_st_rae"])
        if abs(sum(float(per[task]) for task in TASKS) / len(TASKS) - macro) > 1e-12:
            raise ValueError(f"T121 macro/per-task mismatch for {path}")
        result[key] = {
            "cell_key": key,
            "repetition": int(row["repetition"]),
            "fold": int(row["fold"]),
            "training_seed": int(row["seed"]),
            "macro_st_rae": macro,
            "per_endpoint_st_rae": {task: float(per[task]) for task in TASKS},
            "path": str(path.relative_to(ROOT)),
            "config_name": row.get("config_name"),
            "trial_index": row.get("trial_index"),
            "status": row.get("status"),
        }
    if len(result) != 150:
        raise ValueError(f"expected 150 T121 cells, found {len(result)}")
    return result


def load_b_c_metrics(d012_rows: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    b = {}
    c = {}
    for key, row in d012_rows.items():
        b_metrics = read_json(ROOT / row["b_metric_path"])
        c_metrics = read_json(ROOT / row["c_metric_path"])
        b[key] = {
            "macro_st_rae": float(b_metrics["best_macro_st_rae"]),
            "per_endpoint_st_rae": b_metrics.get("best_per_endpoint_st_rae"),
            "path": row["b_metric_path"],
        }
        c[key] = {
            "macro_st_rae": float(c_metrics["score"]["macro_st_rae"]),
            "per_endpoint_st_rae": {task: float(c_metrics["score"]["per_task"][task]) for task in TASKS},
            "path": row["c_metric_path"],
        }
    return b, c


def audit_a() -> dict[str, Any]:
    d012_summary = read_json(D012_AGG / "d012_final_aggregate_summary.json")
    d012_rows = load_d012_rows()
    p1_payload = read_json(P1_O150)
    p1_cells = load_p1_cells()
    t121_cells = load_t121_cells()
    inventory = read_json(D009_INVENTORY)

    d012_labeled_t121 = {key: float(row["t121_macro_st_rae"]) for key, row in d012_rows.items()}
    p1_cell_macros = {key: float(row["macro_st_rae"]) for key, row in p1_cells.items()}
    corrected_t121 = {key: float(row["macro_st_rae"]) for key, row in t121_cells.items()}
    keys_equal = set(d012_labeled_t121) == set(p1_cell_macros) == set(corrected_t121)
    labeled_equals_p1_cells = all(abs(d012_labeled_t121[key] - p1_cell_macros[key]) <= 1e-15 for key in d012_labeled_t121)
    labeled_equals_t121_cells = all(abs(d012_labeled_t121[key] - corrected_t121[key]) <= 1e-15 for key in d012_labeled_t121)
    corrected_o150 = sum(corrected_t121.values()) / 150
    p1_o150_from_cells = sum(p1_cell_macros.values()) / 150

    locally_preserved = inventory.get("locally_preserved_files", [])
    inv_p1 = [row for row in locally_preserved if row.get("path") == str(P1_O150.relative_to(ROOT))]
    inv_t121_results = [row for row in locally_preserved if row.get("category") == "t121_cell_result"]

    return {
        "artifact_type": "D012_posthoc_audit_A_comparator_provenance_v1",
        "defect_exists": labeled_equals_p1_cells and not labeled_equals_t121_cells,
        "semantic_identity": {
            "d012_labeled_t121_o150": d012_summary["o150"]["T121"],
            "d012_labeled_t121_semantics": "P1 cells[] from p1_d009_o150_result_20261108.json, not D007-T121 comparator cells",
            "p1_o150_from_cells": p1_o150_from_cells,
            "p1_o150_summary_field": p1_payload.get("o150"),
            "p1_advances": p1_payload.get("p1_advances"),
            "authoritative_t121_o150_in_p1_file_summary_field": p1_payload.get("t121_o150"),
            "authoritative_t121_o150_in_d009_final_verdict": inventory.get("final_verdict", {}).get("t121_o150"),
            "corrected_t121_o150_from_saved_cells": corrected_o150,
            "labeled_t121_equals_p1_cells_exactly": labeled_equals_p1_cells,
            "labeled_t121_equals_saved_t121_cells_exactly": labeled_equals_t121_cells,
            "ensemble_evidence_found": False,
            "ensemble_note": "No D012 comparator source field or D009 T121 result artifact used ensemble predictions; T121 saved cell results identify config_name D007-T121/trial_index 121.",
        },
        "cell_population": {
            "d012_labeled_t121_cells": len(d012_labeled_t121),
            "p1_cells": len(p1_cell_macros),
            "saved_t121_cells": len(corrected_t121),
            "all_key_sets_equal": keys_equal,
            "key_schema": "repetition + fold + training_seed",
        },
        "metric_definition": {
            "held_out_validation_macro_st_rae": "Arithmetic mean of four endpoint ST-RAE values in saved best_validation/per_task or per_endpoint_st_rae fields.",
            "o150": "Arithmetic mean of macro ST-RAE over exact 150 frozen cells; no weighting or filtering.",
            "macro_checks_passed": True,
        },
        "prediction_recompute_limitation": {
            "saved_t121_validation_predictions_found": False,
            "saved_t121_validation_metric_results_found": True,
            "action": "Used preserved T121 validation metrics only; did not regenerate predictions.",
        },
        "provenance": {
            "d012_summary": {"path": str((D012_AGG / "d012_final_aggregate_summary.json").relative_to(ROOT)), "sha256": sha256_file(D012_AGG / "d012_final_aggregate_summary.json")},
            "d012_joined_rows": {"path": str((D012_AGG / "d012_cell_level_joined_results.tsv").relative_to(ROOT)), "sha256": sha256_file(D012_AGG / "d012_cell_level_joined_results.tsv")},
            "d012_labeled_t121_source": d012_summary["t121_source"],
            "p1_o150_file": {"path": str(P1_O150.relative_to(ROOT)), "sha256": sha256_file(P1_O150), "inventory_entries": inv_p1},
            "t121_cells_dir": {"path": str((T121_DIR / "cells").relative_to(ROOT)), "result_file_count": len(inv_t121_results)},
            "t121_proposal": {"path": str((T121_DIR / "proposal.json").relative_to(ROOT)), "sha256": sha256_file(T121_DIR / "proposal.json")},
            "d009_characterization": {"path": str(D009_CHARACTERIZATION.relative_to(ROOT)), "sha256": sha256_file(D009_CHARACTERIZATION)},
        },
    }


def build_correction(audit: dict[str, Any]) -> dict[str, Any]:
    d012_rows = load_d012_rows()
    t121 = load_t121_cells()
    b, c = load_b_c_metrics(d012_rows)
    rows = []
    for key in sorted(d012_rows):
        rep = int(d012_rows[key]["repetition"])
        fold = int(d012_rows[key]["fold"])
        seed = int(d012_rows[key]["training_seed"])
        b_macro = b[key]["macro_st_rae"]
        c_macro = c[key]["macro_st_rae"]
        t_macro = t121[key]["macro_st_rae"]
        rows.append({
            "cell_key": key,
            "repetition": rep,
            "fold": fold,
            "training_seed": seed,
            "b_macro_st_rae": b_macro,
            "c_macro_st_rae": c_macro,
            "corrected_t121_macro_st_rae": t_macro,
            "delta_c_minus_b": c_macro - b_macro,
            "corrected_delta_b_minus_t121": b_macro - t_macro,
            "corrected_delta_c_minus_t121": c_macro - t_macro,
        })
    return {
        "artifact_type": "D012_posthoc_audit_A_comparator_correction_v1",
        "original_d012_outputs_preserved": True,
        "defect_corrected_in_this_artifact_only": bool(audit["defect_exists"]),
        "unchanged_quantities": ["B", "C", "C_minus_B"],
        "corrected_o150": {
            "B": sum(row["b_macro_st_rae"] for row in rows) / 150,
            "C": sum(row["c_macro_st_rae"] for row in rows) / 150,
            "T121": sum(row["corrected_t121_macro_st_rae"] for row in rows) / 150,
        },
        "corrected_paired_differences": {
            "B_minus_T121": diff_stats([row["corrected_delta_b_minus_t121"] for row in rows]),
            "C_minus_T121": diff_stats([row["corrected_delta_c_minus_t121"] for row in rows]),
            "C_minus_B": diff_stats([row["delta_c_minus_b"] for row in rows]),
        },
        "rows": rows,
    }


def grouped_descriptions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in records:
        for dimension in ("all_150_cells", "fold", "repetition", "training_seed"):
            if dimension == "all_150_cells":
                level = "all"
            else:
                level = str(row[dimension])
            groups[(row["metric"], dimension, level)].append(float(row["value"]))
    output = []
    for (metric, dimension, level), values in sorted(groups.items()):
        row = {"metric": metric, "dimension": dimension, "level": level}
        row.update(diff_stats(values))
        output.append(row)
    return output


def audit_b() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    d012_rows = load_d012_rows()
    t121 = load_t121_cells()
    b, c = load_b_c_metrics(d012_rows)
    records = []
    for key, row in d012_rows.items():
        base = {
            "cell_key": key,
            "fold": int(row["fold"]),
            "repetition": int(row["repetition"]),
            "training_seed": int(row["training_seed"]),
        }
        macro_values = {
            "macro:B_minus_T121": b[key]["macro_st_rae"] - t121[key]["macro_st_rae"],
            "macro:C_minus_T121": c[key]["macro_st_rae"] - t121[key]["macro_st_rae"],
            "macro:C_minus_B": c[key]["macro_st_rae"] - b[key]["macro_st_rae"],
        }
        for metric, value in macro_values.items():
            records.append({**base, "metric": metric, "value": value})
        for task in TASKS:
            records.append({
                **base,
                "metric": f"{task}:C_minus_T121",
                "value": c[key]["per_endpoint_st_rae"][task] - t121[key]["per_endpoint_st_rae"][task],
            })
    descriptions = grouped_descriptions(records)
    unavailable = {
        "B_endpoint_values_available": all(b[key]["per_endpoint_st_rae"] is not None for key in b),
        "unavailable_endpoint_comparisons": ["endpoint:B_minus_T121", "endpoint:C_minus_B"],
        "reason": "Arm B staged target metrics preserve best_macro_st_rae and history only; no per-endpoint B values were present, and predictions were not regenerated.",
    }
    summary = {
        "artifact_type": "D012_posthoc_audit_B_descriptive_decomposition_v1",
        "cell_count": 150,
        "difference_sign_convention": "Negative = first method better/lower ST-RAE.",
        "independence_boundary": "Descriptive only; the 150 cells are not treated as independent replicates and no significance claims are made.",
        "available_comparisons": ["macro:B_minus_T121", "macro:C_minus_T121", "macro:C_minus_B", "endpoint:C_minus_T121"],
        "limitations": unavailable,
        "all_150_main": [row for row in descriptions if row["dimension"] == "all_150_cells"],
    }
    return summary, descriptions


def openadmet_name_to_key() -> dict[str, str]:
    result = {}
    for row in read_tsv(CROSS_SOURCE):
        if row["openadmet_partitions"] != "train":
            continue
        for name in row["openadmet_molecule_names"].split(";"):
            if name:
                result[name] = row["full_standard_inchikey"]
    return result


def audit_c() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    aux_all = read_tsv(AUX_INSTANCES)
    overlap = [row for row in aux_all if row["routing_class"] == "overlap_openadmet_train" and row["overlap_openadmet_train"] == "true"]
    name_to_key = openadmet_name_to_key()
    train_by_key: dict[str, dict[str, str]] = {}
    for row in read_csv(TRAIN):
        key = name_to_key.get(row["Molecule_Name"])
        if key:
            train_by_key[key] = row

    observation_rows = []
    missing_train_endpoint = 0
    for row in overlap:
        iso = row["isoform"]
        task = ISO_TO_TASK[iso]
        key = row["full_standard_inchikey"]
        train_row = train_by_key.get(key)
        train_value = safe_float(train_row.get(task) if train_row else None)
        aux_value = float(row["median_derived_pic50"])
        if train_value is None:
            missing_train_endpoint += 1
        observation_rows.append({
            "isoform": iso,
            "task_unit_id": row["task_unit_id"],
            "source_assay_chembl_ids": row["source_assay_chembl_ids"],
            "full_standard_inchikey": key,
            "auxiliary_pic50": aux_value,
            "train_pic50": train_value,
            "paired": train_value is not None,
            "auxiliary_minus_train": aux_value - train_value if train_value is not None else None,
        })

    counts_by_cyp = Counter(row["isoform"] for row in overlap)
    matched_by_cyp = Counter(row["isoform"] for row in observation_rows if row["paired"])
    multiplicity = Counter((row["full_standard_inchikey"], row["isoform"]) for row in observation_rows)
    multiplicity_rows = [
        {"full_standard_inchikey": key, "isoform": iso, "assay_observation_count": count}
        for (key, iso), count in sorted(multiplicity.items())
    ]

    by_cyp = []
    for iso in sorted(ISO_TO_TASK):
        pairs = [(row["auxiliary_pic50"], row["train_pic50"]) for row in observation_rows if row["isoform"] == iso and row["paired"]]
        stat = correlation_stats(pairs)
        by_cyp.append({"level": "assay_observation", "isoform": iso, **stat})

    molecule_pairs_by_cyp: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    train_value_by_cyp_key: dict[tuple[str, str], float] = {}
    for row in observation_rows:
        if not row["paired"]:
            continue
        molecule_pairs_by_cyp[row["isoform"]][row["full_standard_inchikey"]].append(row["auxiliary_pic50"])
        train_value_by_cyp_key[(row["isoform"], row["full_standard_inchikey"])] = row["train_pic50"]
    molecule_rows = []
    molecule_by_cyp = []
    for iso in sorted(ISO_TO_TASK):
        pairs = []
        for key, values in sorted(molecule_pairs_by_cyp[iso].items()):
            aux_median = statistics.median(values)
            train_value = train_value_by_cyp_key[(iso, key)]
            pairs.append((aux_median, train_value))
            molecule_rows.append({
                "isoform": iso,
                "full_standard_inchikey": key,
                "auxiliary_molecule_endpoint_median_pic50": aux_median,
                "train_pic50": train_value,
                "assay_observation_count": len(values),
                "auxiliary_minus_train": aux_median - train_value,
            })
        molecule_by_cyp.append({"level": "molecule_endpoint_median", "isoform": iso, **correlation_stats(pairs)})

    paired_keys = {row["full_standard_inchikey"] for row in observation_rows if row["paired"]}
    summary = {
        "artifact_type": "D012_posthoc_audit_C_auxiliary_train_identity_concordance_v1",
        "rules": {
            "identity": "full Standard InChIKey identity",
            "train_labels_only": True,
            "auxiliary_records": "already-eligible exact uncensored IC50 records",
            "conversion": "existing IC50 to pIC50 conversion from D012 auxiliary population",
            "aggregation": "existing median within assay x identity replicate aggregation; established duplicate-assay consolidation preserved",
            "endpoint_pairing": "corresponding CYP endpoint only; missing TRAIN endpoint counted but not paired",
        },
        "population_counts": {
            "overlap_openadmet_train_auxiliary_task_identity_instances": len(overlap),
            "unique_auxiliary_full_standard_inchikeys": len({row["full_standard_inchikey"] for row in overlap}),
            "train_molecules_represented_by_identity_bridge": len(train_by_key),
            "assay_identity_observations": len(observation_rows),
            "missing_train_endpoint_observations_counted_not_paired": missing_train_endpoint,
            "matched_nonmissing_observations": sum(row["paired"] for row in observation_rows),
            "matched_unique_molecules": len(paired_keys),
            "counts_by_cyp": dict(sorted(counts_by_cyp.items())),
            "matched_nonmissing_observations_by_cyp": dict(sorted(matched_by_cyp.items())),
            "assay_observation_multiplicity_per_molecule_endpoint": stats(list(multiplicity.values())),
        },
        "assay_observation_concordance_by_cyp": by_cyp,
        "molecule_level_concordance_by_cyp": molecule_by_cyp,
        "interpretation_boundary": "Poor agreement supports measurement/task mismatch as a hypothesis, not causation; strong agreement in this selected overlap does not clear other assays or chemical space.",
        "provenance": {
            "auxiliary_instances": {"path": str(AUX_INSTANCES.relative_to(ROOT)), "sha256": sha256_file(AUX_INSTANCES)},
            "cross_source_matches": {"path": str(CROSS_SOURCE.relative_to(ROOT)), "sha256": sha256_file(CROSS_SOURCE)},
            "train_labels": {"path": str(TRAIN.relative_to(ROOT)), "sha256": sha256_file(TRAIN)},
        },
    }
    return summary, observation_rows, molecule_rows


def write_outputs(output_dir: Path = OUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    a = audit_a()
    correction = build_correction(a)
    b_summary, b_rows = audit_b()
    c_summary, c_obs_rows, c_mol_rows = audit_c()
    summary = {
        "artifact_type": "D012_posthoc_audit_A_C_summary_v1",
        "scope": "Descriptive post-hoc audit A-C only; no training, prediction regeneration, filtering, tuning, D/E, or D013.",
        "A": a,
        "A_correction": {key: value for key, value in correction.items() if key != "rows"},
        "B": b_summary,
        "C": c_summary,
    }

    files = {
        "d012_posthoc_audit_ac_summary.json": summary,
        "d012_posthoc_audit_A_comparator_provenance.json": a,
        "d012_posthoc_audit_A_comparator_correction.json": correction,
        "d012_posthoc_audit_B_decomposition_summary.json": b_summary,
        "d012_posthoc_audit_C_concordance_summary.json": c_summary,
    }
    for name, payload in files.items():
        write_json(output_dir / name, payload)
    write_tsv(output_dir / "d012_posthoc_audit_B_decomposition.tsv", b_rows, [
        "metric", "dimension", "level", "n", "mean", "sample_sd", "median", "min", "max",
        "first_method_wins_lower_is_better", "second_method_wins_lower_is_better", "ties",
    ])
    write_tsv(output_dir / "d012_posthoc_audit_C_observation_pairs.tsv", c_obs_rows, [
        "isoform", "task_unit_id", "source_assay_chembl_ids", "full_standard_inchikey", "auxiliary_pic50", "train_pic50", "paired", "auxiliary_minus_train",
    ])
    write_tsv(output_dir / "d012_posthoc_audit_C_molecule_view.tsv", c_mol_rows, [
        "isoform", "full_standard_inchikey", "auxiliary_molecule_endpoint_median_pic50", "train_pic50", "assay_observation_count", "auxiliary_minus_train",
    ])

    output_hashes = {}
    for path in sorted(output_dir.iterdir()):
        if path.name in {"SHA256SUMS.txt", "d012_posthoc_audit_ac_summary.json"} or not path.is_file():
            continue
        output_hashes[path.name] = sha256_file(path)
    summary["output_hashes"] = output_hashes
    summary["summary_hash_note"] = "d012_posthoc_audit_ac_summary.json is hashed in SHA256SUMS.txt after this embedded map is written to avoid self-referential hashing."
    write_json(output_dir / "d012_posthoc_audit_ac_summary.json", summary)
    with (output_dir / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for path in sorted(output_dir.iterdir()):
            if path.name == "SHA256SUMS.txt" or not path.is_file():
                continue
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D012 post-hoc audit A-C only.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    if args.write_outputs:
        summary = write_outputs(args.output_dir)
    else:
        a = audit_a()
        correction = build_correction(a)
        b, _ = audit_b()
        c, _, _ = audit_c()
        summary = {"A": a, "A_correction": {key: value for key, value in correction.items() if key != "rows"}, "B": b, "C": c}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
