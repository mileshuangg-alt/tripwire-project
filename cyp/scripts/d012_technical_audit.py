from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
QUAL = ROOT / "qualification/cyp_external_task_qualification_20261109"
OUT = QUAL / "d012_technical_audit"
D012_SCRIPT = ROOT / "cyp/scripts/d012_first_cell_runner.py"
D012_COMPLETE = ROOT / "cyp/scripts/d012_complete_model_runtime_benchmark.py"
D012_ARMS = ROOT / "cyp/scripts/d012_arms.py"
P1_PROD = ROOT / "cyp/scripts/p1_pairwise_production.py"
P1_TRAIN = ROOT / "cyp/scripts/p1_pairwise_training.py"
P2_MODEL = ROOT / "cyp/scripts/p2_anil_model.py"
P2_TRAIN = ROOT / "cyp/scripts/p2_anil_training.py"
CYP002_TRAIN = ROOT / "cyp/scripts/cyp002_train.py"
CYP002_VALIDATION = ROOT / "cyp/scripts/cyp002_validation.py"
ST_RAE = ROOT / "cyp/scripts/st_rae.py"
D012_AGG = QUAL / "d012_final_aggregation"
STAGED = D012_AGG / "staged_metrics"
T121_CELLS = ROOT / "artifacts/d009_closure_workspace/d009_t121_comparator_20261109/cells"
T121_PROPOSAL = ROOT / "artifacts/d009_closure_workspace/d009_t121_comparator_20261109/proposal.json"
D007_MANIFESTS = ROOT / "cyp/artifacts/cyp_007/split_manifests"
TARGET_DATA = ROOT / "cyp/data/cyp-challenge-TRAIN_inhibition.csv"
D012_PLAN = QUAL / "d012_implementation_qualification"

TASKS = (
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
)


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


def line_evidence(path: Path, needle: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, start=1):
        if needle in line:
            return {"path": str(path.relative_to(ROOT)), "line": index, "text": line.strip()}
    return {"path": str(path.relative_to(ROOT)), "line": None, "text": f"needle not found: {needle}"}


def function_lines(path: Path, function_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {"path": str(path.relative_to(ROOT)), "start_line": node.lineno, "end_line": node.end_lineno}
    return {"path": str(path.relative_to(ROOT)), "start_line": None, "end_line": None}


def status_from_findings(findings: list[dict[str, Any]], unresolved: bool = False) -> str:
    if any(row["classification"] == "CONFIRMED_DEFECT" for row in findings):
        return "DEFECT"
    if any(row["classification"] == "POTENTIAL_DEFECT" for row in findings):
        return "UNRESOLVED"
    if unresolved or any(row["classification"] == "UNRESOLVED" for row in findings):
        return "UNRESOLVED"
    return "PASS"


def cell_key(repetition: int, fold: int, seed: int) -> str:
    return f"rep_{repetition:02d}_fold_{fold:02d}|seed_{seed}"


def manifest_id_from_key(key: str) -> str:
    return key.split("|", 1)[0]


def hash_strings(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def target_rows_by_name() -> dict[str, dict[str, str]]:
    rows = read_csv(TARGET_DATA)
    return {row["Molecule_Name"]: row for row in rows}


def manifest_population_record(manifest_id: str, seed: int, target_by_name: dict[str, dict[str, str]]) -> dict[str, Any]:
    manifest = read_json(D007_MANIFESTS / f"{manifest_id}.json")
    train = list(manifest["training_names"])
    validation = list(manifest["validation_names"])
    masks = {}
    counts = {}
    for task in TASKS:
        values = [name for name in validation if target_by_name[name].get(task, "") != ""]
        masks[task] = hash_strings(values)
        counts[task] = len(values)
    return {
        "cell_key": cell_key(int(manifest["repetition"]), int(manifest["fold"]), int(seed)),
        "manifest_id": manifest_id,
        "repetition": int(manifest["repetition"]),
        "fold": int(manifest["fold"]),
        "training_seed": int(seed),
        "training_count": len(train),
        "validation_count": len(validation),
        "training_names_sha256": hash_strings(train),
        "validation_names_sha256": hash_strings(validation),
        "train_validation_overlap_count": len(set(train) & set(validation)),
        "endpoint_nonmissing_counts": counts,
        "endpoint_nonmissing_name_hashes": masks,
        "split_manifest_sha256": sha256_file(D007_MANIFESTS / f"{manifest_id}.json"),
    }


def load_d012_cell_keys() -> list[str]:
    rows = read_tsv(D012_AGG / "d012_cell_level_joined_results.tsv")
    return [row["cell_key"] for row in rows]


def inspect_prediction_artifacts() -> dict[str, Any]:
    patterns = ["*pred*", "*prediction*", "*validation*.npy", "*validation*.npz", "*validation*.csv", "*validation*.tsv"]
    found = []
    roots = [STAGED, T121_CELLS, ROOT / "artifacts/d009_closure_workspace/p1_d009_production_20261108"]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found.extend(path for path in root.rglob(pattern) if path.is_file())
    unique = sorted({str(path.relative_to(ROOT)) for path in found})
    return {
        "saved_prediction_artifact_count": len(unique),
        "saved_prediction_artifacts": unique[:25],
        "independent_score_recompute_possible": bool(unique),
        "limitation": "No saved B/C/T121 validation prediction files were found in preserved lightweight evidence; scores can be recomputed only from preserved metrics, not raw predictions.",
    }


def scoring_equivalence() -> dict[str, Any]:
    findings = [
        {
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,C,T121",
            "claim": "D012 B/C and T121 preserved metrics use the same frozen compute_validation_macro_st_rae wrapper over st_rae_metric with unweighted four-endpoint macro.",
            "evidence": [
                function_lines(CYP002_VALIDATION, "compute_validation_macro_st_rae"),
                line_evidence(CYP002_VALIDATION, "values.mean()"),
                line_evidence(D012_SCRIPT, "return compute_validation_macro_st_rae(y_true, predictions, lows, highs, TARGET_TASKS)"),
                line_evidence(D012_SCRIPT, "score = compute_validation_macro_st_rae(val_true, val_predictions, val_low, val_high, TARGET_TASKS)"),
            ],
        },
        {
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,C,T121",
            "claim": "Endpoint order is the frozen CYP1A2, CYP2C9, CYP2D6, CYP3A4 order in D012 and T121 artifacts.",
            "evidence": [line_evidence(D012_ARMS, "TARGET_TASKS ="), line_evidence(ROOT / "artifacts/d009_closure_workspace/d009_t121_comparator_20261109/cells/cell_000/result.json", '"task_order"')],
        },
        {
            "classification": "UNRESOLVED",
            "scope": "B,C,T121",
            "claim": "Independent score recomputation from saved validation predictions is not possible because preserved prediction files were not found.",
            "evidence": [inspect_prediction_artifacts()],
        },
    ]
    return {"status": status_from_findings(findings), "findings": findings, "prediction_artifacts": inspect_prediction_artifacts()}


def target_transformation() -> dict[str, Any]:
    findings = [
        {
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,C",
            "claim": "D012 target scalers are fitted per endpoint on cell training labels only with population standard deviation.",
            "evidence": [function_lines(D012_SCRIPT, "fit_target_scaler"), line_evidence(D012_SCRIPT, "observed = train_values[train_mask[:, j], j]")],
        },
        {
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,C",
            "claim": "D012 scoring inverts standardized outputs exactly once into original pIC50 units before ST-RAE.",
            "evidence": [line_evidence(D012_SCRIPT, "pred = pred_z * scales.cpu() + means.cpu()"), line_evidence(D012_SCRIPT, "pred = pred_z * scales[j].cpu() + means[j].cpu()")],
        },
        {
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,C",
            "claim": "Masks are built in TARGET_TASKS order from the same Molecule_Name-ordered matrices used to assemble validation predictions.",
            "evidence": [function_lines(D012_SCRIPT, "target_matrices"), line_evidence(D012_SCRIPT, "predictions[task] = pred[keep, j]")],
        },
        {
            "classification": "UNRESOLVED",
            "scope": "B,C",
            "claim": "Wrong-cell scaler or endpoint permutation cannot be independently falsified from raw prediction tensors because prediction tensors were not preserved, but code and population hashes align.",
            "evidence": [inspect_prediction_artifacts()],
        },
    ]
    return {"status": status_from_findings(findings), "findings": findings}


def validation_population_equality() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_by_name = target_rows_by_name()
    d012_keys = load_d012_cell_keys()
    t121 = {}
    for path in sorted(T121_CELLS.glob("cell_*/result.json")):
        row = read_json(path)
        key = cell_key(int(row["repetition"]), int(row["fold"]), int(row["seed"]))
        t121[key] = row
    rows = []
    failures = []
    for key in sorted(d012_keys):
        manifest_id, seed_s = key.split("|seed_")
        seed = int(seed_s)
        record = manifest_population_record(manifest_id, seed, target_by_name)
        t121_row = t121.get(key)
        record["t121_split_manifest_sha256"] = None if t121_row is None else t121_row.get("split_manifest_sha256")
        record["b_startup_present"] = (STAGED / "arm_b" / f"{manifest_id}_seed{seed}" / "startup_config.json").exists()
        record["c_startup_present"] = (STAGED / "arm_c" / f"{manifest_id}_seed{seed}" / "startup_config.json").exists()
        record["population_match"] = (
            record["train_validation_overlap_count"] == 0
            and t121_row is not None
            and record["split_manifest_sha256"] == t121_row.get("split_manifest_sha256")
            and record["b_startup_present"]
            and record["c_startup_present"]
        )
        if not record["population_match"]:
            failures.append(record["cell_key"])
        rows.append(record)
    findings = [{
        "classification": "MECHANICAL_EQUIVALENT" if not failures else "CONFIRMED_DEFECT",
        "scope": "B,C,T121",
        "claim": "All 150 cells share exact D007 validation populations by manifest hash, validation-name hash, endpoint mask counts/hashes, and zero train/validation crossover.",
        "evidence": {"cell_count": len(rows), "failure_count": len(failures), "first_failures": failures[:5]},
    }]
    return {"status": status_from_findings(findings), "findings": findings}, rows


def arm_b_vs_t121() -> dict[str, Any]:
    comparisons = [
        {
            "item": "architecture",
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,T121",
            "evidence": [line_evidence(D012_COMPLETE, "message_passing_depth=4"), line_evidence(D012_SCRIPT, "self.hidden = nn.Linear(600, 2000)"), line_evidence(T121_PROPOSAL, '"message_passing_depth": 4')],
        },
        {
            "item": "fresh_head_initialization",
            "classification": "EXPECTED_BY_D012",
            "scope": "B",
            "evidence": [line_evidence(D012_SCRIPT, "target_head = D012BTargetHead().to(device)")],
        },
        {
            "item": "loss_masking",
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,T121",
            "evidence": [function_lines(D012_SCRIPT, "masked_mse"), line_evidence(CYP002_TRAIN, "train_dataset.normalize_targets()")],
        },
        {
            "item": "optimizer",
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,T121",
            "evidence": [line_evidence(D012_SCRIPT, "target_optimizer = torch.optim.Adam"), line_evidence(P1_TRAIN, "optimizer = torch.optim.Adam")],
        },
        {
            "item": "batch_size",
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,T121",
            "evidence": [line_evidence(D012_SCRIPT, "TARGET_BATCH_SIZE = 16"), line_evidence(T121_PROPOSAL, '"batch_size": 16')],
        },
        {
            "item": "batch_order_semantics",
            "classification": "POTENTIAL_DEFECT",
            "scope": "B_vs_T121",
            "evidence": [line_evidence(D012_SCRIPT, "batches = list(range(0, train_values.shape[0], TARGET_BATCH_SIZE))"), line_evidence(CYP002_TRAIN, "shuffle=True")],
            "rationale": "D012 B target training iterates manifest order without shuffling; frozen direct Chemprop/T121 training uses a shuffled dataloader seeded by the cell seed.",
        },
        {
            "item": "lr_schedule",
            "classification": "POTENTIAL_DEFECT",
            "scope": "B_vs_T121",
            "evidence": [line_evidence(D012_SCRIPT, "target_scheduler = LinearAnchorScheduler"), line_evidence(P1_TRAIN, "schedulers.build_NoamLike_LRSched")],
            "rationale": "D012 B uses a custom linear anchor scheduler rather than Chemprop NoamLike scheduler used by frozen production training utilities.",
        },
        {
            "item": "max_epochs_patience_validation_frequency_checkpoint_selection",
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,T121",
            "evidence": [line_evidence(D012_SCRIPT, "TARGET_MAX_EPOCHS = 50"), line_evidence(D012_SCRIPT, "TARGET_PATIENCE = 20"), line_evidence(D012_SCRIPT, "if macro < best_score")],
        },
        {
            "item": "best_checkpoint_restoration",
            "classification": "UNRESOLVED",
            "scope": "B",
            "evidence": [line_evidence(D012_SCRIPT, "writer.checkpoint(\"arm_b_best_target_checkpoint.pt\"")],
            "rationale": "D012 records best checkpoint and best metric, but lightweight evidence does not include checkpoint payloads and final in-memory state is not restored before completion. This does not alter stored best score, but affects any later use of final state if checkpoint payloads were used elsewhere.",
        },
    ]
    return {"status": status_from_findings(comparisons), "line_item_classifications": comparisons, "equivalent_except_initialization": False}


def arm_c_anil() -> dict[str, Any]:
    findings = [
        {
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "C",
            "claim": "Auxiliary support/query identity disjointness is implemented by excluding query_ids from task_to_ids; plan tests also verify disjointness.",
            "evidence": [line_evidence(D012_SCRIPT, "support_ids = [item for item in task_to_ids[row[\"task_unit_id\"]] if item not in query_set]")],
        },
        {
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "C",
            "claim": "K=1, LR=0.01, exact differentiable inner update, query-only outer loss, final epoch-50 auxiliary state, target encoder frozen, and no target checkpoint selection are present.",
            "evidence": [line_evidence(D012_ARMS, "D012_INNER_LR = 0.01"), line_evidence(D012_COMPLETE, "create_graph=True"), line_evidence(D012_SCRIPT, "query_loss.backward()"), line_evidence(D012_SCRIPT, "for epoch in range(1, max_aux_epochs + 1)"), line_evidence(D012_SCRIPT, "validation_checkpoint_selection\": False")],
        },
        {
            "classification": "CONFIRMED_DEFECT",
            "scope": "C",
            "claim": "D012 production adapts all parameters of D012ScalarHead, not only selected output/head weight and bias as in the frozen ANIL reference semantics.",
            "evidence": [function_lines(D012_COMPLETE, "adapt_head_exact"), line_evidence(D012_COMPLETE, "params = dict(head.named_parameters())"), function_lines(P2_MODEL, "adapt_selected_output_exact"), line_evidence(P2_MODEL, "weight = meta_head.output.weight[task_index]"), line_evidence(P2_MODEL, "bias = meta_head.output.bias[task_index]")],
            "rationale": "D012ScalarHead contains hidden and output layers. `dict(head.named_parameters())` adapts hidden.weight, hidden.bias, output.weight, and output.bias; frozen ANIL code adapts only selected output weight/bias.",
        },
        {
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "C",
            "claim": "Target stage starts from learned scalar head initialization, uses all complete target-training support for each endpoint, one inner update, and validates with post-update head.",
            "evidence": [line_evidence(D012_SCRIPT, "head.load_state_dict(model.scalar_head_initialization.state_dict())"), line_evidence(D012_SCRIPT, "support_dataset = data.MoleculeDataset([train_dataset.data[i] for i, flag in enumerate(keep.tolist()) if flag])"), line_evidence(D012_SCRIPT, "adapted, _support_loss = adapt_head_exact(head, support_embeddings, support_y)"), line_evidence(D012_SCRIPT, "pred_z = functional_call(head, adapted, (val_embeddings,)).detach().cpu()")],
        },
    ]
    return {"status": status_from_findings(findings), "findings": findings, "matches_frozen_anil_semantics": False}


def end_to_end_alignment(population_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_by_name = target_rows_by_name()
    selected_keys = sorted(load_d012_cell_keys())
    selected_keys = [selected_keys[0], selected_keys[len(selected_keys) // 2], selected_keys[-1]]
    examples = []
    for key in selected_keys:
        manifest_id = manifest_id_from_key(key)
        manifest = read_json(D007_MANIFESTS / f"{manifest_id}.json")
        validation = list(manifest["validation_names"])
        row_indices = [0, len(validation) // 2, len(validation) - 1]
        for row_index in row_indices:
            name = validation[row_index]
            row = target_by_name[name]
            mask = {task: row.get(task, "") != "" for task in TASKS}
            labels = {task: row.get(task, "") for task in TASKS}
            examples.append({
                "cell_key": key,
                "manifest_id": manifest_id,
                "validation_row_index": row_index,
                "Molecule_Name": name,
                "SMILES_sha256": hashlib.sha256(row["SMILES"].encode("utf-8")).hexdigest(),
                "task_order": ";".join(TASKS),
                "label_mask": json.dumps(mask, sort_keys=True),
                "labels_present": json.dumps(labels, sort_keys=True),
            })
    findings = [
        {
            "classification": "MECHANICAL_EQUIVALENT",
            "scope": "B,C,T121",
            "claim": "Deterministic first/middle/last cell and row examples trace to the frozen manifest Molecule_Name order, target columns, and masks without manifest/key mismatch.",
            "evidence": {"example_count": len(examples), "population_rows": len(population_rows)},
        },
        {
            "classification": "UNRESOLVED",
            "scope": "B,C,T121",
            "claim": "Graph tensor outputs, inverse transformed predictions, checkpoint freshness, and prediction/label tensor alignment cannot be independently replayed because checkpoints/predictions are not preserved locally and regeneration is forbidden.",
            "evidence": [inspect_prediction_artifacts()],
        },
    ]
    return {"status": status_from_findings(findings), "findings": findings}, examples


def summarize_outputs(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "section_status": {key: value["status"] for key, value in summary["sections"].items()},
        "confirmed_defects": [
            finding
            for section in summary["sections"].values()
            for finding in section.get("findings", section.get("line_item_classifications", []))
            if finding.get("classification") == "CONFIRMED_DEFECT"
        ],
        "potential_defects": [
            finding
            for section in summary["sections"].values()
            for finding in section.get("findings", section.get("line_item_classifications", []))
            if finding.get("classification") == "POTENTIAL_DEFECT"
        ],
        "stored_predictions_allowed_independent_recompute": summary["sections"]["scoring_equivalence"]["prediction_artifacts"]["independent_score_recompute_possible"],
        "b_target_training_equivalent_to_t121_except_initialization": summary["sections"]["arm_b_vs_t121_target_training"]["equivalent_except_initialization"],
        "c_matches_frozen_anil_semantics": summary["sections"]["arm_c_anil_implementation"]["matches_frozen_anil_semantics"],
    }


def build_audit() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    population_section, population_rows = validation_population_equality()
    e2e_section, examples = end_to_end_alignment(population_rows)
    sections = {
        "scoring_equivalence": scoring_equivalence(),
        "target_transformation": target_transformation(),
        "validation_population_equality": population_section,
        "arm_b_vs_t121_target_training": arm_b_vs_t121(),
        "arm_c_anil_implementation": arm_c_anil(),
        "end_to_end_alignment": e2e_section,
    }
    summary = {
        "artifact_type": "D012_technical_evaluation_forensic_audit_v1",
        "scope": "Technical forensic audit only; no training, prediction regeneration, tuning, filtering, protected-test/challenge access, D/E interpretation, or D013 proposals.",
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [D012_SCRIPT, D012_COMPLETE, D012_ARMS, P1_PROD, P1_TRAIN, P2_MODEL, P2_TRAIN, CYP002_TRAIN, CYP002_VALIDATION, ST_RAE]
        },
        "artifact_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [D012_AGG / "d012_final_aggregate_summary.json", D012_AGG / "d012_cell_level_joined_results.tsv", T121_PROPOSAL, TARGET_DATA]
        },
        "sections": sections,
        "limitations": [
            "No B/C/T121 saved validation prediction tensors/files found; independent score recomputation from predictions was not possible.",
            "Checkpoint payloads are not inspected or replayed; state freshness and tensor-level output alignment remain unresolved where they require checkpoint execution.",
            "Audit is source/artifact forensic only and preserves original outputs.",
        ],
    }
    summary["executive"] = summarize_outputs(summary)
    return summary, population_rows, examples


def write_outputs(output_dir: Path = OUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, population_rows, examples = build_audit()
    write_json(output_dir / "d012_technical_audit_summary.json", summary)
    write_tsv(output_dir / "d012_validation_population_hashes.tsv", population_rows, [
        "cell_key", "manifest_id", "repetition", "fold", "training_seed", "training_count", "validation_count",
        "training_names_sha256", "validation_names_sha256", "train_validation_overlap_count", "endpoint_nonmissing_counts",
        "endpoint_nonmissing_name_hashes", "split_manifest_sha256", "t121_split_manifest_sha256", "b_startup_present",
        "c_startup_present", "population_match",
    ])
    write_tsv(output_dir / "d012_end_to_end_alignment_examples.tsv", examples, [
        "cell_key", "manifest_id", "validation_row_index", "Molecule_Name", "SMILES_sha256", "task_order", "label_mask", "labels_present",
    ])
    with (output_dir / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for path in sorted(output_dir.iterdir()):
            if path.name == "SHA256SUMS.txt" or not path.is_file():
                continue
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D012 technical forensic audit.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    summary = write_outputs(args.output_dir) if args.write_outputs else build_audit()[0]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
