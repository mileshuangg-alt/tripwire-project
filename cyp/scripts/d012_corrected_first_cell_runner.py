from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path

from chemprop import data
import torch

from d012_arms import D012_FINAL_LR, D012_INITIAL_LR, D012_MAX_LR, ROOT, TARGET_TASKS, configure_determinism, load_manifest, scheduler_budget_for_cell, split_semicolon, stable_seed
from d012_complete_model_runtime_benchmark import D012ScalarHead, LinearAnchorScheduler, mse
from d012_corrective_patch import (
    CORRECTED_FIRST_CELL_ROOT,
    adapt_selected_output_exact_d012,
    chemprop_noam_target_scheduler,
    functional_scalar_head_prediction,
)
from d012_first_cell_runner import (
    ArmBProductionModel,
    ArmCProductionModel,
    ArtifactWriter,
    TARGET_BATCH_SIZE,
    TARGET_MAX_EPOCHS,
    TARGET_PATIENCE,
    build_startup_config,
    dataset_for_instances,
    fit_target_scaler,
    instance_lookup,
    load_instances,
    masked_mse,
    memory_snapshot,
    plan_rows,
    predict_target,
    read_csv,
    retained_instance_ids_from_b_plan,
    score_validation,
    target_matrices,
    D012_PLAN,
    TARGET_DATA,
)
from cyp002_validation import compute_validation_macro_st_rae


def run_arm_b_corrected(
    writer: ArtifactWriter,
    device: torch.device,
    *,
    manifest_id: str,
    training_seed: int,
    max_aux_epochs: int,
    max_target_epochs: int,
) -> None:
    instances = load_instances()
    lookup = instance_lookup(instances)
    task_ids = sorted({row["task_unit_id"] for row in instances})
    b_rows = plan_rows(D012_PLAN / "d012_arm_b_epoch_plan.tsv", manifest_id=manifest_id, training_seed=training_seed)
    budget = scheduler_budget_for_cell(manifest_id, training_seed)

    with writer.time_phase("model_construction"):
        model = ArmBProductionModel(task_ids, seed=training_seed).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=D012_INITIAL_LR)
        scheduler = LinearAnchorScheduler(optimizer, int(budget["b_warmup_steps"]), int(budget["b_total_steps"]))

    with writer.time_phase("auxiliary_training"):
        update = 0
        for epoch in range(1, max_aux_epochs + 1):
            for row in b_rows:
                dataset, labels, task = dataset_for_instances(split_semicolon(row["instance_ids"]), lookup)
                labels = labels.to(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                pred = model.auxiliary_heads[task](model.encoder.forward_dataset(dataset, device=device, batch_size=16))
                loss = (pred - labels).square().mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                update += 1
            writer.event("arm_b_aux_epoch_complete", epoch=epoch, updates=update)
        writer.checkpoint("arm_b_auxiliary_final.pt", {"encoder": model.encoder.state_dict(), "auxiliary_heads": model.auxiliary_heads.state_dict(), "updates": update})

    with writer.time_phase("target_preparation"):
        manifest = load_manifest(manifest_id)
        target_rows = read_csv(TARGET_DATA)
        train_dataset, train_values, train_mask, _train_true, _train_low, _train_high = target_matrices(target_rows, list(manifest["training_names"]))
        val_dataset, val_values, val_mask, val_true, val_low, val_high = target_matrices(target_rows, list(manifest["validation_names"]))
        means, scales = fit_target_scaler(train_values, train_mask)
        train_z = (train_values - means) / scales
        train_z[~train_mask] = float("nan")
        val_values = val_values.to(device)
        val_mask = val_mask.to(device)
        means = means.to(device)
        scales = scales.to(device)

    with writer.time_phase("target_training_checkpoint_selection"):
        target_head = __import__("d012_first_cell_runner", fromlist=["D012BTargetHead"]).D012BTargetHead().to(device)
        target_optimizer = torch.optim.Adam([*model.encoder.parameters(), *target_head.parameters()], lr=D012_INITIAL_LR)
        batches = list(range(0, train_values.shape[0], TARGET_BATCH_SIZE))
        target_scheduler = chemprop_noam_target_scheduler(
            target_optimizer,
            updates_per_epoch=len(batches),
            warmup_epochs=10,
            max_epochs=TARGET_MAX_EPOCHS,
            init_lr=D012_INITIAL_LR,
            max_lr=D012_MAX_LR,
            final_lr=D012_FINAL_LR,
        )
        best_score = math.inf
        best_epoch = None
        bad = 0
        history = []
        for epoch in range(1, max_target_epochs + 1):
            for start in batches:
                stop = min(start + TARGET_BATCH_SIZE, train_values.shape[0])
                batch_dataset = data.MoleculeDataset([train_dataset.data[i] for i in range(start, stop)])
                batch_y = train_z[start:stop].to(device)
                batch_mask = train_mask[start:stop].to(device)
                target_optimizer.zero_grad(set_to_none=True)
                pred = predict_target(model.encoder, target_head, batch_dataset, device=device)
                loss = masked_mse(pred, batch_y, batch_mask)
                loss.backward()
                target_optimizer.step()
                target_scheduler.step()
            score = score_validation(model.encoder, target_head, val_dataset, val_values, val_mask, means, scales, val_true, val_low, val_high, device=device)
            macro = float(score["macro_st_rae"])
            history.append({"epoch": epoch, "macro_st_rae": macro})
            writer.event("arm_b_validation", epoch=epoch, macro_st_rae=macro)
            if macro < best_score:
                best_score = macro
                best_epoch = epoch
                bad = 0
                writer.checkpoint("arm_b_best_target_checkpoint.pt", {"encoder": model.encoder.state_dict(), "target_head": target_head.state_dict(), "epoch": epoch, "macro_st_rae": macro})
            else:
                bad += 1
                if bad >= TARGET_PATIENCE:
                    break
        writer.write_json("arm_b_target_metrics.json", {"best_epoch": best_epoch, "best_macro_st_rae": best_score, "history": history, "validation_checkpoint_selection": True, "correction": "target_scheduler_chemprop_noamlike"})


def run_arm_c_corrected(
    writer: ArtifactWriter,
    device: torch.device,
    *,
    manifest_id: str,
    training_seed: int,
    max_aux_epochs: int,
) -> None:
    instances = load_instances()
    lookup = instance_lookup(instances)
    task_ids = sorted({row["task_unit_id"] for row in instances})
    task_to_ids = {}
    for instance_id in retained_instance_ids_from_b_plan(manifest_id, training_seed):
        task_to_ids.setdefault(lookup[instance_id]["task_unit_id"], []).append(instance_id)
    c_rows = plan_rows(D012_PLAN / "d012_arm_c_epoch_plan.tsv", manifest_id=manifest_id, training_seed=training_seed)
    budget = scheduler_budget_for_cell(manifest_id, training_seed)

    with writer.time_phase("model_construction"):
        model = ArmCProductionModel(seed=training_seed).to(device)
        optimizer = torch.optim.Adam([*model.encoder.parameters(), *model.scalar_head_initialization.parameters()], lr=D012_INITIAL_LR)
        scheduler = LinearAnchorScheduler(optimizer, int(budget["c_warmup_steps"]), int(budget["c_total_steps"]))

    with writer.time_phase("auxiliary_training"):
        update = 0
        for epoch in range(1, max_aux_epochs + 1):
            for row in c_rows:
                query_ids = split_semicolon(row["query_instance_ids"])
                query_set = set(query_ids)
                support_ids = [item for item in task_to_ids[row["task_unit_id"]] if item not in query_set]
                support_dataset, support_y, _task = dataset_for_instances(support_ids, lookup)
                query_dataset, query_y, _task = dataset_for_instances(query_ids, lookup)
                support_y = support_y.to(device)
                query_y = query_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                support_embeddings = model.encoder.forward_dataset(support_dataset, device=device, batch_size=16)
                adapted, _support_loss = adapt_selected_output_exact_d012(model.scalar_head_initialization, support_embeddings, support_y)
                query_embeddings = model.encoder.forward_dataset(query_dataset, device=device, batch_size=16)
                query_loss = mse(functional_scalar_head_prediction(model.scalar_head_initialization, adapted, query_embeddings), query_y)
                query_loss.backward()
                optimizer.step()
                scheduler.step()
                update += 1
            writer.event("arm_c_aux_epoch_complete", epoch=epoch, updates=update)
        writer.checkpoint("arm_c_auxiliary_final.pt", {"encoder": model.encoder.state_dict(), "scalar_head_initialization": model.scalar_head_initialization.state_dict(), "updates": update})

    with writer.time_phase("target_preparation"):
        manifest = load_manifest(manifest_id)
        target_rows = read_csv(TARGET_DATA)
        train_dataset, train_values, train_mask, _train_true, _train_low, _train_high = target_matrices(target_rows, list(manifest["training_names"]))
        val_dataset, val_values, val_mask, val_true, val_low, val_high = target_matrices(target_rows, list(manifest["validation_names"]))
        means, scales = fit_target_scaler(train_values, train_mask)
        train_z = (train_values - means) / scales
        train_z[~train_mask] = float("nan")
        val_predictions = {}
        model.encoder.eval()

    with writer.time_phase("target_one_step_adaptation_and_validation"):
        for j, task in enumerate(TARGET_TASKS):
            keep = train_mask[:, j]
            support_dataset = data.MoleculeDataset([train_dataset.data[i] for i, flag in enumerate(keep.tolist()) if flag])
            support_y = train_z[keep, j].to(device)
            head = D012ScalarHead().to(device)
            head.load_state_dict(model.scalar_head_initialization.state_dict())
            support_embeddings = model.encoder.forward_dataset(support_dataset, device=device, batch_size=TARGET_BATCH_SIZE).detach()
            adapted, _support_loss = adapt_selected_output_exact_d012(head, support_embeddings, support_y)
            val_embeddings = model.encoder.forward_dataset(val_dataset, device=device, batch_size=TARGET_BATCH_SIZE).detach()
            pred_z = functional_scalar_head_prediction(head, adapted, val_embeddings).detach().cpu()
            pred = pred_z * scales[j].cpu() + means[j].cpu()
            val_predictions[task] = pred[val_mask[:, j]].numpy().astype("float64")
        score = compute_validation_macro_st_rae(val_true, val_predictions, val_low, val_high, TARGET_TASKS)
        writer.write_json("arm_c_validation_metrics.json", {"score": score, "validation_checkpoint_selection": False, "target_update": "selected_output_weight_bias_only_lr_0.01_k1_encoder_frozen", "correction": "selected_output_only_anil"})
        writer.checkpoint("arm_c_final_transfer_state.pt", {"encoder": model.encoder.state_dict(), "scalar_head_initialization": model.scalar_head_initialization.state_dict(), "validation_score": score})


def run(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest_id)
    if int(args.training_seed) not in set(manifest["training_seeds"]):
        raise ValueError("training seed is not present in the frozen D007 manifest")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cpu")
    configure_determinism(stable_seed("D012-corrected-production-cell", args.arm, args.manifest_id, args.training_seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    writer = ArtifactWriter(args.output_dir, arm=args.arm)
    start = time.perf_counter()
    writer.write_json("startup_config.json", {**build_startup_config(args.arm, device, manifest_id=args.manifest_id, training_seed=args.training_seed), "correction_artifact_type": "D012_corrected_first_cell_startup_v1"})
    writer.event("startup", arm=args.arm, corrected=True)
    try:
        if args.arm == "B":
            run_arm_b_corrected(writer, device, manifest_id=args.manifest_id, training_seed=args.training_seed, max_aux_epochs=args.max_aux_epochs, max_target_epochs=args.max_target_epochs)
        else:
            run_arm_c_corrected(writer, device, manifest_id=args.manifest_id, training_seed=args.training_seed, max_aux_epochs=args.max_aux_epochs)
        writer.write_json("completion.json", {"status": "completed", "arm": args.arm, "corrected": True, "elapsed_seconds": time.perf_counter() - start, "phase_timings_seconds": writer.timings, "memory": memory_snapshot(device)})
        writer.event("completed")
    except Exception as exc:
        writer.write_json("failure.json", {"status": "failed", "arm": args.arm, "corrected": True, "error": repr(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - start, "phase_timings_seconds": writer.timings, "memory": memory_snapshot(device)})
        writer.event("failed", error=repr(exc))
        raise
    finally:
        writer.finalize_hashes()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run corrected D012 first production cell for Arm B or C.")
    parser.add_argument("--arm", choices=("B", "C"), required=True)
    parser.add_argument("--manifest-id", default="rep_01_fold_01")
    parser.add_argument("--training-seed", default=20261101, type=int)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-aux-epochs", default=50, type=int)
    parser.add_argument("--max-target-epochs", default=50, type=int)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.output_dir is None:
        parsed.output_dir = CORRECTED_FIRST_CELL_ROOT / f"arm_{parsed.arm.lower()}" / f"{parsed.manifest_id}_seed{parsed.training_seed}"
    run(parsed)
