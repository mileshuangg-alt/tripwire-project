"""Validation-only inference for completed updated-T121 cells (Tripwire-CYP-007).

For each requested cell this rebuilds the cell's own training/validation fold
through the existing D007 path (route_d007_fold, which carries the
protected-test guard), builds the same untrained model and loaders as the
original fit (build_updated_t121_pic50_prefit_state), loads the cell's
fit/best.ckpt strictly, and runs ONE validation pass through the model's own
validation_step / on_validation_epoch_end. The arrays the frozen scorer
consumed are captured by wrapping cyp002_validation.compute_validation_macro_st_rae,
so the reproduction check uses the original scoring path unchanged.

Writes only into a new, empty output directory. Nothing is written into cell,
fit or checkpoint directories. No training, no optimizer, no resume. The
protected internal test and the challenge set are never read: only the
molecules named in the cell's split manifest enter, and route_d007_fold
refuses any protected molecule.

Nothing here has a default path or tolerance. Inference must be authorized
explicitly. Chemprop-dependent steps import lazily so the protocol and the
metric code are testable without a GPU.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import socket
import sys
import time

import numpy as np

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
TASK_SUFFIX = "_pIC50_direct_inhibition"
TASKS = tuple(f"{name}{TASK_SUFFIX}" for name in ISOFORMS)

PREDICTION_COLUMNS = (
    "trial_index", "cell_index", "repetition", "fold", "seed", "isoform",
    "Molecule_Name", "y_true", "y_pred", "conf_low", "conf_high",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json_exclusive(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    with open(path, "x") as handle:
        handle.write(text)


def soft_error(y_true_low, y_true_high, y_pred):
    """Distance from the prediction to the nearest edge of [low, high]; zero inside."""
    return np.clip(y_pred - y_true_high, 0.0, None) + np.clip(
        y_true_low - y_pred, 0.0, None
    )


def isoform_metrics(y_true, y_pred, conf_low, conf_high):
    """Metrics for one isoform from individual rows (no averaging of predictions).

    ST-RAE follows the scorer's stated definition: summed soft error of the
    model over the summed soft error of a constant predictor at mean(y_true) of
    the same rows, both under the same tolerance band. It is cross-checked
    against the frozen scorer in the tests and in every cell's reproduction.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    low = np.asarray(conf_low, dtype=np.float64)
    high = np.asarray(conf_high, dtype=np.float64)
    soft = soft_error(low, high, y_pred)
    baseline = soft_error(low, high, np.full_like(y_true, y_true.mean()))
    return {
        "n": int(y_true.size),
        "plain_mae": float(np.mean(np.abs(y_pred - y_true))),
        "signed_bias": float(np.mean(y_pred - y_true)),
        "soft_mae": float(np.mean(soft)),
        "baseline_soft_mae": float(np.mean(baseline)),
        "st_rae": (
            float(soft.sum() / baseline.sum()) if baseline.sum() > 0 else float("nan")
        ),
    }


def metrics_from_rows(rows):
    """rows: list of dicts with isoform, y_true, y_pred, conf_low, conf_high."""
    result = {}
    for isoform in ISOFORMS:
        subset = [row for row in rows if row["isoform"] == isoform]
        result[isoform] = isoform_metrics(
            [r["y_true"] for r in subset], [r["y_pred"] for r in subset],
            [r["conf_low"] for r in subset], [r["conf_high"] for r in subset],
        )
    result["macro_st_rae"] = float(
        np.mean([result[i]["st_rae"] for i in ISOFORMS])
    )
    return result


def prediction_rows(*, identity, validation_dataframe, captured):
    """Per-compound rows. Row order is dataframe order, labeled rows only.

    Labels are taken exactly from the dataframe. The captured scorer inputs
    must agree with them (a float32 round trip makes them differ by ~1e-7);
    a larger disagreement means the captured rows are not these compounds.
    """
    rows = []
    for isoform, task in zip(ISOFORMS, TASKS):
        labels = validation_dataframe[task].to_numpy(dtype=np.float64)
        low = validation_dataframe[f"{task}_conf_low"].to_numpy(dtype=np.float64)
        high = validation_dataframe[f"{task}_conf_high"].to_numpy(dtype=np.float64)
        present = np.isfinite(labels)
        names = validation_dataframe["Molecule_Name"].to_numpy()[present]
        cap = captured[task]
        if len(cap["y_pred"]) != int(present.sum()):
            raise RuntimeError(f"{task}: captured row count differs from labeled rows.")
        if not np.allclose(cap["y_true"], labels[present], atol=1e-4, rtol=0):
            raise RuntimeError(f"{task}: captured labels do not match the fold's labels.")
        if not (
            np.array_equal(cap["conf_low"], low[present])
            and np.array_equal(cap["conf_high"], high[present])
        ):
            raise RuntimeError(f"{task}: captured CI bounds do not match the fold's.")
        for name, y, p, lo, hi in zip(
            names, labels[present], cap["y_pred"], low[present], high[present]
        ):
            rows.append({
                **identity, "isoform": isoform, "Molecule_Name": name,
                "y_true": float(y), "y_pred": float(p),
                "conf_low": float(lo), "conf_high": float(hi),
            })
    return rows


def write_predictions_csv(path, rows):
    with open(path, "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (repr(row[key]) if isinstance(row[key], float) else row[key])
                for key in PREDICTION_COLUMNS
            })


def compare_scores(original, recomputed, tolerance):
    """Per-isoform and macro absolute differences against the original scores."""
    diffs = {
        task: abs(float(recomputed["per_task"][task]) - float(original["per_task"][task]))
        for task in TASKS
    }
    macro = abs(float(recomputed["macro_st_rae"]) - float(original["macro_st_rae"]))
    return {
        "per_task_abs_diff": diffs,
        "macro_abs_diff": macro,
        "max_abs_diff": max([macro, *diffs.values()]),
        "tolerance": tolerance,
        "within_tolerance": max([macro, *diffs.values()]) <= tolerance,
    }


def load_ops():
    import pandas as pd
    import torch
    import cyp007_cell_fit as fit
    import cyp002_validation as validation
    import t121_weighted_training as weighted

    return {
        "pd": pd, "torch": torch, "fit": fit, "validation": validation,
        "weighted": weighted,
    }


def infer_cell(
    *,
    run_root,
    trial_index,
    cell_index,
    dataset_path,
    original_split_path,
    split_manifest_root,
    output_dir,
    tolerance,
    accelerator="gpu",
    authorize_inference=False,
    ops=None,
):
    """Validation-only inference for one completed cell. Returns its record."""
    if authorize_inference is not True:
        raise RuntimeError("Updated-T121 validation inference is not authorized.")
    ops = load_ops() if ops is None else ops
    pd, torch = ops["pd"], ops["torch"]
    fit, validation, weighted = ops["fit"], ops["validation"], ops["weighted"]

    total_start = time.time()
    output_dir = Path(output_dir)
    trial_dir, cell_dir = fit._cell_paths(run_root, trial_index, cell_index)
    for required in ("COMPLETED.json", "result.json", "cell_plan.json"):
        if not (cell_dir / required).is_file():
            raise RuntimeError(f"Cell is not complete: missing {required}.")
    if (cell_dir / "INCOMPLETE.json").exists():
        raise RuntimeError("Cell has INCOMPLETE.json.")

    plan = read_json(cell_dir / "cell_plan.json")
    result = read_json(cell_dir / "result.json")
    proposal = read_json(trial_dir / "proposal.json")
    if plan["trial_index"] != trial_index or plan["cell_index"] != cell_index:
        raise RuntimeError("Cell identity does not match its frozen plan.")
    if result["trial_index"] != trial_index or result["cell_index"] != cell_index:
        raise RuntimeError("Cell identity does not match its result.")

    checkpoint_path = cell_dir / "fit" / "best.ckpt"
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != result["checkpoint_sha256"]:
        raise RuntimeError("best.ckpt does not match the SHA-256 recorded in result.json.")

    manifest_path = (
        Path(split_manifest_root)
        / f"rep_{plan['repetition']:02d}_fold_{plan['fold']:02d}.json"
    )
    manifest = read_json(manifest_path)
    if sha256_file(manifest_path) != result["split_manifest_sha256"]:
        raise RuntimeError("Split manifest does not match the SHA-256 recorded in result.json.")
    if plan["seed"] not in tuple(manifest["training_seeds"]):
        raise RuntimeError("Cell seed is not in the frozen split manifest.")
    if (manifest["repetition"], manifest["fold"]) != (plan["repetition"], plan["fold"]):
        raise RuntimeError("Split manifest does not match the cell plan.")

    sampled_config = proposal["sampled_config"]
    config_name = proposal.get("config_name", f"D007-T{trial_index:03d}")

    build_start = time.time()
    routed = fit.route_d007_fold(
        dataframe=pd.read_csv(dataset_path),
        original_split_dataframe=pd.read_csv(original_split_path),
        split_manifest=manifest,
        training_seed=plan["seed"],
    )
    validation_df = routed["validation_dataframe"]
    prefit = weighted.build_updated_t121_pic50_prefit_state(
        training_dataframe=routed["training_dataframe"],
        validation_dataframe=validation_df,
        sampled_config=sampled_config,
        config_name=config_name,
        training_seed=plan["seed"],
    )
    model = prefit["model"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    build_seconds = time.time() - build_start

    # Capture exactly what the frozen scorer receives, then call it unchanged.
    captured = {}
    original_fn = validation.compute_validation_macro_st_rae

    def recording(y_true, y_pred, conf_low, conf_high, task_names):
        for task in task_names:
            captured[task] = {
                "y_true": np.array(y_true[task], dtype=np.float64),
                "y_pred": np.array(y_pred[task], dtype=np.float64),
                "conf_low": np.array(conf_low[task], dtype=np.float64),
                "conf_high": np.array(conf_high[task], dtype=np.float64),
            }
        return original_fn(y_true, y_pred, conf_low, conf_high, task_names)

    from lightning import pytorch as pl

    trainer = pl.Trainer(
        accelerator=accelerator, devices=1, deterministic=True,
        logger=False, enable_progress_bar=False, enable_checkpointing=False,
        enable_model_summary=False,
    )
    validation.compute_validation_macro_st_rae = recording
    infer_start = time.time()
    try:
        trainer.validate(model, dataloaders=prefit["validation_loader"], verbose=False)
    finally:
        validation.compute_validation_macro_st_rae = original_fn
    infer_seconds = time.time() - infer_start

    if len(model.validation_history) != 1 or set(captured) != set(TASKS):
        raise RuntimeError("Expected exactly one scored validation pass over all four isoforms.")
    rescored = model.validation_history[0]
    original_scores = result["best_validation"]
    comparison = compare_scores(original_scores, rescored, tolerance)

    identity = {
        "trial_index": trial_index, "cell_index": cell_index,
        "repetition": plan["repetition"], "fold": plan["fold"], "seed": plan["seed"],
    }
    rows = prediction_rows(
        identity=identity, validation_dataframe=validation_df, captured=captured
    )
    from_rows = metrics_from_rows(rows)
    from_rows_cmp = compare_scores(
        original_scores,
        {"per_task": {t: from_rows[i]["st_rae"] for i, t in zip(ISOFORMS, TASKS)},
         "macro_st_rae": from_rows["macro_st_rae"]},
        tolerance,
    )

    record = {
        **identity,
        "status": "reproduced" if comparison["within_tolerance"] else "MISMATCH",
        "checkpoint_sha256": checkpoint_sha,
        "split_manifest_sha256": result["split_manifest_sha256"],
        "validation_molecules": int(len(validation_df)),
        "prediction_rows": len(rows),
        "original_scores": original_scores,
        "rescored_via_frozen_scorer": {
            "per_task": rescored["per_task"], "macro_st_rae": rescored["macro_st_rae"],
        },
        "comparison_frozen_scorer_path": comparison,
        "comparison_recomputed_from_saved_rows": from_rows_cmp,
        "metrics_from_rows": from_rows,
        "accelerator": accelerator,
        "hostname": socket.gethostname(),
        "timing_seconds": {
            "build_and_load": build_seconds, "validation_pass": infer_seconds,
            "cell_total": time.time() - total_start,
        },
    }
    name = f"cell_{cell_index:03d}"
    write_predictions_csv(output_dir / f"predictions_{name}.csv", rows)
    write_json_exclusive(output_dir / f"reproduction_{name}.json", record)
    return record


def parse_cells(text):
    cells = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cells.extend(range(int(lo), int(hi) + 1))
        else:
            cells.append(int(part))
    if len(set(cells)) != len(cells):
        raise ValueError("Duplicate cell indices.")
    return cells


def run(args, infer=infer_cell):
    """Run the requested cells serially into a new directory; stop on a mismatch."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cells = parse_cells(args.cells)
    records = []
    started = time.time()
    for cell_index in cells:
        record = infer(
            run_root=args.run_root, trial_index=args.trial_index,
            cell_index=cell_index, dataset_path=args.dataset,
            original_split_path=args.original_split,
            split_manifest_root=args.split_manifest_root,
            output_dir=output_dir, tolerance=args.tolerance,
            accelerator=args.accelerator,
            authorize_inference=args.authorize_inference,
        )
        records.append(record)
        print(json.dumps({
            "cell": cell_index, "status": record["status"],
            "max_abs_diff": record["comparison_frozen_scorer_path"]["max_abs_diff"],
            "seconds": record["timing_seconds"]["cell_total"],
        }), flush=True)
        if record["status"] != "reproduced":
            write_json_exclusive(output_dir / "STOPPED_ON_MISMATCH.json", {
                "cell_index": cell_index,
                "comparison": record["comparison_frozen_scorer_path"],
                "note": "Stopped; no patching or substitution. Needs human review.",
            })
            raise SystemExit(f"Reproduction mismatch in cell {cell_index}; stopped.")
    summary = {
        "cells": cells,
        "max_abs_diff_frozen_scorer_path": max(
            r["comparison_frozen_scorer_path"]["max_abs_diff"] for r in records
        ),
        "max_abs_diff_recomputed_from_rows": max(
            r["comparison_recomputed_from_saved_rows"]["max_abs_diff"] for r in records
        ),
        "tolerance": args.tolerance,
        "mean_cell_seconds": float(np.mean(
            [r["timing_seconds"]["cell_total"] for r in records])),
        "wall_seconds_loop": time.time() - started,
        "per_cell": [
            {k: r[k] for k in ("cell_index", "repetition", "fold", "seed", "status")}
            | {"metrics": r["metrics_from_rows"]} for r in records
        ],
    }
    write_json_exclusive(output_dir / "summary.json", summary)
    write_json_exclusive(output_dir / "completion.json", {
        "status": "completed", "cells": cells,
        "summary_sha256": sha256_file(output_dir / "summary.json"),
    })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("run-root", "dataset", "original-split", "split-manifest-root",
                 "output-dir", "executor-source-dir", "executor-source-sha256",
                 "cells", "accelerator", "package-root"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--tolerance", required=True, type=float)
    parser.add_argument("--authorize-inference", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.executor_source_dir) / "cyp007_cell_fit.py"
    digest = sha256_file(source)
    if digest != args.executor_source_sha256:
        raise SystemExit(f"Executor source hash mismatch: {source} {digest}")
    package_root = Path(args.package_root)
    for directory in (
        args.executor_source_dir, str(package_root / "cyp/scripts"), str(package_root),
    ):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    summary = run(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_cell"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
