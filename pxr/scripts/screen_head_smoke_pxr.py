"""PXR-004 stage-1 screen-head loading smoke (Tripwire-CYP-010). Authorizes no run by itself.

Verifies, on CPU and with no training, that one PXR-004 stage-1 checkpoint (the sequential
arm's `stage1/best.ckpt`, three log2FC heads) loads strictly into the model the trainer
built, that head 0 is the 8.25 uM head, and that eval-mode outputs are native log2fc. It runs
one eval-mode forward pass on a small deterministic set of stage-1 monitor molecules. It
scores nothing. The stage-2 potency checkpoint is never opened; only `stage2/manifest.json`
is read, to confirm which stage-1 checkpoint fed stage 2.

Scaling and eval mode (resolved from Chemprop 2.3.1 source, and different from the CYP
screening model):
  * the stage-1 model carries UnscaleTransform.from_standard_scaler(y_scaler); its mean and
    scale are registered buffers, so they are stored in the checkpoint's state_dict;
  * UnscaleTransform returns its input unchanged while `training` is true, else
    X * scale + mean. MPNN.on_validation_model_eval sets the model to eval but forces
    output_transform.train(), so the stage-1 monitor metrics were computed in STANDARDIZED
    units. A plain model.eval() forward returns NATIVE log2fc. Inference must therefore use
    model.eval() and must not use the validation hook;
  * the script checks the identity native == standardized * scale + mean on one batch, and
    that the scaler agrees three ways: recomputed through the existing loaders, the stage-1
    manifest, and the checkpoint buffers.

Source of truth for paths and hashes is the frozen pins file
(pxr/artifacts/screen_heads_smoke/pxr_stage1_pins.json), whose own SHA-256 must be passed on
the command line. Code is imported only from the staged PXR-004 package, never from this
checkout.

Usage (CoreHPC, run by the project owner):
    /scratch/user/mhuang8/envs/tripwire_cyp002/bin/python -I screen_head_smoke_pxr.py \
        --pins PINS.json --pins-sha256 HEX --seed 20261125 --out-dir NEW_DIR --threads 2
The out-dir must not exist. The script never writes anywhere else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path


# ----------------------------------------------------------------------- pure helpers
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def new_output_dir(path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {path}")
    path.mkdir(parents=True)
    return path


def verify_stage_files(stage_root, files: dict) -> int:
    """Hash every staged file named in the pins (relative path -> sha256)."""
    stage_root = Path(stage_root)
    for relative, expected in sorted(files.items()):
        path = stage_root / relative
        if not path.is_file():
            raise ValueError(f"Staged file missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Hash mismatch: {path} (actual {actual}, expected {expected})")
    return len(files)


def stage1_paths(run_dir) -> dict:
    """The only run files this smoke touches. Stage-2 contributes its manifest, never a checkpoint."""
    run_dir = Path(run_dir)
    return {
        "checkpoint": run_dir / "stage1" / "best.ckpt",
        "metrics": run_dir / "stage1" / "metrics.json",
        "manifest": run_dir / "stage1" / "manifest.json",
        "stage2_manifest": run_dir / "stage2" / "manifest.json",
    }


def expected_encoder_source(checkpoint_sha256: str) -> str:
    return f"stage1_best_ckpt_sha256={checkpoint_sha256}"


def verify_seed_records(paths: dict, pin: dict, expected: dict, weights_sha256: str) -> dict:
    """Cross-check the checkpoint hash, stage-1 metrics, stage-1 manifest and stage-2 manifest."""
    actual = sha256_file(paths["checkpoint"])
    if actual != pin["best_checkpoint_sha256"]:
        raise ValueError(f"Stage-1 checkpoint hash {actual} differs from the pinned hash.")
    metrics = read_json(paths["metrics"])
    manifest = read_json(paths["manifest"])
    stage2 = read_json(paths["stage2_manifest"])
    if metrics["best_checkpoint_sha256"] != actual:
        raise ValueError("metrics.json best_checkpoint_sha256 differs from the checkpoint file.")
    if stage2["encoder_source"] != expected_encoder_source(actual):
        raise ValueError("Stage-2 encoder_source does not name this stage-1 checkpoint.")
    if metrics["best_epoch"] != pin["best_epoch"] or metrics["epochs_run"] != pin["epochs_run"]:
        raise ValueError("Stage-1 best_epoch/epochs_run differ from the pinned record.")
    head = metrics["best_heads"][expected["head_order"][expected["screen_head_index"]]]
    for key in ("mae", "spearman", "n"):
        if head[key] != pin["head_8p25"][key]:
            raise ValueError(f"Stage-1 8.25 uM monitor {key} differs from the pinned record.")
    if manifest["hashes"]["weights_sha256"] != weights_sha256:
        raise ValueError("Stage-1 manifest weights hash differs from the pinned weights hash.")
    if manifest["train_label_counts"] != expected["train_label_counts"]:
        raise ValueError("Stage-1 manifest train label counts differ from the pinned counts.")
    if manifest["monitor_label_counts"] != expected["monitor_label_counts"]:
        raise ValueError("Stage-1 manifest monitor label counts differ from the pinned counts.")
    return {
        "checkpoint_sha256": actual,
        "manifest_scaler_mean": manifest["scaler_mean"],
        "manifest_scaler_scale": manifest["scaler_scale"],
        "manifest_n_train_molecules": manifest["n_train_molecules"],
        "manifest_n_monitor_molecules": manifest["n_monitor_molecules"],
    }


def scaler_from_state_dict(state_dict) -> tuple:
    """(mean, scale) vectors from the checkpoint's output-transform buffers; exactly one each."""
    import numpy as np

    means = [k for k in state_dict if k.endswith("output_transform.mean")]
    scales = [k for k in state_dict if k.endswith("output_transform.scale")]
    if len(means) != 1 or len(scales) != 1:
        raise ValueError(f"Expected one output-transform mean and scale; found {means}, {scales}.")
    mean = np.asarray(state_dict[means[0]].detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    scale = np.asarray(state_dict[scales[0]].detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    return mean, scale


def check_scaler_agreement(recomputed, manifest, buffers, tol64: float, tol32: float) -> dict:
    """Three-way agreement of (mean, scale): recomputed vs stage-1 manifest vs checkpoint buffers."""
    import numpy as np

    result = {}
    for name, index in (("mean", 0), ("scale", 1)):
        a = np.asarray(recomputed[index], dtype=np.float64)
        b = np.asarray(manifest[index], dtype=np.float64)
        c = np.asarray(buffers[index], dtype=np.float64)
        if not (a.shape == b.shape == c.shape):
            raise ValueError(f"Scaler {name} shapes differ: {a.shape}, {b.shape}, {c.shape}.")
        d_ab = float(np.max(np.abs(a - b)))
        d_ac = float(np.max(np.abs(a - c)))
        if d_ab > tol64:
            raise ValueError(f"Recomputed scaler {name} differs from the manifest by {d_ab}.")
        if d_ac > tol32:
            raise ValueError(f"Recomputed scaler {name} differs from the checkpoint buffer by {d_ac}.")
        result[name] = {"recomputed_vs_manifest": d_ab, "recomputed_vs_checkpoint_buffer": d_ac}
    return result


def unscale_consistency(native, standardized, mean, scale, tol: float) -> dict:
    """native must equal standardized * scale + mean, and must differ from standardized."""
    import numpy as np

    native = np.asarray(native, dtype=np.float64)
    standardized = np.asarray(standardized, dtype=np.float64)
    expected = standardized * np.asarray(scale, dtype=np.float64) + np.asarray(mean, dtype=np.float64)
    max_diff = float(np.max(np.abs(native - expected)))
    if max_diff > tol:
        raise ValueError(f"Eval output differs from standardized*scale+mean by {max_diff}.")
    if np.allclose(native, standardized, atol=tol):
        raise ValueError("Eval and train-mode outputs are identical: the transform is not applied.")
    return {"max_abs_difference": max_diff}


def check_head_order(head_order, expected_order, screen_head_index: int) -> str:
    if tuple(head_order) != tuple(expected_order):
        raise ValueError(f"Head order {tuple(head_order)} differs from expected {tuple(expected_order)}.")
    if expected_order[screen_head_index] != "log2fc_8p25uM":
        raise ValueError("The pinned screen head index is not the 8.25 uM head.")
    return expected_order[screen_head_index]


def check_native_outputs(values, n_rows: int, n_cols: int, abs_max: float) -> dict:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n_rows, n_cols):
        raise ValueError(f"Output shape {array.shape} differs from {(n_rows, n_cols)}.")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite model outputs.")
    if float(np.abs(array).max()) > abs_max:
        raise ValueError(f"An output exceeds the sanity bound {abs_max}.")
    return {
        "shape": list(array.shape),
        "column_min": [float(v) for v in array.min(axis=0)],
        "column_max": [float(v) for v in array.max(axis=0)],
        "column_mean": [float(v) for v in array.mean(axis=0)],
    }


class Checks:
    def __init__(self):
        self.items = []

    def record(self, name: str, ok: bool, detail=None) -> None:
        self.items.append({"check": name, "pass": bool(ok), "detail": detail})

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(item["pass"] for item in self.items)


# ------------------------------------------------------------------------- the smoke
def run(args) -> dict:
    started = time.time()
    if sha256_file(args.pins) != args.pins_sha256:
        raise ValueError("Pins file hash differs from --pins-sha256.")
    pins = read_json(args.pins)
    seed_key = str(args.seed)
    if seed_key not in pins["seeds"]:
        raise ValueError(f"Seed {args.seed} is not one of the pinned seeds {sorted(pins['seeds'])}.")
    pin = pins["seeds"][seed_key]
    expected = pins["expected"]
    smoke_cfg = pins["smoke"]
    checks = Checks()
    checks.record("pins_file_hash", True, args.pins_sha256)

    # 1. Hash the staged package, then this seed's stage-1 records, before any scientific import.
    stage_root = Path(pins["stage_root"])
    n_files = verify_stage_files(stage_root, pins["stage_files"])
    checks.record("staged_package_hash", True, n_files)
    paths = stage1_paths(pin["run_dir"])
    seed_records = verify_seed_records(paths, pin, expected, pins["weights_sha256"])
    checks.record("stage1_checkpoint_and_records", True, {
        "checkpoint_sha256": seed_records["checkpoint_sha256"], "best_epoch": pin["best_epoch"],
    })

    # 2. Import only the staged, verified package code.
    if str(stage_root) not in sys.path:
        sys.path.insert(0, str(stage_root))
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(args.threads)
    import chemprop
    from chemprop import data as chemprop_data

    from pxr.scripts import pxr004_data as d
    from pxr.scripts import pxr004_train as t

    for module in (d, t):
        if stage_root.resolve() not in Path(module.__file__).resolve().parents:
            raise ValueError(f"Imported {module.__name__} is not from the staged package.")
    checks.record("imported_modules_are_staged_files", True, [d.__file__, t.__file__])

    # 3. Frames, monitor and scaler through the existing loaders (training frames only).
    artifact_dir = stage_root / pins["artifact_dir_relative"]
    assignments_path = stage_root / pins["assignments_relative"]
    monitor_dir = stage_root / pins["monitor_dir_relative"]
    if d.sha256(assignments_path) != pins["frozen_split_sha256"]:
        raise ValueError("Split assignments do not match the frozen D014 hash.")
    head_name = check_head_order(d.AUX_TASKS, expected["head_order"], expected["screen_head_index"])
    checks.record("head_order_index0_is_8p25", True, {"head_order": list(d.AUX_TASKS), "index0": head_name})

    foundation = t.load_foundation(stage_root / pins["weights_relative_path"])
    if foundation["sha256"] != pins["weights_sha256"]:
        raise ValueError("CheMeleon weights hash differs from the pinned hash.")
    frames = d.load_frames(artifact_dir)
    frame = d.attach_groups(frames["train"], frames["eligibility"], pd.read_csv(assignments_path))
    groups = d.load_monitor(monitor_dir, frame)
    train_frame, monitor_frame = d.stage1_frames(frame, groups)
    if set(train_frame["row_id"]) & set(monitor_frame["row_id"]):
        raise ValueError("Monitor molecules overlap stage-1 training.")
    sets = d.build_stage1_datasets(train_frame, monitor_frame)
    y_scaler = sets["y_scaler"]
    counts_ok = (
        len(train_frame) == expected["n_stage1_train_molecules"]
        and len(monitor_frame) == expected["n_monitor_molecules_labelled"]
        and {k: int(train_frame[k].notna().sum()) for k in d.AUX_TASKS} == expected["train_label_counts"]
        and {k: int(monitor_frame[k].notna().sum()) for k in d.AUX_TASKS} == expected["monitor_label_counts"]
    )
    checks.record("stage1_cohort_counts", counts_ok, {
        "train_molecules": len(train_frame), "monitor_molecules": len(monitor_frame),
    })

    # 4. Build the model as the trainer did, then load the stage-1 checkpoint strictly.
    encoder = t.make_encoder(foundation["hyper_parameters"], foundation["state_dict"])
    model = t.build_model("stage1", encoder, y_scaler)
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no state_dict.")
    buffers = scaler_from_state_dict(checkpoint["state_dict"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    checks.record("strict_state_dict_load", True, {
        "keys": len(checkpoint["state_dict"]), "global_step": checkpoint.get("global_step"),
    })
    changed = any(
        not torch.equal(value, foundation["state_dict"][key])
        for key, value in model.message_passing.state_dict().items() if key in foundation["state_dict"]
    )
    checks.record("encoder_differs_from_foundation_weights", changed, None)

    # 5. Scaler three-way agreement (units).
    scaler_diffs = check_scaler_agreement(
        (np.asarray(y_scaler.mean_), np.asarray(y_scaler.scale_)),
        (np.asarray(seed_records["manifest_scaler_mean"]), np.asarray(seed_records["manifest_scaler_scale"])),
        buffers, smoke_cfg["scaler_tolerance_float64"], smoke_cfg["scaler_tolerance_float32"],
    )
    pinned_ok = (
        np.allclose(y_scaler.mean_, expected["scaler_mean"], atol=smoke_cfg["scaler_tolerance_float64"], rtol=0)
        and np.allclose(y_scaler.scale_, expected["scaler_scale"], atol=smoke_cfg["scaler_tolerance_float64"], rtol=0)
    )
    checks.record("scaler_three_way_agreement", True, scaler_diffs)
    checks.record("scaler_equals_pinned_values", pinned_ok, None)

    # 6. Eval-mode forward: native units, and the standardized/native identity.
    model.eval()
    transform = model.predictor.output_transform
    if transform.training or any(module.training for module in model.modules()):
        raise ValueError("Model has modules in training mode after eval().")
    n = smoke_cfg["n_molecules"]
    chosen = monitor_frame.sort_values("row_id").head(n).reset_index(drop=True)
    if len(chosen) != n:
        raise ValueError("Fewer monitor molecules than the smoke size.")
    loader = chemprop_data.build_dataloader(
        d.frame_to_dataset(chosen, d.AUX_TASKS), batch_size=n, num_workers=0, shuffle=False
    )
    batch = next(iter(loader))
    bmg, V_d, X_d = batch[0], batch[1], batch[2]
    t1 = time.time()
    with torch.inference_mode():
        native = model(bmg, V_d, X_d).detach().cpu().numpy()
        native_again = model(bmg, V_d, X_d).detach().cpu().numpy()
        transform.train()  # identity in train mode: the standardized head output
        standardized = model(bmg, V_d, X_d).detach().cpu().numpy()
        transform.eval()
    forward_seconds = time.time() - t1
    if transform.training:
        raise ValueError("Output transform was not restored to eval mode.")
    unscale = unscale_consistency(native, standardized, buffers[0], buffers[1], smoke_cfg["unscale_tolerance"])
    checks.record("eval_native_equals_standardized_times_scale_plus_mean", True, unscale)
    stats = check_native_outputs(native, n, expected["n_outputs"], smoke_cfg["plausible_abs_max"])
    checks.record("forward_shape_finite_sane", True, stats)
    checks.record("forward_repeatable_in_eval_mode", bool(np.array_equal(native, native_again)), None)

    return {
        "status": "PASS" if checks.passed else "FAIL",
        "script": "pxr/scripts/screen_head_smoke_pxr.py",
        "script_sha256": sha256_file(__file__),
        "pins_sha256": args.pins_sha256,
        "seed": args.seed,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": seed_records["checkpoint_sha256"],
        "stage": "stage-1 (three log2FC heads); stage-2 potency checkpoint not opened",
        "output_columns": list(d.AUX_TASKS),
        "screen_head": {"name": head_name, "index": expected["screen_head_index"]},
        "units": "native log2fc (eval mode applies the stage-1 UnscaleTransform; validation-hook metrics were standardized)",
        "molecules": [str(v) for v in chosen["row_id"]],
        "predictions": [[float(v) for v in row] for row in native],
        "checks": checks.items,
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__, "chemprop": chemprop.__version__,
        },
        "host": platform.node(),
        "device": "cpu",
        "threads": args.threads,
        "timing_seconds": {
            "three_forward_passes": forward_seconds,
            "total": time.time() - started,
            "note": "single CPU smoke timing; not a benchmark and not a production estimate",
        },
        "labels_used": False,
        "scored": False,
        "stage2_checkpoint_opened": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pins", required=True, type=Path)
    parser.add_argument("--pins-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=int)
    args = parser.parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.out_dir}")
    result = run(args)
    out_dir = new_output_dir(args.out_dir)
    with open(out_dir / "result.json", "x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "result": str(out_dir / "result.json")}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
