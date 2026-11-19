"""Tripwire PXR-004 feasibility timing (D020). A runtime check only; authorizes no run.

Measures how long the pilot's stages take, never whether they learn. It builds the real
datasets and the real CheMeleon-initialised model for each stage, then times a few
forward/backward/optimizer steps on a throwaway model, one full validation pass, and one
checkpoint write. Losses are never read, printed or stored, and nothing is used to choose a
cap, patience, learning rate or epoch budget (those are fixed in D020).

It does NOT start any training run, writes nothing under pxr/artifacts, and refuses to
overwrite --out-json. Run it interactively on the hardware you want a number for (never
through Slurm); the accelerator is your choice and is recorded in the output.

The extrapolation is a worst case: every run goes to the 100-epoch maximum with no early
stopping, a validation pass every epoch and a checkpoint write every epoch.

Usage (repository root, tripwire_cyp002 interpreter, run by the user):
    python -I pxr/scripts/pxr004_timing.py --accelerator <cpu|gpu|mps> --out-json <new file> \
        --artifact-dir pxr/artifacts/pxr002_aux --assignments pxr/artifacts/pxr001_split/pxr001_split_assignments.csv \
        --monitor-dir pxr/artifacts/pxr004_monitor --weights pxr/artifacts/pxr004_weights/chemeleon_mp.pt \
        [--requested-wall-hours H]
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr004_data as d  # noqa: E402
from pxr.scripts import pxr004_train as t  # noqa: E402

N_STAGE1_RUNS = 3  # one stage-1 run per seed, sequential arm
N_STAGE2_RUNS = 6  # direct and sequential arms, three seeds each


# ------------------------------------------------------------------- pure arithmetic
def summarize(values: list[float]) -> dict:
    if not values:
        raise ValueError("no timings")
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
    return {"n": len(values), "median": statistics.median(values), "p90": p90, "mean": statistics.fmean(values)}


def batches_per_epoch(n_molecules: int, batch_size: int = t.BATCH_SIZE) -> int:
    return math.ceil(n_molecules / batch_size)


def worst_case_run_seconds(
    train_batches: int, val_batches: int, train_step_s: float, train_fetch_s: float,
    val_batch_s: float, checkpoint_s: float, epochs: int = t.MAX_EPOCHS,
) -> float:
    """One run at the epoch maximum: every epoch trains, validates and writes a checkpoint."""
    per_epoch = train_batches * (train_step_s + train_fetch_s) + val_batches * val_batch_s + checkpoint_s
    return epochs * per_epoch


def pilot_totals(stage1_run_s: float, stage2_run_s: float, requested_wall_hours: float | None = None) -> dict:
    out = {
        "serial_all_runs_hours": (N_STAGE1_RUNS * stage1_run_s + N_STAGE2_RUNS * stage2_run_s) / 3600,
        "sequential_arm_one_seed_hours": (stage1_run_s + stage2_run_s) / 3600,
        "direct_arm_one_seed_hours": stage2_run_s / 3600,
    }
    if requested_wall_hours is not None:
        out["requested_wall_hours"] = requested_wall_hours
        out["fits_serial_all_runs"] = out["serial_all_runs_hours"] <= requested_wall_hours
        out["fits_sequential_arm_one_seed"] = out["sequential_arm_one_seed_hours"] <= requested_wall_hours
    return out


# --------------------------------------------------------------------------- hardware
class Device:
    def __init__(self, accelerator: str):
        import torch

        self.torch = torch
        self.name = {"cpu": "cpu", "gpu": "cuda", "mps": "mps"}[accelerator]
        if self.name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--accelerator gpu requested but CUDA is not available")
        if self.name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--accelerator mps requested but MPS is not available")
        self.device = torch.device(self.name)

    def sync(self) -> None:
        if self.name == "cuda":
            self.torch.cuda.synchronize()
        elif self.name == "mps":
            self.torch.mps.synchronize()

    def description(self) -> str:
        if self.name == "cuda":
            return self.torch.cuda.get_device_name(0)
        return platform.processor() or platform.machine()

    def peak_memory_bytes(self) -> dict:
        import resource

        out = {"process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
        if self.name == "cuda":
            out["cuda_max_allocated_bytes"] = int(self.torch.cuda.max_memory_allocated())
        elif self.name == "mps":
            out["mps_driver_allocated_bytes"] = int(self.torch.mps.driver_allocated_memory())
        return out


def _move(batch, device):
    from lightning.fabric.utilities.apply_func import move_data_to_device

    return move_data_to_device(batch, device)


# ---------------------------------------------------------------------------- stages
def time_stage(kind: str, model, train_ds, val_ds, dev: Device, warmup: int, timed: int, seed: int = 0) -> dict:
    """Time `timed` optimizer steps after `warmup`, one validation pass and a checkpoint write."""
    import torch
    from chemprop import data

    train_loader = data.build_dataloader(train_ds, batch_size=t.BATCH_SIZE, num_workers=0, seed=seed, shuffle=True)
    val_loader = data.build_dataloader(val_ds, batch_size=t.BATCH_SIZE, num_workers=0, seed=seed, shuffle=False)
    n_train_batches, n_val_batches = len(train_loader), len(val_loader)
    if warmup + timed > n_train_batches:
        raise ValueError(f"warmup+timed ({warmup + timed}) exceeds the {n_train_batches} training batches per epoch")

    model = model.to(dev.device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), t.INITIAL_LR)
    if dev.name == "cuda":
        torch.cuda.reset_peak_memory_stats()

    fetch_s, step_s = [], []
    iterator = iter(train_loader)
    for i in range(warmup + timed):
        dev.sync()
        t0 = time.perf_counter()
        batch = _move(next(iterator), dev.device)
        dev.sync()
        t1 = time.perf_counter()
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        opt.zero_grad(set_to_none=True)
        preds = model.predictor.train_step(model.fingerprint(bmg, V_d, X_d))
        loss = model.criterion(preds, targets, mask, weights, lt_mask, gt_mask)
        loss.backward()
        opt.step()
        del loss, preds  # never read: this is a timing run, not a learning run
        dev.sync()
        t2 = time.perf_counter()
        if i >= warmup:
            fetch_s.append(t1 - t0)
            step_s.append(t2 - t1)

    model.on_validation_model_eval()
    val_batch_s, collected = [], 0
    with torch.no_grad():
        for batch in val_loader:
            dev.sync()
            t0 = time.perf_counter()
            bmg, V_d, X_d, *_ = _move(batch, dev.device)
            out = model(bmg, V_d, X_d)
            out.cpu()  # includes the device-to-host copy the metrics need
            dev.sync()
            val_batch_s.append(time.perf_counter() - t0)
            collected += out.shape[0]
    if collected != len(val_ds.data):
        raise RuntimeError("validation pass did not cover every molecule")

    ckpt_s, ckpt_bytes = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(3):
            path = Path(tmp) / f"timing_{k}.ckpt"
            t0 = time.perf_counter()
            torch.save({"state_dict": model.state_dict(), "optimizer_states": [opt.state_dict()]}, path)
            ckpt_s.append(time.perf_counter() - t0)
            ckpt_bytes = path.stat().st_size

    return {
        "kind": kind, "train_molecules": len(train_ds.data), "validation_molecules": len(val_ds.data),
        "train_batches_per_epoch": n_train_batches, "validation_batches_per_pass": n_val_batches,
        "warmup_steps_excluded": warmup, "timed_steps": timed,
        "train_step_s": summarize(step_s), "train_batch_fetch_s": summarize(fetch_s),
        "validation_batch_s": summarize(val_batch_s), "checkpoint_write_s": summarize(ckpt_s),
        "checkpoint_bytes": ckpt_bytes, "memory": dev.peak_memory_bytes(),
    }


def estimate(stage: dict) -> dict:
    """Worst-case seconds for one run, from median and from p90 timings."""
    out = {}
    for label in ("median", "p90"):
        out[label] = worst_case_run_seconds(
            stage["train_batches_per_epoch"], stage["validation_batches_per_pass"],
            stage["train_step_s"][label], stage["train_batch_fetch_s"][label],
            stage["validation_batch_s"][label], stage["checkpoint_write_s"][label],
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu", "mps"])
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--assignments", required=True, type=Path)
    p.add_argument("--monitor-dir", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--timed-steps", type=int, default=20)
    p.add_argument("--requested-wall-hours", type=float, default=None)
    a = p.parse_args()

    if a.out_json.exists():
        raise RuntimeError(f"{a.out_json} exists; refusing to overwrite")
    if "pxr/artifacts" in str(a.out_json.resolve()):
        raise RuntimeError("write timing output outside pxr/artifacts")
    import chemprop
    import torch

    torch.use_deterministic_algorithms(True)  # as in the real runs
    dev = Device(a.accelerator)

    t0 = time.perf_counter()
    foundation = t.load_foundation(a.weights)
    load_weights_s = time.perf_counter() - t0
    if d.sha256(a.assignments) != d.FROZEN_SPLIT_SHA256:
        raise RuntimeError("split assignments do not match the frozen D014 hash")
    frames = d.load_frames(a.artifact_dir)
    frame = d.attach_groups(frames["train"], frames["eligibility"], pd.read_csv(a.assignments))
    monitor_groups = d.load_monitor(a.monitor_dir, frame)

    train_f, monitor_f = d.stage1_frames(frame, monitor_groups)
    s1 = d.build_stage1_datasets(train_f, monitor_f)
    s2 = d.build_stage2_datasets(frame, frames["validation"])

    stage1 = time_stage(
        "stage1", t.build_model("stage1", t.make_encoder(foundation["hyper_parameters"], foundation["state_dict"]), s1["y_scaler"]),
        s1["train"], s1["monitor"], dev, a.warmup_steps, a.timed_steps)
    stage2 = time_stage(
        "stage2", t.build_model("stage2", t.make_encoder(foundation["hyper_parameters"], foundation["state_dict"]), s2["y_scaler"]),
        s2["train"], s2["validation"], dev, a.warmup_steps, a.timed_steps)

    est1, est2 = estimate(stage1), estimate(stage2)
    totals = {label: pilot_totals(est1[label], est2[label], a.requested_wall_hours) for label in ("median", "p90")}
    result = {
        "purpose": "feasibility timing only; not evidence of learning; no loss was read",
        "decision": "D020", "accelerator": a.accelerator, "device": dev.description(), "host": platform.node(),
        "deterministic_algorithms": True,
        "versions": {"chemprop": chemprop.__version__, "torch": torch.__version__, "python": platform.python_version()},
        "hashes": {
            "weights_sha256": foundation["sha256"],
            "scripts": {n: d.sha256(Path(__file__).with_name(n)) for n in ("pxr004_data.py", "pxr004_train.py", "pxr004_timing.py")},
            "monitor": d.verify_sums(a.monitor_dir), "frames": frames["hashes"],
        },
        "weights_load_s": load_weights_s,
        "stage1": stage1, "stage2": stage2,
        "worst_case_run_seconds": {"stage1": est1, "stage2": est2},
        "worst_case_totals": totals,
        "assumptions": [
            f"every run reaches the {t.MAX_EPOCHS}-epoch maximum; early stopping would shorten real runs",
            "a validation pass and a checkpoint write every epoch",
            f"{N_STAGE1_RUNS} stage-1 runs and {N_STAGE2_RUNS} stage-2 runs",
            "timings come from a few steps after warmup on this hardware; first-epoch and long-run effects are not captured",
            "stage-2 cost is the same for the direct arm and the sequential arm",
        ],
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, st, est in (("stage 1", stage1, est1), ("stage 2", stage2, est2)):
        print(f"{name}: {st['train_batches_per_epoch']} train batches/epoch, step median {st['train_step_s']['median']:.3f}s, "
              f"fetch median {st['train_batch_fetch_s']['median']:.3f}s, validation pass {st['validation_batches_per_pass']} batches; "
              f"worst-case run {est['median'] / 3600:.2f} h (median) / {est['p90'] / 3600:.2f} h (p90)")
    for label in ("median", "p90"):
        print(f"worst case, {label}: {json.dumps(totals[label], sort_keys=True)}")
    print(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
