"""Tripwire PXR-002 interactive runtime estimate (D018). Run on the intended host.

Measures, on the real training frame, descriptor build time and the time per training
batch for the two grid corners, then extrapolates the full grid. It never trains to
completion and must not be run through Slurm. Worst case assumes every run uses all
100 epochs (no early stopping); the expected case uses --assumed-epochs.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu", "mps"])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--assumed-epochs", type=int, default=60)
    p.add_argument("--requested-wall-hours", type=float, default=None)
    a = p.parse_args()

    import torch
    from chemprop import data
    from lightning import pytorch as pl

    from pxr.scripts.pxr002_aux_data import AUX_TASKS, PRIMARY_TASK, label_counts
    from pxr.scripts.pxr002_dataset import build_datasets, load_artifact
    from pxr.scripts.pxr002_loss_weights import task_weight_vector
    from pxr.scripts.pxr002_train import BATCH_SIZE, MAX_EPOCHS, PXR002Config, build_model

    torch.use_deterministic_algorithms(True)  # match pxr002_train.py
    artifact = load_artifact(a.artifact_dir)
    train, validation = artifact["train"], artifact["validation"]
    t0 = time.perf_counter()
    sets = build_datasets(train, validation, artifact["train_x"], artifact["val_x"])
    build_s = time.perf_counter() - t0
    counts = label_counts(train)
    weights = task_weight_vector(counts[PRIMARY_TASK], {t: counts[t] for t in AUX_TASKS})
    batches = math.ceil(len(train) / BATCH_SIZE)
    loader = data.build_dataloader(sets["train"], batch_size=BATCH_SIZE, num_workers=0, seed=0, shuffle=True)
    val_loader = data.build_dataloader(sets["validation"], batch_size=BATCH_SIZE, num_workers=0, seed=0, shuffle=False)

    print(f"training molecules {len(train)}, batches/epoch {batches}, dataset build from cached descriptors {build_s:.1f}s (one-time)")
    corners = {"smallest": PXR002Config(300, 3, 1, 0.0), "largest": PXR002Config(600, 5, 2, 0.2)}
    per_epoch = {}
    for label, cfg in corners.items():
        model = build_model(cfg, weights, sets["y_scaler"], sets["xd_scaler"], sets["xd_dim"])
        warm = pl.Trainer(accelerator=a.accelerator, devices=1, max_steps=5, logger=False, deterministic=True,
                          enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0)
        warm.fit(model, train_dataloaders=loader)
        trainer = pl.Trainer(accelerator=a.accelerator, devices=1, max_steps=a.steps, logger=False, deterministic=True,
                             enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0)
        t = time.perf_counter()
        trainer.fit(model, train_dataloaders=loader)
        step_s = (time.perf_counter() - t) / a.steps
        v = time.perf_counter()
        pl.Trainer(accelerator=a.accelerator, devices=1, logger=False, deterministic=True, enable_progress_bar=False).validate(model, val_loader)
        val_s = time.perf_counter() - v
        per_epoch[label] = step_s * batches + val_s
        print(f"{label} {cfg.name}: {step_s*1000:.1f} ms/step, validation {val_s:.1f}s, ~{per_epoch[label]:.1f}s/epoch")

    runs = 36 + 6
    mean_epoch = (per_epoch["smallest"] + per_epoch["largest"]) / 2
    worst = per_epoch["largest"] * MAX_EPOCHS * runs / 3600
    expected = mean_epoch * a.assumed_epochs * runs / 3600
    print(f"projection over {runs} runs: worst case {worst:.1f} h (100 epochs, largest-config time), "
          f"expected {expected:.1f} h ({a.assumed_epochs} epochs, mean-corner time)")
    if a.requested_wall_hours is not None:
        print(f"requested wall {a.requested_wall_hours} h: worst case {'FITS' if worst <= a.requested_wall_hours else 'EXCEEDS'}")


if __name__ == "__main__":
    main()
