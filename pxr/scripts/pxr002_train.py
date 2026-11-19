"""Tripwire PXR-002 single-configuration trainer (D018). Authorizes no run by itself.

One primary pEC50 head + three auxiliary log2FC heads, stock Chemprop MSE with an
explicit task-weight vector calibrated to the nominal 65:35 allocation
(pxr002_loss_weights). Checkpointing and early stopping monitor primary-head
validation MAE in original units; the pooled multitask validation loss is never
logged or used. Holdout and official-test labels are not loaded.

Chemprop-default optimizer/architecture settings that D018 does not list (batch 64,
LR 1e-4 -> 1e-3 -> 1e-4 with 2 warmup epochs, ReLU, norm aggregation 100, no batch
norm) are stock defaults; "hidden size" sets both the message-passing and FFN width.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts.pxr002_loss_weights import (  # noqa: E402
    aux_off_weight_vector, nominal_allocation, task_weight_vector,
)

BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 20
MIN_DELTA = 0.0
INITIAL_LR, MAX_LR, FINAL_LR, WARMUP_EPOCHS = 1e-4, 1e-3, 1e-4, 2
MONITOR = "val_primary_mae"


@dataclass(frozen=True)
class PXR002Config:
    hidden_size: int
    depth: int
    ffn_layers: int
    dropout: float

    @property
    def name(self) -> str:
        return f"h{self.hidden_size}_d{self.depth}_f{self.ffn_layers}_p{self.dropout:g}"


def primary_validation_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Primary-head MAE and Spearman in original units (column 0 only)."""
    import pandas as pd

    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if preds.shape != targets.shape or not np.isfinite(targets).all():
        raise ValueError("primary validation needs aligned, finite targets")
    return {
        "mae": float(np.mean(np.abs(preds - targets))),
        "spearman": float(pd.Series(preds).corr(pd.Series(targets), method="spearman")),
    }


def _validation_model_class():
    import torch
    from chemprop import models

    class PXR002ValidationMPNN(models.MPNN):
        """Collects primary-head validation predictions; never uses pooled loss."""

        def __init__(self, *args, primary_mean: float, primary_scale: float, **kwargs):
            super().__init__(*args, **kwargs)
            # Chemprop's on_validation_model_eval puts the output transform in train mode
            # (identity), so validation predictions come back in scaled units; un-scale the
            # primary head here, because validation targets are kept in original units.
            self.primary_mean = float(primary_mean)
            self.primary_scale = float(primary_scale)
            self.validation_history: list[dict] = []
            self._preds: list = []
            self._targets: list = []

        def on_validation_epoch_start(self):
            self._preds, self._targets = [], []

        def validation_step(self, batch, batch_idx: int = 0):
            bmg, V_d, X_d, targets, *_ = batch
            preds = self(bmg, V_d, X_d)
            if self.predictor.n_targets > 1:
                preds = preds[..., 0]
            primary = preds[:, 0] * self.primary_scale + self.primary_mean
            self._preds.append(primary.detach().cpu())
            self._targets.append(targets[:, 0].detach().cpu())

        def on_validation_epoch_end(self):
            if getattr(self.trainer, "sanity_checking", False):
                return
            metrics = primary_validation_metrics(
                torch.cat(self._preds).numpy(), torch.cat(self._targets).numpy()
            )
            metrics["epoch"] = int(self.current_epoch)
            self.validation_history.append(metrics)
            self.log(MONITOR, torch.tensor(metrics["mae"], device=self.device), prog_bar=True)

    return PXR002ValidationMPNN


def build_model(config: PXR002Config, task_weights, y_scaler, xd_scaler, xd_dim: int):
    from chemprop import nn

    transform = nn.transforms.UnscaleTransform.from_standard_scaler(y_scaler)
    predictor = nn.RegressionFFN(
        n_tasks=len(task_weights),
        input_dim=config.hidden_size + xd_dim,
        hidden_dim=config.hidden_size,
        n_layers=config.ffn_layers,
        dropout=config.dropout,
        activation="relu",
        criterion=nn.MSE(task_weights=task_weights),
        output_transform=transform,
    )
    return _validation_model_class()(
        message_passing=nn.BondMessagePassing(
            d_h=config.hidden_size, depth=config.depth, dropout=config.dropout, activation="relu"
        ),
        agg=nn.NormAggregation(norm=100),
        predictor=predictor,
        batch_norm=False,
        warmup_epochs=WARMUP_EPOCHS,
        init_lr=INITIAL_LR,
        max_lr=MAX_LR,
        final_lr=FINAL_LR,
        X_d_transform=nn.ScaleTransform.from_standard_scaler(xd_scaler),
        primary_mean=float(y_scaler.mean_[0]),
        primary_scale=float(y_scaler.scale_[0]),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_training(
    config: PXR002Config, seed: int, artifact_dir: Path, out_dir: Path, accelerator: str, aux_off: bool = False
) -> dict:
    import chemprop
    import torch
    from chemprop import data
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from pxr.scripts.pxr002_aux_data import AUX_TASKS, PRIMARY_TASK, label_counts
    from pxr.scripts.pxr002_dataset import build_datasets, load_artifact, sha256

    out_dir = Path(out_dir)
    if (out_dir / "metrics.json").exists():
        raise RuntimeError(f"{out_dir}/metrics.json exists; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True)

    artifact = load_artifact(artifact_dir)
    train, validation, hashes = artifact["train"], artifact["validation"], artifact["hashes"]
    counts = label_counts(train)
    if aux_off:  # auxiliary-off ablation: same cohort and batches, auxiliary loss coefficients 0
        weights = aux_off_weight_vector(len(AUX_TASKS))
    else:
        weights = task_weight_vector(counts[PRIMARY_TASK], {t: counts[t] for t in AUX_TASKS})
    allocation = nominal_allocation(weights, [counts[PRIMARY_TASK], *[counts[t] for t in AUX_TASKS]])

    sets = build_datasets(train, validation, artifact["train_x"], artifact["val_x"])
    train_loader = data.build_dataloader(sets["train"], batch_size=BATCH_SIZE, num_workers=0, seed=seed, shuffle=True)
    val_loader = data.build_dataloader(sets["validation"], batch_size=BATCH_SIZE, num_workers=0, seed=seed, shuffle=False)
    model = build_model(config, weights, sets["y_scaler"], sets["xd_scaler"], sets["xd_dim"])

    manifest = {
        "session": "Tripwire-PXR-002",
        "decision": "D018",
        "config": {**asdict(config), "name": config.name},
        "seed": seed,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "monitor": MONITOR,
        "train_label_counts": counts,
        "task_weights": weights,
        "aux_off": aux_off,
        "nominal_allocation": allocation,
        "artifact_hashes": hashes,
        "excluded_rows": artifact["amendment"]["row_id"].tolist(),
        "descriptors_cached_before_scaling": True,
        "versions": {"chemprop": chemprop.__version__, "torch": torch.__version__, "python": platform.python_version()},
        "host": platform.node(),
        "accelerator": accelerator,
        "start_utc": utc_now(),
        "holdout_or_test_labels_used": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    checkpoint = ModelCheckpoint(dirpath=out_dir, filename="best", monitor=MONITOR, mode="min", save_top_k=1, auto_insert_metric_name=False)
    stopping = EarlyStopping(monitor=MONITOR, mode="min", patience=PATIENCE, min_delta=MIN_DELTA)
    trainer = pl.Trainer(
        accelerator=accelerator, devices=1, deterministic=True, max_epochs=MAX_EPOCHS,
        logger=False, enable_progress_bar=True, callbacks=[checkpoint, stopping],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best = min(model.validation_history, key=lambda h: h["mae"])
    metrics = {
        "configuration": config.name,
        "seed": seed,
        "best_primary_validation_mae": best["mae"],
        "best_primary_validation_spearman": best["spearman"],
        "best_epoch": best["epoch"],
        "epochs_run": len(model.validation_history),
        "validation_history": model.validation_history,
        "best_checkpoint_sha256": sha256(out_dir / "best.ckpt"),
        "end_utc": utc_now(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu", "mps"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--hidden-size", required=True, type=int)
    p.add_argument("--depth", required=True, type=int)
    p.add_argument("--ffn-layers", required=True, type=int)
    p.add_argument("--dropout", required=True, type=float)
    p.add_argument("--aux-off", action="store_true", help="auxiliary-off ablation: auxiliary loss coefficients 0")
    a = p.parse_args()
    run_training(
        PXR002Config(a.hidden_size, a.depth, a.ffn_layers, a.dropout),
        a.seed, a.artifact_dir, a.out_dir, a.accelerator, aux_off=a.aux_off,
    )


if __name__ == "__main__":
    main()
