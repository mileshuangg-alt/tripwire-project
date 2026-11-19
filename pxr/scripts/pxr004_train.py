"""Tripwire PXR-004 two-arm CheMeleon trainer (D020). Authorizes no run by itself.

Arm ``direct``     : CheMeleon encoder + fresh FFN head, fine-tuned on pEC50 (stage 2 only).
Arm ``sequential`` : the same encoder, stage 1 on the three auxiliary log2FC heads (fixed
                     D014-group monitor selects the checkpoint), then stage 2 exactly as in
                     the direct arm, starting from the stage-1 encoder.

Settings and their basis (D020):
  * encoder fully fine-tuned in both stages (CheMeleon paper section 3.1); mean aggregation and
    encoder dimensions come from the checkpoint (d_h 2048, depth 6, dropout 0);
  * fresh FFN: 300 hidden units, 1 layer, dropout 0 (Chemprop CLI library defaults);
  * batch 64, Chemprop stock LR schedule 1e-4 -> 1e-3 -> 1e-4 with 2 warmup epochs (executed
    PXR-002 precedent); maximum 100 epochs, patience 20, min_delta 0 for both stages (stage 2
    follows PXR-002; stage 1 is a pilot convention). Maximum epochs also sets the LR decay;
  * stage 1 selects on pooled masked MSE in standardized units on the monitor; stage 2 selects
    on primary-head validation MAE in original units;
  * stage 2 never inherits stage-1 optimizer or scheduler state: only the encoder weights are
    carried over, and the optimizer and scheduler are rebuilt from scratch;
  * both arms seed stage 2 identically, so head initialization, shuffling and dropout streams
    are paired per seed.
The holdout and official-test labels are never loaded.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr004_data as d  # noqa: E402

BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 20
MIN_DELTA = 0.0
INITIAL_LR, MAX_LR, FINAL_LR, WARMUP_EPOCHS = 1e-4, 1e-3, 1e-4, 2
FFN_HIDDEN, FFN_LAYERS, FFN_DROPOUT = 300, 1, 0.0
STAGE1_MONITOR = "val_monitor_mse"
STAGE2_MONITOR = "val_primary_mae"
WEIGHTS_NAME = "chemeleon_mp.pt"
EXPECTED_ENCODER = {"d_v": 72, "d_e": 14, "d_h": 2048, "depth": 6}


# ------------------------------------------------------------------------------ metrics
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def stage1_metrics(preds: np.ndarray, targets: np.ndarray, scale: np.ndarray) -> dict:
    """Monitor metrics. ``preds`` and ``targets`` are in standardized units (NaN = missing).

    Selection signal: pooled masked MSE over every finite monitor label, the same reduction
    Chemprop's loss uses. Per-head MAE (original units) and Spearman are reported only.
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    mask = np.isfinite(targets)
    if not mask.any():
        raise ValueError("monitor has no finite labels")
    pooled = float(np.mean((preds[mask] - targets[mask]) ** 2))
    heads = {}
    for j, name in enumerate(d.AUX_TASKS):
        m = mask[:, j]
        err = np.abs(preds[m, j] - targets[m, j]) * float(scale[j])
        heads[name] = {
            "n": int(m.sum()),
            "mae": float(err.mean()) if m.any() else float("nan"),
            "spearman": _spearman(preds[m, j], targets[m, j]),
        }
    return {"pooled_mse": pooled, "heads": heads}


def stage2_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Primary-head MAE, RMSE and Spearman in original units."""
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if preds.shape != targets.shape or not np.isfinite(targets).all():
        raise ValueError("primary validation needs aligned, finite targets")
    return {
        "mae": float(np.mean(np.abs(preds - targets))),
        "rmse": float(np.sqrt(np.mean((preds - targets) ** 2))),
        "spearman": _spearman(preds, targets),
    }


# ------------------------------------------------------------------------------- weights
def load_foundation(weights_path: Path) -> dict:
    """Hash-verified CheMeleon checkpoint, loaded with weights_only=True."""
    import torch

    weights_path = Path(weights_path)
    hashes = d.verify_sums(weights_path.parent, required=(weights_path.name,))
    ck = torch.load(weights_path, weights_only=True, map_location="cpu")
    if set(ck) != {"hyper_parameters", "state_dict"}:
        raise ValueError(f"unexpected checkpoint keys {sorted(ck)}")
    hp = ck["hyper_parameters"]
    for key, value in EXPECTED_ENCODER.items():
        if hp.get(key) != value:
            raise ValueError(f"checkpoint {key}={hp.get(key)!r}; expected {value!r}")
    if d.make_featurizer().shape != (hp["d_v"], hp["d_e"]):
        raise ValueError("V2 featurizer dimensions do not match the checkpoint")
    return {"hyper_parameters": hp, "state_dict": ck["state_dict"], "sha256": hashes[weights_path.name]}


def make_encoder(hyper_parameters: dict, state_dict: dict):
    from chemprop import nn

    encoder = nn.BondMessagePassing(**hyper_parameters)
    encoder.load_state_dict(state_dict)
    return encoder


def encoder_state_from_checkpoint(path: Path) -> dict:
    """Message-passing weights only (never optimizer or scheduler state) from a Lightning ckpt."""
    import torch

    ck = torch.load(path, weights_only=False, map_location="cpu")
    prefix = "message_passing."
    state = {k[len(prefix):]: v for k, v in ck["state_dict"].items() if k.startswith(prefix)}
    if not state:
        raise ValueError(f"{path} has no message_passing weights")
    return state


# --------------------------------------------------------------------------------- model
def _base_model_class(kind: str):
    import torch
    from chemprop import models

    class _Stage1(models.MPNN):
        def __init__(self, *args, scale, **kwargs):
            super().__init__(*args, **kwargs)
            self.monitor_scale = np.asarray(scale, dtype=np.float64)
            self.validation_history: list[dict] = []
            self._preds: list = []
            self._targets: list = []

        def on_validation_epoch_start(self):
            self._preds, self._targets = [], []

        def validation_step(self, batch, batch_idx: int = 0):
            # Chemprop evaluates validation with the output transform in train mode (identity),
            # so predictions are in standardized units, matching the standardized targets.
            bmg, V_d, X_d, targets, *_ = batch
            preds = self(bmg, V_d, X_d)
            if self.predictor.n_targets > 1:
                preds = preds[..., 0]
            self._preds.append(preds.detach().cpu())
            self._targets.append(targets.detach().cpu())

        def on_validation_epoch_end(self):
            if getattr(self.trainer, "sanity_checking", False):
                return
            m = stage1_metrics(
                torch.cat(self._preds).numpy(), torch.cat(self._targets).numpy(), self.monitor_scale
            )
            m["epoch"] = int(self.current_epoch)
            self.validation_history.append(m)
            self.log(STAGE1_MONITOR, torch.tensor(m["pooled_mse"], device=self.device), prog_bar=True)

    class _Stage2(models.MPNN):
        def __init__(self, *args, primary_mean: float, primary_scale: float, **kwargs):
            super().__init__(*args, **kwargs)
            self.primary_mean = float(primary_mean)
            self.primary_scale = float(primary_scale)
            self.validation_history: list[dict] = []
            self.epoch_predictions: list[np.ndarray] = []
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
            preds = torch.cat(self._preds).numpy()
            m = stage2_metrics(preds, torch.cat(self._targets).numpy())
            m["epoch"] = int(self.current_epoch)
            self.validation_history.append(m)
            self.epoch_predictions.append(preds.astype(np.float64))
            self.log(STAGE2_MONITOR, torch.tensor(m["mae"], device=self.device), prog_bar=True)

    return {"stage1": _Stage1, "stage2": _Stage2}[kind]


def build_model(kind: str, encoder, y_scaler):
    """Encoder + fresh FFN head + mean aggregation; a new model always gets a new optimizer."""
    from chemprop import nn

    n_tasks = len(y_scaler.mean_)
    predictor = nn.RegressionFFN(
        n_tasks=n_tasks,
        input_dim=encoder.output_dim,
        hidden_dim=FFN_HIDDEN,
        n_layers=FFN_LAYERS,
        dropout=FFN_DROPOUT,
        activation="relu",
        criterion=nn.MSE(),
        output_transform=nn.transforms.UnscaleTransform.from_standard_scaler(y_scaler),
    )
    common = dict(
        message_passing=encoder,
        agg=nn.MeanAggregation(),
        predictor=predictor,
        batch_norm=False,
        warmup_epochs=WARMUP_EPOCHS,
        init_lr=INITIAL_LR,
        max_lr=MAX_LR,
        final_lr=FINAL_LR,
    )
    cls = _base_model_class(kind)
    if kind == "stage1":
        return cls(**common, scale=y_scaler.scale_)
    return cls(**common, primary_mean=float(y_scaler.mean_[0]), primary_scale=float(y_scaler.scale_[0]))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fit(model, train_loader, val_loader, monitor: str, out_dir: Path, accelerator: str):
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    checkpoint = ModelCheckpoint(
        dirpath=out_dir, filename="best", monitor=monitor, mode="min", save_top_k=1, auto_insert_metric_name=False
    )
    stopping = EarlyStopping(monitor=monitor, mode="min", patience=PATIENCE, min_delta=MIN_DELTA)
    trainer = pl.Trainer(
        accelerator=accelerator, devices=1, deterministic=True, max_epochs=MAX_EPOCHS,
        logger=False, enable_progress_bar=True, callbacks=[checkpoint, stopping],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return checkpoint


def _prepare(out_dir: Path, name: str) -> Path:
    out_dir = Path(out_dir)
    if (out_dir / "metrics.json").exists():
        raise RuntimeError(f"{out_dir}/metrics.json exists; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _versions() -> dict:
    import chemprop
    import torch

    return {"chemprop": chemprop.__version__, "torch": torch.__version__, "python": platform.python_version()}


def run_stage1(foundation: dict, frame: pd.DataFrame, monitor_groups: set[int], seed: int, out_dir: Path, accelerator: str, hashes: dict) -> dict:
    import torch
    from chemprop import data
    from lightning import pytorch as pl

    out_dir = _prepare(out_dir, "stage1")
    train_frame, monitor_frame = d.stage1_frames(frame, monitor_groups)
    if set(train_frame["row_id"]) & set(monitor_frame["row_id"]):
        raise RuntimeError("monitor molecules overlap stage-1 training")
    sets = d.build_stage1_datasets(train_frame, monitor_frame)

    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True)
    encoder = make_encoder(foundation["hyper_parameters"], foundation["state_dict"])
    model = build_model("stage1", encoder, sets["y_scaler"])
    train_loader = data.build_dataloader(sets["train"], batch_size=BATCH_SIZE, num_workers=0, seed=seed, shuffle=True)
    monitor_loader = data.build_dataloader(sets["monitor"], batch_size=BATCH_SIZE, num_workers=0, seed=seed, shuffle=False)

    manifest = {
        "session": "Tripwire-PXR-004", "decision": "D020", "stage": 1, "seed": seed,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
        "monitor": STAGE1_MONITOR, "ffn": [FFN_HIDDEN, FFN_LAYERS, FFN_DROPOUT],
        "lr": [INITIAL_LR, MAX_LR, FINAL_LR, WARMUP_EPOCHS],
        "n_train_molecules": len(train_frame), "n_monitor_molecules": len(monitor_frame),
        "train_label_counts": {t: int(train_frame[t].notna().sum()) for t in d.AUX_TASKS},
        "monitor_label_counts": {t: int(monitor_frame[t].notna().sum()) for t in d.AUX_TASKS},
        "scaler_mean": sets["y_scaler"].mean_.tolist(), "scaler_scale": sets["y_scaler"].scale_.tolist(),
        "hashes": hashes, "versions": _versions(), "host": platform.node(), "accelerator": accelerator,
        "start_utc": utc_now(), "holdout_or_test_labels_used": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checkpoint = _fit(model, train_loader, monitor_loader, STAGE1_MONITOR, out_dir, accelerator)

    best = min(model.validation_history, key=lambda h: h["pooled_mse"])
    if abs(float(checkpoint.best_model_score) - best["pooled_mse"]) > 1e-5:
        raise RuntimeError("checkpoint best score disagrees with the recorded monitor history")
    metrics = {
        "stage": 1, "seed": seed, "best_monitor_pooled_mse": best["pooled_mse"], "best_epoch": best["epoch"],
        "best_heads": best["heads"], "epochs_run": len(model.validation_history),
        "validation_history": model.validation_history,
        "best_checkpoint_sha256": d.sha256(out_dir / "best.ckpt"), "end_utc": utc_now(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def run_stage2(encoder_hp: dict, encoder_state: dict, encoder_source: str, arm: str, frame: pd.DataFrame,
               validation: pd.DataFrame, seed: int, out_dir: Path, accelerator: str, hashes: dict) -> dict:
    import torch
    from chemprop import data
    from lightning import pytorch as pl

    out_dir = _prepare(out_dir, "stage2")
    sets = d.build_stage2_datasets(frame, validation)
    if sets["n_train"] != int((frame["role"] == "primary_train").sum()):
        raise RuntimeError("stage 2 must train on every primary-train molecule")

    # Identical in both arms: same seed, same construction order, so the fresh head, shuffling
    # and dropout streams are paired. A new model also means a new optimizer and scheduler.
    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True)
    encoder = make_encoder(encoder_hp, encoder_state)
    model = build_model("stage2", encoder, sets["y_scaler"])
    train_loader = data.build_dataloader(sets["train"], batch_size=BATCH_SIZE, num_workers=0, seed=seed, shuffle=True)
    val_loader = data.build_dataloader(sets["validation"], batch_size=BATCH_SIZE, num_workers=0, seed=seed, shuffle=False)

    manifest = {
        "session": "Tripwire-PXR-004", "decision": "D020", "stage": 2, "arm": arm, "seed": seed,
        "encoder_source": encoder_source, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "min_delta": MIN_DELTA, "monitor": STAGE2_MONITOR,
        "ffn": [FFN_HIDDEN, FFN_LAYERS, FFN_DROPOUT], "lr": [INITIAL_LR, MAX_LR, FINAL_LR, WARMUP_EPOCHS],
        "n_train_molecules": sets["n_train"], "n_validation_molecules": len(validation),
        "scaler_mean": sets["y_scaler"].mean_.tolist(), "scaler_scale": sets["y_scaler"].scale_.tolist(),
        "hashes": hashes, "versions": _versions(), "host": platform.node(), "accelerator": accelerator,
        "start_utc": utc_now(), "holdout_or_test_labels_used": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checkpoint = _fit(model, train_loader, val_loader, STAGE2_MONITOR, out_dir, accelerator)

    history = model.validation_history
    best_i = min(range(len(history)), key=lambda i: history[i]["mae"])
    best = history[best_i]
    if abs(float(checkpoint.best_model_score) - best["mae"]) > 1e-5:
        raise RuntimeError("checkpoint best score disagrees with the recorded validation history")
    pd.DataFrame({
        "row_id": validation["row_id"].to_numpy(),
        "pEC50_true": validation[d.PRIMARY_TASK].to_numpy(dtype=np.float64),
        "pEC50_pred": model.epoch_predictions[best_i],
    }).to_csv(out_dir / "validation_predictions_best.csv", index=False)
    metrics = {
        "stage": 2, "arm": arm, "seed": seed, "best_primary_validation_mae": best["mae"],
        "best_primary_validation_rmse": best["rmse"], "best_primary_validation_spearman": best["spearman"],
        "best_epoch": best["epoch"], "epochs_run": len(history), "validation_history": history,
        "best_checkpoint_sha256": d.sha256(out_dir / "best.ckpt"),
        "predictions_sha256": d.sha256(out_dir / "validation_predictions_best.csv"), "end_utc": utc_now(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=["direct", "sequential"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--assignments", required=True, type=Path)
    p.add_argument("--monitor-dir", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu", "mps"])
    a = p.parse_args()

    if d.sha256(a.assignments) != d.FROZEN_SPLIT_SHA256:
        raise RuntimeError("split assignments do not match the frozen D014 hash")
    foundation = load_foundation(a.weights)
    frames = d.load_frames(a.artifact_dir)
    frame = d.attach_groups(frames["train"], frames["eligibility"], pd.read_csv(a.assignments))
    monitor_groups = d.load_monitor(a.monitor_dir, frame)
    hashes = {
        "weights_sha256": foundation["sha256"], "frames": frames["hashes"],
        "monitor": d.verify_sums(a.monitor_dir), "split_assignments_sha256": d.FROZEN_SPLIT_SHA256,
        "scripts": {n: d.sha256(Path(__file__).with_name(n)) for n in ("pxr004_data.py", "pxr004_train.py")},
    }
    run_dir = a.out_dir / f"{a.arm}_seed{a.seed}"
    hp, state, source = foundation["hyper_parameters"], foundation["state_dict"], "chemeleon_foundation"
    if a.arm == "sequential":
        s1 = run_stage1(foundation, frame, monitor_groups, a.seed, run_dir / "stage1", a.accelerator, hashes)
        state = encoder_state_from_checkpoint(run_dir / "stage1" / "best.ckpt")
        source = f"stage1_best_ckpt_sha256={s1['best_checkpoint_sha256']}"
    run_stage2(hp, state, source, a.arm, frame, frames["validation"], a.seed, run_dir / "stage2", a.accelerator, hashes)


if __name__ == "__main__":
    main()
