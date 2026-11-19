"""Tripwire PXR-002 final prediction (approved protocol, 2026-11-18). Predicts; reads no labels.

For every final model (variant x seed) this reloads the validation-selected best.ckpt,
first REPLICATES its recorded validation MAE through the inference path (a hard gate: a
mismatch means the reload or scaling path is wrong and nothing is predicted), then predicts
the structure-only holdout artifact. Output: holdout_predictions.csv (row_id, variant,
seed, prediction) plus a manifest with the gate results and hashes. The holdout labels are
not available here; pxr002_final_eval.py reads them locally at the single evaluation step.

Inference feeds the RAW cached descriptors: the model's own X_d scale transform (fit on
the training cohort and stored in the checkpoint) standardizes them in eval mode, and its
output transform un-scales predictions to original pEC50 units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts.pxr002_grid import build_grid  # noqa: E402

GATE_TOLERANCE = 1e-3  # |replicated validation MAE - recorded best MAE|


def load_final_model(ckpt_path: str | Path, config, xd_dim: int):
    """Rebuild the architecture and load the checkpoint's weights and fitted scalers."""
    import torch

    from pxr.scripts.pxr002_train import build_model

    y_scaler = SimpleNamespace(mean_=np.zeros(4), scale_=np.ones(4))   # real values come from the state dict
    xd_scaler = SimpleNamespace(mean_=np.zeros(xd_dim), scale_=np.ones(xd_dim))
    model = build_model(config, [1.0, 0.0, 0.0, 0.0], y_scaler, xd_scaler, xd_dim)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict_primary(model, smiles: list[str], descriptors: np.ndarray, accelerator: str) -> np.ndarray:
    """Primary-head predictions in original pEC50 units from RAW descriptors."""
    import torch
    from chemprop import data
    from lightning import pytorch as pl

    points = [data.MoleculeDatapoint.from_smi(s, x_d=x) for s, x in zip(smiles, descriptors)]
    loader = data.build_dataloader(data.MoleculeDataset(points), batch_size=64, num_workers=0, shuffle=False)
    trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False, enable_progress_bar=False,
                         enable_checkpointing=False, enable_model_summary=False)
    batches = trainer.predict(model, loader)
    return torch.cat(batches)[:, 0].detach().cpu().numpy().astype(np.float64)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_predict(artifact_dir: Path, holdout_dir: Path, final_root: Path, out_dir: Path, accelerator: str) -> int:
    from pxr.scripts.pxr002_dataset import load_artifact
    from pxr.scripts.pxr002_holdout_artifact import load_holdout_artifact

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)  # refuses to overwrite earlier predictions
    plan = json.loads((Path(final_root) / "final_plan.json").read_text())
    config = {c.name: c for c in build_grid()}[plan["configuration"]]
    art = load_artifact(artifact_dir)
    validation, val_x = art["validation"], art["val_x"]
    holdout, hold_x, holdout_hashes = load_holdout_artifact(holdout_dir)

    rows, gates, ckpt_hashes = [], [], {}
    for variant in plan["variants"]:
        for seed in plan["seeds"]:
            run = Path(final_root) / f"{variant}_seed{seed}"
            ckpt = run / "best.ckpt"
            recorded = json.loads((run / "metrics.json").read_text())["best_primary_validation_mae"]
            model = load_final_model(ckpt, config, art["val_x"].shape[1])
            val_pred = predict_primary(model, validation["SMILES"].tolist(), val_x, accelerator)
            replicated = float(np.mean(np.abs(val_pred - validation["pEC50"].to_numpy(dtype=float))))
            gate = {"variant": variant, "seed": seed, "recorded_val_mae": recorded,
                    "replicated_val_mae": replicated, "abs_difference": abs(replicated - recorded)}
            gates.append(gate)
            print(f"=== gate {variant} seed {seed}: recorded {recorded:.6f} replicated {replicated:.6f}", flush=True)
            if gate["abs_difference"] > GATE_TOLERANCE:
                (out_dir / "gate_failure.json").write_text(json.dumps(gates, indent=2) + "\n")
                print("=== GATE FAILED: inference path does not reproduce the recorded validation MAE; "
                      "no holdout predictions were made for this or later models", flush=True)
                return 1
            pred = predict_primary(model, holdout["SMILES"].tolist(), hold_x, accelerator)
            rows.append(pd.DataFrame({"row_id": holdout["row_id"], "variant": variant, "seed": seed, "prediction": pred}))
            ckpt_hashes[f"{variant}_seed{seed}"] = sha256_file(ckpt)

    predictions = pd.concat(rows, ignore_index=True)
    if not np.isfinite(predictions["prediction"]).all():
        raise RuntimeError("non-finite holdout predictions")
    path = out_dir / "holdout_predictions.csv"
    predictions.to_csv(path, index=False)
    manifest = {"configuration": plan["configuration"], "variants": plan["variants"], "seeds": plan["seeds"],
                "gate_tolerance": GATE_TOLERANCE, "gates": gates, "checkpoint_sha256": ckpt_hashes,
                "holdout_artifact_hashes": holdout_hashes, "v2_artifact_hashes": art["hashes"],
                "predictions_sha256": sha256_file(path), "holdout_labels_read": False}
    (out_dir / "final_predict_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"=== wrote {path}\n=== predictions_sha256 {manifest['predictions_sha256']}", flush=True)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--holdout-dir", required=True, type=Path)
    p.add_argument("--final-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu", "mps"])
    a = p.parse_args()
    sys.exit(run_predict(a.artifact_dir, a.holdout_dir, a.final_root, a.out_dir, a.accelerator))


if __name__ == "__main__":
    main()
