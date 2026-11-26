"""PXR-002 final-model predictions for an explicit molecule frame (Tripwire-CYP-011).

Uses the ten frozen D018 Amendment 6 checkpoints (multitask and aux_off, seeds 20261120 to
20261124, configuration h600_d4_f2_p0.2) exactly as pxr002_final_predict.py does, but on a
hash-verified frame built by pxr002_build_predict_frame.py instead of the holdout artifact.
Nothing is trained, tuned or selected.

Per checkpoint the original hard gate is kept: the checkpoint is first reloaded and must
reproduce its recorded best validation MAE on the v2 validation frame (tolerance 1e-3); a
failed gate stops the run before any prediction is made for that or later models.

Report values: the primary column is the arithmetic mean of the five multitask seeds
(PXR_pEC50_multitask_mean5); the aux_off five-seed mean (PXR_pEC50_aux_off_mean5) is the
secondary column. Every per-seed prediction is also written. No variant or seed is selected.

Outputs (new, empty directory): pxr_potency_seed_predictions.csv (row_id, variant, seed,
prediction), pxr_potency_means.csv (row_id plus the two mean columns), pxr_potency_manifest.json.

Usage (CoreHPC, run by the project owner; every argument is required):
    python -I pxr/scripts/pxr002_predict_frame.py --artifact-dir D --predict-dir P \
        --predict-sums-sha256 H --final-root R --final-plan-sha256 H --out-dir NEW --accelerator cpu|gpu|mps
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr002_dataset as ds  # noqa: E402
from pxr.scripts import pxr002_final_predict as fp  # noqa: E402

VARIANTS = ("multitask", "aux_off")
SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)
CONFIGURATION = "h600_d4_f2_p0.2"
PRIMARY_COLUMN = "PXR_pEC50_multitask_mean5"
SECONDARY_COLUMN = "PXR_pEC50_aux_off_mean5"
PREDICT_FRAME = "pxr002_predict_frame.csv"
PREDICT_DESCRIPTORS = "pxr002_predict_descriptors.npy"
OUT_SEEDS = "pxr_potency_seed_predictions.csv"
OUT_MEANS = "pxr_potency_means.csv"
OUT_MANIFEST = "pxr_potency_manifest.json"


class PredictStop(RuntimeError):
    """A condition that stops the run; no variant or seed is substituted."""


def check_plan(plan: dict) -> None:
    """The plan must be exactly the approved ten models; anything else stops the run."""
    if plan.get("configuration") != CONFIGURATION:
        raise PredictStop(f"Unexpected configuration {plan.get('configuration')!r}.")
    if set(plan.get("variants", ())) != set(VARIANTS) or len(plan.get("variants", ())) != len(VARIANTS):
        raise PredictStop(f"Unexpected variants {plan.get('variants')!r}.")
    if sorted(int(s) for s in plan.get("seeds", ())) != list(SEEDS):
        raise PredictStop(f"Unexpected seeds {plan.get('seeds')!r}.")


def load_predict_frame(predict_dir, expected_sums_sha256: str):
    """Hash-verified frame (row_id, SMILES only) and raw descriptors."""
    d = Path(predict_dir)
    actual = ds.sha256(d / "SHA256SUMS")
    if actual != expected_sums_sha256:
        raise PredictStop(f"SHA256SUMS hash {actual} differs from the declared {expected_sums_sha256}.")
    listed = ds.verify_artifact(d)
    if set(listed) != {PREDICT_FRAME, PREDICT_DESCRIPTORS}:
        raise PredictStop(f"Artifact lists {sorted(listed)}; expected the frame and descriptors only.")
    frame = pd.read_csv(d / PREDICT_FRAME, dtype=str, keep_default_na=False)
    if list(frame.columns) != ["row_id", "SMILES"]:
        raise PredictStop("Prediction frame must contain only row_id and SMILES.")
    if frame["row_id"].duplicated().any() or (frame["row_id"].str.strip() == "").any():
        raise PredictStop("Prediction frame has duplicate or empty row_id values.")
    x = np.load(d / PREDICT_DESCRIPTORS, allow_pickle=False)
    if x.dtype != np.float64 or x.shape != (len(frame), ds.N_DESCRIPTORS) or not np.isfinite(x).all():
        raise PredictStop("Descriptor cache has an unexpected shape, dtype or non-finite values.")
    return frame, x, listed


def parse_timing_target(text: str) -> tuple:
    """'variant:seed' naming exactly one of the ten approved models, or stop."""
    variant, _, seed = str(text).partition(":")
    if variant not in VARIANTS or not seed.isdigit() or int(seed) not in SEEDS:
        raise PredictStop(f"--timing-only-checkpoint must be variant:seed of an approved model; got {text!r}.")
    return variant, int(seed)


def five_seed_mean(values) -> float:
    values = [float(v) for v in values]
    if len(values) != len(SEEDS) or not all(math.isfinite(v) for v in values):
        raise PredictStop("A mean needs exactly five finite seed predictions.")
    return math.fsum(values) / len(values)


def build_means(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-row five-seed means by variant; every (row, variant) must have all five seeds once."""
    counts = predictions.groupby(["row_id", "variant"]).size()
    if not (counts == len(SEEDS)).all() or len(counts) != predictions["row_id"].nunique() * len(VARIANTS):
        raise PredictStop("Each row and variant must have exactly five seed predictions.")
    records = []
    for row_id, block in predictions.groupby("row_id", sort=False):
        record = {"row_id": row_id}
        for variant, column in (("multitask", PRIMARY_COLUMN), ("aux_off", SECONDARY_COLUMN)):
            part = block[block["variant"] == variant]
            record[column] = repr(five_seed_mean(part["prediction"].tolist()))
        records.append(record)
    return pd.DataFrame(records, columns=["row_id", PRIMARY_COLUMN, SECONDARY_COLUMN])


def run_predict_frame(args, loader=fp.load_final_model, predictor=fp.predict_primary) -> int:
    from pxr.scripts.pxr002_grid import build_grid

    out_dir = Path(args.out_dir)
    plan_path = Path(args.final_root) / "final_plan.json"
    if ds.sha256(plan_path) != args.final_plan_sha256:
        raise PredictStop("final_plan.json hash differs from the declared hash.")
    plan = json.loads(plan_path.read_text())
    check_plan(plan)
    frame, descriptors, frame_hashes = load_predict_frame(args.predict_dir, args.predict_sums_sha256)
    out_dir.mkdir(parents=True, exist_ok=False)  # refuses to overwrite earlier predictions

    from pxr.scripts.pxr002_dataset import load_artifact
    art = load_artifact(args.artifact_dir)
    validation, val_x = art["validation"], art["val_x"]
    config = {c.name: c for c in build_grid()}[plan["configuration"]]

    timing = getattr(args, "timing_only_checkpoint", None)
    targets = [(v, s) for v in VARIANTS for s in SEEDS]
    if timing is not None:
        targets = [parse_timing_target(timing)]
    rows, gates, ckpt_hashes, seconds = [], [], {}, []
    for variant, seed in targets:
        started = time.time()
        run = Path(args.final_root) / f"{variant}_seed{seed}"
        ckpt = run / "best.ckpt"
        recorded = json.loads((run / "metrics.json").read_text())["best_primary_validation_mae"]
        model = loader(ckpt, config, val_x.shape[1])
        val_pred = predictor(model, validation["SMILES"].tolist(), val_x, args.accelerator)
        replicated = float(np.mean(np.abs(val_pred - validation["pEC50"].to_numpy(dtype=float))))
        gate = {"variant": variant, "seed": seed, "recorded_val_mae": recorded,
                "replicated_val_mae": replicated, "abs_difference": abs(replicated - recorded)}
        gates.append(gate)
        print(f"=== gate {variant} seed {seed}: recorded {recorded:.6f} replicated {replicated:.6f}", flush=True)
        if gate["abs_difference"] > fp.GATE_TOLERANCE:
            (out_dir / "gate_failure.json").write_text(json.dumps(gates, indent=2) + "\n")
            print("=== GATE FAILED: no predictions were made for this or later models", flush=True)
            return 1
        pred = predictor(model, frame["SMILES"].tolist(), descriptors, args.accelerator)
        rows.append(pd.DataFrame({"row_id": frame["row_id"], "variant": variant, "seed": seed, "prediction": pred}))
        ckpt_hashes[f"{variant}_seed{seed}"] = fp.sha256_file(ckpt)
        seconds.append(time.time() - started)

    if timing is not None:
        total = seconds[0]
        record = {
            "mode": "timing_only", "checkpoint": f"{targets[0][0]}_seed{targets[0][1]}",
            "accelerator": args.accelerator, "gate_molecules": int(len(validation)),
            "input_molecules_predicted": int(len(frame)), "gate": gates[0],
            "seconds": {"member_total_including_load_gate_and_prediction": total},
            "extrapolated_10_model_seconds": total * len(VARIANTS) * len(SEEDS),
            "extrapolation_note": "one model times ten; one-model cold start included; not a guarantee",
            "means_written": False, "script_sha256": ds.sha256(__file__),
        }
        (out_dir / "pxr_potency_timing.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"=== timing record {out_dir / 'pxr_potency_timing.json'}", flush=True)
        return 0

    predictions = pd.concat(rows, ignore_index=True)
    if not np.isfinite(predictions["prediction"]).all():
        raise PredictStop("non-finite predictions")
    means = build_means(predictions)
    predictions.to_csv(out_dir / OUT_SEEDS, index=False)
    means.to_csv(out_dir / OUT_MEANS, index=False)
    manifest = {
        "decision": "D018 Amendment 6 (models); D027 and CYP-011 report scope",
        "configuration": plan["configuration"], "variants": list(VARIANTS), "seeds": list(SEEDS),
        "gate_tolerance": fp.GATE_TOLERANCE, "gates": gates, "checkpoint_sha256": ckpt_hashes,
        "predict_artifact_hashes": frame_hashes, "predict_sums_sha256": args.predict_sums_sha256,
        "final_plan_sha256": args.final_plan_sha256, "v2_artifact_hashes": art["hashes"],
        "molecules": int(len(frame)), "primary_column": PRIMARY_COLUMN, "secondary_column": SECONDARY_COLUMN,
        "outputs": {OUT_SEEDS: ds.sha256(out_dir / OUT_SEEDS), OUT_MEANS: ds.sha256(out_dir / OUT_MEANS)},
        "script_sha256": ds.sha256(__file__), "labels_read_for_prediction_frame": False,
        "training_performed": False,
    }
    (out_dir / OUT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"=== wrote {out_dir / OUT_MEANS}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--predict-dir", required=True, type=Path)
    p.add_argument("--predict-sums-sha256", required=True)
    p.add_argument("--final-root", required=True, type=Path)
    p.add_argument("--final-plan-sha256", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu", "mps"])
    p.add_argument("--timing-only-checkpoint", default=None,
                   help="variant:seed; run exactly that one model and write only a timing record")
    args = p.parse_args(argv)
    try:
        return run_predict_frame(args)
    except PredictStop as exc:
        raise SystemExit(f"STOPPED: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
