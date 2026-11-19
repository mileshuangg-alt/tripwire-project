"""PXR-003 freeze and final five-model ensembles. Holdout labels are never read here.

  freeze: combine the two tuning selections into frozen_config.json and print its SHA-256
          (record this hash before any holdout scoring).
  train:  verify the frozen-config hash, train 5 models per method on train only, and write
          validation predictions plus structure-only holdout predictions.

Run from the repository root:
  python -m pxr.scripts.pxr003_final freeze --lgbm-selection A/selection.json --rf-selection B/selection.json --out-dir F
  python -m pxr.scripts.pxr003_final train --frozen-config F/frozen_config.json --frozen-sha256 <hash> --out-dir T --threads 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pxr.scripts import pxr003_common as common
from pxr.scripts import pxr003_models as models
from pxr.scripts.pxr001_baseline import ID_COLUMN


def build_frozen(lgbm_selection: dict, rf_selection: dict, config_sha256: str) -> dict:
    for selection, name in ((lgbm_selection, "lgbm"), (rf_selection, "rf")):
        if selection["model"] != name:
            raise ValueError(f"selection for {name} reports model {selection['model']}")
        if selection["config_sha256"] != config_sha256:
            raise ValueError(f"{name} selection was produced with a different pxr003_config.json")
        if selection["input_hashes"]["train_csv"] != common.EXPECTED_TRAIN_SHA256 or \
                selection["input_hashes"]["assignments_csv"] != common.EXPECTED_ASSIGNMENTS_SHA256:
            raise ValueError(f"{name} selection used unrecorded input files")
    cfg = common.load_config()
    return {
        "config_sha256": config_sha256,
        "seeds_final": cfg["seeds"]["final"],
        "selection_metric": "validation MAE",
        "models": {
            "lgbm": {"params": lgbm_selection["params"], "trial": lgbm_selection["trial"],
                     "confirm_mean_val_mae": lgbm_selection["confirm_mean_val_mae"]},
            "rf": {"params": rf_selection["params"], "trial": rf_selection["trial"],
                   "confirm_mean_val_mae": rf_selection["confirm_mean_val_mae"]},
        },
        "input_hashes": lgbm_selection["input_hashes"],
        "note": "Frozen before any holdout scoring. Holdout scores may not change these choices.",
    }


def verify_frozen(path: Path, expected_sha256: str) -> dict:
    observed = common.sha256_file(path)
    if observed != expected_sha256:
        raise SystemExit(f"frozen config hash {observed} != supplied {expected_sha256}")
    return json.loads(Path(path).read_text())


def cmd_freeze(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frozen = build_frozen(
        json.loads(args.lgbm_selection.read_text()),
        json.loads(args.rf_selection.read_text()),
        common.sha256_file(common.CONFIG_PATH),
    )
    target = args.out_dir / "frozen_config.json"
    common.write_json(target, frozen)
    print(json.dumps(frozen, indent=2, sort_keys=True))
    print(f"frozen_config.json sha256: {common.sha256_file(target)}")


def cmd_train(args) -> None:
    frozen = verify_frozen(args.frozen_config, args.frozen_sha256)
    cfg = common.load_config()
    if common.sha256_file(common.CONFIG_PATH) != frozen["config_sha256"]:
        raise SystemExit("pxr003_config.json changed since the selections were frozen")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    train, validation, holdout_ids, hashes = common.load_cohort()
    if hashes != frozen["input_hashes"]:
        raise SystemExit("input file hashes differ from the frozen configuration")
    data = common.build_features(train, validation)
    structures = common.load_holdout_structures()
    x_holdout = common.holdout_matrix(structures)
    if set(structures[ID_COLUMN]) != holdout_ids:
        raise AssertionError("holdout structure ids differ from the holdout assignments")

    rows, per_model, best_iterations = [], [], {}
    for model in models.MODELS:
        params = frozen["models"][model]["params"]
        for seed in frozen["seeds_final"]:
            predict, info = models.fit_model(model, params, cfg[model], data["x_train"], data["y_train"],
                                             data["x_validation"], data["y_validation"], seed, args.threads)
            val_pred, hold_pred = predict(data["x_validation"]), predict(x_holdout)
            for split, ids, pred in (("validation", validation[ID_COLUMN].tolist(), val_pred),
                                     ("holdout", structures[ID_COLUMN].tolist(), hold_pred)):
                rows.append(pd.DataFrame({"row_id": ids, "split": split, "model": model, "seed": seed,
                                          "prediction": pred}))
            m = common.metrics(val_pred, data["y_validation"])
            per_model.append({"model": model, "seed": seed, **m, **info})
            print(f"[{model}] seed {seed} validation MAE={m['mae']:.5f} {info}")

    predictions = pd.concat(rows, ignore_index=True)
    if not np.isfinite(predictions["prediction"]).all():
        raise AssertionError("non-finite predictions")
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)

    train_mean = float(data["y_train"].mean())
    summary = {"n_train": len(train), "train_mean_pEC50": train_mean,
               "constant_baseline_validation": common.constant_metrics(train_mean, data["y_validation"]),
               "per_model_validation": per_model, "ensemble_validation": {}}
    for model in models.MODELS:
        sub = predictions[(predictions["model"] == model) & (predictions["split"] == "validation")]
        ensemble = sub.groupby("row_id")["prediction"].mean().loc[validation[ID_COLUMN]].to_numpy()
        summary["ensemble_validation"][model] = common.metrics(ensemble, data["y_validation"])
    common.write_json(args.out_dir / "validation_summary.json", summary)
    common.write_json(args.out_dir / "manifest.json", {
        "frozen_config_sha256": args.frozen_sha256,
        "predictions_sha256": common.sha256_file(args.out_dir / "predictions.csv"),
        "input_hashes": hashes, "library_versions": common.library_versions(),
        "threads": args.threads, "runtime_seconds": time.perf_counter() - started,
        "holdout_labels_read": False,
    })
    print(json.dumps(summary["ensemble_validation"], indent=2))
    print(f"predictions.csv sha256: {common.sha256_file(args.out_dir / 'predictions.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--lgbm-selection", required=True, type=Path)
    f.add_argument("--rf-selection", required=True, type=Path)
    f.add_argument("--out-dir", required=True, type=Path)
    f.set_defaults(func=cmd_freeze)
    t = sub.add_parser("train")
    t.add_argument("--frozen-config", required=True, type=Path)
    t.add_argument("--frozen-sha256", required=True)
    t.add_argument("--out-dir", required=True, type=Path)
    t.add_argument("--threads", type=int, default=os.cpu_count())
    t.set_defaults(func=cmd_train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
