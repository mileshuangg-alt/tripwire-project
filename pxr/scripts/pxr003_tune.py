"""PXR-003 validation-only tuning for one baseline model (stage 1 search, stage 2 confirmation).

Reads: pxr-challenge_TRAIN.csv and the frozen D014 assignments (hash-verified), pxr003_config.json.
Writes (new directory, refuses to overwrite): trials.csv, confirm.csv, selection.json.
Holdout labels are never used; selection is by validation MAE only.

Run from the repository root:
  python -m pxr.scripts.pxr003_tune --model lgbm --out-dir <new dir> --threads 8
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from pxr.scripts import pxr003_common as common
from pxr.scripts import pxr003_models as models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=models.MODELS)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()

    cfg = common.load_config()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    train, validation, _holdout_ids, hashes = common.load_cohort()
    data = common.build_features(train, validation)
    print(f"cohort: train={len(train)} validation={len(validation)}; threads={args.threads}")

    trials = models.run_search(args.model, cfg, data, args.threads)
    trials.to_csv(args.out_dir / "trials.csv", index=False)
    confirm = models.run_confirm(args.model, cfg, data, args.threads, trials)
    confirm.to_csv(args.out_dir / "confirm.csv", index=False)

    best = models.choose_best(confirm)
    param_names = list(cfg[args.model]["ranges"])
    chosen = confirm.loc[confirm["trial"].eq(best["trial"])].iloc[0]
    params = {n: (int(chosen[n]) if cfg[args.model]["ranges"][n][0] == "int" else float(chosen[n])) for n in param_names}

    selection = {
        "model": args.model,
        "trial": best["trial"],
        "params": params,
        "confirm_mean_val_mae": best["confirm_mean_val_mae"],
        "per_trial_summary": best["per_trial_summary"],
        "trials_at_iteration_cap": int(trials.get("best_iteration_at_cap", []).sum()) if args.model == "lgbm" else None,
        "config_sha256": common.sha256_file(common.CONFIG_PATH),
        "input_hashes": hashes,
        "library_versions": common.library_versions(),
        "threads": args.threads,
        "runtime_seconds": time.perf_counter() - started,
        "selection_metric": "validation MAE",
    }
    common.write_json(args.out_dir / "selection.json", selection)
    print(f"selected trial {best['trial']} mean validation MAE {best['confirm_mean_val_mae']:.5f}")
    print(f"selection.json sha256: {common.sha256_file(args.out_dir / 'selection.json')}")


if __name__ == "__main__":
    main()
