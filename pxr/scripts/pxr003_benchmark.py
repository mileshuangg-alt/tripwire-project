"""PXR-003 interactive runtime benchmark (run locally; never through Slurm).

Times representative fits on the real train/validation data, then extrapolates the full
workload per method: search trials + 9 confirmation fits + 5 final fits (+ their predictions).
Typical = mean of the first three configurations the real search would draw; upper bound =
slowest of those plus three hand-built worst-case corners. Reads train and validation only.

  python -m pxr.scripts.pxr003_benchmark --threads 8 --requested-walltime-minutes 240
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from pxr.scripts import pxr003_common as common
from pxr.scripts import pxr003_models as models

CORNERS = {
    "lgbm": [
        {"learning_rate": 0.01, "num_leaves": 63, "min_child_samples": 5, "colsample_bytree": 0.5,
         "subsample": 1.0, "reg_lambda": 0.001},
        {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 50, "colsample_bytree": 0.05,
         "subsample": 0.5, "reg_lambda": 10.0},
    ],
    "rf": [
        {"max_features": 0.5, "min_samples_leaf": 1, "max_samples": 1.0},
        {"max_features": 0.02, "min_samples_leaf": 8, "max_samples": 0.5},
    ],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--requested-walltime-minutes", type=float, default=None)
    a = p.parse_args()

    cfg = common.load_config()
    t0 = time.perf_counter()
    train, validation, _, _ = common.load_cohort()
    data = common.build_features(train, validation)
    print(f"load + ECFP4 featurization: {time.perf_counter() - t0:.1f}s; threads={a.threads}")

    grand_typical = grand_upper = 0.0
    for model in models.MODELS:
        model_cfg = cfg[model]
        rng = np.random.default_rng(cfg["seeds"]["search"])
        drawn = [models.sample_params(rng, model_cfg["ranges"]) for _ in range(3)]
        typical, worst = [], []
        for label, params_list, bucket in (("drawn", drawn, typical), ("corner", CORNERS[model], None)):
            for params in params_list:
                row = models.evaluate_fit(model, params, model_cfg, data, cfg["seeds"]["search"], a.threads)
                print(f"[{model}] {label} {row['seconds']:.1f}s val_mae={row['val_mae']:.4f} "
                      f"{ {k: row[k] for k in ('best_iteration', 'n_trees') if k in row} } {params}")
                if bucket is not None:
                    bucket.append(row["seconds"])
                worst.append(row["seconds"])
        n_fits = model_cfg["n_trials"] + cfg["confirm_top_k"] * len(cfg["seeds"]["confirm"]) + len(cfg["seeds"]["final"])
        est_typical, est_upper = n_fits * float(np.mean(typical)), n_fits * float(np.max(worst))
        grand_typical += est_typical
        grand_upper += est_upper
        print(f"[{model}] {n_fits} fits: typical {est_typical / 60:.1f} min, upper bound {est_upper / 60:.1f} min")

    print(f"TOTAL (both models, run sequentially): typical {grand_typical / 60:.1f} min, "
          f"upper bound {grand_upper / 60:.1f} min")
    if a.requested_walltime_minutes is not None:
        verdict = "FITS" if grand_upper / 60 <= a.requested_walltime_minutes else "EXCEEDS"
        print(f"upper bound vs requested {a.requested_walltime_minutes:.0f} min: {verdict}")


if __name__ == "__main__":
    main()
