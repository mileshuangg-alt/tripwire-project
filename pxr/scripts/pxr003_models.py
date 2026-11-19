"""PXR-003 model fitting, search and confirmation (ECFP4 + LightGBM, ECFP4 + random forest).

Selection metric is validation MAE only. No holdout data enters this module.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from pxr.scripts.pxr003_common import metrics

MODELS = ("lgbm", "rf")


def sample_value(rng: np.random.Generator, spec: list):
    kind, low, high = spec
    if kind == "loguniform":
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    if kind == "uniform":
        return float(rng.uniform(low, high))
    if kind == "int":
        return int(rng.integers(low, high + 1))
    raise ValueError(f"unknown range kind: {kind}")


def sample_params(rng: np.random.Generator, ranges: dict) -> dict:
    return {name: sample_value(rng, spec) for name, spec in ranges.items()}


def fit_model(model: str, params: dict, model_cfg: dict, x_train, y_train, x_val, y_val, seed: int, threads: int):
    """Fit one model. Returns (predict_fn, info). LightGBM early-stops on validation MAE."""
    if model == "lgbm":
        import lightgbm as lgb

        cap = model_cfg["n_estimators_cap"]
        learner = lgb.LGBMRegressor(
            n_estimators=cap, random_state=seed, n_jobs=threads, verbosity=-1,
            **model_cfg["fixed"], **params,
        )
        learner.fit(
            x_train, y_train, eval_X=x_val, eval_y=y_val,
            callbacks=[lgb.early_stopping(model_cfg["early_stopping_rounds"], first_metric_only=True, verbose=False)],
        )
        best = int(learner.best_iteration_ or cap)
        info = {"best_iteration": best, "best_iteration_at_cap": best >= cap}
        return (lambda x: learner.predict(x, num_iteration=best)), info
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor

        learner = RandomForestRegressor(
            n_estimators=model_cfg["n_estimators"], random_state=seed, n_jobs=threads,
            **model_cfg["fixed"], **params,
        )
        learner.fit(x_train, y_train)
        return learner.predict, {"n_trees": model_cfg["n_estimators"]}
    raise ValueError(f"unknown model: {model}")


def evaluate_fit(model, params, model_cfg, data, seed, threads) -> dict:
    start = time.perf_counter()
    predict, info = fit_model(model, params, model_cfg, data["x_train"], data["y_train"],
                              data["x_validation"], data["y_validation"], seed, threads)
    prediction = predict(data["x_validation"])
    row = {"val_mae": metrics(prediction, data["y_validation"])["mae"],
           "val_rmse": metrics(prediction, data["y_validation"])["rmse"],
           "val_spearman": metrics(prediction, data["y_validation"])["spearman"],
           "seconds": time.perf_counter() - start}
    row.update(info)
    return row


def run_search(model, cfg, data, threads, n_trials=None, log=print) -> pd.DataFrame:
    model_cfg = cfg[model]
    n_trials = model_cfg["n_trials"] if n_trials is None else n_trials
    rng = np.random.default_rng(cfg["seeds"]["search"])
    rows = []
    for trial in range(1, n_trials + 1):
        params = sample_params(rng, model_cfg["ranges"])
        row = {"trial": trial, "seed": cfg["seeds"]["search"] + trial, **params}
        row.update(evaluate_fit(model, params, model_cfg, data, row["seed"], threads))
        rows.append(row)
        log(f"[{model}] trial {trial}/{n_trials} val_mae={row['val_mae']:.5f} seconds={row['seconds']:.1f}")
    return pd.DataFrame(rows)


def select_top(trials: pd.DataFrame, k: int) -> pd.DataFrame:
    """Lowest validation MAE; ties go to the lowest trial number."""
    return trials.sort_values(["val_mae", "trial"], kind="mergesort").head(k)


def run_confirm(model, cfg, data, threads, trials: pd.DataFrame, log=print) -> pd.DataFrame:
    model_cfg = cfg[model]
    param_names = list(model_cfg["ranges"])
    rows = []
    for _, candidate in select_top(trials, cfg["confirm_top_k"]).iterrows():
        params = {name: candidate[name] for name in param_names}
        params = {n: (int(v) if model_cfg["ranges"][n][0] == "int" else float(v)) for n, v in params.items()}
        for seed in cfg["seeds"]["confirm"]:
            row = {"trial": int(candidate["trial"]), "seed": seed, **params}
            row.update(evaluate_fit(model, params, model_cfg, data, seed, threads))
            rows.append(row)
            log(f"[{model}] confirm trial {row['trial']} seed {seed} val_mae={row['val_mae']:.5f}")
    return pd.DataFrame(rows)


def choose_best(confirm: pd.DataFrame) -> dict:
    """Best mean validation MAE across confirmation seeds; ties go to the lowest trial number."""
    summary = confirm.groupby("trial")["val_mae"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values(["mean", "trial"], kind="mergesort")
    best = summary.iloc[0]
    return {"trial": int(best["trial"]), "confirm_mean_val_mae": float(best["mean"]),
            "per_trial_summary": summary.to_dict(orient="records")}
