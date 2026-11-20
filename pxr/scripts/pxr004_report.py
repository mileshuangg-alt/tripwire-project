"""PXR-004 pilot report (D020): descriptive per-seed results and paired differences.

Reads the six stage-2 `metrics.json` files (and, for the sequential arm, the stage-1
`metrics.json`) written by `pxr004_train.py`, plus each stage-2 `validation_predictions_best.csv`.
Prints a markdown report and optionally writes a JSON copy. It trains nothing, reads no
holdout or official-test label, and applies no threshold or statistical test: D020 asks for
validation MAE, Spearman and RMSE per seed and mean, with per-seed paired differences,
descriptive only.

Layout read: <pilot-root>/seed<SEED>/<arm>_seed<SEED>/stage1|stage2/metrics.json, where
<pilot-root> is the directory that holds the per-seed OUT_ROOT directories. Seeds are passed
explicitly (the approved values live in D020, not in this script). Context points are passed
on the command line as LABEL=VALUE and are printed as descriptive reference only.

Integrity checks (a failed check raises; nothing is silently skipped):
  * every expected file exists, and `arm`/`seed`/`stage` in each metrics.json match its path;
  * MAE and RMSE recomputed from validation_predictions_best.csv match metrics.json
    (tolerance 1e-6), which guards against a metrics file paired with the wrong predictions;
  * every arm has the same validation row ids for a given seed, and row ids are unique.
Spearman is taken from metrics.json as written by the trainer (no recomputation here).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ARMS = ("direct", "sequential")
METRICS = ("mae", "rmse", "spearman")
LOWER_IS_BETTER = {"mae": True, "rmse": True, "spearman": False}
_KEY = {"mae": "best_primary_validation_mae", "rmse": "best_primary_validation_rmse",
        "spearman": "best_primary_validation_spearman"}
CHECK_TOL = 1e-6


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing expected file: {path}")
    return json.loads(path.read_text())


def run_dir(pilot_root: Path, arm: str, seed: int) -> Path:
    return pilot_root / f"seed{seed}" / f"{arm}_seed{seed}"


def read_predictions(path: Path) -> tuple[list[str], list[float], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing expected file: {path}")
    ids, y, p = [], [], []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            ids.append(row["row_id"])
            y.append(float(row["pEC50_true"]))
            p.append(float(row["pEC50_pred"]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate row_id in {path}")
    if not ids:
        raise ValueError(f"no rows in {path}")
    return ids, y, p


def mae_rmse(y: list[float], p: list[float]) -> tuple[float, float]:
    err = [a - b for a, b in zip(y, p)]
    return sum(abs(e) for e in err) / len(err), math.sqrt(sum(e * e for e in err) / len(err))


def load_arm_seed(pilot_root: Path, arm: str, seed: int) -> dict:
    d = run_dir(pilot_root, arm, seed)
    m = _load(d / "stage2" / "metrics.json")
    if (m.get("stage"), m.get("arm"), m.get("seed")) != (2, arm, seed):
        raise ValueError(f"{d / 'stage2' / 'metrics.json'} reports stage/arm/seed "
                         f"{(m.get('stage'), m.get('arm'), m.get('seed'))}, expected {(2, arm, seed)}")
    ids, y, p = read_predictions(d / "stage2" / "validation_predictions_best.csv")
    mae, rmse = mae_rmse(y, p)
    for name, value in (("mae", mae), ("rmse", rmse)):
        if abs(value - m[_KEY[name]]) > CHECK_TOL:
            raise ValueError(f"{arm} seed {seed}: recomputed {name} {value:.9f} disagrees with "
                             f"metrics.json {m[_KEY[name]]:.9f}")
    out = {k: float(m[_KEY[k]]) for k in METRICS}
    out.update(best_epoch=int(m["best_epoch"]), epochs_run=int(m["epochs_run"]), n_validation=len(ids),
               row_ids=ids)
    return out


def load_stage1(pilot_root: Path, seed: int) -> dict:
    path = run_dir(pilot_root, "sequential", seed) / "stage1" / "metrics.json"
    m = _load(path)
    if (m.get("stage"), m.get("seed")) != (1, seed):
        raise ValueError(f"{path} reports stage/seed {(m.get('stage'), m.get('seed'))}, expected {(1, seed)}")
    return {"best_monitor_pooled_mse": float(m["best_monitor_pooled_mse"]), "best_epoch": int(m["best_epoch"]),
            "epochs_run": int(m["epochs_run"]), "heads": m["best_heads"]}


def build_report(pilot_root: Path, seeds: list[int]) -> dict:
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be a non-empty list of distinct values")
    per = {arm: {s: load_arm_seed(pilot_root, arm, s) for s in seeds} for arm in ARMS}
    for s in seeds:
        if per["direct"][s]["row_ids"] != per["sequential"][s]["row_ids"] \
                or per["direct"][s]["n_validation"] != per["sequential"][s]["n_validation"]:
            raise ValueError(f"seed {s}: the two arms were scored on different validation rows")
    n_val = {per[a][s]["n_validation"] for a in ARMS for s in seeds}
    if len(n_val) != 1:
        raise ValueError(f"validation row counts differ across runs: {sorted(n_val)}")
    for arm in ARMS:
        for s in seeds:
            del per[arm][s]["row_ids"]
    paired = {m: {s: per["sequential"][s][m] - per["direct"][s][m] for s in seeds} for m in METRICS}
    means = {arm: {m: statistics.fmean(per[arm][s][m] for s in seeds) for m in METRICS} for arm in ARMS}
    mean_paired = {m: statistics.fmean(paired[m].values()) for m in METRICS}
    sd = {arm: {m: (statistics.stdev(per[arm][s][m] for s in seeds) if len(seeds) > 1 else None)
                for m in METRICS} for arm in ARMS}
    return {"seeds": seeds, "n_validation": n_val.pop(), "per_seed": per, "arm_means": means,
            "arm_sd_across_seeds": sd, "paired_sequential_minus_direct": paired,
            "mean_paired_sequential_minus_direct": mean_paired,
            "stage1_monitor": {s: load_stage1(pilot_root, s) for s in seeds}}


def parse_context(items: list[str]) -> list[tuple[str, float]]:
    out = []
    for item in items:
        label, sep, value = item.rpartition("=")
        if not sep or not label:
            raise ValueError(f"--context must be LABEL=VALUE, got {item!r}")
        out.append((label, float(value)))
    return out


def render_markdown(rep: dict, context: list[tuple[str, float]]) -> str:
    seeds = rep["seeds"]
    L = ["# PXR-004 pilot report (descriptive; validation set only)", "",
         f"Validation compounds scored: {rep['n_validation']}. Selection was on primary-head validation MAE. "
         "Differences are sequential minus direct (lower is better for MAE and RMSE, higher for Spearman). "
         "No threshold and no statistical test (D020).", "",
         "## Stage 2, per seed", "", "| Seed | Arm | MAE | RMSE | Spearman | Best epoch | Epochs run |",
         "|---|---|---|---|---|---|---|"]
    for s in seeds:
        for arm in ARMS:
            r = rep["per_seed"][arm][s]
            L.append(f"| {s} | {arm} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['spearman']:.4f} | "
                     f"{r['best_epoch']} | {r['epochs_run']} |")
    L += ["", "## Paired differences (sequential minus direct)", "",
          "| Metric | " + " | ".join(str(s) for s in seeds) + " | Mean |", "|---|" + "---|" * (len(seeds) + 1)]
    for m in METRICS:
        L.append(f"| {m.upper() if m != 'spearman' else 'Spearman'} | "
                 + " | ".join(f"{rep['paired_sequential_minus_direct'][m][s]:+.4f}" for s in seeds)
                 + f" | {rep['mean_paired_sequential_minus_direct'][m]:+.4f} |")
    L += ["", "## Arm means across seeds (SD across seeds in brackets)", "",
          "| Arm | MAE | RMSE | Spearman |", "|---|---|---|---|"]
    for arm in ARMS:
        cells = []
        for m in METRICS:
            sd = rep["arm_sd_across_seeds"][arm][m]
            cells.append(f"{rep['arm_means'][arm][m]:.4f}" + (f" [{sd:.4f}]" if sd is not None else ""))
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["", "## Stage 1 monitor (sequential arm)", "",
          "| Seed | Pooled std. MSE | Best epoch | Epochs run | Head MAE (" + " / ".join(
              sorted(next(iter(rep["stage1_monitor"].values()))["heads"])) + ") |", "|---|---|---|---|---|"]
    for s in seeds:
        st = rep["stage1_monitor"][s]
        heads = " / ".join(f"{st['heads'][h]['mae']:.4f}" for h in sorted(st["heads"]))
        L.append(f"| {s} | {st['best_monitor_pooled_mse']:.4f} | {st['best_epoch']} | {st['epochs_run']} | {heads} |")
    if context:
        L += ["", "## Context points (descriptive, not matched comparators)", "", "| Label | Validation MAE |",
              "|---|---|"] + [f"| {label} | {value:.4f} |" for label, value in context]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot-root", required=True, type=Path,
                    help="directory containing seed<SEED>/<arm>_seed<SEED>/stage1|stage2")
    ap.add_argument("--seeds", required=True, type=int, nargs="+")
    ap.add_argument("--context", action="append", default=[], metavar="LABEL=VALUE")
    ap.add_argument("--out-json", type=Path, help="also write the full report as JSON (must not exist)")
    a = ap.parse_args(argv)
    if a.out_json is not None and a.out_json.exists():
        raise SystemExit(f"--out-json already exists: {a.out_json}")
    rep = build_report(a.pilot_root, a.seeds)
    sys.stdout.write(render_markdown(rep, parse_context(a.context)))
    if a.out_json is not None:
        a.out_json.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
