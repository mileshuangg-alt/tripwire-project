"""Cluster-level error concentration for D022 arm R replicate 1 (read-only diagnostic).

Reads only the existing saved validation predictions of the 150 updated-T121 cells
(t121_valinf_full_20261118/full_cells_000_149), the frozen cluster assignments and the frozen
D007 split manifests. Writes only into a NEW output directory. It trains nothing, runs no
inference, loads no checkpoint and opens no protected-test data.

What it computes, per isoform (CYP1A2, CYP2C9, CYP2D6 and CYP3A4 are treated alike):
  * signed error (predicted - measured pIC50), absolute error, and the confidence-band-aware
    ("soft") error of the frozen scorer (references/cyp_upstream/custom_scoring_functions.py,
    rae_soft_threshold_absolute_error, lines 136-138: clip(pred - high, 0) + clip(low - pred, 0));
  * prediction-row counts AND unique-molecule counts. A molecule is validated once per
    repetition and with three seeds, so rows are not independent. Cluster and molecule results
    are therefore computed on molecule-level means: rows -> seed mean within a repetition ->
    mean over repetitions, so each repetition counts equally and each molecule counts once;
  * how concentrated the error is across clusters (cluster share of total error vs share of
    molecules, top-k shares, cumulative curve);
  * descriptive variation across repetitions, folds, seeds and cells, and prediction spread
    across seeds and repetitions. These cells share partitions and are NOT independent
    observations; no significance statement is made.

Limits that apply to every output (also written to run_metadata.json):
  * The saved predictions are analysed exactly as they were scored (post-heuristic). Their
    pre-heuristic scale provenance is unresolved: the branch taken by the scorer's scaling
    heuristic (cyp002_validation.py lines 195-218) was not recorded for any cell. The range check
    only flags anomalies. It cannot prove the historical branch, and matching scores cannot
    either. This script does not reapply the heuristic and does not require residuals to be
    centred as a condition for analysis.
  * Descriptive only. No overfitting or generalization-gap diagnosis is possible here: no
    matched training-side evidence exists. No penalty, training or experiment is proposed.
  * Potency compression is not the focus (session 007 covers it).

Usage:
    python -I cyp/scripts/t121_error_diagnostic.py --selftest
    python -I cyp/scripts/t121_error_diagnostic.py \
        --predictions-dir DIR --cluster-assignments CSV --split-manifest-dir DIR \
        --output-dir NEW_DIR [--cells I J K ...] [--sha256sums FILE] [--reproduction-dir DIR]
Without --cells all 150 cells are required. A subset is labelled partial in the metadata.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile

import numpy as np
import pandas as pd

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
TASK_SUFFIX = "_pIC50_direct_inhibition"
COLUMNS = [
    "trial_index", "cell_index", "repetition", "fold", "seed", "isoform",
    "Molecule_Name", "y_true", "y_pred", "conf_low", "conf_high",
]
TRIAL_INDEX = 121
N_CELLS = 150
TOP_K = (1, 5, 10, 20)
ERROR_SHARE_TARGETS = (0.5, 0.8)
MOLECULE_TOP_FRACTIONS = (0.01, 0.05, 0.10)
# Anomaly flags only (not a proof of the historical scale branch).
PRED_RANGE = (0.0, 10.0)
MEDIAN_BAND_DEVIATION_LIMIT = 3.0
NOTES = [
    "Saved post-heuristic predictions analysed exactly as scored; pre-heuristic scale provenance remains unresolved.",
    "Range checks flag anomalies only; they cannot prove the historical scaling branch, and matching scores cannot either.",
    "The scorer's scaling heuristic is not reapplied; residuals are not required to be centred.",
    "Cells share partitions and compounds; counts are rows vs unique molecules; no significance claim; results are descriptive.",
    "No matched training-side evidence is used; nothing here diagnoses overfitting or a generalization gap.",
    "Molecule-level means: seed mean within a repetition, then mean over repetitions (each repetition counts equally).",
]


# ----------------------------------------------------------------------------- input
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sums(path):
    """`sha256sum` format: '<hash>  <relative name>' per line."""
    sums = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if len(digest) != 64 or not name:
            raise ValueError(f"Unreadable SHA-256 line: {line!r}")
        sums[name.strip()] = digest
    return sums


def load_predictions(predictions_dir, cells, sums=None):
    predictions_dir = Path(predictions_dir)
    names = {i: f"predictions_cell_{i:03d}.csv" for i in cells}
    missing = [n for n in names.values() if not (predictions_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} required prediction file(s) missing in {predictions_dir}: {missing[:5]}"
        )
    frames, hashes = [], {}
    for cell, name in names.items():
        path = predictions_dir / name
        digest = sha256_file(path)
        if sums is not None:
            if name not in sums:
                raise ValueError(f"{name} is not listed in the SHA-256 file.")
            if sums[name] != digest:
                raise ValueError(f"{name} does not match its recorded SHA-256.")
        hashes[name] = digest
        frame = pd.read_csv(path)
        if list(frame.columns) != COLUMNS:
            raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
        if frame.empty:
            raise ValueError(f"{name}: no rows")
        if not (frame["cell_index"] == cell).all():
            raise ValueError(f"{name}: cell_index does not match the file name")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if set(df["trial_index"]) != {TRIAL_INDEX}:
        raise ValueError(f"Unexpected trial_index values: {sorted(set(df['trial_index']))}")
    unknown = set(df["isoform"]) - set(ISOFORMS)
    if unknown:
        raise ValueError(f"Unknown isoform labels: {sorted(unknown)}")
    numeric = df[["y_true", "y_pred", "conf_low", "conf_high"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite value in y_true, y_pred, conf_low or conf_high.")
    if (df["conf_low"] > df["conf_high"]).any():
        raise ValueError("conf_low exceeds conf_high in some row.")
    if df.duplicated(["cell_index", "isoform", "Molecule_Name"]).any():
        raise ValueError("Duplicate (cell, isoform, molecule) prediction rows.")
    identity = df.groupby("cell_index")[["repetition", "fold", "seed"]].nunique()
    if (identity != 1).any().any():
        raise ValueError("A cell has more than one repetition, fold or seed.")
    return df, hashes


def load_clusters(path):
    clusters = pd.read_csv(path)
    for column in ("Molecule_Name", "cluster_id", "split"):
        if column not in clusters.columns:
            raise ValueError(f"Cluster assignments lack column {column}.")
    if clusters["Molecule_Name"].duplicated().any():
        raise ValueError("Cluster assignments list a molecule more than once.")
    return clusters


def load_manifests(df, manifest_dir):
    manifests, hashes = {}, {}
    keys = sorted({(int(r), int(f)) for r, f in zip(df["repetition"], df["fold"])})
    for repetition, fold in keys:
        path = Path(manifest_dir) / f"rep_{repetition:02d}_fold_{fold:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Split manifest missing: {path}")
        manifest = json.loads(path.read_text())
        if (int(manifest["repetition"]), int(manifest["fold"])) != (repetition, fold):
            raise ValueError(f"{path.name}: repetition/fold do not match its name")
        manifests[(repetition, fold)] = {
            "names": set(manifest["validation_names"]),
            "clusters": {int(c) for c in manifest["validation_cluster_ids"]},
            "seeds": {int(s) for s in manifest["training_seeds"]},
        }
        hashes[path.name] = sha256_file(path)
    return manifests, hashes


def attach_and_validate(df, clusters, manifests):
    merged = df.merge(
        clusters[["Molecule_Name", "cluster_id", "split"]],
        on="Molecule_Name", how="left", validate="many_to_one",
    )
    if merged["cluster_id"].isna().any():
        bad = sorted(set(merged.loc[merged["cluster_id"].isna(), "Molecule_Name"]))[:5]
        raise ValueError(f"Molecules without a cluster assignment: {bad}")
    if (merged["split"] == "test").any():
        raise ValueError("A protected internal-test molecule appears in the predictions; stopping.")
    merged["cluster_id"] = merged["cluster_id"].astype(int)
    warnings = []
    for (repetition, fold), group in merged.groupby(["repetition", "fold"]):
        manifest = manifests[(int(repetition), int(fold))]
        outside = set(group["Molecule_Name"]) - manifest["names"]
        if outside:
            raise ValueError(
                f"rep {repetition} fold {fold}: molecules not in the manifest's validation names: {sorted(outside)[:5]}"
            )
        bad_seeds = set(group["seed"].astype(int)) - manifest["seeds"]
        if bad_seeds:
            raise ValueError(f"rep {repetition} fold {fold}: seeds not in the manifest: {sorted(bad_seeds)}")
        off = int((~group["cluster_id"].isin(manifest["clusters"])).sum())
        if off:
            warnings.append(f"rep {repetition} fold {fold}: {off} rows whose cluster id is not in validation_cluster_ids")
    constant = merged.groupby(["isoform", "Molecule_Name"])[["y_true", "conf_low", "conf_high"]].nunique()
    if (constant != 1).any().any():
        raise ValueError("A molecule's label or confidence band differs between cells.")
    folds_per_rep = merged.groupby(["isoform", "Molecule_Name", "repetition"])["fold"].nunique()
    if (folds_per_rep != 1).any():
        raise ValueError("A molecule is validated in more than one fold within a repetition.")
    return merged, warnings


# ------------------------------------------------------------------------- errors
def soft_error(y_pred, low, high):
    """Frozen scorer per-row error: clip(pred - high, 0) + clip(low - pred, 0)."""
    return np.clip(y_pred - high, 0, None) + np.clip(low - y_pred, 0, None)


def add_errors(df):
    df = df.copy()
    df["signed"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["signed"].abs()
    df["soft"] = soft_error(df["y_pred"].to_numpy(), df["conf_low"].to_numpy(), df["conf_high"].to_numpy())
    df["inside_band"] = df["soft"] == 0
    return df


def st_rae(y_true, y_pred, low, high):
    """Per-cell ST-RAE as in the frozen scorer (baseline: constant mean of y_true)."""
    numerator = soft_error(y_pred, low, high).sum()
    mean_true = np.mean(y_true)
    denominator = (np.clip(mean_true - high, 0, None) + np.clip(low - mean_true, 0, None)).sum()
    return float(numerator / denominator) if denominator > 0 else float("nan")


def cell_metrics(df, reproduction_dir=None):
    rows = []
    for (cell, isoform), g in df.groupby(["cell_index", "isoform"], sort=True):
        row = {
            "cell_index": int(cell), "isoform": isoform,
            "repetition": int(g["repetition"].iloc[0]), "fold": int(g["fold"].iloc[0]),
            "seed": int(g["seed"].iloc[0]), "n_rows": int(len(g)),
            "mean_signed": float(g["signed"].mean()), "mean_abs": float(g["abs_error"].mean()),
            "mean_soft": float(g["soft"].mean()), "share_inside_band": float(g["inside_band"].mean()),
            "st_rae_recomputed": st_rae(
                g["y_true"].to_numpy(), g["y_pred"].to_numpy(),
                g["conf_low"].to_numpy(), g["conf_high"].to_numpy(),
            ),
            "st_rae_recorded_original": float("nan"), "st_rae_recorded_from_rows": float("nan"),
        }
        if reproduction_dir is not None:
            path = Path(reproduction_dir) / f"reproduction_cell_{int(cell):03d}.json"
            if path.is_file():
                record = json.loads(path.read_text())
                row["st_rae_recorded_original"] = float(record["original_scores"]["per_task"][isoform + TASK_SUFFIX])
                row["st_rae_recorded_from_rows"] = float(record["metrics_from_rows"][isoform]["st_rae"])
        row["abs_diff_vs_recorded_original"] = abs(row["st_rae_recomputed"] - row["st_rae_recorded_original"])
        rows.append(row)
    return pd.DataFrame(rows)


def range_check(df):
    rows = []
    for (cell, isoform), g in df.groupby(["cell_index", "isoform"], sort=True):
        midpoint = (g["conf_low"] + g["conf_high"]) / 2.0
        median_dev = float(np.median(np.abs(g["y_pred"] - midpoint)))
        low_flag = bool(g["y_pred"].min() < PRED_RANGE[0] or g["y_pred"].max() > PRED_RANGE[1])
        dev_flag = bool(median_dev > MEDIAN_BAND_DEVIATION_LIMIT)
        rows.append({
            "cell_index": int(cell), "isoform": isoform, "n_rows": int(len(g)),
            "y_true_min": float(g["y_true"].min()), "y_true_max": float(g["y_true"].max()),
            "y_pred_min": float(g["y_pred"].min()), "y_pred_max": float(g["y_pred"].max()),
            "median_abs_dev_from_band_midpoint": median_dev,
            "flag_pred_outside_range": low_flag, "flag_median_deviation_large": dev_flag,
            "flagged": low_flag or dev_flag,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- molecule level
def molecule_tables(df):
    """Rep-level (seed mean within a repetition) and molecule-level (mean over repetitions)."""
    rep_level = df.groupby(["isoform", "Molecule_Name", "repetition"], sort=True).agg(
        signed=("signed", "mean"), abs_error=("abs_error", "mean"), soft=("soft", "mean"),
        pred_mean=("y_pred", "mean"), pred_sd_across_seeds=("y_pred", "std"),
        n_rows=("y_pred", "size"), y_true=("y_true", "first"), cluster_id=("cluster_id", "first"),
    ).reset_index()
    mol = rep_level.groupby(["isoform", "Molecule_Name"], sort=True).agg(
        n_repetitions=("repetition", "nunique"), n_prediction_rows=("n_rows", "sum"),
        mean_signed=("signed", "mean"), mean_abs=("abs_error", "mean"), mean_soft=("soft", "mean"),
        sd_across_seeds_mean=("pred_sd_across_seeds", "mean"),
        sd_across_repetitions=("pred_mean", "std"),
        y_true=("y_true", "first"), cluster_id=("cluster_id", "first"),
    ).reset_index()
    mol["abs_of_mean_signed"] = mol["mean_signed"].abs()
    return rep_level, mol


def isoform_summary(df, mol, cm):
    rows = []
    for isoform in ISOFORMS:
        g = df[df["isoform"] == isoform]
        m = mol[mol["isoform"] == isoform]
        c = cm[cm["isoform"] == isoform]
        if g.empty:
            continue
        per_molecule_rows = m["n_prediction_rows"]
        rows.append({
            "isoform": isoform,
            "n_cells": int(g["cell_index"].nunique()),
            "n_prediction_rows": int(len(g)),
            "n_unique_molecules": int(len(m)),
            "n_unique_clusters": int(m["cluster_id"].nunique()),
            "rows_per_molecule_mean": float(per_molecule_rows.mean()),
            "rows_per_molecule_min": int(per_molecule_rows.min()),
            "rows_per_molecule_max": int(per_molecule_rows.max()),
            "row_mean_signed": float(g["signed"].mean()),
            "row_mean_abs": float(g["abs_error"].mean()),
            "row_mean_soft": float(g["soft"].mean()),
            "row_share_inside_band": float(g["inside_band"].mean()),
            "row_share_overpredicted": float((g["signed"] > 0).mean()),
            "row_share_underpredicted": float((g["signed"] < 0).mean()),
            "molecule_mean_signed": float(m["mean_signed"].mean()),
            "molecule_sd_of_mean_signed": float(m["mean_signed"].std()),
            "molecule_mean_abs": float(m["mean_abs"].mean()),
            "molecule_mean_abs_of_mean_signed": float(m["abs_of_mean_signed"].mean()),
            "molecule_mean_soft": float(m["mean_soft"].mean()),
            "molecule_share_overpredicted": float((m["mean_signed"] > 0).mean()),
            "molecule_share_underpredicted": float((m["mean_signed"] < 0).mean()),
            "molecule_share_zero_soft_error": float((m["mean_soft"] == 0).mean()),
            "cell_st_rae_mean": float(c["st_rae_recomputed"].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- clusters
def cluster_tables(mol, cluster_sizes):
    summary_parts, cumulative_parts, concentration_rows = [], [], []
    for isoform in ISOFORMS:
        m = mol[mol["isoform"] == isoform]
        if m.empty:
            continue
        n_molecules = len(m)
        total = {"soft": float(m["mean_soft"].sum()), "abs": float(m["mean_abs"].sum())}
        g = m.groupby("cluster_id").agg(
            n_unique_molecules=("Molecule_Name", "size"),
            n_prediction_rows=("n_prediction_rows", "sum"),
            mean_signed_error=("mean_signed", "mean"),
            mean_abs_of_molecule_signed=("abs_of_mean_signed", "mean"),
            mean_abs_error=("mean_abs", "mean"),
            mean_soft_error=("mean_soft", "mean"),
            sum_abs_error=("mean_abs", "sum"),
            sum_soft_error=("mean_soft", "sum"),
            n_molecules_over=("mean_signed", lambda s: int((s > 0).sum())),
            n_molecules_under=("mean_signed", lambda s: int((s < 0).sum())),
            mean_y_true=("y_true", "mean"),
        ).reset_index()
        g.insert(0, "isoform", isoform)
        g["cluster_size_all_splits"] = g["cluster_id"].map(cluster_sizes)
        g["share_of_molecules"] = g["n_unique_molecules"] / n_molecules
        for kind, column in (("soft", "sum_soft_error"), ("abs", "sum_abs_error")):
            g[f"share_of_{kind}_error"] = g[column] / total[kind] if total[kind] > 0 else np.nan
            g[f"concentration_ratio_{kind}"] = g[f"share_of_{kind}_error"] / g["share_of_molecules"]
        g["rank_by_soft_error"] = g["sum_soft_error"].rank(ascending=False, method="first").astype(int)
        g = g.sort_values("rank_by_soft_error").reset_index(drop=True)
        summary_parts.append(g)

        sizes_one = g["n_unique_molecules"] == 1
        for kind, column in (("soft", "sum_soft_error"), ("abs", "sum_abs_error")):
            ordered = g.sort_values(column, ascending=False, kind="mergesort").reset_index(drop=True)
            if total[kind] > 0:
                cum_error = ordered[column].cumsum() / total[kind]
            else:
                cum_error = pd.Series(np.nan, index=ordered.index)
            cum_molecules = ordered["n_unique_molecules"].cumsum() / n_molecules
            cumulative_parts.append(pd.DataFrame({
                "isoform": isoform, "error_measure": kind, "rank": np.arange(1, len(ordered) + 1),
                "cluster_id": ordered["cluster_id"], "cum_error_share": cum_error,
                "cum_molecule_share": cum_molecules,
            }))

            def add(statistic, value):
                concentration_rows.append(
                    {"isoform": isoform, "error_measure": kind, "statistic": statistic, "value": value}
                )

            add("n_clusters", len(ordered))
            add("n_unique_molecules", n_molecules)
            for k in TOP_K:
                top = ordered.head(k)
                add(f"top_{k}_clusters_error_share", float(top[column].sum() / total[kind]) if total[kind] > 0 else np.nan)
                add(f"top_{k}_clusters_molecule_share", float(top["n_unique_molecules"].sum() / n_molecules))
            for target in ERROR_SHARE_TARGETS:
                if total[kind] > 0:
                    needed = int(np.searchsorted(cum_error.to_numpy(), target - 1e-12) + 1)
                    needed = min(needed, len(ordered))
                    add(f"n_clusters_for_{int(target * 100)}pct_error", needed)
                    add(f"molecule_share_in_clusters_for_{int(target * 100)}pct_error",
                        float(ordered["n_unique_molecules"].head(needed).sum() / n_molecules))
            add("share_of_clusters_with_one_molecule", float(sizes_one.mean()))
            add("error_share_in_one_molecule_clusters",
                float(g.loc[sizes_one, column].sum() / total[kind]) if total[kind] > 0 else np.nan)
            per_molecule = np.sort(m["mean_soft" if kind == "soft" else "mean_abs"].to_numpy())[::-1]
            if total[kind] > 0:
                for fraction in MOLECULE_TOP_FRACTIONS:
                    count = max(1, int(round(fraction * n_molecules)))
                    add(f"top_{int(fraction * 100)}pct_molecules_error_share", float(per_molecule[:count].sum() / total[kind]))
    return (
        pd.concat(summary_parts, ignore_index=True),
        pd.concat(cumulative_parts, ignore_index=True),
        pd.DataFrame(concentration_rows),
    )


# ------------------------------------------------------------------- variation
def variation_table(df, cm):
    df = df.assign(repetition_x_fold=df["repetition"].astype(int).astype(str) + "x" + df["fold"].astype(int).astype(str))
    cm = cm.assign(repetition_x_fold=cm["repetition"].astype(int).astype(str) + "x" + cm["fold"].astype(int).astype(str))
    rows = []
    for grouping in ("repetition", "fold", "seed", "repetition_x_fold"):
        for (isoform, value), g in df.groupby(["isoform", grouping], sort=True):
            c = cm[(cm["isoform"] == isoform) & (cm[grouping] == value)]
            rows.append({
                "isoform": isoform, "grouping": grouping, "group_value": value,
                "n_cells": int(c["cell_index"].nunique()), "n_prediction_rows": int(len(g)),
                "n_unique_molecules": int(g["Molecule_Name"].nunique()),
                "mean_signed": float(g["signed"].mean()), "mean_abs": float(g["abs_error"].mean()),
                "mean_soft": float(g["soft"].mean()), "share_inside_band": float(g["inside_band"].mean()),
                "cell_st_rae_mean": float(c["st_rae_recomputed"].mean()),
                "cell_st_rae_min": float(c["st_rae_recomputed"].min()),
                "cell_st_rae_max": float(c["st_rae_recomputed"].max()),
                "cell_st_rae_sd": float(c["st_rae_recomputed"].std()),
            })
    return pd.DataFrame(rows)


def spread_table(df, rep_level, mol):
    rows = []
    for isoform in ISOFORMS:
        reference = df.loc[df["isoform"] == isoform, "abs_error"]
        if reference.empty:
            continue
        sources = (
            ("sd_across_seeds_within_repetition", rep_level.loc[rep_level["isoform"] == isoform, "pred_sd_across_seeds"]),
            ("sd_across_repetitions_of_seed_mean", mol.loc[mol["isoform"] == isoform, "sd_across_repetitions"]),
        )
        for source, values in sources:
            values = values.dropna().to_numpy()
            rows.append({
                "isoform": isoform, "source": source, "n_values": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "p90": float(np.percentile(values, 90)) if len(values) else np.nan,
                "reference_row_mean_abs_error": float(reference.mean()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- run
def write_csv(frame, path):
    frame.to_csv(path, index=False)


def run(args):
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {output_dir}")
    cells = list(range(N_CELLS)) if args.cells is None else sorted(set(int(c) for c in args.cells))
    if not cells or min(cells) < 0 or max(cells) >= N_CELLS:
        raise ValueError(f"Cells must lie in 0..{N_CELLS - 1}.")
    partial = len(cells) != N_CELLS
    sums = read_sums(args.sha256sums) if args.sha256sums else None

    df, prediction_hashes = load_predictions(args.predictions_dir, cells, sums)
    clusters = load_clusters(args.cluster_assignments)
    manifests, manifest_hashes = load_manifests(df, args.split_manifest_dir)
    df, warnings = attach_and_validate(df, clusters, manifests)
    df = add_errors(df)

    cm = cell_metrics(df, args.reproduction_dir)
    checks = range_check(df)
    rep_level, mol = molecule_tables(df)
    summary = isoform_summary(df, mol, cm)
    cluster_sizes = clusters["cluster_id"].value_counts()
    cluster_summary, cumulative, concentration = cluster_tables(mol, cluster_sizes)
    variation = variation_table(df, cm)
    spread = spread_table(df, rep_level, mol)

    output_dir.mkdir(parents=True)
    write_csv(summary, output_dir / "isoform_summary.csv")
    write_csv(cluster_summary, output_dir / "cluster_summary.csv")
    write_csv(concentration, output_dir / "cluster_concentration.csv")
    write_csv(cumulative, output_dir / "cluster_cumulative_share.csv")
    write_csv(mol, output_dir / "molecule_level.csv")
    write_csv(cm, output_dir / "cell_metrics.csv")
    write_csv(variation, output_dir / "variation_by_group.csv")
    write_csv(spread, output_dir / "prediction_spread.csv")
    write_csv(checks, output_dir / "scale_range_check.csv")

    recorded = cm["abs_diff_vs_recorded_original"].dropna()
    metadata = {
        "mode": "partial" if partial else "full",
        "cells": cells,
        "n_cells": len(cells),
        "inputs": {
            "predictions_dir": str(args.predictions_dir),
            "cluster_assignments": str(args.cluster_assignments),
            "cluster_assignments_sha256": sha256_file(args.cluster_assignments),
            "split_manifest_dir": str(args.split_manifest_dir),
            "sha256sums_used": bool(args.sha256sums),
            "reproduction_dir": None if args.reproduction_dir is None else str(args.reproduction_dir),
            "prediction_file_sha256": prediction_hashes,
            "split_manifest_sha256": manifest_hashes,
        },
        "script_sha256": sha256_file(__file__),
        "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
        "range_check": {
            "n_cell_isoform_checks": int(len(checks)),
            "n_flagged": int(checks["flagged"].sum()),
            "thresholds": {
                "pred_range": list(PRED_RANGE),
                "median_abs_deviation_from_band_midpoint": MEDIAN_BAND_DEVIATION_LIMIT,
            },
            "meaning": "anomaly flags only; cannot prove the historical scaling branch",
        },
        "st_rae_recomputed_vs_recorded": {
            "n_compared": int(len(recorded)),
            "max_abs_diff": float(recorded.max()) if len(recorded) else None,
            "meaning": "integrity cross-check only; matching scores cannot prove the scaling branch",
        },
        "warnings": warnings,
        "notes": NOTES,
        "outputs": sorted(p.name for p in output_dir.iterdir()) + ["run_metadata.json"],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"mode: {metadata['mode']}  cells: {len(cells)}  output: {output_dir}")
    print(summary[[
        "isoform", "n_prediction_rows", "n_unique_molecules", "n_unique_clusters",
        "row_mean_signed", "molecule_mean_signed", "row_mean_abs", "row_mean_soft", "row_share_inside_band",
    ]].round(4).to_string(index=False))
    flagged = metadata["range_check"]["n_flagged"]
    print(f"range-check anomaly flags: {flagged} of {len(checks)} (flags only; scale provenance remains unresolved)")
    if len(recorded):
        print(f"recomputed vs recorded cell ST-RAE: max abs diff {metadata['st_rae_recomputed_vs_recorded']['max_abs_diff']:.3e} (integrity check only)")
    for warning in warnings:
        print("warning:", warning)
    return {
        "summary": summary, "cluster_summary": cluster_summary, "concentration": concentration,
        "cumulative": cumulative, "molecule": mol, "cell_metrics": cm, "variation": variation,
        "spread": spread, "range_check": checks, "metadata": metadata,
    }


# ----------------------------------------------------------------------- selftest
SELF_SEEDS = (20261101, 20261102, 20261103)
SELF_OFFSETS = (0.0, 0.05, -0.05)
SELF_TRUE = {"M1": 4.0, "M2": 5.0, "M3": 6.0, "M4": 7.0, "M5": 5.5, "M6": 4.5}
SELF_DELTA = {"M1": 1.0, "M2": -0.5, "M3": 0.1, "M4": -0.3, "M5": 0.0, "M6": 0.7}
SELF_CLUSTER = {"M1": 10, "M2": 10, "M3": 20, "M4": 30, "M5": 30, "M6": 30}
SELF_HALF_BAND = 0.2


def _self_fixture(root, mutate=None):
    """Two repetitions x three seeds. M6 is validated only in repetition 1 (rows != molecules)."""
    root = Path(root)
    predictions = root / "predictions"
    manifests = root / "manifests"
    predictions.mkdir(parents=True)
    manifests.mkdir(parents=True)
    rows_by_cell = {}
    for repetition in (1, 2):
        for seed_index, seed in enumerate(SELF_SEEDS):
            cell = (repetition - 1) * 3 + seed_index
            rows = []
            for molecule in ("M1", "M2", "M3", "M4", "M5", "M6"):
                if repetition == 2 and molecule == "M6":
                    continue
                y = SELF_TRUE[molecule]
                rows.append({
                    "trial_index": TRIAL_INDEX, "cell_index": cell, "repetition": repetition, "fold": 1,
                    "seed": seed, "isoform": "CYP1A2", "Molecule_Name": molecule, "y_true": y,
                    "y_pred": y + SELF_DELTA[molecule] + SELF_OFFSETS[seed_index],
                    "conf_low": y - SELF_HALF_BAND, "conf_high": y + SELF_HALF_BAND,
                })
            for molecule, delta in (("M1", -0.5), ("M4", 0.5)):
                y = SELF_TRUE[molecule]
                rows.append({
                    "trial_index": TRIAL_INDEX, "cell_index": cell, "repetition": repetition, "fold": 1,
                    "seed": seed, "isoform": "CYP2D6", "Molecule_Name": molecule, "y_true": y,
                    "y_pred": y + delta + SELF_OFFSETS[seed_index],
                    "conf_low": y - SELF_HALF_BAND, "conf_high": y + SELF_HALF_BAND,
                })
            rows_by_cell[cell] = rows
    if mutate is not None:
        mutate(rows_by_cell)
    for cell, rows in rows_by_cell.items():
        # reversed order: results must not depend on row order
        pd.DataFrame(rows[::-1], columns=COLUMNS).to_csv(predictions / f"predictions_cell_{cell:03d}.csv", index=False)
    names = {1: sorted(SELF_TRUE), 2: sorted(SELF_TRUE)[:5]}
    for repetition in (1, 2):
        (manifests / f"rep_{repetition:02d}_fold_01.json").write_text(json.dumps({
            "repetition": repetition, "fold": 1, "validation_names": names[repetition],
            "validation_cluster_ids": [10, 20, 30], "training_seeds": list(SELF_SEEDS),
        }))
    cluster_rows = [
        {"row_index": i, "Molecule_Name": m, "cluster_id": SELF_CLUSTER[m], "split": "train"}
        for i, m in enumerate(sorted(SELF_TRUE))
    ]
    cluster_rows.append({"row_index": 6, "Molecule_Name": "P1", "cluster_id": 99, "split": "test"})
    clusters = root / "cluster_assignments.csv"
    pd.DataFrame(cluster_rows).to_csv(clusters, index=False)
    return predictions, manifests, clusters


def _args(predictions, clusters, manifests, output, cells=None, sums=None, reproduction=None):
    return argparse.Namespace(
        predictions_dir=predictions, cluster_assignments=clusters, split_manifest_dir=manifests,
        output_dir=output, cells=cells, sha256sums=sums, reproduction_dir=reproduction,
    )


def _close(a, b, tolerance=1e-9):
    return abs(float(a) - float(b)) <= tolerance


def selftest():
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        predictions, manifests, clusters = _self_fixture(tmp / "fixture")
        out = tmp / "out"
        cells = list(range(6))
        result = run(_args(predictions, clusters, manifests, out, cells=cells))

        # counts: 6 cells; CYP1A2 has 6 molecules in rep 1 and 5 in rep 2
        summary = result["summary"].set_index("isoform")
        one = summary.loc["CYP1A2"]
        check(one["n_prediction_rows"] == 33, f"CYP1A2 rows {one['n_prediction_rows']} != 33")
        check(one["n_unique_molecules"] == 6, "CYP1A2 unique molecules != 6")
        check(one["n_unique_clusters"] == 3, "CYP1A2 unique clusters != 3")
        check(summary.loc["CYP2D6"]["n_unique_molecules"] == 2, "CYP2D6 unique molecules != 2")
        check(summary.loc["CYP2D6"]["n_prediction_rows"] == 12, "CYP2D6 rows != 12")

        # row-level vs molecule-level means differ because M6 appears in one repetition only
        expected_row_signed = (3 * sum(SELF_DELTA.values()) + 3 * (sum(SELF_DELTA.values()) - SELF_DELTA["M6"])) / 33
        check(_close(one["row_mean_signed"], expected_row_signed), "CYP1A2 row-level mean signed error is wrong")
        check(_close(one["molecule_mean_signed"], sum(SELF_DELTA.values()) / 6), "CYP1A2 molecule-level mean signed error is wrong")
        check(not _close(one["row_mean_signed"], one["molecule_mean_signed"], 1e-6), "row-level and molecule-level means should differ in this fixture")

        # independent per-molecule soft-error means (offsets average to zero)
        expected_soft = {"M1": 0.8, "M2": 0.3, "M3": 0.0, "M4": 0.1, "M5": 0.0, "M6": 0.5}
        mol = result["molecule"]
        for isoform_name, molecule, expected in [("CYP1A2", k, v) for k, v in expected_soft.items()]:
            value = mol[(mol["isoform"] == isoform_name) & (mol["Molecule_Name"] == molecule)]["mean_soft"].iloc[0]
            check(_close(value, expected), f"{molecule} mean soft error {value} != {expected}")
        m1 = mol[(mol["isoform"] == "CYP1A2") & (mol["Molecule_Name"] == "M1")].iloc[0]
        check(m1["n_repetitions"] == 2 and m1["n_prediction_rows"] == 6, "M1 repetition/row counts are wrong")
        m6 = mol[(mol["isoform"] == "CYP1A2") & (mol["Molecule_Name"] == "M6")].iloc[0]
        check(m6["n_repetitions"] == 1 and m6["n_prediction_rows"] == 3, "M6 repetition/row counts are wrong")
        m2 = mol[(mol["isoform"] == "CYP1A2") & (mol["Molecule_Name"] == "M2")].iloc[0]
        check(_close(m2["mean_signed"], -0.5) and _close(m2["abs_of_mean_signed"], 0.5), "M2 signed/abs-of-signed are wrong")
        check(_close(m2["sd_across_seeds_mean"], 0.05), "M2 seed SD should be 0.05")
        check(_close(m2["sd_across_repetitions"], 0.0), "M2 repetition SD should be 0")

        # cluster concentration
        clusters_table = result["cluster_summary"]
        a = clusters_table[(clusters_table["isoform"] == "CYP1A2") & (clusters_table["cluster_id"] == 10)].iloc[0]
        b = clusters_table[(clusters_table["isoform"] == "CYP1A2") & (clusters_table["cluster_id"] == 20)].iloc[0]
        c = clusters_table[(clusters_table["isoform"] == "CYP1A2") & (clusters_table["cluster_id"] == 30)].iloc[0]
        check(a["n_unique_molecules"] == 2 and a["n_prediction_rows"] == 12, "cluster 10 counts are wrong")
        check(c["n_unique_molecules"] == 3 and c["n_prediction_rows"] == 15, "cluster 30 counts are wrong")
        check(_close(a["share_of_soft_error"], 1.1 / 1.7), "cluster 10 soft-error share is wrong")
        check(_close(b["share_of_soft_error"], 0.0), "cluster 20 soft-error share should be 0")
        check(_close(c["share_of_soft_error"], 0.6 / 1.7), "cluster 30 soft-error share is wrong")
        check(_close(a["share_of_molecules"], 2 / 6), "cluster 10 molecule share is wrong")
        check(_close(clusters_table[clusters_table["isoform"] == "CYP1A2"]["share_of_soft_error"].sum(), 1.0), "shares do not sum to 1")
        check(int(a["rank_by_soft_error"]) == 1, "cluster 10 should rank first")
        check(a["n_molecules_over"] == 1 and a["n_molecules_under"] == 1, "cluster 10 over/under counts are wrong")
        stats = result["concentration"]
        top1 = stats[(stats["isoform"] == "CYP1A2") & (stats["error_measure"] == "soft") & (stats["statistic"] == "top_1_clusters_error_share")]["value"].iloc[0]
        check(_close(top1, 1.1 / 1.7), "top-1 cluster error share is wrong")
        needed = stats[(stats["isoform"] == "CYP1A2") & (stats["error_measure"] == "soft") & (stats["statistic"] == "n_clusters_for_50pct_error")]["value"].iloc[0]
        check(int(needed) == 1, "one cluster should cover 50% of CYP1A2 soft error")
        d2 = clusters_table[clusters_table["isoform"] == "CYP2D6"]
        check(all(_close(v, 0.5) for v in d2["share_of_soft_error"]), "CYP2D6 shares should be 0.5 each")

        # independent per-cell ST-RAE for cell 0, CYP1A2 (offset 0)
        y = np.array([SELF_TRUE[m] for m in sorted(SELF_TRUE)])
        p = np.array([SELF_TRUE[m] + SELF_DELTA[m] for m in sorted(SELF_TRUE)])
        lo, hi = y - SELF_HALF_BAND, y + SELF_HALF_BAND
        numerator = np.maximum(p - hi, 0).sum() + np.maximum(lo - p, 0).sum()
        mean_y = y.mean()
        denominator = np.maximum(mean_y - hi, 0).sum() + np.maximum(lo - mean_y, 0).sum()
        cm = result["cell_metrics"]
        got = cm[(cm["cell_index"] == 0) & (cm["isoform"] == "CYP1A2")]["st_rae_recomputed"].iloc[0]
        check(_close(got, numerator / denominator), f"cell 0 ST-RAE {got} != {numerator / denominator}")

        variation = result["variation"]
        rep_rows = variation[(variation["isoform"] == "CYP1A2") & (variation["grouping"] == "repetition")].set_index("group_value")
        check(int(rep_rows.loc[1]["n_prediction_rows"]) == 18 and int(rep_rows.loc[2]["n_prediction_rows"]) == 15, "per-repetition row counts are wrong")
        check(int(rep_rows.loc[2]["n_unique_molecules"]) == 5, "repetition 2 should have 5 unique molecules")
        check(int(result["range_check"]["flagged"].sum()) == 0, "range check flagged a clean fixture")
        meta = result["metadata"]
        check(meta["mode"] == "partial" and meta["n_cells"] == 6, "a 6-cell run must be labelled partial")
        for name in ("isoform_summary.csv", "cluster_summary.csv", "cluster_concentration.csv", "cluster_cumulative_share.csv",
                     "molecule_level.csv", "cell_metrics.csv", "variation_by_group.csv", "prediction_spread.csv",
                     "scale_range_check.csv", "run_metadata.json"):
            check((out / name).is_file(), f"missing output {name}")

        # refusals
        def expect(error, label, build):
            try:
                build()
            except error:
                return
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{label}: raised {type(exc).__name__}, expected {error.__name__}: {exc}")
                return
            failures.append(f"{label}: did not raise {error.__name__}")

        expect(FileExistsError, "reuse of an output directory",
               lambda: run(_args(predictions, clusters, manifests, out, cells=cells)))
        expect(FileNotFoundError, "missing prediction file",
               lambda: run(_args(predictions, clusters, manifests, tmp / "o_missing", cells=cells + [99])))
        sums = tmp / "sums.txt"
        good = {f"predictions_cell_{i:03d}.csv": sha256_file(predictions / f"predictions_cell_{i:03d}.csv") for i in cells}
        sums.write_text("".join(f"{h}  {n}\n" for n, h in good.items()))
        run(_args(predictions, clusters, manifests, tmp / "o_sums", cells=cells, sums=sums))
        bad = dict(good, **{"predictions_cell_000.csv": "0" * 64})
        sums.write_text("".join(f"{h}  {n}\n" for n, h in bad.items()))
        expect(ValueError, "SHA-256 mismatch",
               lambda: run(_args(predictions, clusters, manifests, tmp / "o_sha", cells=cells, sums=sums)))

        def with_extra(label, extra_rows_builder, error):
            fixture = tmp / f"fx_{label}"

            def mutate(rows_by_cell):
                rows_by_cell[0].extend(extra_rows_builder(rows_by_cell[0][0]))

            p2, m2_, c2 = _self_fixture(fixture, mutate=mutate)
            expect(error, label, lambda: run(_args(p2, c2, m2_, tmp / f"o_{label}", cells=cells)))

        with_extra("unknown_molecule", lambda r: [dict(r, Molecule_Name="ZZ", isoform="CYP2C9")], ValueError)
        with_extra("protected_molecule", lambda r: [dict(r, Molecule_Name="P1", isoform="CYP2C9")], ValueError)
        with_extra("duplicate_row", lambda r: [dict(r)], ValueError)

    if failures:
        print("SELFTEST FAILED:")
        for message in failures:
            print(" -", message)
        return 1
    print("SELFTEST PASSED: counts, row vs molecule means, cluster shares, ST-RAE, variation and refusals all match the independent expectations.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="synthetic-data check; reads and writes nothing in the repository")
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--cluster-assignments", type=Path)
    parser.add_argument("--split-manifest-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="must not exist")
    parser.add_argument("--cells", type=int, nargs="+", help="cell indices; omit for all 150")
    parser.add_argument("--sha256sums", type=Path, help="optional sha256sum file for the prediction files")
    parser.add_argument("--reproduction-dir", type=Path, help="optional dir with reproduction_cell_NNN.json for the ST-RAE cross-check")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    required = ("predictions_dir", "cluster_assignments", "split_manifest_dir", "output_dir")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing: " + ", ".join("--" + n.replace("_", "-") for n in missing))
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
