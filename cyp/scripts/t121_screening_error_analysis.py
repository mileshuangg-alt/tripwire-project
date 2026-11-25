"""Does a weaker screening response line up with over-prediction? (read-only diagnostic)

Joins the existing single-concentration screening log2fc values to the saved validation
predictions of D022 arm R replicate 1 (150 cells) by verified compound identity and relates the
screening response to the signed error (predicted - measured pIC50) at molecule level. It reuses
the loading, validation and error machinery of t121_error_diagnostic.py. It trains nothing, runs no
inference, loads no checkpoint, opens no protected-test data and fits no rule.

Documented facts (sources) and what is inference
  * Join key: Molecule_Name (OCNT-...). It is the key of the screening file, of the training CSV
    and of the saved predictions. Identity is verified against inputs/adjudication.tsv (class
    full_key_match or largest_fragment_match, tier train or validation), which the frozen loader
    also uses (cyp/scripts/actfound_model_side.py, identity adjudication block, ~lines 980-996),
    and the SMILES of the two files are compared and reported.
  * The frozen screening loader SKIPS every identity-linked compound, so the screening stage never
    trained on these validation compounds' screening values (same file; notes/Handoffs/
    Tripwire_CYP_007_Handoff_20261119.md line 88).
  * log2fc_estimate, log2fc_std_error, p_value, log2fc_fdr, log2fc_median, cohens_d, plate_id and
    concentration_M are columns of inputs/screen_full.csv (one concentration value in the file,
    4.95049505e-05 M; the public documentation does not state the assay concentration).
  * Sign: the OpenADMET challenge announcement states that a negative log-fold change indicates
    that a compound inhibits the CYP (https://openadmet.ghost.io/announcing-openadmets-cyp-
    inhibition-blind-challenge/). It describes fluorescence of a probe metabolite for CYP1A2,
    CYP2C9 and CYP3A4 and mass-spectrometry parent depletion (dextromethorphan) for CYP2D6, and a
    per-plate linear model contrasting each compound with the negative control. The exact
    numerator and denominator of the fold change, its normalization beyond that, and the
    concentration are NOT stated. The dataset card (huggingface.co/datasets/openadmet/
    cyp-challenge-train-test) defines none of these. For CYP2D6 the file-level direction is
    therefore documented only through the general statement; Spearman(log2fc, measured pIC50)
    is reported as a coherence check (inference, not a definition) and CYP2D6 interpretation
    is held until it is reviewed.
  * "Weaker response" is not given a cutoff and is not equated with confirmed inactivity. Outputs
    keep the raw log2fc sign. Binned summaries use fixed equal-count quintiles and a fixed top-decile
    tail, both labelled exploratory; nothing is tuned to the residuals.

Limits (also in run_metadata.json): descriptive only, no significance claim; each molecule has 15
predictions (5 repetitions x 3 seeds) and counts as ONE compound; measured pIC50 enters the signed
error, so associations with the screening response are not a causal statement; this does not
authorize selecting or fitting any penalty on validation labels, and any future penalty must use
only information permitted inside its training fold.

Usage:
    python -I cyp/scripts/t121_screening_error_analysis.py --selftest
    python -I cyp/scripts/t121_screening_error_analysis.py \
        --predictions-dir DIR --cluster-assignments CSV --split-manifest-dir DIR \
        --screen-tar TAR --train-csv CSV --isoforms CYP1A2 CYP2D6 --output-dir NEW_DIR \
        [--cells I J K ...] [--sha256sums FILE]
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import tarfile
import tempfile

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` omits the script dir
import t121_error_diagnostic as diag  # noqa: E402

ISOFORMS = diag.ISOFORMS
SCREEN_MEMBER = "inputs/screen_full.csv"
IDENTITY_MEMBER = "inputs/adjudication.tsv"
CONTRACT_MEMBER = "qualification/actfound/screen_loader_20261117/contract.json"
SCREEN_COLUMNS = [
    "Molecule_Name", "SMILES", "OCNT_Batch", "enzyme", "plate_id", "concentration_M",
    "log2fc_estimate", "log2fc_std_error", "p_value", "log2fc_fdr", "log2fc_median", "cohens_d",
]
IDENTITY_COLUMNS = ["molecule_name", "smiles", "class", "tiers"]
IDENTITY_OK = ("full_key_match", "largest_fragment_match")
PROTECTED_TIERS = ("internal_test", "challenge_blinded_test")
SCREEN_VALUE = "log2fc_estimate"
TAIL_FRACTION = 0.10
TAIL_SIGNED_THRESHOLD = 1.0
N_BINS = 5
MIN_N = 3
MIN_BIN_N = 10
OUTCOMES = (
    ("measured_pIC50_context", "y_true"),
    ("predicted_pIC50_context", "pred_mean"),
    ("signed_error", "mean_signed"),
    ("absolute_error", "mean_abs"),
    ("soft_error", "mean_soft"),
)
NOTES = [
    "Join key is Molecule_Name; identity verified through adjudication.tsv (identity-linked, tier train or validation) and SMILES comparison; counts reported.",
    "The frozen screening loader skips identity-linked compounds, so these screening values were not used to train the screening stage.",
    "Documented direction: a negative log-fold change indicates inhibition (OpenADMET challenge announcement). Numerator, denominator and assay concentration are not stated; CYP2D6 uses mass-spectrometry parent depletion, so its file-level direction is documented only generally.",
    "Raw log2fc sign is kept; no weak/inactive cutoff is defined; weak response is not confirmed inactivity.",
    "Each molecule counts once (15 predictions = 5 repetitions x 3 seeds); no significance claim; descriptive only.",
    "Signed error contains the measured value; signed and absolute error are reported separately.",
    "Bins (equal-count quintiles of log2fc or of predicted pIC50) and the tail (top 10% of molecule-level signed error; signed error above +1.0) are exploratory and fixed in advance.",
    "This analysis uses validation labels diagnostically; it does not select or fit a penalty. Scale provenance of the saved predictions remains unresolved.",
]


# --------------------------------------------------------------------------- inputs
def load_screening(tar_path):
    """Read the three members straight from the tarball (nothing is extracted) and verify hashes."""
    tar_path = Path(tar_path)
    if not tar_path.is_file():
        raise FileNotFoundError(f"Screening tarball missing: {tar_path}")
    raw = {}
    with tarfile.open(tar_path, "r:*") as tar:
        for name in (SCREEN_MEMBER, IDENTITY_MEMBER, CONTRACT_MEMBER):
            try:
                handle = tar.extractfile(name)
            except KeyError:
                handle = None
            if handle is None:
                raise FileNotFoundError(f"{name} is not in {tar_path}")
            raw[name] = handle.read()
    contract = json.loads(raw[CONTRACT_MEMBER])
    screen_sha = hashlib.sha256(raw[SCREEN_MEMBER]).hexdigest()
    identity_sha = hashlib.sha256(raw[IDENTITY_MEMBER]).hexdigest()
    if screen_sha != contract["sha256"]["screen"]:
        raise ValueError("screen_full.csv does not match the contract's SHA-256.")
    if identity_sha != contract["sha256"]["identity"]:
        raise ValueError("adjudication.tsv does not match the contract's SHA-256.")
    screen = pd.read_csv(io.BytesIO(raw[SCREEN_MEMBER]))
    identity = pd.read_csv(io.BytesIO(raw[IDENTITY_MEMBER]), sep="\t")
    for column in SCREEN_COLUMNS:
        if column not in screen.columns:
            raise ValueError(f"screen_full.csv lacks column {column}.")
    for column in IDENTITY_COLUMNS:
        if column not in identity.columns:
            raise ValueError(f"adjudication.tsv lacks column {column}.")
    if identity["molecule_name"].duplicated().any():
        raise ValueError("adjudication.tsv lists a molecule more than once.")
    hashes = {
        "screen_tar_sha256": diag.sha256_file(tar_path),
        "screen_full_sha256": screen_sha,
        "adjudication_sha256": identity_sha,
        "contract_sha256": hashlib.sha256(raw[CONTRACT_MEMBER]).hexdigest(),
    }
    return screen, identity, hashes


def load_train_smiles(train_csv, molecules):
    train = pd.read_csv(train_csv, usecols=["Molecule_Name", "SMILES"])
    train = train[train["Molecule_Name"].isin(set(molecules))]
    if train["Molecule_Name"].duplicated().any():
        raise ValueError("The training CSV lists a molecule more than once.")
    missing = set(molecules) - set(train["Molecule_Name"])
    if missing:
        raise ValueError(f"Molecules not in the training CSV (cannot verify SMILES): {sorted(missing)[:5]}")
    return dict(zip(train["Molecule_Name"], train["SMILES"]))


# ------------------------------------------------------------------------------ join
def build_join(molecules, isoforms, screen, identity, train_smiles):
    """One row per (isoform, molecule) with a status; nothing is dropped silently."""
    identity = identity.set_index("molecule_name")
    rows = []
    for isoform in isoforms:
        names = sorted(molecules[isoform])
        sub = screen[(screen["enzyme"] == isoform) & screen["Molecule_Name"].isin(set(names))]
        groups = {name: g for name, g in sub.groupby("Molecule_Name")}
        for name in names:
            g = groups.get(name)
            row = {
                "isoform": isoform, "Molecule_Name": name,
                "n_screening_rows": 0 if g is None else int(len(g)),
                "identity_class": None, "identity_tier": None, "smiles_exact_match": None,
                "log2fc_estimate": np.nan, "log2fc_std_error": np.nan, "p_value": np.nan,
                "log2fc_fdr": np.nan, "log2fc_median": np.nan, "cohens_d": np.nan,
                "plate_id": None, "concentration_M": np.nan,
                "log2fc_min_over_duplicates": np.nan, "log2fc_max_over_duplicates": np.nan,
            }
            if name in identity.index:
                row["identity_class"] = str(identity.loc[name, "class"])
                row["identity_tier"] = str(identity.loc[name, "tiers"])
            if g is None:
                row["status"] = "no_screening_row"
            elif len(g) > 1:
                row["status"] = "ambiguous_duplicate"
                row["log2fc_min_over_duplicates"] = float(g[SCREEN_VALUE].min())
                row["log2fc_max_over_duplicates"] = float(g[SCREEN_VALUE].max())
            else:
                record = g.iloc[0]
                for column in ("log2fc_estimate", "log2fc_std_error", "p_value", "log2fc_fdr",
                               "log2fc_median", "cohens_d", "concentration_M"):
                    row[column] = float(record[column]) if pd.notna(record[column]) else np.nan
                row["plate_id"] = str(record["plate_id"])
                row["smiles_exact_match"] = bool(record["SMILES"] == train_smiles[name])
                if row["identity_class"] not in IDENTITY_OK:
                    row["status"] = "identity_unverified"
                elif pd.isna(record[SCREEN_VALUE]):
                    row["status"] = "missing_value"
                else:
                    row["status"] = "ok"
            if row["identity_tier"] in PROTECTED_TIERS:
                raise ValueError(f"{name}: adjudicated as a protected tier ({row['identity_tier']}); stopping.")
            rows.append(row)
    return pd.DataFrame(rows)


def join_audit(join, isoforms):
    rows = []
    for isoform in isoforms:
        j = join[join["isoform"] == isoform]
        counts = j["status"].value_counts()
        ok = j[j["status"] == "ok"]
        rows.append({
            "isoform": isoform,
            "n_molecules_in_predictions": int(len(j)),
            "n_ok": int(counts.get("ok", 0)),
            "n_no_screening_row": int(counts.get("no_screening_row", 0)),
            "n_ambiguous_duplicate": int(counts.get("ambiguous_duplicate", 0)),
            "n_identity_unverified": int(counts.get("identity_unverified", 0)),
            "n_missing_value": int(counts.get("missing_value", 0)),
            "ok_smiles_exact_match": int((ok["smiles_exact_match"] == True).sum()),  # noqa: E712
            "ok_smiles_differ": int((ok["smiles_exact_match"] == False).sum()),  # noqa: E712
            "ok_identity_full_key_match": int((ok["identity_class"] == "full_key_match").sum()),
            "ok_identity_largest_fragment_match": int((ok["identity_class"] == "largest_fragment_match").sum()),
            "ok_distinct_concentration_M": ";".join(sorted({repr(v) for v in ok["concentration_M"].dropna()})),
            "ok_n_plates": int(ok["plate_id"].nunique()),
            "ok_log2fc_min": float(ok["log2fc_estimate"].min()) if len(ok) else np.nan,
            "ok_log2fc_median": float(ok["log2fc_estimate"].median()) if len(ok) else np.nan,
            "ok_log2fc_max": float(ok["log2fc_estimate"].max()) if len(ok) else np.nan,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ statistics
def _finite(*arrays):
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return [a[mask] for a in arrays]


def _ranks(a):
    return pd.Series(a).rank(method="average").to_numpy()


def pearson(x, y):
    x, y = _finite(x, y)
    if len(x) < MIN_N or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x, y = _finite(x, y)
    if len(x) < MIN_N:
        return float("nan")
    return pearson(_ranks(x), _ranks(y))


def partial_spearman(x, y, z):
    """Rank-based partial correlation of x and y given z (residuals of ranks on the rank of z)."""
    x, y, z = _finite(x, y, z)
    if len(x) < MIN_N + 1:
        return float("nan")
    rx, ry, rz = _ranks(x), _ranks(y), _ranks(z)
    if np.var(rz) == 0:
        return float("nan")

    def residual(r):
        slope = np.cov(r, rz, bias=True)[0, 1] / np.var(rz)
        return r - slope * rz

    return pearson(residual(rx), residual(ry))


def tail_auc(tail, rest):
    """P(a random tail value exceeds a random non-tail value); 0.5 means no difference. Ties count 1/2."""
    tail, rest = _finite(tail)[0], _finite(rest)[0]
    if len(tail) == 0 or len(rest) == 0:
        return float("nan")
    ranks = _ranks(np.concatenate([tail, rest]))
    n1, n2 = len(tail), len(rest)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def summarize(values):
    values = _finite(values)[0]
    if len(values) == 0:
        return {"mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)), "q75": float(np.percentile(values, 75)),
    }


# ------------------------------------------------------------------------ analysis
def molecule_frames(df, isoforms):
    sub = df[df["isoform"].isin(isoforms)]
    rep = sub.groupby(["isoform", "Molecule_Name", "repetition", "fold"], sort=True).agg(
        signed=("signed", "mean"), abs_error=("abs_error", "mean"), soft=("soft", "mean"),
        pred_mean=("y_pred", "mean"), y_true=("y_true", "first"), n_rows=("y_pred", "size"),
    ).reset_index()
    mol = rep.groupby(["isoform", "Molecule_Name"], sort=True).agg(
        n_repetitions=("repetition", "nunique"), n_prediction_rows=("n_rows", "sum"),
        mean_signed=("signed", "mean"), mean_abs=("abs_error", "mean"), mean_soft=("soft", "mean"),
        pred_mean=("pred_mean", "mean"), y_true=("y_true", "first"),
    ).reset_index()
    return rep, mol


def relationship_table(analysis):
    rows = []
    for isoform, a in analysis.groupby("isoform", sort=True):
        x = a[SCREEN_VALUE].to_numpy(dtype=np.float64)
        for outcome, column in OUTCOMES:
            y = a[column].to_numpy(dtype=np.float64)
            rows.append({
                "isoform": isoform, "outcome": outcome, "n_molecules": int(len(_finite(x, y)[0])),
                "spearman_with_log2fc": spearman(x, y), "pearson_with_log2fc": pearson(x, y),
            })
        rows.append({
            "isoform": isoform, "outcome": "signed_error_given_predicted_pIC50_partial",
            "n_molecules": int(len(_finite(x, a["mean_signed"], a["pred_mean"])[0])),
            "spearman_with_log2fc": partial_spearman(x, a["mean_signed"], a["pred_mean"]),
            "pearson_with_log2fc": float("nan"),
        })
    return pd.DataFrame(rows)


def quantile_bins(values, n_bins):
    return pd.qcut(pd.Series(values), n_bins, duplicates="drop")


def screening_bin_table(analysis):
    rows = []
    for isoform, a in analysis.groupby("isoform", sort=True):
        a = a.reset_index(drop=True)
        if len(a) < MIN_BIN_N:
            continue
        bins = quantile_bins(a[SCREEN_VALUE], N_BINS)
        for interval, g in a.groupby(bins, observed=True):
            rows.append({
                "isoform": isoform, "grouping": "log2fc_equal_count_quintile (exploratory)",
                "bin": str(interval), "n_molecules": int(len(g)),
                "log2fc_min": float(g[SCREEN_VALUE].min()), "log2fc_max": float(g[SCREEN_VALUE].max()),
                "mean_signed": float(g["mean_signed"].mean()), "median_signed": float(g["mean_signed"].median()),
                "share_overpredicted": float((g["mean_signed"] > 0).mean()),
                "share_signed_above_1": float((g["mean_signed"] > TAIL_SIGNED_THRESHOLD).mean()),
                "mean_abs_error": float(g["mean_abs"].mean()), "mean_soft_error": float(g["mean_soft"].mean()),
                "mean_measured_pIC50": float(g["y_true"].mean()), "mean_predicted_pIC50": float(g["pred_mean"].mean()),
            })
    return pd.DataFrame(rows)


def stratified_table(analysis):
    """Screening-error association inside equal-count strata of the model's own predicted pIC50."""
    rows = []
    for isoform, a in analysis.groupby("isoform", sort=True):
        a = a.reset_index(drop=True)
        if len(a) < MIN_BIN_N:
            continue
        bins = quantile_bins(a["pred_mean"], N_BINS)
        for interval, g in a.groupby(bins, observed=True):
            rows.append({
                "isoform": isoform, "grouping": "predicted_pIC50_equal_count_quintile (exploratory)",
                "bin": str(interval), "n_molecules": int(len(g)),
                "spearman_log2fc_vs_signed": spearman(g[SCREEN_VALUE], g["mean_signed"]),
                "spearman_log2fc_vs_measured": spearman(g[SCREEN_VALUE], g["y_true"]),
                "mean_signed": float(g["mean_signed"].mean()), "mean_measured_pIC50": float(g["y_true"].mean()),
            })
    return pd.DataFrame(rows)


def tail_table(analysis):
    rows = []
    for isoform, a in analysis.groupby("isoform", sort=True):
        a = a.sort_values("mean_signed", ascending=False, kind="mergesort").reset_index(drop=True)
        if a.empty:
            continue
        count = max(1, int(round(TAIL_FRACTION * len(a))))
        definitions = (
            (f"top_{int(TAIL_FRACTION * 100)}pct_of_signed_error (exploratory)", a.index < count),
            (f"signed_error_above_{TAIL_SIGNED_THRESHOLD} (exploratory)", (a["mean_signed"] > TAIL_SIGNED_THRESHOLD).to_numpy()),
        )
        for label, mask in definitions:
            tail, rest = a[mask], a[~mask]
            row = {
                "isoform": isoform, "tail_definition": label, "n_tail": int(len(tail)), "n_rest": int(len(rest)),
                "tail_mean_signed": float(tail["mean_signed"].mean()) if len(tail) else np.nan,
                "tail_mean_measured_pIC50": float(tail["y_true"].mean()) if len(tail) else np.nan,
                "rest_mean_measured_pIC50": float(rest["y_true"].mean()) if len(rest) else np.nan,
                "tail_mean_predicted_pIC50": float(tail["pred_mean"].mean()) if len(tail) else np.nan,
                "rest_mean_predicted_pIC50": float(rest["pred_mean"].mean()) if len(rest) else np.nan,
                "auc_tail_log2fc_exceeds_rest": tail_auc(tail[SCREEN_VALUE], rest[SCREEN_VALUE]),
            }
            for prefix, part in (("tail", tail), ("rest", rest)):
                for key, value in summarize(part[SCREEN_VALUE]).items():
                    row[f"{prefix}_log2fc_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def partition_table(rep, join):
    ok = join[join["status"] == "ok"][["isoform", "Molecule_Name", SCREEN_VALUE]]
    merged = rep.merge(ok, on=["isoform", "Molecule_Name"], how="inner")
    rows = []
    for (isoform, repetition, fold), g in merged.groupby(["isoform", "repetition", "fold"], sort=True):
        rows.append({
            "isoform": isoform, "repetition": int(repetition), "fold": int(fold), "n_molecules": int(len(g)),
            "mean_signed": float(g["signed"].mean()),
            "spearman_log2fc_vs_signed": spearman(g[SCREEN_VALUE], g["signed"]),
            "spearman_log2fc_vs_absolute": spearman(g[SCREEN_VALUE], g["abs_error"]),
            "spearman_log2fc_vs_measured": spearman(g[SCREEN_VALUE], g["y_true"]),
            "spearman_log2fc_vs_predicted": spearman(g[SCREEN_VALUE], g["pred_mean"]),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- run
def run(args):
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {output_dir}")
    isoforms = list(dict.fromkeys(args.isoforms))
    unknown = set(isoforms) - set(ISOFORMS)
    if not isoforms or unknown:
        raise ValueError(f"--isoforms must be a non-empty subset of {ISOFORMS}.")
    cells = list(range(diag.N_CELLS)) if args.cells is None else sorted(set(int(c) for c in args.cells))
    if not cells or min(cells) < 0 or max(cells) >= diag.N_CELLS:
        raise ValueError(f"Cells must lie in 0..{diag.N_CELLS - 1}.")
    partial = len(cells) != diag.N_CELLS
    sums = diag.read_sums(args.sha256sums) if args.sha256sums else None

    df, prediction_hashes = diag.load_predictions(args.predictions_dir, cells, sums)
    clusters = diag.load_clusters(args.cluster_assignments)
    manifests, manifest_hashes = diag.load_manifests(df, args.split_manifest_dir)
    df, warnings = diag.attach_and_validate(df, clusters, manifests)
    df = diag.add_errors(df)
    df = df[df["isoform"].isin(isoforms)].copy()
    if df.empty:
        raise ValueError("No prediction rows for the requested isoforms.")

    screen, identity, screen_hashes = load_screening(args.screen_tar)
    molecules = {iso: set(df.loc[df["isoform"] == iso, "Molecule_Name"]) for iso in isoforms}
    train_smiles = load_train_smiles(args.train_csv, set().union(*molecules.values()))
    join = build_join(molecules, isoforms, screen, identity, train_smiles)
    audit = join_audit(join, isoforms)

    rep, mol = molecule_frames(df, isoforms)
    merged = mol.merge(join, on=["isoform", "Molecule_Name"], how="left", validate="one_to_one")
    analysis = merged[merged["status"] == "ok"].copy()
    relationship = relationship_table(analysis)
    bins = screening_bin_table(analysis)
    strata = stratified_table(analysis)
    tails = tail_table(analysis)
    partitions = partition_table(rep, join)

    rows = df[[
        "cell_index", "repetition", "fold", "seed", "isoform", "Molecule_Name", "cluster_id",
        "y_true", "y_pred", "signed", "abs_error", "soft",
    ]].merge(
        join[["isoform", "Molecule_Name", "status", SCREEN_VALUE, "log2fc_std_error", "plate_id"]],
        on=["isoform", "Molecule_Name"], how="left", validate="many_to_one",
    )

    output_dir.mkdir(parents=True)
    audit.to_csv(output_dir / "screening_join_audit.csv", index=False)
    merged.to_csv(output_dir / "molecule_screening_error.csv", index=False)
    rows.to_csv(output_dir / "prediction_rows_with_screening.csv", index=False)
    relationship.to_csv(output_dir / "relationship_summary.csv", index=False)
    bins.to_csv(output_dir / "log2fc_quintile_summary.csv", index=False)
    strata.to_csv(output_dir / "prediction_strata_summary.csv", index=False)
    tails.to_csv(output_dir / "overprediction_tail_comparison.csv", index=False)
    partitions.to_csv(output_dir / "partition_relationship.csv", index=False)

    metadata = {
        "mode": "partial" if partial else "full",
        "isoforms": isoforms, "cells": cells, "n_cells": len(cells),
        "inputs": {
            "predictions_dir": str(args.predictions_dir),
            "cluster_assignments": str(args.cluster_assignments),
            "cluster_assignments_sha256": diag.sha256_file(args.cluster_assignments),
            "split_manifest_dir": str(args.split_manifest_dir),
            "train_csv": str(args.train_csv), "train_csv_sha256": diag.sha256_file(args.train_csv),
            "screen_tar": str(args.screen_tar), "sha256sums_used": bool(args.sha256sums),
            "prediction_file_sha256": prediction_hashes, "split_manifest_sha256": manifest_hashes,
            **screen_hashes,
        },
        "script_sha256": diag.sha256_file(__file__),
        "diagnostic_script_sha256": diag.sha256_file(diag.__file__),
        "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
        "constants": {
            "tail_fraction": TAIL_FRACTION, "tail_signed_threshold": TAIL_SIGNED_THRESHOLD,
            "n_bins": N_BINS, "min_n_correlation": MIN_N, "min_n_for_bins": MIN_BIN_N,
        },
        "bins_skipped_for_small_n": bool(any(len(a) < MIN_BIN_N for _, a in analysis.groupby("isoform"))) if len(analysis) else True,
        "warnings": warnings,
        "notes": NOTES,
        "outputs": sorted(p.name for p in output_dir.iterdir()) + ["run_metadata.json"],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"mode: {metadata['mode']}  isoforms: {isoforms}  cells: {len(cells)}  output: {output_dir}")
    print(audit[[
        "isoform", "n_molecules_in_predictions", "n_ok", "n_no_screening_row", "n_ambiguous_duplicate",
        "n_identity_unverified", "n_missing_value", "ok_smiles_exact_match", "ok_smiles_differ", "ok_distinct_concentration_M",
    ]].to_string(index=False))
    show = relationship[relationship["outcome"].isin([
        "measured_pIC50_context", "predicted_pIC50_context", "signed_error", "absolute_error",
        "signed_error_given_predicted_pIC50_partial",
    ])]
    print(show.round(4).to_string(index=False))
    if not tails.empty:
        print(tails[["isoform", "tail_definition", "n_tail", "n_rest", "auc_tail_log2fc_exceeds_rest"]].round(4).to_string(index=False))
    for warning in warnings:
        print("warning:", warning)
    return {
        "audit": audit, "analysis": analysis, "merged": merged, "relationship": relationship,
        "bins": bins, "strata": strata, "tails": tails, "partitions": partitions, "rows": rows,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- selftest
SELF_LOG2FC = {"M1": 0.6, "M3": 0.2, "M5": 0.0, "M4": -0.2, "M2": -0.4}


def _self_screening(root, classes=None, smiles_override=None, tamper=False, drop_member=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    classes = classes or {m: "full_key_match" for m in ("M1", "M2", "M3", "M4", "M5", "M6")}
    base_smiles = {m: f"C{'C' * i}O" for i, m in enumerate(("M1", "M2", "M3", "M4", "M5", "M6", "X1"))}
    screen_rows = []

    def add(name, enzyme, value, plate="P1"):
        smi = (smiles_override or {}).get(name, base_smiles[name])
        screen_rows.append({
            "Molecule_Name": name, "SMILES": smi, "OCNT_Batch": f"{name}-B", "enzyme": enzyme, "plate_id": plate,
            "concentration_M": 4.95049505e-05, "log2fc_estimate": value, "log2fc_std_error": 0.1,
            "p_value": 0.01, "log2fc_fdr": 0.02, "log2fc_median": value, "cohens_d": -1.0,
        })

    for name, value in SELF_LOG2FC.items():
        add(name, "CYP1A2", value)
    add("M6", "CYP1A2", 0.1)
    add("M6", "CYP1A2", 0.3, plate="P2")          # ambiguous duplicate
    add("X1", "CYP1A2", 0.0)                      # not in the predictions
    add("M1", "CYP2D6", -0.3)
    add("M4", "CYP2D6", np.nan)                   # missing value
    add("M1", "CYP2C9", 0.5)
    screen_csv = pd.DataFrame(screen_rows).to_csv(index=False).encode()
    identity_rows = [
        {"molecule_name": m, "smiles": base_smiles[m], "class": classes.get(m, "no_identity_link"), "tiers": "train"}
        for m in ("M1", "M2", "M3", "M4", "M5", "M6", "X1")
    ]
    identity_tsv = pd.DataFrame(identity_rows).to_csv(index=False, sep="\t").encode()
    contract = {"sha256": {
        "screen": hashlib.sha256(screen_csv).hexdigest(),
        "identity": hashlib.sha256(identity_tsv).hexdigest(), "similarity": "unused",
    }}
    payload = {SCREEN_MEMBER: screen_csv, IDENTITY_MEMBER: identity_tsv, CONTRACT_MEMBER: json.dumps(contract).encode()}
    if tamper:
        payload[SCREEN_MEMBER] = screen_csv + b"\n"
    if drop_member:
        payload.pop(drop_member)
    tar_path = root / "screen.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, data in payload.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    train = pd.DataFrame([{"Molecule_Name": m, "SMILES": base_smiles[m]} for m in ("M1", "M2", "M3", "M4", "M5", "M6")])
    train_csv = root / "train.csv"
    train.to_csv(train_csv, index=False)
    return tar_path, train_csv


def _close(a, b, tolerance=1e-9):
    return abs(float(a) - float(b)) <= tolerance


def selftest():
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    def expect(error, label, build):
        try:
            build()
        except error:
            return
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: raised {type(exc).__name__}, expected {error.__name__}: {exc}")
            return
        failures.append(f"{label}: did not raise {error.__name__}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        predictions, manifests, clusters = diag._self_fixture(tmp / "fixture")
        tar_path, train_csv = _self_screening(tmp / "screening")

        def args(out, tar=tar_path, cells=None, isoforms=("CYP1A2", "CYP2D6")):
            return argparse.Namespace(
                predictions_dir=predictions, cluster_assignments=clusters, split_manifest_dir=manifests,
                screen_tar=tar, train_csv=train_csv, isoforms=list(isoforms), output_dir=out,
                cells=list(range(6)) if cells is None else cells, sha256sums=None,
            )

        out = tmp / "out"
        result = run(args(out))
        audit = result["audit"].set_index("isoform")
        a = audit.loc["CYP1A2"]
        check(a["n_molecules_in_predictions"] == 6, "CYP1A2 molecules in predictions != 6")
        check(a["n_ok"] == 5 and a["n_ambiguous_duplicate"] == 1, "CYP1A2 ok/ambiguous counts are wrong")
        check(a["n_no_screening_row"] == 0 and a["n_identity_unverified"] == 0 and a["n_missing_value"] == 0, "CYP1A2 other statuses are wrong")
        check(a["ok_smiles_exact_match"] == 5 and a["ok_smiles_differ"] == 0, "CYP1A2 SMILES match counts are wrong")
        check(a["ok_identity_full_key_match"] == 5, "CYP1A2 identity class count is wrong")
        d = audit.loc["CYP2D6"]
        check(d["n_molecules_in_predictions"] == 2 and d["n_ok"] == 1 and d["n_missing_value"] == 1, "CYP2D6 audit counts are wrong")

        rel = result["relationship"]
        rel = rel[rel["isoform"] == "CYP1A2"].set_index("outcome")
        check(int(rel.loc["signed_error", "n_molecules"]) == 5, "relationship n != 5")
        check(_close(rel.loc["signed_error", "spearman_with_log2fc"], 1.0), "Spearman(log2fc, signed) should be 1.0")
        check(_close(rel.loc["absolute_error", "spearman_with_log2fc"], 0.1), "Spearman(log2fc, absolute) should be 0.1")
        check(_close(rel.loc["measured_pIC50_context", "spearman_with_log2fc"], -0.3), "Spearman(log2fc, measured) should be -0.3")
        check(_close(rel.loc["predicted_pIC50_context", "spearman_with_log2fc"], 0.1), "Spearman(log2fc, predicted) should be 0.1")
        check(np.isfinite(rel.loc["signed_error_given_predicted_pIC50_partial", "spearman_with_log2fc"]), "partial Spearman should be finite")
        rel_d = result["relationship"]
        rel_d = rel_d[(rel_d["isoform"] == "CYP2D6") & (rel_d["outcome"] == "signed_error")]
        check(np.isnan(rel_d["spearman_with_log2fc"].iloc[0]), "one-molecule CYP2D6 correlation should be NaN")

        tails = result["tails"]
        top = tails[(tails["isoform"] == "CYP1A2") & tails["tail_definition"].str.startswith("top_10pct")].iloc[0]
        check(top["n_tail"] == 1 and top["n_rest"] == 4, "top-decile tail should be 1 vs 4 molecules")
        check(_close(top["auc_tail_log2fc_exceeds_rest"], 1.0), "tail AUC should be 1.0")
        check(_close(top["tail_log2fc_median"], 0.6), "tail median log2fc should be 0.6")

        parts = result["partitions"]
        p1 = parts[(parts["isoform"] == "CYP1A2")]
        check(len(p1) == 2 and set(p1["n_molecules"]) == {5}, "CYP1A2 should have 2 partitions of 5 molecules")
        check(all(_close(v, 1.0) for v in p1["spearman_log2fc_vs_signed"]), "per-partition Spearman should be 1.0")
        check(result["bins"].empty and result["strata"].empty, "bins must be skipped for n < 10")

        rows = result["rows"]
        check(len(rows) == 45, f"row-level output should have 45 rows, has {len(rows)}")
        check(int((rows["status"] == "ambiguous_duplicate").sum()) == 3, "M6 rows should be flagged ambiguous (3 rows)")
        check({"repetition", "fold", "seed", "cell_index"} <= set(rows.columns), "row-level output must keep repetition, fold, seed and cell")
        for name in ("screening_join_audit.csv", "molecule_screening_error.csv", "prediction_rows_with_screening.csv",
                     "relationship_summary.csv", "log2fc_quintile_summary.csv", "prediction_strata_summary.csv",
                     "overprediction_tail_comparison.csv", "partition_relationship.csv", "run_metadata.json"):
            check((out / name).is_file(), f"missing output {name}")

        # identity class mismatch is excluded and counted, not silently used
        bad_classes = {m: "full_key_match" for m in ("M1", "M2", "M4", "M5", "M6")}
        bad_classes["M3"] = "no_identity_link"
        tar2, train2 = _self_screening(tmp / "screening2", classes=bad_classes, smiles_override={"M2": "CCCCCCCC"})
        train_csv_backup = train_csv
        r2 = run(argparse.Namespace(
            predictions_dir=predictions, cluster_assignments=clusters, split_manifest_dir=manifests,
            screen_tar=tar2, train_csv=train2, isoforms=["CYP1A2"], output_dir=tmp / "out2", cells=list(range(6)), sha256sums=None,
        ))
        a2 = r2["audit"].iloc[0]
        check(a2["n_identity_unverified"] == 1 and a2["n_ok"] == 4, "identity-unverified molecule should be excluded and counted")
        check(a2["ok_smiles_differ"] == 1, "a SMILES difference should be counted")
        del train_csv_backup

        # refusals
        expect(FileExistsError, "reuse of an output directory", lambda: run(args(out)))
        tar3, _ = _self_screening(tmp / "screening3", tamper=True)
        expect(ValueError, "screening hash mismatch", lambda: run(args(tmp / "o3", tar=tar3)))
        tar4, _ = _self_screening(tmp / "screening4", drop_member=IDENTITY_MEMBER)
        expect(FileNotFoundError, "missing screening member", lambda: run(args(tmp / "o4", tar=tar4)))
        expect(FileNotFoundError, "missing screening tarball", lambda: run(args(tmp / "o5", tar=tmp / "none.tar.gz")))
        expect(ValueError, "unknown isoform", lambda: run(args(tmp / "o6", isoforms=("CYP9Z9",))))
        shortened = pd.read_csv(train_csv).iloc[:5]
        shortened.to_csv(train_csv, index=False)
        expect(ValueError, "molecule absent from the training CSV", lambda: run(args(tmp / "o7")))

    if failures:
        print("SELFTEST FAILED:")
        for message in failures:
            print(" -", message)
        return 1
    print("SELFTEST PASSED: join audit, ambiguity and identity handling, Spearman/partial/tail/partition values and refusals match the independent expectations.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="synthetic-data check; reads and writes nothing in the repository")
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--cluster-assignments", type=Path)
    parser.add_argument("--split-manifest-dir", type=Path)
    parser.add_argument("--screen-tar", type=Path, help="actfound_screen_loader_20261117.tar.gz (read in place, not extracted)")
    parser.add_argument("--train-csv", type=Path, help="cyp-challenge-TRAIN_inhibition.csv (SMILES check for analysed molecules only)")
    parser.add_argument("--isoforms", nargs="+", choices=ISOFORMS)
    parser.add_argument("--output-dir", type=Path, help="must not exist")
    parser.add_argument("--cells", type=int, nargs="+", help="cell indices; omit for all 150")
    parser.add_argument("--sha256sums", type=Path, help="optional sha256sum file for the prediction files")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    required = ("predictions_dir", "cluster_assignments", "split_manifest_dir", "screen_tar", "train_csv", "isoforms", "output_dir")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing: " + ", ".join("--" + n.replace("_", "-") for n in missing))
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
