"""Descriptive development-validation diagnostics (Tripwire-CYP-007).

Reads the saved updated-T121 per-compound validation predictions
(t121_validation_inference.py output) and, optionally, ActFound-CYP's saved
per-compound development predictions (cyp_adaptation_predictions.json), and
writes tables into a NEW output directory. Descriptive only: associations are
not causal claims, no bootstrap, no significance statements. Only molecules
named in the frozen development folds enter; the protected internal test is
asserted absent. All error metrics come from individual prediction errors; no
predictions are averaged across seeds, repetitions or models.

Units. Updated T121: 150 cells (50 partitions x 3 seeds), each with its own
predictions. ActFound: 200 adaptations (50 partitions x 4 isoforms); the three
seed slots are replicates, so there is one prediction set per partition.
Partitions share compounds across repetitions and folds are not independent.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t121_validation_inference as inf  # noqa: E402

ISO = inf.ISOFORMS
TASK = {i: f"{i}{inf.TASK_SUFFIX}" for i in ISO}


def partition_key(rep, fold):
    return f"rep_{int(rep):02d}_fold_{int(fold):02d}"


def load_t121(directory):
    frames = [pd.read_csv(p) for p in sorted(Path(directory).glob("predictions_cell_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["partition"] = [partition_key(r, f) for r, f in zip(df.repetition, df.fold)]
    df["group"] = "cell_" + df.cell_index.astype(str)
    df["model"] = "updated_t121"
    return df


def load_actfound(path):
    payload = json.load(open(path))
    rows = []
    for cell in payload["cells"]:
        n = len(cell["target_names"])
        rows.append(pd.DataFrame({
            "partition": cell["partition"], "repetition": cell["repetition"],
            "fold": cell["fold"], "isoform": cell["endpoint"],
            "Molecule_Name": cell["target_names"], "y_true": cell["target_y"],
            "y_pred": cell["target_predictions"], "conf_low": cell["target_conf_low"],
            "conf_high": cell["target_conf_high"], "seed": -1, "cell_index": -1,
        }))
    df = pd.concat(rows, ignore_index=True)
    df["group"] = df.partition
    df["model"] = "actfound_cyp"
    return df


def add_errors(df):
    low, high = df.conf_low.to_numpy(), df.conf_high.to_numpy()
    df["signed_err"] = df.y_pred - df.y_true
    df["abs_err"] = df.signed_err.abs()
    df["soft_err"] = inf.soft_error(low, high, df.y_pred.to_numpy())
    mean_true = df.groupby(["model", "group", "isoform"]).y_true.transform("mean")
    df["base_soft_err"] = inf.soft_error(low, high, mean_true.to_numpy())
    df["ci_width"] = high - low
    return df


def check_alignment(t121, actfound):
    """ActFound and T121 must refer to the same compounds, labels and CIs."""
    first = t121[t121.seed == t121.seed.min()]
    key = ["partition", "isoform", "Molecule_Name"]
    merged = first.merge(actfound, on=key, suffixes=("_t", "_a"), validate="one_to_one")
    if len(merged) != len(first) or len(merged) != len(actfound):
        raise RuntimeError("T121 and ActFound rows do not cover the same compounds.")
    for col in ("y_true", "conf_low", "conf_high"):
        if not np.allclose(merged[f"{col}_t"], merged[f"{col}_a"], atol=1e-6, rtol=0):
            raise RuntimeError(f"T121 and ActFound disagree on {col}.")
    return len(merged)


def max_similarity(t121_or_a, dataset_path, split_path, manifest_root):
    """Max Morgan(r2, 2048) Tanimoto of each validation compound to the
    compounds with a label for that isoform in that partition's training set."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    data = pd.read_csv(dataset_path).set_index("Molecule_Name")
    split = pd.read_csv(split_path).set_index("Molecule_Name")
    used = set(t121_or_a.Molecule_Name)
    protected = set(split.index[split.split == "test"])
    if used & protected:
        raise RuntimeError("Protected internal-test molecule in the analysis rows.")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {}

    def fingerprint(name):
        if name not in fps:
            fps[name] = gen.GetFingerprint(Chem.MolFromSmiles(data.loc[name, "SMILES"]))
        return fps[name]

    records = []
    for (partition, isoform), group in t121_or_a.groupby(["partition", "isoform"]):
        manifest = json.load(open(Path(manifest_root) / f"{partition}.json"))
        if set(manifest["training_names"]) & protected:
            raise RuntimeError("Protected molecule in a training manifest.")
        trained = [n for n in manifest["training_names"] if np.isfinite(data.loc[n, TASK[isoform]])]
        train_fps = [fingerprint(n) for n in trained]
        for name in group.Molecule_Name.unique():
            sims = DataStructs.BulkTanimotoSimilarity(fingerprint(name), train_fps)
            records.append((partition, isoform, name, float(max(sims)), len(trained)))
    return pd.DataFrame(records, columns=["partition", "isoform", "Molecule_Name", "max_sim", "n_train_labeled"])


def metrics_table(df):
    """Per model/isoform: mean over groups (cells or partitions) of the row-based metrics."""
    out = []
    for (model, group, isoform), g in df.groupby(["model", "group", "isoform"]):
        m = inf.isoform_metrics(g.y_true, g.y_pred, g.conf_low, g.conf_high)
        r = np.corrcoef(g.y_pred, g.y_true)[0, 1]
        slope = np.polyfit(g.y_pred, g.y_true, 1)[0]
        out.append({"model": model, "group": group, "isoform": isoform, **m,
                    "pearson_r": r, "ols_slope_true_on_pred": slope,
                    "sd_pred_over_sd_true": g.y_pred.std() / g.y_true.std()})
    per_group = pd.DataFrame(out)
    summary = per_group.groupby(["model", "isoform"]).agg(
        groups=("group", "count"), n_rows_per_group=("n", "mean"),
        plain_mae=("plain_mae", "mean"), signed_bias=("signed_bias", "mean"),
        soft_mae=("soft_mae", "mean"), baseline_soft_mae=("baseline_soft_mae", "mean"),
        st_rae=("st_rae", "mean"), pearson_r=("pearson_r", "mean"),
        ols_slope_true_on_pred=("ols_slope_true_on_pred", "mean"),
        sd_pred_over_sd_true=("sd_pred_over_sd_true", "mean"),
    ).reset_index()
    return per_group, summary


def binned(df, column, edges_by_isoform, labels_prefix):
    out = []
    for (model, isoform), g in df.groupby(["model", "isoform"]):
        edges = edges_by_isoform[isoform]
        bins = pd.cut(g[column], edges, include_lowest=True)
        for interval, b in g.groupby(bins, observed=True):
            out.append({
                "model": model, "isoform": isoform, f"{labels_prefix}_bin": str(interval),
                "n_rows": len(b), "plain_mae": b.abs_err.mean(), "signed_bias": b.signed_err.mean(),
                "soft_mae": b.soft_err.mean(), "baseline_soft_mae": b.base_soft_err.mean(),
                "soft_over_baseline": b.soft_err.sum() / b.base_soft_err.sum(),
            })
    return pd.DataFrame(out)


def quartile_edges(df, column):
    unique = df.drop_duplicates(["partition", "isoform", "Molecule_Name"])
    edges = {}
    for isoform, g in unique.groupby("isoform"):
        e = np.unique(np.quantile(g[column], [0, 0.25, 0.5, 0.75, 1.0]))
        e[0] -= 1e-9
        edges[isoform] = e
    return edges


def constant_bias_sensitivity(df):
    """Post hoc, uses validation labels (diagnostic only, not a model): ST-RAE
    per isoform after subtracting each model's pooled signed bias for that
    isoform (one scalar per model and isoform, not per fold)."""
    out = []
    for (model, isoform), g in df.groupby(["model", "isoform"]):
        offset = g.signed_err.mean()
        scores = [
            inf.isoform_metrics(b.y_true, b.y_pred - offset, b.conf_low, b.conf_high)["st_rae"]
            for _, b in g.groupby("group")
        ]
        out.append({"model": model, "isoform": isoform, "pooled_signed_bias": offset,
                    "st_rae_original": float(np.mean([
                        inf.isoform_metrics(b.y_true, b.y_pred, b.conf_low, b.conf_high)["st_rae"]
                        for _, b in g.groupby("group")])),
                    "st_rae_bias_removed": float(np.mean(scores))})
    return pd.DataFrame(out)


def head_to_head(t121, actfound):
    """Per compound row: T121's soft error (mean over its three seeds' errors)
    against ActFound's. Errors are averaged, not predictions."""
    key = ["partition", "isoform", "Molecule_Name"]
    t = t121.groupby(key).agg(t121_soft=("soft_err", "mean"), max_sim=("max_sim", "first"),
                              ci_width=("ci_width", "first")).reset_index()
    a = actfound[key + ["soft_err"]].rename(columns={"soft_err": "actfound_soft"})
    merged = t.merge(a, on=key, validate="one_to_one")
    merged["t121_lower"] = merged.t121_soft < merged.actfound_soft
    merged["tie"] = merged.t121_soft == merged.actfound_soft
    return merged


def seed_instability(t121):
    sd = t121.groupby(["isoform", "partition", "Molecule_Name"]).y_pred.std(ddof=0)
    out = sd.groupby("isoform").mean().rename("mean_sd_of_prediction_across_3_seeds").reset_index()
    mae = t121.groupby("isoform").abs_err.mean().rename("pooled_plain_mae").reset_index()
    return out.merge(mae, on="isoform")


def spearman_table(df):
    out = []
    for (model, isoform), g in df.groupby(["model", "isoform"]):
        out.append({
            "model": model, "isoform": isoform,
            "spearman_maxsim_vs_abs_err": g.max_sim.corr(g.abs_err, method="spearman"),
            "spearman_maxsim_vs_soft_err": g.max_sim.corr(g.soft_err, method="spearman"),
            "spearman_ciwidth_vs_abs_err": g.ci_width.corr(g.abs_err, method="spearman"),
            "spearman_ciwidth_vs_soft_err": g.ci_width.corr(g.soft_err, method="spearman"),
            "spearman_maxsim_vs_ciwidth": g.max_sim.corr(g.ci_width, method="spearman"),
        })
    return pd.DataFrame(out)


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    t121 = add_errors(load_t121(args.t121_dir))
    frames = [t121]
    actfound = None
    if args.actfound_predictions:
        actfound = add_errors(load_actfound(args.actfound_predictions))
        aligned = check_alignment(t121, actfound)
        frames.append(actfound)
        print(f"ActFound and T121 aligned on {aligned} compound rows.")
    allrows = pd.concat(frames, ignore_index=True)
    sim = max_similarity(allrows, args.dataset, args.original_split, args.split_manifest_root)
    allrows = allrows.merge(sim, on=["partition", "isoform", "Molecule_Name"], validate="many_to_one")
    allrows.to_csv(out / "rows_with_similarity.csv", index=False)

    per_group, summary = metrics_table(allrows)
    per_group.to_csv(out / "metrics_per_group.csv", index=False)
    summary.to_csv(out / "metrics_per_isoform.csv", index=False)
    seeds = (t121.assign(g=t121.group)
             .pipe(lambda d: d.groupby(["seed", "group", "isoform"]).apply(
                 lambda g: inf.isoform_metrics(g.y_true, g.y_pred, g.conf_low, g.conf_high)["st_rae"],
                 include_groups=False).rename("st_rae").reset_index())
             .groupby(["seed", "isoform"]).st_rae.mean().unstack("isoform"))
    seeds["macro"] = seeds.mean(axis=1)
    seeds.to_csv(out / "t121_st_rae_by_seed.csv")

    sim_edges = quartile_edges(allrows, "max_sim")
    ci_edges = quartile_edges(allrows, "ci_width")
    by_sim = binned(allrows, "max_sim", sim_edges, "max_sim")
    by_ci = binned(allrows, "ci_width", ci_edges, "ci_width")
    by_sim.to_csv(out / "error_by_similarity_quartile.csv", index=False)
    by_ci.to_csv(out / "error_by_ci_width_quartile.csv", index=False)
    label_edges = quartile_edges(allrows, "y_true")
    binned(allrows, "y_true", label_edges, "y_true").to_csv(
        out / "error_by_label_quartile.csv", index=False)
    constant_bias_sensitivity(allrows).to_csv(
        out / "constant_bias_sensitivity.csv", index=False)
    spearman_table(allrows).to_csv(out / "spearman_descriptive.csv", index=False)
    seed_instability(t121).to_csv(out / "t121_seed_instability.csv", index=False)
    if actfound is not None:
        h2h = head_to_head(allrows[allrows.model == "updated_t121"], allrows[allrows.model == "actfound_cyp"])
        h2h.to_csv(out / "head_to_head_rows.csv", index=False)
        rows = []
        for iso, g in h2h.groupby("isoform"):
            rows.append({"isoform": iso, "n_rows": len(g),
                         "frac_t121_lower_soft_err": g.t121_lower.mean(),
                         "frac_tie": g.tie.mean(),
                         "mean_actfound_minus_t121_soft": (g.actfound_soft - g.t121_soft).mean()})
        pd.DataFrame(rows).to_csv(out / "head_to_head_by_isoform.csv", index=False)
        bins = []
        for iso, g in h2h.groupby("isoform"):
            cut = pd.cut(g.max_sim, sim_edges[iso], include_lowest=True)
            for interval, b in g.groupby(cut, observed=True):
                bins.append({"isoform": iso, "max_sim_bin": str(interval), "n_rows": len(b),
                             "frac_t121_lower": b.t121_lower.mean(),
                             "mean_actfound_minus_t121_soft": (b.actfound_soft - b.t121_soft).mean()})
        pd.DataFrame(bins).to_csv(out / "head_to_head_by_similarity.csv", index=False)
    print("written:", sorted(p.name for p in out.iterdir()))


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ("t121-dir", "dataset", "original-split", "split-manifest-root", "output-dir"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--actfound-predictions")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
