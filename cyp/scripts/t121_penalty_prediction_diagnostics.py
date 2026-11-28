"""Exploratory diagnostics of saved CYP validation predictions: penalty arms against
unchanged updated T121 (Tripwire-CYP-011, after the D029 result).

Reads saved prediction CSVs, the adoption reports, the frozen tail cohorts and
the screening inputs. It trains nothing, runs no inference, loads no checkpoint,
changes no adoption rule and writes only into a NEW output directory.

Pairing: every prediction file must carry the per-cell SHA-256 its adoption
report recorded, and each arm must have exactly the unchanged arm's
(cell, isoform, molecule) keys with identical cell identity, label and bounds. An
arm that is missing or fails is recorded as stopped and left out; nothing is
regenerated. A failure of the unchanged arm stops the run.

Aggregation (the session-009 molecule-level rule, t121_error_diagnostic.py):
rows -> seed mean within a repetition -> mean over repetitions, so each molecule
counts once. Absolute error is the mean of the row absolute errors, not the
error of the averaged prediction. Per seed: the mean over that seed's
repetitions. Prediction change is arm minus unchanged.

Limits that apply to every output: descriptive and exploratory only; development
partitions already inspected; the tail cohorts were selected from unchanged's own
errors, so comparisons on them are baseline-conditioned; arm and unchanged are
separate training runs, so a change includes run-to-run variation; screening
coverage is an association only. The mean-shift counterfactual uses evaluation
predictions, is not a correction or a candidate and is not eligible for adoption.

Exit status: 0 complete; 2 complete with at least one stopped comparison.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` omits the script dir
import t121_screening_penalty as penalty  # noqa: E402

BASELINE = "unchanged"
COLUMNS = [
    "trial_index", "cell_index", "repetition", "fold", "seed", "isoform",
    "Molecule_Name", "y_true", "y_pred", "conf_low", "conf_high",
]
KEY = ["cell_index", "isoform", "Molecule_Name"]
PAIRED = ["trial_index", "repetition", "fold", "seed", "y_true", "conf_low", "conf_high"]
MOLECULE = ["isoform", "Molecule_Name"]
VALUES = ["pred", "signed", "abs_error"]
TAILS = {"over": "over_tail", "under": "under_tail"}
GROUPS = ("all", "over_tail", "under_tail", "remaining")
COUNTERFACTUAL_GROUPS = ("all", "over_tail", "under_tail")
COUNTERFACTUAL_STATUS = "diagnostic_counterfactual_not_a_candidate"
TOLERANCE = 1e-12  # reconciliation with the reports' own tail MAEs
NOTES = [
    "Descriptive and exploratory only; no new gate, no change to the adoption rule, no next experiment selected.",
    "Molecule level: seed mean within a repetition, then mean over repetitions; absolute error is the mean of row absolute errors.",
    "'equal' means exactly equal molecule-level absolute error (no tolerance).",
    "Tail cohorts were selected from the unchanged arm's own errors: tail comparisons are baseline-conditioned.",
    "Raw-MAE contributions are not a decomposition of macro ST-RAE; macro values are copied from the reports.",
    "Screening coverage is a descriptive association; it does not show how the training penalty moved a held-out molecule.",
    "The mean-shift counterfactual uses evaluation predictions; not deployable, not a candidate, not causal, not eligible for adoption.",
]


class Stop(Exception):
    """A comparison that cannot be made from the saved artifacts."""


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_json(path, expected, label):
    path = Path(path)
    if not path.is_file():
        raise Stop(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise Stop(f"{label} does not match its pinned SHA-256: {actual}")
    return json.loads(path.read_text())


def named(items, option):
    result = {}
    for item in items or []:
        name, _, path = item.partition("=")
        if not name or not path or name in result:
            raise SystemExit(f"{option} needs distinct ARM=DIR values, got {item!r}")
        result[name] = Path(path)
    return result


# --------------------------------------------------------------------------- input
def load_arm(directory, arm, report, config):
    """One arm's rows, indexed by KEY, verified against the report's hashes."""
    entry = report.get("inputs", {}).get(arm)
    if entry is None or arm not in report.get("metrics", {}):
        raise Stop(f"the report has no entry for arm {arm!r}")
    seeds = [int(s) for s in config["cell_seeds"]]
    per_molecule = int(config["predictions_per_molecule"])
    frames = []
    for cell in range(len(seeds) * int(config["cells_per_seed"])):
        path = Path(directory) / f"predictions_cell_{cell:03d}.csv"
        if not path.is_file():
            raise Stop(f"missing prediction file: {path}")
        if sha256_file(path) != entry["prediction_sha256_by_cell"].get(str(cell)):
            raise Stop(f"{path.name} does not match the report's SHA-256 for this arm")
        frame = pd.read_csv(path, float_precision="round_trip", dtype={"Molecule_Name": str})
        if list(frame.columns) != COLUMNS:
            raise Stop(f"{path.name}: unexpected columns {list(frame.columns)}")
        if frame.empty or not (frame["cell_index"] == cell).all():
            raise Stop(f"{path.name}: empty, or rows from another cell")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    trial = re.search(r"trial_(\d+)", entry["cells_dir"])
    if trial is None or set(df["trial_index"]) != {int(trial.group(1))}:
        raise Stop(f"trial_index differs from the report's cell directory: {sorted(set(df['trial_index']))}")
    if not np.isfinite(df[["y_true", "y_pred", "conf_low", "conf_high"]].to_numpy(float)).all():
        raise Stop("non-finite label, prediction or bound")
    if df.duplicated(KEY).any():
        raise Stop("duplicate (cell, isoform, molecule) rows")
    if (df.groupby("cell_index")[["repetition", "fold", "seed"]].nunique() != 1).any().any():
        raise Stop("a cell has more than one repetition, fold or seed")
    if set(df["isoform"]) != set(config["isoforms"]) or set(df["seed"]) != set(seeds):
        raise Stop("isoforms or seeds differ from the evaluation config")
    counts = df.groupby(MOLECULE).agg(
        rows=("seed", "size"), seeds=("seed", "nunique"), repetitions=("repetition", "nunique"),
    )
    repetitions = per_molecule // len(seeds)
    pairs = df.drop_duplicates(MOLECULE + ["seed", "repetition"]).groupby(MOLECULE).size()
    if not (
        (counts["rows"] == per_molecule).all() and (counts["seeds"] == len(seeds)).all()
        and (counts["repetitions"] == repetitions).all() and (pairs == per_molecule).all()
    ):
        raise Stop("a molecule does not have each seed x repetition exactly once")
    if (df.groupby(MOLECULE)[["y_true", "conf_low", "conf_high"]].nunique() != 1).any().any():
        raise Stop("a molecule's label or bounds differ between cells")

    df["signed"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["signed"].abs()
    df["pred"] = df["y_pred"]
    return df.set_index(KEY).sort_index()


def check_paired(base, other):
    if not base.index.equals(other.index):
        only_base = len(base.index.difference(other.index))
        only_other = len(other.index.difference(base.index))
        raise Stop(
            f"(cell, isoform, molecule) keys differ: {only_base} only in unchanged, {only_other} only in this arm"
        )
    for column in PAIRED:
        different = int((base[column].to_numpy() != other[column].to_numpy()).sum())
        if different:
            label = "label or bound" if column in ("y_true", "conf_low", "conf_high") else "cell identity"
            raise Stop(f"{label} column {column} differs from unchanged on {different} rows")


def load_cohorts(path, expected, config, base):
    data = pinned_json(path, expected, "cohort file")["cohorts"]
    for isoform in config["isoforms"]:
        molecules = set(base.index.get_level_values("Molecule_Name")[
            base.index.get_level_values("isoform") == isoform])
        cohort = data[isoform]
        over, under = set(cohort["over"]), set(cohort["under"])
        if (
            cohort["n"] != len(molecules) or over & under or not (over | under) <= molecules
            or len(over) != cohort["size"] or len(under) != cohort["size"]
        ):
            raise Stop(f"{isoform}: cohort membership is inconsistent with the unchanged predictions")
    return data


def check_tails(rows, cohorts, reported, arm):
    """The report's row-pooled tail MAEs must be reproduced from these rows."""
    isoforms = rows.index.get_level_values("isoform")
    molecules = rows.index.get_level_values("Molecule_Name")
    largest = 0.0
    for isoform, cohort in cohorts.items():
        for tail in TAILS:
            chosen = (isoforms == isoform) & molecules.isin(cohort[tail])
            value = float(rows.loc[chosen, "abs_error"].mean())
            difference = abs(value - reported[isoform][tail]["mae"])
            if not difference <= TOLERANCE:
                raise Stop(f"{arm} {isoform} {tail}-tail MAE differs from the report by {difference}")
            largest = max(largest, difference)
    return largest


# --------------------------------------------------------------------------- aggregation
def molecule_level(rows):
    """Seed mean within a repetition, then mean over repetitions."""
    repetition = rows.groupby(MOLECULE + ["repetition"])[VALUES].mean()
    return repetition.groupby(level=MOLECULE).mean()


def seed_level(rows):
    """Per downstream seed: each molecule's mean over that seed's repetitions."""
    return rows.groupby(MOLECULE + ["seed"])[VALUES].mean()


def group_labels(index, cohorts):
    """over_tail / under_tail / remaining for a (isoform, molecule) index."""
    labels = []
    members = {i: {t: set(c[t]) for t in TAILS} for i, c in cohorts.items()}
    for isoform, molecule in index:
        label = "remaining"
        for tail, name in TAILS.items():
            if molecule in members[isoform][tail]:
                label = name
        labels.append(label)
    return pd.Series(labels, index=index, name="group")


def select(frame, labels, isoform, group):
    """Rows of one isoform and group from a (isoform, molecule)-indexed frame."""
    chosen = frame.index.get_level_values("isoform") == isoform
    if group != "all":
        chosen &= (labels.reindex(frame.index) == group).to_numpy()
    return frame[chosen]


def group_summary(base, arm, labels, isoform, group):
    b, a = select(base, labels, isoform, group), select(arm, labels, isoform, group)
    difference = a["abs_error"] - b["abs_error"]
    change = a["pred"] - b["pred"]
    reduction = float((-difference[difference < 0]).sum())
    increase = float(difference[difference > 0].sum())
    return {
        "n_molecules": len(b),
        "mae_unchanged": float(b["abs_error"].mean()), "mae_arm": float(a["abs_error"].mean()),
        "delta_mae": float(difference.mean()),
        "n_lower": int((difference < 0).sum()), "n_equal": int((difference == 0).sum()),
        "n_higher": int((difference > 0).sum()),
        "total_error_reduction": reduction, "total_error_increase": increase,
        "net_error_change": float(difference.sum()),
        "mean_prediction_change": float(change.mean()),
        "median_prediction_change": float(change.median()),
        "mean_signed_residual_unchanged": float(b["signed"].mean()),
        "mean_signed_residual_arm": float(a["signed"].mean()),
    }


def concentration(base, arm, labels, isoform):
    """Molecules ordered by their own contribution, separately for each direction."""
    b, a = select(base, labels, isoform, "all"), select(arm, labels, isoform, "all")
    difference = a["abs_error"] - b["abs_error"]
    parts = []
    for direction, values in (("improvement", -difference), ("deterioration", difference)):
        values = values[values > 0]
        if values.empty:
            continue
        table = values.rename("contribution").reset_index()
        table = table.sort_values(
            ["contribution", "Molecule_Name"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        table["direction"] = direction
        table["rank"] = np.arange(1, len(table) + 1)
        table["cumulative_contribution"] = table["contribution"].cumsum()
        table["cumulative_share"] = table["cumulative_contribution"] / table["contribution"].sum()
        table["share_of_contributing_molecules"] = table["rank"] / len(table)
        table["share_of_all_molecules"] = table["rank"] / len(b)
        table["group"] = [labels[(isoform, m)] for m in table["Molecule_Name"]]
        parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else None


def counterfactual(base_rows, arm_rows, base, labels, isoform):
    """Remove the arm's mean prediction change for one isoform, then re-score."""
    chosen = arm_rows.index.get_level_values("isoform") == isoform
    b, a = base_rows[chosen], arm_rows[chosen].copy()
    shift = float((a["pred"] - b["pred"]).mean())
    actual = molecule_level(a)
    a["pred"] = a["pred"] - shift
    a["signed"] = a["pred"] - a["y_true"]
    a["abs_error"] = a["signed"].abs()
    shifted = molecule_level(a)
    result = []
    for group in COUNTERFACTUAL_GROUPS:
        u = float(select(base, labels, isoform, group)["abs_error"].mean())
        x = float(select(actual, labels, isoform, group)["abs_error"].mean())
        y = float(select(shifted, labels, isoform, group)["abs_error"].mean())
        result.append({
            "group": group, "status": COUNTERFACTUAL_STATUS, "mean_prediction_change_removed": shift,
            "mae_unchanged": u, "mae_actual": x, "mae_counterfactual": y,
            "delta_actual": x - u, "delta_counterfactual": y - u,
        })
    return result


# --------------------------------------------------------------------------- coverage
def screening_coverage(args, base):
    """(status, covered) per (isoform, molecule) under the D024 verified-value rule.

    The rebuilt grouping of the frozen file's isoform must equal that file."""
    frozen = pinned_json(args.coverage, args.coverage_sha256, "coverage file")
    for label, path, key in (
        ("coverage config", args.coverage_config, "coverage_config_sha256"),
        ("coverage penalty config", args.coverage_penalty_config, "penalty_config_sha256"),
    ):
        if not Path(path).is_file() or sha256_file(path) != frozen[key]:
            raise Stop(f"{label} is not the file the frozen coverage grouping was built with")
    config = json.loads(Path(args.coverage_config).read_text())
    if not Path(args.screen_tar).is_file():
        raise Stop(f"screening archive is missing: {args.screen_tar}")
    arm = penalty.PenaltyArm.load(
        args.coverage_penalty_config, expected_sha256=frozen["penalty_config_sha256"])
    screen = penalty.load_screen_tar(args.screen_tar, arm)
    if screen["sha256"] != frozen["screen_input_sha256"]:
        raise Stop("screening inputs differ from those of the frozen coverage grouping")
    status = {}
    for isoform, molecule in base.index:
        value = penalty.lookup(screen, arm, molecule, isoform)[0]
        status[(isoform, molecule)] = (value, value in config["covered_statuses"])
    isoform = frozen["isoform"]
    rebuilt = {m: s for (i, m), s in status.items() if i == isoform}
    if (
        {m: s for m, (s, _) in rebuilt.items()} != frozen["statuses"]
        or {m: "covered" if c else "not_covered" for m, (_, c) in rebuilt.items()} != frozen["groups"]
    ):
        raise Stop(f"rebuilt {isoform} coverage differs from the frozen grouping")
    return status


def coverage_rows(base, arm, status, isoform):
    b = base[base.index.get_level_values("isoform") == isoform]
    a = arm.loc[b.index]
    covered = np.array([status[key][1] for key in b.index], dtype=bool)
    result = []
    for name, chosen in (("covered", covered), ("not_covered", ~covered)):
        n = int(chosen.sum())
        entry = {"coverage": name, "n_molecules": n, "mean_prediction_change": None,
                 "mae_unchanged": None, "mae_arm": None, "delta_mae": None}
        if n:
            u, x = float(b["abs_error"][chosen].mean()), float(a["abs_error"][chosen].mean())
            entry.update(
                mean_prediction_change=float((a["pred"] - b["pred"])[chosen].mean()),
                mae_unchanged=u, mae_arm=x, delta_mae=x - u)
        result.append(entry)
    return result


# --------------------------------------------------------------------------- run
def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {output}")
    config = json.loads(Path(args.config).read_text())
    candidates = named(args.predictions, "--predictions")
    context = named(args.context_predictions, "--context-predictions")
    if BASELINE not in candidates or set(candidates) & set(context) or BASELINE in context:
        raise SystemExit("--predictions must include unchanged; arm names must not repeat.")
    stopped, checks = {}, {}

    # The unchanged arm and everything every comparison needs: any failure stops the run.
    report = pinned_json(args.report, args.report_sha256, "report")
    if report["cohorts_sha256"] != args.cohorts_sha256:
        raise Stop("the report was not made with this cohort file")
    base_rows = load_arm(candidates.pop(BASELINE), BASELINE, report, config)
    base = molecule_level(base_rows)
    cohorts = load_cohorts(args.cohorts, args.cohorts_sha256, config, base_rows)
    checks["tail_mae_max_difference"] = {
        BASELINE: check_tails(base_rows, cohorts, report["metrics"][BASELINE]["tails"], BASELINE)}
    labels = group_labels(base.index, cohorts)

    context_report = None
    if context:
        try:
            if not args.context_report or not args.context_report_sha256:
                raise Stop("no context report was given")
            context_report = pinned_json(
                args.context_report, args.context_report_sha256, "context report")
            if context_report["cohorts_sha256"] != args.cohorts_sha256:
                raise Stop("the context report used another cohort file")
            if (context_report["inputs"][BASELINE]["prediction_sha256_by_cell"]
                    != report["inputs"][BASELINE]["prediction_sha256_by_cell"]):
                raise Stop("the context report used other unchanged predictions")
            if context_report["metrics"][BASELINE]["O150"] != report["metrics"][BASELINE]["O150"]:
                raise Stop("the reports disagree on the unchanged macro value")
        except Stop as reason:
            context_report = None
            for name in context:
                stopped[name] = str(reason)

    arms, roles, reports = {}, {}, {}
    for role, items, source in (("candidate", candidates, report), ("context", context, context_report)):
        for name, directory in items.items():
            if name in stopped:
                continue
            try:
                rows = load_arm(directory, name, source, config)
                check_paired(base_rows, rows)
                checks["tail_mae_max_difference"][name] = check_tails(
                    rows, cohorts, source["metrics"][name]["tails"], name)
            except Stop as reason:
                stopped[name] = str(reason)
                continue
            arms[name], roles[name], reports[name] = rows, role, source
    if not arms:
        raise Stop(f"no arm can be compared: {stopped}")

    status = None
    try:
        if report["coverage_sha256"] != args.coverage_sha256:
            raise Stop("the report was not made with this coverage file")
        status = screening_coverage(args, base)
    except (Stop, ValueError, KeyError, OSError) as reason:
        stopped["screening_coverage"] = f"{type(reason).__name__}: {reason}"

    isoforms = config["isoforms"]
    summaries, curves, seeds, coverage, shifted = [], [], [], [], []
    wide = base_rows.groupby(MOLECULE)[["y_true", "conf_low", "conf_high"]].first()
    wide.insert(0, "group", labels.reindex(wide.index))
    if status is not None:
        wide["screen_status"] = [status[key][0] for key in wide.index]
        wide["screen_covered"] = [status[key][1] for key in wide.index]
    wide["n_predictions"] = base_rows.groupby(MOLECULE).size()
    for column in VALUES:
        wide[f"{column}_{BASELINE}"] = base[column]
    base_seed = seed_level(base_rows)

    for name, rows in arms.items():
        arm = molecule_level(rows)
        for column in VALUES:
            wide[f"{column}_{name}"] = arm[column]
        wide[f"change_{name}"] = arm["pred"] - base["pred"]
        wide[f"abs_error_change_{name}"] = arm["abs_error"] - base["abs_error"]
        arm_seed = seed_level(rows)
        for isoform in isoforms:
            tag = {"isoform": isoform, "arm": name, "role": roles[name]}
            for group in GROUPS:
                summaries.append({**tag, "group": group, **group_summary(base, arm, labels, isoform, group)})
            table = concentration(base, arm, labels, isoform)
            if table is not None:
                curves.append(table.assign(**{k: v for k, v in tag.items() if k != "isoform"}))
            for seed in config["cell_seeds"]:
                b = base_seed.xs(seed, level="seed")
                a = arm_seed.xs(seed, level="seed")
                for group in GROUPS:
                    u = select(b, labels, isoform, group)["abs_error"]
                    x = select(a, labels, isoform, group)["abs_error"]
                    seeds.append({
                        **tag, "seed": seed, "group": group, "n_molecules": len(u),
                        "mae_unchanged": float(u.mean()), "mae_arm": float(x.mean()),
                        "delta_mae": float((x - u).mean()),
                    })
            shifted.extend({**tag, **row} for row in counterfactual(base_rows, rows, base, labels, isoform))
            if status is not None:
                coverage.extend({**tag, **row} for row in coverage_rows(base, arm, status, isoform))

    macro = []
    for name in (BASELINE, *arms):
        source = report if name == BASELINE else reports[name]
        values = source["metrics"][name]
        reference = source["metrics"][BASELINE]
        for scope, value, other in (
            ("macro", values["O150"], reference["O150"]),
            *((i, values["per_isoform"][i], reference["per_isoform"][i]) for i in isoforms),
        ):
            macro.append({
                "arm": name, "role": roles.get(name, BASELINE), "scope": scope, "st_rae": value,
                "difference_from_unchanged": value - other,
                "source_report_decision": source.get("decision_record"),
            })

    output.mkdir(parents=True)
    tables = {
        "group_summary.csv": pd.DataFrame(summaries),
        "molecule_level.csv": wide.reset_index(),
        "concentration.csv": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(
            columns=["isoform", "arm", "role", "direction", "rank", "Molecule_Name"]),
        "seed_breakdown.csv": pd.DataFrame(seeds),
        "counterfactual.csv": pd.DataFrame(shifted),
        "macro_st_rae.csv": pd.DataFrame(macro),
    }
    if status is not None:
        tables["coverage_summary.csv"] = pd.DataFrame(coverage)
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)

    metadata = {
        "arms": roles, "isoforms": isoforms, "seeds": config["cell_seeds"], "groups": list(GROUPS),
        "stopped": stopped, "checks": checks, "notes": NOTES,
        "inputs": {
            "config_sha256": sha256_file(args.config), "report_sha256": args.report_sha256,
            "context_report_sha256": args.context_report_sha256 if context_report else None,
            "cohorts_sha256": args.cohorts_sha256,
            "coverage_sha256": args.coverage_sha256 if status is not None else None,
            "prediction_dirs": {
                name: str(path) for name, path in {
                    **named(args.predictions, "--predictions"),
                    **named(args.context_predictions, "--context-predictions")}.items()},
        },
        "script_sha256": sha256_file(__file__),
        "penalty_module_sha256": sha256_file(penalty.__file__),
        "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
        "outputs": {name: sha256_file(output / name) for name in tables},
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    overall = tables["group_summary.csv"]
    print(overall[["isoform", "arm", "group", "n_molecules", "mae_unchanged", "mae_arm", "delta_mae",
                   "n_lower", "n_higher", "mean_prediction_change"]].round(5).to_string(index=False))
    print("largest tail-MAE difference from the reports:",
          max(checks["tail_mae_max_difference"].values()))
    for name, reason in stopped.items():
        print(f"STOPPED {name}: {reason}")
    print("wrote", output)
    return 2 if stopped else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("config", "report", "report-sha256", "cohorts", "cohorts-sha256", "coverage",
                 "coverage-sha256", "coverage-config", "coverage-penalty-config", "screen-tar", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--predictions", action="append", required=True, metavar="ARM=DIR")
    parser.add_argument("--context-predictions", action="append", metavar="ARM=DIR")
    parser.add_argument("--context-report")
    parser.add_argument("--context-report-sha256")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
