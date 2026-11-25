"""D024 comparison: unchanged updated T121 versus the Q3-cutoff and rank penalty arms.

Adoption rule (D024 Amendments 1 and 2). A penalty arm passes only if, relative
to the unchanged arm over the matched 150 cells:
  (a) its mean macro ST-RAE is strictly lower;
  (b) none of the 12 enzyme x cell-seed means (per-cell per-enzyme ST-RAE
      averaged over that seed's 50 cells) is worse (equality passes);
  (c) its MAE on every enzyme's fixed over-prediction tail is strictly lower;
  (d) its MAE on every enzyme's fixed under-prediction tail is not higher.
If both pass, the lower macro ST-RAE wins and an exact tie goes to the cutoff
arm. If neither passes, the unchanged arm is kept.

Tail cohorts: per enzyme, molecules are ranked by their mean signed error
(predicted - measured) over all their baseline predictions; the top and bottom
max(1, round(fraction * n)) are the over and under cohorts, ties by molecule
name. They are built once, from the baseline only. Tail MAE is over all
predictions of a cohort's molecules. Membership comes from the baseline's own
validation errors, so changes on the cohorts are not independent evidence.

Report-only addition (not an adoption clause): results for one enzyme (CYP3A4)
are also reported separately for molecules with and without screening
coverage ("covered" = a verified screening value exists under the penalty's
verified-value rule), next to the full-enzyme and fixed-tail results, from the
same saved predictions. The grouping is built once and hash-pinned like the
cohorts.

Reads result files and saved prediction CSVs only; runs no model code.

    cohorts   --baseline-predictions D --sha256sums F --config C --output NEW.json
    coverage  --baseline-predictions D --sha256sums F --config C --coverage-config K
              --penalty-config P --screen-tar T --output NEW.json
    evaluate  --config C --cohorts F --cohorts-sha256 H --coverage F --coverage-sha256 H
              --cells unchanged=D --cells q3=D --cells rank=D
              --predictions unchanged=D --predictions q3=D --predictions rank=D
              --output NEW.json
"""
from pathlib import Path
import argparse
import csv
import json
import math
import sys

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` omits the script dir
import t121_d021_aggregate as base  # noqa: E402
import t121_validation_inference as inference  # noqa: E402

ARMS = ("unchanged", "q3", "rank")
PENALTY_ARMS = ("q3", "rank")
sha256_file = base.sha256_file
mean = base.mean


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {
        "isoforms", "task_suffix", "cell_seeds", "cells_per_seed", "predictions_per_molecule",
        "tail_fraction", "arm_labels", "tie_preference", "baseline_o150", "baseline_o150_tolerance",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Evaluation config lacks: {sorted(missing)}")
    return config


def sums_table(path):
    table = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            table[Path(name.strip().lstrip("*")).name] = digest
    return table


# --------------------------------------------------------------------- predictions
def load_predictions(directory, cell_indices, sums=None):
    """Rows keyed (cell, isoform, molecule) from reproduced validation inference.

    Returns (rows, checkpoint SHA-256 by cell, prediction-file SHA-256 by cell).
    """
    directory = Path(directory)
    rows, checkpoints, files = {}, {}, {}
    for cell in cell_indices:
        name = f"cell_{cell:03d}"
        csv_path = directory / f"predictions_{name}.csv"
        record_path = directory / f"reproduction_{name}.json"
        digest = sha256_file(csv_path)
        if sums is not None:
            for path, actual in ((csv_path, digest), (record_path, sha256_file(record_path))):
                if sums.get(path.name) != actual:
                    raise ValueError(f"{path.name} does not match the checksum file.")
        record = json.loads(record_path.read_text())
        if record["status"] != "reproduced" or record["cell_index"] != cell:
            raise ValueError(f"{name}: inference record is not a reproduced record of this cell.")
        checkpoints[cell] = record["checkpoint_sha256"]
        files[cell] = digest
        count = 0
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["cell_index"]) != cell:
                    raise ValueError(f"{csv_path.name}: row from another cell.")
                key = (cell, row["isoform"], row["Molecule_Name"])
                if key in rows:
                    raise ValueError(f"Duplicate prediction row: {key}")
                values = tuple(float(row[c]) for c in ("y_true", "y_pred", "conf_low", "conf_high"))
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(f"Non-finite prediction row: {key}")
                rows[key] = values
                count += 1
        if count != record["prediction_rows"]:
            raise ValueError(f"{name}: prediction row count differs from its record.")
    return rows, checkpoints, files


# ------------------------------------------------------------------------- cohorts
def build_cohorts(rows, config):
    cohorts = {}
    for isoform in config["isoforms"]:
        errors = {}
        for (_, row_isoform, molecule), (y_true, y_pred, _, _) in rows.items():
            if row_isoform == isoform:
                errors.setdefault(molecule, []).append(y_pred - y_true)
        if not errors:
            raise ValueError(f"No predictions for {isoform}.")
        counts = {len(v) for v in errors.values()}
        if counts != {config["predictions_per_molecule"]}:
            raise ValueError(f"{isoform}: unexpected predictions per molecule: {sorted(counts)}")
        signed = {m: math.fsum(v) / len(v) for m, v in errors.items()}
        n = len(signed)
        size = max(1, round(config["tail_fraction"] * n))
        over = sorted(signed, key=lambda m: (-signed[m], m))[:size]
        under = sorted(signed, key=lambda m: (signed[m], m))[:size]
        if set(over) & set(under):
            raise ValueError(f"{isoform}: over and under cohorts overlap.")
        cohorts[isoform] = {"n": n, "size": size, "over": sorted(over), "under": sorted(under)}
    return cohorts


# ------------------------------------------------------------------------- metrics
def tail_metrics(rows, cohorts):
    result = {}
    for isoform, cohort in cohorts.items():
        entry = {}
        for tail in ("over", "under"):
            members = set(cohort[tail])
            errors = [
                y_pred - y_true
                for (_, row_isoform, molecule), (y_true, y_pred, _, _) in rows.items()
                if row_isoform == isoform and molecule in members
            ]
            entry[tail] = {
                "rows": len(errors),
                "mae": mean(abs(e) for e in errors),
                "signed_error": mean(errors),
            }
        result[isoform] = entry
    return result


def cell_metrics(run, config):
    """O150, per-enzyme means and the enzyme x seed means of one 150-cell run."""
    suffix = config["task_suffix"]
    by_seed = {}
    for isoform in config["isoforms"]:
        by_seed[isoform] = {}
        for seed in config["cell_seeds"]:
            cells = [c for c in run["cells"] if c["seed"] == seed]
            if len(cells) != config["cells_per_seed"]:
                raise ValueError(f"Seed {seed}: expected {config['cells_per_seed']} cells, found {len(cells)}")
            by_seed[isoform][str(seed)] = mean(c["per_task"][isoform + suffix] for c in cells)
    summary = base.summarize(run)
    return {
        "O150": summary["O150"],
        "per_isoform": {i: summary["per_isoform"][i + suffix] for i in config["isoforms"]},
        "by_isoform_seed": by_seed,
    }


# ---------------------------------------------------------------------- coverage
def load_coverage_config(path):
    config = json.loads(Path(path).read_text())
    required = {"isoform", "covered_statuses", "groups"}
    if required - set(config) or set(config["groups"]) != {"covered", "not_covered"}:
        raise ValueError("Coverage config must name an isoform, covered_statuses and the two groups.")
    return config


def build_coverage(rows, coverage_config, status_of):
    """Group every molecule predicted for the coverage isoform by screening status."""
    isoform = coverage_config["isoform"]
    molecules = sorted({m for (_, i, m) in rows if i == isoform})
    if not molecules:
        raise ValueError(f"No predictions for {isoform}.")
    statuses = {m: status_of(m) for m in molecules}
    groups = {
        m: "covered" if s in coverage_config["covered_statuses"] else "not_covered"
        for m, s in statuses.items()
    }
    counts = {name: sum(1 for g in groups.values() if g == name) for name in ("covered", "not_covered")}
    by_status = {}
    for status in statuses.values():
        by_status[status] = by_status.get(status, 0) + 1
    return {"isoform": isoform, "groups": groups, "statuses": statuses,
            "counts": counts, "status_counts": dict(sorted(by_status.items()))}


def _cell_st_rae(y_true, y_pred, low, high):
    """ST-RAE of one cell's rows (same definition as the inference script's)."""
    total = inference.soft_error(low, high, np.full_like(y_true, y_true.mean())).sum()
    if total <= 0:
        return None
    return float(inference.soft_error(low, high, y_pred).sum() / total)


def group_metrics(rows, isoform, members, seed_of, cohorts):
    by_cell, errors, soft = {}, [], []
    for (cell, row_isoform, molecule), (y, p, lo, hi) in rows.items():
        if row_isoform == isoform and molecule in members:
            by_cell.setdefault(cell, []).append((y, p, lo, hi))
            errors.append(p - y)
            soft.append(float(inference.soft_error(lo, hi, p)))
    if not errors:
        raise ValueError("A coverage group has no predictions.")
    cell_scores, by_seed = {}, {}
    for cell, items in by_cell.items():
        y, p, lo, hi = (np.array(column, dtype=np.float64) for column in zip(*items))
        score = _cell_st_rae(y, p, lo, hi)
        if score is not None:
            cell_scores[cell] = score
            by_seed.setdefault(str(seed_of[cell]), []).append(score)
    tails = {}
    for tail in ("over", "under"):
        chosen = set(cohorts[isoform][tail]) & members
        tail_errors = [
            p - y for (_, row_isoform, molecule), (y, p, _, _) in rows.items()
            if row_isoform == isoform and molecule in chosen
        ]
        tails[tail] = {
            "members": len(chosen), "rows": len(tail_errors),
            "mae": mean(abs(e) for e in tail_errors) if tail_errors else None,
        }
    return {
        "molecules": len(members), "rows": len(errors),
        "mae": mean(abs(e) for e in errors), "signed_error": mean(errors), "soft_mae": mean(soft),
        "st_rae_mean": mean(cell_scores.values()) if cell_scores else None,
        "st_rae_cells": len(cell_scores), "cells_without_score": len(by_cell) - len(cell_scores),
        "st_rae_by_seed": {s: mean(v) for s, v in sorted(by_seed.items())},
        "tails": tails,
    }


def _difference(a, b):
    return None if a is None or b is None else a - b


def group_report(predictions, coverage, runs, cohorts):
    """Per-arm group metrics and their differences from the unchanged arm. Descriptive only."""
    isoform = coverage["isoform"]
    seed_of = {c["cell_index"]: c["seed"] for c in runs["unchanged"]["cells"]}
    members = {
        name: {m for m, g in coverage["groups"].items() if g == name}
        for name in ("covered", "not_covered")
    }
    groups = {
        arm: {name: group_metrics(predictions[arm][0], isoform, members[name], seed_of, cohorts)
              for name in members}
        for arm in ARMS
    }
    differences = {}
    for arm in PENALTY_ARMS:
        differences[arm] = {}
        for name in members:
            mine, theirs = groups[arm][name], groups["unchanged"][name]
            differences[arm][name] = {
                key: _difference(mine[key], theirs[key]) for key in ("mae", "signed_error", "soft_mae", "st_rae_mean")
            }
            differences[arm][name]["st_rae_by_seed"] = {
                s: _difference(mine["st_rae_by_seed"].get(s), theirs["st_rae_by_seed"].get(s))
                for s in theirs["st_rae_by_seed"]
            }
            differences[arm][name]["over_tail_mae"] = _difference(
                mine["tails"]["over"]["mae"], theirs["tails"]["over"]["mae"])
            differences[arm][name]["under_tail_mae"] = _difference(
                mine["tails"]["under"]["mae"], theirs["tails"]["under"]["mae"])
    return {
        "isoform": isoform, "counts": coverage["counts"], "groups": groups,
        "differences_from_unchanged": differences,
        "note": "Descriptive only; not an adoption clause. Same saved predictions, no new training. "
                "Group sizes differ, so group means are not directly comparable between groups.",
    }


# ---------------------------------------------------------------------------- rule
def arm_clauses(unchanged, arm, config):
    isoforms = config["isoforms"]
    comparisons = [(i, str(s)) for i in isoforms for s in config["cell_seeds"]]
    worse = [
        [i, s] for i, s in comparisons
        if arm["by_isoform_seed"][i][s] > unchanged["by_isoform_seed"][i][s]
    ]
    over_failed = [
        i for i in isoforms if not arm["tails"][i]["over"]["mae"] < unchanged["tails"][i]["over"]["mae"]
    ]
    under_failed = [
        i for i in isoforms if arm["tails"][i]["under"]["mae"] > unchanged["tails"][i]["under"]["mae"]
    ]
    clauses = {
        "a_lower_mean_macro_st_rae": arm["O150"] < unchanged["O150"],
        "b_no_worse_enzyme_seed_mean": not worse,
        "c_lower_over_tail_mae": not over_failed,
        "d_no_higher_under_tail_mae": not under_failed,
    }
    return {
        "clauses": clauses,
        "passes": all(clauses.values()),
        "comparisons": len(comparisons),
        "worse_comparisons": worse,
        "over_tail_failed": over_failed,
        "under_tail_failed": under_failed,
    }


def decide(metrics, config):
    judged = {arm: arm_clauses(metrics["unchanged"], metrics[arm], config) for arm in PENALTY_ARMS}
    passing = [arm for arm in PENALTY_ARMS if judged[arm]["passes"]]
    if not passing:
        selected = "unchanged"
    else:
        best = min(metrics[arm]["O150"] for arm in passing)
        tied = [arm for arm in passing if metrics[arm]["O150"] == best]
        selected = config["tie_preference"] if config["tie_preference"] in tied else tied[0]
    return {"arms": judged, "selected": selected}


# --------------------------------------------------------------------------- runs
def load_arm(cells_dir, arm, config):
    run = base.load_run(cells_dir)
    checkpoints = {}
    for cell in run["cells"]:
        index = cell["cell_index"]
        path = Path(cells_dir) / f"cell_{index:03d}" / "result.json"
        result = json.loads(path.read_text())
        if result.get("arm") != config["arm_labels"][arm]:
            raise ValueError(f"{arm}: cell {index} carries arm label {result.get('arm')!r}")
        checkpoints[index] = result["checkpoint_sha256"]
    run["checkpoints"] = checkpoints
    return run


def evaluate(runs, predictions, cohorts, config, coverage):
    """runs: {arm: load_arm}; predictions: {arm: load_predictions}."""
    def layout(run):
        return {(c["cell_index"], c["repetition"], c["fold"], c["seed"], c["split_manifest_sha256"])
                for c in run["cells"]}

    reference = layout(runs["unchanged"])
    for arm in PENALTY_ARMS:
        if layout(runs[arm]) != reference:
            raise ValueError(f"{arm}: cell/partition layout differs from the unchanged arm")
    if len({runs[arm]["screen_checkpoint_sha256"] for arm in ARMS}) != 1:
        raise ValueError("The three arms were not fed by the same screening checkpoint.")

    def labels(rows):
        return {key: (v[0], v[2], v[3]) for key, v in rows.items()}

    for arm in ARMS:
        rows, checkpoints, _ = predictions[arm]
        if checkpoints != runs[arm]["checkpoints"]:
            raise ValueError(f"{arm}: predictions were not made from this run's best checkpoints.")
        if labels(rows) != labels(predictions["unchanged"][0]):
            raise ValueError(f"{arm}: prediction rows, labels or bounds differ from the unchanged arm.")
    if cohorts != build_cohorts(predictions["unchanged"][0], config):
        raise ValueError("Cohort file does not match cohorts rebuilt from the baseline predictions.")

    expected = {m for (_, i, m) in predictions["unchanged"][0] if i == coverage["isoform"]}
    if set(coverage["groups"]) != expected:
        raise ValueError("Coverage groups do not cover exactly the baseline's molecules.")
    metrics = {
        arm: {**cell_metrics(runs[arm], config), "tails": tail_metrics(predictions[arm][0], cohorts)}
        for arm in ARMS
    }
    difference = metrics["unchanged"]["O150"] - config["baseline_o150"]
    if abs(difference) > config["baseline_o150_tolerance"]:
        raise ValueError(
            f"Unchanged arm does not reproduce its recorded O150 (difference {difference})."
        )
    return {
        "rule": "D024 Amendments 1 and 2; development evidence on already-inspected partitions; "
                "no significance claim; tail membership comes from the baseline's own errors.",
        "metrics": metrics,
        "differences_from_unchanged": {
            arm: metrics[arm]["O150"] - metrics["unchanged"]["O150"] for arm in PENALTY_ARMS
        },
        "decision": decide(metrics, config),
        "baseline_reproduction_difference": difference,
        "coverage_groups": group_report(predictions, coverage, runs, cohorts),
    }


def write_new(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def named_paths(items, label):
    paths = {}
    for item in items:
        name, _, path = item.partition("=")
        if name in paths or name not in ARMS or not path:
            raise ValueError(f"Bad or duplicate {label}: {item}")
        paths[name] = path
    if set(paths) != set(ARMS):
        raise ValueError(f"{label} must name exactly {ARMS}.")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cohort = sub.add_parser("cohorts")
    for name in ("baseline-predictions", "sha256sums", "config", "output"):
        cohort.add_argument(f"--{name}", required=True)
    group = sub.add_parser("coverage")
    for name in ("baseline-predictions", "sha256sums", "config", "coverage-config",
                 "penalty-config", "screen-tar", "output"):
        group.add_argument(f"--{name}", required=True)
    judge = sub.add_parser("evaluate")
    for name in ("config", "cohorts", "cohorts-sha256", "coverage", "coverage-sha256", "output"):
        judge.add_argument(f"--{name}", required=True)
    judge.add_argument("--cells", action="append", required=True, metavar="ARM=CELLS_DIR")
    judge.add_argument("--predictions", action="append", required=True, metavar="ARM=DIR")
    args = parser.parse_args(argv)

    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    config = load_config(args.config)
    cells = range(len(config["cell_seeds"]) * config["cells_per_seed"])

    if args.command == "cohorts":
        rows, _, files = load_predictions(
            args.baseline_predictions, cells, sums=sums_table(args.sha256sums)
        )
        cohorts = build_cohorts(rows, config)
        write_new(args.output, {
            "cohorts": cohorts,
            "config_sha256": sha256_file(args.config),
            "sha256sums_sha256": sha256_file(args.sha256sums),
            "prediction_sha256_by_cell": {str(k): v for k, v in files.items()},
            "script_sha256": sha256_file(__file__),
        })
        for isoform, entry in cohorts.items():
            print(f"{isoform}: n={entry['n']}  cohort size={entry['size']}")
        print("wrote", args.output)
        print("cohorts sha256", sha256_file(args.output))
        return 0

    if args.command == "coverage":
        import t121_screening_penalty as penalty

        rows, _, files = load_predictions(
            args.baseline_predictions, cells, sums=sums_table(args.sha256sums)
        )
        coverage_config = load_coverage_config(args.coverage_config)
        arm = penalty.PenaltyArm(
            json.loads(Path(args.penalty_config).read_bytes()), sha256_file(args.penalty_config)
        )
        screen = penalty.load_screen_tar(args.screen_tar, arm)
        isoform = coverage_config["isoform"]
        built = build_coverage(
            rows, coverage_config, lambda m: penalty.lookup(screen, arm, m, isoform)[0]
        )
        write_new(args.output, {
            **built,
            "coverage_config_sha256": sha256_file(args.coverage_config),
            "penalty_config_sha256": arm.config_sha256,
            "screen_input_sha256": screen["sha256"],
            "sha256sums_sha256": sha256_file(args.sha256sums),
            "prediction_sha256_by_cell": {str(k): v for k, v in files.items()},
            "script_sha256": sha256_file(__file__),
        })
        print(f"{isoform}: molecules {len(built['groups'])}  {built['counts']}  by status {built['status_counts']}")
        print("wrote", args.output)
        print("coverage sha256", sha256_file(args.output))
        return 0

    if sha256_file(args.cohorts) != args.cohorts_sha256:
        raise ValueError("Cohort file does not match its pinned SHA-256.")
    if sha256_file(args.coverage) != args.coverage_sha256:
        raise ValueError("Coverage file does not match its pinned SHA-256.")
    frozen = json.loads(Path(args.cohorts).read_text())
    coverage = json.loads(Path(args.coverage).read_text())
    cell_dirs = named_paths(args.cells, "--cells")
    prediction_dirs = named_paths(args.predictions, "--predictions")
    runs = {arm: load_arm(cell_dirs[arm], arm, config) for arm in ARMS}
    predictions = {arm: load_predictions(prediction_dirs[arm], cells) for arm in ARMS}
    if frozen["prediction_sha256_by_cell"] != {
        str(k): v for k, v in predictions["unchanged"][2].items()
    }:
        raise ValueError("Baseline prediction files differ from those the cohorts were built from.")
    if coverage["prediction_sha256_by_cell"] != frozen["prediction_sha256_by_cell"]:
        raise ValueError("Coverage groups were built from different baseline predictions.")
    report = evaluate(runs, predictions, frozen["cohorts"], config, coverage)
    report["config_sha256"] = sha256_file(args.config)
    report["cohorts_sha256"] = args.cohorts_sha256
    report["coverage_sha256"] = args.coverage_sha256
    report["script_sha256"] = sha256_file(__file__)
    report["d021_aggregate_sha256"] = sha256_file(base.__file__)
    report["inputs"] = {
        arm: {
            "cells_dir": cell_dirs[arm],
            "predictions_dir": prediction_dirs[arm],
            "result_sha256_by_cell": {str(c["cell_index"]): c["result_sha256"] for c in runs[arm]["cells"]},
            "prediction_sha256_by_cell": {str(k): v for k, v in predictions[arm][2].items()},
        }
        for arm in ARMS
    }
    write_new(args.output, report)

    for arm in ARMS:
        entry = report["metrics"][arm]
        print(f"{arm:<10} O150 {entry['O150']:.9f}")
        for isoform in config["isoforms"]:
            tails = entry["tails"][isoform]
            print(f"  {isoform}  ST-RAE {entry['per_isoform'][isoform]:.6f}  "
                  f"over-tail MAE {tails['over']['mae']:.6f}  under-tail MAE {tails['under']['mae']:.6f}")
    for arm, judged in report["decision"]["arms"].items():
        print(f"{arm}: passes={judged['passes']}  {judged['clauses']}")
    print("selected:", report["decision"]["selected"])
    groups = report["coverage_groups"]
    print(f"{groups['isoform']} by screening coverage (descriptive; not an adoption clause): {groups['counts']}")
    for arm in ARMS:
        for name, entry in groups["groups"][arm].items():
            score = entry["st_rae_mean"]
            print(f"  {arm:<10} {name:<12} molecules {entry['molecules']:>5}  MAE {entry['mae']:.6f}  "
                  f"signed {entry['signed_error']:+.6f}  mean cell ST-RAE "
                  f"{'n/a' if score is None else format(score, '.6f')}")
    for arm, by_group in groups["differences_from_unchanged"].items():
        for name, entry in by_group.items():
            seeds = " ".join(
                f"{s}:{'n/a' if v is None else format(v, '+.5f')}" for s, v in entry["st_rae_by_seed"].items()
            )
            print(f"  {arm:<5} - unchanged, {name:<12} MAE {entry['mae']:+.6f}  ST-RAE "
                  f"{'n/a' if entry['st_rae_mean'] is None else format(entry['st_rae_mean'], '+.6f')}  by seed {seeds}")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
