"""D021 item 6 aggregation: O150 per (arm, screening seed), Delta_s, Delta-bar.

O150(arm, s) = mean macro ST-RAE over the 150 cells fed by screening
checkpoint s. Delta_s = O150(D, s) - O150(R, s); Delta-bar = mean of the three
Delta_s. D is favoured iff Delta-bar < 0 strictly. No effect-size floor, no
bootstrap, no significance claim; replicates share partitions and are not
independent.

Reads only cell result.json files, each checked against its COMPLETED.json
(status completed, result_sha256 equal to the file's actual hash). Stops on any
missing, incomplete, duplicated or inconsistent cell. Runs no model code.

Usage:
    python -I t121_d021_aggregate.py \
        --run r1=<cells_dir> --run r2=<cells_dir> --run r3=<cells_dir> \
        --run d1=<cells_dir> --run d2=<cells_dir> --run d3=<cells_dir> \
        --output NEW.json
Run names are r1..r3 and d1..d3 (arm letter, screening replicate). The output
path must not exist.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

RUN_NAMES = ("r1", "r2", "r3", "d1", "d2", "d3")
N_CELLS = 150
R1_RECORDED_O150 = 0.6769540372619428  # D021: R replicate 1, recorded before D runs
R1_TOLERANCE = 1e-12
TASKS = (
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def load_cell(cell_dir):
    cell_dir = Path(cell_dir)
    completed_path = cell_dir / "COMPLETED.json"
    if not completed_path.is_file():
        raise ValueError(f"Cell not completed: {cell_dir}")
    if (cell_dir / "INCOMPLETE.json").exists():
        raise ValueError(f"Cell marked incomplete: {cell_dir}")
    completed = json.loads(completed_path.read_text())
    if completed.get("status") != "completed":
        raise ValueError(f"Cell status is not completed: {cell_dir}")
    result_path = cell_dir / "result.json"
    actual = sha256_file(result_path)
    if actual != completed.get("result_sha256"):
        raise ValueError(f"result.json hash differs from COMPLETED.json: {cell_dir}")
    result = json.loads(result_path.read_text())
    if result.get("status") != "completed":
        raise ValueError(f"result.json status is not completed: {cell_dir}")
    per_task = result["best_validation"]["per_task"]
    if tuple(sorted(per_task)) != tuple(sorted(TASKS)):
        raise ValueError(f"Unexpected task set in {cell_dir}: {sorted(per_task)}")
    macro = result["macro_st_rae"]
    if not (isinstance(macro, float) and math.isfinite(macro)):
        raise ValueError(f"Non-finite macro ST-RAE in {cell_dir}")
    if abs(macro - mean(per_task[t] for t in TASKS)) > 1e-9:
        raise ValueError(f"macro_st_rae is not the mean of its per-task values: {cell_dir}")
    if macro != result["best_validation"]["macro_st_rae"]:
        raise ValueError(f"macro_st_rae differs from best_validation: {cell_dir}")
    return {
        "cell_index": result["cell_index"],
        "repetition": result["repetition"],
        "fold": result["fold"],
        "seed": result["seed"],
        "split_manifest_sha256": result["split_manifest_sha256"],
        "screen_checkpoint_sha256": result["screen_checkpoint_sha256"],
        "macro_st_rae": macro,
        "per_task": {t: per_task[t] for t in TASKS},
        "result_sha256": actual,
    }


def load_run(cells_dir):
    cells_dir = Path(cells_dir)
    directories = sorted(p for p in cells_dir.iterdir() if p.is_dir() and p.name.startswith("cell_"))
    if len(directories) != N_CELLS:
        raise ValueError(f"{cells_dir}: expected {N_CELLS} cell directories, found {len(directories)}")
    cells = [load_cell(p) for p in directories]
    if sorted(c["cell_index"] for c in cells) != list(range(N_CELLS)):
        raise ValueError(f"{cells_dir}: cell indices are not exactly 0..{N_CELLS - 1}")
    # 150 cells = 5 repetitions x 10 folds x 3 cell seeds (cell_plan.json, read on CoreHPC).
    if len({(c["repetition"], c["fold"], c["seed"]) for c in cells}) != N_CELLS:
        raise ValueError(f"{cells_dir}: duplicate (repetition, fold, seed) cells")
    screens = {c["screen_checkpoint_sha256"] for c in cells}
    if len(screens) != 1:
        raise ValueError(f"{cells_dir}: cells were not all fed by one screening checkpoint")
    return {"cells": sorted(cells, key=lambda c: c["cell_index"]), "screen_checkpoint_sha256": screens.pop()}


def summarize(run):
    cells = run["cells"]
    return {
        "screen_checkpoint_sha256": run["screen_checkpoint_sha256"],
        "n_cells": len(cells),
        "O150": mean(c["macro_st_rae"] for c in cells),
        "per_isoform": {t: mean(c["per_task"][t] for c in cells) for t in TASKS},
    }


def aggregate(runs):
    """runs: {name: load_run output}. Returns the report dict."""
    if set(runs) != set(RUN_NAMES):
        raise ValueError(f"Runs must be exactly {RUN_NAMES}; got {sorted(runs)}")
    # Shared partitions: every run must score the same (repetition, fold) splits.
    reference = {(c["cell_index"], c["repetition"], c["fold"], c["seed"], c["split_manifest_sha256"])
                 for c in runs["r1"]["cells"]}
    for name, run in runs.items():
        layout = {(c["cell_index"], c["repetition"], c["fold"], c["seed"], c["split_manifest_sha256"])
                  for c in run["cells"]}
        if layout != reference:
            raise ValueError(f"{name}: cell/partition layout differs from r1")
    screens = [runs[name]["screen_checkpoint_sha256"] for name in RUN_NAMES]
    if len(set(screens)) != len(screens):
        raise ValueError("Two runs share a screening checkpoint hash.")

    summary = {name: summarize(runs[name]) for name in RUN_NAMES}
    deltas = [summary[f"d{s}"]["O150"] - summary[f"r{s}"]["O150"] for s in (1, 2, 3)]
    delta_bar = mean(deltas)
    per_isoform_delta = {
        t: mean(summary[f"d{s}"]["per_isoform"][t] - summary[f"r{s}"]["per_isoform"][t] for s in (1, 2, 3))
        for t in TASKS
    }
    r1_difference = summary["r1"]["O150"] - R1_RECORDED_O150
    return {
        "rule": "D021 item 6: D favoured iff mean Delta_s < 0 strictly; no effect-size floor, no bootstrap, no significance claim; replicates share partitions and are not independent.",
        "runs": summary,
        "delta_s": {f"s{s}": deltas[s - 1] for s in (1, 2, 3)},
        "delta_bar": delta_bar,
        "d_favoured": bool(delta_bar < 0),
        "O_bar_R": mean(summary[f"r{s}"]["O150"] for s in (1, 2, 3)),
        "O_bar_D": mean(summary[f"d{s}"]["O150"] for s in (1, 2, 3)),
        "per_isoform_delta_bar": per_isoform_delta,
        "r1_reproduction": {
            "recorded_O150": R1_RECORDED_O150,
            "recomputed_O150": summary["r1"]["O150"],
            "difference": r1_difference,
            "matches": abs(r1_difference) <= R1_TOLERANCE,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="NAME=CELLS_DIR")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")

    paths = {}
    for item in args.run:
        name, _, path = item.partition("=")
        if name in paths or name not in RUN_NAMES or not path:
            raise ValueError(f"Bad or duplicate --run: {item}")
        paths[name] = path
    runs = {name: load_run(path) for name, path in paths.items()}
    report = aggregate(runs)
    if not report["r1_reproduction"]["matches"]:
        raise ValueError(
            "R replicate 1 does not reproduce its recorded O150: "
            f"{report['r1_reproduction']}. Stopping; no report written."
        )
    report["script_sha256"] = sha256_file(__file__)
    report["inputs"] = {
        name: {"cells_dir": paths[name],
               "result_sha256_by_cell": {str(c["cell_index"]): c["result_sha256"] for c in runs[name]["cells"]}}
        for name in RUN_NAMES
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    print(f"{'run':<4} {'O150':>12}  screen_checkpoint_sha256")
    for name in RUN_NAMES:
        row = report["runs"][name]
        print(f"{name:<4} {row['O150']:>12.9f}  {row['screen_checkpoint_sha256'][:16]}")
    for key, value in report["delta_s"].items():
        print(f"Delta_{key}  {value:+.9f}")
    print(f"Delta-bar   {report['delta_bar']:+.9f}   D favoured (strict < 0): {report['d_favoured']}")
    print(f"O_bar(R) {report['O_bar_R']:.9f}   O_bar(D) {report['O_bar_D']:.9f}")
    for task, value in report["per_isoform_delta_bar"].items():
        print(f"  per-isoform Delta-bar {task}: {value:+.9f}")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
