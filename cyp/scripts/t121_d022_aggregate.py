"""D022 Amendment 1 acceptance: CheMeleon versus descriptor-free updated T121 (arm R).

O150(arm, s) = mean macro ST-RAE over the 150 cells fed by screening
checkpoint s (identical computation and cell checks to D021, reused from
t121_d021_aggregate). For the same three screening seeds,
    Delta_s = O150(C, s) - O150(R, s);  Delta-bar = mean of the three.
CheMeleon is preferred only if Delta-bar < 0 strictly (lower mean macro
ST-RAE). Paired seed differences and per-isoform results (per seed and
averaged) are reported beside it. No significance claim: replicates share
partitions and are not independent. No effect floor, no bootstrap.

Arm R runs r1..r3 are the existing D021 comparator runs (r1 recorded before
any D021 run; r2 and r3 from D021), read unchanged.

Usage:
    python -I t121_d022_aggregate.py \
        --run r1=<cells_dir> --run r2=<cells_dir> --run r3=<cells_dir> \
        --run c1=<cells_dir> --run c2=<cells_dir> --run c3=<cells_dir> \
        --output NEW.json
Run names: r1..r3 (arm R, screening replicate) and c1..c3 (CheMeleon). The
output path must not exist.
"""
from pathlib import Path
import argparse
import json
import sys

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` omits the script dir
import t121_d021_aggregate as base  # noqa: E402

RUN_NAMES = ("r1", "r2", "r3", "c1", "c2", "c3")
SEEDS = (1, 2, 3)
TASKS = base.TASKS
mean = base.mean


def aggregate(runs):
    """runs: {name: base.load_run output}. Returns the report dict."""
    if set(runs) != set(RUN_NAMES):
        raise ValueError(f"Runs must be exactly {RUN_NAMES}; got {sorted(runs)}")

    def layout(run):
        return {(c["cell_index"], c["repetition"], c["fold"], c["seed"], c["split_manifest_sha256"])
                for c in run["cells"]}

    reference = layout(runs["r1"])
    for name in RUN_NAMES:
        if layout(runs[name]) != reference:
            raise ValueError(f"{name}: cell/partition layout differs from r1")
    screens = [runs[name]["screen_checkpoint_sha256"] for name in RUN_NAMES]
    if len(set(screens)) != len(screens):
        raise ValueError("Two runs share a screening checkpoint hash.")

    summary = {name: base.summarize(runs[name]) for name in RUN_NAMES}
    deltas = [summary[f"c{s}"]["O150"] - summary[f"r{s}"]["O150"] for s in SEEDS]
    delta_bar = mean(deltas)
    per_isoform_by_seed = {
        f"s{s}": {t: summary[f"c{s}"]["per_isoform"][t] - summary[f"r{s}"]["per_isoform"][t] for t in TASKS}
        for s in SEEDS
    }
    r1_difference = summary["r1"]["O150"] - base.R1_RECORDED_O150
    return {
        "rule": "D022 Amendment 1 item 5: CheMeleon preferred iff mean Delta_s < 0 strictly (lower mean macro ST-RAE vs arm R, same three screening seeds, matched O150); paired seed differences and per-isoform results reported; no significance claim; replicates share partitions and are not independent.",
        "runs": summary,
        "delta_s": {f"s{s}": deltas[s - 1] for s in SEEDS},
        "delta_bar": delta_bar,
        "chemeleon_preferred": bool(delta_bar < 0),
        "O_bar_R": mean(summary[f"r{s}"]["O150"] for s in SEEDS),
        "O_bar_C": mean(summary[f"c{s}"]["O150"] for s in SEEDS),
        "per_isoform_delta_s": per_isoform_by_seed,
        "per_isoform_delta_bar": {
            t: mean(per_isoform_by_seed[f"s{s}"][t] for s in SEEDS) for t in TASKS
        },
        "r1_reproduction": {
            "recorded_O150": base.R1_RECORDED_O150,
            "recomputed_O150": summary["r1"]["O150"],
            "difference": r1_difference,
            "matches": abs(r1_difference) <= base.R1_TOLERANCE,
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
    runs = {name: base.load_run(path) for name, path in paths.items()}
    report = aggregate(runs)
    if not report["r1_reproduction"]["matches"]:
        raise ValueError(
            "R replicate 1 does not reproduce its recorded O150: "
            f"{report['r1_reproduction']}. Stopping; no report written."
        )
    report["script_sha256"] = base.sha256_file(__file__)
    report["d021_aggregate_sha256"] = base.sha256_file(base.__file__)
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
        print(f"Delta_{key}  {value:+.9f}   (CheMeleon - R)")
    print(f"Delta-bar   {report['delta_bar']:+.9f}   CheMeleon preferred (strict < 0): {report['chemeleon_preferred']}")
    print(f"O_bar(R) {report['O_bar_R']:.9f}   O_bar(C) {report['O_bar_C']:.9f}")
    for task in TASKS:
        seeds = "  ".join(f"{report['per_isoform_delta_s'][f's{s}'][task]:+.6f}" for s in SEEDS)
        print(f"  {task}: Delta-bar {report['per_isoform_delta_bar'][task]:+.9f}  per seed {seeds}")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
