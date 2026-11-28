"""Plots of the tables written by t121_penalty_prediction_diagnostics.py.

Reads only those tables (hash-checked against run_metadata.json) and writes only
into a NEW output directory. Needs matplotlib, numpy and pandas; no project code.

    change_vs_unchanged.png  molecule-level prediction change (arm - unchanged)
                             against the unchanged prediction
    concentration.png        cumulative share of total error improvement and
                             deterioration, molecules ordered by contribution
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, MUTED_INK, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Fixed assignments. The two highlight hues pass the all-pairs colour-vision
# check on this surface; the remaining molecules are context grey. Marker shape
# and line style repeat the identity without colour.
POINTS = (
    ("remaining", "remaining", "#b0afa8", "o", 8, 0.45),
    ("over_tail", "fixed over-prediction tail", "#eb6834", "^", 18, 0.85),
    ("under_tail", "fixed under-prediction tail", "#2a78d6", "v", 18, 0.85),
)
LINES = (("improvement", "#2a78d6", "-"), ("deterioration", "#eb6834", "--"))


def checked_table(directory, metadata, name):
    path = directory / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["outputs"][name]:
        raise ValueError(f"{name} differs from the hash in run_metadata.json")
    return pd.read_csv(path)


def grid(rows, columns, title, note):
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.1 * rows + 1.5), squeeze=False,
        facecolor=SURFACE, sharey="row",
    )
    for axis in axes.flat:
        axis.set_facecolor(SURFACE)
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED_INK, labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
    height = figure.get_size_inches()[1]
    figure.suptitle(title, color=INK, fontsize=12, y=1 - 0.2 / height)
    figure.text(0.5, 0.12 / height, note, ha="center", va="bottom", color=MUTED_INK, fontsize=8)
    return figure, axes


def finish(figure, handles, path):
    height = figure.get_size_inches()[1]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.4 / height),
                  ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout(rect=(0, 0.6 / height, 1, 1 - 0.75 / height))
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def change_plot(molecules, arms, roles, isoforms, path):
    figure, axes = grid(
        len(isoforms), len(arms),
        "Prediction change against the unchanged prediction (one point per molecule)",
        "Solid line: no change. Dashed line: mean change. A downward slope can arise from run-to-run noise "
        "alone, because the unchanged prediction enters both axes.\nTails were selected from the unchanged "
        "arm's own errors. Descriptive only.",
    )
    for r, isoform in enumerate(isoforms):
        rows = molecules[molecules["isoform"] == isoform]
        for c, arm in enumerate(arms):
            axis = axes[r][c]
            for group, _, colour, marker, size, alpha in POINTS:
                chosen = rows[rows["group"] == group]
                axis.scatter(chosen["pred_unchanged"], chosen[f"change_{arm}"], s=size, c=colour,
                             marker=marker, alpha=alpha, linewidths=0)
            mean = float(rows[f"change_{arm}"].mean())
            axis.axhline(0.0, color=INK, linewidth=0.9)
            axis.axhline(mean, color=INK, linewidth=0.9, linestyle="--")
            axis.set_title(f"{isoform}  {arm} ({roles[arm]})  mean {mean:+.3f}", color=INK, fontsize=9)
            if r == len(isoforms) - 1:
                axis.set_xlabel("unchanged prediction (pIC50)", color=MUTED_INK, fontsize=9)
            if c == 0:
                axis.set_ylabel("arm - unchanged (pIC50)", color=MUTED_INK, fontsize=9)
    handles = [
        plt.Line2D([], [], linestyle="none", marker=marker, color=colour, markersize=7, label=label)
        for _, label, colour, marker, _, _ in POINTS
    ]
    finish(figure, handles, path)


def concentration_plot(curves, arms, roles, isoforms, path):
    figure, axes = grid(
        len(isoforms), len(arms),
        "Concentration of absolute-error improvement and deterioration",
        "Each curve is ordered by its own molecules' contributions and ends at 100% of its own total, at the "
        "share of all molecules that moved in that direction.\nThe two totals differ in size. Descriptive "
        "only; no cutoff is defined.",
    )
    for r, isoform in enumerate(isoforms):
        for c, arm in enumerate(arms):
            axis = axes[r][c]
            totals = []
            for direction, colour, style in LINES:
                chosen = curves[(curves["isoform"] == isoform) & (curves["arm"] == arm)
                                & (curves["direction"] == direction)].sort_values("rank")
                if chosen.empty:
                    totals.append(f"{direction} 0")
                    continue
                axis.plot(np.r_[0.0, chosen["share_of_all_molecules"]],
                          np.r_[0.0, chosen["cumulative_share"]],
                          color=colour, linestyle=style, linewidth=2)
                totals.append(f"{direction} {chosen['contribution'].sum():.1f}")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.02)
            axis.set_title(f"{isoform}  {arm} ({roles[arm]})\ntotals: {', '.join(totals)}",
                           color=INK, fontsize=8.5)
            if r == len(isoforms) - 1:
                axis.set_xlabel("share of all molecules", color=MUTED_INK, fontsize=9)
            if c == 0:
                axis.set_ylabel("cumulative share of the total", color=MUTED_INK, fontsize=9)
    handles = [plt.Line2D([], [], color=colour, linestyle=style, linewidth=2, label=direction)
               for direction, colour, style in LINES]
    finish(figure, handles, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="must not exist")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {args.output_dir}")
    metadata = json.loads((args.tables / "run_metadata.json").read_text())
    molecules = checked_table(args.tables, metadata, "molecule_level.csv")
    curves = checked_table(args.tables, metadata, "concentration.csv")
    roles, isoforms = metadata["arms"], metadata["isoforms"]
    # candidates first, then context arms
    arms = sorted(roles, key=lambda a: (roles[a] != "candidate", a))
    args.output_dir.mkdir(parents=True)
    change_plot(molecules, arms, roles, isoforms, args.output_dir / "change_vs_unchanged.png")
    concentration_plot(curves, arms, roles, isoforms, args.output_dir / "concentration.png")
    for name, reason in metadata["stopped"].items():
        print(f"not plotted, stopped earlier: {name}: {reason}")
    print("wrote", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
