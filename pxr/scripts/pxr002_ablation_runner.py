"""Tripwire PXR-002 auxiliary-off ablation (approved 2026-11-18 as proposed).

Trains the stage B selected configuration (top of stage_b_ranking.json) at the three seeds
20261117, 20261118 and 20261119 with the auxiliary loss coefficients set to 0 and the
primary coefficient 1. Cohort, artifact, descriptors, batches, optimizer and stopping rule
are identical to the multitask runs; only the loss changes. Chemprop still divides by the
count of all observed labels, so the primary loss is scaled by n_primary / n_observed;
the manifest records aux_off and the weight vector.

Interpretation rule (descriptive; no statistical threshold): with the three-seed mean and
range of each variant, multitask is "supported" only if its mean is lower AND its range
does not overlap the ablation's; otherwise "not_distinguishable" (a mirrored
"ablation_better_ranges_disjoint" label is printed when the ablation is clearly lower).
The multitask side of the comparison reads the existing stage A (seed 20261117) and
stage B (seeds 20261118, 20261119) metrics. Because the stage A seed was the best of 36,
the summary also reports the fresh-seed-only (20261118, 20261119) means for both variants.

No holdout or official-test labels are used. This authorizes no final ensemble or
holdout evaluation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts.pxr002_grid import COMMON_SEED, STAGE_B_SEEDS, build_grid, read_stage_a  # noqa: E402

TRAIN_SCRIPT = REPO_ROOT / "pxr/scripts/pxr002_train.py"
FRESH_SEEDS = tuple(s for s in STAGE_B_SEEDS if s != COMMON_SEED)


def selected_configuration(stage_b_root: Path) -> str:
    ranking = json.loads((Path(stage_b_root) / "stage_b_ranking.json").read_text())["ranking_by_mean_primary_validation_mae"]
    return ranking[0][0]


def multitask_mae_by_seed(name: str, stage_a_root: Path, stage_b_root: Path) -> dict[int, float]:
    out = {COMMON_SEED: read_stage_a(stage_a_root)[name]}
    for seed in FRESH_SEEDS:
        metrics = json.loads((Path(stage_b_root) / f"{name}_seed{seed}" / "metrics.json").read_text())
        out[seed] = metrics["best_primary_validation_mae"]
    return out


def build_command(config, seed: int, artifact_dir: Path, out_dir: Path, accelerator: str) -> list[str]:
    return [
        sys.executable, "-I", str(TRAIN_SCRIPT),
        "--artifact-dir", str(artifact_dir), "--out-dir", str(out_dir),
        "--accelerator", accelerator, "--seed", str(seed),
        "--hidden-size", str(config.hidden_size), "--depth", str(config.depth),
        "--ffn-layers", str(config.ffn_layers), "--dropout", str(config.dropout),
        "--aux-off",
    ]


def _stats(values: list[float]) -> dict:
    return {"values": values, "mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def summarize(multitask: dict[int, float], ablation: dict[int, float]) -> dict:
    """Descriptive comparison under the approved interpretation rule."""
    if set(multitask) != set(ablation):
        raise ValueError("multitask and ablation must cover the same seeds")
    seeds = sorted(multitask)
    m, a = _stats([multitask[s] for s in seeds]), _stats([ablation[s] for s in seeds])
    overlap = not (m["max"] < a["min"] or a["max"] < m["min"])
    if m["mean"] < a["mean"] and not overlap:
        outcome = "multitask_supported"
    elif a["mean"] < m["mean"] and not overlap:
        outcome = "ablation_better_ranges_disjoint"
    else:
        outcome = "not_distinguishable"
    fresh = [s for s in seeds if s in FRESH_SEEDS]
    fresh_summary = None
    if fresh:
        fm, fa = _stats([multitask[s] for s in fresh]), _stats([ablation[s] for s in fresh])
        fresh_summary = {"seeds": fresh, "multitask_mean": fm["mean"], "ablation_mean": fa["mean"],
                         "mean_difference_ablation_minus_multitask": fa["mean"] - fm["mean"]}
    return {
        "seeds": seeds, "multitask": m, "ablation": a,
        "mean_difference_ablation_minus_multitask": a["mean"] - m["mean"],
        "ranges_overlap": overlap, "multitask_mean_lower": m["mean"] < a["mean"],
        "outcome": outcome, "fresh_seed_comparison": fresh_summary,
    }


def run_ablation(artifact_dir: Path, stage_a_root: Path, stage_b_root: Path, out_root: Path, accelerator: str) -> int:
    name = selected_configuration(stage_b_root)
    multitask = multitask_mae_by_seed(name, stage_a_root, stage_b_root)
    config = {c.name: c for c in build_grid()}[name]
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=False)  # refuses to reuse an existing output root
    print(f"=== ablation of {name}: multitask MAE by seed {multitask}", flush=True)

    ablation: dict[int, float] = {}
    failures = 0
    for i, seed in enumerate(sorted(multitask), start=1):
        out_dir = out_root / f"{name}_auxoff_seed{seed}"
        t0 = time.time()
        print(f"=== [{i}/3] {name} aux-off seed {seed} start {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
        rc = subprocess.run(build_command(config, seed, artifact_dir, out_dir, accelerator)).returncode
        mae = epochs = None
        if rc == 0 and (out_dir / "metrics.json").exists():
            metrics = json.loads((out_dir / "metrics.json").read_text())
            mae, epochs = metrics["best_primary_validation_mae"], metrics["epochs_run"]
            ablation[seed] = mae
        elif rc == 0:
            rc = 98
        with open(out_root / "ablation_status.jsonl", "a") as handle:
            handle.write(json.dumps({"configuration": name, "aux_off": True, "seed": seed, "returncode": rc,
                                     "seconds": round(time.time() - t0, 1), "epochs_run": epochs,
                                     "best_primary_validation_mae": mae}) + "\n")
        failures += rc != 0
        print(f"=== [{i}/3] aux-off seed {seed} rc={rc} epochs={epochs} mae={mae}", flush=True)

    if failures:
        print(f"=== ablation finished with {failures} failed run(s); no summary written", flush=True)
        return 1
    summary = {"configuration": name, **summarize(multitask, ablation)}
    (out_root / "ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("=== ablation summary (descriptive; rule: multitask supported only if mean lower AND ranges disjoint):", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--stage-a-root", required=True, type=Path)
    p.add_argument("--stage-b-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu", "mps"])
    a = p.parse_args()
    sys.exit(run_ablation(a.artifact_dir, a.stage_a_root, a.stage_b_root, a.out_root, a.accelerator))


if __name__ == "__main__":
    main()
