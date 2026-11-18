"""Tripwire PXR-002 stage B runner (D018): top three stage-A configurations across three seeds.

Reads the stage A results (metrics.json files, seed 20261117), selects the three
configurations with the lowest primary-head validation MAE, and runs only the seeds not
already run (the common seed's result is reused as the first of the three seeds). Each run
is its own subprocess of pxr002_train.py in a new output directory; nothing resumes. At
the end it writes stage_b_ranking.json with each configuration's mean primary-head
validation MAE over the three seeds. Selection is by that mean and by validation only.

The seed values (20261117 common, 20261118, 20261119) are the project's date-shaped
proposal. This authorizes no auxiliary-off ablation, final ensemble or holdout/test
evaluation.
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

from pxr.scripts.pxr002_grid import (  # noqa: E402
    COMMON_SEED, STAGE_B_SEEDS, build_grid, rank_by_mae, read_stage_a, select_top, stage_b_plan,
)

TRAIN_SCRIPT = REPO_ROOT / "pxr/scripts/pxr002_train.py"


def plan_from_stage_a(stage_a_root: Path) -> dict:
    stage_a = read_stage_a(stage_a_root)
    if len(stage_a) != 36:
        raise RuntimeError(f"expected 36 stage A results at {stage_a_root}, found {len(stage_a)}")
    top = select_top(stage_a)
    return {"top": top, "stage_a_mae": {n: stage_a[n] for n in top}, "extra_runs": stage_b_plan(top)}


def build_command(config, seed: int, artifact_dir: Path, out_dir: Path, accelerator: str) -> list[str]:
    return [
        sys.executable, "-I", str(TRAIN_SCRIPT),
        "--artifact-dir", str(artifact_dir), "--out-dir", str(out_dir),
        "--accelerator", accelerator, "--seed", str(seed),
        "--hidden-size", str(config.hidden_size), "--depth", str(config.depth),
        "--ffn-layers", str(config.ffn_layers), "--dropout", str(config.dropout),
    ]


def aggregate(plan: dict, new_mae: dict[tuple[str, int], float]) -> list[tuple[str, float]]:
    """Mean MAE per top configuration: the stage A (common-seed) value plus the new seeds."""
    results = {}
    for name in plan["top"]:
        values = [plan["stage_a_mae"][name]] + [new_mae[(name, s)] for s in STAGE_B_SEEDS if s != COMMON_SEED]
        if len(values) != len(STAGE_B_SEEDS):
            raise RuntimeError(f"{name} has {len(values)} results; expected {len(STAGE_B_SEEDS)}")
        results[name] = values
    return rank_by_mae(results)


def run_stage_b(artifact_dir: Path, stage_a_root: Path, out_root: Path, accelerator: str) -> int:
    plan = plan_from_stage_a(stage_a_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=False)  # refuses to reuse an existing output root
    (out_root / "stage_b_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"=== stage B plan: top {plan['top']}; {len(plan['extra_runs'])} extra runs", flush=True)

    by_name = {c.name: c for c in build_grid()}
    new_mae: dict[tuple[str, int], float] = {}
    failures = 0
    for i, run in enumerate(plan["extra_runs"], start=1):
        name, seed = run["configuration"], run["seed"]
        out_dir = out_root / f"{name}_seed{seed}"
        t0 = time.time()
        print(f"=== [{i}/{len(plan['extra_runs'])}] {name} seed {seed} start {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
        rc = subprocess.run(build_command(by_name[name], seed, artifact_dir, out_dir, accelerator)).returncode
        metrics_path = out_dir / "metrics.json"
        mae = epochs = None
        if rc == 0 and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            mae, epochs = metrics["best_primary_validation_mae"], metrics["epochs_run"]
            new_mae[(name, seed)] = mae
        elif rc == 0:
            rc = 98
        with open(out_root / "stage_b_status.jsonl", "a") as handle:
            handle.write(json.dumps({"configuration": name, "seed": seed, "returncode": rc,
                                     "seconds": round(time.time() - t0, 1), "epochs_run": epochs,
                                     "best_primary_validation_mae": mae}) + "\n")
        failures += rc != 0
        print(f"=== [{i}/{len(plan['extra_runs'])}] {name} seed {seed} rc={rc} epochs={epochs} mae={mae}", flush=True)

    if failures:
        print(f"=== stage B finished with {failures} failed run(s); no ranking written", flush=True)
        return 1
    ranking = aggregate(plan, new_mae)
    (out_root / "stage_b_ranking.json").write_text(json.dumps(
        {"ranking_by_mean_primary_validation_mae": ranking, "seeds": list(STAGE_B_SEEDS)}, indent=2) + "\n")
    print("=== stage B ranking (mean primary-head validation MAE over 3 seeds):", flush=True)
    for name, mean in ranking:
        print(f"    {name}: {mean:.5f}", flush=True)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--stage-a-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu", "mps"])
    a = p.parse_args()
    sys.exit(run_stage_b(a.artifact_dir, a.stage_a_root, a.out_root, a.accelerator))


if __name__ == "__main__":
    main()
