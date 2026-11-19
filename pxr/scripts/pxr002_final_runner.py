"""Tripwire PXR-002 final training (approved protocol, 2026-11-18). Trains; never evaluates.

Trains the stage B selected configuration (top of stage_b_ranking.json) on the train split
only (the v2 cohort; no refit on train+validation) at five fresh seeds, for two variants:
  multitask  primary result (calibrated 65:35 auxiliary coefficients)
  aux_off    secondary, reported alongside (auxiliary coefficients 0)
Each run is its own subprocess of pxr002_train.py in a new output directory; checkpoints
are the validation-selected best.ckpt files. Nothing resumes. No holdout or official-test
labels are loaded here; prediction and evaluation are separate steps.
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

from pxr.scripts.pxr002_ablation_runner import selected_configuration  # noqa: E402
from pxr.scripts.pxr002_grid import build_grid  # noqa: E402

TRAIN_SCRIPT = REPO_ROOT / "pxr/scripts/pxr002_train.py"
FINAL_SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)
VARIANTS = ("multitask", "aux_off")


def run_name(variant: str, seed: int) -> str:
    return f"{variant}_seed{seed}"


def build_command(config, seed: int, variant: str, artifact_dir: Path, out_dir: Path, accelerator: str) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    cmd = [
        sys.executable, "-I", str(TRAIN_SCRIPT),
        "--artifact-dir", str(artifact_dir), "--out-dir", str(out_dir),
        "--accelerator", accelerator, "--seed", str(seed),
        "--hidden-size", str(config.hidden_size), "--depth", str(config.depth),
        "--ffn-layers", str(config.ffn_layers), "--dropout", str(config.dropout),
    ]
    return cmd + (["--aux-off"] if variant == "aux_off" else [])


def run_final(artifact_dir: Path, stage_b_root: Path, out_root: Path, accelerator: str) -> int:
    name = selected_configuration(stage_b_root)
    config = {c.name: c for c in build_grid()}[name]
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=False)  # refuses to reuse an existing output root
    (out_root / "final_plan.json").write_text(json.dumps(
        {"configuration": name, "variants": list(VARIANTS), "seeds": list(FINAL_SEEDS)}, indent=2) + "\n")
    total = len(VARIANTS) * len(FINAL_SEEDS)
    failures = i = 0
    for variant in VARIANTS:
        for seed in FINAL_SEEDS:
            i += 1
            out_dir = out_root / run_name(variant, seed)
            t0 = time.time()
            print(f"=== [{i}/{total}] {name} {variant} seed {seed} start {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
            rc = subprocess.run(build_command(config, seed, variant, artifact_dir, out_dir, accelerator)).returncode
            mae = epochs = None
            if rc == 0 and (out_dir / "metrics.json").exists() and (out_dir / "best.ckpt").exists():
                metrics = json.loads((out_dir / "metrics.json").read_text())
                mae, epochs = metrics["best_primary_validation_mae"], metrics["epochs_run"]
            elif rc == 0:
                rc = 98
            with open(out_root / "final_status.jsonl", "a") as handle:
                handle.write(json.dumps({"configuration": name, "variant": variant, "seed": seed, "returncode": rc,
                                         "seconds": round(time.time() - t0, 1), "epochs_run": epochs,
                                         "best_primary_validation_mae": mae}) + "\n")
            failures += rc != 0
            print(f"=== [{i}/{total}] {variant} seed {seed} rc={rc} epochs={epochs} val_mae={mae}", flush=True)
    print(f"=== final training finished: {total - failures} ok, {failures} failed", flush=True)
    return 1 if failures else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--stage-b-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu", "mps"])
    a = p.parse_args()
    sys.exit(run_final(a.artifact_dir, a.stage_b_root, a.out_root, a.accelerator))


if __name__ == "__main__":
    main()
