"""Tripwire PXR-002 stage A runner (D018): the 36-configuration grid, one common seed.

Runs every grid configuration serially, each in its own subprocess of pxr002_train.py
and its own new output directory. Nothing resumes: the output root must not exist, and a
failed configuration is recorded and the rest continue (the exit code is nonzero if any
failed). A per-run status line (configuration, return code, seconds, best primary-head
validation MAE) is appended to stage_a_status.jsonl, which also yields measured runtime.

Holdout and official-test labels are never loaded (see pxr002_dataset/pxr002_train).
This authorizes no stage B, ablation, ensemble or holdout evaluation.
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

from pxr.scripts.pxr002_grid import COMMON_SEED, build_grid  # noqa: E402

TRAIN_SCRIPT = REPO_ROOT / "pxr/scripts/pxr002_train.py"


def build_command(config, seed: int, artifact_dir: Path, out_root: Path, accelerator: str) -> list[str]:
    return [
        sys.executable, "-I", str(TRAIN_SCRIPT),
        "--artifact-dir", str(artifact_dir),
        "--out-dir", str(Path(out_root) / config.name),
        "--accelerator", accelerator,
        "--seed", str(seed),
        "--hidden-size", str(config.hidden_size),
        "--depth", str(config.depth),
        "--ffn-layers", str(config.ffn_layers),
        "--dropout", str(config.dropout),
    ]


def run_stage_a(artifact_dir: Path, out_root: Path, accelerator: str) -> int:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=False)  # refuses to reuse an existing output root
    configs = build_grid()
    failures = 0
    status_path = out_root / "stage_a_status.jsonl"
    started = time.time()
    for i, config in enumerate(configs, start=1):
        t0 = time.time()
        print(f"=== [{i}/{len(configs)}] {config.name} start {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
        result = subprocess.run(build_command(config, COMMON_SEED, artifact_dir, out_root, accelerator))
        seconds = time.time() - t0
        metrics_path = out_root / config.name / "metrics.json"
        mae = epochs = None
        if result.returncode == 0 and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            mae, epochs = metrics["best_primary_validation_mae"], metrics["epochs_run"]
        elif result.returncode == 0:
            result.returncode = 98  # exited cleanly but wrote no metrics
        record = {"configuration": config.name, "seed": COMMON_SEED, "returncode": result.returncode,
                  "seconds": round(seconds, 1), "epochs_run": epochs, "best_primary_validation_mae": mae}
        with open(status_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        failures += result.returncode != 0
        print(f"=== [{i}/{len(configs)}] {config.name} rc={result.returncode} {seconds:.0f}s "
              f"epochs={epochs} best_primary_val_mae={mae} elapsed_total={(time.time() - started) / 3600:.2f}h", flush=True)
    print(f"=== stage A finished: {len(configs) - failures} ok, {failures} failed", flush=True)
    return 1 if failures else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--accelerator", required=True, choices=["cpu", "gpu", "mps"])
    a = p.parse_args()
    sys.exit(run_stage_a(a.artifact_dir, a.out_root, a.accelerator))


if __name__ == "__main__":
    main()
