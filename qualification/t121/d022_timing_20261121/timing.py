"""Interactive runtime measurement for the D022 CheMeleon arm (run
in an interactive GPU session; never through Slurm without a recorded approval).

Measures with the production code, on the production hardware class:
  (a) screening: prepare time, then TIMING_EPOCHS real epochs of fit_screen_stage
      (the module's epoch constant is set after preparation, as in the smoke);
      per-epoch seconds are extrapolated to 50 epochs;
  (b) one pIC50 cell (cell 0, CheMeleon arm) through the production cell runner
      from the selected checkpoint of (a), into a new scratch run root.
The extrapolation is arithmetic on measured values, not a prediction of
checkpoint quality. Nothing is written inside the package or any run root.

Usage: python -I timing.py --out NEW_DIR
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

PACKAGE = Path("/scratch/user/mhuang8/tripwire_staging/t121_d022_package_20261121")
PR = Path("/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project")
TIMING_EPOCHS = 3
SEED = 20261101

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
if args.out.exists():
    raise SystemExit(f"STOP: refusing to reuse {args.out}")
sys.path[:0] = [str(PACKAGE / "cyp/scripts"), str(PACKAGE / "executor_source")]

import torch

if not torch.cuda.is_available():
    raise SystemExit("STOP: no CUDA device in this session.")

import t121_plan_cells as planner
import t121_production_cell as cell_runner
import t121_screen_chain_run as launcher
import t121_screen_stage as stage
from cyp007_production_helpers import cell_keys
from t121_chemeleon import ChemeleonArm

pins = json.loads((PACKAGE / "d022_pins.json").read_text())
verified = launcher.verify_pinned_files(pins["pinned_files"])
spec = pins["chemeleon"]
arm = ChemeleonArm.load(spec["config_path"], spec["weights_path"], expected_config_sha256=spec["config_sha256"])
args.out.mkdir(parents=True)

t0 = time.perf_counter()
prepared = stage.prepare_screen_stage(
    **dict(pins["screen_inputs"]), proposal_path=pins["proposal_path"],
    output_dir=args.out / "screen" / "preparation", training_seed=SEED, chemeleon=arm,
)
prepare_s = time.perf_counter() - t0
stage.SCREEN_EPOCHS = TIMING_EPOCHS
t0 = time.perf_counter()
selected = stage.fit_screen_stage(prepared, authorize_training=True)
fit_s = time.perf_counter() - t0
stage.SCREEN_EPOCHS = 50  # restore: the cell runner re-prepares the screening shell (warmup < epochs check)
per_epoch = fit_s / TIMING_EPOCHS
record = args.out / "screen" / "selected_screen_record.json"
record.write_text(json.dumps(selected, indent=2, default=str) + "\n")

cell_root = args.out / "cell_root"
planner.plan_cells(
    run_root=cell_root, proposal_path=pins["proposal_path"],
    proposal_sha256=verified["proposal"]["sha256"], cell_keys=cell_keys(),
)
t0 = time.perf_counter()
result = cell_runner.run_updated_t121_cell(
    run_root=cell_root, trial_index=121, cell_index=0,
    dataset_path=PR / "cyp/data/cyp-challenge-TRAIN_inhibition.csv",
    original_split_path=PR / "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv",
    split_manifest_root=PR / "cyp/artifacts/cyp_007/split_manifests",
    screen_selection_path=record,
    screen_selection_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
    screen_proposal_path=pins["proposal_path"], screen_inputs=dict(pins["screen_inputs"]),
    authorize_training=True, chemeleon=arm,
)
cell_s = time.perf_counter() - t0
cell_ckpt = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
cell_groups = cell_ckpt["optimizer_states"][0]["param_groups"]
assert len(cell_groups) == 2, len(cell_groups)
assert abs(cell_groups[1]["lr"] / cell_groups[0]["lr"] - 0.1) < 1e-9

report = {
    "device": torch.cuda.get_device_name(0),
    "screen_prepare_seconds": prepare_s,
    "screen_epochs_timed": TIMING_EPOCHS,
    "screen_fit_seconds": fit_s,
    "screen_seconds_per_epoch": per_epoch,
    "screen_50_epoch_extrapolated_seconds": per_epoch * 50 + prepare_s,
    "cell_0_seconds": cell_s,
    "cell_0_optimizer_groups": len(cell_groups),
    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    "cell_0_macro_st_rae_from_3_epoch_screen_checkpoint": result["macro_st_rae"],
    "cell_array_extrapolated_serial_seconds": cell_s * 150,
    "note": "measured interactively; extrapolation is arithmetic; cell score is from an untrained-for-50-epochs screening checkpoint and is not a result",
}
(args.out / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
