"""Bounded real-fit smoke for the T121 screening stage (D015).

Runs the production prepare_screen_stage and fit_screen_stage unchanged except
that the module's epoch constant is set to 2 after preparation, so the real
Lightning Trainer, ScreenEpochCheckpoints callback, per-epoch checkpointing,
selection and the selected-checkpoint -> pIC50 handoff all execute on GPU.
It is not the 50-epoch run and qualifies nothing about trained performance.
Not executed: the 50-epoch screen, any pIC50 fit, the cell array.
"""
from pathlib import Path
import hashlib
import json
import sys

PACKAGE = Path("/scratch/user/mhuang8/tripwire_staging/t121_production_chain_20261117")
PR = Path("/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project")
DATASET = PR / "cyp/data/cyp-challenge-TRAIN_inhibition.csv"
DATASET_SHA256 = "b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278"
ORIGINAL_SPLIT = PR / "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv"
ORIGINAL_SPLIT_SHA256 = "4cea719334a0836e4b9c0d50dd093872d472142b33337eb98362f95aa10c1902"
SPLIT_MANIFEST = PR / "cyp/artifacts/cyp_007/split_manifests/rep_01_fold_01.json"
SMOKE_EPOCHS = 2
SEED = 20261101

out = PACKAGE / "results/screen_fit_smoke"
if out.exists():
    raise SystemExit(f"STOP: refusing to overwrite {out}")
sys.path[:0] = [str(PACKAGE / "cyp/scripts"), str(PACKAGE / "executor_source")]

import pandas as pd
import torch

if not torch.cuda.is_available():
    raise SystemExit("STOP: batch job cannot access CUDA.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


import t121_screen_chain_run as launcher

manifest = json.loads((PACKAGE / "screen_manifest.json").read_text())
launcher.validate_manifest(manifest)
verified_files = launcher.verify_pinned_files(manifest["pinned_files"])
verified_evidence = launcher.verify_evidence(manifest["evidence"])
assert manifest["training_seed"] == SEED
assert sha(DATASET) == DATASET_SHA256
assert sha(ORIGINAL_SPLIT) == ORIGINAL_SPLIT_SHA256

import cyp007_cell_fit as cell_fit
import t121_screen_stage as stage

assert Path(stage.__file__).resolve() == Path(verified_files["screen_stage"]["path"]).resolve()
assert Path(cell_fit.__file__).resolve() == Path(verified_files["executor_cell_fit"]["path"]).resolve()

out.mkdir(parents=True)
prepared = stage.prepare_screen_stage(
    **dict(manifest["screen_inputs"]),
    proposal_path=manifest["proposal_path"],
    output_dir=out / "preparation",
    training_seed=SEED,
)
assert stage.SCREEN_EPOCHS == 50
stage.SCREEN_EPOCHS = SMOKE_EPOCHS  # read at call time by fit_screen_stage
selected = stage.fit_screen_stage(prepared, authorize_training=True)

directory = out / "preparation" / "epoch_checkpoints"
selection = json.loads((directory / "selection.json").read_text())
records = selection["records"]
assert [r["epoch"] for r in records] == list(range(SMOKE_EPOCHS))
for record in records:
    assert Path(record["checkpoint"]).is_file()
    assert sha(record["checkpoint"]) == record["sha256"]
    assert record["score"] == record["score"] and abs(record["score"]) < float("inf")
best = min(records, key=lambda r: (r["score"], r["epoch"]))
assert selection["selected"] == best == selected

checkpoints = [torch.load(r["checkpoint"], map_location="cpu", weights_only=False) for r in records]
steps = [c["global_step"] for c in checkpoints]
assert steps[0] > 0 and steps[1] == 2 * steps[0], steps
assert all(c["optimizer_states"] for c in checkpoints)
assert [c["epoch"] for c in checkpoints] == [0, 1]
assert not torch.equal(
    checkpoints[0]["state_dict"][next(iter(checkpoints[0]["state_dict"]))],
    checkpoints[1]["state_dict"][next(iter(checkpoints[1]["state_dict"]))],
)

routed = cell_fit.route_d007_fold(
    dataframe=pd.read_csv(DATASET),
    original_split_dataframe=pd.read_csv(ORIGINAL_SPLIT),
    split_manifest=json.loads(SPLIT_MANIFEST.read_text()),
    training_seed=SEED,
)
prefit = stage.prepare_selected_screen_handoff(
    prepared_screen=prepared,
    selected_record=selected,
    training_dataframe=routed["training_dataframe"],
    validation_dataframe=routed["validation_dataframe"],
    config_name="D007-T121",
    training_seed=SEED,
)
assert prefit["stage_initialization"] == "screen_shared_weights_fresh_pic50_state"
assert getattr(prefit["model"], "_trainer", None) is None

result = {
    "status": "PASS",
    "scope": "T121 screening real Trainer.fit smoke (2 epochs) and selected-checkpoint pIC50 handoff",
    "epochs_run": SMOKE_EPOCHS,
    "global_steps_per_epoch": steps[0],
    "scores": [r["score"] for r in records],
    "selected_epoch": selected["epoch"],
    "selected_checkpoint_sha256": selected["sha256"],
    "handoff": "PASS",
    "pic50_destination_has_no_trainer": True,
    "verified_files": verified_files,
    "verified_evidence": verified_evidence,
    "split_manifest_sha256": sha(SPLIT_MANIFEST),
    "check_py_sha256": sha(__file__),
    "not_executed": [
        "50-epoch screening run",
        "any pIC50 fit",
        "150-cell array",
        "trained screening checkpoint performance",
    ],
}
(out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
print(json.dumps(result, indent=2, default=str), flush=True)
print("SAVED:", out, flush=True)
