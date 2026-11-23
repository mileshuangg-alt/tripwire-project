"""Bounded real-fit smoke for the D022 CheMeleon screening stage and handoff.

Same design as the D021 descriptor smoke (qualification/t121/
d021_descriptor_smoke_20261119/check.py), with the CheMeleon arm: the
production prepare_screen_stage and fit_screen_stage execute unchanged except
that the module's epoch constant is set to 2 after preparation, so the real
Lightning Trainer, per-epoch checkpointing, selection and the selected-
checkpoint -> pIC50 handoff run on GPU with the real pretrained weights. It
also checks, from the saved checkpoints, the two optimizer parameter groups
and the 0.1 encoder ratio, that the encoder starts from the pretrained
tensors and then moves, and the pIC50 destination (2048-wide head input, mean
aggregation, encoder equal to the selected screening encoder, fresh final
projection, no Trainer, authorization gate). It is not the 50-epoch run and
qualifies nothing about trained performance.
Not executed: the 50-epoch screen, any pIC50 fit, the cell array.
Verification of pinned files only (no evidence list: this check IS evidence).
"""
from pathlib import Path
import hashlib
import json
import sys

PACKAGE = Path("/scratch/user/mhuang8/tripwire_staging/t121_d022_package_20261121")
PR = Path("/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project")
DATASET = PR / "cyp/data/cyp-challenge-TRAIN_inhibition.csv"
DATASET_SHA256 = "b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278"
ORIGINAL_SPLIT = PR / "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv"
ORIGINAL_SPLIT_SHA256 = "4cea719334a0836e4b9c0d50dd093872d472142b33337eb98362f95aa10c1902"
SPLIT_MANIFEST = PR / "cyp/artifacts/cyp_007/split_manifests/rep_01_fold_01.json"
SMOKE_EPOCHS = 2
SEED = 20261101

out = PACKAGE / "results/d022_chemeleon_smoke"
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
from t121_chemeleon import ChemeleonArm

pins = json.loads((PACKAGE / "d022_pins.json").read_text())
verified_files = launcher.verify_pinned_files(pins["pinned_files"])
spec = pins["chemeleon"]
arm = ChemeleonArm.load(spec["config_path"], spec["weights_path"], expected_config_sha256=spec["config_sha256"])
assert sha(DATASET) == DATASET_SHA256
assert sha(ORIGINAL_SPLIT) == ORIGINAL_SPLIT_SHA256

import cyp007_cell_fit as cell_fit
import t121_chemeleon as chemeleon_module
import t121_screen_stage as stage
import t121_weighted_training as weighted

for module, role in (
    (stage, "screen_stage"), (weighted, "weighted_training"),
    (cell_fit, "executor_cell_fit"), (chemeleon_module, "chemeleon_module"),
):
    assert Path(module.__file__).resolve() == Path(verified_files[role]["path"]).resolve(), role

out.mkdir(parents=True)
prepared = stage.prepare_screen_stage(
    **dict(pins["screen_inputs"]),
    proposal_path=pins["proposal_path"],
    output_dir=out / "preparation",
    training_seed=SEED,
    chemeleon=arm,
)
model = prepared["model"]
from chemprop import nn

assert isinstance(model.agg, nn.MeanAggregation)
assert model.encoder_lr_ratio == 0.1
assert model.message_passing.output_dim == 2048
for key, value in model.message_passing.state_dict().items():
    assert torch.equal(value, arm.state_dict[key]), f"encoder differs from pretrained before fit: {key}"
assert prepared["sampled_config"]["batch_size"] == 16
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
assert [c["epoch"] for c in checkpoints] == [0, 1]
ratios = []
for c in checkpoints:
    groups = c["optimizer_states"][0]["param_groups"]
    assert len(groups) == 2, len(groups)
    ratios.append(groups[1]["lr"] / groups[0]["lr"])
    assert abs(ratios[-1] - 0.1) < 1e-9, ratios
encoder_keys = [k for k in checkpoints[0]["state_dict"] if k.startswith("message_passing.")]
assert encoder_keys
moved = [k for k in encoder_keys
         if not torch.equal(checkpoints[0]["state_dict"][k], arm.state_dict[k[len("message_passing."):]])]
assert moved, "encoder did not move from the pretrained tensors after one epoch"
assert any(not torch.equal(checkpoints[0]["state_dict"][k], checkpoints[1]["state_dict"][k]) for k in encoder_keys)
head_widths = sorted({
    tuple(v.shape) for k, v in checkpoints[0]["state_dict"].items()
    if v.ndim == 2 and k.startswith("predictor.ffn.")  # FFN weights only; excludes criterion.task_weights (1, 1)
})
assert head_widths == [(4, 2000), (2000, 2048)], head_widths

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
    chemeleon=arm,
)
destination = prefit["model"]
assert prefit["stage_initialization"] == "screen_shared_weights_fresh_pic50_state"
assert getattr(destination, "_trainer", None) is None
assert isinstance(destination.agg, nn.MeanAggregation)
assert destination.encoder_lr_ratio == 0.1
selected_state = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
for key, value in destination.state_dict().items():
    if key.startswith("message_passing."):
        assert torch.equal(value, selected_state[key]), f"encoder not transferred: {key}"
final = prefit["screen_to_pic50_transition"]["final_projection"]
assert not torch.equal(destination.state_dict()[final + ".weight"], selected_state[final + ".weight"]), "final projection was transferred"
assert prefit["screen_to_pic50_transition"]["optimizer_state_transferred"] is False
assert prefit["screen_to_pic50_transition"]["scheduler_state_transferred"] is False
assert prefit["train_loader"].batch_sampler.batch_size == 16
try:
    weighted.fit_prepared_t121_pic50_stage(prefit_state=prefit, checkpoint_dir=out / "unused")
except RuntimeError as exc:
    assert "not authorized" in str(exc)
else:
    raise AssertionError("pIC50 fit ran without authorization")
assert not (out / "unused").exists()

result = {
    "status": "PASS",
    "scope": "D022 CheMeleon T121 screening real Trainer.fit smoke (2 epochs) with the pretrained encoder, two optimizer groups at 0.1, selected-checkpoint pIC50 handoff, authorization gate",
    "epochs_run": SMOKE_EPOCHS,
    "encoder_lr_to_head_lr_ratios": ratios,
    "head_weight_shapes": [list(w) for w in head_widths],
    "encoder_tensors_moved_after_epoch_0": len(moved),
    "global_steps_per_epoch": steps[0],
    "scores": [r["score"] for r in records],
    "selected_epoch": selected["epoch"],
    "selected_checkpoint_sha256": selected["sha256"],
    "handoff": "PASS",
    "pic50_destination_has_no_trainer": True,
    "pic50_authorization_gate": "PASS",
    "chemeleon_config_sha256": spec["config_sha256"],
    "chemeleon_weights_sha256": arm.weights_sha256,
    "chemeleon_module_sha256": verified_files["chemeleon_module"]["sha256"],
    "screen_stage_sha256": verified_files["screen_stage"]["sha256"],
    "weighted_training_sha256": verified_files["weighted_training"]["sha256"],
    "verified_files": verified_files,
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
