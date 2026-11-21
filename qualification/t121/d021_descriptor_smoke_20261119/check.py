"""Bounded real-fit smoke for the D021 descriptor screening stage (arm D).

Same design as the D015 screen_fit_smoke check (qualification/t121/
screen_fit_smoke_20261117/check.py), run with the frozen descriptor cache:
the production prepare_screen_stage and fit_screen_stage execute unchanged
except that the module's epoch constant is set to 2 after preparation, so the
real Lightning Trainer, per-epoch checkpointing, selection and the selected-
checkpoint -> pIC50 handoff run on GPU. It also checks the pIC50 destination
(FFN input 600 + 196, strict transfer, no Trainer, authorization gate). It is
not the 50-epoch run and qualifies nothing about trained performance.
Not executed: the 50-epoch screen, any pIC50 fit, the cell array.
Verification of pinned files only (no evidence list: this check IS evidence).
"""
from pathlib import Path
import hashlib
import json
import sys

PACKAGE = Path("/scratch/user/mhuang8/tripwire_staging/t121_d021_package_20261119")
PR = Path("/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project")
DATASET = PR / "cyp/data/cyp-challenge-TRAIN_inhibition.csv"
DATASET_SHA256 = "b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278"
ORIGINAL_SPLIT = PR / "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv"
ORIGINAL_SPLIT_SHA256 = "4cea719334a0836e4b9c0d50dd093872d472142b33337eb98362f95aa10c1902"
SPLIT_MANIFEST = PR / "cyp/artifacts/cyp_007/split_manifests/rep_01_fold_01.json"
SMOKE_EPOCHS = 2
SEED = 20261101

out = PACKAGE / "results/d021_descriptor_smoke"
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
from t121_descriptor_cache import DescriptorCache, attach_descriptor_columns

pins = json.loads((PACKAGE / "d021_pins.json").read_text())
verified_files = launcher.verify_pinned_files(pins["pinned_files"])
spec = pins["descriptor_cache"]
cache = DescriptorCache.load(spec["directory"], expected_manifest_sha256=spec["manifest_sha256"])
assert cache.dim == 196, cache.dim
assert sha(DATASET) == DATASET_SHA256
assert sha(ORIGINAL_SPLIT) == ORIGINAL_SPLIT_SHA256

import cyp007_cell_fit as cell_fit
import t121_screen_stage as stage
import t121_weighted_training as weighted

for module, role in ((stage, "screen_stage"), (weighted, "weighted_training"), (cell_fit, "executor_cell_fit")):
    assert Path(module.__file__).resolve() == Path(verified_files[role]["path"]).resolve(), role

out.mkdir(parents=True)
prepared = stage.prepare_screen_stage(
    **dict(pins["screen_inputs"]),
    proposal_path=pins["proposal_path"],
    output_dir=out / "preparation",
    training_seed=SEED,
    descriptor_cache=cache,
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
first_key = next(iter(checkpoints[0]["state_dict"]))
assert not torch.equal(checkpoints[0]["state_dict"][first_key], checkpoints[1]["state_dict"][first_key])
message_dim = prepared["sampled_config"]["message_hidden_dim"]
widths = sorted({
    tuple(v.shape) for k, v in checkpoints[0]["state_dict"].items()
    if v.ndim == 2 and "predictor" in k and v.shape[1] == message_dim + cache.dim
})
assert widths, "no predictor weight with input width message_hidden_dim + 196 in the screening checkpoint"

routed = cell_fit.route_d007_fold(
    dataframe=pd.read_csv(DATASET),
    original_split_dataframe=pd.read_csv(ORIGINAL_SPLIT),
    split_manifest=json.loads(SPLIT_MANIFEST.read_text()),
    training_seed=SEED,
)
# Descriptors attach after routing (protected rows are dropped by the guard and
# are deliberately absent from the cache), exactly as the cell runner does.
routed = dict(routed)
for _key in ("training_dataframe", "validation_dataframe"):
    routed[_key] = attach_descriptor_columns(routed[_key], cache)
prefit = stage.prepare_selected_screen_handoff(
    prepared_screen=prepared,
    selected_record=selected,
    training_dataframe=routed["training_dataframe"],
    validation_dataframe=routed["validation_dataframe"],
    config_name="D007-T121",
    training_seed=SEED,
    descriptor_dim=cache.dim,
)
assert prefit["stage_initialization"] == "screen_shared_weights_fresh_pic50_state"
assert prefit["descriptor_dim"] == cache.dim
assert getattr(prefit["model"], "_trainer", None) is None
pic50_widths = sorted({
    tuple(v.shape) for k, v in prefit["model"].state_dict().items()
    if v.ndim == 2 and "predictor" in k and v.shape[1] == message_dim + cache.dim
})
assert pic50_widths == widths, (pic50_widths, widths)
try:
    weighted.fit_prepared_t121_pic50_stage(prefit_state=prefit, checkpoint_dir=out / "unused")
except RuntimeError as exc:
    assert "not authorized" in str(exc)
else:
    raise AssertionError("pIC50 fit ran without authorization")
assert not (out / "unused").exists()

result = {
    "status": "PASS",
    "scope": "D021 descriptor T121 screening real Trainer.fit smoke (2 epochs), selected-checkpoint pIC50 handoff with 796-input first FFN layer, authorization gate",
    "epochs_run": SMOKE_EPOCHS,
    "descriptor_dim": cache.dim,
    "predictor_input_widths": [list(w) for w in widths],
    "global_steps_per_epoch": steps[0],
    "scores": [r["score"] for r in records],
    "selected_epoch": selected["epoch"],
    "selected_checkpoint_sha256": selected["sha256"],
    "handoff": "PASS",
    "pic50_destination_has_no_trainer": True,
    "pic50_authorization_gate": "PASS",
    "descriptor_manifest_sha256": spec["manifest_sha256"],
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
