"""0-step GPU recheck of screening validation with weighted adaptation.

Derived from the job 3103473 script (check.py, SHA-256 17c12c16...). Differences:
  1. New package/output locations; nothing from the job 3103473 folders is
     written to or overwritten. Its upstream_minimal folder is read only.
  2. Pins the runner and model-side SHA-256 and refuses to run on a mismatch.
  3. Records the SHA-256 of every source module actually imported.
  4. Spies on run_task during validation and asserts every call used
     use_observation_weights=True, is_training_phase=False, and finite,
     positive weights on 5-field arrays.
  5. Reports (does not assert) the score difference from job 3103473's
     0.9987984361996203 under the same seeds, as evidence the weights act.
All other checks are unchanged. Still 0 optimizer steps; no training.
"""
from pathlib import Path
import json
import os
import random
import sys

import numpy as np
import torch

EXPECTED_RUNNER_SHA256 = "af92e152208e339bbfcfdbbe609eed0e7ae3ab4229367f26071b309c2928e68d"
EXPECTED_MODEL_SIDE_SHA256 = "86ccf4b135a0388c147d2838e5d12ead2b5aaf5968f7140ec341e7950d3a62fd"
PRIOR_UNWEIGHTED_SCORE = 0.9987984361996203  # job 3103473 result.json

package = Path("/scratch/user/mhuang8/tripwire_staging/actfound_weighted_validation_check_20261117")
old_stage = Path("/scratch/user/mhuang8/tripwire_staging/actfound_stage_transfer_20261117")
upstream = old_stage / "upstream_minimal/external/ActFound"
inputs = Path("/scratch/user/mhuang8/tripwire_staging/actfound_screen_loader_20261117")
split = Path(
    "/scratch/user/mhuang8/tripwire_staging/t121_screen_complete_20261117/"
    "results/approved_screen_preparation/split_assignments.csv"
)
out = package / "results/screen_validation_gpu_recheck"
if out.exists():
    raise SystemExit(f"STOP: refusing to overwrite {out}")
if not torch.cuda.is_available():
    raise SystemExit("STOP: batch job cannot access CUDA.")

os.chdir(package)
sys.path[:0] = [str(package / "cyp/scripts"), str(upstream), str(package)]
import actfound_model_side
import actfound_production_train_v2 as runner
from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR

runner_path = Path(runner.__file__).resolve()
model_side_path = Path(actfound_model_side.__file__).resolve()
assert runner_path.parent == (package / "cyp/scripts").resolve(), runner_path
assert model_side_path.parent == (package / "cyp/scripts").resolve(), model_side_path
runner_sha = runner.file_sha256(runner_path)
model_side_sha = runner.file_sha256(model_side_path)
if runner_sha != EXPECTED_RUNNER_SHA256:
    raise SystemExit(f"STOP: runner hash {runner_sha} != pinned")
if model_side_sha != EXPECTED_MODEL_SIDE_SHA256:
    raise SystemExit(f"STOP: model-side hash {model_side_sha} != pinned")

prepared = runner.prepare_actfound_partitioned_screen_stage(
    screen_path=inputs / "inputs/screen_full.csv",
    identity_path=inputs / "inputs/adjudication.tsv",
    similarity_path=inputs / "inputs/cluster_adjudication.tsv",
    contract_path=inputs / "qualification/actfound/screen_loader_20261117/contract.json",
    split_assignments_path=split,
    hyperparameters=dict(runner.UPSTREAM_DEFAULTS),
)
args = prepared["args"]
runner.seed_like_upstream(args)
model = FCNReLUNormNetworkQSAR(
    input_shape=(2, args.dim_w), args=args, meta=True
).cuda()
rates = torch.nn.Parameter(
    torch.ones(args.num_updates + 1, device="cuda") * args.update_lr,
    requires_grad=bool(args.learnable_per_layer_per_step_inner_loop_learning_rate),
)
model.train()
before = {k: v.detach().clone() for k, v in model.state_dict().items()}
before_rates = rates.detach().clone()
before_modes = [m.training for m in model.modules()]
python_rng = random.getstate()
numpy_rng = np.random.get_state()
torch_rng = torch.get_rng_state().clone()
cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]

# Spy on run_task: record how validation calls it, then delegate unchanged.
original_run_task = runner.run_task
spy_calls = []


def spying_run_task(regressor, inner_lrs, arrays, args_, epoch, is_training_phase,
                    use_observation_weights=None):
    spy_calls.append({
        "is_training_phase": is_training_phase,
        "use_observation_weights": use_observation_weights,
        "n_fields": len(arrays),
        "weights_finite_positive": bool(
            len(arrays) == 5
            and np.all(np.isfinite(arrays[3]))
            and np.all(np.asarray(arrays[3]) > 0)
        ),
    })
    return original_run_task(
        regressor, inner_lrs, arrays, args_, epoch, is_training_phase,
        use_observation_weights=use_observation_weights,
    )


runner.run_task = spying_run_task
try:
    validation = runner.evaluate_actfound_screen_stage(model, rates, prepared)
finally:
    runner.run_task = original_run_task

assert len(spy_calls) == len(prepared["validation_episodes"]) == 8, len(spy_calls)
assert all(c["is_training_phase"] is False for c in spy_calls)
assert all(c["use_observation_weights"] is True for c in spy_calls)
assert all(c["n_fields"] == 5 and c["weights_finite_positive"] for c in spy_calls)

for key, value in before.items():
    assert torch.equal(model.state_dict()[key], value), key
assert torch.equal(rates.detach(), before_rates)
assert [m.training for m in model.modules()] == before_modes
assert random.getstate() == python_rng
after_numpy = np.random.get_state()
assert after_numpy[0] == numpy_rng[0]
np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
assert after_numpy[2:] == numpy_rng[2:]
assert torch.equal(torch.get_rng_state(), torch_rng)
assert all(
    torch.equal(a, b)
    for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng)
)
assert all(p.grad is None for p in model.parameters())
assert rates.grad is None
assert len(validation["per_enzyme"]) == 4
assert np.isfinite(validation["score"])
for name, repeats in validation["per_repeat"].items():
    assert len(repeats) == 2
    np.testing.assert_allclose(
        validation["per_enzyme"][name], np.mean(list(repeats.values()))
    )
np.testing.assert_allclose(
    validation["score"], np.mean(list(validation["per_enzyme"].values()))
)
assert runner.actfound_screen_query_wmse(
    [100, 1, 3], [0, 0, 0], [1, 0, 0], [999, 1, 3]
) == 7.0
assert runner.select_actfound_screen_checkpoint([
    {"epoch": 4, "score": 2.0},
    {"epoch": 3, "score": 1.0},
    {"epoch": 2, "score": 1.0},
])["epoch"] == 2

blocked = package / "SHOULD_NOT_EXIST_SCREEN_TRAINING"
try:
    runner.fit_actfound_screen_stage(
        regressor=model, inner_lrs=rates, prepared=prepared,
        output_dir=blocked, config={},
    )
except RuntimeError as error:
    assert "not authorized" in str(error)
else:
    raise AssertionError("Training authorization gate failed.")
assert not blocked.exists()

out.mkdir(parents=True)
config_path = out / "qualification_config.json"
config_path.write_text(json.dumps({"qualification_only": True}) + "\n")
outer = runner.build_fresh_actfound_outer_state(model, rates, args)
checkpoint = out / "qualification_model_0.pt"
runner.save_checkpoint(
    checkpoint, model, rates, outer["optimizer"], outer["scheduler"],
    0, {"config_path": str(config_path)},
)
saved = torch.load(checkpoint, map_location="cpu")
assert saved["epoch"] == 0
assert not saved["optimizer_state_dict"]["state"]
assert "scheduler_state_dict" in saved
for key, value in before.items():
    assert torch.equal(saved["regressor_state_dict"][key], value.cpu()), key
assert torch.equal(saved["inner_lrs"], before_rates.cpu())

result = {
    "status": "PASS",
    "scope": "GPU fixed-episode validation and checkpoint correctness",
    "validation": validation,
    "validation_episodes": len(prepared["validation_episodes"]),
    "shared_state_modes_and_rng_restored": True,
    "no_accumulated_parameter_gradients": True,
    "repeat_then_equal_enzyme_aggregation": "PASS",
    "manual_metric_and_tie_rule": "PASS",
    "checkpoint_writer_round_trip": "PASS",
    "authorization_gate": "PASS",
    "weighted_adaptation_spy": {
        "run_task_calls": len(spy_calls),
        "all_use_observation_weights_true": True,
        "all_not_training_phase": True,
        "all_weights_finite_positive_5_field": True,
    },
    "score_vs_job_3103473_unweighted": {
        "prior_score": PRIOR_UNWEIGHTED_SCORE,
        "this_score": validation["score"],
        "difference": validation["score"] - PRIOR_UNWEIGHTED_SCORE,
        "note": "reported, not asserted; same seeds and random-init procedure",
    },
    "optimizer_steps": 0,
    "model_initialization": "random qualification model",
    "runner_sha256": runner_sha,
    "model_side_sha256": model_side_sha,
    "check_py_sha256": runner.file_sha256(Path(__file__).resolve()),
    "checkpoint_sha256": runner.file_sha256(checkpoint),
    "not_executed": [
        "screening training loop",
        "foundation pretraining",
        "production staged execution",
    ],
}
(out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
print("SAVED:", out, flush=True)
