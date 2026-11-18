# Tripwire-CYP-006 — Supporting Record

**Purpose:** detail behind `notes/Tripwire_CYP_006.md`: hashes, line references, test history and carried-in items.

**Provenance:** extracted from the pre-restructuring log, preserved byte-identical as `notes/archive/Tripwire_CYP_006_archive_20261117.md` (SHA-256 `cc91b757df01d4024d1830cce324ea5c6bcbe74196893008843ac55e110b0c9e`). Approval wording lives there and in `notes/Decisions.md`.

## S1. Runner edit and hashes

- File: `cyp/scripts/actfound_production_train_v2.py`. `run_task` gained keyword `use_observation_weights: Optional[bool] = None`; `None` keeps the old rule (weights only if `is_training_phase`), `True` applies weights from 5-field arrays outside training. The screening-validation call (about line 1373) passes `True`; the foundation-validation call (about line 698) is unchanged.
- Backup: `cyp/scripts/actfound_production_train_v2.py.pre_screen_weights_20261117.bak`, SHA-256 `9627ff17c2d0132dea2c8d34eae6cf38a5921479898e91a2eb1f465450534303`.
- Edited runner SHA-256: `af92e152208e339bbfcfdbbe609eed0e7ae3ab4229367f26071b309c2928e68d` (not re-hashed after the test-only fix below). The CoreHPC staging copy and the staged-integration archive (`79c4ace8…`) differ from it.

## S2. Inspection finding (read-only; line numbers approximate)

- `run_task` builds observation weights only when `is_training_phase=True` (about lines 573–581).
- Screening validation (about line 1369) and the pIC50 evaluator (`qualification/actfound/d013_target_qualification_20261116/run_evaluation.py`, about line 155; 3-field arrays from `qualify_adaptation.assemble_arrays`) called it with `False`.
- The evaluator's inputs and config reference no uncertainty column; no weight source is identified in the records.
- The training-only gate first appears in commit `8caf07e`; no recorded approval for it was found.

## S3. Local tests

Files: `cyp/tests/test_actfound_run_task_weights.py` (new, 4 tests, stubbed forward capturing `observation_weights`), `test_actfound_weighting.py`, `test_actfound_lr_penalty.py`, `test_actfound_stage_adapters.py`, `test_actfound_stage_controller.py`; miniforge python3, torch 2.15.0, `-I -m pytest`.

- First run: 38 passed, 1 failed. `test_actfound_lr_penalty.py::test_runner_applies_penalty_once_in_training_only` failed with KeyError `epoch`; it predated the weights edit. The test expected `epoch` as a keyword, but `add_inner_learning_rate_penalty(meta_loss, learned_lr_parameters, epoch, args, *, is_training_phase)` receives it positionally in the runner and in the `.bak`.
- Fix (test only): the test now asserts `epoch` is not a keyword and `call.args[2]` is the Name `epoch`. No `.bak` of the test file was made.
- Re-run: 39 passed, 0 failed. Mock-level only.

## S4. Items carried in from Session 005

Source: `notes/archive/Tripwire_CYP_005_archive_20261117.md` §12. (1) wiring list: pIC50 support weights reaching adaptation, normalization/reset semantics, qualification evidence bound to deployed hashes, real adapter/config/input binding, T121 production/O150 routing and epoch-end checkpoint behavior; (2) support weights at evaluation (addressed by D016 and the edit above); (3) T121 not connected to the controller; (4) comparator provenance (`C0_COMPARISON.json` absent, `comparison_is_final=false`); (5) whether D015 supersedes D013's retrain-once and checkpoint-freeze terms (settled by D015 Amendment 1, 2026-11-17: new separate artifacts, D013 artifacts frozen); (6) the `d014_lr_penalty_20261117` folder name colliding with PXR D014; (7) job 3103473 hash binding not enforced by the controller.
