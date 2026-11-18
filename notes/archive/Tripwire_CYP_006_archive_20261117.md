# Tripwire-CYP-006 — Session Log

**Session:** Tripwire-CYP-006
**Date:** start 2026-11-17 (Miles: "this will be Tripwire-CYP-006")
**Status:** OPEN. One finding and two Miles statements are recorded (§2, §3): evaluation-phase adaptation weighting is intended for screening validation only; the pIC50 evaluator stays as it is (unweighted, no uncertainty input). Entered as D016 in `notes/Decisions.md`; written and approved 2026-11-17 (Miles: "Approve D016"). Whether to annotate D015's open-items line is undecided. One local code edit (screening-validation weights) and local mock-level tests (39 passed, 0 failed after a test-only fix) are recorded (§4); Miles's stated position is that a GPU check is not required unless GPU-specific behavior is left untested (not yet determined). No GPU check, run, training or submission is done or authorized.
**Workstream:** Tripwire — CYP inhibition arm; ActFound-CYP whole-package study
**Governing decision:** D013 and D013 Amendments 1–4 (`notes/Decisions.md`), all Approved; D015 (2026-11-17 comparison plan; form approved, authorizes no run; whether it supersedes any D013 term is an open question). Carried in unchanged from Session 005.
**Predecessor record:** `notes/Tripwire_CYP_005.md` (CLOSED 2026-11-17). Its archives A1 `notes/archive/Tripwire_CYP_005_archive_20261115.md` and A2 `notes/archive/Tripwire_CYP_005_archive_20261116.md` remain the detailed records for Session 005.

---

## 1. Carried in from Session 005 (unresolved at its close, 2026-11-17)

Source: `notes/Tripwire_CYP_005.md` §12. Session 005 closed on Miles's declaration of 2026-11-17. None of the following was resolved in that session; each is carried forward as it stands in Session 005 §5.14, §9 and §10. This list adds no Session 006 content and no decision.

1. **Wiring list (Miles's list, none done):** pIC50 support weights reaching adaptation; normalization/reset semantics; qualification evidence bound to deployed hashes; real adapter/config/input binding; T121 production/O150 routing and epoch-end checkpoint behavior. Component/mock tests do not establish launch readiness; no resume support established.
2. **Support weights at evaluation:** `run_task` builds observation weights only when `is_training_phase=True`; screening validation and the pIC50 evaluator call it with `False`. Whether that matches the approved weighting rule is not recorded (Session 005 §10).
3. **T121 connection to the controller:** the stage controller and adapters cover only ActFound foundation → screening → pIC50; T121 is not connected (Session 005 §10).
4. **Comparator provenance:** `C0_COMPARISON.json` still absent; the O150 comparison (ActFound-CYP 0.8699544509650656 vs T121 0.701212576796) was computed against the corroborating D009 artifact, so `comparison_is_final=false` (Session 005 §5.9, §10).
5. **Open question on D015:** whether D015 supersedes D013's "retrain once; checkpoint freeze" terms is not decided; epoch-51 checkpoint `4db29a04…` and job 3076848 are preserved unchanged (Session 005 §10, `notes/Decisions.md` D015).
6. **Naming note:** `qualification/actfound/d014_lr_penalty_20261117/` uses "d014", but D014 is the PXR holdout decision; the ActFound penalty work has no D-number; nothing renamed (Session 005 §10).
7. **Job 3103473 binding:** the controller does not enforce the hash binding; model-side has no in-job hash record (Session 005 §5.14, §10).

Standing at Session 005 close: launch readiness is not established. No training, model run or SLURM submission is authorized.

---

## 2. Support weights at evaluation: inspection finding (2026-11-17)

Read-only inspection; nothing edited or run. This addresses carried item 2 in §1.

- `run_task` in `cyp/scripts/actfound_production_train_v2.py` (about lines 573-581) builds observation weights only when `is_training_phase=True`.
- Screening validation (same file, about line 1369) calls it with `False`, so adaptation there is currently unweighted.
- The pIC50 evaluator (`qualification/actfound/d013_target_qualification_20261116/run_evaluation.py`, about line 155) also calls it with `False`. It uses 3-field arrays from `qualify_adaptation.assemble_arrays`, so adaptation there is currently unweighted.
- The evaluator's inputs and config reference no uncertainty column. Weighting pIC50 adaptation would therefore need a weight source, and none is identified in the records.
- The training-only gate first appears in commit 8caf07e. No recorded approval for it was found.
- The only weighting rule on record is D015 item 5: "Preserve exact approved uncertainty and harmonic pair weighting from qualified code and session evidence. Do not recreate formulas from defaults."

Line numbers are approximate, as reported from the inspection.

## 3. Decisions and approvals

- **Evaluation-phase weighting (Miles, 2026-11-17).** Asked "Is the evaluation adaptation meant to be weighted?", Miles answered, verbatim: "Weighted intended". This is recorded as stated. Nothing in `notes/Decisions.md` has been added or changed.
- **Scope of that answer (Miles, 2026-11-17).** Asked whether "Weighted intended" covers screening validation only or also pIC50 evaluation, Miles answered, verbatim: "Screening only". Effect: weighted adaptation is intended for screening validation; the pIC50 evaluator stays as it is (unweighted, no uncertainty input). The earlier scope question is therefore answered, and the pIC50 weight-source question is moot for this scope. Recorded as D016 in `notes/Decisions.md`.
- **Code edit approval (Miles, 2026-11-17).** Asked whether he approved the `run_task` change as described (screening validation only; `.bak` copy; new tests; runner hash change), and whether the bookkeeper should check `Decisions.md` format and draft an entry for approval, Miles answered "yes to both". The assistant read this as covering the edit as described with foundation validation (about line 698) not included; Miles was not asked to confirm that reading. The approval covers the edit only, not any GPU check, run, training or submission.
- **D016 (Miles, 2026-11-17).** Miles approved the D016 wording ("Approve D016"); D016 is written in `notes/Decisions.md` with status Approved. It authorizes no run, code change, training or submission. D015's open-items line (about line 6720) was not edited; Miles flagged a tension with it, and whether to annotate that line as moot is undecided.
- **Not authorized.** Beyond the §4 edit, these statements authorize no run, training or submission.

## 4. Screening-validation weights: code edit and local test (2026-11-17)

Status: edit made and tested locally with mocks only (39 passed, 0 failed). This is not a deployment and not a GPU correctness check. It addresses carried item 2 in §1 for the screening scope stated in §3.

**Edit.** `cyp/scripts/actfound_production_train_v2.py`:
- `run_task` gained keyword `use_observation_weights: Optional[bool] = None`. `None` keeps the old rule (weights only if `is_training_phase`). `True` applies weights from 5-field arrays outside training.
- The screening-validation call (about line 1373) passes `True`. No other call changed; the foundation-validation call (about line 698) is unchanged.
- Backup: `cyp/scripts/actfound_production_train_v2.py.pre_screen_weights_20261117.bak`, SHA-256 `9627ff17c2d0132dea2c8d34eae6cf38a5921479898e91a2eb1f465450534303`.
- New runner SHA-256: `af92e152208e339bbfcfdbbe609eed0e7ae3ab4229367f26071b309c2928e68d`. The CoreHPC staging copy and the staged-integration archive (`79c4ace8…`) now differ from the local runner.

**Tests.** New file `cyp/tests/test_actfound_run_task_weights.py` (4 tests, stubbed forward capturing `observation_weights`). Local run (miniforge python3, torch 2.15.0, `-I -m pytest`) of that file plus `test_actfound_weighting.py`, `test_actfound_lr_penalty.py`, `test_actfound_stage_adapters.py`, `test_actfound_stage_controller.py`: 38 passed, 1 failed.
- That run had one failure, `cyp/tests/test_actfound_lr_penalty.py::test_runner_applies_penalty_once_in_training_only` (KeyError `epoch`). It predated the screening-weights edit: the test expected `epoch` as a keyword in the `add_inner_learning_rate_penalty` call, but the signature is `(meta_loss, learned_lr_parameters, epoch, args, *, is_training_phase)` and the runner (and the `.bak`) pass `epoch` as the third positional argument.
- **Fix (test only, at Miles's request).** The test now asserts that `epoch` is not a keyword and that `call.args[2]` is the Name `epoch`. The runner is unchanged (SHA-256 still `af92e152…` as recorded above; not re-hashed in this step). No `.bak` of the test file was made; the change is in the uncommitted git diff.
- **Re-run:** the same five test files (`python3 -I -m pytest`): 39 passed, 0 failed (previously 38 passed, 1 failed). Mock-level only; not a GPU check and not real-model behavior.

**GPU check (Miles's stated position, 2026-11-17).** For the screening-weights change, a GPU check should not be required unless the change leaves some GPU-specific behavior untested. This is a statement of position only. It is not an approval of any run, training or submission. It does not state that no GPU-specific behavior is untested; that determination (for example device/dtype handling of the weights tensor) has not been made. Not entered in `notes/Decisions.md`.

**Not verified.**
- No GPU check has been done, and none is currently stated as required (see position above); which GPU-specific behavior, if any, remains untested is UNKNOWN. No run, training or submission.
- The tests use a stubbed forward, so they do not establish real-model behavior or launch readiness.
- Whether foundation validation (about line 698) should also be weighted is not decided; it was left unchanged.
- The edited runner's hash is not yet bound to any staged package or qualification evidence.
