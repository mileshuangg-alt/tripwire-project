# Tripwire-CYP-006 Session Log

**Date:** 2026-11-17 (updated 2026-11-18)  
**Status:** Closed on Miles's declaration, 2026-11-18. Screening-validation weighting decided (D016), implemented and GPU-rechecked (job 3105525 PASS); updated-T121 O150 (0.6769540372619428) and a non-final ActFound-CYP O150 (0.8513335058029842) now exist (§4, §4a); comparison finality, challenger selection and the open items in §5 are carried forward unresolved.

## 1. Session purpose

Session 006 continues the ActFound-CYP whole-package study after Session 005 closed. Its first task was the carried question of whether evaluation-phase adaptation should be weighted (`notes/Tripwire_CYP_005.md`, §8–§9).

Governing decisions: D013 and Amendments 1–4, D015 (comparison plan; authorizes no run) and D016.

## 2. Weighting at evaluation

Inspection found that `run_task` in `cyp/scripts/actfound_production_train_v2.py` built observation weights only when `is_training_phase=True`. Screening-validation adaptation and the pIC50 evaluator therefore both ran unweighted. No weight source for the pIC50 evaluator exists in the records, and the training-only gate has no recorded approval. Details: supporting record S2.

**D016** (`notes/Decisions.md`) fixes the intent: weighted adaptation for screening validation only; the pIC50 evaluator stays unweighted. D015's open-items line on pIC50 support weights is therefore not required for the evaluator; whether to annotate it is undecided.

## 3. Screening-validation weights: edit and local test

`run_task` gained a `use_observation_weights` keyword, and the screening-validation call passes `True`. The foundation-validation call is unchanged; whether it should also be weighted is not decided. The edited runner has SHA-256 `af92e152…` (backup and details: supporting record S1).

Local mock-level tests of the five relevant files give 39 passed, 0 failed (S3). One earlier failure was a test written against a keyword that the runner passes positionally; the test was corrected and the runner was not changed. The tests use a stubbed forward, so they establish neither real-model behavior nor launch readiness.

GPU recheck: job 3105525 PASSED on runner SHA-256 `af92e152…` with model-side SHA-256 `86ccf4b1…` (full hashes and output: supporting record where recorded; otherwise on CoreHPC). Its position that a GPU check was initially not required is not entered in `notes/Decisions.md`. The same session also produced D015 Amendment 1 (`notes/Decisions.md`), a hash-bound controller evidence gate, and ActFound and updated-T121 staged-chain launchers (the code changes after commit `6519419` are uncommitted; see §5).

## 4. Updated-T121 pIC50 stage (CoreHPC, 2026-11-18)

The updated-T121 pIC50 stage (D015; D016 weighting applies to screening validation only, pIC50 evaluation unweighted) finished. Array job 3105880 (cells 1-149): 149/149 COMPLETED per sacct; cell 0 was the earlier pilot. A read-only file-level check run on the CoreHPC login host over `/scratch/user/mhuang8/tripwire_staging/t121_production_chain_20261117_cells_run/trials/trial_121/cells` found 150 cell directories, all valid (COMPLETED.json present, no INCOMPLETE.json, result.json SHA-256 matching COMPLETED.json `result_sha256`, finite `macro_st_rae`), 150 distinct (repetition, fold, seed) keys, no problems.

Updated-T121 O150 (mean `macro_st_rae` over the 150 cells, lower is better) = 0.6769540372619428; per-cell min 0.5979100210008923, max 0.7393726578456778. The frozen T121 O150 is 0.701212576796 (`T121_O150` in `cyp/scripts/d011_inner_adaptation.py`), reproduced from the saved comparator cells to 3e-13. Descriptive comparison only (the pipeline differs, so this is not a verified like-for-like change): updated minus frozen T121 = -0.0243; per isoform -0.0075 (1A2), -0.0232 (2C9), -0.0438 (2D6), -0.0225 (3A4); per repetition -0.0193 to -0.0279. All 150 updated-T121 fold checkpoints are present on CoreHPC (about 24.4 MB each; SHA-256 match the recorded values).

Descriptive diagnostic of what limits T121 on CYP1A2 and CYP2D6: absolute implied soft errors are 1A2 0.386, 2C9 0.179, 2D6 0.365, 3A4 0.237 against mean-predictor values 0.486 / 0.307 / 0.418 / 0.527; 2D6 labels are uncorrelated with the other isoforms; train-validation similarity is lowest on 2D6; models across families rank the isoforms the same way. Conclusion: no model change is supported yet. Per-cell validation predictions were never saved.

## 4a. ActFound staged chain job 3105596 failed (CoreHPC, 2026-11-18)

Job 3105596 ended FAILED (exit 1:0) after 11:24:33 elapsed against a 72 h limit, so it was not a timeout. Evidence is on CoreHPC: Slurm err log `actfound_d015_chain_3105596.err` and run dir `/scratch/user/mhuang8/tripwire_staging/actfound_production_chain_20261117_run`.

- Foundation stage: `foundation/completion.json` and `foundation/training/completion.json` were written at 08:50 on Nov 18; `progress.jsonl` logged 60 epochs complete. The `status` field inside `completion.json` has not been inspected.
- Screening stage: failed before any screening training. In `actfound_stage_adapters.py` `screening_adapter`, `write_new_json` of `effective` (which includes `vars(args)`) raised TypeError (Object of type RandomState is not JSON serializable); `make_args` in `actfound_production_train_v2.py` sets `args.rng = np.random.RandomState`. `screening/` holds a partial `effective_screening_config.json` (1466 B), `initialization.json` and `failure.json`. No screening training and no pIC50 evaluation ran.
- The adapter unit tests used a stubbed runner and so did not exercise the real args namespace.
- At the time of failure no fix or rerun was yet approved; the Option B continuation below followed. The T121 arm result above is unaffected.

Continuation (Option B, approved and implemented): the adapter config fix (`effective_screening_config` omits `args.rng`; `write_new_json` serializes before opening), a controller `completed_foundation` verification and a launcher continuation manifest; 76 ActFound tests pass locally (`cyp` tests `test_actfound*`). Package `actfound_production_continue_20261118.tar.gz` (SHA-256 `d6cacfd52094fb5694c3a55cce526e66e913ca1d1b1974677c89f1d690a79ad1`; manifest SHA-256 `49466cae9939eaf070551ffae1f58fea678e349fe53ce6dd6a2fd486c5eea401`) pins the foundation checkpoint (`1b34a2c7571fadebd00777474099cd85e6ef908a2af88715569b96bcee31eeb3`) and foundation `completion.json` (`076fd14c4df6f9d316da056d95a92a753ce450dfb7b85b867011ba0233537c2c`) from job 3105596's run dir. Job 3117821 (submitted on CoreHPC) ended FAILED 1:0 after 00:11:31 (Slurm elapsed). The foundation lineage check passed; screening COMPLETED (80/80 epochs, 1,920 batches, about 15 minutes); the pIC50 stage FAILED. The predictions step succeeded (`pic50/predictions/cyp_adaptation_predictions.json`), then the scoring subprocess `score_evaluation.py` exited 1 with comparator artifact missing or SHA-256 mismatch for `<package>/artifacts/d009_closure_workspace/p1_d009_o150_result_20261108.json`. Cause: that file, which the scorer hash-checks (expected SHA-256 `b277f4ccd9138b688ff20bfaf2378ebfcb66bc118ab5943df48027168adc0e2b`, matching the local copy), was not included in the package; the other scorer inputs (`st_rae.py`, `cyp002_validation.py`, `custom_scoring_functions.py`, evaluation scripts) are. No ActFound O150 came from this attempt; it was produced by the rebuilt package and job 3118356 below.

Rebuilt package `actfound_production_continue_20261118b.tar.gz` (SHA-256 `410dc6e1322730e128ed07bd10bdb358f662993d1228a325c3d9126dc2ac210b`; manifest SHA-256 `7dc300ca8d6920599e262104468a0d1d12ba8330c18e365e387fc1272856be11`) adds the hash-pinned corroborating comparator file `artifacts/d009_closure_workspace/p1_d009_o150_result_20261108.json` (SHA-256 `b277f4cc...`); sources unchanged. Job 3118356 (submitted by Miles) COMPLETED 0:0 in 00:11:40: foundation lineage check OK (checkpoint `1b34a2c7...eb3` from job 3105596's run); screening 80/80 epochs (1,920 batches, 884 s); pIC50 completed. Evidence is on CoreHPC: `/scratch/user/mhuang8/tripwire_staging/actfound_production_continue_20261118b_run/pic50/scores/cyp_o150_result.json` (SHA-256 `1777918cb1ad156e568623b9f5357a231ab24bc46cdda7c3a5fa580343776725`; `evaluation_manifest.json` SHA-256 `15b1092c645fc1fc2cef6a2d40019d1ebdc37d4c0be9cd92f3d46eee6690c3c1`).

**ActFound-CYP O150 = 0.8513335058029842** (150 cells; 200 unique adaptations recorded into three seed slots as replicates; D013 evaluator, unweighted). The result reports `favours_actfound_cyp` false and `comparison_is_final` false (the scorer used the corroborating D009 file; the cited `C0_COMPARISON.json` was not in the scoring package). Context, descriptive only: frozen T121 O150 0.701212576796 (difference about +0.150), updated T121 O150 0.6769540372619428 (about +0.174), earlier D013 ActFound O150 0.8699544509650656 from a historical checkpoint (about -0.019).

Screening reproducibility: the chain ran screening twice (3117821 and 3118356); the second feeds this result. The selected screening checkpoints differ in bytes (SHA-256 `7b4f1209032f74d9c87cfbc30a22bea30167891bde5906917ea1d284ee3464f3` for 3117821; `6573cd51ec52f673ab622acbc9996aeebf7c70a12b1d13f31f85c222249f8a46` for 3118356), but a read-only tensor-by-tensor comparison on CoreHPC found every tensor equal (regressor state, optimizer state, inner LRs, scheduler state, epoch, upstream commit). The only differing entry is the embedded `config_path` (the two package directory names differ by `b`). The screening step therefore reproduced exactly, which resolves the earlier unexplained-hash question for the checkpoints. Still unexplained: the predictions-file SHA-256 differs between the attempts (`1a68cfdf...` in this result versus `adcd5892...` in job 3117821).

Per-isoform comparison (descriptive; from the local copy of `cyp_o150_result.json`, SHA-256 `1777918c...6725`; keys match the T121 cells, macro equals the mean of endpoints, and seed slots are identical within each of the 50 partitions, confirming the replicate topology). Mean ST-RAE, lower is better:

| Isoform | ActFound | Updated T121 | Frozen T121 |
|---|---|---|---|
| CYP1A2 | 0.9425 | 0.7964 | 0.8039 |
| CYP2C9 | 0.7946 | 0.5866 | 0.6098 |
| CYP2D6 | 1.0028 | 0.8745 | 0.9183 |
| CYP3A4 | 0.6654 | 0.4502 | 0.4728 |
| Macro | 0.8513 | 0.6770 | 0.7012 |

ActFound minus updated T121 by repetition (30 cells each): +0.1745, +0.1731, +0.1755, +0.1735, +0.1754; minus frozen T121: +0.1485, +0.1537, +0.1545, +0.1465, +0.1475. At partition level (50 partitions, T121 averaged over its 3 seeds because ActFound seed slots are replicates), ActFound is higher than updated T121 in 50/50 (mean +0.1744, range +0.1244 to +0.2495) and higher than frozen T121 in 50/50 (mean +0.1501, range +0.0749 to +0.2266). For context, the earlier D013 ActFound per-isoform values were 0.936 / 0.807 / 1.009 / 0.728. All descriptive; the comparison remains non-final and no challenger is selected.

Not decided: final status of the comparison, adoption of the located `C0_COMPARISON.json`, any challenger choice.

Status: the ActFound arm now has a non-final result (above); `C0_COMPARISON.json` was located on CoreHPC by a read-only search at `/scratch/user/mhuang8/tripwire-d006/d006_bundle/cyp/artifacts/cyp_007/finalist_completion_v1/C0_COMPARISON.json` (a different project root, `d006_bundle`; earlier searches of `d012_runtime_qualification_20261110` and `tripwire_staging` found nothing). Its sha256sum is `48bd619ebd42b30b0c8862f915b7696e2156e822f48e220789b1563eb46817a8`, equal to the SHA-256 cited in D013/D015, the scorer config and the beside-it `C0_COMPARISON_SHA256.txt`; it contains `c0_o150` 0.7181033743880986 and `trial_121_o150` 0.701212576796, matching the comparator value O150(T121)=0.701212576796. It has not been placed in a scoring package and no ActFound score has used it; its adoption is not made. Whether this satisfies D013 Amendment 4 item 6 (comparator provenance) and permits calling the comparison final is undecided (Miles's decision); `comparison_is_final` remains false. Not yet checked: the cells were not compared byte-for-byte against `prepare_trial_cells` output (planner `t121_plan_cells_v2.py` duplicates its layout).

## 5. Stopping point and next step

Session 006 is closed (Miles's declaration, 2026-11-18). Outcomes: updated-T121 O150 0.6769540372619428 (§4); non-final ActFound-CYP O150 0.8513335058029842 from job 3118356 (§4a); no run is pending or authorized. D015 Amendment 1 (`notes/Decisions.md`) settles the D013 question: the fresh corrected-penalty initialization writes new, separate artifacts, and the D013 epoch-51 checkpoint, job 3076848 and results stay frozen; it authorizes no run.

Open items carried forward, none decided:
- Final status of the comparison, and whether the located `C0_COMPARISON.json` satisfies D013 Amendment 4 item 6.
- No challenger has been selected.
- Held-out validation-prediction inference pass for the 150 updated-T121 cells: planned only (Step 0 passed). The script, tests, provisional 1e-6 tolerance and interactive timing on gpu-dev1 are still to do; no files written, nothing run. The analysis design (plain MAE, signed bias, soft MAE, ST-RAE per isoform; error against training-set similarity and CI width; no bootstrap, no new training) is approved in outline only.
- D012 record discrepancy: the local D012 aggregate summary shows T121 0.7075 / B 0.7948 / C 1.0364 against the decision record's 0.7012 / 0.7966 / 1.0292. Unresolved and kept separate.
- The predictions-file hashes of the two ActFound pIC50 attempts differ; unexplained.
- Code changes after commit `6519419` (`actfound_stage_adapters.py`, `actfound_stage_controller.py`, `actfound_stage_chain_run.py` and their three tests) are uncommitted, as are the new tarballs (untracked).
- The T121 planner duplicates the `prepare_trial_cells` layout and has not been compared byte-for-byte.
- Carried from Session 005, still open: the wiring list, the `d014_lr_penalty` folder name colliding with PXR D014, and the job 3103473 hash binding (S4).
