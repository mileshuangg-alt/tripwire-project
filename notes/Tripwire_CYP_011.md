# Tripwire-CYP-011 Session Log

**Date:** 2026-11-26 (opened)  
**Status (current):** Open. D027 (descriptive screen-response reporting scope), D028 (CYP potency source: 150-member updated-T121 mean; writing scope) and D029 (A/B penalty design and evaluation definitions; record of the writing approval) are recorded. For the report work, CYP and PXR potency outputs exist for the 208 input rows and the report join has been run locally (208 rows; sections 2c and 8); nothing is committed. Separately, the A/B penalty arms (D029, section 5a) are renamed to `d029`, previewed and packaged locally, and uncommitted; the wall-time sizing is decided (D029 Amendment 1). The package is transferred to CoreHPC staging, the login-host load check passed (pasted output) and the environments for both arms are written. The input-file check passed, and cell 0 of both arms passed the integrity check (pasted output). Under D029 Amendment 2, all 150 cells of both arms, validation-only inference and aggregation are complete (pasted output). Development-only result: both arms lower mean macro ST-RAE (`twosided` 0.669120, `huber` 0.669652, unchanged 0.676954) but fail adoption clauses (c) and (d), so unchanged updated T121 stays selected (section 5a). The penalty code, configs and tests remain uncommitted. The test run of the renamed files and the transfer hash checks are reported as expected, without pasted output. D030 (label-rank mirrored band penalty; design, evaluation definitions and implementation approval) is also recorded: its code, configs and tests are applied to the checkout and uncommitted, tests are reported passed (output not pasted), the local 50-fold preview is done and the D030 package is built locally. D030 Amendment 1 (2026-11-27) records a one-off wall-time sizing and approves the D030 execution scope. The D030 run has completed through validation-only inference and aggregation (2026-11-27). Development-only result: `mirrored` raises mean macro ST-RAE (0.679699, unchanged 0.676954) and fails all four adoption clauses, so unchanged updated T121 stays selected. The package-verification, load-check and input-file-check outputs have not been shown, and whether the cell 0 integrity check had passed before cells 1-149 were submitted is not reported (section 5c).

## 1. Session purpose and starting state

Session 011 follows Session 010, which was closed on declaration on 2026-11-26 (`notes/Tripwire_CYP_010.md`). It takes up the screen-response report version now being prepared, which exposes the existing screen-head predictions beside the unchanged potency outputs.

Starting state, from the 010 records:
- D024 three-arm evaluation complete; unchanged updated T121 selected.
- D025 (with Amendments 1-3) and D026 (with Amendments 1-2) recorded; screen-prediction script run on CoreHPC on the 208-molecule set; descriptive evaluation on 197 molecules and a CYP-tier-split diagnostic done (descriptive only).
- Uncommitted at 010 close: the D025/D026 scripts, tests, pins, analysis outputs and the record edits made after commit `99a3301`.

## 2. Reporting scope decision (D027, 2026-11-26)

D027 in `notes/Decisions.md` (Approved as a reporting scope only; no code, run or trigger authorized) fixes what the report version shows: the four native-unit CYP screen log2fc predictions from the r1 `epoch_016` checkpoint and the PXR 8.25 uM values of the three stage-1 seeds with their mean, no separate spread column, placed beside the five unchanged potency outputs and not blended. It has no automatic trigger, review flag, top-fraction queue or native-unit cutoff and no claim of confirmed PXR-specific activity. Seed differences are described only as model disagreement.

D026 supports descriptive error and ranking performance on the 197 molecules, which were excluded from training and selection of both screen heads (pretrained-encoder overlap unchecked); it does not validate the combined report or any trigger. The CYP train and validation tiers refer to the potency models, not screen-head membership, which supersedes the earlier tier-split concern to that extent.

## 2a. CYP potency source decision (D028, 2026-11-26)

D028 in `notes/Decisions.md` (Approved as a report-input scope and a writing authorization; no training, inference run, transfer, Slurm submission or execution by the assistant) fixes the CYP potency columns of the screen-response report. The updated-T121 150-checkpoint pool (array job 3105880; cells under `/scratch/user/mhuang8/tripwire_staging/t121_production_chain_20261117_cells_run/trials/trial_121/cells`) is reused without training, and the CYP potency value per enzyme is the equal-weight mean over all 150 members on the original pIC50 scale. All 150 are required; a member failing its hash or identity check stops the run; the member SD is descriptive and not a report column. This extends the D007 post-deadline blind-inference ensemble rule, whose text covers only the frozen D007-T121 checkpoints, to the updated-T121 pool. It selects no final challenger, does not make the comparison final and does not alter D007 or D008. The PXR potency columns follow D018 Amendment 6 and D027, with no new PXR decision. The 208 molecules are CYP train/validation-tier molecules on which most members trained, so the CYP values are deployment predictions and are labelled so, not held-out predictions.

Writing approved (these files only): `cyp/scripts/t121_potency_predict.py`, `pxr/scripts/pxr002_predict_frame.py`, `cyp/scripts/build_screen_report.py`, a test file for each, and a PXR input-artifact builder. Inference jobs, transfers and Slurm submissions remain unapproved; per-unit timing and extrapolated total against requested wall time are required before any submission. None of these files has been written or run.

## 2c. D028 files written and locally tested (2026-11-26)

The D028 files are written locally (uncommitted): `cyp/scripts/t121_potency_predict.py`, `pxr/scripts/pxr002_build_predict_frame.py`, `pxr/scripts/pxr002_predict_frame.py`, `cyp/scripts/build_screen_report.py` and their tests. Source: the project owner's report of local pytest runs in the `tripwire_cyp002` interpreter (console output not pasted). The first run over the three test files gave 40 passed and 2 failed (10.17 s); both failures were in the CYP potency EndToEnd tests and came from the test fixture writing the repr of a numpy float64 into the reference CSV, not from the script. After a one-line fixture fix the CYP potency test file ran 19 passed (10.11 s); the other two files' results are those of the first run (all passed). The earlier plan expectation of 22 tests in the CYP file was a miscount; the file has 19. File hashes (current and superseded) are in `notes/Tripwire_CYP_011_supporting.md`.

**Timing-only mode (added after approval).** `--timing-only-cell N` in `cyp/scripts/t121_potency_predict.py` (one member through build, reference gate and input prediction; writes only a timing record; `--cells` must still name 150 cells) and `--timing-only-checkpoint variant:seed` in `pxr/scripts/pxr002_predict_frame.py` (one of the ten approved models; writes only a timing record) were added, with `TimingOnly` test classes in the two potency test files. Reason: measured interactive timing is required before any submission, and the scripts otherwise could not run a single member. After the additions the owner reported all tests passing for the three test files; counts were not supplied. The earlier 40-passed, 2-failed and 19-passed figures are from before this addition.

**CoreHPC read-only lookups (reported 2026-11-26, login host).** 150 cells with `proposal.json` and 302 reference files are present under the updated-T121 cell root. The dataset and split-assignments hashes equal the local files, which closes the Session 007 equality check that had never been reported. The executor source in the production-chain package equals the pinned value (the open lookup in section 2b is resolved). The PXR final root `/scratch/user/mhuang8/pxr002_final_20261118` holds `final_plan.json` and ten model directories, each with `best.ckpt`, `manifest.json` and `metrics.json`; the PXR package is `/scratch/user/mhuang8/tripwire_staging/pxr002_stage_20261118_r6`. Hashes are in the supporting record. No inference, transfer or job has run, and no decision was made.

**First CoreHPC timing attempt for one CYP member (cell 0; failed at the reference gate).** Login host CPU, 2 threads, batch 64. The run stopped at the reference gate: max absolute difference 5.071113783007601 over 560 compared rows against a tolerance of 1e-5. No timing or ensemble file was written; output directory `cyp_potency_timing_20261126` holds only the stop record. Cause, confirmed from the installed Chemprop 2.3.1 source: `RegressionFFN` defines `train_step` as `forward` and applies the output transform, which unscales in eval mode, so the script's own manual unscale doubled the scaling.

**Fix (approved).** `predict_native` now calls `model(bmg, V_d, X_d)` in eval mode and uses the native output. It first checks that the output transform is an `UnscaleTransform` not in training mode, with mean and scale agreeing with the cell's rebuilt scaler (atol 1e-5). A new regression test class `RealModelSeam` (a tiny real Chemprop MPNN with a real `UnscaleTransform`) and source-invariant additions (`train_step` and manual rescale forbidden) were added. A test-helper bug (the `drop_last` default dropped one molecule in the reference loader) was fixed in the test only. Local result reported by the owner: `cyp/tests/test_t121_potency_predict.py` 31 passed in 15.77 s. This supersedes the earlier 19-test and post-timing-addition figures for that file. The corrected script was transferred and the staged copy replaced (old copy kept as `.pre_unscale_fix_20261126`), with the hash verified by the owner before the run. Hashes are in the supporting record.

**Second timing run (one member, cell 0).** Same host and settings; output directory `/scratch/user/mhuang8/tripwire_staging/cyp_potency_timing_20261126b`. The reference gate passed: max absolute difference 8.064889511771867e-07 over 560 rows (tolerance 1e-5), 418 gate molecules, 208 input molecules predicted. Seconds: build and load 1.709; reference gate pass 0.927; input prediction 0.499; member total 3.135. Extrapolated 150 members: 470.3 s (one member times 150, cold start included; not a guarantee). No ensemble files were written and no full run has been made. The 1e-5 gate tolerance was an assistant proposal for the report run and was not exceeded. No decision was made.

**Full 150-member CYP potency inference (2026-11-26).** The updated-T121 pool was run on the CoreHPC login host (CPU, 2 threads via `OMP_NUM_THREADS` and `MKL_NUM_THREADS`, batch 64, detached run, process 4165577) with the corrected script (SHA-256 `1e236efe...0f37`) on the 208-row `screen_input_set.csv` (SHA-256 `ec850989...1537`). Output `/scratch/user/mhuang8/tripwire_staging/cyp_potency_run_20261126`, fetched to `analysis/cyp_011_potency_20261126/cyp_potency_run_20261126/` locally. Verified locally from the fetched manifest: 150 members; all 150 reference gates within the 1e-5 tolerance (maximum 1.050954927084291e-06); 208 of 208 rows ok, none invalid; member time total 457.19 s against the 470.3 s extrapolation; manifest output hashes equal recomputed hashes; manifest input hashes equal the pinned dataset, split, executor, input and protected-list hashes. Output: `cyp_potency_ensemble.csv` (208 rows; four `pred_mean150` and four `member_sd150` columns), `cyp_potency_members.csv` (31,200 rows = 150 x 208) and the manifest; hashes are in the supporting record.

Ensemble means per enzyme (min / mean / max, pIC50): CYP1A2 3.351 / 4.953 / 5.949; CYP2C9 3.085 / 4.736 / 6.063; CYP2D6 4.298 / 4.702 / 5.832; CYP3A4 1.875 / 4.522 / 5.953. Member SD (ddof=1) ranges 0.047 to 0.535. These are deployment predictions of the pool and most members trained on these molecules (D028), descriptive only, with no label comparison. The CoreHPC-side `ls` and `sha256sum` output was not pasted; the local hashes match the manifest. PXR potency is not yet generated and the report join has not been run. No decision was made.

**PXR potency inference (2026-11-26; D018 Amendment 6 models; D027/D028 report scope).** Run on the CoreHPC login host, CPU, 2 threads, detached (process 1719). A preceding one-model timing (multitask seed 20261120) gave a gate absolute difference of 1.326e-08 against the recorded validation MAE 0.5490965845, 13.36 s for one model, and an extrapolation of 133.6 s for ten (one model times ten, cold start included; not a guarantee). The package is a new copy of the frozen r6 package, `/scratch/user/mhuang8/tripwire_staging/pxr_predict_pkg_20261126` (r6 untouched), containing `pxr/scripts/pxr002_predict_frame.py` (SHA-256 `561274ce...5d3b`) and the artifact `pxr/artifacts/pxr002_predict_208_20261126` (208 rows; descriptors 208 x 200 float64, finite; ids and SMILES equal the mapping `pxr_row_id` and `pxr_source_smiles` in order; built locally with no labels read). Run output `/scratch/user/mhuang8/tripwire_staging/pxr_potency_run_20261126`, fetched to `analysis/cyp_011_potency_20261126/pxr_potency_run_20261126/`.

Verified locally by the assistant from the fetched manifest: 10 gates all passed (maximum absolute difference 2.19e-08, tolerance 1e-3); configuration `h600_d4_f2_p0.2`, variants multitask and aux_off, seeds 20261120-20261124; the ten checkpoint hashes equal the PXR-002 `final_predict_manifest`; `training_performed` false and `labels_read` false; manifest output hashes equal recomputed hashes; 2,080 seed rows (208 x 10) with exactly five seeds per row and variant; the means recompute exactly from the seed rows (0 mismatches); row ids equal the mapping `pxr_row_id` in order. Hashes are in the supporting record.

Descriptive ranges (pEC50, min / mean / max): multitask five-seed mean 2.118 / 4.569 / 5.712 (primary column); aux_off mean 2.111 / 4.624 / 5.743 (secondary). Multitask mean by role: primary_validation (160) 4.708; aux_only (48) 4.106. The 160 primary_validation molecules were the model-selection validation set (not trained on); aux-only molecules have no primary pEC50 label. No label comparison was made and no decision was made.

Not covered: the PXR tests use fake model loading; the real PXR path is now exercised on CoreHPC by the one-model gate and the ten-model run.

**Report join (run locally, 2026-11-26).** `cyp/scripts/build_screen_report.py` (SHA-256 `21a98e82...10e8`) was run by the project owner with the `tripwire_cyp002` interpreter; output `analysis/cyp_011_report_20261126/`. Inputs were hash-pinned: the 208-row screen predictions, the mapping file, `cyp_potency_ensemble.csv` and `pxr_potency_means.csv` (hashes in the supporting record). `screen_report.csv` has 208 rows and 22 columns in this order: five identifier columns (`OCNT_ID`, `cyp_molecule_name`, `pxr_row_id`, `pxr_role`, `pxr_eligibility_reason`); the potency block (four CYP `*_pIC50_direct_inhibition_pred_mean150` columns, then the PXR multitask five-seed mean as primary and the aux-off five-seed mean as secondary); the screen block (four `*_screen_log2fc` columns, three PXR 8.25 uM per-seed log2fc columns and their mean3); then `screen_status`, `join_status_cyp` and `join_status_pxr`.

Join counts (`join_report.json`, checked locally by the assistant): 208 rows in each of the screen, mapping, CYP potency, PXR potency and report tables; `join_status_cyp` matched 208, `join_status_pxr` matched 208, `screen_status` ok 208; no duplicate keys in any of the six keyed columns; no screen id missing from the mapping, no mapping row without a screen row, no potency id missing from the mapping; `cyp_molecule_name` equals `OCNT_ID` for all 208; no screen/CYP SMILES disagreements. Local check by the assistant: every screen, CYP potency and PXR potency value in the report equals its source cell as text (0 mismatches); no empty cells; 208 unique ids; roles primary_validation 160 and aux_only 48. The 11 aux_only rows with reason `aux_only_group_contains_test_structure` remain in the 208-row report; the 197-molecule evaluation set is a separate subset and no evaluation was rerun. The report contains no combined score, spread column, flag, queue or cutoff (D027). The CYP potency values are deployment predictions of the 150-member pool, not held-out predictions (D028). The report has not been committed. Remaining follow-ups, none decided: committing the 011 files and any use of the report. No decision was made.

## 2b. Findings from read-only inspection (2026-11-26)

- The CYP potency cross-validation rows (`t121_valinf_*`) are not deployment inputs.
- No PXR Chemprop potency predictions exist for the 208 molecules: the PXR-002 final predictions cover only the 621 holdout molecules, and the PXR-003 baselines are classical comparators.
- The frozen-T121 blind-inference code is not in the repository.
- Open lookup (not a decision) at the time: the CoreHPC directory holding the executor source `cyp007_cell_fit.py` (SHA-256 `625a64fa...1a00b`, as supplied in abbreviated form) and the dataset and split hash-check outputs. Resolved later on 2026-11-26 (section 2c): the executor source equals the pinned value and the dataset and split hashes equal the local files.

## 3. Correction to the D026 descriptive reading

On the 197 molecules, CYP2D6 has the weakest overall baseline-relative MAE (ratio 0.938) and the weakest Spearman (0.478); CYP2C9 has the lowest MAE ratio (0.550); CYP3A4 has the highest Spearman (0.701) with MAE ratio 0.585 (`notes/Tripwire_CYP_010.md` section 11). MAE alone does not establish classification changes near any cutoff. The 010 record carries this wording and the D027 supersession note.

## 4. Conditional requirements for any later trigger (notes only, not decisions)

- An opposing-predicted-signal flag needs a stated rule and residual and classification analysis near that rule.
- A batch-relative review queue is a different feature and needs reviewer-supplied capacity and direction.
- A native-unit cutoff, or any hit or specific-activity label, needs its own definition and evidence.

## 5. Proposed next step (proposal status, not a decision)

Prepare a join and report implementation plan using the existing explicit CYP/PXR mapping (`analysis/cyp_010_screen_input_set_20261125/screen_input_mapping.csv`, columns `OCNT_ID`, `cyp_molecule_name`, `pxr_row_id`, and others). `input_id`, `Molecule_Name` and `row_id` are not interchangeable; unmatched and duplicate mappings are reported with counts, not dropped or guessed. No edit or execution for it is approved yet, and the approved A/B penalty experiments are unchanged.

## 5a. Penalty-form arms (A/B): implementation, package, CoreHPC run and development result (2026-11-26)

A topic separate from the report work. The design source is the A/B penalty plan in `notes/Handoffs/Tripwire_CYP_010_Handoff_20261126.md` section 3.1. D029 in `notes/Decisions.md` (Approved as design and evaluation definitions and as the record of the writing approval; no training, inference, transfer, benchmarking, GPU job, Slurm submission, commit or execution by the assistant) records this penalty design. Its Amendment 2 (approved execution scope, below) closes, to that extent, the bar on training, inference and Slurm submission in D029 and Amendment 1.

**Design.** Two exploratory arms against unchanged updated T121, both with the D024 response-rank weights and λ = 1:
- A (`twosided`): two-sided squared confidence-band penalty. Missing or incompatible bounds stop A.
- B (`huber`): upper-side penalty equal to 2 x Huber, with δ the per-enzyme, per-training-fold median standardized confidence half-width over observed training cells. A missing, non-finite or zero δ stops B.

Both use all 150 matched r1 cells, the saved unchanged baseline and the same four adoption requirements as D024; if both pass, the lower macro ST-RAE wins. An exact tie goes to `twosided`: this rule was proposed by the assistant, and no objection was raised when writing the implementation was approved.

**Code (uncommitted, this checkout).** Writing the tests, and then the implementation, was approved and done in this session.
- `cyp/scripts/t121_screening_penalty.py`: the two new forms, the bound and δ stop checks, and a preview comparison against a reference preview.
- `cyp/scripts/t121_d024_aggregate.py`: arm names are read from the evaluation config.
- `cyp/artifacts/d029_penalty/`: two arm configs and `evaluation.json`.
- `cyp/scripts/t121_d029_layout.py` and `cyp/scripts/t121_d029_build_package.py`: no tests; the builder produced the package below.
- `cyp/tests/test_t121_penalty_forms.py`.
- The training module, cell runner, Slurm script, executor and inference script are unchanged.

**Naming collision (closed).** The penalty files are now named `d029`, matching D029. They were earlier named `d027` and then `d028`, which collided with the unrelated D027 and D028 records. The report work's D028 files are unaffected.

**Tests** (source: pasted pytest output; pytest 9.1.1, Python 3.12.14; five files covering the new penalty forms, screening penalty, D024 aggregate, D024 layout and production cell): 90 collected, 89 passed, 1 failed. The failure was a `TypeError` from a helper signature in the new test file (a test-file bug, not penalty code). The helper was then fixed (positional-only first argument; test file SHA-256 at that point `2f4f748b...0a40`). A later test run of the renamed files is reported as expected (expected result: 90 passed); its console output was not pasted, so no count or output from it has been seen, and the 89 of 90 run is the only supplied output.

**Local weight and bound preview with the final configs** (50 folds, CPU, no model; source: pasted console output, with the report read back locally): `analysis/cyp_011_penalty_preview_20261126_d029/preview.json`, SHA-256 `77cd3a5451add98d1f7729ff82c54dcd903b49f88f1b844723e08477008af0e5`.
- 50 folds for each arm; disabled 0; SMILES differences 0; bound stops 0; no arm stopped.
- Weights equal the D024 rank preview in every fold; screening input hashes equal.
- Median half-width ranges are as in the earlier preview (table below). That earlier run, with the `d028`-named configs (`analysis/cyp_011_penalty_preview_20261126/preview.json`), also reported 366-391 CYP3A4 labelled molecules per fold without a screening row, as in D024.

| Enzyme | Median native confidence half-width over folds (pIC50) |
|---|---|
| CYP1A2 | 0.161731-0.166918 |
| CYP2C9 | 0.255323-0.267217 |
| CYP2D6 | 0.135017-0.136884 |
| CYP3A4 | 0.180731-0.195033 |

Final config SHA-256: `twosided` `347d48dac367db193152ffba83a32b8413ed358a681ccff88bb251818530232b`, `huber` `8f7dce50c7f980bdfd44414fe2e38c0119ac5606c081e62676481fc98b25a6af`.

**Local package** (source: pasted builder output, then verified locally): `t121_d029_build_20261126/t121_d029_package_20261126.tar.gz`, SHA-256 `ef84a8022b8d597cf38d137c2bcd198fb44b2e1774acdc1ea66da319a58051ce`, 121 files in its `SHA256SUMS.txt`.
- Every listed file and archive member matches the sums; all 36 overlay files equal the checkout.
- Compared with the D024 package, the only changed files are the penalty module (SHA-256 `b6d72307...0c34`), the aggregator (`e7b712f5...f44d`) and `PACKAGE_SCOPE.txt`. Added: three `d029` configs, the `d029` layout and builder, the new test file and `d029_pins.json`. Removed: `d024_pins.json`.

**Wall-time sizing** (D029 Amendment 1 in `notes/Decisions.md`; sizing only, authorizes no run or transfer). A one-off deviation for the D029 comparison: sizing from past D024 batch timings, cell 0 of each arm first, 12 h per-cell request retained. Extrapolated about 4.8-4.9 GPU-hours per 150-cell arm; the added compute of the D029 terms is not measured.

**CoreHPC staging and load check.**
- Package `t121_d029_package_20261126` was transferred to `/scratch/user/mhuang8/tripwire_staging/` on CoreHPC. The archive hash check and the `ALL 121 FILES OK` step are reported as expected; their output was not pasted.
- Load check on the CoreHPC login host (source: pasted output): the 14 listed modules import from inside the package with hashes matching its `SHA256SUMS.txt`; both frozen configs load against their pinned SHA-256 (`twosided`: `two_sided_squared`, `response_rank`, λ 1.0; `huber`: `upper_huber`, `response_rank`, λ 1.0); `ALL CHECKS PASSED`. The library-version line was not in the pasted part (reported as expected).
- Environments for both arms were written to `/scratch/user/mhuang8/tripwire_staging/t121_d029_env_20261126`, with 150 cells planned for each arm under `t121_d029_twosided_cells_run` and `t121_d029_huber_cells_run` (source: pasted output).

**Input-file check on CoreHPC** (source: pasted output): passed. The dataset and original split are OK, and the combined split-manifest digest `de2aea27a9b286290111f30d344ac791ecdc17cba0aa386556eff4789ce47921` equals that of the local copies. The pasted order does not establish whether the check was run before or after the cell 0 submission.

**Cell 0 submission.** Batch jobs 3274350 (`twosided`) and 3274351 (`huber`), with elapsed times 00:02:08 and 00:02:03 (source: pasted CoreHPC output). Both are within the D024 per-cell range; the per-cell request is 12 h.

**Cell 0 integrity check** (source: pasted output): `CELL 0 CHECK PASSED` for both arms (repetition 1, fold 1).
- Arm labels, config hashes and penalty forms checked; λ 1.0.
- Penalty weights equal the local preview (`3f2a9adb...364e`, the same as D024 rank cell 0); the screen checkpoint equals r1 `96b75ceb...`.
- No bound stops; one epoch-log line per epoch.
- For `huber`, the half-widths equal the preview and δ equals half-width over scale. It ran 35 epochs, with standardized δ CYP1A2 0.1613, CYP2C9 0.3323, CYP2D6 0.1462 and CYP3A4 0.1709.

| Arm | Base / penalty loss, epoch 0 | Base / penalty loss, later epoch |
|---|---|---|
| `huber` | 0.8405 / 0.0748 | epoch 34: 0.2334 / 0.0272 |
| `twosided` | 0.848 / 0.2959 | last logged epoch 36: 0.2239 / 0.0649 |

The `twosided` header line with the epoch count was not in the pasted text. These are integrity records only, not performance results.

**Execution approval** (D029 Amendment 2 in `notes/Decisions.md`; given after the cell 0 check passed; execution by the project owner). Scope: cells 1-149, validation-only inference (cells 0-149) and aggregation for both arms. It does not authorize the r2/r3 confirmation stage, a commit, execution by the assistant, protected data, an unadjudicated rerun of a failed cell, or any change to the arm definitions or evaluation rule.

**Cells 1-149, validation inference and aggregation** (source: pasted CoreHPC output).
- Cells 1-149 ran as arrays 3274362 and 3274363 (`twosided`, `huber`; the arm order follows the submission loop). Both arms have 150 cells completed and 0 incomplete.
- Validation inference: jobs 3275307 (`twosided`, COMPLETED, 00:04:32) and 3275308 (`huber`). The post-inference check found, for each arm, 150 prediction files, no `STOPPED_ON_MISMATCH.json` and 150 reproduced lines.
- Aggregation: `t121_d024_aggregate.py evaluate` with the D029 config, the frozen tail cohorts and the coverage file. It completed, so the unchanged arm reproduced its recorded O150. Report on CoreHPC (not copied to the Mac): `/scratch/user/mhuang8/tripwire_staging/t121_d029_report_20261126.json`, SHA-256 `6e8a39aee515e3263e43c20f69e8b407bbf3714a1a77fe9462f6b9da1afa579d`.

**Result** (development-only; lower is better). Both arms lower the mean macro ST-RAE, but neither passes the D029 adoption rule; the selected model is unchanged updated T121.

| Arm | Mean macro ST-RAE | Difference from unchanged | Clauses (a), (b) | Clauses (c), (d) |
|---|---|---|---|---|
| unchanged | 0.676954037 | - | - | - |
| `twosided` | 0.669120358 | -0.007834 | pass, pass | fail, fail |
| `huber` | 0.669652223 | -0.007302 | pass, pass | fail, fail |

Clauses (D029 item 7): (a) lower mean macro ST-RAE; (b) no worse enzyme x seed mean; (c) lower over-tail MAE for every enzyme; (d) no higher under-tail MAE for any enzyme. Pass or fail is from the aggregator output.

Per enzyme, each entry is ST-RAE / over-tail MAE / under-tail MAE:

| Enzyme | unchanged | `twosided` | `huber` |
|---|---|---|---|
| CYP1A2 | 0.796420 / 2.034759 / 1.141123 | 0.787918 / 1.868739 / 1.233180 | 0.786929 / 1.956170 / 1.179351 |
| CYP2C9 | 0.586615 / 1.155224 / 0.853427 | 0.580634 / 1.153767 / 0.857117 | 0.582305 / 1.137918 / 0.864218 |
| CYP2D6 | 0.874548 / 1.904297 / 1.240689 | 0.863303 / 1.793159 / 1.308456 | 0.865970 / 1.852136 / 1.272439 |
| CYP3A4 | 0.450233 / 1.405430 / 0.953722 | 0.444627 / 1.424333 / 0.917079 | 0.443405 / 1.420012 / 0.913512 |

All four per-enzyme ST-RAE means are lower than unchanged for both arms. The following breakdown of the failed clauses is read from the table, not from the report JSON:
- (c) fails for both arms only on CYP3A4 over-tail MAE (`twosided` +0.018903, `huber` +0.014582).
- (d) fails for both arms on CYP1A2, CYP2C9 and CYP2D6 under-tail MAE (`twosided` +0.092057, +0.003690, +0.067767; `huber` +0.038228, +0.010791, +0.031750).

**CYP3A4 by screening coverage** (descriptive, not an adoption clause; covered 1565, not covered 421).

| Group | Arm | MAE difference from unchanged | Mean cell ST-RAE difference from unchanged | Mean signed error: arm (unchanged) |
|---|---|---|---|---|
| covered | `twosided` | +0.010260 | -0.006061 | 0.198975 (0.172417) |
| covered | `huber` | +0.003010 | -0.006209 | 0.199361 (0.172417) |
| not covered | `twosided` | +0.004592 | -0.003132 | 0.118045 (0.089514) |
| not covered | `huber` | -0.001901 | -0.008283 | 0.129132 (0.089514) |

Mean signed error is more positive than unchanged in both groups for both arms.

**Limits.** This is development evidence on already-inspected partitions, with no significance claim. Tail membership comes from the baseline's own errors, so the tail changes are not independent evidence. No cause is established. No arm passed, so this result does not trigger the r2/r3 confirmation stage, which still requires separate approval.

**Open for D029.** The penalty code, configs and tests are uncommitted.

## 5b. Saved-prediction diagnostics after the D029 result (exploratory, 2026-11-26)

**Purpose.** Investigate the saved CYP validation predictions to inform a next experiment. No training, new inference, penalty sweep or change to the adoption rule. Arms: unchanged updated T121, `twosided` (A) and `huber` (B), with D024 rank as context, on the 150 matched r1 cells.

**Code and run.** Uncommitted, this checkout: `cyp/scripts/t121_penalty_prediction_diagnostics.py` (SHA-256 prefix `2612cc8c0161`), `cyp/scripts/t121_penalty_prediction_plots.py` and `cyp/tests/test_t121_penalty_prediction_diagnostics.py` (16 tests; no test run output supplied). Outputs are in `analysis/cyp_011_penalty_diagnostics_20261126/` (source: pasted console output, outputs read back locally; `run_metadata.json` SHA-256 prefix `d4d74dcb2804`, output hashes match it). No comparison stopped. Report tail MAEs reproduced for all arms (largest difference 4.4e-16), and the rebuilt CYP3A4 coverage matched the frozen grouping.

**Findings** (descriptive; lower is better; change = arm minus unchanged). Molecule-level rule: seed mean within a repetition, then mean over repetitions; absolute error is the mean of row absolute errors.

| Enzyme | MAE change: A / B / rank | Mean prediction change: A / B / rank |
|---|---|---|
| CYP1A2 | -0.0094 / -0.0072 / -0.0063 | -0.096 / -0.042 / -0.122 |
| CYP2C9 | -0.0012 / -0.0034 / +0.0000 | -0.005 / -0.012 / -0.019 |
| CYP2D6 | -0.0051 / -0.0041 / -0.0065 | -0.065 / -0.033 / -0.077 |
| CYP3A4 | +0.0091 / +0.0020 / +0.0051 | +0.027 / +0.030 / +0.022 |

- CYP3A4 MAE rose while its per-enzyme ST-RAE fell for A and B (section 5a), so MAE is not a decomposition of ST-RAE.
- Fixed tails: on CYP1A2 and CYP2D6 (and by small amounts on CYP2C9) over-tail MAE fell and under-tail MAE rose; on CYP3A4 the reverse. A and B lower the absolute error of 47-54% of molecules overall.
- Direction: on CYP1A2 and CYP2D6 all three groups (the two tails and the remaining molecules) move down on average; on CYP3A4 the upward move is largest at low unchanged predictions (this pattern can partly arise from run-to-run noise, because the unchanged prediction is on both axes).
- Concentration: in every enzyme and arm, half of each improvement or deterioration total comes from about 10-12% of all molecules, and the largest single molecule is at most 1.6% of a total. Remaining (non-tail) molecules hold about 68-81% of both totals.
- Seeds: on CYP1A2 and CYP2D6 the over-tail gain and under-tail loss recur in all three seeds for all arms. CYP2C9 under-tail MAE rises in all seeds; CYP2C9 overall is mixed across seeds for all arms, and the CYP2C9 over-tail of A is mixed. CYP3A4 under-tail MAE falls in all seeds; its over-tail is mixed for B and rank and rises in all seeds for A.
- Coverage (association only): only CYP3A4 has uncovered molecules (421 of 1986). Covered / uncovered mean prediction change: A +0.027 / +0.029, B +0.027 / +0.040, rank +0.022 / +0.023. Covered / uncovered MAE change: A +0.0103 / +0.0046, B +0.0030 / -0.0019, rank +0.0061 / +0.0012.
- Diagnostic counterfactual (each arm's per-enzyme mean change subtracted; it uses evaluation predictions and is not a correction, a candidate, causal evidence or adoption-eligible): on CYP1A2 and CYP2D6 the under-tail rise nearly vanishes (A +0.092 to -0.004 and +0.068 to +0.002; B +0.038 to -0.004 and +0.032 to -0.001), and about 37-47% of the over-tail gain remains. For B the shifted values are lower than unchanged overall and on both tails for all four enzymes; for A and rank this does not hold.

**Limits.** The development partitions were already inspected. Tails are selected from the unchanged model's own errors, so tail comparisons are baseline-conditioned. The arms are separate training runs, with no repeated unchanged run to size run-to-run variation. No significance claim is made and no next experiment is selected.

## 5c. D030 label-rank mirrored penalty: implementation, run and development result (2026-11-26 to 2026-11-27)

D030 in `notes/Decisions.md` (Approved as design and evaluation definitions and as the record of the implementation approval; its own text authorizes no training, inference, transfer, benchmarking, GPU job, Slurm submission, commit or execution by the assistant) records the `mirrored` arm. It follows the D029 result (section 5a). D030 Amendment 1 (2026-11-27, in `notes/Decisions.md`) records the one-off wall-time sizing and approves the D030 execution scope for the package below, closing the open timing item and lifting the bar on the listed execution items only; its conditions and exclusions, including no r2/r3 stage, no commit and no execution by the assistant, are stated there.

**Files** (uncommitted, this checkout; code, configs and tests applied under the implementation approval):

| File | SHA-256 prefix |
|---|---|
| `cyp/scripts/t121_screening_penalty.py` | `593ec1c46241` |
| `cyp/scripts/t121_d030_layout.py` | `c8bb59028017` |
| `cyp/scripts/t121_d030_build_package.py` | `6bb16042b6f6` |
| `cyp/artifacts/d030_penalty/penalty_label_mirrored.json` | `cc7afefcea42` |
| `cyp/artifacts/d030_penalty/evaluation.json` | `27f9142fd1a0` |
| `cyp/tests/test_t121_penalty_mirrored.py` | `78b60a3c6d5e` |

The layout and builder have no tests; the builder produced the package below.

**Tests** (source: the project owner's report; output not pasted): all passed, with 113 expected across six files.

**Local 50-fold preview** (source: pasted output, with the saved report read back): `analysis/cyp_011_penalty_preview_20261126_d030/preview.json`, SHA-256 prefix `f8319d8f89b4`.
- 50 folds; disabled 0; weight identity failures 0; bound stops 0; no screening file read.
- Ranked counts per fold: CYP1A2 1062-1101, CYP2C9 965-1008, CYP2D6 1123-1157, CYP3A4 1766-1815.
- Every fold-enzyme ranked set equals its labelled training molecules, and the weights equal an independent rank calculation.

**Local package** (source: pasted builder output, verified locally): `t121_d030_build_20261126/t121_d030_package_20261126.tar.gz`, SHA-256 `1b2cc7e97f803ca757873ce64f46fd0b3285d1e5ae7ddfc02179d80659b7b062`, 124 files in its `SHA256SUMS.txt`.
- All listed files and archive members match; all 39 overlay files equal the checkout.
- Compared with the D029 package, only the penalty module and `PACKAGE_SCOPE.txt` changed. Added: the D030 configs, layout, builder, test and `d030_pins.json`. Removed: the D029 layout, builder and pins.

**CoreHPC run** (executed by the project owner; source: pasted output unless stated). The package is the one above (SHA-256 `1b2cc7e9...`). Per-check evidence and identifiers are in `notes/Tripwire_CYP_011_supporting.md`.
- Cell 0: job 3285377 (task `3285377_0`), COMPLETED, elapsed 00:02:03 against the 12:00:00 limit.
- Cells 1-149: submitted as array 3285386 (job ID from pasted submission output; per-task accounting not supplied). A live monitor output showed D030 cells completed 150/150, incomplete 0.
- Validation-only inference: job 3285392; 150 prediction files and 150 "reproduced" statuses in the logs. The aggregation guard (no `STOPPED_ON_MISMATCH.json`) passed.
- Not shown: the package-verification output (`ALL 124 FILES OK`) and the load-check and input-file-check outputs (planned steps before cell planning) were never supplied. No pass is recorded for them, and no failure is reported.

**Cell 0 integrity check** (the D030 Amendment 1 condition; source: pasted output): all 14 checks OK, `CELL 0 CHECK PASSED`, 35 epochs. Covered checks include the config hash (`cc7afefc...`), a weights hash equal to the local preview (`bb4aec47...`), no screening input in the penalty, the r1 screening checkpoint (`96b75ceb...`), no bound stops, ranked counts 1069 / 980 / 1141 / 1799, and upper plus lower penalty sums and counts equal to the totals. These are integrity records only.

| Epoch | Base loss per observed cell | Penalty per observed cell (upper / lower) |
|---|---|---|
| 0 | 0.8397 | 0.9172 (0.2269 / 0.6902) |
| 34 | 0.237 | 0.1519 (0.0725 / 0.0794) |

Order relative to the condition: the passing output was supplied after cells 1-149 had completed. The array was submitted before any check output was shown, and whether the check had passed before that submission is not reported (UNKNOWN). D030 Amendment 1 item 5 conditions cells 1-149 on cell 0 passing.

**Aggregation report:** `analysis/t121_d030_report_20261127.json`, SHA-256 `9d8e4e8bfe07c3dabe586e15ea78cbc793799052f3d60996eee79e0eaf160825`, identical on CoreHPC (pasted `sha256sum`) and in the local copy (hashed locally). The first aggregation terminal output was not supplied; a rerun refused to overwrite the existing report. The local copy parses completely; its config, cohort, coverage and aggregator-script hashes match the pinned local values; the unchanged-arm metrics and prediction hashes are identical to those in the D029 report; the baseline reproduction difference is 0.0.

**Result** (development-only, exploratory; lower is better; read from the report). `mirrored` does not pass the adoption rule; the selected model is unchanged updated T121.

| Arm | Mean macro ST-RAE | Difference from unchanged | Clauses (a), (b) | Clauses (c), (d) |
|---|---|---|---|---|
| unchanged | 0.676954 | - | - | - |
| `mirrored` | 0.679699 | +0.002745 (worse) | fail, fail | fail, fail |

- (b): 8 of 12 enzyme x seed means are worse (CYP1A2 all three seeds, CYP2C9 all three, CYP2D6 seed 20261101, CYP3A4 seed 20261103).
- (c): over-tail MAE is higher on CYP2C9 and CYP3A4.
- (d): under-tail MAE is higher on CYP1A2 and CYP2D6.

Per enzyme, each entry is ST-RAE / over-tail MAE / under-tail MAE:

| Enzyme | unchanged | `mirrored` | `mirrored` minus unchanged |
|---|---|---|---|
| CYP1A2 | 0.796420 / 2.034759 / 1.141123 | 0.799665 / 1.915845 / 1.204503 | +0.003245 / -0.118914 / +0.063380 |
| CYP2C9 | 0.586615 / 1.155224 / 0.853427 | 0.595438 / 1.183927 / 0.847010 | +0.008823 / +0.028703 / -0.006416 |
| CYP2D6 | 0.874548 / 1.904297 / 1.240689 | 0.873479 / 1.840731 / 1.248900 | -0.001070 / -0.063567 / +0.008211 |
| CYP3A4 | 0.450233 / 1.405430 / 0.953722 | 0.450216 / 1.452467 / 0.903310 | -0.000017 / +0.047037 / -0.050412 |

Observation computed from the report: within each enzyme, mean signed error changed in the same direction in both fixed tails (over-tail, under-tail): CYP1A2 -0.119, -0.063; CYP2C9 +0.029, +0.006; CYP2D6 -0.064, -0.008; CYP3A4 +0.047, +0.051. Reading this as an enzyme-level shift is an interpretation; the overall mean prediction change was not computed for D030.

**CYP3A4 by screening coverage** (D030 item 9; descriptive, not an adoption clause; `coverage_groups` in the report; covered 1565, not covered 421). Each entry is `mirrored` minus unchanged.

| Measure | Covered | Not covered |
|---|---|---|
| MAE | +0.016290 | +0.007067 |
| Mean signed error | +0.043815 | +0.032453 |
| Over-tail MAE | +0.048673 | +0.033173 |
| Under-tail MAE | -0.052893 | -0.037463 |
| Mean ST-RAE | -0.000581 | +0.001726 |
| ST-RAE, seed 20261101 | -0.000391 | -0.001767 |
| ST-RAE, seed 20261102 | -0.002239 | +0.002730 |
| ST-RAE, seed 20261103 | +0.000887 | +0.004216 |

Group sizes differ, so group means are not directly comparable between groups. The over-tail and under-tail groups have 178 and 167 molecules (covered) and 21 and 32 (not covered).

**Limits.** This is development evidence on already-inspected partitions, with no significance claim, and the design follows inspected validation results. Tail membership comes from the baseline's own errors, so the tail changes are not independent evidence. D030 Amendment 1 authorizes no r2/r3 stage and no commit. No next experiment is decided.

## 6. Carried from Session 010 (none decided)

- D025 Amendment 3 versus D026 label-use reconciliation remains unanswered.
- The PXR hit definition, the review trigger and thresholds, and a training-only median baseline or confidence intervals are not decided (an explicit decision entry is required for each).
- Also carried per `notes/Tripwire_CYP_010.md` section 13: the 009 handoff correction on validation prediction scaling (needs owner approval), the r2/r3 confirmation stage, the 6,251 versus 6,252 `n_train_labels` gap, and the pytest output for the `evaluate_screen_predictions` tests (never supplied).

## 7. Claim boundary

Everything on the screen heads is descriptive and development-side; it is not challenge performance, not validation of the combined potency-plus-screen report, and not evidence of confirmed PXR-specific activity. No trigger or threshold exists.

## 8. Stopping point and next step

Session opened 2026-11-26; D027, D028 and D029 are recorded; the report files named under D028 (section 2a) are written locally and pass their local tests (section 2c). CYP potency (150-member updated-T121 ensemble) and PXR potency (five-seed multitask mean, aux_off secondary) both now exist for the 208 input rows, each run on the CoreHPC login host with all reference gates passed and the outputs fetched and hash-verified locally (section 2c). The report join (`cyp/scripts/build_screen_report.py`) was run locally and produced the 208-row, 22-column `screen_report.csv` with all joins matched and no duplicates or unmatched rows (section 2c). The 011 report files and the report are uncommitted. Remaining for the report work, none decided: committing the 011 files and any use of the report. Penalty arms (section 5a): the D029 comparison has run through aggregation; neither arm passed the adoption rule and unchanged updated T121 stays selected. The r2/r3 confirmation stage is not triggered. The penalty code, configs and tests are uncommitted; D029 Amendment 2 authorizes no commit. Exploratory saved-prediction diagnostics on these arms are in section 5b (descriptive only, diagnostic code uncommitted; the diagnostics themselves select no decision, gate or next experiment). D030 (section 5c): recorded; the code, configs and tests are in the checkout and uncommitted, tests are reported passed (output not pasted), the local 50-fold preview showed no disabled folds, weight identity failures or bound stops, and the D030 package is built and verified locally. Under D030 Amendment 1, the run has completed through validation-only inference and aggregation; `mirrored` fails all four adoption clauses and unchanged updated T121 stays selected (section 5c, which also states the check outputs not shown and the unreported order of the cell 0 check and the array submission). No r2/r3 stage or commit is authorized by that amendment, and no next experiment is decided.
