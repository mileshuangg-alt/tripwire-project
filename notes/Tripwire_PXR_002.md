# Tripwire-PXR-002 Session Log

**Date:** 2026-11-17 to 2026-11-18
**Status:** CLOSED on Miles's declaration, 2026-11-18.

**Summary (final).**
- **Objective.** Multitask Chemprop with one primary pEC50 head and three auxiliary log2FC heads (8.25, 33, 99 uM) on the frozen D014 split (D017; D018 Amendments 1-6).
- **Last completed step.** The single holdout evaluation of 621 molecules (§5l), evidence class internal-retrospective: multitask five-seed ensemble MAE 0.4843 / Spearman 0.7357; auxiliary-off MAE 0.4847 / Spearman 0.7367; constant training-mean floor MAE 0.9237 (descriptive comparison per D018 Amendment 5). The variants are indistinguishable. Cohort v2 (OADMET-0003727 excluded; 10,818 training molecules); stage B selected h600_d4_f2_p0.2 (not separable from the other two); the holdout is consumed.
- **Next step for a future session.** None authorized or started. Candidates for Miles to direct are listed under pending decisions.
- **Pending decisions / open items.** Official-test evaluation and use of firewalled test labels; retraining on train plus validation; standardization (unchanged, released SMILES); PXR-001 items (crude and semi-pure HT-chem eligibility, the `OCNT-2395416` == `OADMET-0006234` match); holdout MAE (0.48) lower than validation (about 0.54) at similar floor MAE, not investigated; early unpasted steps (tests, holdout-artifact build) UNKNOWN; the `notes/Tripwire_CYP_002.md` line-527 `$$` block is still unconverted; Rev 3 not amended (not a repository file); individual hashes of the r2-r6 packages were never recorded.
- **Active constraints.** The holdout must not be used for tuning or selection; no official-test use; no further holdout evaluation; nothing beyond what D018 Amendments 1-6 authorized.
- **Artifacts.** Evaluation-artifact hashes and commits are in `notes/Tripwire_PXR_002_supporting.md`; decisions in `notes/Decisions.md` (D017, D018). Commits 59fe1c0, fa31dab and aca631d are local and not pushed.
**Workstream:** Tripwire — PXR induction arm; continuation after Tripwire-PXR-001 (closed 2026-11-15; `notes/Tripwire_PXR_001.md`).
**Execution target:** local checkout plus CoreHPC staging and batch job 3106209 (runtime estimate, failed).

## 1. Session purpose and starting state

Session start declared by Miles on 2026-11-17, using the PXR-001 log and a continuation handoff (`/Users/mileshuang/Downloads/Tripwire_PXR_002_Claude_Handoff.md`, not in this repository, hash UNKNOWN) as prior context. The handoff's request to restart a PXR leaderboard review is superseded: Miles states that the review was completed on 2026-11-17 in another conversation. Its report is a Google Doc titled openadmet_pxr_model_report (https://docs.google.com/document/d/1OY1OPicY4OD6uFvbqvXGfeLcRJbxTRY6Ds8uXZusBR4/edit, sign-in gated). Its contents are not in this repository or visible to Claude Code; statements about it here rest on Miles's report only.

Starting state is the PXR-001 close: release pinned, audit complete, D014 holdout frozen (`pxr/artifacts/pxr001_split/pxr001_split_assignments.csv`, SHA-256 `c6028f1b7216ab2ea4bab96616fc02b95ce011be20b1b96f807d1e7d582ece46`; train 2,897 / validation 621 / holdout 621). Governing decisions: D014 with Amendments 1–4, D017 with Amendment 1, and D018.

## 2. Modeling direction (D017)

D017 (`notes/Decisions.md`, Approved, scope limited to modeling direction): multitask Chemprop with one primary pEC50 head and auxiliary log2FC heads, informed by component 1 of the winning matcha-croissant entry in the report above (not a reproduction of its full ensemble). The ECFP4 gradient-boosting/random-forest ladder and a stock-Chemprop-first run are skipped. This supersedes the PXR-001 next step (Rev 3 line 90 mandatory ECFP4 + gradient-boosting-first baseline); Rev 3 is a Downloads file and is not amended. The frozen D014 split, MAE (primary) with Spearman, and evaluation discipline are preserved. CYP's staged-training and uncertainty-weighting recipe is not transferred automatically.

**Head-count correction (D017 Amendment 1).** The handoff's "four auxiliary heads" was an error. The data have three library concentrations (8.25, 33 and 99 µM), so the model has one primary pEC50 head and three auxiliary log2FC heads, four outputs in total. Positive-control rows (27 rows at 0.98 µM, compound_class Positive Control) are excluded. The 99 µM readout occurs only in experiment OPA_PXR-030, so concentration and experiment effects cannot be disentangled for that head.

## 3. Design recipe (D018, design only)

Approved by Miles on 2026-11-17 as design; authorizes no execution (Miles runs execution).

| Topic | Approved design |
|---|---|
| Auxiliary data | Released library-compound single-concentration measurements (`single_concentration_TRAIN.csv`, `log2_fc_estimate`, keyed on OCNT_ID). Counter-assay, crude HT-chem and semi-pure data left out of this recipe. |
| Leakage | Auxiliary labels on primary validation and holdout molecules kept out of training. Auxiliary-only molecules included only if they pass the existing D014 scaffold/connectivity-group exclusions against validation, holdout and official-test structures. D014 split and grouping preserved; no new similarity cutoff. |
| Selection | Checkpoints chosen on primary-head validation MAE, not pooled multitask loss. Holdout untouched until final evaluation; official-test labels not used for selection. |
| Weighting | Nominal primary : combined-auxiliary allocation 65:35, not tuned; equal coefficients across the three auxiliary heads (count-proportional nominal allocation among them). Auxiliary coefficient calibrated relative to the primary from verified eligible label counts and Chemprop 2.3.1's actual loss reduction (sum(L·task_weight·mask)/sum(mask); task weights scale only the numerator), not by entering raw 0.65 and 0.35. Nominal allocations, not guaranteed shares of loss or gradient. Train-only target scaling preserved. |
| Inputs | Molecular graph plus `v1_rdkit_2d_normalized` descriptors (Miles's descriptor-set choice; not a claim it matches the winning entry's undocumented set). |
| Tuning | Full grid hidden {300, 600} × depth {3, 4, 5} × FFN layers {1, 2} × dropout {0, 0.1, 0.2} = 36 configurations, one common seed. Top three compared across three seeds by mean primary-head validation MAE. Max 100 epochs, early stopping patience 20 on primary-head validation MAE. |

## 4. Implementation and frozen auxiliary eligibility artifact

**Implementation.** Claude Code wrote new files only, under the plan Miles approved; no existing file was edited and nothing is committed: `pxr/scripts/pxr002_aux_data.py`, `pxr002_loss_weights.py`, `pxr002_dataset.py`, `pxr002_train.py`, `pxr002_grid.py`, `pxr002_runtime_estimate.py` and `pxr/tests/test_pxr002.py`. Miles ran them in the local conda env `tripwire_cyp002` (Chemprop 2.3.1, RDKit 2026.03.6). `pxr/tests/test_pxr002.py` passed in full (exact pytest summary line not reported). Nothing has been trained and no runtime measurement exists.

**Eligibility artifact build (succeeded), `pxr/artifacts/pxr002_aux/`:**

| Quantity | Value |
|---|---|
| Train molecules | 10,819 (2,897 primary-train + 7,922 eligible auxiliary-only) |
| Validation molecules | 621 |
| Train labels | pEC50 2,897; log2fc_8p25uM 9,737; log2fc_33uM 8,540; log2fc_99uM 606 (auxiliary total 18,883) |
| Library molecules total | 10,869 |
| Auxiliary-only excluded | 204 = 49 sharing a D014 group with an official-test structure + 155 sharing a group with validation or holdout molecules |

These match the earlier read-only scan counts. Artifact file hashes, implementation-file hashes and the v1 stage archive (`pxr002_stage_20261117.tar.gz`, transferred and extracted on CoreHPC at `/scratch/user/mhuang8/tripwire_staging/pxr002_stage_20261117`; job 3106209 verified the 13 staged files there) are recorded in `notes/Tripwire_PXR_002_supporting.md`.

**Nominal-allocation coefficient (v1, superseded).** From the v1 counts, with primary coefficient 1, the auxiliary-head coefficient was (35/65) × 2897 / 18883 = 0.0826099 (computed by Claude Code). Superseded by the v2 value in §5a.

## 5. Findings during implementation

- The project `.venv` (Python 3.15, RDKit 2026.09.2) fails Bemis-Murcko scaffold decomposition ("bad any cast"); PXR scaffold work therefore uses `tripwire_cyp002`.
- The library single-concentration table has no duplicate (molecule, concentration) rows. The code refuses duplicates; averaging is not approved.
- 730 of 5,703 library rows on primary molecules carry SMILES that differ textually from the primary table. Primary-train molecules use the released primary SMILES; auxiliary-only molecules use the single-concentration SMILES.

## 5a. Runtime-estimate attempt and descriptor failure (2026-11-17 evening, CoreHPC)

- **Interactive allocation failed.** The interactive srun allocation failed with repeated "Insane message length" socket errors and queued stray jobs 3106182, 3106184, 3106189, 3106196 and 3106200. A squeue check reported by Miles on 2026-11-18 showed none of these still present; the only job running was the unrelated 3105596 (actfound-d015-chain, CYP workstream). How they ended (cancelled or otherwise) is not recorded.
- **Batch deviation.** Because srun failed, Miles chose to submit the runtime estimate as a batch job, a deviation from the interactive-benchmark practice. The wrapper `pxr/scripts/pxr002_runtime_estimate.slurm` is new, written by Claude Code; its local hash was not reported.
- **Job 3106209** (node ggpu1-12, NVIDIA L40S 46068 MiB, driver 595.91.07; start 2026-11-17T23:26:43, elapsed 00:03:11): State FAILED, ExitCode 1:0. All 13 staged files passed `sha256sum -c` against the reviewed list; chemprop 2.3.1. Then a ValueError in `pxr002_dataset.compute_descriptors`: 4 non-finite descriptor values in 1 molecule (imputation was not approved by design). No timing was measured; the runtime estimate remains outstanding.
- **Diagnosis (local, `tripwire_cyp002`, by Claude Code).** The molecule is OADMET-0003727 (auranofin, primary_train; SMILES contains Au, P, S). The four descriptors are positions 39, 41, 43, 45 (MaxAbsPartialCharge, MaxPartialCharge, MinAbsPartialCharge, MinPartialCharge) of `v1_rdkit_2d_normalized`. The raw `v1_rdkit_2d` values are already NaN, so the fault is in the raw descriptor, not the normalization. No other molecule in the training or validation frames is non-finite.
- **Decision (D018 Amendment 1, 2026-11-18).** Exclude this molecule from the descriptor-based cohort rather than impute, via a versioned one-row eligibility amendment. Frozen `pxr002_aux` and the D014 split stay untouched. Same cohort for any auxiliary-isolating comparison; auxiliary coefficient to be recalculated from amended counts (the 0.0826099 in section 4 is superseded); descriptors cached once before learned scaling, learned scaling training-only; runtime pilot to be rerun before any production training. Design/preparation only; no training authorized. 

**v2 amended artifact (built locally 2026-11-18, `pxr/artifacts/pxr002_aux_v2/`).** Built by `pxr/scripts/pxr002_artifact_v2.py` from the frozen v1 artifact (v1 `SHA256SUMS` hash recorded as parent in the supporting record; v1 not modified), in `tripwire_cyp002`. `pxr/tests/test_pxr002.py` passes in full after the v2 additions (exact summary line not reported).

| Quantity | v2 value |
|---|---|
| cohort_version | v2 |
| Excluded | exactly one row, OADMET-0003727 (pEC50 label 2.54; no auxiliary labels) |
| Train molecules | 10,818 |
| Validation molecules | 621 |
| Descriptors | cached before scaling; 200 `v1_rdkit_2d_normalized` values per molecule |
| Train labels | pEC50 2,896 (v1: 2,897); log2fc_8p25uM 9,737; log2fc_33uM 8,540; log2fc_99uM 606 (auxiliary total 18,883, unchanged) |

Recalculated nominal-65:35 auxiliary coefficient (computed by Claude): (35/65) × 2896 / 18883 = 0.0825814, superseding 0.0826099. The trainer recomputes it from the artifact at run time. v2 file hashes not yet reported.

## 5b. v2 runtime pilot (2026-11-18, CoreHPC)

The v2 package (archive `pxr002_stage_20261118.tar.gz`, hash list `pxr002_stage_hashes_20261118.txt`) was staged to CoreHPC; Miles reports the pre-transfer and hash checks as done (individual hashes not pasted). Runtime-estimate batch job 3111835 (wrapper `pxr002_runtime_estimate_v2.slurm`) ran on ggpu1-08 (L40S 46068 MiB, driver 595.91.07), elapsed 22 s, COMPLETED 0:0. All 16 staged files (scripts including `pxr002_artifact_v2.py`; the v2 artifact including both descriptor `.npy` caches and the amendment CSV) passed `sha256sum -c`; chemprop 2.3.1. The `.err` held only Lightning info and warnings. Source: Miles's report of CoreHPC output.

| Quantity | Measured |
|---|---|
| Training molecules | 10,818; 170 batches/epoch (batch 64) |
| Dataset build from cached descriptors | 1.0 s |
| Smallest corner h300_d3_f1_p0 | 25.0 ms/step; validation 0.2 s; about 4.5 s/epoch |
| Largest corner h600_d5_f2_p0.2 | 28.4 ms/step; validation 0.3 s; about 5.1 s/epoch |

Step time is nearly independent of model size, consistent with a CPU/data-loading-bound loop at num_workers=0 (Lightning warns about worker count; this reading is an inference, not tested). Script projection over 42 runs (36 grid plus 6 stage-B extra seeds): worst case 5.9 h (100 epochs every run at the largest-config epoch time); expected 3.4 h (60 assumed epochs at mean-corner time; the 60-epoch figure is an assumption, not a measurement).

**Validation-MAE defect found in the pilot output (Claude Code).** Logged val_primary_mae after 100 steps was 4.29 and 4.37, against 0.9216 for a constant predictor at the training mean (train mean pEC50 4.328). Cause: Chemprop 2.3.1 puts the output transform in train mode (identity) during validation, so validation predictions are in scaled units while validation targets stay in pEC50 units. The primary-head validation MAE in `pxr002_train.py` was therefore wrong and would have mis-selected checkpoints and early stopping. The measured timings are unaffected. Fix made by Claude Code: the validation model un-scales the primary head with the training-set mean and scale; a CPU regression test was added (primary validation MAE must be in original units).

**Fix tested, restaged and confirmed (2026-11-18).** Local `pxr/tests/test_pxr002.py` in `tripwire_cyp002`: 17 passed, including the new regression test (its sklearn invalid-value warnings come from all-NaN auxiliary columns in the test's synthetic frame and do not apply to production data). The fixed package was restaged to CoreHPC as `pxr002_stage_20261118_r2.tar.gz` with hash list `pxr002_stage_hashes_20261118_r2.txt` (hash values and extraction checks reported done, not pasted); wrapper `pxr002_runtime_estimate_v2.slurm` reused unchanged. This supersedes the earlier "fix untested/unstaged" status and the 20261118 archive and `pxr002_train.py` hash as the current package.

Pilot job 3113883 (ggpu1-08, L40S 46068 MiB, driver 595.91.07; 2026-11-18T01:36:14 to 01:36:35; exit 0): all 16 listed files passed `sha256sum -c`; chemprop 2.3.1; 10,818 training molecules, 170 batches/epoch; cached-descriptor dataset build 1.0 s. Source: Miles's report of CoreHPC output.

| Quantity | Job 3113883 |
|---|---|
| val_primary_mae after 100 steps, smallest corner h300_d3_f1_p0 | 0.640 |
| val_primary_mae after 100 steps, largest corner h600_d5_f2_p0.2 | 0.606 |
| Constant training-mean predictor on validation | 0.9216 |
| Smallest corner | 24.2 ms/step; about 4.3 s/epoch |
| Largest corner | 28.7 ms/step; about 5.1 s/epoch |
| Projection over 42 runs | worst case 6.0 h; expected 3.3 h (60-epoch assumption, not a measurement) |

The MAE values are in the expected range, confirming the un-scaling fix (job 3111835 read 4.29 and 4.37). They come from 100 optimizer steps of an untuned model and carry no model-selection or performance meaning. These timings and projection supersede those of job 3111835. The `run_training` path (ModelCheckpoint, EarlyStopping, metrics.json, manifest) has not yet been exercised on real data.

## 5c. Stage A launch (2026-11-18, CoreHPC)

**Launch approval (stage A only).** Stage A is a single serial job running all 36 configurations of the D018 grid, not three depth-split jobs. Common seed, selection on primary-head validation MAE. The seed value 20261117 was proposed by Claude; Miles's launch used the package he was given, so the seed actually used is not independently confirmed in the records. The wall-time request was briefly raised to 12 h and returned to 8 h before the files were transferred. This approval does not authorize stage B, the auxiliary-off ablation, a final ensemble, or any holdout or test evaluation.

**New files (written by Claude Code, reviewed and transferred by Miles; uncommitted).** `pxr/scripts/pxr002_stage_a_runner.py`; `pxr/scripts/pxr002_stage_a.slurm` (8 h, 1 L40S, 8 CPUs, 96G, `--no-requeue`; requires `AUTHORIZE_STAGE_A=yes`, refuses an existing output root, verifies the staged hash list before running); and a stage A runner test added to `pxr/tests/test_pxr002.py`. Local SHA-256 of the 8 h wrapper is in the supporting record.

**Submission.** Miles reported submitting batch job 3113907 (stage A) on CoreHPC. Package and output names were proposed by Claude: archive `pxr002_stage_20261118_r3.tar.gz`, hash list `pxr002_stage_hashes_20261118_r3.txt`, output root `/scratch/user/mhuang8/pxr002_stage_a_20261118`. Miles did not report his test run, hash comparisons or the exact sbatch line, so whether these names and paths were used, and whether the staged files matched the reviewed hashes, is UNKNOWN.

**Runtime expectation.** From pilot job 3113883: worst case 36 runs x 100 epochs x about 5.1 s = about 5.1 h, within the 8 h request; shorter expected with early stopping (patience 20). The `run_training` path (checkpointing, early stopping, metrics.json, manifest) has not previously run on real data. Job outcome: see §5d.

## 5d. Stage A results (job 3113907, 2026-11-18, CoreHPC)

Source: Miles's pasted sacct and `stage_a_status.jsonl`; job logs are not in this checkout. The `.out` hash-check lines were not pasted.

**Job.** 3113907 COMPLETED 0:0 on ggpu1-08, 2026-11-18T02:25:39 to 03:41:47, elapsed 01:16:08 against the 8 h request. The pilot worst-case projection was about 5.1 h; its 60-epoch expected-case assumption was too high. All 36 configurations returned code 0 with seed 20261117 (so the seed used is now confirmed by the status file), output root `/scratch/user/mhuang8/pxr002_stage_a_20261118`. The `run_training` path (checkpointing, early stopping, metrics.json, manifest) therefore ran on real data without failure.

**Runs.** 114-143 s per run (mean about 126 s). Epochs run were 24-27 for every configuration, i.e. early stopping (patience 20) ended every run; best epochs were therefore roughly 4-7 of 100 (inferred as epochs run minus patience, not read from metrics.json).

**Best primary-head validation MAE (single seed).** Range across the 36 configurations 0.5296-0.5671; the full table (with epochs and seconds per run) is in `notes/Tripwire_PXR_002_supporting.md`. For context, the constant training-mean predictor scores 0.9216 on this validation set. The three lowest:

| Configuration | Validation MAE | Epochs run |
|---|---|---|
| h600_d4_f2_p0.2 | 0.5296 | 26 |
| h300_d5_f2_p0.2 | 0.5357 | 26 |
| h300_d3_f2_p0.2 | 0.5368 | 24 |

Claude's analysis (not a decision): all three have two FFN layers and dropout 0.2; every f2 configuration is at or below about 0.55 while f1 configurations lie between about 0.544 and 0.567; single-seed gaps among the top few (about 0.006) may be within seed noise, which stage B would test. The official ranking is to be produced by `pxr002_grid.py` from the metrics.json files; the figures above come from the status report, not that ranking.

**Boundaries.** Selection used validation only; holdout and official-test labels were not used. At the time of stage A, stage B, the auxiliary-off ablation, final ensemble size, and any holdout or test evaluation were not authorized; stage B was later approved (§5e).

## 5e. Stage B approval and preparation (2026-11-18)

**Approval (D018 Amendment 3).** Miles approved stage B after asking what it entails. Scope: the three best stage A configurations by primary-head validation MAE, selected by `pxr/scripts/pxr002_stage_b_runner.py` from the stage A `metrics.json` files, each run with three seeds (20261117, 20261118, 20261119; six new runs), ranked by mean primary-head validation MAE. Same artifact, features, loss weights, 100-epoch maximum and patience 20. Wall-time request 1 h. The two new seeds were proposed by Claude and are fixed by this approval. Not authorized: the auxiliary-off ablation and its interpretation rule, final ensemble size, any holdout or official-test evaluation, retraining on train plus validation, any statistical improvement threshold or the Rev 3 line 129 bar.

**Files (written by Claude Code; uncommitted).** `pxr/scripts/pxr002_stage_b_runner.py` and `pxr/scripts/pxr002_stage_b.slurm` (requires `AUTHORIZE_STAGE_B=yes`), plus stage B runner tests added to `pxr/tests/test_pxr002.py`. 

**Local tests.** First run of `pxr/tests/test_pxr002.py` by Miles: 21 passed, 1 failed (22 collected). The failure, `test_stage_b_aggregate_uses_stage_a_value_plus_new_seeds`, was a wrong expectation in the test itself: with means x 0.5733, y 0.5300, z 0.5467 the correct ranking is y, z, x, and the test expected y, x, z. Production code was correct and unchanged; Claude corrected the test only. Scripts and the r4 package are unaffected (tests are not packaged). Miles then reported the tests pass; the full summary line was not pasted (22 passed expected).

**Submission.** Miles staged the stage B package and submitted stage B as batch job 3119543 on CoreHPC. Names proposed by Claude: archive `pxr002_stage_20261118_r4.tar.gz`, hash list `pxr002_stage_hashes_20261118_r4.txt`, wrapper `pxr002_stage_b.slurm` (1 h, 1 L40S, 8 CPUs, 96G, requires `AUTHORIZE_STAGE_B=yes`), stage A root `/scratch/user/mhuang8/pxr002_stage_a_20261118`, output root `/scratch/user/mhuang8/pxr002_stage_b_20261118`. The package names, hash comparisons, the 36-result count check and the exact sbatch line were not reported, so whether these were used and whether the staged files matched the reviewed hashes is UNKNOWN. Outcome of job 3119543: UNKNOWN.

## 5f. Stage B results (job 3119543, 2026-11-18, CoreHPC)

Source: Miles's report of the `.out`; job logs are not in this checkout. Stage B output root (proposed `/scratch/user/mhuang8/pxr002_stage_b_20261118`) is not shown in the output.

**Job.** 3119543 COMPLETED 0:0 on ggpu1-11 (L40S), 14:28:32 to 14:42:05, elapsed 00:13:33 against the 1 h request. All 18 listed files passed `sha256sum -c` (r4 package, including `pxr002_stage_a_runner.py` and `pxr002_stage_b_runner.py`); chemprop 2.3.1. The plan line selected h600_d4_f2_p0.2, h300_d5_f2_p0.2 and h300_d3_f2_p0.2 (matching Claude's hand-sort of stage A) and 6 extra runs. All six runs returned code 0, 24-30 epochs, about 2.2 min each.

**Best primary-head validation MAE by seed** (20261117 is the stage A value):

| Configuration | 20261117 | 20261118 | 20261119 | Mean |
|---|---|---|---|---|
| h600_d4_f2_p0.2 | 0.52960 | 0.53900 (27 ep) | 0.54396 (25 ep) | 0.53752 |
| h300_d5_f2_p0.2 | 0.53574 | 0.53768 (27 ep) | 0.54985 (30 ep) | 0.54109 |
| h300_d3_f2_p0.2 | 0.53681 | 0.55002 (28 ep) | 0.54755 (24 ep) | 0.54479 |

Ranking by mean (D018 item 4 rule): 1 h600_d4_f2_p0.2 (0.53752), 2 h300_d5_f2_p0.2 (0.54109), 3 h300_d3_f2_p0.2 (0.54479). The final-epoch val_primary_mae printed in the log (0.572-0.589) is the last epoch, not the saved best checkpoint.

**Claude's analysis (not a decision).** Per-configuration seed spread is about 0.007 (SD) and 0.013-0.014 (range), so the mean gaps of 0.0036 and 0.0073 are about the size of the standard error of a three-seed mean (about 0.004); the three configurations are not separable on this evidence, although the D018 rule names h600_d4_f2_p0.2. For each configuration the stage A seed was its lowest value, consistent with selection from 36 candidates at one seed (winner's curse). On the two fresh seeds alone the means are h600_d4 0.5415, h300_d5 0.5438 and h300_d3 0.5488 (same order), so fresh-seed MAE is about 0.54, not 0.53.

**Boundaries.** Validation only; no holdout or official-test labels used. Not authorized or decided: auxiliary-off ablation and its interpretation rule, final ensemble size and final-training protocol, any holdout or official-test evaluation, retraining on train plus validation, statistical improvement threshold, Rev 3 line 129 numerical bar and its comparator.

## 5g. Auxiliary-off ablation approval and preparation (2026-11-18)

**Approval (D018 Amendment 4).** Miles approved the ablation proposal as Claude put it. The stage B selected configuration (top of `stage_b_ranking.json`; by the reported stage B results h600_d4_f2_p0.2) is run at seeds 20261117, 20261118 and 20261119 with the three auxiliary loss coefficients 0 and primary coefficient 1; cohort, v2 artifact, descriptors, batches, optimizer and stopping rule unchanged, so only the loss changes. Caveat: Chemprop divides by all observed labels, so the primary loss is scaled by n_primary / n_observed. Interpretation rule (descriptive, no statistical threshold): compare three-seed mean and range of each variant; multitask is supported only if its mean is lower and the ranges do not overlap, otherwise "not distinguishable". The runner also prints a mirrored `ablation_better_ranges_disjoint` label when the ablation is clearly lower (Claude's descriptive addition) and reports fresh-seed-only (20261118, 20261119) means for both variants, since the stage A seed was the best of 36. Wall-time request 1 h. Not authorized: final ensemble size and protocol, any holdout or official-test evaluation, retraining on train plus validation, statistical improvement threshold, Rev 3 line 129 bar and comparator.

**Files (written or edited by Claude Code; uncommitted).** New: `pxr/scripts/pxr002_ablation_runner.py`, `pxr/scripts/pxr002_ablation.slurm` (requires `AUTHORIZE_ABLATION=yes`). Edited: `pxr/scripts/pxr002_loss_weights.py` (`aux_off_weight_vector`) and `pxr/scripts/pxr002_train.py` (`--aux-off` flag, default off; stage A and B behaviour unchanged; manifest records `aux_off`). Four tests added.

**Submission.** Miles reported submitting the ablation as batch job 3120327 on CoreHPC. His test run, hash comparisons, stage A/B result-file checks and exact sbatch line were not pasted. Names proposed by Claude: archive `pxr002_stage_20261118_r5.tar.gz`, hash list `pxr002_stage_hashes_20261118_r5.txt`, wrapper `pxr002_ablation.slurm` (1 h, 1 L40S, requires `AUTHORIZE_ABLATION=yes`), stage A root `/scratch/user/mhuang8/pxr002_stage_a_20261118`, stage B root `/scratch/user/mhuang8/pxr002_stage_b_20261118`, output root `/scratch/user/mhuang8/pxr002_ablation_20261118`. Whether these were used, whether the tests passed (Claude expected 26) and whether the staged files matched the reviewed hashes is UNKNOWN. Outcome of job 3120327: UNKNOWN.

## 5h. Auxiliary-off ablation results (job 3120327, 2026-11-18, CoreHPC)

Source: Miles's report of the `.out`; job logs are not in this checkout.

**Job.** 3120327 COMPLETED 0:0 on ggpu1-11, 15:36:56 to 15:43:38, elapsed 00:06:42 against the 1 h request. All 19 listed files passed `sha256sum -c` (r5 package, including `pxr002_ablation_runner.py`); chemprop 2.3.1. The plan line shows the selected configuration h600_d4_f2_p0.2.

**Best primary-head validation MAE by seed** (configuration h600_d4_f2_p0.2):

| Variant | 20261117 | 20261118 | 20261119 | Mean | Range |
|---|---|---|---|---|---|
| Multitask (stage A/B) | 0.52960 | 0.53900 | 0.54396 | 0.53752 | 0.52960-0.54396 |
| Auxiliary-off | 0.53036 (26 ep) | 0.54472 (25 ep) | 0.54911 (25 ep) | 0.54140 | 0.53036-0.54911 |

Mean difference (ablation minus multitask) +0.00388. Fresh-seed-only means (20261118, 20261119): ablation 0.54692, multitask 0.54148, difference +0.00544. **Outcome under the approved rule (D018 Amendment 4): not_distinguishable** (multitask mean lower, ranges overlap). The final-epoch val_primary_mae in the log (0.589-0.593) is the last epoch, not the saved best checkpoint.

**Claude's analysis (exploratory; not part of the approved rule; no threshold).** The ablation was worse at all three seeds, by +0.0008, +0.0057 and +0.0051 (mean +0.0039, SD about 0.0027). With n=3 this does not establish an effect, and whether same-seed pairs are a valid pairing is not established. The auxiliary heads therefore show at most a small benefit (about 0.004-0.005 MAE, about 1%) relative to the seed SD of about 0.007; the evidence does not distinguish the variants under the approved rule.

**Boundaries.** Validation only; no holdout or official-test labels used. Not authorized or decided: final ensemble size and protocol, which variant(s) go to final evaluation, any holdout or official-test evaluation, retraining on train plus validation, statistical improvement threshold, Rev 3 line 129 bar and comparator. A separate structure-only check by Claude (SMILES only, labels not read) is not recorded here.

## 5i. Final ensemble and holdout evaluation protocol: approval and preparation (2026-11-18)

**Approval (D018 Amendment 6).** Miles approved the protocol Claude proposed. Primary model: the stage B selected configuration (top of `stage_b_ranking.json`; by the reported results multitask h600_d4_f2_p0.2), trained on the train split only (v2 cohort), no refit on train plus validation; ensemble of five fresh seeds 20261120-20261124, predictions averaged. Secondary variant: auxiliary-off (coefficients 0), same five seeds, reported alongside. No selection between variants or configurations on holdout numbers. One holdout evaluation of the 621 molecules reporting MAE and Spearman; evidence class internal-retrospective (Rev 3 line 53); official test and its firewalled labels unused. Improvement criterion per Amendment 5 (descriptive comparison with the constant mean-pEC50 predictor). Not authorized: any run of these steps, and any holdout label read before the final evaluation step.

**Claude's design choices within that approval (not separate decisions).** Holdout labels never leave the local machine: the CoreHPC job predicts from a structure-only holdout artifact, and evaluation runs locally only with an explicit flag, a predictions-file SHA-256 gate and a one-shot output directory. Each model's inference path must first reproduce its recorded validation MAE within 1e-3, or no holdout predictions are made. Wall-time request 1.5 h.

**Files (written by Claude Code; uncommitted; not yet tested by Miles, hashed or staged).** `pxr/scripts/pxr002_holdout_artifact.py`, `pxr002_final_runner.py`, `pxr002_final_predict.py`, `pxr002_final_eval.py`, `pxr002_final.slurm` (requires `AUTHORIZE_FINAL_TRAINING=yes`), and tests in `pxr/tests/test_pxr002.py`. 

**Submission.** Miles reported submitting the final training and prediction job as batch job 3120551 on CoreHPC, his separate go-ahead for that run under Amendment 6 (the launch command carried `AUTHORIZE_FINAL_TRAINING=yes`). The job trains ten models and writes holdout predictions from the structure-only artifact; it reads no holdout labels. No holdout evaluation has run. His test run, holdout-artifact build, hash comparisons, stage B ranking-file check and exact sbatch line were not pasted. Names proposed by Claude: archive `pxr002_stage_20261118_r6.tar.gz`, hash list `pxr002_stage_hashes_20261118_r6.txt`, wrapper `pxr002_final.slurm` (1.5 h, 1 L40S, 8 CPUs, 96G), structure-only artifact `pxr/artifacts/pxr002_holdout_v1`, stage B root `/scratch/user/mhuang8/pxr002_stage_b_20261118`, output root `/scratch/user/mhuang8/pxr002_final_20261118`. Whether these were used, whether the tests passed (Claude expected 34, count uncertain) and whether the staged files matched the reviewed hashes is UNKNOWN. Outcome of job 3120551: see §5j.

## 5j. Final training and prediction results (job 3120551, 2026-11-18, CoreHPC)

Source: Miles's report of the `.out`; job logs are not in this checkout.

**Job.** 3120551 COMPLETED 0:0 on ggpu1-07 (L40S 46068 MiB, driver 595.91.07), 16:23:17 to 16:46:15, elapsed 00:22:58 against the 1.5 h request. All 27 listed files (r6 package, including the holdout artifact) passed `sha256sum -c`; chemprop 2.3.1; configuration h600_d4_f2_p0.2; ten runs rc=0 ("10 ok, 0 failed"). Whether Miles's earlier unpasted steps (tests, holdout-artifact build, hash comparisons) went as proposed is UNKNOWN, except that the job's own hash check passed.

**Best validation MAE by seed (epochs).**

| Variant | 20261120 | 20261121 | 20261122 | 20261123 | 20261124 |
|---|---|---|---|---|---|
| Multitask | 0.54910 (28) | 0.54084 (24) | 0.55170 (23) | 0.54996 (30) | 0.53744 (27) |
| Auxiliary-off | 0.56271 (29) | 0.54129 (23) | 0.54213 (23) | 0.56134 (29) | 0.54211 (25) |

Claude's arithmetic (not part of any rule): five-seed mean validation MAE 0.5458 for multitask (range 0.5374-0.5517) and 0.5499 for auxiliary-off (range 0.5412-0.5627); mean difference +0.0041 in favour of multitask, ranges overlap, consistent with the earlier not_distinguishable ablation (§5h).

**Predictions.** The prediction step's validation-replication gate passed for all ten models: replicated validation MAE equalled the recorded value to six decimals in each case (tolerance 1e-3). Wrote `/scratch/user/mhuang8/pxr002_final_20261118/predictions/holdout_predictions.csv`, predictions SHA-256 `45c84c925b9dfc1c5c959e3a95c6b01961667dd19e2338470e8ae26655aa07aa`. The structure-only holdout artifact was used; no holdout labels were read on CoreHPC.

**Boundary.** At the time of this job the holdout evaluation had not run and holdout labels were unread; the evaluation was run afterwards (§5l).

## 5l. Holdout evaluation (2026-11-18, local)

Miles ran the single holdout evaluation (`pxr/scripts/pxr002_final_eval.py`, local, `--authorize-holdout-evaluation`) under D018 Amendments 5 and 6. Output: `pxr/artifacts/pxr002_final_20261118/evaluation/final_evaluation.json` (SHA-256 not reported). The predictions file SHA-256 `45c84c925b9dfc1c5c959e3a95c6b01961667dd19e2338470e8ae26655aa07aa` matched the evaluator's gate. This was the first and only read of the 621 holdout labels; evidence class internal-retrospective; the official test was not used. No selection was made on holdout numbers. The protocol's primary result is the multitask ensemble.

| Quantity (n_holdout 621) | Multitask 5-seed ensemble | Auxiliary-off 5-seed ensemble |
|---|---|---|
| Ensemble MAE | 0.48427 | 0.48469 |
| Ensemble Spearman | 0.73567 | 0.73675 |
| Mean per-model MAE | 0.51357 | 0.51034 |
| Per-model MAE | 0.52899, 0.49491, 0.51685, 0.49694, 0.53019 | 0.51236, 0.50360, 0.51022, 0.50821, 0.51733 |
| Ensemble MAE minus constant floor | -0.43945 | -0.43902 |

Constant training-mean predictor (mean 4.32822, the v2 primary-train mean): MAE 0.92372. Auxiliary-off minus multitask ensemble MAE: +0.00042. Configuration h600_d4_f2_p0.2, seeds 20261120-24.

**Claude's reading (descriptive; no threshold; no selection).** Under the Amendment 5 criterion both ensembles score about 0.484 MAE against 0.924 for the constant floor (about 47.5% lower). The two variants are indistinguishable on the holdout (ensemble MAE differs by 0.0004, auxiliary-off Spearman higher by 0.0011, auxiliary-off mean per-model MAE lower by 0.0033), consistent with the validation ablation, where multitask was lower by about 0.004 but not distinguishable; the single-concentration auxiliary heads therefore show no demonstrated benefit in this setup. Ensembling lowered MAE by about 5.7% for multitask and 5.0% for auxiliary-off relative to the mean single model. Holdout MAE (0.48) is lower than validation MAE (about 0.54) at similar constant-floor MAE (0.924 against 0.922); this may reflect partition composition and was not investigated. The constant floor is a weak comparator and no other method was compared (tree and stock-Chemprop baselines were skipped by decision), so no claim against other models or the leaderboard is supported.

**Boundary.** The holdout is consumed: it must not be used for further tuning or selection. Not authorized: any official-test evaluation or use of firewalled test labels, retraining on train plus validation, any further holdout evaluation.

## 5k. Records housekeeping (2026-11-18)

Claude, with Miles's approval, converted the math in `notes/Decisions.md` and in the 20261117 archive to fenced blocks and code spans. Miles then had both archives deleted, since `notes/Decisions.md` is the authoritative log, and approved committing: commits 59fe1c0 and fa31dab (local, not pushed). The CLAUDE.md rule that archives are never edited was set aside once by Miles's explicit approval; CLAUDE.md itself is unchanged.

## 6. Still pending (not approved)

- Stage B: completed (§5f).
- Auxiliary-off ablation and its interpretation rule: approved and completed as job 3120327; outcome not_distinguishable under the approved rule (§5h). Which variant(s) go to final evaluation is not decided.
- Unlisted fixed settings proposed as Chemprop stock defaults: batch size 64, LR 1e-4/1e-3/1e-4, warmup 2, ReLU, norm aggregation 100, no batch norm, hidden size sets both message and FFN width. Proposal only.
- final ensemble size and final-evaluation protocol: approved as a protocol in D018 Amendment 6 (§5i); completed: training and prediction job 3120551 (§5j) and the single holdout evaluation (§5l);
- any statistical improvement threshold (none set; D018 Amendment 5 sets no threshold or test);
- numerical improvement bar (Rev 3 line 129): decided in D018 Amendment 5 (2026-11-18). For the PXR multitask path it is satisfied by reporting the final holdout MAE against the constant training-mean predictor as a floor, descriptively, with no numeric threshold or statistical test. Claude's clarifications: the predictor is the mean primary-train pEC50 of the v2 cohort (2,896 labels; v1 mean 4.328, validation MAE 0.9216); a floor, not a comparator method; no defined Spearman, so only MAE is compared. Rev 3 is not a repository file and is not amended;
- standardization remains unchanged (released SMILES).

PXR-001 open items not addressed this session: crude and semi-pure HT-chem eligibility, the `OCNT-2395416` == `OADMET-0006234` match, ringless-molecule distribution (19 / 3 / 8).

## 7. Stopping point and next step

Implementation files, the v1 artifact and the v2 amended artifact exist locally (uncommitted). Runtime pilot job 3111835 completed (§5b); nothing trained; no training authorized. The validation-MAE fix is tested locally (17 passed), restaged (r2) and confirmed on the real GPU path (job 3113883). Stage A (36-configuration grid, single serial job) completed as job 3113907 in 01:16:08 with all 36 runs exit 0 (§5d). Stage B was then approved (D018 Amendment 3, §5e); it completed as job 3119543 (§5f); the D018 mean-MAE ranking names h600_d4_f2_p0.2 first (0.53752), ahead of h300_d5_f2_p0.2 (0.54109) and h300_d3_f2_p0.2 (0.54479). The auxiliary-off ablation was then approved (D018 Amendment 4, §5g); it completed as job 3120327 (§5h): auxiliary-off mean 0.54140 against multitask 0.53752, ranges overlapping, so not_distinguishable under the approved rule. The improvement criterion was then decided (D018 Amendment 5, §6): descriptive comparison of the final holdout MAE with the constant training-mean floor, no threshold or test. Completed work, 2026-11-18: Claude ran a structure-only check reading only holdout SMILES and identities (the pEC50 column was not read); all 621 holdout molecules have finite `v1_rdkit_2d_normalized` descriptors (0 non-finite). The final ensemble and holdout evaluation protocol was then approved (D018 Amendment 6, §5i); the final training and prediction job completed as job 3120551 (§5j): ten models trained, validation-replication gate passed for all ten, holdout predictions written (predictions SHA-256 `45c84c92...aa07aa`, full value in §5j) without reading holdout labels. Five-seed mean validation MAE: multitask 0.5458, auxiliary-off 0.5499 (Claude's arithmetic; ranges overlap). Miles then ran the single holdout evaluation locally (§5l): multitask five-seed ensemble MAE 0.48427 and Spearman 0.73567; auxiliary-off ensemble MAE 0.48469 and Spearman 0.73675; constant training-mean floor MAE 0.92372; the variants are indistinguishable on the holdout (Claude's reading). The holdout is consumed and must not be used for tuning or selection. The session was closed on Miles's declaration, 2026-11-18; no further step is authorized or started (see the summary at the top). Not authorized: any official-test evaluation or use of firewalled test labels, retraining on train plus validation, any further holdout evaluation, any threshold or test. Records housekeeping on archives and commits is in §5k. The earlier stray srun jobs are no longer in the queue (§5a).
