# Tripwire-CYP-010 Session Log

**Date:** 2026-11-24 (opened)  
**Status (current):** Closed on the project owner's declaration, 2026-11-26. The D024 three-arm evaluation is complete (section 10): neither penalty arm passes the adoption rule, so unchanged is selected. The evaluation report SHA-256 is 7cefd735d92c61f4b18822bde0ca18afcd15aeba9ec23d2ff448b628cdf462b1 (previously missing, recovered 2026-11-26 from CoreHPC `sha256sum` output; report produced 2026-11-25). The post-inference integrity check is reported satisfied by the project owner (output not pasted; not independently verified). Commit `99a3301` carries the D024 code, configs, tests, packaging and records through the package build; later work (sections 11 onward: D025 and D026 code, tests, analysis outputs and record edits) is uncommitted.

## 1. Session purpose and starting state

Session 010 takes up the three-arm comparison of the updated-T121 recipe with and without a screening-informed extra loss term: updated T121 unchanged, a Q3-cutoff penalty and a response-rank penalty. It commenced on 2026-11-24 from `notes/Handoffs/Tripwire_CYP_009_Handoff_20261124.md`.

Starting state:
- The handoff was read. D024 and its Amendment 1 were confirmed to be present in `notes/Decisions.md` and consistent with the handoff.
- Commit `9bcf355` holds the session 009 record edits, which the handoff listed as uncommitted.

## 2. Implementation plan and approval (2026-11-24)

A consolidated plan was drafted from code reading, revised, and approved. It is recorded as D024 Amendment 2 in `notes/Decisions.md` (Approved), which holds the design, the settled definitions, the one-off runtime-sizing deviation and the execution order. It lifts the bar of D024 and Amendment 1 on implementation of code, configurations and tests, closes the items Amendment 1 left open on adoption readings, per-compound predictions and runtime sizing, and states that planned items are not re-submitted for approval. The bar it names as lifted is the implementation bar; it states no commit authorization. Execution commands are run by the project owner, as before.

The CYP2C9 and CYP3A4 coverage and direction check is not closed by the amendment, and the r2 and r3 confirmation stage remains separately authorized.

**Runtime estimate.** About 5.8 GPU-hours per arm if a cell takes the D022-array mean of 139.867 s (`notes/Tripwire_CYP_009.md` section 9). That mean comes from a different model; this is an estimate only.

## 3. Source-reading finding: prediction scale at validation (unverified)

- Finding from reading source, not verified by execution: installed Chemprop 2.3.1 keeps the output transform in train mode during validation. This implies that the scorer receives standardized predictions and unscales them.
- It is contrary to the statement in the CYP-009 handoff (section 4.2: validation in eval mode returns unscaled predictions to the frozen scorer) and to the keep-raw branch inference in `notes/Tripwire_CYP_009_supporting.md` (the same inference appears in `notes/Tripwire_CYP_009.md` section 12). Those records mark the keep-raw branch as an inference.
- Saved post-scorer predictions are expected to be unaffected.
- The Chemprop version on CoreHPC is unchecked.
- Status: unresolved. A planned test is to check the finding before any record correction; no record has been corrected.

## 4. Local tests

The focused CYP-010 test files and the full local suite (excluding `test_t121_training_weights.py`) were reported as all passing (console output of local commands). An earlier attempt used the project `.venv` Python, which lacks torch and chemprop, so torch-based tests could not run; the passing run used the `tripwire_cyp002` environment's own Python. Test counts were not captured and are not recorded. Equivalence at lambda = 0 against the pre-edit module is part of the planned tests (D024 Amendment 2 item 1).

## 5. Local weight preview

Run over all 50 folds x 4 enzymes x 2 arms; output directory `analysis/cyp_010_penalty_preview_20261124`, report `preview.json` (SHA-256 `06c4d49c9c4ffad1e616cbaf231e5b0cb224c2cd37d58bfa755759c72b77741e`). Screen input hash `4f61be8ffdec767a61981ab89e3f06d28850603c3fa4ad0715a2183d4261a706`; identity input hash `2f3045124dbf138ec535abfd8b697a6e2e2cd605d2bccd9f30a58f4ba5464f54`.

Result: no disabled enzyme-fold-arm, no training-SMILES differences, no automatic stop. No missing-value, unverified-identity or conflicting records in any enzyme.

| Enzyme | Labelled training molecules per fold (range over 50 folds) | Population per fold |
|---|---|---|
| CYP1A2 | 1062-1101 | 1062-1101 |
| CYP2C9 | 965-1008 | 965-1008 |
| CYP2D6 | 1123-1157 | 1123-1157 |
| CYP3A4 | 1766-1815 | 1388-1432 |

CYP3A4: 366-391 labelled molecules per fold have no screening row; none of the other enzymes have any. Flagged fraction: cutoff arm 0.250 (0.250-0.251 for CYP2C9); rank arm 0.999 (all but the lowest-ranked cell receive a positive weight).

Decision on the CYP3A4 gap: handled by the D024 missing-value rule (no extra penalty; ordinary supervised loss and denominator kept); the plan continues to packaging (D024 Amendment 3 item 1).

## 6. Frozen configurations

Under `cyp/artifacts/d024_penalty/` (SHA-256):
- `penalty_q3.json` `ddfe1f01e853e990fabd3ac60e0e7fec5b4bb4e998800b46440b50a4d372af7b`
- `penalty_rank.json` `a20552a33392891dcd1e9eac4af178825c7fc476a7b596004e964f9fc58edefe`
- `evaluation.json` `8e7aabecdaa364f33faf0f59a9292e97a6455d9fcaa9c94dfec34863fd669c77`

## 7. Tail cohorts

Built once from the saved baseline predictions (`t121_valinf_full_20261118/full_cells_000_149`, checked against `full_cells_000_149_SHA256SUMS.txt`) into `analysis/cyp_010_tail_cohorts_20261124.json` (SHA-256 `cc336d4ab3de85a5c8da0ccfd2408b43a27e02c7b5ea2ee41b407388c0d73e7e`). The file is frozen before any penalty-arm result exists.

| Enzyme | Molecules | Cohort size per tail |
|---|---|---|
| CYP1A2 | 1201 | 120 |
| CYP2C9 | 1093 | 109 |
| CYP2D6 | 1269 | 127 |
| CYP3A4 | 1986 | 199 |

## 8. CYP3A4 coverage split (report-only)

D024 Amendment 3 item 2 adds a descriptive report of CYP3A4 results separately for molecules with and without screening coverage, alongside the full-enzyme and fixed-tail results, from the same saved predictions with no new training. It is not an adoption clause and does not change Amendments 1 or 2. "Covered" means a verified CYP3A4 screening value exists under the verified-value rule (Amendment 2 item 2); the reading is open to correction.

The grouping was built once from the screening inputs (screen files hash-checked against the contract) and the baseline predictions into `analysis/cyp_010_cyp3a4_coverage_20261124.json` (SHA-256 `3640127b4fb5b4af135bf27fe9e31c29be6de89138ffa5525ea729fdc2342fb2`), before any penalty-arm result exists. Of 1986 molecules, 1565 are covered and 421 are not covered; all 421 have no CYP3A4 screening row and no other status occurred. The report-only split therefore uses fixed groups of 1565 and 421 molecules.

## 9. Package, baseline provenance and timing

**Package (local only).** Built from the unchanged reference package plus the listed overlay files: `t121_d024_build_20261124/t121_d024_package_20261124.tar.gz`, SHA-256 `83b61e6ae961ad62115f0e2335622bb0c45ee07463a5a5171e7a4b977ae4edf3`, 115 files listed in its `SHA256SUMS.txt`. It contains the pins file, the three frozen configs (section 6), the coverage config and the new modules and tests. It was built after the coverage addition. The focused test rerun after that addition has not been reported, so its result is not recorded. Nothing has been transferred or run.

**Baseline selected record (CoreHPC).** `/scratch/user/mhuang8/tripwire_staging/t121_production_chain_20261117_screen_run/selected_screen_record.json`, SHA-256 `4da7a07de2926a62894441da3cd0183b0e6c38f87902266fba40fc3195b1a0fc`. It records epoch 16 (zero-based), score 0.2029570272781568, and checkpoint `preparation/epoch_checkpoints/epoch_016.ckpt` with SHA-256 `96b75ceb4ddd629373e2356ba5cff3ae8ba58d27a4de1c48db0091646bcd5e7c`. This agrees with the checkpoint hash prefix recorded for the baseline cells (`96b75ceb4ddd6293`).

**Measured same-recipe runtime.** Baseline array job 3105880 (sacct, cells 1-149, all COMPLETED): per-cell elapsed 00:01:29 to 00:02:56, typically about two minutes (cell 0 was an earlier pilot and is not in that list). This is the measured basis for the Amendment 2 sizing, about 5 GPU-hours per 150-cell arm, with the 12 h per-cell request retained. It supersedes the section 2 estimate, which came from a different model.

## 10. Penalty arms: cell-0 integrity gate and three-arm evaluation (CoreHPC)

Cell 0 (repetition 1, fold 1, seed 20261101) is a production cell of each arm, run first as the planned integrity gate. Source: console output of the integrity check over `/scratch/user/mhuang8/tripwire_staging/t121_d024_{q3,rank}_cells_run/trials/trial_121/cells/cell_000`; Slurm jobs 3254977 (cutoff arm, array task 0) and 3254978 (rank arm, array task 0), both COMPLETED. Scores were not printed and change nothing.

**Gate result: PASS for both arms.**
- `COMPLETED.json` present, no INCOMPLETE marker.
- Arm labels `updated_t121_q3_penalty_from_screen_checkpoint` and `updated_t121_rank_penalty_from_screen_checkpoint`.
- Screen checkpoint hash prefix `96b75ceb4ddd6293` (the r1 checkpoint, matching section 9).
- Penalty weights hash equals the local preview value for that fold: cutoff `3346f1d25608bc3237599a0649a4e92071548ea092330e94cdaaf861a331a584`, rank `3f2a9adb332294e2ab0286901c44f3a19c9eb5c45c4617f08837f72bf649363e`.
- Epoch-log lines equal completed epochs (cutoff 36 and 36, rank 35 and 35); lambda 1.0 in every record shown.

**Loss components (cell-count-weighted; first and last logged epoch).** This is the first measurement of the lambda-scaled penalty relative to the base loss. It is a one-cell observation, descriptive only.

| Arm | Epoch 0 base | Epoch 0 unscaled penalty | Last epoch | Last base | Last penalty | Penalty / base at epoch 0 | Penalty / base at last epoch |
|---|---|---|---|---|---|---|---|
| Cutoff | 0.8726 | 0.3595 | 35 | 0.2604 | 0.0271 | about 41% | about 10% |
| Rank | 0.8544 | 0.1929 | 34 | 0.2502 | 0.0378 | about 23% | about 15% |

Flagged (positive-weight) observed cells at the last epoch, by enzyme (CYP1A2, CYP2C9, CYP2D6, CYP3A4): cutoff 267, 245, 285, 354; rank 1068, 979, 1140, 1413.

**Not seen.** Elapsed times of the two jobs; the outputs of the archive/checksum check, the loaded-module check and the cell-planning step (the cells ran from the package paths, so the code ran, but those printed checks were not returned); the cluster library versions remain unrecorded.

**Three-arm evaluation result (reported 2026-11-25; development-only).** Unchanged updated T121, Q3-cutoff penalty and response-rank penalty, lambda 1.0, r1 screening checkpoint, matched 150 cells, 4 enzymes. Produced by `cyp/scripts/t121_d024_aggregate.py evaluate` on CoreHPC; report `/scratch/user/mhuang8/tripwire_staging/t121_d024_report_20261124.json`, SHA-256 7cefd735d92c61f4b18822bde0ca18afcd15aeba9ec23d2ff448b628cdf462b1 (previously missing, recovered 2026-11-26 from CoreHPC `sha256sum` output; report produced 2026-11-25). Both penalty arms ran as single batch jobs (q3 3273438, rank 3273439), 150 cells each, tolerance 1e-06. The aggregation completed without error, including its check that the unchanged arm reproduces its recorded score 0.6769540372619428 (O150 0.676954037 printed). The post-inference integrity check (150 prediction files per arm, no `STOPPED_ON_MISMATCH.json`, 150 reproduced lines per arm) is reported satisfied by the project owner before the aggregation was run (the aggregation was run only after it); the check output was not pasted and the result is not independently verified.

Mean macro ST-RAE (lower is better): unchanged 0.676954037; q3 0.689167299; rank 0.672843630.

Per-enzyme ST-RAE / over-tail MAE / under-tail MAE:

| Arm | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|
| Unchanged | 0.796420 / 2.034759 / 1.141123 | 0.586615 / 1.155224 / 0.853427 | 0.874548 / 1.904297 / 1.240689 | 0.450233 / 1.405430 / 0.953722 |
| Q3 | 0.828250 / 1.743023 / 1.316247 | 0.606279 / 1.114983 / 0.890924 | 0.873811 / 1.746413 / 1.343281 | 0.448329 / 1.406323 / 0.899042 |
| Rank | 0.795941 / 1.839150 / 1.269253 | 0.591742 / 1.136288 / 0.875994 | 0.860740 / 1.772758 / 1.320731 | 0.442951 / 1.415664 / 0.912363 |

**D024 adoption-rule outcome (existing rule, not a new decision).** Clauses: (a) lower mean macro ST-RAE, (b) no enzyme-seed mean worse, (c) lower over-tail MAE, (d) no higher under-tail MAE.

| Arm | a | b | c | d |
|---|---|---|---|---|
| Q3 | False | False | False | False |
| Rank | True | False | False | False |

Neither penalty arm passes; the selected arm under D024 is unchanged. Clause-level reading of c and d per enzyme (from the table above, not from the report JSON): rank over-tail MAE rises only for CYP3A4; rank under-tail MAE rises for CYP1A2, CYP2C9 and CYP2D6; rank CYP2C9 ST-RAE is worse than unchanged (0.591742 vs 0.586615).

**CYP3A4 coverage report (descriptive, D024 Amendment 3; covered 1565, not covered 421 molecules).** MAE / signed error / mean cell ST-RAE:

| Arm | Covered | Not covered |
|---|---|---|
| Unchanged | 0.555906 / +0.172417 / 0.458775 | 0.437730 / +0.089514 / 0.446079 |
| Q3 | 0.565305 / +0.201835 / 0.455342 | 0.445285 / +0.115077 / 0.450953 |
| Rank | 0.562028 / +0.193962 / 0.451220 | 0.438897 / +0.112522 / 0.438728 |

Differences from unchanged (MAE; ST-RAE; ST-RAE by seed 20261101 / 20261102 / 20261103):
- Q3 covered: +0.009400; -0.003433 (-0.00564 / -0.00063 / -0.00403).
- Q3 not covered: +0.007555; +0.004874 (+0.00145 / +0.00200 / +0.01117).
- Rank covered: +0.006123; -0.007555 (-0.01013 / -0.00713 / -0.00541).
- Rank not covered: +0.001167; -0.007351 (-0.00970 / -0.00676 / -0.00560).

## 11. Screen-head loading verification and smoke-test preparation (read-only; no decision made)

No inference, training, scoring, benchmark or Slurm job was run for this work. Hashes and identifiers are in `notes/Tripwire_CYP_010_supporting.md`.

**Scope.** A plan to verify the existing screen checkpoints and prepare a minimal loading smoke test was reviewed and approved for writing the scripts and tests only. Anything run on CoreHPC beyond read-only checks, tier-2 score reproductions, the login-node-versus-batch choice for the CPU smoke, and the choice of a single PXR seed versus a three-seed mean for the 8.25 uM prediction are undecided: an explicit decision entry is required. No threshold or flag logic is added.

**CYP r1 screening run (CoreHPC read-only checks, 2026-11-24).**
- `selected_screen_record.json` and `epoch_016.ckpt` match their recorded hashes (section 9); 50 epoch checkpoints are present. Hashes for `preparation.json`, `split_assignments.csv` and `selection.json` are newly recorded.
- Epoch 16 (zero-based) was selected of 50, score 0.2029570272781568, seed 20261101, no full-screen refit. Per-enzyme native weighted MSE at epoch 16 (CYP1A2, CYP2C9, CYP2D6, CYP3A4): 0.1004, 0.1027, 0.0742, 0.5345. CYP3A4 (index 3) is 5 to 7 times the others; no per-enzyme baseline is recorded, so this is not interpreted, and the validation partition is selection-biased.
- Preparation: train 6,252, validation 695, split seed 20261001; message_hidden_dim 600, depth 4, FFN 2000 x 1, dropout 0.4, batch size 16, native log2fc, no target normalization.
- 16 of 16 pinned sources and inputs of the reviewed r1 manifest pass `sha256sum -c`; `actfound_model_side.py` equals the hash in `notes/Tripwire_CYP_006.md`. Environment `tripwire_cyp002` on CoreHPC: chemprop 2.3.1, torch 2.14.1+cu126, lightning 2.6.6. This resolves the section 3 "Chemprop version on CoreHPC unchecked" item for that environment; the section 3 finding itself remains a source reading.
- Local `t121_screen_stage.py` and `t121_weighted_training.py` differ from the r1-pinned versions, so a load test must import from the r1 package on CoreHPC, not from this checkout.

**PXR-004 stage-1 (sequential arm), three seeds (20261125, 20261126, 20261127).** `best.ckpt` hashes equal the metrics.json and stage-2 manifest values for all three seeds; the staged package passes 16 of 16. 8.25 uM monitor metrics (n 974): MAE 0.1988 / 0.1945 / 0.2070; Spearman 0.5714 / 0.5705 / 0.5441 (seed order as above). These agree with the earlier PXR-004 records, which were transcribed from a paste, and are now directly confirmed on CoreHPC.

**Resolved from source (Chemprop 2.3.1 and repository code; not executed).** The CYP screening model has no output scaler (identity output transform, native log2fc; enzyme order CYP1A2, CYP2C9, CYP2D6, CYP3A4, so CYP3A4 is index 3). The PXR stage-1 model carries an UnscaleTransform whose mean and scale are checkpoint buffers. In Chemprop 2.3.1 the validation hook forces the output transform to train mode (identity), so stage-1 monitor metrics are in standardized units, while a plain eval-mode forward returns native log2fc. PXR head order is 8.25, 33, 99 uM, so the 8.25 uM head is output index 0. The two pipelines scale differently.

**Files written locally (uncommitted; not run under pytest locally; syntax parse only; executed on CoreHPC, see "CoreHPC smoke results" below).** `cyp/scripts/screen_head_smoke_cyp.py`, `pxr/scripts/screen_head_smoke_pxr.py`, their two test files, and frozen pins `cyp/artifacts/screen_heads_smoke/cyp_r1_pins.json` and `pxr/artifacts/screen_heads_smoke/pxr_stage1_pins.json`. Design deviation from the reviewed plan: two scripts (one per package tree on CoreHPC) instead of one script with two subcommands. The scripts verify pins and hashes before any scientific import, import code only from the pinned CoreHPC packages, run on CPU with an explicit thread count, write one result.json to a new directory, never open a stage-2 checkpoint, use no labels and score nothing. Tier-2 score reproductions are not implemented.

**CoreHPC smoke results (2026-11-24).** The CYP smoke and the three PXR stage-1 smokes were run on the CoreHPC login host, CPU only, 2 threads, from `/scratch/user/mhuang8/tripwire_staging/screen_heads_smoke_20261124/`, after hash verification of the transferred script and pins copies. No labels were used and nothing was scored. Per-check detail and hashes are in `notes/Tripwire_CYP_010_supporting.md`.
- **CYP: PASS, 13 of 13 checks.** Pinned sources, inputs and run files hash-verified; selection records consistent (epoch 16); split 6252 train and 695 validation; the imported `t121_weighted_training` is the pinned r1 file; task order puts CYP3A4 at index 3; the head is an Identity output transform with 4 outputs; the strict `state_dict` load of `epoch_016.ckpt` succeeded; forward shape [16, 4], finite; two eval-mode forwards identical. Native log2fc predictions on 16 sorted-id molecules (8 train, 8 validation) lie within -0.264 to 0.359 across the four enzymes. These values are descriptive only: the molecules are not representative, the smoke does not test discrimination, and they are not comparable with the development-compound label ranges in `notes/Tripwire_CYP_009.md` section 12. Login-node timing (not a benchmark): 9.46 s total.
- **PXR stage-1: PASS for all three seeds (20261125, 20261126, 20261127).** All 13 checks were true in each run: hash checks, strict load (best epochs 2, 4, 20), head order with the 8.25 uM head at output index 0, a stage-1 cohort of 8,862 training and 985 monitor molecules, scaler three-way agreement and equality with the pinned values (differences at float32 rounding level), and an eval-mode output equal to standardized output times scale plus mean (max absolute difference at most 1.21e-07, tolerance 1e-5). The stage-1 checkpoints load and the eval-mode path returns native log2fc, consistent with the section 11 source reading. Native 8.25 uM predictions on 16 monitor molecules (lowest sorted row ids; descriptive only, nothing scored) have means 0.73, 0.82, 0.82 for the three seeds, with seed-to-seed Spearman 0.84 to 0.88 on those molecules. These 16 molecules are probably hit-enriched primary-train molecules (an inference, not yet confirmed from the molecule list), are not representative of the screen-only population, and support no statement about discrimination or the three-seed-mean question. The stage-2 potency checkpoint was not opened by design. Result file hashes and per-check detail are in the supporting record.
- The smoke establishes loading and head-shape behaviour only. It does not reproduce any score, and tier-2 reproductions remain unimplemented.

**Decided (D025, Approved as scope and design decisions only; no code writing, inference, run, Slurm submission or protected-data use is authorized).** Tier-2 score reproduction is not a mandatory gate and settled checks are not repeated unless a named implementation change introduces a new failure mode; the CYP screen value is the single r1 `epoch_016.ckpt` with four native log2fc outputs (CYP3A4 index 3) and no r2/r3 averaging; the PXR screen value is the three verified stage-1 checkpoints at the 8.25 uM head (index 0), each seed retained and the arithmetic mean reported, with no seed selection and no stage-2 substitution; no new fits, tuning or checkpoint selection; any inference takes an explicit input-file argument with no default dataset and the target molecule set is not named; predictions go to a new separate output with existing potency outputs unchanged. These replace the earlier open items on tier-2 reproductions, single seed versus three-seed mean, and the single r1 checkpoint.

**Implementation written (D025 Amendment 1, Approved as a writing authorization only).** Code, tests and frozen pins and input files for screen-prediction exposure were written locally (uncommitted, not run, syntax parse only): `cyp/scripts/screen_predict.py`, `cyp/scripts/screen_predict_reference.py`, their two test files, a synthetic smoke input and a 13-entry protected-file hash guard (exact-file only; it does not catch derived or excerpted protected data). Corrections adopted: the CYP model is built through the pinned r1 builder `build_t121_screen_prefit_state` (which reads the pinned screening files to construct the model and dataset object, with no prediction on them); the manifest names four checkpoints; input identifiers are literal text with duplicates recorded and no assumed join to CYP `Molecule_Name` or PXR `row_id`; inference never switches a model or output transform to training mode; the test plan includes a small comparison against the saved 16-molecule PXR smoke predictions as a targeted check of changed code, reading the three PXR smoke `result.json` files and `pxr/artifacts/pxr002_aux/pxr002_train_frame.csv` (SMILES lookup only). File hashes are in the supporting record. No run, inference, transfer, Slurm submission, protected-data use, threshold or flag logic is authorized, and complete commands are to be provided before any execution approval.

**Screen-prediction runs on CoreHPC (reported 2026-11-25).** `screen_predict.py` was run on the CoreHPC login host, CPU, 2 threads, on a synthetic input and on a 16-molecule reference input; details and hashes are in the supporting record. No decision was made in this step.
- **Synthetic input (14 rows, batch sizes 3 and 64):** both runs completed with 10 ok, 1 missing id, 1 empty SMILES and 2 invalid structures, all reported per row with none dropped; the duplicate id `00042` was recorded; the manifest names exactly the four checkpoints. Predictions were identical within 1e-6 across the two batch sizes. The PXR stage-1 transform check on a deep copy left the inference model untouched, with max difference 2.73e-08.
- **Targeted comparison of the changed PXR loading path:** on the 16 smoke molecules (all primary-train `OADMET-` ids, so hit-enriched and not representative), 16 of 16 rows were ok and the predictions equal the saved smoke predictions for all three seeds (max difference 0.0, tolerance 1e-5); the three-seed mean equals the arithmetic mean of the retained seed columns. Labels were not used and nothing was scored. Exact equality is expected because batch size and order match the smoke; this tests the pinned-constant scaler against the recomputed one, not predictive validity.
- **Script identity confirmed (2026-11-25):** CoreHPC `sha256sum` of `screen_predict.py` and `screen_predict_reference.py` equals the local hashes, and the manifest `script_sha256` of all three runs equals the `screen_predict.py` hash, so the runs used that script. The script verifies the synthetic input and protected-hash list against declared hashes at run time.
- **Synthetic output read:** rows contiguous in input order; ids preserved literally (including leading zeros, `NA`, `nan`, a CSV-quoted comma, and a trailing space with a warning); the duplicate id flagged on both rows; invalid rows kept with empty prediction cells and a reason; `C[Hg]Cl` ok. On the 10 ok rows, which are small, mostly out-of-domain molecules chosen for input handling and are not a test of predictive performance, CYP outputs lie between -0.47 and +0.34 native log2fc, with CYP2D6 positive on all rows (as in the loading smoke), and PXR 8.25 uM seed means range -0.09 to 0.54, with the largest seed disagreement on tiny out-of-domain molecules. No interpretation is established and nothing was scored.
- **Local tests (2026-11-25; source: pasted console output):** pytest 9.1.1, Python 3.12.14, interpreter `/opt/homebrew/Caskroom/miniforge/base/envs/tripwire_cyp002/bin/python`: 38 items collected, 38 passed in 7.79 s, none skipped, across `cyp/tests/test_screen_predict.py` (including its torch/chemprop model-behaviour tests) and `cyp/tests/test_screen_predict_reference.py`.
- **Not yet seen:** timing.
- That these runs used the login host CPU is a fact only; where inference may run remains open (an explicit decision entry is required). No named input file or protected-data use exists. The review trigger, thresholds and the PXR hit definition remain unrecorded.

**Execution location settled (D025 Amendment 2, Approved as a record correction and command-preparation authorization only).** Inference and evaluation runs for the existing screen heads use the established CoreHPC CPU route: login host, environment `/scratch/user/mhuang8/envs/tripwire_cyp002`, CPU, explicit thread count, outputs to new directories under `/scratch/user/mhuang8/tripwire_staging/`. This continues the route used for the completed checks and smokes; it is not permission for unlimited login-host compute, and SLURM training stays separate. No new interactive-allocation or policy-search prerequisite is introduced absent a new restriction or a materially different workload. The five-minute limit remains not adopted. The amendment supersedes, to that extent, the D025 open location item (D025's original text is unchanged). Closed and not to be reopened or rebuilt: the verified checkpoints, loaders, native units, seed package and the tested `screen_predict.py`; the loading smoke is not repeated and tier-2 reproduction is not reopened.

**Real input set defined (D025 Amendment 3, Approved as an input-set definition and a writing authorization only).** The first real run is on molecules with measured CYP3A4 and PXR 8.25 uM responses that both screen models excluded from training and checkpoint selection, with protected material removed by construction: CYP internal-test and challenge-blinded tiers (77), PXR holdout-role molecules (190), the 4 CYP `bridges_protected` molecules, and molecules whose CYP and PXR source SMILES differ in canonical isomeric structure (2 expected). Earlier read-only analysis counts: 449 base, 213 after the first two removals, 209 after bridges, about 207 after the SMILES rule; the exact count is fixed when the file is built. The input SMILES is the CYP screening-file string; `input_id` is `OCNT_ID` (equal to the CYP `Molecule_Name`), with an explicit mapping file listing `OCNT_ID`, `Molecule_Name`, PXR `row_id` and both SMILES strings (a later join still needs review of that mapping). Writing a read-only input-building script and tests is authorized, reading only presence flags, identities, tiers or roles and SMILES strings, with no label values, and writing to a new directory. The selected set will contain PXR aux-only molecules whose group touches official-test structures; the builder reports their count and their exclusion is not part of the decision. Running the builder, transfer and `screen_predict.py` are not authorized by this amendment. The builder and its tests are now written locally (uncommitted, not run); it uses the role-appropriate PXR comparison SMILES, which may change the earlier count of about 207, so the exact final count comes from the builder report.

**First real screen-prediction run (reported 2026-11-25).** `screen_predict.py` was run on the CoreHPC login host (CPU, 2 threads, batch size 64) on the real input built by the approved builder: 208 input rows, 208 ok, no invalid rows and no duplicate ids; the output has 208 prediction rows. The manifest script hash equals the verified script and names exactly the four checkpoints. The run took 34.29 s including model loading (single run, not a benchmark). The final set size of 208 replaces the earlier estimate of about 207. No predictions were reviewed, scored or evaluated, and no label was used. The input file, predictions file and manifest hashes on CoreHPC are recorded in the supporting record. The builder report (read locally, 2026-11-25) confirms the step counts 449, 213, 209 and 208 (equal to the recorded counts) and that the local input-file hash equals the CoreHPC copy, so the transfer is intact; the role-appropriate SMILES rule removed 1 molecule (earlier estimate 2). The selected 208 include 11 PXR aux-only molecules whose group touches official-test structures, identifiable in the mapping file; whether to set them aside is open (an explicit decision entry is required). Not yet seen: the builder test results and the builder's exit code as such (exit 0 is inferred from the report, not observed).

**Descriptive evaluation scope decided (D026, Approved as an evaluation scope; planning and command preparation only, no execution by the assistant).** The 11 aux-only molecules whose group touches official-test structures are excluded from the primary descriptive evaluation, leaving 197 molecules (160 PXR primary-validation and 37 aux-only, reported as separate strata); the 208-row input and existing predictions stay unchanged; existing predictions are used with no inference rerun to select the 197. The crosswalk is verified from the builder's mapping file (identifiers not assumed interchangeable; coverage, duplicates, structure matches and CYP-tier by PXR-role counts after exclusions reported). Proposed metrics: native-unit MAE and Spearman for the four CYP outputs and the PXR 8.25 uM three-seed mean, label counts per output and stratum, and a constant-prediction MAE baseline from existing training-only statistics (not from evaluation labels; none invented). The evaluation is not challenge performance, not validation of the potency-plus-screen report and not evidence of PXR-specific activity; no thresholds are chosen from it. Protected CYP tiers, holdout-role and bridging molecules are not reintroduced.

**Baselines decided (D026 Amendment 1, Approved as an evaluation-definition and writing authorization).** The CYP constant baseline for each of the four outputs is the unweighted mean of observed native log2fc over the 6,252-molecule screening-training partition (existing `preparation/split_assignments.csv`, to be transferred, not rebuilt; evaluation molecules checked to be outside it; no evaluation label enters). It is a training-only reference constant, not the MAE-optimal constant and not a bound on achievable error (D026 Amendment 2 supersedes the earlier "plain floor" wording), because the screening model trained with inverse-standard-error weights, and it has no meaningful Spearman. The PXR baseline is the recorded stage-1 training-label mean at 8.25 uM, 0.301972839854651 (8,763 labels, monitor excluded). Writing `cyp/scripts/evaluate_screen_predictions.py` and `cyp/tests/test_evaluate_screen_predictions.py` is in scope; execution stays with the project owner. The relation between the D025 Amendment 3 no-label-values bullet (builder) and this evaluation is unreconciled; D026 treats the evaluation as a separate descriptive step.

**Evaluation script written (2026-11-25).** `cyp/scripts/evaluate_screen_predictions.py` and `cyp/tests/test_evaluate_screen_predictions.py` are written locally (uncommitted, syntax parse only, not run; hashes in the supporting record). A read-only check of the builder's mapping file found identical ids and order in input and mapping, no duplicate ids, `cyp_molecule_name` equal to `OCNT_ID` for 208 of 208, all 208 SMILES pairs canonically identical, and 241 exclusion rows with none among the kept. After setting aside the 11, the evaluated set is 197 (160 primary_validation, 37 aux_only); CYP tier by PXR role: train/primary_validation 137, train/aux_only 32, validation/primary_validation 23, validation/aux_only 5. All four CYP enzymes and the PXR 8.25 uM value are present for all 197 (presence only; no label values read), with no duplicate label rows. 

**D026 descriptive evaluation results (2026-11-25; source: owner-pasted output of a read-only command on the local machine reading `analysis/cyp_010_screen_eval_20261125/report/`).** `evaluate_screen_predictions.py` ran locally on the 208-row predictions and the builder mapping; its own assertions on hashes, crosswalk and recorded counts passed (inferred from completion). The 197 evaluated molecules split as recorded above, with label counts n 197 per output (160 primary_validation, 37 aux_only). Native log2fc results (MAE; Spearman; MAE as a ratio to the training-mean constant baseline), full table in the supporting record:

| Output (all strata, n 197) | MAE | Spearman | MAE / baseline MAE |
|---|---|---|---|
| CYP1A2 | 0.509 | 0.568 | 0.804 |
| CYP2C9 | 0.283 | 0.681 | 0.550 |
| CYP2D6 | 0.745 | 0.478 | 0.938 |
| CYP3A4 | 0.903 | 0.701 | 0.585 |
| PXR 8.25 uM three-seed mean | 0.357 | 0.641 | 0.632 |

Descriptive reading (corrected 2026-11-25; not a decision): over all 197 molecules CYP2C9 has the lowest MAE/baseline ratio (0.550), ahead of CYP3A4 (0.585) and PXR (0.632); CYP3A4 has the highest overall Spearman (0.701); CYP2D6 has the weakest ratio (0.938) and the weakest Spearman (0.478) among the five outputs. Every output has MAE below its constant baseline and a positive Spearman. Per-stratum figures are as tabulated in the supporting record; in the aux_only stratum PXR has the lowest Spearman (0.354). Not established, and removed from the earlier reading: any explanation of the CYP3A4 baseline MAE by the training mean (-0.974), since the baseline MAE depends on the distance of the evaluation labels from the constant and nothing about label spread is inferred from the constant's value; any statement that the positive CYP2D6 training mean is consistent with earlier head outputs near +0.2 to +0.35, since no prediction distribution has been inspected and no claim about where CYP2D6 predictions cluster is made; and the word "floor" with the statement that a training-only median would give a lower baseline MAE, since the training-mean constant is a reference point and not a bound, and a training-only median minimizes training absolute error but is not guaranteed to give a lower or higher MAE on these 197 evaluation molecules (no median baseline was computed). CYP caveat: 169 of the 197 molecules are CYP training rows and 28 are CYP validation rows (train/primary_validation 137 + train/aux_only 32 = 169; validation/primary_validation 23 + validation/aux_only 5 = 28); the primary_validation and aux_only strata describe PXR roles, not CYP holdout status, so the pooled CYP metrics cannot establish held-out CYP performance. Other limits: descriptive only; molecules share scaffold groups and are not independent; no confidence intervals; not challenge performance, not validation of the full potency-plus-screen report, not evidence of confirmed PXR-specific activity. No threshold was chosen and no decision was made. MAE alone does not establish classification changes near any cutoff. Superseded 2026-11-26 (D027): the CYP train and validation tiers refer to the potency models, not to screen-head membership; the 197 molecules were excluded from training and selection of both screen heads, so the CYP-tier caveat above does not bear on screen-head held-out status (pretrained-encoder overlap is unchecked). Unresolved: `evaluation_report.json` lists 6,251 training labels per CYP enzyme (`n_train_labels`) while the records describe 6,252 screening-training molecules; the one-molecule gap is not yet explained (detail in the supporting record).

**CYP-tier-split diagnostic (D026 Amendment 2; written 2026-11-25, run locally and reported 2026-11-26).** `cyp/scripts/evaluate_screen_cyp_tier_split.py` and `cyp/tests/test_evaluate_screen_cyp_tier_split.py` are written locally (uncommitted). The tier-split unit tests were run locally (reported 2026-11-26 by the project owner, with the tripwire_cyp002 interpreter) and all passed; the individual test output and count were not pasted. SHA-256 of the files as they now exist, computed 2026-11-26 after the run (the run-time identity of the script was not hashed at run time): script f916846846b97c32511f453c1dd59e21a933fc80ef990c9cdf30766b6c8e43f7; tests 96dfa23b6d666497bdf00089ffcdd77f41a3ded2178a0037b7f515132d023a04. The script was run locally (source: output pasted by the project owner) on the saved `evaluation_rows.csv` and `evaluation_metrics.csv` only, both hash-checked by the script; its pooled all-stratum consistency check passed (inferred from completion). Output is in `analysis/cyp_010_screen_eval_20261125/cyp_tier_diagnostic/` (`cyp_tier_split.csv`, `cyp_tier_split_report.json`; hashes in the supporting record). Tier counts are train 169 and validation 28 (the CYP tier is one molecule-level value, the same for all four outputs). Native log2fc; ratio is MAE over the in-tier baseline MAE (existing training-only constants); signed mean residual is prediction minus label.

| Output | Tier | n | MAE | Spearman | Baseline MAE | Ratio | Mean residual |
|---|---|---|---|---|---|---|---|
| CYP1A2 | train | 169 | 0.4931 | 0.5947 | 0.6206 | 0.7945 | +0.3181 |
| CYP1A2 | validation | 28 | 0.6046 | 0.3892 | 0.7101 | 0.8514 | +0.3801 |
| CYP2C9 | train | 169 | 0.2821 | 0.6971 | 0.5090 | 0.5543 | +0.0881 |
| CYP2C9 | validation | 28 | 0.2875 | 0.5977 | 0.5457 | 0.5267 | +0.0915 |
| CYP2D6 | train | 169 | 0.7480 | 0.5100 | 0.7977 | 0.9377 | +0.5843 |
| CYP2D6 | validation | 28 | 0.7252 | 0.2742 | 0.7737 | 0.9374 | +0.5508 |
| CYP3A4 | train | 169 | 0.8889 | 0.7177 | 1.5409 | 0.5769 | +0.6432 |
| CYP3A4 | validation | 28 | 0.9866 | 0.5807 | 1.5521 | 0.6356 | +0.4970 |

Baseline constants: CYP1A2 -0.1932, CYP2C9 -0.3715, CYP2D6 +0.3457, CYP3A4 -0.9739.

Descriptive reading (not a decision; no cause explained): the validation-tier MAE/baseline ratio is below 1 for all four outputs (0.851, 0.527, 0.937, 0.636). It is lower than the train-tier ratio only for CYP2C9 and equal to it to three decimals for CYP2D6. Spearman is lower in the validation tier than in the train tier for all four outputs; with n 28 and no confidence intervals the size of these differences is not interpreted. The signed mean residual is positive in every output and tier (+0.088 to +0.643): predictions are above labels on average. The residual measures aggregate offset only and does not separate bias from noise. No new baseline, bootstrap, threshold or inference was used. Limits as in D026 and Amendment 2: descriptive; molecules share scaffold groups; not challenge performance. Not supplied: the pytest output for the evaluation tests (see section 13).

**Open (an explicit decision entry is required for each).** The review trigger and thresholds, the definition of a PXR hit, the explicit CYP/PXR join for the potency report, and whether to add a training-only median baseline or confidence intervals.

## 12. Claim boundary

The comparison is development evaluation under D024 and its amendments; protected test data are untouched by it. The section 3 finding is a source reading only. The penalty-arm results are the cell-0 integrity observation and the development-only three-arm evaluation (section 10), which is an adoption-rule outcome under D024 and not challenge performance; its post-inference integrity check is reported satisfied by the owner (output not pasted, not independently verified), and its report SHA-256 is 7cefd735d92c61f4b18822bde0ca18afcd15aeba9ec23d2ff448b628cdf462b1 (previously missing, recovered 2026-11-26 from CoreHPC `sha256sum` output; report produced 2026-11-25). The section 11 work is read-only verification and a CPU loading smoke on CoreHPC; it shows that the checkpoints load and the heads have the expected shape, and it scores nothing. D025 fixes the scope and design of screen-head prediction exposure but authorizes no code, inference or run; the loading smoke is loading-path verification, not predictive validation.

## 13. Stopping point and next step

**Session closed 2026-11-26 on the project owner's declaration.** Completed: the D024 three-arm evaluation (unchanged selected; section 10); the screen-head loading smoke and D025 screen-prediction exposure through the first real run on the 208-molecule set, 208 of 208 rows ok (section 11); and the D026 descriptive evaluation on 197 molecules with its Amendment 2 CYP-tier-split diagnostic (section 11). D025 (with Amendments 1 to 3) and D026 (with Amendments 1 and 2) are recorded in `notes/Decisions.md`. Uncommitted at close: the D025/D026 scripts, tests, pins and analysis outputs and the record edits made after commit `99a3301`.

**Carried forward (none decided or authorized here):**
- The 009 handoff record correction concerning validation prediction scaling (section 3) needs the project owner's approval; no record has been corrected.
- The r2 and r3 confirmation stage remains separately authorized for later.
- The 6,251 versus 6,252 `n_train_labels` gap in `evaluation_report.json` is unexplained.
- The D025 Amendment 3 no-label-values bullet and the D026 evaluation remain unreconciled.
- The pytest output for the `evaluate_screen_predictions` tests was never supplied (recorded 2026-11-26); no test result for those tests is established.
- The review trigger and thresholds, the PXR hit definition, the CYP/PXR join for the potency report, and whether to add a training-only median baseline or confidence intervals are undecided (an explicit decision entry is required for each).
