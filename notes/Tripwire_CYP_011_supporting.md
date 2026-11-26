# Tripwire-CYP-011 Supporting Record

Supports `notes/Tripwire_CYP_011.md`. Identifiers and hashes only.

## D028 files and local test run (2026-11-26; source: owner's report of local pytest runs, tripwire_cyp002 interpreter; console output not pasted)

Local, uncommitted. No inference, transfer or job has run.

| File | SHA-256 |
|---|---|
| `cyp/scripts/t121_potency_predict.py` (current, after the unscale fix) | 1e236efe54d5ec58e632b5f6dcf5b8904cdb7d5ce1f63ce0b67d387fd0830f37 |
| `cyp/tests/test_t121_potency_predict.py` (current, with `RealModelSeam` tests) | b1dc8f2848e0a690d278fe2ec2a0becca9365a0b2521f09909861bd38e02bb9d |
| `pxr/scripts/pxr002_build_predict_frame.py` | 434bf261dd4f45be7a4418eab795a4264f3a7d6d7664a25b2b0348f7314d6eba |
| `pxr/scripts/pxr002_predict_frame.py` (current, with timing-only mode) | 561274ce955761801c716c05e3a95ab7b14b8a4c0febb640e11aaa36f25b5d3b |
| `pxr/tests/test_pxr002_predict_frame.py` (current, with TimingOnly tests) | 88edbc7e2e982793f9100b8ece3c02dd3d4ba286e701f533712b44ebd46a50a8 |
| `cyp/scripts/build_screen_report.py` | 21a98e82e95eb9cd77d2da43dc41e60e5ea3b89c86c1ab2f596284b503d910e8 |
| `cyp/tests/test_build_screen_report.py` | 44f91dfa2ee993e0ab6c61b325f3aef40c5c5fc30d6301c88a4d0e7de30c3af9 |

Superseded hashes: the timing-only-mode versions were `t121_potency_predict.py` 30673db9...9896d and `test_t121_potency_predict.py` 275866c2...0c3b (the first CoreHPC timing attempt below used the former). The versions the first two test runs below used, before the timing-only additions, were: `t121_potency_predict.py` 71b8d54a...0decb; `test_t121_potency_predict.py` cfe964fa...6a90 (after fixture fix); `pxr002_predict_frame.py` dcec01ad...9630b; `test_pxr002_predict_frame.py` bff20da8...f15713. The report-join and `pxr002_build_predict_frame.py` files are unchanged.

### Timing-only mode (added after approval, 2026-11-26)

`--timing-only-cell N` in `cyp/scripts/t121_potency_predict.py` runs one member through build, reference gate and input prediction and writes only a timing record (`--cells` must still name 150 cells). `--timing-only-checkpoint variant:seed` in `pxr/scripts/pxr002_predict_frame.py` runs one of the ten approved models and writes only a timing record. `TimingOnly` test classes were added to `cyp/tests/test_t121_potency_predict.py` and `pxr/tests/test_pxr002_predict_frame.py`. Reason: measured interactive timing is required before any submission, and the scripts otherwise cannot run a single member. After the additions the project owner reported that all tests pass for the three test files (`cyp/tests/test_t121_potency_predict.py`, `cyp/tests/test_build_screen_report.py`, `pxr/tests/test_pxr002_predict_frame.py`); counts were not supplied.

Test runs:
1. First run over `cyp/tests/test_t121_potency_predict.py`, `cyp/tests/test_build_screen_report.py` and `pxr/tests/test_pxr002_predict_frame.py`: 40 passed, 2 failed in 10.17 s. Both failures were in the CYP potency EndToEnd tests and came from the test fixture writing the repr of a numpy float64 (text `np.float64(2.0)`) into the reference CSV, not from the script.
2. After a one-line fixture fix, the CYP potency test file ran 19 passed in 10.11 s. The results for the other two files are those of the first run (all passed).
3. The earlier plan expectation of 22 tests in the CYP potency file was a miscount; the file has 19.

## CYP potency one-member timing (2026-11-26; source: owner-reported CoreHPC login-host output, CPU, 2 threads, batch 64)

| Run | Script SHA-256 | Output directory (under `/scratch/user/mhuang8/tripwire_staging/` unless stated) | Outcome |
|---|---|---|---|
| 1 (cell 0) | 30673db9f329c2483d0bbf4ddc7dca0d292515673771f9899633b50aa719896d | `cyp_potency_timing_20261126` | Stopped at the reference gate: max abs difference 5.071113783007601 over 560 rows, tolerance 1e-5; only the stop record written |
| 2 (cell 0) | 1e236efe54d5ec58e632b5f6dcf5b8904cdb7d5ce1f63ce0b67d387fd0830f37 | `cyp_potency_timing_20261126b` | Gate passed: max abs difference 8.064889511771867e-07 over 560 rows; 418 gate molecules; 208 input molecules predicted |

Run 1 cause: Chemprop 2.3.1 `RegressionFFN.train_step` is `forward` and applies the output transform, which unscales in eval mode, so the manual unscale in the script doubled the scaling. The corrected staged copy replaced the old one (old kept as `.pre_unscale_fix_20261126`); the owner verified its hash before run 2.

Run 2 timing (seconds): build and load 1.709; reference gate pass 0.927; input prediction 0.499; member total 3.135. Extrapolated 150 members 470.3 s (one member times 150, cold start included; not a guarantee). No ensemble file written.

Local test result after the fix (owner-reported): `cyp/tests/test_t121_potency_predict.py` 31 passed in 15.77 s (includes `RealModelSeam` and source-invariant tests; a test-helper `drop_last` bug fixed in the test only).

## Full 150-member CYP potency run (2026-11-26; source: assistant verification of the fetched manifest and outputs on the local machine; the CoreHPC-side `ls` and `sha256sum` output was not pasted)

- Host and settings: CoreHPC login host, CPU, 2 threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`), batch 64, detached run, process 4165577.
- Script SHA-256 1e236efe54d5ec58e632b5f6dcf5b8904cdb7d5ce1f63ce0b67d387fd0830f37; input `screen_input_set.csv` (208 rows) SHA-256 ec85098928d21928f59002b32985ae7db530fbe0dc9f15bfe9405e7c90f91537.
- Run output `/scratch/user/mhuang8/tripwire_staging/cyp_potency_run_20261126` (log `cyp_potency_run_20261126.log`); fetched to `analysis/cyp_011_potency_20261126/cyp_potency_run_20261126/`.
- Checks (local, from the fetched manifest): 150 members; 150 reference gates within 1e-5, maximum difference 1.050954927084291e-06; 208 of 208 rows ok, zero invalid; `member_seconds_total` 457.19 s against the 470.3 s extrapolation; manifest output hashes equal recomputed hashes; manifest input hashes equal the pinned dataset, split, executor, input and protected-list hashes.

| File | SHA-256 |
|---|---|
| `cyp_potency_ensemble.csv` (208 rows; four `pred_mean150`, four `member_sd150`) | 4032dc19e2a192f721ecedb6d7dd7efc24a1dd7a58dbe5577ee29ea466504499 |
| `cyp_potency_members.csv` (31,200 rows = 150 x 208) | bab3507752c07e2db0e72d07b5f80ded4e83f1415f81299a4eb7cafe6c8ff314 |
| `cyp_potency_manifest.json` | not computed |

- Ensemble means per enzyme (min / mean / max, pIC50): CYP1A2 3.351 / 4.953 / 5.949; CYP2C9 3.085 / 4.736 / 6.063; CYP2D6 4.298 / 4.702 / 5.832; CYP3A4 1.875 / 4.522 / 5.953. Member SD (ddof=1) ranges 0.047 to 0.535.
- Deployment predictions of the pool (D028; most members trained on these molecules); descriptive only; no label comparison. PXR potency is not yet generated and the report join has not been run.

## PXR potency run (2026-11-26; source: assistant verification of the fetched manifest and outputs on the local machine)

- Host and settings: CoreHPC login host, CPU, 2 threads, detached (process 1719).
- One-model timing (multitask seed 20261120): gate absolute difference 1.326e-08 against recorded validation MAE 0.5490965845; 13.36 s for one model; extrapolation 133.6 s for ten (one model times ten, cold start included; not a guarantee).
- Package: `/scratch/user/mhuang8/tripwire_staging/pxr_predict_pkg_20261126` (new copy of the frozen r6 package; r6 untouched). `pxr/scripts/pxr002_predict_frame.py` SHA-256 561274ce955761801c716c05e3a95ab7b14b8a4c0febb640e11aaa36f25b5d3b. Artifact `pxr/artifacts/pxr002_predict_208_20261126`: mapping SHA-256 fd9ab197b736039f497c62085ce0617e8ce3d4f0631729f8c93fa761887111e3; 208 rows; descriptors 208 x 200 float64, finite; `SHA256SUMS` SHA-256 c42fd4a36605e1c7512cd072f953d7086da9589f979638aa005ea0a949884b4c; ids and SMILES equal the mapping `pxr_row_id` and `pxr_source_smiles` in order; built on the local machine, no labels read.
- Run output `/scratch/user/mhuang8/tripwire_staging/pxr_potency_run_20261126` (log `pxr_potency_run_20261126.log`); fetched to `analysis/cyp_011_potency_20261126/pxr_potency_run_20261126/`.
- Checks (local, from the fetched manifest): 10 gates all passed, maximum absolute difference 2.19e-08 (tolerance 1e-3); configuration `h600_d4_f2_p0.2`; variants multitask and aux_off; seeds 20261120-20261124; the ten checkpoint hashes equal the PXR-002 `final_predict_manifest`; `final_plan.json` SHA-256 3862c5dc38f93952337bfea4edbb60be74614373b4fd6fc88dbd2d9e68455dcf; `training_performed` false, `labels_read` false; manifest output hashes equal recomputed; 2,080 seed rows (208 x 10), exactly five seeds per row and variant; means recompute exactly from the seed rows (0 mismatches); row ids equal the mapping `pxr_row_id` in order.

| File | SHA-256 |
|---|---|
| `pxr_potency_means.csv` | 1255e03d602448697356d7f4f0fae6879dfddc84fd8c26695c4161859825badb |
| `pxr_potency_seed_predictions.csv` | 1ac9c94db2e7bf995a48cdb3e6fd2c6bd87d9b3dbbe48024628d5da36525b616 |
| `pxr_potency_manifest.json` | 922246c335920a99ea923500f6126a7f20a2a04913fa1d6769fff6d7e09b43e2 |

- Descriptive ranges (pEC50, min / mean / max): multitask five-seed mean 2.118 / 4.569 / 5.712 (primary column); aux_off mean 2.111 / 4.624 / 5.743 (secondary). Multitask mean by role: primary_validation (160) 4.708; aux_only (48) 4.106. The 160 primary_validation molecules were the model-selection validation set (not trained on); aux-only molecules have no primary pEC50 label. No label comparison was made.

## Report join (2026-11-26; run locally by the project owner; counts checked locally by the assistant)

- Script `cyp/scripts/build_screen_report.py` SHA-256 21a98e82e95eb9cd77d2da43dc41e60e5ea3b89c86c1ab2f596284b503d910e8; interpreter `tripwire_cyp002`; output directory `analysis/cyp_011_report_20261126/`.

| Input (hash-pinned) | SHA-256 |
|---|---|
| screen predictions `analysis/cyp_010_screen_eval_20261125/inputs/screen_predictions.csv` | 0a04fe735ea5d55e2fbd0ad72ab62e7f696d137f2865cec1213211a48eeb5d1f |
| mapping `screen_input_mapping.csv` | fd9ab197b736039f497c62085ce0617e8ce3d4f0631729f8c93fa761887111e3 |
| CYP potency `cyp_potency_ensemble.csv` | 4032dc19e2a192f721ecedb6d7dd7efc24a1dd7a58dbe5577ee29ea466504499 |
| PXR potency `pxr_potency_means.csv` | 1255e03d602448697356d7f4f0fae6879dfddc84fd8c26695c4161859825badb |

| Output | SHA-256 |
|---|---|
| `screen_report.csv` (208 rows, 22 columns) | 8dff51a1a2b00c77fbae6e02b315714b9c387c1de07b06fabc2d29e76fa7dd33 |
| `join_report.json` | 5d079ffe4a29fd917f05a3dde33055ea8df29cdb0d9687eaa1866388b70cfb54 |

Column order: `OCNT_ID`, `cyp_molecule_name`, `pxr_row_id`, `pxr_role`, `pxr_eligibility_reason`; four `*_pIC50_direct_inhibition_pred_mean150`; `PXR_pEC50_multitask_mean5` (primary); `PXR_pEC50_aux_off_mean5` (secondary); four `*_screen_log2fc`; `PXR_screen_8p25uM_log2fc_seed20261125`, `_seed20261126`, `_seed20261127`; `PXR_screen_8p25uM_log2fc_mean3`; `screen_status`; `join_status_cyp`; `join_status_pxr`.

Join counts (`join_report.json`): 208 screen, mapping, CYP potency, PXR potency and report rows; `join_status_cyp` matched 208; `join_status_pxr` matched 208; `screen_status` ok 208; zero duplicate keys in any of the six keyed columns; zero screen ids not in the mapping; zero mapping rows without a screen row; zero potency ids not in the mapping; mapping `cyp_molecule_name` equals `OCNT_ID` for all 208; zero screen/CYP SMILES disagreements.

Assistant's local check: every screen value, CYP potency value and PXR potency value in the report equals its source cell as text (0 mismatches); no empty cells; 208 unique OCNT ids; roles primary_validation 160 and aux_only 48; the 11 aux_only rows with reason `aux_only_group_contains_test_structure` remain in the report (the 197-molecule evaluation set is a separate subset; no evaluation rerun). No combined score, spread column, flag, queue or cutoff (D027); CYP potency values are deployment predictions of the 150-member pool, not held-out (D028). Not committed.

## CoreHPC read-only lookups (reported 2026-11-26 from the login host)

- 150 cells present under the updated-T121 cell root with `proposal.json`; 302 reference files.
- Dataset SHA-256 b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278 and split-assignments SHA-256 4cea719334a0836e4b9c0d50dd093872d472142b33337eb98362f95aa10c1902 equal the local files (closes the Session 007 unreported equality check).
- Executor source in the production-chain package `executor_source` directory: 625a64fae191573b5a860cdba9a6ae9456d17bc6cd5b0337cc0b7cff1df1a00b (equals the pinned value; this resolves the abbreviated `625a64fa...1a00b` recorded earlier).
- PXR final root `/scratch/user/mhuang8/pxr002_final_20261118`: `final_plan.json` SHA-256 3862c5dc38f93952337bfea4edbb60be74614373b4fd6fc88dbd2d9e68455dcf; ten model directories, each with `best.ckpt`, `manifest.json` and `metrics.json`.
- PXR package with `pxr/scripts` and `pxr002_aux_v2`: `/scratch/user/mhuang8/tripwire_staging/pxr002_stage_20261118_r6`.
- No inference, transfer or job has run.

Not covered by these tests (as of the first run; the CYP real Chemprop path was later covered by the `RealModelSeam` tests and a one-member CoreHPC gate, see above): the real Chemprop path (`predict_native`, `build_member_model`) was untested locally; the PXR tests use fake model loading.
