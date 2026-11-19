# Tripwire-PXR-002 Supporting Record

**Date:** 2026-11-17, continuing into 2026-11-18
**Report:** `notes/Tripwire_PXR_002.md`

## Auxiliary eligibility artifact, `pxr/artifacts/pxr002_aux/` (local checkout, built in conda env `tripwire_cyp002`)

| File | SHA-256 |
|---|---|
| pxr002_eligibility.csv | 0d5f2d0093ee46763323f801f155730ec591890fd96a835ff4b51ef24ac7e7a9 |
| pxr002_train_frame.csv | 024ed1072a399fc13df0fac5dc26b0d913416c0fb62fef1c8b66547907feec06 |
| pxr002_validation_frame.csv | 09bab2a2f76003432692942d84b731b9ad0376585c494360fb66f4c6ec19ff82 |
| pxr002_counts.json | 7e030365542b8f583d4e5b166152a7c1c4910db71de27bd7f91cb6b13c6d500a |

Hashes as reported by Miles; not recomputed by the bookkeeper.

## Implementation files (local, repo root; not yet committed)

Hashes as reported by Miles from local commands; not recomputed by the bookkeeper. The four artifact data-file hashes above are unchanged.

| File | SHA-256 |
|---|---|
| pxr/scripts/pxr001_split.py | 1778cc7c6ff4156f5df51ccbe5db5221410b1e89cf9ee0826fc80fe67a3b8455 |
| pxr/scripts/pxr002_aux_data.py | b6c2903106f36ae19416de48c046c3ce9af1b1d02daa34ed60dfcdc89168a10e |
| pxr/scripts/pxr002_dataset.py | 0ede072855ce41e996cca2359b248371d6b8a8462bb6fa2ba584a625f59e4ae2 |
| pxr/scripts/pxr002_grid.py | 07e42db1d3da595ed3bda6b94a8e0a4ffde6d43398f26fbf26ff6c4b516d3946 |
| pxr/scripts/pxr002_loss_weights.py | 70d4c50cfdc6f2ceda216ac3dda5051aa35199712e9e429b9cc00984db9e0e3f |
| pxr/scripts/pxr002_runtime_estimate.py | 7c3a3ce36f707cd454057d01d24b1f476dfd96dfd1e38110df8e01548aac1a6e (after it was made deterministic-mode, matching pxr002_train.py) |
| pxr/scripts/pxr002_train.py | 529755c14e4979c2ce46b667dab8af488cd79fad47cdcbc4e19539bca712b238 |
| pxr/artifacts/pxr002_aux/PROVENANCE.txt | 24d41f138fbf74388df4d6bcb197b91f105bf311da7a07069e134e89b254e685 |
| pxr/artifacts/pxr002_aux/SHA256SUMS | 29e6bd3495a3bb50354ec1d0fc969ad2957e422aafc966b2055e39a144a81c3a |

`pxr/tests/test_pxr002.py`: hash not reported.

## Runtime-estimate job 3106209 (CoreHPC, 2026-11-17)

Wrapper `pxr/scripts/pxr002_runtime_estimate.slurm`: local hash not reported. Node ggpu1-12, L40S 46068 MiB, driver 595.91.07; start 2026-11-17T23:26:43, elapsed 00:03:11; FAILED, ExitCode 1:0. 13 staged files passed `sha256sum -c`; chemprop 2.3.1; ValueError in `compute_descriptors` (4 non-finite values, 1 molecule, OADMET-0003727). Stray queued jobs from failed srun: 3106182, 3106184, 3106189, 3106196, 3106200 (cancellation unconfirmed). Source: Miles's report of CoreHPC output; job logs not in this checkout.

## v2 amended artifact, `pxr/artifacts/pxr002_aux_v2/` (local, 2026-11-18)

Built by `pxr/scripts/pxr002_artifact_v2.py` (hash not reported) in `tripwire_cyp002` from the frozen v1 artifact. Parent: v1 `SHA256SUMS` sha256 29e6bd3495a3bb50354ec1d0fc969ad2957e422aafc966b2055e39a144a81c3a (v1 not modified). Counts and coefficient are in the report, section 5a. Hashes of the v2 files: not yet reported. `pxr/tests/test_pxr002.py` passes after the v2 additions; exact summary line not pasted. Source: Miles's report of local runs.

## Stage archive (v1, 2026-11-17)

`pxr002_stage_20261117.tar.gz` (repo root, untracked; contains `pxr001_split.py`, `pxr002_*.py` and `pxr/artifacts/pxr002_aux`): SHA-256 acd3149dfb62e82d670f952d01b842621d8eaebaba5ef74be0a09a60d6defbb7, as reported by Miles. Hash list saved as `pxr002_stage_hashes_20261117.txt` (repo root). Transferred and extracted on CoreHPC at `/scratch/user/mhuang8/tripwire_staging/pxr002_stage_20261117`; job 3106209 verified the 13 staged files there (corrects the earlier statement that it was never transferred).

## Stage archive (v2, 2026-11-18) and job 3111835

`pxr002_stage_20261118.tar.gz` and hash list `pxr002_stage_hashes_20261118.txt` (repo root, untracked); individual hashes not pasted to the records. Staged to CoreHPC; pre-transfer and hash checks reported done by Miles. Batch job 3111835 (wrapper `pxr002_runtime_estimate_v2.slurm`; local hash not reported), node ggpu1-08, L40S 46068 MiB, driver 595.91.07; 2026-11-18T00:40:26 to 00:40:48, elapsed 22 s, COMPLETED 0:0. 16 files passed `sha256sum -c` against the 20261118 list (scripts including `pxr002_artifact_v2.py`; v2 artifact including both descriptor `.npy` caches and the amendment CSV); chemprop 2.3.1. Timings and projection are in report §5b. Source: Miles's report of CoreHPC output; job logs not in this checkout.

**Superseded by the validation-MAE fix.** The `pxr002_train.py` hash (529755c1...) and the 20261118 archive reflect the version before the fix. Replaced by the r2 package below.

## Stage archive (r2, 2026-11-18) and job 3113883

`pxr002_stage_20261118_r2.tar.gz` and hash list `pxr002_stage_hashes_20261118_r2.txt` (repo root, untracked); hash values and extraction checks reported done by Miles, not pasted (no hashes recorded). Local `pxr/tests/test_pxr002.py` in `tripwire_cyp002`: 17 passed (summary line pasted by Miles; includes the primary-validation-MAE-in-original-units regression test; sklearn invalid-value warnings there come from all-NaN auxiliary columns in the synthetic frame). Wrapper `pxr002_runtime_estimate_v2.slurm` reused unchanged. Batch job 3113883, node ggpu1-08, L40S 46068 MiB, driver 595.91.07; 2026-11-18T01:36:14 to 01:36:35, exit 0. All 16 listed files passed `sha256sum -c`; chemprop 2.3.1. Results are in report §5b. Source: Miles's report of CoreHPC output; job logs not in this checkout.

## Stage A launch (r3 package, job 3113907, 2026-11-18)

Report section: 5c. New files (local, uncommitted): `pxr/scripts/pxr002_stage_a_runner.py` (hash not reported), `pxr/scripts/pxr002_stage_a.slurm` (8 h wrapper; SHA-256 74d5377277be3d05936f1d46ffe63e707c26d3127252a1ea4eabdefce9222882, local, as reported), stage A runner test in `pxr/tests/test_pxr002.py` (result of Miles's run not reported). Wrapper settings: 8 h, 1 L40S, 8 CPUs, 96G, `--no-requeue`; requires `AUTHORIZE_STAGE_A=yes`, refuses an existing output root, verifies the staged hash list first. The wall time was briefly 12 h, then 8 h before transfer.

Names proposed by Claude: archive `pxr002_stage_20261118_r3.tar.gz`, hash list `pxr002_stage_hashes_20261118_r3.txt` (both present untracked at repo root per git status; hash values not recorded), output root `/scratch/user/mhuang8/pxr002_stage_a_20261118`. Not reported by Miles: his test run, hash comparisons, the exact sbatch line. Whether these names and paths were used, and whether staged files matched the reviewed hashes: UNKNOWN. Seed: 20261117 proposed by Claude; actual seed used not confirmed. Job 3113907 submitted on CoreHPC (source: Miles's report); outcome in the next section; no logs in this checkout.

## Stage A results, job 3113907 (2026-11-18)

Report section: 5d. sacct: COMPLETED 0:0, node ggpu1-08, 2026-11-18T02:25:39 to 03:41:47, elapsed 01:16:08 (8 h requested). Source: Miles's pasted sacct and `stage_a_status.jsonl`; not stored in this checkout. `.out` hash-check lines not pasted. All 36 configurations returned code 0, seed 20261117, output root `/scratch/user/mhuang8/pxr002_stage_a_20261118`. Values are best primary-head validation MAE as reported (single seed); not the official `pxr002_grid.py` ranking. Configuration naming: h{hidden}_d{depth}_f{ffn layers}_p{dropout}.

| # | Configuration | Validation MAE | Epochs run | Seconds |
|---|---|---|---|---|
| 1 | h300_d3_f1_p0 | 0.56166 | 26 | 120.4 |
| 2 | h300_d3_f1_p0.1 | 0.55773 | 24 | 115.1 |
| 3 | h300_d3_f1_p0.2 | 0.55961 | 25 | 118.6 |
| 4 | h300_d3_f2_p0 | 0.54221 | 24 | 114.4 |
| 5 | h300_d3_f2_p0.1 | 0.54971 | 27 | 129.3 |
| 6 | h300_d3_f2_p0.2 | 0.53681 | 24 | 117.1 |
| 7 | h300_d4_f1_p0 | 0.55901 | 26 | 124.1 |
| 8 | h300_d4_f1_p0.1 | 0.54396 | 26 | 125.5 |
| 9 | h300_d4_f1_p0.2 | 0.55334 | 26 | 125.4 |
| 10 | h300_d4_f2_p0 | 0.54533 | 24 | 117.0 |
| 11 | h300_d4_f2_p0.1 | 0.54248 | 24 | 117.7 |
| 12 | h300_d4_f2_p0.2 | 0.54167 | 26 | 126.1 |
| 13 | h300_d5_f1_p0 | 0.55829 | 26 | 128.1 |
| 14 | h300_d5_f1_p0.1 | 0.56023 | 26 | 131.1 |
| 15 | h300_d5_f1_p0.2 | 0.55098 | 26 | 131.0 |
| 16 | h300_d5_f2_p0 | 0.54260 | 24 | 120.8 |
| 17 | h300_d5_f2_p0.1 | 0.53861 | 24 | 120.9 |
| 18 | h300_d5_f2_p0.2 | 0.53574 | 26 | 129.5 |
| 19 | h600_d3_f1_p0 | 0.56706 | 24 | 119.9 |
| 20 | h600_d3_f1_p0.1 | 0.55832 | 24 | 118.9 |
| 21 | h600_d3_f1_p0.2 | 0.56523 | 26 | 128.2 |
| 22 | h600_d3_f2_p0 | 0.54247 | 24 | 119.4 |
| 23 | h600_d3_f2_p0.1 | 0.54591 | 24 | 121.5 |
| 24 | h600_d3_f2_p0.2 | 0.54513 | 26 | 128.5 |
| 25 | h600_d4_f1_p0 | 0.55724 | 26 | 130.5 |
| 26 | h600_d4_f1_p0.1 | 0.55836 | 24 | 123.0 |
| 27 | h600_d4_f1_p0.2 | 0.54847 | 26 | 134.1 |
| 28 | h600_d4_f2_p0 | 0.53970 | 24 | 124.8 |
| 29 | h600_d4_f2_p0.1 | 0.54907 | 27 | 141.2 |
| 30 | h600_d4_f2_p0.2 | 0.52960 | 26 | 134.3 |
| 31 | h600_d5_f1_p0 | 0.56237 | 25 | 134.7 |
| 32 | h600_d5_f1_p0.1 | 0.55980 | 26 | 139.5 |
| 33 | h600_d5_f1_p0.2 | 0.55379 | 25 | 132.3 |
| 34 | h600_d5_f2_p0 | 0.54903 | 24 | 131.7 |
| 35 | h600_d5_f2_p0.1 | 0.54319 | 26 | 142.1 |
| 36 | h600_d5_f2_p0.2 | 0.54063 | 26 | 142.6 |

Seconds per run were supplied as a separate list and paired to rows by the stated order (same order as the MAE list).

## Stage B preparation (2026-11-18)

Report section: 5e. New files (local, uncommitted, hashes not reported): `pxr/scripts/pxr002_stage_b_runner.py`, `pxr/scripts/pxr002_stage_b.slurm` (1 h wall request; requires `AUTHORIZE_STAGE_B=yes`); stage B runner tests added to `pxr/tests/test_pxr002.py` (not yet run by Miles). Local test run: 21 passed, 1 failed of 22 (failing test `test_stage_b_aggregate_uses_stage_a_value_plus_new_seeds` had a wrong expected ranking; means x 0.5733, y 0.5300, z 0.5467 rank y, z, x); test expectation corrected, production code unchanged; Miles then reported pass (summary line not pasted). Stage B submitted as batch job 3119543 on CoreHPC. Proposed (not confirmed) names: `pxr002_stage_20261118_r4.tar.gz`, `pxr002_stage_hashes_20261118_r4.txt`, `pxr002_stage_b.slurm`, stage A root `/scratch/user/mhuang8/pxr002_stage_a_20261118`, output root `/scratch/user/mhuang8/pxr002_stage_b_20261118`. Hash comparisons, 36-result count check, exact sbatch line and file hashes not reported; outcome UNKNOWN. Source: Miles's reports relayed in the call.

## Stage B results, job 3119543 (2026-11-18)

Report section: 5f. Node ggpu1-11 (L40S), 14:28:32 to 14:42:05, elapsed 00:13:33, COMPLETED 0:0. 18 listed files (r4 package) passed `sha256sum -c`; chemprop 2.3.1; six runs rc=0, 24-30 epochs, about 2.2 min each. Best primary-head validation MAE (20261117 = stage A value):

| Configuration | 20261117 | 20261118 | 20261119 | Mean |
|---|---|---|---|---|
| h600_d4_f2_p0.2 | 0.52960 | 0.53900 | 0.54396 | 0.53752 |
| h300_d5_f2_p0.2 | 0.53574 | 0.53768 | 0.54985 | 0.54109 |
| h300_d3_f2_p0.2 | 0.53681 | 0.55002 | 0.54755 | 0.54479 |

Epochs: h600_d4 27, 25; h300_d5 27, 30; h300_d3 28, 24 (seeds 20261118, 20261119). Fresh-seed-only means (Claude's calculation): 0.5415, 0.5438, 0.5488. Final-epoch log values 0.572-0.589 are not the best checkpoint. Output root path not shown. Source: Miles's report of CoreHPC output; logs not in this checkout.

## Auxiliary-off ablation preparation (2026-11-18)

Report section: 5g; decision D018 Amendment 4. New files (local, uncommitted, hashes not reported): `pxr/scripts/pxr002_ablation_runner.py`, `pxr/scripts/pxr002_ablation.slurm` (1 h; requires `AUTHORIZE_ABLATION=yes`). Edited files (hashes not reported; prior hashes in earlier sections are superseded for these two): `pxr/scripts/pxr002_loss_weights.py` (`aux_off_weight_vector`), `pxr/scripts/pxr002_train.py` (`--aux-off`, default off; manifest records `aux_off`). Four tests added to `pxr/tests/test_pxr002.py`; not yet run by Miles. Submitted by Miles as batch job 3120327 on CoreHPC. Proposed (not confirmed) names: `pxr002_stage_20261118_r5.tar.gz`, `pxr002_stage_hashes_20261118_r5.txt`, `pxr002_ablation.slurm` (1 h, 1 L40S), stage A root `/scratch/user/mhuang8/pxr002_stage_a_20261118`, stage B root `/scratch/user/mhuang8/pxr002_stage_b_20261118`, output root `/scratch/user/mhuang8/pxr002_ablation_20261118`. Test run (26 passed expected), hash comparisons, stage A/B result-file checks and the sbatch line not reported; outcome UNKNOWN. Source: Miles's decision and report, relayed in the call.

## Ablation results, job 3120327 (2026-11-18)

Report section: 5h. Node ggpu1-11, 15:36:56 to 15:43:38, elapsed 00:06:42, COMPLETED 0:0. 19 listed files (r5 package) passed `sha256sum -c`; chemprop 2.3.1. Configuration h600_d4_f2_p0.2. Best primary-head validation MAE by seed 20261117 / 20261118 / 20261119: multitask 0.52960 / 0.53900 / 0.54396 (mean 0.53752); auxiliary-off 0.53036 (26 ep) / 0.54472 (25 ep) / 0.54911 (25 ep) (mean 0.54140, range 0.53036-0.54911). Mean difference +0.00388; fresh-seed-only means ablation 0.54692, multitask 0.54148 (+0.00544); per-seed differences +0.0008, +0.0057, +0.0051 (Claude's calculation). Label: not_distinguishable. Final-epoch log values 0.589-0.593 are not the best checkpoint. Source: Miles's report of CoreHPC output; logs not in this checkout. The structure-only check is not recorded.

## Holdout structure-only check (2026-11-18)

Claude read only holdout SMILES and identities from the D014 split (pEC50 column not read). Result: all 621 holdout molecules have finite `v1_rdkit_2d_normalized` descriptors; 0 non-finite. Not a holdout evaluation. Source: Claude's report relayed in the call; script and command not recorded.

## Improvement criterion (2026-11-18)

Decision recorded as D018 Amendment 5; see report §6. Floor reference values (Claude's): v2 cohort 2,896 primary labels; v1 training mean pEC50 4.328; constant-predictor validation MAE 0.9216. Claude's protocol proposal (5 seeds 20261120-20261124, multitask primary, aux-off secondary) is pending Miles's answer.

## Final protocol preparation (2026-11-18)

Report section: 5i; decision D018 Amendment 6. New files (local, uncommitted, hashes not reported): `pxr/scripts/pxr002_holdout_artifact.py`, `pxr002_final_runner.py`, `pxr002_final_predict.py`, `pxr002_final_eval.py`, `pxr002_final.slurm` (1.5 h; requires `AUTHORIZE_FINAL_TRAINING=yes`); tests added to `pxr/tests/test_pxr002.py` (not yet run by Miles). Seeds 20261120-20261124. Submitted by Miles as batch job 3120551 on CoreHPC (launch carried `AUTHORIZE_FINAL_TRAINING=yes`); trains ten models, writes holdout predictions from the structure-only artifact, reads no holdout labels; no holdout evaluation run. Proposed (not confirmed) names: `pxr002_stage_20261118_r6.tar.gz`, `pxr002_stage_hashes_20261118_r6.txt`, `pxr002_final.slurm` (1.5 h, 1 L40S, 8 CPUs, 96G), `pxr/artifacts/pxr002_holdout_v1`, stage B root `/scratch/user/mhuang8/pxr002_stage_b_20261118`, output root `/scratch/user/mhuang8/pxr002_final_20261118`. Test run (34 expected, count uncertain), holdout-artifact build, hash comparisons, ranking-file check and sbatch line not reported; outcome UNKNOWN. Source: Miles's report, relayed in the call.

## Final training and prediction results, job 3120551 (2026-11-18)

Report section: 5j. Node ggpu1-07 (L40S 46068 MiB, driver 595.91.07), 2026-11-18T16:23:17 to 16:46:15, elapsed 00:22:58, COMPLETED 0:0. 27 listed files (r6 package incl. holdout artifact) passed `sha256sum -c`; chemprop 2.3.1; h600_d4_f2_p0.2; "10 ok, 0 failed". Best validation MAE (epochs), seeds 20261120 / 21 / 22 / 23 / 24: multitask 0.54910 (28) / 0.54084 (24) / 0.55170 (23) / 0.54996 (30) / 0.53744 (27); aux_off 0.56271 (29) / 0.54129 (23) / 0.54213 (23) / 0.56134 (29) / 0.54211 (25). Claude's arithmetic: means 0.5458 (range 0.5374-0.5517) and 0.5499 (range 0.5412-0.5627), difference +0.0041. Replication gate: all ten replicated validation MAE to six decimals (tolerance 1e-3). Output: `/scratch/user/mhuang8/pxr002_final_20261118/predictions/holdout_predictions.csv`, predictions_sha256 45c84c925b9dfc1c5c959e3a95c6b01961667dd19e2338470e8ae26655aa07aa. No holdout labels read on CoreHPC; holdout evaluation not run. Source: Miles's report of CoreHPC output; logs not in this checkout.

## Holdout evaluation (2026-11-18, local)

Report section: 5l. `pxr/scripts/pxr002_final_eval.py` with `--authorize-holdout-evaluation`; output `pxr/artifacts/pxr002_final_20261118/evaluation/final_evaluation.json` (SHA-256 not reported). Predictions SHA-256 45c84c925b9dfc1c5c959e3a95c6b01961667dd19e2338470e8ae26655aa07aa matched the evaluator's gate. n_holdout 621. Constant predictor (mean 4.32822): MAE 0.92372. Multitask ensemble: MAE 0.48427, Spearman 0.73567, mean per-model MAE 0.51357, per-model 0.52899 / 0.49491 / 0.51685 / 0.49694 / 0.53019, minus floor -0.43945. Aux-off ensemble: MAE 0.48469, Spearman 0.73675, mean per-model MAE 0.51034, per-model 0.51236 / 0.50360 / 0.51022 / 0.50821 / 0.51733, minus floor -0.43902. Aux-off minus multitask ensemble MAE +0.00042. First and only holdout label read. Source: Miles's report, relayed in the call; result file is in the local checkout but was not re-read by the bookkeeper.

## Records housekeeping (2026-11-18)

Math in `notes/Decisions.md` and the 20261117 archive converted to fenced blocks and code spans by Claude with Miles's approval; Miles then had both archives deleted (`notes/Decisions.md` is the authoritative log) and approved committing: 59fe1c0 and fa31dab, local, not pushed. The archives-never-edited rule was set aside once by Miles's explicit approval; CLAUDE.md unchanged. Source: Miles's report via the call.

## Queue check (2026-11-18)

Miles pasted a squeue check: none of the stray srun jobs 3106182, 3106184, 3106189, 3106196, 3106200 remained; only unrelated job 3105596 (actfound-d015-chain, CYP workstream) was running. Output not stored in this checkout.
