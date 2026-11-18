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

## Queue check (2026-11-18)

Miles pasted a squeue check: none of the stray srun jobs 3106182, 3106184, 3106189, 3106196, 3106200 remained; only unrelated job 3105596 (actfound-d015-chain, CYP workstream) was running. Output not stored in this checkout.
