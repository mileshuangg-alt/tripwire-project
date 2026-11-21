# Tripwire-CYP-008 Session Log

**Date:** 2026-11-19 to 2026-11-21  
**Status:** Open. The D021 comparison (updated T121 plus molecule-level descriptors, arm D, versus updated T121, arm R) has been run to its D021 item 6 statistic: 750 new cells over five new screening runs, aggregated. Outcome: arm D is not favoured by the D021 rule. No follow-up for this outcome is decided. Nothing is committed to git; no session close or advance has been declared.

## 1. Session purpose and starting state

Session 008 begins from the closed-out direction in `notes/Tripwire_CYP_007.md` §7 and the handoff `notes/Handoffs/Tripwire_CYP_007_Handoff_20261119.md` (transfers state; not decision authority). Starting state: updated T121 (arm R) O150 = 0.6769540372619428 (replicate 1, seed 20261101, selected epoch 16); D021 approved as a plan in `notes/Decisions.md` (authorizes local implementation and the descriptor cache build; no training, screening, cell run, GPU job or Slurm submission without separate approval after its item 8 gates). Per-step run approvals are not itemized in this log beyond what is stated below; the one explicit deviation approval is in §3.

Purpose: run the D021 comparison and report its item 6 result. CheMeleon remains queued after the descriptor comparison.

## 2. D021 launch preparation

A separate hash-bound descriptor package `t121_d021_package_20261119` (tarball SHA-256 bd73e8c3a6f1257e24587eb78a8102a03e3eb1b534ebe68b5b6e2005601a0978) was built from the reference production-chain tarball (pinned 9ecb6ccf...) plus the D021 overlay. The existing reference package was left untouched.

Code (local, uncommitted): `cyp/scripts/t121_d021_layout.py`, `t121_d021_build_package.py`, `t121_descriptor_cache.py`; `t121_screen_chain_run.py` (optional `descriptor_cache` section); `t121_production_cell.py` (descriptors attached after fold routing); `t121_cells.slurm` (optional `DESCRIPTOR_CACHE_DIR` / `DESCRIPTOR_MANIFEST_SHA256`). A dry run caught a real bug: attaching descriptors before fold routing raised KeyError on protected molecules absent from the cache; the attach step was moved after routing. Local tests passed (61 in conda env `tripwire_cyp002`; 26 inside the built package).

Qualification evidence for arm D was regenerated under the existing standard (PASS file, pinned SHA-256, bound fields): descriptor smoke job 3155222, PASS, COMPLETED 0:0, 32 s, ggpu1-09, descriptor_dim 196, predictor widths [[2000, 796]].

## 3. Timing (batch-Slurm deviation)

Interactive srun failed with "Insane message length" (as in 007 §3). **Approval (scope: this timing only):** a one-off batch-Slurm timing run. It is not a standing permission.

Job 3158166 failed: `timing.py` left SCREEN_EPOCHS at 3, and the trial warmup contradicted the 50-epoch schedule. Fixed (restored to 50) and rerun as job 3158201 (L40S, ggpu1-07, 2:56):

| Quantity | Measured | Extrapolated |
|---|---|---|
| Screening | 4.90 s/epoch | about 251 s for 50 epochs |
| Cell 0 | 147.6 s | about 22,137 s serial for 150 cells |

The cell 0 score 0.7015 came from a 3-epoch screening checkpoint and is not a result.

## 4. Screening runs (five new, no full-screen refit)

Jobs 3161795-3161799, all COMPLETED 0:0, status completed, 50 epochs.

| Run | Seed | Selected epoch | Record SHA-256 (abbreviated) |
|---|---|---|---|
| r_s2 | 20261102 | 31 | 0bab1c8a...eeb3 |
| r_s3 | 20261103 | 29 | c7c71844...513f |
| d_s1 | 20261101 | 29 | a03d270c...d5bc |
| d_s2 | 20261102 | 46 | ce1666a8...c40b |
| d_s3 | 20261103 | 28 | 9a415bd1...fd68 |

R replicate 1 is the existing run (selected epoch 16). Full hashes are in the supporting record.

## 5. 150-cell arrays

Five 150-cell arrays, planned from the D package and submitted with `--time=00:30:00`: r_s2 3161888, r_s3 3161889, d_s1 3161890, d_s2 3161892, d_s3 3161893. All 750 cells completed: 150 COMPLETED.json and 0 INCOMPLETE.json per run, every task COMPLETED 0:0. Cell layout: 5 repetitions x 10 folds x 3 cell seeds. Per-cell time about 1.5 to 2.5 min.

## 6. Aggregation and result

New script `cyp/scripts/t121_d021_aggregate.py` (SHA-256 55ade624c4cd67a9c4fbc03430e29274e179f20f44517851457a9a2037d42c39), written after the cell-output format was supplied and a go-ahead was given; tests `cyp/tests/test_t121_d021_aggregate.py` (11 passed locally). Checks: 150 cells per run; `result.json` hash against COMPLETED.json; one screening checkpoint per run; shared partition layout across runs; reproduction of R replicate 1's recorded O150 within 1e-12. The first attempt stopped on an over-strict uniqueness check keyed on (repetition, fold); legitimate duplicates exist because of the three cell seeds. It was corrected to (repetition, fold, seed) and rerun; the failed attempt wrote no report.

Report: `/scratch/user/mhuang8/tripwire_staging/t121_d021_o150_report_20261121.json` (CoreHPC; its own hash not computed).

D021 item 6 statistic: O150 is the mean macro ST-RAE over 150 cells per screening checkpoint (lower is better); Delta_s = D - R.

| Seed slot | O150 R | O150 D | Delta_s |
|---|---|---|---|
| 1 | 0.676954037 (reproduces recorded 0.6769540372619428 within 1e-12) | 0.682479984 | +0.005525946 |
| 2 | 0.668499342 | 0.680816837 | +0.012317495 |
| 3 | 0.669626167 | 0.686278215 | +0.016652049 |

Delta-bar = +0.011498497; O_bar(R) 0.671693182; O_bar(D) 0.683191679. By the D021 rule (D favoured only if Delta-bar < 0 strictly), **D is not favoured**.

Per-isoform Delta-bar: CYP1A2 +0.012091654, CYP2C9 +0.003409566, CYP2D6 +0.009012727, CYP3A4 +0.021480040. All three Delta_s are positive. No significance claim is made; the replicates share partitions and are not independent.

## 7. Claim boundary

Descriptive result on development validation data under D021 item 6 only. No descriptor-aware inference or diagnostics scripts have been built. The report JSON hash has not been computed. Nothing is committed to git. A discussion of the screening-score concept (the log2fc assay definition is not confirmed in the repo) is not a result. D021 states no follow-up for a not-favoured outcome, and no next step is decided. No held-out evaluation is authorized and no protected internal test or challenge data were used. No entry has been made in `notes/Decisions.md` for this session.

## 8. Stopping point and next step

Stopping point: D021 item 6 result computed and recorded (§6). Next step: undecided; any follow-up (including CheMeleon, queued from 007 §7) requires an explicit decision entry. Session close or advance has not been declared.

Supporting record: `notes/Tripwire_CYP_008_supporting.md`.
