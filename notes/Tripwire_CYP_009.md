# Tripwire-CYP-009 Session Log

**Date:** 2026-11-21 (opened); closed 2026-11-23  
**Status:** Closed on declaration, 2026-11-23. D022 (as completed by Amendment 1) was implemented locally, deployed to CoreHPC, qualified (smoke job 3202933; approved one-off batch timing job 3203300), run through three 50-epoch screening runs (jobs 3203564-3203566) and three 150-cell arrays (jobs 3207090-3207092), and aggregated (section 9): CheMeleon is not preferred under D022 Amendment 1 item 5. The results are frozen by D023 (section 10). Next session: Tripwire-CYP-010; its scope and direction have not been stated.  
**Supporting record:** `notes/Tripwire_CYP_009_supporting.md` (hashes and identifiers).

## 1. Session purpose and starting state

Session 009 takes up the CheMeleon work queued in `notes/Tripwire_CYP_008.md` §8a. Starting state: D022 (Approved as a design only) fixes the pretrained CheMeleon encoder, fully fine-tuned through screening then pIC50, with head learning rate equal to the updated-T121 Noam schedule and encoder rate 0.1 times it. It left aggregation, dropout, batch size and the acceptance rule open. D021 outcome (arm D not favoured) is unchanged and unrelated.

## 2. Design decisions (D022 Amendment 1)

Recorded in `notes/Decisions.md` as D022 Amendment 1 (2026-11-21); D022's existing text is unchanged.

- Mean aggregation, matching the pretrained encoder, replaces the current norm aggregation for this design.
- Encoder dropout 0.0 (from the CheMeleon checkpoint); head dropout 0.4 (updated T121's head setting); batch size 16 (updated T121's). The pretrained encoder's setup and updated-T121 head settings are kept distinct.
- Acceptance rule: CheMeleon versus the descriptor-free updated T121 (arm R) over the same three screening seeds with matched O150 evaluation. CheMeleon is preferred only if its mean macro ST-RAE (mean over seeds of O150, lower is better) is lower. Paired seed differences and per-isoform results are reported alongside; no significance claim for a lower mean.

## 3. Read-only implementation audit (2026-11-21)

Source: code inspection of `cyp/scripts/t121_weighted_training.py`, `t121_screen_stage.py`, `cyp002_model.py`, `cyp007_infrastructure.py`, `pxr/scripts/pxr004_train.py`, and local checks in conda env `tripwire_cyp002` with chemprop 2.3.1.

- **Weights.** `pxr/artifacts/pxr004_weights/chemeleon_mp.pt`, SHA-256 c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651. Encoder d_h 2048, depth 6, dropout 0.0, d_v 72, d_e 14.
- **Featurizer.** The CYP default featurizer produces arrays identical to CheMeleon's V2 featurizer on a test molecule; no featurizer change is needed.
- **Learning rates.** Chemprop `MPNN.configure_optimizers` uses one Adam and one LambdaLR Noam schedule scaling each group's initial rate by the same factor, so two groups with encoder initial rate 0.1 times the head's give exactly 0.1 times at initial, peak and final values. A `configure_optimizers` override is needed on the screening and pIC50 model classes.
- **Stage transfer.** `transfer_screen_initialization` copies all shared keys including the encoder, except the final projection and output transform, with no optimizer or scheduler state; the D022 reset rule is met by the existing path.
- **Blocker found.** `build_cyp002_model` builds its own `BondMessagePassing` from `config.message_hidden_dim`, and `config_from_trial` validates against the frozen D007 search space (maximum 2400, step 100), so 2048 is not representable; the encoder must be injected in the screening builder and `build_updated_t121_pic50_prefit_state`.
- **Updated T121 reference values** (reconstructed config, `artifacts/d009_closure_workspace` manifests): message dim 600, depth 4, FFN 2000 x 1 layer, dropout 0.4, batch 16.
- **Not audited:** the model-rebuild path in the validation, inference and diagnostics scripts.

## 4. Implementation (D022 as amended; local, within the approved plan)

All edits are local and uncommitted. The frozen D007 search space and the pinned executor `cyp007_cell_fit.py` are untouched.

- `cyp/scripts/t121_chemeleon.py` (new): hash-verified CheMeleon loader; encoder injection with mean aggregation and a head sized to 2048; a two-parameter-group optimizer mixin with encoder learning rate 0.1 times the head's. Descriptors combined with chemeleon are refused.
- Frozen config `cyp/artifacts/d022_chemeleon/chemeleon_config.json`, SHA-256 444d37bc3f2501237e5734893a48522bd60b93b8cbed5ece3a3324830eb67e4b.
- Opt-in `chemeleon` parameter in `t121_weighted_training.py`, `t121_screen_stage.py` and `t121_production_cell.py` (CLI `--chemeleon-config`, `--chemeleon-config-sha256`, `--chemeleon-weights`); `t121_screen_chain_run.py` (manifest section `chemeleon`, pinned roles chemeleon_module, config, weights); `t121_cells.slurm` (optional trio).
- `t121_d022_aggregate.py` (acceptance per D022 Amendment 1 item 5), `t121_d022_layout.py`, `t121_d022_build_package.py`; qualification smoke and timing scripts under `qualification/t121/d022_*`.
- `cyp/tests/test_t121_chemeleon.py`, `test_t121_d022_aggregate.py`, `test_t121_d022_layout.py`.

**Local verification** (env `tripwire_cyp002`): 535 tests passed excluding `cyp/tests/test_t121_training_weights.py`, which fails at collection both with and without these edits (`cyp002_model.py` lacks functions that test parses; pre-existing and unrelated); 53 passed inside the built package; graph-only golden hashes of the arm R and descriptor builds are unchanged.

## 5. Package and deployment

`t121_d022_package_20261121.tar.gz`, SHA-256 781ba3f327e53fd7f624bfe529b0c133da0a95d66b6c85331a1d962ea814900f, 32,857,437 bytes, 113 files in `SHA256SUMS.txt`, 22 pinned roles. Built from the unchanged reference package plus the D022 overlay; CheMeleon weights SHA-256 c376624d...183651 checked against the frozen config. Deployed to CoreHPC `/scratch/user/mhuang8/tripwire_staging/t121_d022_package_20261121`; the `SHA256SUMS` check was run there and reported no non-OK lines.

## 6. Qualification smoke

**v1 failed (job 3191973, FAILED 1:0, 32 s; package tarball SHA-256 4202a97a...befb39).** The check's head-width assertion (check.py line 122) counted every 2-D tensor whose key contained 'predictor', including `predictor.criterion.task_weights` (1,1). No `result.json` was written, so nothing from v1 counts as evidence. Cause: a defect in the check script (an over-loose filter written without inspecting real key names and without running the assertions beforehand), not in the model code. Fix: filter to keys starting `predictor.ffn.`. A weak assertion in `test_t121_chemeleon.py` (projection filter matched nothing) was also fixed. A local CPU harness with a synthetic screen, run on the post-preparation assertions, passes with the fixed filter and reproduces the v1 failure with the old filter. The v1 package directory and tarball were set aside on CoreHPC (`t121_d022_package_20261121_v1_smoke_failed`, `..._v1.tar.gz`) and locally (`t121_d022_package_20261121_v1_smoke_failed.tar.gz`); nothing was deleted.

**v2 passed (job 3202933; partition gpu, node ggpu1-08, COMPLETED 0:0, 00:00:42; L40S was the requested GPU type, the node's GPU type was not separately reported).** Package tarball SHA-256 781ba3f3...814900f. Evidence source: the readout pasted into the session (saved-result line and log tail). The top of `result.json` with its "status" field was not in the excerpt; the script writes `result.json` only after all assertions pass and the log ends with its SAVED line. `result.json` (CoreHPC `.../t121_d022_package_20261121/results/d022_chemeleon_smoke/result.json`) SHA-256 62c03470...ed6; pasted fields confirm the check.py hash equals the one shipped in the tarball (hashes in the supporting record).

Scope of the smoke: 2-epoch real `Trainer.fit` with the pretrained weights; two optimizer groups at ratio 0.1; encoder starts equal to the pretrained tensors, then moves; head shapes (2000x2048, 4x2000); mean aggregation; batch 16; selected-checkpoint pIC50 handoff with encoder transferred and a fresh final projection; authorization gate. Not executed: 50-epoch screening, any pIC50 fit, the 150-cell array, trained performance.

The timing script's two-optimizer-group assertion on a cell checkpoint, untested at the smoke stage, passed in the timing run (section 7).

## 7. Interactive GPU access

`salloc` and `srun` both fail on the CoreHPC login VM with "Insane message length" (observed 2026-11-18 with `srun`, and again in this session with `salloc`). The interactive timing measurement the working rules require therefore cannot currently be made. `large_gpu` is closed to the account (job QOS not permitted).

**Approved deviation (2026-11-23; scope: D022 CheMeleon arm timing only).** A one-off batch-Slurm timing run is approved: `qualification/t121/d022_timing_20261121/timing.py` via its `timing.sbatch`, one L40S, a 2-hour wall-time cap (a cap, not an estimate), measuring 3 screening epochs and one full pIC50 cell and writing only a new scratch output directory. Basis: the working rule requires a measured interactive runtime before production submission, and interactive allocation is unavailable. Same form as the one-off deviation recorded for D021 timing in `notes/Tripwire_CYP_008.md` section 3 (recorded in the session log only; no Decisions.md entry exists for it, so none is added here). This is not a standing permission. It does not authorize the three 50-epoch screening runs or the three 150-cell arrays, each of which still requires separate approval, and it does not waive the extrapolation of measured time against requested wall time.

**Timing result (job 3203300).** The approved batch timing ran: partition gpu, node ggpu1-11, NVIDIA L40S, COMPLETED 0:0, elapsed 00:02:49; package `t121_d022_package_20261121` (tarball SHA-256 781ba3f3...814900f); output directory `/scratch/user/mhuang8/tripwire_staging/t121_d022_timing_20261121`. Evidence source: the job readout and the printed `timing.json`, pasted into the session from CoreHPC; the hash of `timing.json` itself was not computed. The `note` text in `timing.json` says "measured interactively"; it was run as the approved batch job (script text carried over from D021).

| Quantity | Measured |
|---|---|
| Screening prepare | 6.337 s |
| Screening fit, 3 epochs | 24.727 s (8.242 s/epoch) |
| 50-epoch screening extrapolation (includes prepare) | 418.45 s (about 7.0 min) |
| Cell 0 (CheMeleon arm, from a 3-epoch screening checkpoint) | 129.63 s |
| Optimizer groups in cell 0 | 2 (assertion passed) |
| Peak GPU memory | 642,503,680 bytes (about 0.64 GB) |
| Serial 150-cell extrapolation, one array | 19,444 s (about 5.4 GPU-hours) per 150-cell array; arithmetic on one cell. Three arrays: about 16.2 GPU-hours in total (serial arithmetic, not a prediction of wall time) |

- The cell's macro ST-RAE (0.7544) comes from a 3-epoch screening checkpoint and is not a result.
- The logged cell fit reached epoch 25 of a 50-epoch maximum (early stopping, patience 20), so cell time depends on when stopping occurs. By arithmetic on the shown epoch count, per-epoch cost excluding setup is roughly 4.6 s, so a full 50 epochs would be roughly 4 minutes per cell. This is an upper-bound estimate, not measured.
- Comparison: the D021 descriptor timing (`notes/Tripwire_CYP_008.md`) was 4.90 s/epoch screening and 147.6 s for cell 0.

## 8. Launcher validation and screening runs

**Validate-only launcher run.** All three screening manifests (`t121_d022_c_s1`..`s3`) were validated on CoreHPC in validate-only mode; the launcher's checks accepted the smoke result as bound PASS evidence. Source: pasted CoreHPC output.

**Screening runs (approved explicitly; scope: the three 50-epoch D022 CheMeleon screening runs only).** Submitted as batch jobs 3203564, 3203565 and 3203566, each with `--time=00:30:00` (a cap set from the measured 7.0 min extrapolation in section 7, not an estimate). Manifests `t121_d022_manifests/screen_manifest_t121_d022_c_s{1,2,3}.json` were hash-checked with `sha256sum -c` before submission (hashes in the supporting record). The job-to-run mapping was assumed to follow submission order (3203564 = c_s1) and is confirmed by the seeds in the completion records. Source: pasted CoreHPC output; the `completion.json` files were in the excerpt, the `sacct` job-state lines were not.

All three `completion.json` report status completed, 50 epochs recorded, no full-screen refit. Run roots: `/scratch/user/mhuang8/tripwire_staging/t121_d022_c_s{1,2,3}_screen_run`.

| Run | Seed | Selected epoch (zero-based) |
|---|---|---|
| c_s1 | 20261101 | 4 |
| c_s2 | 20261102 | 45 |
| c_s3 | 20261103 | 4 |

Observation only, no claim about quality: two of three seeds selected an early epoch under the existing selection rule. For reference, `notes/Tripwire_CYP_008.md` section 4 records arm R selected epochs 16 (seed 20261101, existing run), 31 (s2), 29 (s3) and arm D 29, 46, 28 (s1 to s3); whether that log's epoch numbering is zero-based is not stated there. Screening scores were not in the pasted output. No screening metric or acceptance comparison has been made.

**Correction.** An earlier chat estimate of about 5.4 GPU-hours for the arrays was wrong. 5.4 GPU-hours (19,444 s) is the serial extrapolation for one 150-cell array; the three arrays are about 16.2 GPU-hours in total (serial arithmetic on one cell, not a prediction of wall time).

## 9. Cell arrays and D022 acceptance aggregation (2026-11-23)

**Cell arrays (approved explicitly; scope: the three 150-cell CheMeleon arrays only).** Three 150-cell arrays were run as batch jobs 3207090, 3207091 and 3207092, each task with `--time=00:30:00` (a command-line override of the script's 12 h; cap, not estimate). The job-to-run mapping (c_s1, c_s2, c_s3 in order) follows submission order and is assumed; the aggregation's per-run screening checkpoint hashes differ for all six runs, and the aggregator checks that each run's cells share one screening checkpoint. Final counts as pasted: 150/150 completed and 0 incomplete for each root (`t121_d022_c_s{1,2,3}_cells_run`; logs `t121_d022_c_s{n}_cells_logs`). Mean elapsed of the first array's 150 tasks was 139.867 s per task (sacct, as pasted) against 129.6 s measured for cell 0 in the timing run (section 7); implied total about 17.5 GPU-hours versus the earlier 16.2 estimate.

**Aggregation.** `t121_d022_aggregate.py` (SHA-256 7a2be5f7...d34c9) and `t121_d021_aggregate.py` (55ade624...2c39) were verified with `sha256sum -c` on CoreHPC before running (both OK). Comparator arm R runs: r1 = `t121_production_chain_20261117_cells_run`, r2 = `t121_d021_r_s2_cells_run`, r3 = `t121_d021_r_s3_cells_run`. The aggregator stops if r1 does not reproduce the recorded O150 0.6769540372619428; it wrote the report, so that check passed. Report: `/scratch/user/mhuang8/tripwire_staging/t121_d022_o150_report_20261123.json`, SHA-256 3f70eb45...a288ab (full hashes in the supporting record). Source: stdout pasted from CoreHPC, plus the report hash printed after it.

O150 (macro ST-RAE; lower is better):

| Seed | R (arm) | CheMeleon | Delta_s = CheMeleon - R |
|---|---|---|---|
| s1 | 0.676954037 | 0.706368862 | +0.029414825 |
| s2 | 0.668499342 | 0.693407594 | +0.024908252 |
| s3 | 0.669626167 | 0.710411351 | +0.040785184 |
| Mean | 0.671693182 | 0.703395936 | +0.031702754 |

Under the D022 Amendment 1 item 5 rule (CheMeleon preferred only if mean Delta_s is strictly below zero), **CheMeleon is not preferred.** Mean Delta_s is positive and every per-seed difference is positive.

Per-isoform Delta-bar (per seed s1, s2, s3), CheMeleon minus R:

| Isoform | Delta-bar | s1 | s2 | s3 |
|---|---|---|---|---|
| CYP1A2 | +0.032520341 | +0.031493 | +0.026601 | +0.039467 |
| CYP2C9 | +0.035926921 | +0.035242 | +0.016496 | +0.056044 |
| CYP2D6 | +0.039066549 | +0.040730 | +0.039479 | +0.036991 |
| CYP3A4 | +0.019297204 | +0.010195 | +0.017058 | +0.030639 |

Every per-seed, per-isoform difference is positive (CheMeleon worse than R).

Observation only, no explanation or claim: the seed with the latest selected screening epoch (c_s2, epoch 45; section 8) has the smallest Delta_s; the other two selected epoch 4. With three seeds this is descriptive only.

## 10. Result freeze (D023, 2026-11-23)

D023 in `notes/Decisions.md` (Approved and frozen) freezes the D022 results: the O150 report (`t121_d022_o150_report_20261123.json`, SHA-256 3f70eb45...a288ab) and its stated result (CheMeleon not preferred; mean paired difference +0.031702754), the three screening runs and three cell roots (150 completed cells each), the deployed package (tarball SHA-256 781ba3f3...814900f), manifests, smoke result and timing output, and the frozen config (SHA-256 444d37bc...e4b). They are preserved unmodified, not overwritten, not re-run into the same roots, and the result is not revised unless a later explicit decision says otherwise. Recorded as a new short entry (D023) rather than a D022 amendment, because D022's amendments settle design items and D012 is the precedent for a stand-alone results-freeze record. The freeze decides no follow-up, does not change D022 or D021, and does not close or alter the Governing-principle question or the leftover untracked files noted in the Session 008 handoff (`notes/Handoffs/Tripwire_CYP_008_Handoff_20261121.md`, §5: the D021 Governing-principle sentence is awaiting an explicit decision entry, and the handling of the leftover untracked files is undecided). Hashes are in `notes/Tripwire_CYP_009_supporting.md`.

## 11. Claim boundary

Implementation, a 2-epoch qualification smoke, timing (section 7), three 50-epoch screening runs (section 8), three 150-cell arrays and the O150 comparison to arm R (section 9). No significance claim: the replicates share partitions and are not independent. The result tests the complete CheMeleon-based setup (D022 design with its screening-then-pIC50 schedule and the 0.1 encoder ratio), not pretraining alone; the 0.1 ratio was a chosen design and was not tuned. No explanation of the difference is established. No held-out evaluation is authorized and no protected internal test or challenge data were used.

## 12. Stopping point and unresolved items

Stopping point: session closed 2026-11-23 on declaration, after the D022 result was computed (section 9) and frozen (D023). No next step is decided; Session CYP-010 follows, its scope and direction not yet stated.

Unresolved at close:
- The batch-timing run (section 7) was a one-off deviation, not a standing permission.
- Interactive `salloc`/`srun` on the CoreHPC login VM fails ("Insane message length"); unresolved.
- Leftover untracked files in the checkout, and the untracked status of `pxr/artifacts/pxr004_weights/chemeleon_mp.pt` (the CheMeleon weights file), are undecided.
- This log, its supporting record and the `notes/Decisions.md` edits (D022 Amendment 1, D023) are uncommitted at closing, as are the section 4 code edits.
