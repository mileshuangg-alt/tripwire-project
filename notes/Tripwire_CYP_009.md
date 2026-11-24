# Tripwire-CYP-009 Session Log

**Date:** 2026-11-21 (opened); closed 2026-11-23; reopened 2026-11-23  
**Status (current):** Reopened 2026-11-23 for a read-only diagnostic analysis (section 12). D022 and D023 remain frozen and unchanged. The status below is the earlier closed status, kept as history.  
**Status (history, at close):** Closed on declaration, 2026-11-23. D022 (as completed by Amendment 1) was implemented locally, deployed to CoreHPC, qualified (smoke job 3202933; approved one-off batch timing job 3203300), run through three 50-epoch screening runs (jobs 3203564-3203566) and three 150-cell arrays (jobs 3207090-3207092), and aggregated (section 9): CheMeleon is not preferred under D022 Amendment 1 item 5. The results are frozen by D023 (section 10). Next session: Tripwire-CYP-010; its scope and direction have not been stated.  
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

## 12. Reopened scope: read-only error diagnostic (2026-11-23)

The session was reopened after its close. The reopening does not change D022 or D023, which remain frozen, and no decision entry is added.

**Scope.** A read-only diagnostic analysis of the CYP2D6 and CYP1A2 errors, with CYP2C9 and CYP3A4 as reference context, for the selected updated-T121 reference arm R from D022, using existing development artifacts only.

**Constraints.**
- No training, no inference, no SLURM submission, no loss changes, no protected-test access.
- No reconstruction of missing predictions and no checkpoint loading to generate them.
- No modification of existing artifacts or decision records.
- All commands are run by the user.
- Findings are descriptive: repeated evaluations are not independent, and no significance claims are made.
- No penalties, new training or other experiment are proposed at this stage.
- Any descriptive grouping is exploratory and is not an acceptance rule.

**Requested output.** A findings report that distinguishes systematic over- or underprediction from a training-to-validation generalization gap, and does not declare either established unless the artifacts support it.

**Implementation.** `cyp/scripts/t121_error_diagnostic.py` (uncommitted; SHA-256 in the supporting record) was written within the bounded approval (implement and `--selftest` only; the user runs every command). It reads only the 150 saved arm R replicate-1 validation-prediction CSVs, `cyp/artifacts/cyp_baseline/cluster_assignments.csv`, `cyp/artifacts/cyp_007/split_manifests`, and optionally the hash list and reproduction JSONs, and writes only to a new directory. It analyses the saved post-heuristic predictions as scored. The scorer's scaling-heuristic branch (`cyp002_validation.py` lines 195-218) is not recorded in any saved artifact; code-path analysis points to the keep-raw branch, which is an inference and not a record, so pre-heuristic scale provenance is unresolved. The self-test and three-cell smoke were run by the user, but their output was not reported; only the full run's output was.

**Full run (150 cells).** Source: console output from the full run and the output tables in `analysis/cyp_009_error_diagnostic_20261124/full_150/` read locally (output-file hashes not computed). Integrity: the range-check anomaly flags were 0 of 600 (flags only; they cannot prove the scale branch), and recomputed versus recorded cell ST-RAE differed by at most 2.858e-09 over 600 comparisons (recomputed cell-mean ST-RAE 0.7964 / 0.5866 / 0.8745 / 0.4502 for 1A2 / 2C9 / 2D6 / 3A4, matching the session 007 pooled values).

| Quantity | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|
| Rows / unique molecules / unique clusters | 18,015 / 1,201 / 1,197 | 16,395 / 1,093 / 1,093 | 19,035 / 1,269 / 1,264 | 29,790 / 1,986 / 1,940 |
| Mean signed error (pred - measured) | +0.194 | +0.097 | +0.114 | +0.155 |
| Rows overpredicted | 55.8% | 55.7% | 55.8% | 57.4% |
| Median of molecule-mean signed error | +0.089 | +0.078 | +0.083 | +0.109 |
| Molecule-mean signed error, 5th / 95th pct | -1.055 / +1.799 | -0.734 / +1.039 | -1.138 / +2.067 | -0.873 / +1.327 |
| SD of molecule-mean signed error | 0.867 | 0.559 | 0.827 | 0.657 |
| Molecules with abs mean signed error > 1 | 20.7% | 7.9% | 14.9% | 13.8% |
| Row mean absolute error | 0.644 | 0.443 | 0.576 | 0.531 |
| Mean of abs molecule-mean signed error | 0.632 | 0.431 | 0.566 | 0.511 |
| Prediction SD across seeds (within repetition) | 0.114 | 0.107 | 0.103 | 0.147 |
| Prediction SD across repetitions of seed mean | 0.096 | 0.083 | 0.082 | 0.117 |
| Row mean soft (band-aware) error | 0.386 | 0.180 | 0.364 | 0.237 |
| Rows inside the CI band | 21.2% | 42.5% | 20.3% | 36.1% |
| Molecules with zero soft error on all 15 predictions | 1.8% | 11.3% | 1.8% | 11.0% |
| Clusters with one validation molecule | 99.75% | 100% | 99.68% | 98.66% |
| Top 10% of molecules: share of soft / abs error | 35.6 / 32.6% | 43.7 / 28.7% | 36.5 / 37.1% | 39.2 / 28.1% |
| Top 1% of molecules: share of soft / abs error | 5.3 / 4.9% | 8.1 / 4.4% | 5.6 / 5.0% | 7.0 / 4.3% |
| Molecules holding 50% of abs error | 19.2% | 22.8% | 17.1% | 23.6% |
| Molecules holding 50% of soft error | 16.7% | 12.4% | 16.3% | 15.8% |

Findings (descriptive):
- Every molecule has exactly 15 rows (5 repetitions x 3 seeds) and every repetition covers all unique molecules, so rows overstate independent information 15-fold; signed means are identical at row and molecule level because the design is balanced.
- The positive mean signed error is carried by a right tail of strongly overpredicted molecules over a small median shift (about +0.08 to +0.11).
- About 96-98% of mean absolute error is reproducible per molecule across seeds and repetitions; prediction variation across seeds and repetitions is small relative to error.
- Clusters are near-singletons (98.66-100% hold one validation molecule), so cluster-level concentration equals molecule-level concentration and cluster-level hypotheses are not testable with these assignments. Soft error is zero-inflated by the band, so its concentration is not comparable to a uniform benchmark.
- Variation by repetition, seed and fold (descriptive; cells share partitions and compounds and are not independent): mean signed error by repetition 0.164-0.203 (1A2) and 0.108-0.121 (2D6); by seed 0.184-0.201 (1A2) and 0.113-0.118 (2D6); by fold 0.155-0.231 (1A2), 0.066-0.191 (2D6), 0.070-0.135 (2C9) and 0.120-0.209 (3A4). Folds vary more than seeds or repetitions, and all fold-level means are positive. Per-cell ST-RAE (150 cells, 100-220 labeled rows each) ranges 0.626-0.946 (SD 0.063) for 1A2, 0.729-1.013 (SD 0.056) for 2D6, 0.389-0.797 for 2C9 and 0.368-0.567 for 3A4. Per-cell mean signed error ranges -0.024 to +0.433 (1A2) and -0.113 to +0.330 (2D6), so some cells have net negative bias.

**Not examined or unresolved.**
- Training-side evidence per enzyme is absent (Lightning logger off; only the aggregate weighted standardized training loss was ever on screen, never saved). Per-epoch validation history exists only on CoreHPC in each cell's `provenance.json` and was not loaded. A generalization gap or overfitting is therefore neither established nor ruled out, and underfitting is not established.
- Potency dependence was not recomputed; the session 007 pooled label-quartile tables remain the context (lowest-quartile overprediction, highest-quartile underprediction; SD(pred)/SD(label) 0.42 for 1A2 and 0.38 for 2D6 versus 0.64 for 2C9 and 0.71 for 3A4), binned on the measured value and so partly built in.
- How the lowest pIC50 values (about 1.9-2.1) were derived is not documented in the repository; the CYP screening log2fc sign and normalization are not established in the repository.
- Saved predictions exist only for R replicate 1 (r2, r3 and the CheMeleon runs have none).
- The near-singleton cluster granularity limits chemical-cluster conclusions; a coarser grouping would require a decision.
- Scale-branch provenance remains unresolved (see Implementation).

**Status.** Claim boundary: descriptive only; no significance claim; no diagnosis of overfitting; no penalty, training or experiment proposed. D022 and D023 remain frozen; no decision entry is added, and no next step is decided (an explicit decision entry is required for any).

**Follow-on question (2026-11-24): screening response versus overprediction.** Scope: check whether compounds with weaker screening response are the ones receiving overly potent predictions, as the check that informs whether a screening-informed prediction penalty is worth designing; CYP1A2 and CYP2D6 first. Analysis only: no training, inference, checkpoint loading, protected-test access, loss or scorer changes, or commits; the user runs every command. Any future penalty must use only information permitted inside its training fold; this analysis does not authorize selecting or fitting a rule on validation labels. Results are in the block below.

- Screening data facts (read from `actfound_screen_loader_20261117.tar.gz`: `inputs/screen_full.csv`, `inputs/adjudication.tsv`, `qualification/actfound/screen_loader_20261117/contract.json`; contract file hashes match): 45,428 rows, one per (molecule, enzyme), no duplicated keys, no empty `log2fc_estimate`, a single `concentration_M` value (4.95049505e-05 M). Adjudication has 13,097 rows (no_identity_link 8,182; full_key_match 4,910; first_block_near_match_only 5); identity-linked rows are tiered train 3,434, validation 736, internal_test 736, challenge_blinded_test 4. The join key `Molecule_Name` (OCNT-...) is shared with the training CSV and the saved predictions.
- The frozen screening loader (`cyp/scripts/actfound_model_side.py`, identity block) skips every identity-linked class, and the 007 handoff (line 88) records that the screening stage contains no development, test or challenge compounds. The screening values of the validation compounds were therefore not used to train the screening stage.
- Not defined in the repository: what `log2fc` is a fold change of, its sign, normalization and assay concentration (a gap already noted in the 008 handoff and log). Public documentation (fetched 2026-11-24; web sources, not repository facts): the OpenADMET dataset card does not define log2fc meaning, sign, concentration or censoring; the OpenADMET challenge announcement states that a negative log-fold change indicates the compound inhibits the CYP, describes fluorescence readouts for CYP3A4, CYP1A2 and CYP2C9 and parent depletion by mass spectrometry for CYP2D6, and says pIC50 values come from 12-point dose-response fits while the single-concentration screen was used for hit selection. It does not state the assay concentration, the fold-change numerator and denominator, or how weak, inactive or censored compounds appear in the pIC50 columns.
- Open gaps: exact log2fc definition and base; assay concentration (the file column is a data fact only); direction for CYP2D6, documented only through the general statement despite the depletion readout, so its interpretation is held pending the script's coherence check (Spearman between log2fc and measured pIC50; an inference, not a definition); representation of weak compounds in pIC50. Weak screening response is not equated with confirmed inactivity.
- Implementation (written; full run reported below): `cyp/scripts/t121_screening_error_analysis.py` (uncommitted; SHA-256 in the supporting record), reusing `t121_error_diagnostic.py` unchanged. It reads R replicate-1 saved predictions, cluster assignments, split manifests, optional SHA-256 sums, the screening tarball in place (nothing extracted) and the training CSV (SMILES check for analysed molecules only), and writes only a new directory. It verifies identity class (full_key_match or largest_fragment_match; tiers train or validation; a protected tier stops the run), SMILES agreement, duplicates, missing values and per-isoform coverage. Each molecule counts once (15 predictions = 5 repetitions x 3 seeds). Outputs: join audit; molecule-level and row-level tables; Spearman and Pearson relationships of log2fc to measured pIC50, predicted pIC50, signed, absolute and soft error, plus a rank partial correlation with signed error given predicted pIC50 (signed and absolute kept separate); equal-count quintile summaries of log2fc and of predicted pIC50 (exploratory, no cutoff); an overprediction-tail comparison (top 10% of molecule-level signed error and signed error above +1.0, both fixed in advance and exploratory; tail-versus-rest log2fc distribution and rank AUC, no p-values); per-repetition and per-fold relationships.

**Follow-on results (2026-11-24; full mode, 150 cells, CYP1A2 and CYP2D6; R replicate-1 saved predictions; each molecule counted once over its 15 predictions).** Source: console output from the full run and the output tables in `analysis/cyp_009_screening_error_20261124/full_150/` read locally (output-file hashes not computed). The self-test and smoke-run output were not reported. Tables are in the supporting record.

- Join audit: all analysed molecules joined with verified identity (CYP1A2 1,201 of 1,201; CYP2D6 1,269 of 1,269); 0 missing screening rows, duplicates, identity-unverified or missing values; SMILES identical in the screening file and training CSV; one concentration value; no warnings.
- Screening range among the labelled compounds: log2fc -2.905 to +2.703 (CYP1A2; quartiles about -1.77 / -1.555 / -0.90) and -3.273 to -0.592 (CYP2D6). Every labelled CYP2D6 compound has a negative log2fc, so its labelled set has no weak or non-responding compounds. Inference, not a record: the labelled set looks like screening hits, consistent with the public statement that hits were followed up with dose-response curves; the selection mechanism for the pIC50 labels is not established in the repository.
- Direction (inference, not a definition): Spearman(log2fc, measured pIC50) is -0.871 (CYP1A2) and -0.825 (CYP2D6), per-partition ranges -0.936..-0.797 and -0.881..-0.771. More negative log2fc goes with higher measured pIC50, matching the public statement that a negative log-fold change indicates inhibition. With predicted pIC50 the association is weaker (-0.486, -0.396).
- Signed error (predicted - measured) versus log2fc: Spearman +0.758 (CYP1A2) and +0.688 (CYP2D6); rank partial correlation given predicted pIC50 +0.811 and +0.748. Within equal-count predicted-pIC50 quintiles (exploratory) it stays +0.594..+0.927 and +0.711..+0.812. Across the 50 repetition-by-fold partitions (100-146 molecules each; shared compounds, not independent) it ranges +0.624..+0.832 and +0.528..+0.772; no partition reverses sign.
- Absolute error (kept separate): Spearman with log2fc +0.326 (CYP1A2) and -0.086 (CYP2D6); per partition +0.121..+0.514 and -0.274..+0.138. Mean absolute error by log2fc quintile is U- or J-shaped (CYP1A2 0.622, 0.396, 0.320, 0.432, 1.451; CYP2D6 0.757, 0.381, 0.432, 0.603, 0.707).
- Log2fc quintiles (exploratory; strongest to weakest response): weaker responders are overpredicted and the strongest responders underpredicted. Mean signed error runs from -0.519 to +1.438 (CYP1A2; share overpredicted 16.2% to 98.8%) and from -0.684 to +0.690 (CYP2D6; 11.8% to 93.3%). In the weakest CYP1A2 quintile 72.1% have signed error above +1.0 (measured 3.35 versus predicted 4.79). Predicted pIC50 varies much less across quintiles than measured (shrinkage toward the mean).
- Overprediction tail (top 10% of molecule-level signed error, and signed error above +1.0; definitions fixed in advance, exploratory): tail median log2fc -0.195 versus -1.601 (CYP1A2) and -1.068 versus -1.892 (CYP2D6). Rank AUC that a tail compound has a weaker-response log2fc than a non-tail compound is 0.980 (CYP1A2) and 0.787 (CYP2D6) for the top-10% tail, and 0.976 and 0.788 for signed error above +1.0. Tail compounds have mean measured pIC50 about 2.8-3.2 (CYP1A2) and 2.7-2.9 (CYP2D6) against predictions of about 4.8-4.9.

**Interpretation limits.** The signed error contains the measured value, which itself tracks log2fc, so a strong association is expected from compressed predictions. It shows that screening carries information about the measured value that the predictions use less; it does not identify a cause, does not show a penalty is the right mechanism, and does not predict any performance gain. The result concerns relatively weaker hits, not confirmed inactives; weak screening response is not confirmed inactivity. Descriptive only; no significance claim; saved-prediction scale provenance unresolved; no training-side evidence.

**Still unresolved.** How the very low pIC50 values are derived (extrapolated fits possible); the exact log2fc numerator, denominator and assay concentration; and whether screening values would exist for blinded test compounds (only 4 challenge-blinded compounds are identity-linked to the screening file, and the current pipeline's screening stage excluded identity-linked compounds). No next step is decided (an explicit decision entry is required for any), and D022 and D023 remain frozen. This analysis does not authorize selecting or fitting a penalty or rule on validation labels; any future penalty must use only information permitted inside its training fold.

**Design definitions recorded as D024 (2026-11-24).** The three-arm screening-informed penalty comparison (updated T121 unchanged; Q3-cutoff penalty; response-rank penalty) is approved as design definitions only, in `notes/Decisions.md` D024, with the follow-on analysis above as motivating context only. D024 fixes the arms, the matched r1 checkpoint (selected epoch 16) and 150 cells, the primary metric (macro ST-RAE per cell, development evaluation only), per-fold and per-enzyme training-only weights, the penalty form and lambda = 0 identity, handling of missing, duplicate and degenerate cases, the both-tail reporting definition (with its non-independence caveat), per-epoch loss-component logging, and the opt-in implementation approach. It authorizes no implementation, training, inference, checkpoint loading, benchmarking, GPU job, SLURM submission or commit; any run needs separate approval. D021, D022 and D023 are unchanged. Explicitly not decided: numeric lambda and its selection rule, the adoption criterion, extension beyond r1 to all three screening replicates, any lambda = 0 training fit or GPU measurement, the mapping of "condition" to screening-file fields for duplicate collapsing, and the unaudited CYP2C9 and CYP3A4 coverage and direction check. A fresh training reproduction of a saved cell has not been demonstrated (saved baseline scoring was reproduced).

**D024 Amendment 1 (2026-11-24).** Further rulings are recorded as Amendment 1 to D024 in `notes/Decisions.md`, with the same status (design and evaluation definitions only; no implementation, training, inference, checkpoint loading, benchmarking, GPU job, SLURM submission or commit authorized). It closes the items listed above as not decided, except the CYP2C9/CYP3A4 coverage check: lambda = 1.0 fixed for both penalty arms and all four enzymes (a first test, not a calibrated strength; a negative result rejects only this strength); a strict adoption rule (lower mean macro ST-RAE over the matched 150 cells, no worse mean in any of the 12 enzyme x seed comparisons, lower MAE on each enzyme's over-prediction tail, no higher MAE on its under-prediction tail; lower macro ST-RAE decides between two passing arms, an exact tie goes to the cutoff arm; if neither passes the unchanged updated T121 is kept); r1 only for the primary comparison, with r2/r3 confirmation of a provisional winner a separately authorized stage; no separate lambda = 0 training or loss run, replaced by deterministic loss-and-gradient equivalence tests at lambda = 0 plus the required smoke checks; condition key (Molecule_Name, enzyme) with the D024 duplicate policy as safeguard; per-enzyme tails fixed once from baseline mean signed error. Still not decided: how per-compound predictions for the penalty arms are obtained for the tail-MAE criterion (validation-only inference from best checkpoints, which needs checkpoint loading and its own approval, or capture inside training; only the unchanged arm has saved per-compound predictions); operational readings of the rule (equality, the 12 comparisons, tail strictness, cohort size rounding) awaiting confirmation; runtime measurement and wall-time sizing (no timing run authorized; interactive GPU unavailable on the login VM). Caveats: with no lambda = 0 trajectory reproduction, differences from the saved unchanged arm can include non-penalty sources (code version, GPU nondeterminism); tail cohorts come from the baseline's own validation errors, so changes on them are not independent evidence; all evidence is development evidence on inspected partitions.

## 13. Stopping point and unresolved items

Stopping point at the earlier close (superseded as current status by section 12; the session was reopened 2026-11-23): session closed 2026-11-23 on declaration, after the D022 result was computed (section 9) and frozen (D023). No next step is decided; Session CYP-010 follows, its scope and direction not yet stated.

Unresolved at close:
- The batch-timing run (section 7) was a one-off deviation, not a standing permission.
- Interactive `salloc`/`srun` on the CoreHPC login VM fails ("Insane message length"); unresolved.
- Leftover untracked files in the checkout, and the untracked status of `pxr/artifacts/pxr004_weights/chemeleon_mp.pt` (the CheMeleon weights file), are undecided.
- This log, its supporting record and the `notes/Decisions.md` edits (D022 Amendment 1, D023) are uncommitted at closing, as are the section 4 code edits and the section 12 diagnostic script and output directory.
