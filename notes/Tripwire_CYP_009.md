# Tripwire-CYP-009 Session Log

**Date:** 2026-11-21 (opened); updated 2026-11-23  
**Status:** Open. D022 (as completed by Amendment 1) is implemented locally, the package is deployed to CoreHPC, the qualification smoke has passed (job 3202933) and the approved one-off batch timing has run (job 3203300). No production screening, pIC50 production fit or cell array has been executed. No session close or advance has been declared.  
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
| Serial 150-cell extrapolation | 19,444 s (about 5.4 GPU-hours); arithmetic on one cell |

- The cell's macro ST-RAE (0.7544) comes from a 3-epoch screening checkpoint and is not a result.
- The logged cell fit reached epoch 25 of a 50-epoch maximum (early stopping, patience 20), so cell time depends on when stopping occurs. By arithmetic on the shown epoch count, per-epoch cost excluding setup is roughly 4.6 s, so a full 50 epochs would be roughly 4 minutes per cell. This is an upper-bound estimate, not measured.
- Comparison: the D021 descriptor timing (`notes/Tripwire_CYP_008.md`) was 4.90 s/epoch screening and 147.6 s for cell 0.

## 8. Claim boundary

Implementation and a 2-epoch qualification smoke only. No production screening, pIC50, cell-array or performance result exists for CheMeleon; the only other output is the timing measurement (section 7), whose ST-RAE is from a 3-epoch checkpoint and is not a result. No held-out evaluation is authorized and no protected internal test or challenge data were used.

## 9. Stopping point and next step

Stopping point: smoke v2 passed; screening manifests were generated locally from `d022_pins.json` and the smoke result hash (`d022_launch/manifests`: `t121_d022_c_s1`..`s3` and three cells_env files). The timing run has completed (section 7). Proposal only, nothing approved: validate-only run of the launcher on CoreHPC; requested wall times from the measurement (screening about 7 min measured, proposed cap 30 min; cells about 2.2 min measured and about 4 min worst case, proposed per-task cap 30 min by command-line override of the script's 12 h); then explicit approval for the three 50-epoch screening runs and the three 150-cell arrays. Session close or advance has not been declared.
