# Tripwire-CYP-007 Session Log

**Date:** 2026-11-18 to 2026-11-19  
**Status:** Open. Single-cell reproduction (§4) and the full 150-cell validation-only inference (§5) are completed and reproduced. Outputs are transferred, hash-verified and described in a descriptive development-validation diagnostic (§6); no bootstrap, training or held-out evaluation. On 2026-11-19 the direction for Session 008 was stated (§7): scope a descriptor comparison and return a plan for approval. Nothing is approved to implement or run; nothing further is authorized to launch until Miles approves the specific execution step. A handoff to Session 008 is being written; no session close or advance has been declared.

Naming note: "CYP-007" is also an existing experiment name (`cyp/artifacts/cyp_007/`, `cyp007_*` scripts, D007). This log is the seventh session of the CYP thread.

## 1. Session purpose and starting state

Session 007 begins from the closed Session 006 (`notes/Tripwire_CYP_006.md`; handoff `handoff_Tripwire-CYP-007_20261118.md`, which transfers state and is not decision authority). Starting state, from those records: updated-T121 O150 = 0.6769540372619428 (150 cells, job array 3105880, checkpoints present on CoreHPC and SHA-256 matched); non-final ActFound-CYP O150 = 0.8513335058029842 (job 3118356; `comparison_is_final` false); no challenger selected; per-cell validation predictions were never saved. Governing decisions: D013 with Amendments 1-4, D015, D016.

## 2. Session direction (stated by Miles, 2026-11-18; scope: Session 007 only)

- **Direction:** diagnose updated T121 through development-validation inference, meaning per-cell validation predictions from the existing 150 updated-T121 checkpoints. The session does not launch another Chemprop training update and does not rescue ActFound.
- **Candidate:** on the current development-validation results updated T121 is the candidate to carry forward. A further Chemprop change stays open only if the diagnostics support a specific hypothesis and an experiment that would resolve it.
- **PXR:** continues separately in parallel, in its own thread.
- **Checkpoint gate:** the existence and SHA-256 check of the 150 checkpoints already passed and is not repeated.
- **Execution plan:**
  1. Run validation-only inference for one cell first, to verify that the original per-isoform and macro scores are reproduced and to measure runtime. Extrapolate to 150 cells against the proposed wall time before any full submission. The original plan was to run this on the established interactive resource and not through Slurm; see §3 for the deviation.
  2. The full pass saves compound, isoform, repetition, fold and seed identifiers with labels, predictions and CI bounds, in a new output directory.
  3. Metrics (plain MAE, signed bias, soft MAE, ST-RAE) are computed from individual errors, with no averaging of predictions across seeds or repetitions.
  4. Initial analysis is descriptive: error against training-set similarity and against CI width. Bootstrap analysis and new training proposals are deferred.
- **ActFound:** its saved development predictions may support the same diagnostics once transferred. Interpretation constraints stated: ActFound beats the listed 5-NN baseline on all four isoforms; the 50 partitions are not independent observations; a small gain over the mean predictor does not establish an information ceiling; ActFound's poor result does not identify a Chemprop change; ActFound production settings must be confirmed before runner defaults are used to explain its performance.
- **Boundaries:** the D012/comparator provenance discrepancy stays separate and unresolved. The protected internal test and challenge sets are not accessed. Final test evaluation comes only after model choices and evaluation settings are locked, not in this session. No inference, training or job is launched until Miles approves the specific execution step.

These are session-scope directions recorded here; no entry has been made in `notes/Decisions.md`.

## 3. Interactive GPU resource and batch-timing deviation

The interactive GPU invocation is recorded (an earlier statement in this log that it was UNKNOWN was wrong): `srun --partition=gpu --gres=gpu:nvidia_l40s:1 --cpus-per-task=8 --mem=96G --time=2:00:00 --pty bash`, from `notes/archive/Tripwire_CYP_005_archive_20261116.md` entry 38 (job 3076227, used for the D013 benchmark).

On 2026-11-18 this invocation was tried for this session. srun jobs 3119727 and 3119728 queued, then srun failed with "Insane message length" / "Srun communication socket apparently being written to by something other than Slurm", the same failure recorded in `notes/Tripwire_PXR_002.md` §5a. Job 3119728 was cancelled.

**Session-scope decision (Miles, 2026-11-18; scope: this one-cell timing/reproduction check only):** because interactive srun failed, the one-cell timing benchmark is run as a Slurm batch job. This is a recorded deviation from the interactive-benchmark practice (precedent: the PXR-002 runtime estimate, job 3106209). It is not a standing permission to benchmark other work through Slurm. No entry has been made in `notes/Decisions.md`.

**Session-scope approval for the full pass (Miles, 2026-11-18; scope: this pass only):** the validation-only inference for all 150 updated-T121 cells is run as one Slurm batch job with a 30 minute request, under the same deviation (interactive srun failed). Extrapolation presented before approval: 150 x 2.41 s (cell 0, cold) is about 6 minutes plus startup, so 6 to 10 minutes expected against the 30 minute request. This does not extend to other jobs.

## 4. Validation-inference script and single-cell reproduction

**Code (local, uncommitted; written by Claude Code):**

| File | SHA-256 |
|---|---|
| `cyp/scripts/t121_validation_inference.py` | bb73bde01d0d251c82121568c35c8d625999e2b75e5e57035dd9658394e35e5c |
| `cyp/tests/test_t121_validation_inference.py` | 825db192c8a27447cdb5d2d7b870cc0ba98e671377b3d663ec643dc7746d8b81 |
| `cyp/scripts/t121_validation_inference_timing.slurm` | d8f974961da58c80112db72eb5960b0f8f94d595ddb69cc8b3ba8aa9177c80e1 |

**Method:** rebuild each cell's fold via `route_d007_fold` (protected-test guard); build the same prefit state; load `fit/best.ckpt` strictly, checked against the `result.json` checkpoint SHA-256; run one validation pass through the model's own validation step; capture the arrays the frozen scorer receives; compare with `result.json`; write per-compound rows to a new, never-reused directory; stop on mismatch.

**Local test:** 10 passed on CPU in conda env `tripwire_cyp002` (Chemprop 2.3.1, Lightning 2.6.6) with a tiny synthetic fold. Metrics match the frozen `st_rae_metric` to 12 decimals; CPU reproduction difference 0.0. This is a synthetic-fold test, not evidence about the real checkpoints.

**Batch job 3119764** (node ggpu1-07; source: contents of `timing_cell_000/reproduction_cell_000.json` as pasted by Miles). Run directory on CoreHPC: `/scratch/user/mhuang8/tripwire_staging/t121_valinf_20261118/timing_cell_000`; package root `/scratch/user/mhuang8/tripwire_staging/t121_production_chain_20261117`. Package files for the modules used match the local hashes in its `SHA256SUMS.txt`. The CoreHPC `sha256sum -c` output for the package and the dataset/split hash-check outputs were not pasted back, so those two checks are unconfirmed in the record.

Cell 0 (repetition 1, fold 1, seed 20261101; 418 validation molecules, 560 labeled prediction rows): status **reproduced**.

| Quantity | Value |
|---|---|
| Frozen-scorer path difference from original `result.json` scores (macro and all four isoforms) | 0.0 |
| Max difference recomputed from saved rows | 3.38e-10 (macro 1.19e-10), attributed to float32 label reconstruction in the scorer path |
| Macro ST-RAE (original and rescored) | 0.6703237217646288 |
| 1A2 | 0.7326445152952443 |
| 2C9 | 0.5062979410879374 |
| 2D6 | 0.9807246886621953 |
| 3A4 | 0.4616277420131385 |

The reproduction tolerance of 1e-6 stays provisional; observed differences are far below it.

Timing: build_and_load 1.65 s, validation pass 0.67 s, cell_total 2.41 s. Process wall time (imports and CUDA startup) and Slurm elapsed time were not reported and remain UNKNOWN. The result covers one cell only; reproduction for other cells is not yet shown.

## 5. Full 150-cell validation-only inference

**Batch job 3120318** (node ggpu1-11; source: sacct and job output pasted by Miles from CoreHPC). Same inference script (SHA-256 bb73bde01d0d251c82121568c35c8d625999e2b75e5e57035dd9658394e35e5c, as in §4); batch wrapper `t121_validation_inference_timing.slurm` with job name and output path overridden at submit. Output directory on CoreHPC: `/scratch/user/mhuang8/tripwire_staging/t121_valinf_20261118/full_cells_000_149`.

| Quantity | Value |
|---|---|
| State / ExitCode | COMPLETED, 0:0 |
| Slurm elapsed | 00:04:33 |
| process_wall_seconds | 273 |
| Loop wall | 272.04 s (mean 1.708 s per cell; about 1.7 to 1.8 s on the last cells) |
| Cells / reproduced | 150 / 150 (tolerance 1e-6, provisional) |
| Max abs difference from original `result.json` scores, frozen-scorer path | 0.0 |
| Max difference recomputed from saved per-compound rows | 2.86e-9 (float32 label round-off) |
| Output files | 302 (150 predictions CSVs, 150 reproduction records, `summary.json`, `completion.json`); no `STOPPED_ON_MISMATCH.json` |

The run was faster than the 6 to 10 minute extrapolation. Per-compound predictions carry repetition, fold, seed, cell, isoform, Molecule_Name, label, prediction and CI bounds. The process wall time of cell-0 job 3119764 remains UNKNOWN.

The CoreHPC package `sha256sum -c` and dataset/split hash-check outputs (§4) remain unreported. Transfer and analysis are in §6.

## 6. Transfer and descriptive diagnostics of updated T121 and ActFound-CYP (development validation only)

### 6.1 Transfer and verification

Miles ran the transfer; the hashes were pasted by Miles and confirmed locally by Claude Code. All files are untracked at the repo root of this checkout.

| Artifact | SHA-256 | Check |
|---|---|---|
| `full_cells_000_149.tar.gz` | 59cc0edd3b0dd5a53650aee5e7180292470501b5ec2eb49411f2393f78243e2f | CoreHPC and Mac match |
| `full_cells_000_149_SHA256SUMS.txt` | 11a44fd7449642549d4e7b9a5376afa37a98adf3a61ba1d9f1a99e89f8c7b18d | match |
| `cyp_adaptation_predictions.json` (ActFound, job 3118356 run) | 1a68cfdfd79bd777455fab1cc3e979fa892cc7c37f8e2bf37e38d62258de2176 | equals the hash recorded in the ActFound result |

The tarball was extracted to `t121_valinf_full_20261118/`; all 302 files verified OK against the manifest.

### 6.2 Diagnostic script and cross-checks

`cyp/scripts/t121_validation_diagnostics.py` (Claude Code, uncommitted), SHA-256 186098585f2e9d78b4cf2fbfad28c9e885bc484577f269b52a8e4ead99fcb34c. Outputs: `t121_valinf_analysis_20261118/`. An earlier run into a same-named directory was deleted and rerun after two tables were added; the earlier output was scratch and is not evidence. The script asserts that no protected internal-test molecule appears; only molecules named in the frozen development folds are used. Not run: bootstrap, new training, held-out or protected data.

- Labels and CI bounds of the T121 rows and ActFound rows were checked identical on 27,745 compound rows.
- Macro ST-RAE recomputed from saved rows: 0.676954 (updated T121) and 0.851334 (ActFound), matching the recorded O150 values 0.6769540372619428 and 0.8513335058029842.
- Unverified: the package `sha256sum -c` output and the CoreHPC dataset/split hash-check output were never reported, so equality of the local dataset file with the CoreHPC one is supported only by the label and score reproduction.

### 6.3 Descriptive results

Descriptive only, not causal. Partitions share compounds and are not independent; ActFound seed slots are replicates (one prediction set per partition); T121 has 150 cells.

**Per-isoform scores (ActFound / updated T121):**

| Isoform | ST-RAE | Plain MAE | Signed bias, T121 (pred - label) | Signed bias, ActFound |
|---|---|---|---|---|
| CYP1A2 | 0.9425 / 0.7964 | 0.715 / 0.645 | +0.194 | about 0 (abs < 0.004) |
| CYP2C9 | 0.7946 / 0.5866 | 0.508 / 0.442 | +0.097 | about 0 |
| CYP2D6 | 1.0028 / 0.8745 | 0.632 / 0.577 | +0.114 | about 0 |
| CYP3A4 | 0.6654 / 0.4502 | 0.672 / 0.531 | +0.155 | about 0 |

Updated T121 macro ST-RAE by seed: 20261101 0.6845, 20261102 0.6745, 20261103 0.6719. Within-compound SD of T121 predictions across seeds is 0.084 to 0.120 pIC50, against pooled plain MAE of 0.44 to 0.64.

**Constant-bias removal (post hoc, uses validation labels, diagnostic only):** removing T121's pooled constant bias changes ST-RAE 0.7964 to 0.8142 (1A2), 0.5866 to 0.6076 (2C9), 0.8745 to 0.8727 (2D6), 0.4502 to 0.4807 (3A4); worse in 3 of 4 isoforms.

**Error by label quartile (per isoform, pooled compound set):** both models shrink toward the mean.

| Isoform | Lowest-quartile bias T121 / ActFound | Highest-quartile bias T121 / ActFound | Highest-quartile plain MAE T121 / ActFound | SD(pred)/SD(label) T121 / ActFound | Share of soft error, highest / lowest quartile (T121) |
|---|---|---|---|---|---|
| 1A2 | +1.27 / +1.29 | -0.62 / -0.99 | 0.654 / 0.987 | 0.42 / 0.26 | 0.36 / 0.41 |
| 2C9 | +0.66 / +0.78 | -0.42 / -0.72 | 0.477 / 0.728 | 0.64 / 0.42 | 0.47 / 0.26 |
| 2D6 | +0.98 / +1.02 | -0.71 / -1.05 | 0.748 / 1.045 | 0.38 / 0.18 | 0.44 / 0.34 |
| 3A4 | +0.82 / +1.02 | -0.39 / -0.75 | 0.452 / 0.801 | 0.71 / 0.56 | 0.40 / 0.15 |

In the second-lowest label quartile ActFound has lower plain MAE than T121 for all four isoforms.

**CI width:** for both models, narrow-CI compounds are under-predicted and wide-CI compounds over-predicted, the same pattern as label level. CI width appears to track label level here; this is an association and was not tested further.

**Training-set similarity** (max Morgan r2 2048 Tanimoto to isoform-labeled training compounds of that partition): no consistent monotone relation with error (Spearman |rho| <= 0.10 for absolute and soft error). Similarity ranges are narrow because folds are cluster splits, and similarity is confounded with CI width for CYP1A2 (rho 0.18).

**Head to head** (T121 soft error averaged over its three seeds' errors, versus ActFound): T121 is lower in 45 to 54 percent of compound rows (ties 5 to 17 percent), with mean advantage 0.054 to 0.113 pIC50 in T121's favour. Its advantage is therefore concentrated in fewer large errors rather than in most rows.

### 6.4 Interpretation limits

Descriptive; no mechanism established; no decision about a Chemprop change. The earlier hypothesis that chemistry coverage limits CYP2D6 is neither confirmed nor refuted by the within-isoform similarity result. Nothing here is a decision and no entry has been made in `notes/Decisions.md`.

## 7. Direction for Session 008 (2026-11-19)

Directions stated by Miles (scope: direction only; no entry in `notes/Decisions.md` unless he asks)

- **Reference:** updated T121 won the D015 comparison against ActFound-CYP on the current development-validation results and stays the reference (O150 0.6769540372619428 versus non-final ActFound-CYP O150 0.8513335058029842).
- **Flagged, unresolved:** this statement does not by itself change the records, which still show `comparison_is_final` false, D013 Amendment 4 item 6 (comparator provenance, `C0_COMPARISON.json`) undecided, the D012 summary-versus-record discrepancy unresolved, and no held-out evaluation authorized. These are for Miles to decide separately.
- **Next session (Tripwire-CYP-008):** first scope a comparison of updated T121 against updated T121 plus molecule-level descriptors, preserving the rest of the recipe, and return a comparison plan to Miles for approval before any implementation or training. CheMeleon is queued after the descriptor comparison. Nothing is approved to implement or run.
- **Handoff:** `notes/Handoffs/Tripwire_CYP_007_Handoff_20261119.md` is being written. It carries a proposed descriptor-comparison plan, which is a proposal only and not an approval.

## 8. Claim boundary

Reproduction is shown for all 150 updated-T121 cells on the frozen-scorer path. The §6 diagnostics are descriptive on development validation data only. This does not select a final challenger, does not make the comparison final, does not adopt `C0_COMPARISON.json`, and does not authorize a held-out evaluation. The §7 direction approves no implementation, training or run. No protected internal test or challenge data were accessed.

## 9. Stopping point and next step

Stopping point: outputs transferred and verified (§6.1), descriptive diagnostics complete (§6.3), and Session 008 direction stated (§7). Bootstrap analysis and any new training proposal remain deferred. Next step: in Session 008, scope the updated-T121 versus updated-T121-plus-molecule-level-descriptors comparison and return the plan to Miles for approval before any implementation or training; CheMeleon follows. Session close or advance has not been declared.
