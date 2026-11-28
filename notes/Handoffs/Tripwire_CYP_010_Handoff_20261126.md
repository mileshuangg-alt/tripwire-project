# Tripwire-CYP-010 Handoff (for Tripwire-CYP-011)

Written 2026-11-26 at the close of Tripwire-CYP-010. This is a continuity reference. It is not decision authority and it authorizes no implementation, training, inference, submission, branch or edit. The receiving session should confirm the current position and the next action briefly, then wait for Miles's direction.

Source basis: this session's chat record (including tool outputs Miles pasted back) and the repository files named below. Where a statement rests only on a repository record rather than on something seen in this chat, it is marked "per record". Anything the record does not establish is marked UNKNOWN.

Access the receiving session needs: the repository checkout `/Users/mileshuang/Desktop/tripwire-project` (all local paths below are relative to it). CoreHPC paths exist only on CoreHPC (login host `chpc-ucsf-login-vm1.corehpc.ucsf.edu`, via bastion `chpc-ucsf-bastion-vm1.corehpc.ucsf.edu`); Claude cannot reach them. Miles runs every command, test, transfer and Slurm submission and pastes the output back.

---

## 1. Objective

Tripwire-CYP-010 began from `notes/Handoffs/Tripwire_CYP_009_Handoff_20261124.md`. Its purpose was the D024 three-arm comparison on the updated-T121 recipe: unchanged, Q3-cutoff penalty (`q3`) and response-rank penalty (`rank`). Both penalty arms use λ = 1.0, the r1 screening checkpoint (epoch 16), the matched 150 cells (5 repetitions × 10 folds × seeds 20261101–20261103) and all four enzymes. The primary metric is macro ST-RAE; the evaluation is development-only. The session also carried D025 (screen-head prediction exposure) and D026 (descriptive evaluation of existing screen predictions).

Session status: closed by Miles's explicit declaration, 2026-11-26. Session log: `notes/Tripwire_CYP_010.md`; supporting record: `notes/Tripwire_CYP_010_supporting.md`; project-log row added to `notes/project_log.md`.

---

## 2. Completed work (with evidence and source)

### 2.1 D024 three-arm evaluation

Outcome (source: Miles-pasted output of `t121_d024_aggregate.py evaluate`, recorded in the session log section 10):

| Arm | Mean macro ST-RAE (lower is better) |
|---|---|
| unchanged | 0.676954037 |
| q3 | 0.689167299 |
| rank | 0.672843630 |

- Adoption-rule clauses (a lower mean macro ST-RAE, b no worse enzyme × seed mean, c lower over-tail MAE, d no higher under-tail MAE): q3 fails a, b, c, d; rank passes a only (fails b, c, d). Neither passes, so **unchanged is selected** (rule outcome, as printed by the aggregator).
- Per-enzyme ST-RAE / over-tail MAE / under-tail MAE are in the session log section 10. From the printed table, rank's CYP2C9 ST-RAE is worse than unchanged (0.591742 vs 0.586615); its over-tail MAE rises only for CYP3A4; its under-tail MAE rises for CYP1A2, CYP2C9 and CYP2D6. This clause-level reading is from the table, not from the report JSON.
- CYP3A4 coverage-group report (D024 Amendment 3, descriptive only, covered 1565 / not covered 421 molecules): rank's mean cell ST-RAE is lower than unchanged in both groups (covered −0.007555, not covered −0.007351); q3 is lower in covered (−0.003433) and higher in not covered (+0.004874). Full table in the session log.
- Evaluation report: `/scratch/user/mhuang8/tripwire_staging/t121_d024_report_20261124.json` on CoreHPC, SHA-256 `7cefd735d92c61f4b18822bde0ca18afcd15aeba9ec23d2ff448b628cdf462b1` (previously missing; recovered 2026-11-26 from Miles's `sha256sum` output; report produced 2026-11-25). The report file itself has not been copied to the Mac.
- The aggregator checks that the unchanged arm reproduces its recorded score 0.6769540372619428; it completed without error.
- Post-inference integrity check (150 prediction files per arm, no `STOPPED_ON_MISMATCH.json`, 150 `reproduced` lines per arm): reported satisfied by Miles before aggregation; the output was not pasted and was not independently verified. Cells 11–15 of the q3 inference log were seen reproduced with `max_abs_diff` 0.0.

Jobs (last-reported status, source: Miles-pasted `sacct` output, 2026-11-25; not live):
- Cell 0, both arms: batch jobs 3254977 / 3254978; cell-0 integrity output was pasted and passed (arm labels, checkpoint-hash prefix equals r1, weights hashes equal the preview, epoch-log lines equal epochs, λ = 1.0). First loss components: q3 epoch 0 base 0.8726 / penalty 0.3595, epoch 35 base 0.2604 / penalty 0.0271; rank epoch 0 base 0.8544 / penalty 0.1929, epoch 34 base 0.2502 / penalty 0.0378.
- Cells 1–149: array 3255867 (`t121-d024-q3`) and array 3255868 (`t121-d024-rank`), 149 tasks each, all COMPLETED. The mapping of job id to arm rests on my expectation that the first array was q3; the `sacct` listing seen shows job names and counts, not ids.
- Validation inference (cells 0–149, tolerance 1e-06): 3273438 (q3) COMPLETED 00:04:35 and 3273439 (rank) COMPLETED 00:04:34.
- Typical baseline cell time from earlier `sacct`: about 2 min (range 1:29–2:56). Penalty-arm per-cell elapsed times were not pasted.

Locations on CoreHPC (all under `/scratch/user/mhuang8/tripwire_staging/`):
- Package: `t121_d024_package_20261124/`; arm environments: `t121_d024_env_20261124/cells_env_t121_d024_{q3,rank}.sh`; logs: `t121_d024_logs/`.
- Run roots: `t121_d024_q3_cells_run`, `t121_d024_rank_cells_run` (cells under `trials/trial_121/cells`).
- Validation-inference outputs: `t121_d024_valinf_q3_20261124`, `t121_d024_valinf_rank_20261124`.
- Unchanged baseline: cells `t121_production_chain_20261117_cells_run/trials/trial_121/cells`; per-compound predictions `t121_valinf_20261118/full_cells_000_149`; r1 selected record `t121_production_chain_20261117_screen_run/selected_screen_record.json` (SHA-256 `4da7a07de2926a62894441da3cd0183b0e6c38f87902266fba40fc3195b1a0fc`; epoch 16, checkpoint `preparation/epoch_checkpoints/epoch_016.ckpt` SHA-256 `96b75ceb4ddd629373e2356ba5cff3ae8ba58d27a4de1c48db0091646bcd5e7c`).

Local artifacts and code (committed in `99a3301c06790b885c5ca84df2e81ba309a6c94b` unless stated):
- Penalty module `cyp/scripts/t121_screening_penalty.py`; training hook in `cyp/scripts/t121_weighted_training.py`; cell runner `cyp/scripts/t121_production_cell.py` (`--penalty-config`, `--penalty-config-sha256`, refuses combination with descriptors or CheMeleon); Slurm `cyp/scripts/t121_cells.slurm` (optional `PENALTY_ARGS`).
- Frozen configs `cyp/artifacts/d024_penalty/`: `penalty_q3.json` (SHA-256 prefix ddfe1f01…af7b), `penalty_rank.json` (a20552a3…efe), `evaluation.json` (8e7aabec…c669c77), `coverage.json`. Full hashes are in the session records.
- Layout/packaging/aggregation: `cyp/scripts/t121_d024_layout.py`, `t121_d024_build_package.py`, `t121_d024_aggregate.py`; tests `cyp/tests/test_t121_screening_penalty.py`, `test_t121_d024_aggregate.py`, `test_t121_d024_layout.py` and new cases in `test_t121_production_cell.py`.
- Pinned analysis inputs: `analysis/cyp_010_tail_cohorts_20261124.json` (SHA-256 `cc336d4ab3de85a5c8da0ccfd2408b43a27e02c7b5ea2ee41b407388c0d73e7e`; cohort sizes 120/109/127/199), `analysis/cyp_010_cyp3a4_coverage_20261124.json` (`3640127b4fb5b4af135bf27fe9e31c29be6de89138ffa5525ea729fdc2342fb2`), `analysis/cyp_010_penalty_preview_20261124/preview.json` (`06c4d49c9c4ffad1e616cbaf231e5b0cb224c2cd37d58bfa755759c72b77741e`).
- Built package (untracked, deployed artifact): `t121_d024_build_20261124/t121_d024_package_20261124.tar.gz`, SHA-256 `83b61e6ae961ad62115f0e2335622bb0c45ee07463a5a5171e7a4b977ae4edf3`, 115 files.
- Commit `1b22c9b` (35 files, not pushed) carries the later D025/D026 work and the closed records. Neither commit is pushed.

Implementation design facts a new penalty arm would build on (source: code and design as written this session):
- Total loss = base loss + λ·P. P = Σ w·relu(standardized prediction − standardized upper CI bound)² over observed training cells, divided by the same observed-cell count as the base loss. CI-width weights do not enter P.
- Weights `w` are per (repetition, fold, enzyme) from labelled training-fold molecules with verified screening values only; the rank arm uses (average rank − 1)/(n − 1) normalized to mean 1; missing or unverified molecules get zero weight but keep their ordinary loss and the denominator.
- Degenerate cases (n < 2, all-equal responses, nothing above Q3 for the cutoff arm, no labelled molecules) switch the penalty off with a logged reason. Note that this differs from the "stop and report" semantics requested for the new arms (see 5).
- The penalty table is a plain attribute (`screening_penalty`), not a module or buffer, so state_dict and checkpoints are unchanged; the frozen executor `cyp007_cell_fit.py` (SHA-256 prefix 625a64fa…1a00b) is untouched; λ = 0 equivalence was tested against the pre-edit module at commit `9bcf355` (file SHA-256 prefix 50414196…c49e).
- Weights, population and an epoch log are written in each cell directory outside `fit/` (`penalty_weights.csv`, `penalty_population.json`, `penalty_epochs.jsonl`).

### 2.2 D025: screen-head prediction exposure (per record: `notes/Tripwire_CYP_010.md` section 11)

D025 and Amendments 1–3 are recorded. Per record: screen-head CPU loading smoke passed for the CYP r1 checkpoint and the three PXR stage-1 checkpoints; `cyp/scripts/screen_predict.py` ran on CoreHPC on the synthetic and 16-molecule reference inputs and on the 208-molecule built input set, 208 of 208 rows ok; local tests were recorded as 38 of 38 at one point. Scripts: `cyp/scripts/screen_predict.py`, `screen_predict_reference.py`, `screen_head_smoke_cyp.py`, `build_screen_input_set.py`, `pxr/scripts/screen_head_smoke_pxr.py`, with tests under `cyp/tests/` and `pxr/tests/`. The predictions were not evaluated under D025 itself.

### 2.3 D026: descriptive evaluation of existing screen predictions on 197 molecules (complete)

- Scope (D026, Amendments 1 and 2 in `notes/Decisions.md`): 208-row input minus the 11 PXR aux-only molecules whose group touches official-test structures (`aux_only_group_contains_test_structure`), leaving 197 (160 PXR primary-validation, 37 aux-only). Native-unit MAE and Spearman for the four CYP screen outputs and the PXR 8.25 µM three-seed mean, with constant baselines from training-only data (CYP: unweighted native log2fc mean over the 6,252-molecule screening-training partition, source `preparation/split_assignments.csv`, SHA-256 `d26da9ce522653816dab0876cee082e63f7a49c7fe486e863c1d77c5c7a27544`; PXR: 0.301972839854651 from `pxr_stage1_pins.json`, SHA-256 `144695415cb940d52f47717d26d7eae07a4297850b7a0f838886398c58d7257d`).
- Script and tests: `cyp/scripts/evaluate_screen_predictions.py` (SHA-256 `5eaafb5f3338737d48422902bf01c55ecb75bc54b9d608735fbec4e9f73a1dcb`), `cyp/tests/test_evaluate_screen_predictions.py` (`252625c22275dae430d905004cc6b65f338be5473a8b000aa35f3cd022b58351`). The script ran locally; its test result was never pasted (not seen).
- Outputs: `analysis/cyp_010_screen_eval_20261125/report/` — `evaluation_report.json` (SHA-256 `2097f1445b1d417bbe2211085348597520d0acb7460051ed1adf411089127a65`), `evaluation_metrics.csv` (`27b00b3a925212b25a8fb07fb740c535acc909ef9b1b15aeadd510f5d2960156`), `evaluation_rows.csv` (`919f86799938dbe8d58491b355db8e2936e1a57bdcaca40b1431e68811c7945a`; one row per molecule with predictions, labels, `pxr_role`, `cyp_tiers`), `crosswalk_table.csv` (`8cc613a22fbe84b2200d2e0515123b43075843ca33d342e95e8f39e48667303d`). Inputs: `analysis/cyp_010_screen_eval_20261125/inputs/` (predictions SHA-256 `0a04fe735ea5d55e2fbd0ad72ab62e7f696d137f2865cec1213211a48eeb5d1f`). Builder outputs: `analysis/cyp_010_screen_input_set_20261125/` (`screen_input_mapping.csv` SHA-256 `fd9ab197b736039f497c62085ce0617e8ce3d4f0631729f8c93fa761887111e3`).
- Pooled results (all 197, MAE / Spearman / MAE-over-baseline): CYP1A2 0.509 / 0.568 / 0.804; CYP2C9 0.283 / 0.681 / 0.550; CYP2D6 0.745 / 0.478 / 0.938; CYP3A4 0.903 / 0.701 / 0.585; PXR 8.25 µM mean 0.357 / 0.641 / 0.632. Per-stratum tables: supporting record.
- Corrected interpretation (2026-11-25/26, approved by Miles and recorded): CYP2C9 has the lowest ratio, CYP3A4 the highest overall Spearman, CYP2D6 the weakest ratio and Spearman. Removed as not established: an explanation of CYP3A4's baseline MAE from the training mean, a claim that CYP2D6 predictions cluster near its positive training mean (no prediction distribution was inspected), and the "floor" wording and the claim that a training median would give a lower baseline MAE (no median was computed). D026 Amendment 2 supersedes "plain floor" with "a training-only reference constant, not the MAE-optimal constant, and not a bound on achievable error"; the docstring of the evaluated script was deliberately not edited, to preserve its recorded hash.
- Added caveat: 169 of the 197 molecules are CYP training rows and 28 are CYP validation rows. The primary_validation / aux_only strata describe PXR roles, not CYP holdout status, so the pooled CYP metrics cannot establish held-out CYP performance.
- CYP-tier-split diagnostic (D026 Amendment 2): `cyp/scripts/evaluate_screen_cyp_tier_split.py` (SHA-256 `f916846846b97c32511f453c1dd59e21a933fc80ef990c9cdf30766b6c8e43f7`) and `cyp/tests/test_evaluate_screen_cyp_tier_split.py` (`96dfa23b6d666497bdf00089ffcdd77f41a3ded2178a0037b7f515132d023a04`); hashes computed after the run; Miles reported the tests passing (count not pasted). Outputs in `analysis/cyp_010_screen_eval_20261125/cyp_tier_diagnostic/`: `cyp_tier_split.csv` (SHA-256 `80a2f11e4bacc5c49b62446bf788a04380f678b819250fed4b265837f423f2c5`), `cyp_tier_split_report.json` (`e6952e1ea2f827f10dad9cec80ac307caa14897ff33e8e837cb3fd63010606e6`). Validation-tier MAE/baseline ratios (n 28): CYP1A2 0.851, CYP2C9 0.527, CYP2D6 0.937, CYP3A4 0.636, all below 1; validation-tier Spearman is lower than train-tier for all four outputs (no confidence intervals); the signed mean residual (prediction − label) is positive in every output and tier (+0.088 to +0.643), an aggregate offset that does not separate bias from noise. No cause is asserted.

---

## 3. Approved plans for Tripwire-CYP-011 (new in this handoff; approved by Miles in the message that requested it; no execution authorized)

Records lag: neither plan below has a decision entry in `notes/Decisions.md` yet. This is missing documentation, not an unapproved plan. The bookkeeper was not called for it in this turn because the instruction was to write the handoff only.

### 3.1 Penalty experiments (two separate exploratory arms against unchanged updated-T121)

Context: CYP-010 rank improved macro ST-RAE but failed the full adoption rule, so unchanged remains selected.

Common settings: the existing rank screening weights and λ = 1 for both arms. Preserve the base loss, masks, task/uncertainty weighting, observed-cell denominator, splits, scaling and missing-screen handling. Do not apply CI-width weights to the extra term again. All predictions and bounds use the existing training-fold target scaler.

**Arm A — two-sided squared confidence-band penalty.**
P_A = Σ w_i [max(ŷ_i − U_i, 0)² + max(L_i − ŷ_i, 0)²] / N_obs.
Use existing saved lower bounds only. Missing or incompatible bounds stop A; do not reconstruct them. Purpose: add upward correction below the lower bound while leaving predictions inside the band unpenalized.

**Arm B — robust one-sided upper penalty.**
e_i = max(ŷ_i − U_i, 0); P_B = Σ w_i Hδ(e_i) / N_obs; Hδ(e) = e² for e ≤ δ, otherwise 2δe − δ² (2 × ordinary PyTorch Huber, preserving the old squared penalty below δ). δ is set per enzyme per training fold to the median standardized CI half-width, (U − L)/2, over observed training cells. No sweep. Missing, nonfinite or zero δ stops that branch and is reported; no floor and no silent disabling. Purpose: limit the influence of large violations without weakening small ones.

**Scope:** all 150 matched r1 cells and all three downstream seeds, using the saved unchanged baseline artifacts. No new baseline fits. Keep the fixed tail cohorts (`analysis/cyp_010_tail_cohorts_20261124.json`) and all four adoption requirements: lower macro ST-RAE; no worse mean in any enzyme × seed comparison; lower over-tail MAE for every enzyme; no higher under-tail MAE for any enzyme. If both arms pass, lower macro wins. If neither passes, unchanged stays. Any r2/r3 confirmation needs separate approval.

**First deliverable for Session 11 (plan and tests only):** an implementation plan and tests covering: λ = 0 loss and gradient equivalence; Arm A without its lower term reproducing the existing rank arm; Arm B's exact pieces and continuity at δ; preserved masking, weights and denominator; bound scaling and δ failure handling. Use actual repo paths and commands. Before any submission: supply measured per-unit timing and the extrapolated total against the requested wall time; no Slurm benchmarking. Miles executes commands. This handoff authorizes no training or submission.

### 3.2 Reporting next steps

The 197-molecule descriptive screen evaluation and its checks are complete; do not rerun the evaluation or reopen settled loaders or checks. The approved report shows the four native-unit CYP screen predictions and the three PXR 8.25 µM seed values plus their mean, beside the five unchanged potency outputs. No separate spread column, no combined score, no automatic flag, no cutoff. Next step: prepare the explicit CYP/PXR mapping join and the report implementation plan, showing unmatched and duplicate mappings with counts.

---

## 4. Current position

- Session 010 is closed; the working tree has nothing running. No jobs are queued or running to this session's knowledge (last-reported states above are from 2026-11-25 pastes; live status was not checked).
- Next action: Miles's direction on Session 11. The established first steps are the penalty-arm implementation plan with tests (3.1) and the CYP/PXR join and report implementation plan (3.2), both plans only.

Untracked on purpose (still on disk): `analysis/cyp_010_penalty_preview_20261124/{q3,rank}/` weight CSVs (regenerable), `t121_d024_build_20261124/` (deployed package, keep), tarballs, `.bak` files, hash txt files, `pxr/artifacts/pxr004_weights/chemeleon_mp.pt` (33 MB), older handoffs and loose JSON/markdown files.

Execution conventions used this session: the repo `.venv` hijacks `conda run`; use `/opt/homebrew/Caskroom/miniforge/base/envs/tripwire_cyp002/bin/python` locally and `/scratch/user/mhuang8/envs/tripwire_cyp002/bin/python` on CoreHPC; Mac-to-CoreHPC transfers use `scp -o ProxyJump=mhuang8@chpc-ucsf-bastion-vm1.corehpc.ucsf.edu <files> mhuang8@chpc-ucsf-login-vm1.corehpc.ucsf.edu:/scratch/user/mhuang8/tripwire_staging/`.

---

## 5. Unresolved items and unknowns

Needs Miles's approval or an explicit decision entry:
- Record correction to the 009 handoff and supporting record about validation scaling. Source-reading finding: Chemprop 2.3.1 `on_validation_model_eval` keeps `output_transform` in train mode, so the validation scorer receives standardized predictions and unscales them; this contradicts the 009 handoff and 009 supporting record's keep-raw description. A test was written for it. The finding is not yet confirmed or recorded as a correction.
- r2/r3 confirmation stage (separately authorized later).
- Review trigger and thresholds, the PXR hit definition, and the explicit CYP/PXR join for the potency report (the join's plan is 3.2; each of the others still needs an explicit decision entry). The D025 Amendment 3 label-values relation to D026 is unreconciled.

Unverified or not seen:
- Pytest results for `test_evaluate_screen_predictions.py` (not seen); the test rerun after the coverage addition to the D024 package (unreported); the archive/checksum, loaded-module (cluster Chemprop version), planning and cell-0 elapsed outputs for the D024 package.
- `evaluation_report.json` lists 6,251 CYP training labels per enzyme (`n_train_labels`) while the records describe 6,252 screening-training molecules: the one-molecule gap is unexplained.
- The CYP2C9/CYP3A4 direction check is unaudited.
- Post-inference integrity-check output for the D024 inference jobs was not pasted.

Open design questions the new plans raise (not decided here):
- UNKNOWN: whether lower confidence bounds are saved, in which file and under what column name. The D024 implementation used upper bounds only; the screening files the penalty module parses (`parse_screen`, `load_screen_files`) must be inspected. Arm A stops if they are missing or incompatible.
- The existing penalty module switches the penalty off with a logged reason in degenerate cases, while Arm B's δ failure and Arm A's missing-bound case are specified as stop-and-report. The plan must state how these differ.
- The D024 aggregator's clauses c and d were read here per enzyme from the table; the new rule states "for every enzyme". Whether `cyp/artifacts/d024_penalty/evaluation.json` and `t121_d024_aggregate.py` apply c and d per enzyme or pooled was not re-checked after the context summary and should be read before the new rule is written down.
- Measured per-unit timing: the interactive GPU allocation was unavailable (an earlier interactive `srun` failed with "Insane message length"), and D024 used a recorded one-off deviation (sizing from past batch timings, cell 0 first). How the "measured timing" required for the new arms will be obtained without Slurm benchmarking is UNKNOWN and may need a decision.
- UNKNOWN: where the five unchanged potency outputs for the report are saved and in what format; the report implementation plan must locate them. Known from the mapping file (D026 verification): `cyp_molecule_name` equals `OCNT_ID` for 208 of 208, aux-only `pxr_row_id` equals `OCNT_ID`, primary-role `pxr_row_id` differs, `input_id`, `Molecule_Name` and `row_id` are not interchangeable; `analysis/cyp_010_screen_input_set_20261125/screen_input_mapping.csv` columns: `OCNT_ID, cyp_molecule_name, pxr_row_id, pxr_role, pxr_eligibility_reason, cyp_tiers, cyp_smiles, pxr_source_smiles, pxr_smiles_source, smiles_string_equal`.
