# Tripwire-CYP-009 → Tripwire-CYP-010 Handoff

Written 2026-11-24 from the Tripwire-CYP session 009 chat (closed 2026-11-23 under D023, then reopened for read-only diagnostics); revised the same day to carry Miles's follow-up rulings on edge cases, tail reporting, logging and tests (§2, §6). Next session: **Tripwire-CYP-010**, whose purpose is a three-arm comparison of the updated-T121 recipe with and without a screening-informed extra loss term.

This is a continuity reference, not decision authority. It does not authorize implementation, training, inference, checkpoint loading, benchmarking, GPU jobs, Slurm submissions, branches, commits or edits. The receiving session confirms the position and next action briefly, then waits for Miles's direction.

Source note: everything below comes from the 009 chat record and the files that chat wrote or read. Items not established there are marked UNKNOWN. Nothing was re-verified for this document (no commands run, no files read back while writing it). Code line numbers were read in 009 and may drift if files change.

---

## 1. Objective (as stated by Miles in 009, 2026-11-24)

Compare three arms using the updated-T121 recipe (trial 121 configuration, D015/D007):

1. Updated T121 unchanged.
2. Q3-cutoff penalty.
3. Response-rank penalty.

**Agreed scope and evaluation (stated in chat):**
- All four enzymes. Initially the same r1 screening checkpoint (selected epoch 16) and the matched 150 cells and seeds of the saved updated-T121 baseline (5 repetitions x 10 folds x 3 cell seeds 20261101-20261103).
- Preserve the existing model recipe, partitions and evaluation metrics. Primary metric: macro ST-RAE per cell, lower is better, with per-enzyme and both-tail reporting.
- Development evaluation only: these partitions have already been inspected (§4.3). Protected test data stay untouched.

**Agreed weight definitions (stated in chat):**
- Weights are computed separately for each repetition, fold and enzyme from labelled training-fold molecules with verified screening values only. Seeds share a fold's weights. No validation or test molecule enters either calculation.
- Cutoff arm: Q3 of screening log2fc in that population, computed by linear interpolation between neighbouring observations (agreed 2026-11-24; a quartile-computation convention only, with no assumed linear relationship between screening response and potency or penalty strength); flag values strictly above Q3 (relatively weaker responders, not confirmed inactives). The participant's absolute cutoff and any fixed pIC50 ceiling are not imported.
- Rank arm: rank log2fc ascending, ties take the average rank; raw weight = (average rank - 1)/(n - 1).
- Each arm's weights are normalized to population mean 1 within that enzyme and training fold (not within a batch). The cutoff arm uses the actual flagged fraction, not an assumed 0.25. Equal average weights do not guarantee equal realized penalty losses.

**Agreed loss (stated in chat):**
- The existing base loss is unchanged (masking, uncertainty weights, task weights, reduction).
- P = sum of w x relu(standardized prediction - standardized upper confidence bound)^2 over observed training cells, divided by the same observed-cell count the base loss uses.
- Total loss = existing loss + lambda x P. Native upper bounds are standardized with the same training-fold target scaler as the targets. P is not multiplied by the CI-width weights again. The unchanged arm has no extra term. lambda = 0 must reproduce the existing loss calculation exactly.
- Missing screening values contribute zero extra penalty and keep the ordinary supervised loss and the agreed observed-cell denominator.

**Agreed strength, adoption and stages (stated by Miles 2026-11-24, third message):**
- Lambda = 1.0, fixed for both penalty arms and all four enzymes; no lambda search in this session. It is a clear first test, not an empirically calibrated strength. Mean-one weights do not guarantee that the penalty and base loss have equal influence, so their actual contributions are logged as agreed. A negative result rejects only this strength, not every possible lambda.
- Adoption rule (Miles's proposed rule; deliberately strict, a small regression can block adoption; not something the earlier CheMeleon result in D022/D023 established): lower mean macro ST-RAE across the matched 150 cells; no worse mean in any of the 12 enzyme x seed comparisons; lower MAE on each enzyme's fixed over-prediction tail; no higher MAE on its under-prediction tail. If both arms pass, pick the lower macro ST-RAE, and an exact tie goes to the cutoff arm as the simpler rule. If neither passes, keep the unchanged updated T121.
- Stages: r1 only for the primary comparison. A passing arm is a provisional winner; comparing unchanged versus that winner on r2 and r3 is a separately authorized confirmation stage, not part of the current run, to be done before calling the result robust across screening runs. The three downstream seeds on r1 do not test screening-replicate variation.
- Verification: no separate lambda = 0 training reproduction or loss-measurement run. Use the existing saved unchanged arm. Require deterministic loss-and-gradient equivalence tests at lambda = 0 through the modified loss path, plus the implementation smoke checks (§6.5). This checks compatibility without claiming the original training trajectory was reproduced.
- Duplicates: for the current screen file the condition key is (Molecule_Name, enzyme); it has no duplicated keys today. The duplicate policy stays as a safeguard (§6.3).
- Tails are per enzyme (§6.7).

**Implementation approach (stated in chat):** reuse the existing updated-T121 code with opt-in functionality, no parallel training pipeline, frozen baseline artifacts unchanged. Log the base loss, the unscaled penalty component, the lambda-scaled contribution and flagged counts per enzyme and fold.

## 2. Approvals versus proposals

**Explicitly stated by Miles, with scope (009 chat):**
- D022 (with Amendment 1) and its freeze D023 are recorded in `notes/Decisions.md`; both stay in force. The 009 analyses and this three-arm plan do not change them.
- The one-off batch timing deviation for the D022 timing only (recorded in `notes/Tripwire_CYP_009.md`). It is not a standing permission.
- The three-arm design, the Q3 and rank definitions, the normalization rule, the loss form, the missing-value rule and the "opt-in, no parallel pipeline" approach in §1. These were stated as agreed in chat (recording status: §3).
- Rulings in Miles's follow-up message (2026-11-24): (1) Q3 uses linear interpolation, to be recorded explicitly. (2) An unverified screening match gets no extra penalty and keeps its ordinary supervised loss; duplicate records collapse only when identity, enzyme, condition and value agree; conflicting records are flagged and excluded from the extra term, with counts reported and no silent choice. (3) Fewer than two eligible molecules, or all-equal responses, switch the extra penalty off for that enzyme and fold with the reason logged (a deliberate fallback); if nothing is strictly above Q3 only the cutoff arm has no eligible penalties, and the rank arm is assessed separately, not automatically switched off. (4) Both-tail reporting is defined in §6.7 (a reporting definition, not a training cutoff). (5) Logging records the agreed loss components without changing the frozen executor and saves small summaries after every epoch, with no new logging framework. (6) The old test passes are historical evidence; the relevant tests, including the new penalty tests, are rerun after implementation, and tests that do not exist yet stay marked as needing implementation.
- Rulings in Miles's third message (2026-11-24): lambda = 1.0 fixed; the adoption rule; r1-only primary comparison with r2 and r3 as a later separately authorized confirmation stage; no lambda = 0 training reproduction or loss-measurement run; condition key (Molecule_Name, enzyme); per-enzyme tails (all in §1). These close the corresponding open items of the earlier version of this handoff.
- The read-only diagnostic analyses of 009 (error diagnostic, screening-versus-error), each bounded as "implement and self-test, Miles runs every command".
- Reading and planning only was authorized for the grounding step (the findings are in §4.2).

**Not approved (open; do not treat as settled):**
- The r2 and r3 confirmation stage: separately authorized later, not part of the current run.
- How per-compound predictions for the penalty arms will be obtained for the tail-MAE part of the adoption rule (§6.2), and any validation-only inference, checkpoint loading, GPU run, timing run or submission that this needs.
- The items still marked PROPOSED in §6.2 to §6.4 (operational reading of "no worse" at equality and of the 12 comparisons, cohort-size rounding and ties, the per-epoch log file location, float64 precision).
- Any implementation, training, inference, checkpoint loading, benchmarking, submission or commit: nothing in D024, in Miles's rulings or in this document authorizes them. lambda = 1.0 is a fixed first-test strength, not a calibrated one, and its typical contribution relative to the base loss is unmeasured.

## 3. Records lag

- **The settled arms, definitions, loss form and rules in §1 and the follow-up rulings in §2 were not in `notes/Decisions.md` when the 009 chat produced them.** A first bookkeeper call to record them (the intended D024) was blocked by the harness while a read-only instruction was in force. A second call, made after Miles's follow-up rulings asked for explicit recording, reported that it wrote **D024** ("Screening-Informed Prediction-Penalty Comparison: Three-Arm Design Definitions (CYP)", dated 2026-11-24, approved as design definitions only, authorizing no implementation, training, inference, checkpoint loading, benchmarking, GPU job, Slurm submission or commit) at the end of `notes/Decisions.md`, a matching dated paragraph at the end of §12 of `notes/Tripwire_CYP_009.md`, and a pointer in `notes/Tripwire_CYP_009_supporting.md`. That is the bookkeeper's report; the entries were not read back by the session that wrote this handoff, so confirm them before relying on exact wording. The 009 log's §13 stopping-point text still says no next step is decided and now lags slightly behind D024. The third-message rulings (lambda, adoption rule, stages, condition key, tails; §1) were sent to the bookkeeper as an amendment to D024. Its report (read after this handoff was drafted) says Amendment 1 (2026-11-24) was written at the end of D024 in `notes/Decisions.md`, with a dated paragraph in §12 of `notes/Tripwire_CYP_009.md` and a pointer in the supporting record; the entries were not read back, so confirm the wording. The amendment records rulings 1 to 6, lists as not decided the per-compound prediction route for the tail-MAE criterion, the operational readings and the runtime sizing, and keeps the unaudited CYP2C9 and CYP3A4 coverage check open. Neither this handoff nor D024 changes D021, D022 or D023.
- The 009 grounding findings of 2026-11-24 (loss formula references, bound units, normalization, runtime review; §4.2) were not logged in `notes/Tripwire_CYP_009.md` because the same bookkeeper call was blocked.
- The 009 project-log row (`notes/project_log.md`) was written at the close; whether it should note the reopened analyses is UNKNOWN. No entry for the reopened work was written there.
- The self-test and three-cell smoke outputs of the two 009 diagnostic scripts were never pasted back; the 009 log says so. Only the full-run outputs were reported.
- Uncommitted as of the last check in chat (UNKNOWN whether anything changed since): `cyp/scripts/t121_error_diagnostic.py`, `cyp/scripts/t121_screening_error_analysis.py`, `analysis/cyp_009_error_diagnostic_20261124/`, `analysis/cyp_009_screening_error_20261124/`, and edits to `notes/Tripwire_CYP_009.md` and `notes/Tripwire_CYP_009_supporting.md` made after commit `8e78163`. Earlier untracked files (transfer tarballs, `.bak` files, `chemeleon_mp.pt`, `t121_valinf_*` directories, other handoffs) remain undecided as in the 008 handoff.
- The 009 log still has stale statements from before the reopening (the earlier close text in §13); §12 supersedes them as current status.

## 4. Work and artifacts

### 4.1 Completed in 009 before the reopening (recorded; details in `notes/Tripwire_CYP_009.md` and `_supporting.md`)

- D022 CheMeleon comparison: three screening runs (jobs 3203564-3203566) and three 150-cell arrays (jobs 3207090-3207092); mean O150 difference versus arm R +0.031702754, CheMeleon not preferred. Aggregation report on CoreHPC `/scratch/user/mhuang8/tripwire_staging/t121_d022_o150_report_20261123.json` (SHA-256 `3f70eb45a630548fff3b24460b201d9cd546c9e9245950c06f39dc4637a288ab`). Frozen as D023 (`notes/Decisions.md`).
- Commits on `main` (local only, nothing pushed): `69dc80b` (D021), `8592874` (D022 code, tests, package, 009 log), `8e78163` (D023 freeze, closure, project-log backfill of CYP-007 and CYP-008).

### 4.2 Reopened work (2026-11-23 to 2026-11-24)

**Error diagnostic** (source: user-run console output plus the output tables read locally; output-file hashes not computed).
- Script `cyp/scripts/t121_error_diagnostic.py` (SHA-256 `fdc23a4ba0e2df796d4d98c0b5ba1878f970024b1317cfa0f31cb296060ba214`), outputs in `analysis/cyp_009_error_diagnostic_20261124/full_150/`.
- It analyses the saved post-heuristic r1 validation predictions as scored. The scorer's scaling-heuristic branch (`cyp/scripts/cyp002_validation.py` lines 195-218) was not recorded for any cell; the code path suggests the keep-raw branch, which is an inference, not a record. Range flags 0 of 600; recomputed cell scores match recorded ones to 2.858e-09 (integrity checks only).
- Findings (descriptive): net over-prediction in all four enzymes (mean signed error +0.194 CYP1A2, +0.097 CYP2C9, +0.114 CYP2D6, +0.155 CYP3A4), driven by a right tail; per-molecule error is reproducible across seeds and repetitions; clusters are almost all singletons (cluster-level hypotheses untestable); no training-side evidence exists, so neither a generalization gap nor underfitting is established.

**Screening-versus-error analysis** (source: user-run console output plus output tables; output hashes not computed).
- Script `cyp/scripts/t121_screening_error_analysis.py` (SHA-256 `0760bbfde003a027fc4b370ba073a884a410a597975481a3bf41319a04d6035f`), outputs in `analysis/cyp_009_screening_error_20261124/full_150/`. It ran for CYP1A2 and CYP2D6 only.
- Join: all 1,201 CYP1A2 and 1,269 CYP2D6 molecules had verified screening values (identity `full_key_match`, SMILES identical, one concentration value 4.95049505e-05 M in the file). Coverage and the direction check for CYP2C9 and CYP3A4 have not been run.
- Spearman of log2fc with measured pIC50: -0.871 (CYP1A2), -0.825 (CYP2D6); with signed error (predicted - measured): +0.758 and +0.688; rank partial given predicted pIC50: +0.811 and +0.748; with absolute error (kept separate): +0.326 and -0.086. Top-decile over-prediction tail versus the rest: AUC of log2fc 0.980 (CYP1A2) and 0.787 (CYP2D6). Strongest screening responders are under-predicted (quintile mean signed error -0.52 and -0.68). Per-partition ranges and quintile tables are in the output directory and the 009 log §12.
- Limits: signed error contains the measured value, so a strong association is expected under compressed predictions; the labelled set looks like screening hits (every CYP2D6 compound has negative log2fc; max -0.592), which is an inference; nothing here selects or fits a rule on validation labels.

**Screening-source facts** (009 log §12 follow-on block):
- Source file: `inputs/screen_full.csv` inside `actfound_screen_loader_20261117.tar.gz` (repo root, untracked); also on CoreHPC at the `SCREEN_CSV` path in the `cells_env_*` files (`/scratch/user/mhuang8/tripwire_staging/actfound_screen_loader_20261117/inputs/screen_full.csv`). 45,428 rows, one row per (molecule, enzyme), join key `Molecule_Name`; identity adjudication `inputs/adjudication.tsv`; both hashes match `contract.json` (screen `4f61be8f…`, identity `2f304512…`).
- Repository gap: log2fc meaning, numerator and denominator, normalization and assay concentration are not defined in the repository. Public source: the OpenADMET challenge announcement states a negative log-fold change indicates inhibition and that CYP2D6 is read by parent depletion while CYP1A2, CYP2C9 and CYP3A4 use fluorescence; it does not state the concentration or fold-change definition. How the lowest pIC50 values (about 1.9 to 2.1) were derived is undocumented.
- The frozen screening loader skips every identity-linked compound, so these molecules' screening values were not used to train the screening stage (`cyp/scripts/actfound_model_side.py` identity block; `notes/Handoffs/Tripwire_CYP_007_Handoff_20261119.md` line 88). Only 4 challenge-blinded compounds are identity-linked in the screening file, so availability of screening values for blinded test compounds is a separate unresolved question.

**Grounding for the new design** (source: assistant code reading in 009, source-file evidence; not logged):
- Training loss: `cyp/scripts/t121_weighted_training.py` `weighted_training_mse` (lines 12-65) and `T121WeightedTrainingMPNN.training_step` (lines 185-217). Stock Chemprop MSE (`cyp/scripts/cyp002_model.py` lines 55-56; class check at `t121_weighted_training.py` line 18). Per-cell squared error in standardized units; observed-label mask; observed-cell weights 1/(conf_high - conf_low + 0.25) in native pIC50 width (line 276) normalized to mean 1 over each batch's observed cells (lines 36-41); task weights 1; reduction: sum divided by the batch's observed-cell count, pooled across the four tasks (Chemprop `ChempropMetric.update`/`compute`).
- Units: fold scaler from `training_dataset.normalize_targets()` (`qualification/t121_executor_source_20261117/cyp007_cell_fit.py` lines 211-212); training predictions are standardized (the output transform is the identity in train mode); validation in eval mode returns unscaled predictions to the frozen scorer (ST-RAE, best epoch by `val_macro_st_rae`, patience 20). The model carries `target_mean` and `target_scale` (`cyp/scripts/cyp002_validation.py` lines 281-290).
- Bounds: native-unit `<task>_conf_low` and `_conf_high` dataframe columns; used during training only for the width weight (`build_weighted_pic50_train_loader`, `t121_weighted_training.py` from line 220). Standardized upper bound = (conf_high - mean_t)/scale_t.
- Logging: the frozen executor builds its Trainer with `logger=False` (`cyp007_cell_fit.py` line 336); per-epoch training loss was never saved. Any new component logging therefore needs model-held history written into non-frozen provenance code (proposal, §6.4).
- Weight arithmetic (derived, not measured): rank-arm weight 2r has population mean exactly 1 with tie-averaged ranks; cutoff-arm weight flag/f-hat has second moment 1/f-hat (about 4) against 4/3 for the rank arm, so equal average weight does not equalize realized penalty.

### 4.3 Baseline and evidence status

- Saved baseline (r1, updated T121): 150 cells, array job 3105880 (cells 1-149; cell 0 earlier pilot), O150 0.6769540372619428; cell root `/scratch/user/mhuang8/tripwire_staging/t121_production_chain_20261117_cells_run/trials/trial_121/cells`; sampled config in `artifacts/d009_closure_workspace/d009_t121_comparator_20261109/proposal.json` (batch 16, dropout 0.4, FFN 2000 x 1, message 600 x depth 4, warmup 10).
- Scoring of the saved checkpoints was reproduced (session 007: 150 of 150 cells, frozen-scorer path difference 0.0, from saved rows at most 2.9e-9; `t121_valinf_full_20261118/full_cells_000_149/summary.json`). **A fresh training reproduction has not been demonstrated**: no record shows that rerunning a cell with the current code reproduces its saved training result. The graph-only builders are bit-identical to their pre-edit golden hashes (`cyp/tests/test_t121_descriptor_arm.py`), which covers construction, not training.
- Consequence of the 2026-11-24 ruling that no lambda = 0 training reproduction or loss-measurement run will be done: a difference between a penalty arm and the saved unchanged arm can include non-penalty sources (the code version since the baseline was trained, GPU nondeterminism). The equivalence tests check the loss path only; they do not reproduce the original trajectory, as Miles stated.
- Evidence status: the 150 development folds were inspected in detail in 007 and 009, including the relationship that motivates this design. Results on them are development evidence. The protected internal test (736 molecules) and the challenge data are untouched; nothing in 009 opened them.

### 4.4 Runtime evidence (label per source)

| Item | Value | Source type |
|---|---|---|
| Descriptor arm (D021) screening | 4.90 s/epoch; about 251 s for 50 epochs | measured, batch job 3158201 (008 log §3) |
| Descriptor arm cell 0 | 147.6 s | measured, same job |
| R and D arrays per cell | "about 1.5 to 2.5 min" | recorded in 008 log §5, source of the figure not stated |
| CheMeleon arm screening | 8.24 s/epoch | measured, batch job 3203300 (009 log) |
| CheMeleon arm cell 0 | 129.6 s | measured, same job |
| CheMeleon array mean per task | 139.867 s (150 tasks, array 3207090) | measured, sacct pasted by Miles |
| Plain R arm timing | none recorded; r1 array 3105880 elapsed UNKNOWN | missing |
| Cost of a 150-cell arm | about 5 GPU-hours | estimated (150 x about 2 min) |
| Cost of two new arms | about 10 GPU-hours | estimated |
| Concurrency | roughly 3.5 GPUs on average for the three D022 arrays (450 cells in about 5 h wall) | estimated from pasted check-in times, not a record |
| Queue waits | hours in 009 (smoke job pending; L40S nodes busy) | observed in chat, not recorded or predictable |

Applicability: screening need not be rerun if the arms start from the r1 selected screening checkpoint. The extra term adds a few tensor operations per batch, but it may change the early-stopping epoch and therefore cell time; that effect is UNKNOWN. Per-cell time for the penalty arms, the wall times to request and the queue delay are UNKNOWN. Interactive GPU allocation is broken on the CoreHPC login VM (`srun` on 2026-11-18, `salloc` on 2026-11-23, "Insane message length"), so any GPU run needs a batch job under its own recorded approval.

## 5. Current position

- Everything in 009 is read-only analysis plus the D022 work. Nothing is running or queued on CoreHPC as far as the chat knows (last reported: all D022 arrays completed, 2026-11-23; live status UNKNOWN).
- The three-arm design, definitions, lambda (1.0), adoption rule, staging and tail definition are agreed in chat. D024 was reported written for the earlier definitions; whether the later rulings are recorded is addressed in §3. No penalty code exists. No run, inference, checkpoint load or GPU measurement is authorized.
- Awaiting Miles's direction on the remaining open items (§6.2) and on authorizing implementation.

## 6. Next action

### 6.1 Established next step
The established next step is to begin CYP-010 by confirming this handoff's position and waiting for Miles's direction. Implementation has not been authorized by this document.

### 6.2 Open decisions (not approved)
1. How per-compound predictions for the penalty arms will be obtained. The tail-MAE part of the adoption rule needs them, and only the unchanged arm has saved per-compound predictions (r1 baseline, 150 cells). Options: (a) validation-only inference from each cell's best checkpoint with the existing `cyp/scripts/t121_validation_inference.py`, the route that produced the baseline rows (measured about 272 s of loop time for 150 cells in 007, `notes/Tripwire_CYP_007.md` §6); it loads checkpoints and needs a run, so it needs its own approval; (b) capture inside the training run through the model hook, which needs no extra run but must reproduce the scorer's post-heuristic inputs that the baseline rows used. Not decided; nothing is authorized.
2. Operational readings (PROPOSED, need Miles's confirmation): "no worse" at equality passes (arm mean at most the unchanged mean); the 12 comparisons are 4 enzymes x the 3 cell seeds (20261101-20261103), each the mean per-enzyme ST-RAE over that seed's 50 cells; the over-prediction tail MAE must be strictly lower and the under-prediction tail MAE not higher; tail MAE is the mean absolute error over all predictions of the cohort's compounds (15 per compound in the baseline); cohort size per enzyme = max(1, round(0.10 x n)) with ties broken by compound identifier (n = 1,201 / 1,093 / 1,269 / 1,986 gives 120 / 109 / 127 / 199, arithmetic from the 009 counts).
3. Runtime measurement and wall-time sizing before any production submission: Miles's working rule requires a measured interactive runtime with the extrapolated total against the requested wall time and no benchmarking through Slurm. Interactive GPU is broken on the login VM, and the 2026-11-24 rulings authorize no timing run. Any timing deviation would need its own recorded approval.
4. The r2 and r3 confirmation stage (separately authorized later).
5. Confirmation of D024's wording and of whether the later rulings (§1) were appended to it (§3).

Closed by Miles on 2026-11-24 (no longer open): lambda = 1.0 fixed for both arms and all four enzymes; the adoption rule (§1); r1-only primary comparison with r2 and r3 as a later confirmation stage; no lambda = 0 training reproduction or loss-measurement run; condition key (Molecule_Name, enzyme) for the current screen file; per-enzyme tail groups; the Q3 interpolation convention (linear); the both-tail reporting definition (§6.7); handling of unverified, duplicate, conflicting and degenerate cases (§6.3); the logging approach (§6.4).

### 6.3 Edge-case policies (AGREED 2026-11-24 unless marked PROPOSED)
- Q3 convention (AGREED): linear interpolation between neighbouring observations (the numpy default 75th percentile), recorded explicitly; a computation convention only. Flag strictly above Q3. float64 on the population (PROPOSED precision).
- Missing or unverified screening value (AGREED): no extra penalty, weight 0, excluded from the population statistics, ordinary supervised loss and the observed-cell denominator unchanged.
- Duplicate screening records (AGREED): collapse only when identity, enzyme, condition and value all agree; conflicting records are flagged and excluded from the extra term, with counts of collapsed and conflicting records reported and no silent choice. For the current screen file the condition key is (Molecule_Name, enzyme) (AGREED 2026-11-24); the file has no duplicated key today (verified in 009). Safeguard: if the input later adds concentrations or repeat assay conditions, that aggregation stops until the condition key is updated, and records are never silently averaged.
- Fewer than two eligible molecules (AGREED): penalty off for that enzyme and fold, reason logged (a deliberate fallback; rank is undefined and there is no useful response ranking).
- All-equal responses (AGREED): penalty off for that enzyme and fold in both arms, reason logged.
- Cutoff arm with nothing strictly above Q3 (AGREED): the cutoff arm has no eligible penalties for that enzyme and fold (logged). The rank arm is assessed separately and is not automatically switched off.
- Realized flagged fraction, Q3, population size and missing count per (repetition, fold, enzyme) recorded for every cell. No minimum-population threshold is proposed beyond the checks above.
- CYP2C9 and CYP3A4: coverage and direction are unaudited. Expected per-fold labelled populations are roughly 1,000 to 1,800 molecules per enzyme (arithmetic from the validation-molecule counts, an estimate).

### 6.4 Files to change (proposed; paths verified in 009, no changes made)
- `cyp/scripts/t121_weighted_training.py`: opt-in extension of `UncertaintyDataset`/`collate_uncertainty_batch` (per-cell weight and standardized upper bound), `T121WeightedTrainingMPNN.training_step` (add lambda x P with the base loss's observed-cell denominator and λ = 0 identity), `build_updated_t121_pic50_prefit_state` (opt-in parameter like the existing `chemeleon` one). Logging (approach agreed 2026-11-24): record the base loss, the unscaled penalty component, the lambda-scaled contribution and flagged counts per enzyme and fold, without changing the frozen executor, and save small summaries after every epoch so a failed run keeps its history; no new logging framework. Why a file is needed: the frozen executor builds its Trainer with `logger=False`, so `self.log` values are not kept. PROPOSED mechanism (source-file evidence from 009): model-held accumulators written by a LightningModule epoch-end hook in `t121_weighted_training.py`, appended as one small JSON line per epoch to a file in the cell directory that the non-frozen `t121_production_cell.py` supplies; the file must sit outside `fit/`, because `fit_d007_cell` refuses a non-empty checkpoint directory (`cyp007_cell_fit.py`, the "D007 checkpoint directory is not empty" check).
- New module (name to be chosen, e.g. under `cyp/scripts/`): pure functions for population selection, Q3 flag, rank weight, mean-1 normalization and the penalty, plus the screening join with identity verification (reusing the logic of `cyp/scripts/t121_screening_error_analysis.py`, reading `screen_full.csv` and `adjudication.tsv` against their contract hashes).
- `cyp/scripts/t121_screen_stage.py` (`prepare_selected_screen_handoff` pass-through) and `cyp/scripts/t121_production_cell.py` (CLI flags, arm label, provenance, history output); optionally `cyp/scripts/t121_cells.slurm` for new variables.
- A layout and package pair following `cyp/scripts/t121_d022_layout.py` and `cyp/scripts/t121_d022_build_package.py`, and an aggregation script following `cyp/scripts/t121_d022_aggregate.py` (per-enzyme and both-tail reporting added). Names, run roots and decision numbers are for Miles to set.
- Not to change: `qualification/t121_executor_source_20261117/cyp007_cell_fit.py` (hash-pinned, SHA-256 `625a64fae191573b5a860cdba9a6ae9456d17bc6cd5b0337cc0b7cff1df1a00b`), the saved r1 baseline artifacts, and the frozen D022/D023 outputs.

### 6.5 Tests (each marked `[needs implementation]` unless it runs today)
- Rule agreed 2026-11-24: the passes below are historical evidence (2026-11-23, before any penalty code). After implementation the relevant existing tests and the new penalty tests are rerun, and tests that do not exist yet stay marked as needing implementation.
- Runs today (Mac, `/Users/mileshuang/Desktop/tripwire-project`, env `tripwire_cyp002`): the existing golden-hash guard for the default path,
  `conda run -n tripwire_cyp002 python -m pytest cyp/tests/test_t121_descriptor_arm.py -q -p no:warnings -p no:cacheprovider`
  and the full local suite as last run (535 passed on 2026-11-23, excluding `cyp/tests/test_t121_training_weights.py`, whose collection error predates all 009 edits because `cyp002_model.py` lacks the functions that test parses):
  `conda run -n tripwire_cyp002 python -m pytest cyp/tests -q -p no:warnings -p no:cacheprovider --ignore=cyp/tests/test_t121_training_weights.py`
- `[needs implementation]` zero-lambda equivalence (required by Miles's 2026-11-24 ruling in place of any lambda = 0 training reproduction): deterministic loss and gradient equivalence at lambda = 0 through the modified loss path, so that the total loss and the parameter gradients equal the unmodified path on synthetic batches with the penalty disabled and with the penalty enabled at lambda = 0, and the default-path golden hashes are unchanged. This checks compatibility; it does not claim a reproduction of the original training trajectory.
- `[needs implementation]` implementation smoke checks: one optimizer step per arm on a tiny synthetic batch (finite loss, finite gradients, logged fields present, lambda = 1.0 applied from configuration); CPU only, no project data. The exact scope of the "already-required smoke checks" beyond this was not specified (UNKNOWN).
- `[needs implementation]` adoption-rule evaluator: lower mean macro ST-RAE; the 12 enzyme x seed comparisons; over-tail strictly lower and under-tail not higher MAE on fixed cohorts; both arms pass picks the lower macro ST-RAE with an exact tie going to the cutoff arm; neither passing keeps the unchanged arm.
- `[needs implementation]` consistent units: standardized upper bound equals (conf_high - mean)/scale from the fold scaler; a prediction exactly at the bound gives zero excess; native and standardized computations agree.
- `[needs implementation]` training-fold-only weights: validation and held-out molecules change nothing in a fold's Q3, ranks or normalization (permute or drop them and compare).
- `[needs implementation]` normalization: population mean weight is 1 for both arms within each enzyme and fold; the cutoff arm uses the realized flagged fraction.
- `[needs implementation]` ties: tied log2fc values receive the average rank; the Q3 flag is strict at tied values; the Q3 convention fixed by Miles.
- `[needs implementation]` missing values: zero extra penalty, base loss and the observed-cell denominator unchanged.
- `[needs implementation]` degenerate cases: n < 2 and all-equal values switch the penalty off for that enzyme and fold in both arms, with the reason logged; nothing strictly above Q3 leaves the cutoff arm with no eligible penalties while the rank arm stays on and is computed separately; an enzyme without labelled training molecules logs and disables only the penalty.
- `[needs implementation]` duplicates and unverified matches: identical duplicate records (same identity, enzyme, condition, value) collapse to one; conflicting records are flagged, excluded from the extra term and counted; an unverified match gets no extra penalty and keeps its ordinary supervised loss.
- `[needs implementation]` Q3 convention: linear interpolation matches an independent hand computation on small populations and the flag is strict at tied values.
- `[needs implementation]` per-epoch log: one small line per epoch is written to the cell directory outside `fit/`, survives an exception raised mid-run, and never changes the loss; the frozen executor file is byte-identical.
- `[needs implementation]` tail reporting (§6.7): the groups are built once from the saved baseline predictions and stay identical across arms.
- `[needs implementation]` denominator and mask: P uses the same observed-cell count as the base loss; CI-width weights are not applied to P; lambda unset must raise when a penalty arm is enabled.
- `[needs implementation]` logging fields present per enzyme and fold (base loss, unscaled P, lambda-scaled contribution, flagged counts).
- Existing analysis script, if the CYP2C9 and CYP3A4 coverage and direction check is wanted before design work (read-only, uses saved r1 predictions, writes only the new directory below, which did not exist when checked on 2026-11-24; the script refuses to reuse an existing directory; all input paths in the command were confirmed to exist and the flags were checked against the script's argument parser on 2026-11-24):
  `conda run -n tripwire_cyp002 python -I cyp/scripts/t121_screening_error_analysis.py --predictions-dir t121_valinf_full_20261118/full_cells_000_149 --cluster-assignments cyp/artifacts/cyp_baseline/cluster_assignments.csv --split-manifest-dir cyp/artifacts/cyp_007/split_manifests --screen-tar actfound_screen_loader_20261117.tar.gz --train-csv cyp/data/cyp-challenge-TRAIN_inhibition.csv --isoforms CYP2C9 CYP3A4 --output-dir analysis/cyp_009_screening_error_20261124/coverage_cyp2c9_cyp3a4_full_150 --sha256sums full_cells_000_149_SHA256SUMS.txt`
  (the 009 self-test and smoke outputs for this script were never reported; its self-test is `... t121_screening_error_analysis.py --selftest`).

### 6.6 First implementation step (when Miles authorizes implementation)
1. Have the settled definitions recorded (decision entry) if Miles wants them in the records.
2. Write the pure weight and penalty functions with their unit tests first (no training code touched), then integrate behind opt-in switches, then re-run the golden-hash and full local suites. Package, deployment, any measurement and any run follow only with Miles's separate approvals; Miles runs every execution command.

### 6.7 Reporting definition (agreed by Miles 2026-11-24 as a reporting definition, not a training cutoff)
- Both tails: the most over-predicted and the most under-predicted 10% of molecules, defined by each molecule's mean signed error (predicted - measured) in the saved updated-T121 baseline (r1 saved validation predictions, `t121_valinf_full_20261118/full_cells_000_149`). The groups are kept identical across all three arms, so the report asks whether the original problem cases improve or the opposite end worsens. Overall and per-enzyme macro ST-RAE are reported as well, because errors can move to other molecules. Per enzyme, as agreed 2026-11-24: unique compounds are ranked by their baseline mean signed error across all their available predictions, and the top and bottom 10% cohorts are fixed once and reused for every arm. Cohort-size rounding and ties are PROPOSED in §6.2.
- Caveat to carry: membership is chosen from the baseline's own validation errors, so a change on those groups is not independent evidence (selection on error). Per-molecule error was 96 to 98% reproducible across seeds and repetitions in 009, and prediction SD across seeds averages about 0.08 to 0.15 pIC50, which limits but does not remove that effect. This remains development evidence.

## 7. Unknowns

- How per-compound predictions for the penalty arms will be obtained for the tail-MAE criterion; the operational readings in §6.2; runtime measurement and wall-time sizing; whether and when the r2 and r3 confirmation stage runs.
- Typical loss-component magnitudes (training loss was never saved; the only per-cell record is validation history, and the updated-T121 r1 cell provenance is on CoreHPC, not in this checkout; the frozen-T121 comparator cell provenance is local under `artifacts/d009_closure_workspace/d009_t121_comparator_20261109/cells/` and holds validation history only).
- Whether a rerun of a cell with current code reproduces the saved training result.
- The path and hash of the r1 selected screening record on CoreHPC (r1 screening output root name not confirmed in 009; the 008 log records only selected epoch 16).
- CYP2C9 and CYP3A4 screening coverage and direction; the exact log2fc definition, numerator and denominator, and assay concentration; how the labelled molecules were selected from the screen; how the lowest pIC50 values were derived.
- Penalty-arm cell time, wall times to request, queue delay, and whether the penalty changes the early-stopping epoch.
- Live CoreHPC job state; full SHA-256 of output files in the two 009 analysis directories; the unreported self-test and smoke outputs.
- Whether the 009 project-log row needs a note about the reopened work.
- The scaling-heuristic branch actually taken per saved cell (not recorded anywhere); the diagnostics inherit this.

## 8. Access the receiving session needs

- **Repository checkout** at `/Users/mileshuang/Desktop/tripwire-project` (local Mac). The assistant has no CoreHPC access; every CoreHPC fact comes from output Miles pastes.
- **Untracked local inputs that the existing scripts read** (they exist in the working tree, not in git, so a fresh checkout would lack them): `t121_valinf_full_20261118/`, `full_cells_000_149_SHA256SUMS.txt`, `actfound_screen_loader_20261117.tar.gz`, `t121_production_chain_20261117.tar.gz`, `pxr/artifacts/pxr004_weights/chemeleon_mp.pt`, the two `analysis/` directories.
- **CoreHPC:** login `chpc-ucsf-login-vm1.corehpc.ucsf.edu` via bastion `chpc-ucsf-bastion-vm1.corehpc.ucsf.edu`, user `mhuang8`, staging `/scratch/user/mhuang8/tripwire_staging/`, T121 environment `/scratch/user/mhuang8/envs/tripwire_cyp002`. Transfers use `scp -o ProxyJump=mhuang8@chpc-ucsf-bastion-vm1.corehpc.ucsf.edu ...`; Miles runs every command, transfer and submission. Interactive GPU is broken (§4.4).
- **Local test environment:** conda env `tripwire_cyp002` (pandas 3.0.6, numpy 2.5.3, chemprop 2.3.1).
- **Subagents:** `bookkeeper` (records) and `lookup`, per the project instructions. The decision-entry attempt in 009 was blocked by the harness; Miles decides whether to retry.
- **Governing records:** `notes/Decisions.md` (D021, D022 with Amendment 1, D023), `notes/Tripwire_CYP_009.md` and `notes/Tripwire_CYP_009_supporting.md`, `notes/Handoffs/Tripwire_CYP_008_Handoff_20261121.md`.
