# Tripwire PXR: handoff from Tripwire-PXR-003 (closed) to Tripwire-PXR-004

**Handoff date:** 2026-11-19
**From:** Tripwire-PXR-003 (declared closed by the user on 2026-11-18; committed as `319e7a5`)
**To:** Tripwire-PXR-004 (the user named this number in the handoff request)
**Status of this document:** a continuity reference. It is not decision authority and does not authorize implementation, runs, submissions, branches, installs or edits. The receiving session confirms the current position and next action briefly, then waits for the user's direction.
**Sources:** this chat's own record (the user's messages, tool output, bookkeeper hand-back reports) and the PXR-002 handoff `notes/Handoffs/Tripwire_PXR_002_Handoff_20261118.md`, which this chat read at its start. Where a fact comes only from the PXR-002 handoff, it is marked "(via PXR-002 handoff)". Permanent records written by the bookkeeper were checked by targeted searches only, not re-read in full.

## 1. Direction for the next session (the user, 2026-11-19)

**Scope: sequential auxiliary pretraining, two arms.**

- Both arms start from the **same released CheMeleon encoder**.
- **Arm 1:** fine-tune directly on pEC50.
- **Arm 2:** first train on the **eligible single-concentration PXR responses**, then fine-tune on pEC50.
- The **downstream setup is matched** across arms, to isolate the value of the auxiliary-training step.
- **Preserve the existing data exclusions and evaluation boundaries.** The consumed holdout stays closed. The official test stays unused.
- **The next session reviews compatibility and proposes a bounded pilot before any implementation or training.**

This is the "improving auxiliary learning" follow-up that PXR-002 and PXR-003 set aside as separate. It is not recorded in `notes/Decisions.md`. See section 3.

## 2. Approvals versus proposals

**Stated direction (above):** the scope and the review-first constraint. It authorizes no code, install, training or submission.

**Not decided, not stated:**

- whether the CheMeleon encoder is frozen or fine-tuned in either stage
- stage-2 head and loss design, and whether stage 1 uses the three-head log2FC structure from D017/D018
- cohort choice (see 5.4)
- whether Chemprop descriptors are used with CheMeleon
- any comparison criterion between the arms
- whether RMSE continues as a reported metric for this session
- any numeric improvement threshold or statistical test (none exists; none should be invented)

## 3. Records lag

- **The PXR-004 direction is not in `notes/Decisions.md`.** The next session should ask for it to be recorded through the bookkeeper once the user confirms the scope. No `notes/Tripwire_PXR_004.md` exists yet.
- **No D-entry exists for adopting CheMeleon.** Neither D017 nor D018 mentions it. D017's multitask Chemprop direction and D018's auxiliary rules remain the last recorded neural decisions.
- PXR-003 itself is fully recorded: D019, `notes/Tripwire_PXR_003.md`, `notes/Tripwire_PXR_003_supporting.md` and a `notes/project_log.md` row.

## 4. Settled decisions and boundaries to carry forward (do not reopen)

Authoritative log: `notes/Decisions.md` (D014, D017–D019 are the PXR entries).

**Data and identity**

- Release directory `pxr/data/openadmet_pxr_release_2026-11-15/`. Primary table `pxr-challenge_TRAIN.csv` (4,139 rows), SHA-256 `fdc2252bb5763ab5742e78cb1ba692bebe131a972cf83aa1bd45e43d8c5bc223`.
- Identity key is exactly `Molecule Name` (OADMET ids). Columns used: `Molecule Name`, `SMILES`, `pEC50`. Released SMILES are preserved; standardization was never declared and must not be introduced silently.

**Frozen D014 split**

- `pxr/artifacts/pxr001_split/pxr001_split_assignments.csv`, SHA-256 `c6028f1b7216ab2ea4bab96616fc02b95ce011be20b1b96f807d1e7d582ece46`. 2,897 train / 621 validation / 621 holdout, group-aware (Bemis–Murcko scaffold groups, whole groups indivisible). Do not reopen.

**Auxiliary-data eligibility (D018 item 1, D017 Amendment 1)**

- Auxiliary labels are the released library-compound single-concentration measurements (`single_concentration_TRAIN.csv`, `log2_fc_estimate`, keyed on OCNT_ID), at three concentrations: 8.25, 33 and 99 µM. Positive-control rows (27 rows at 0.98 µM) are excluded.
- Counter-assay, crude HT-chem and semi-pure data are left out.
- Auxiliary labels on primary validation and holdout molecules are kept out of training.
- Auxiliary-only molecules enter only if they pass the D014 scaffold and connectivity-group exclusions against validation, holdout and official-test structures. No new similarity cutoff is introduced.
- The 99 µM readout occurs only in experiment OPA_PXR-030, so concentration and experiment effects cannot be separated for that readout (D017 Amendment 1).
- These rules were written for the multitask-head design. Whether the sequential design reuses them unchanged is for the compatibility review, but the user's direction is to preserve the existing exclusions.

**Evaluation boundaries**

- Model selection stays on the 621-compound validation set.
- **The 621-compound holdout is consumed and closed.** It was scored once by the PXR-002 neural ensembles and once, reporting-only, by the PXR-003 baselines (internal-retrospective, Rev 3 line 53). It must not be used for tuning or selection, and the user's direction keeps it closed.
- **The official test (513 blinded molecules) is unused**, and its firewalled labels (`pxr/data/openadmet_pxr_heldout_labels_2026-11-15/`) must not be used.
- No retraining on train plus validation is authorized.
- Primary metric MAE; ranking metric Spearman. RMSE was added for PXR-003. Whether it continues is UNKNOWN (section 2).

## 5. What the closed sessions produced

### 5.1 PXR-003 (this chat), results and artifacts

D019: ECFP4 + LightGBM and ECFP4 + random forest baselines on the full 2,897 train molecules, selected on validation MAE, frozen, then scored once on the holdout (reporting only). Environment: new conda env `tripwire_pxr003` (Python 3.12.16, RDKit 2026.03.6, LightGBM 4.7.0, scikit-learn 1.9.1, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, pytest). It was created without Chemprop.

| | Validation MAE (5-model ensemble) | Holdout MAE | Holdout Spearman | Holdout RMSE |
|---|---|---|---|---|
| Chemprop multitask (PXR-002, 2,896 train) | not recorded for the ensemble (single-model mean 0.5458 via PXR-002 handoff) | 0.4843 | 0.7357 | 0.6794 |
| Chemprop aux-off (PXR-002) | not recorded (single-model mean 0.5499) | 0.4847 | 0.7368 | 0.6837 |
| ECFP4 + LightGBM | 0.6049 | 0.5446 | 0.7278 | 0.7198 |
| ECFP4 + random forest | 0.6745 | 0.6127 | 0.6822 | 0.8086 |
| Constant training mean | 0.9218 | 0.9239 | undefined | 1.1373 |

Source: recorded command output pasted by the user and `pxr/artifacts/pxr003/holdout_report/holdout_report.json` (SHA-256 `b7cb8b8c9b2e9bbdc124724e15bda4e42daf3600f8a83f180b7607d50a919d51`, hashed locally by the assistant). The neural validation Spearman and RMSE were never computed, and neural validation predictions were not saved. The comparison is descriptive; no threshold or test exists.

**Observation, uninvestigated:** holdout MAE is lower than validation MAE for every system (LightGBM 0.605 to 0.545, RF 0.675 to 0.613, neural about 0.546 to 0.484), while the constant baseline is about equal (0.922 versus 0.924). This is not specific to the neural models. No cause has been established.

**PXR-003 artifacts** (all committed in `319e7a5`):

- `pxr/artifacts/pxr003/` (`tune_lgbm/`, `tune_rf/`, `frozen/`, `final/`, `holdout_report/`)
- `pxr/scripts/pxr003_*.py`, `pxr003_config.json`, `pxr/tests/test_pxr003.py`
- frozen config SHA-256 `38238eb784a514289b04d65addebffe2ca1953e23af54157ce9f85cff4090030`
- baseline predictions SHA-256 `85231ba487521d3c8aa4bfb2affc9cace29ce482ff7713ecf8f28c966591f48f`
- all other hashes: `notes/Tripwire_PXR_003_supporting.md`

Tests: 22 passed (pytest on `pxr/tests/test_pxr003.py` and `pxr/tests/test_pxr001_baseline.py`, user's pasted output). Smoke test PASS (user's pasted output).

### 5.2 PXR-002 neural work (via PXR-002 handoff, not re-verified in this chat)

- Multitask Chemprop (one primary pEC50 head, three auxiliary log2FC heads), `v1_rdkit_2d_normalized` descriptors, 65:35 nominal primary:auxiliary weighting, 36-configuration grid, selected configuration `h600_d4_f2_p0.2`.
- Auxiliary-off ablation: `not_distinguishable` from multitask under the approved rule (validation mean MAE 0.53752 versus 0.54140). The auxiliary heads showed no demonstrated benefit.
- Cohort v2: OADMET-0003727 (auranofin) excluded from the descriptor-based training cohort because its RDKit partial-charge descriptors are non-finite; 10,818 training molecules (the 10,819 v1 artifact minus that one). Eligibility artifact v1: 2,897 primary + 7,922 auxiliary-only training molecules, 204 auxiliary-only excluded.
- Artifacts: `pxr/artifacts/pxr002_aux/` (v1), `pxr/artifacts/pxr002_aux_v2/` (cohort v2, cached descriptors, one-row amendment), `pxr/artifacts/pxr002_holdout_v1/`, `pxr/artifacts/pxr002_final_20261118/` (`holdout_predictions.csv` SHA-256 `45c84c925b9dfc1c5c959e3a95c6b01961667dd19e2338470e8ae26655aa07aa`). Code: `pxr/scripts/pxr002_*.py` and `.slurm` (including `pxr002_aux_data.py`, `pxr002_train.py`, `pxr002_dataset.py`, `pxr002_loss_weights.py`), tests `pxr/tests/test_pxr002.py`.
- Last-reported job statuses (all COMPLETED 0:0, via PXR-002 handoff, 2026-11-18): stage A 3113907, stage B 3119543, ablation 3120327, final training 3120551. Nothing from PXR is believed to be running or queued. That is not verified live.

### 5.3 Compute and environments

- CoreHPC (SSH via the user's bastion and login host; the user authenticates). Staging root, stage outputs and the Chemprop environment live under the user's `/scratch/user/<user>/` area. The exact paths are in section 5 of the PXR-002 handoff; they are not repeated here. Chemprop 2.3.1 is recorded there for the `tripwire_cyp002` environment (also present locally with Python 3.12.14, RDKit 2026.03.6, scikit-learn 1.9.1).
- Local environments (checked in this chat): `tripwire_cyp002` (Chemprop, pytest, scikit-learn; no LightGBM), `tripwire_pxr003` (no Chemprop), `oadmet_cyp_tutorial` (LightGBM, no pytest), project `.venv` (Python 3.15; do not use for scaffold work).
- PXR-002 neural training ran on GPU on CoreHPC. The PXR-003 baselines ran locally on CPU (benchmark typical 24.2 min, upper bound 82.6 min; actual tuning about 13 min for LightGBM and about 2 min for the forest).

### 5.4 Open cohort question for this direction

PXR-002 used the 2,896-molecule cohort (auranofin out) because of the descriptor-based input. PXR-003 baselines used the full 2,897 (D019, with the difference disclosed). Which cohort the two CheMeleon arms use is not decided and may depend on whether descriptors are used (UNKNOWN). The arms must share it. D018 Amendment 1 item 3 requires the same eligible cohort in comparisons that isolate auxiliary-data effects, which is what this direction does.

## 6. Current position

- PXR-003 is closed and committed locally as `319e7a5` (not pushed). Earlier local commits: `fabec38`, `1dc6862`, `59fe1c0`, `fa31dab`, `aca631d`, `f406fe9` (via PXR-002 handoff).
- Untracked and intentionally left out of the PXR-003 commit: `notes/Handoffs/Tripwire_PXR_002_Handoff_20261118.md`, `notes/Tripwire_CYP_007.md`, unrelated CYP files, `pxr002_stage_*` staging copies. This handoff file is also new and uncommitted.
- No work toward the CheMeleon arms has started. Nothing was inspected for CheMeleon in this chat.
- No question is awaiting the user.

## 7. Next action (established by the user)

Review compatibility, then propose a bounded pilot **before any implementation or training**. The receiving session confirms the current position, then waits for the user's direction.

Compatibility questions the review would need to answer (open questions, not requirements or approvals):

1. Whether and how the CheMeleon released encoder is available in the existing Chemprop 2.3.1 environment (locally and on CoreHPC), including obtaining the weights, any access restrictions, and whether compute nodes can reach them. UNKNOWN in this chat.
2. How the encoder's featurization and architecture fit the existing PXR-002 pipeline (`pxr/scripts/pxr002_*`), and whether descriptors are used. UNKNOWN.
3. How the eligible single-concentration data (`pxr002_aux` artifacts) map onto a first-stage training objective, which head design, and how stage 2 re-heads to pEC50. UNKNOWN.
4. Which cohort, seeds, epochs, early stopping and validation selection are matched across the two arms, so only the auxiliary-training step differs. UNKNOWN.
5. The cost and the benchmark required before any SLURM submission. A measured interactive runtime with an extrapolated total versus the requested wall time must precede any batch job; no GPU benchmark exists for CheMeleon.

Existing criteria to carry: selection on validation only; descriptive reporting; constant training-mean predictor as the baseline; no invented thresholds or tests.

## 8. Unknowns

- Whether the pinned Chemprop 2.3.1 exposes the CheMeleon foundation model, how weights are obtained, and any license or access constraints.
- Whether RMSE and Spearman continue as reported metrics for PXR-004 (not stated).
- Neural validation-set ensemble metrics (never saved or computed).
- The reason for holdout MAE being below validation MAE.
- Rev 3's file path and byte identity (a Downloads file, not a repository file; carried from PXR-001/PXR-002).
- Items carried from PXR-001, never resolved: eligibility of crude and semi-pure HT-chem material; the structural identity `OCNT-2395416` = `OADMET-0006234`; observations that 3 TRAIN molecules lack dose-response curves, 348 have wide confidence intervals and the primary table has no inactives (observations, not exclusions). The Rev 3 line 129 numerical improvement bar and any statistical test remain undefined. Standardization remains undeclared.
- Any state after 2026-11-18 (CoreHPC job state, repository changes made elsewhere).
