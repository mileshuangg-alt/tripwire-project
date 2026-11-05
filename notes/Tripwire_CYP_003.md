# Tripwire-CYP-003 — Session Log

**Session:** Tripwire-CYP-003
**Date:** 2026-10-10 to 2026-11-04
**Status:** **CLOSED**
**Workstream:** Tripwire — CYP inhibition arm
**Governing document:** `Tripwire_Project_Proposal_Rev3.md`
**Primary execution target:** UCSF CoreHPC
**Primary scientific environment:** `/scratch/user/mhuang8/envs/tripwire_cyp002`

---

## 1. Session Objective

Session 003 completed the post-D006 Chemprop development program under D007.

The session:

1. qualified and executed D007;
2. froze a reduced-fidelity O30 search amendment after the original full-matrix search proved operationally too expensive;
3. completed all 128 search configurations;
4. advanced the frozen top three O30 configurations to authoritative 150-cell O150 evaluation;
5. selected D007 trial 121 as the full-fidelity search winner;
6. evaluated C0 on the identical 150-cell matrix;
7. froze D007-T121 as the preferred Chemprop development configuration;
8. missed the official OpenADMET submission deadline before the completed model-selection workflow was ready;
9. froze a post-deadline 150-checkpoint blind-inference ensemble;
10. generated and froze predictions for all 750 blinded challenge molecules before authoritative label release.

D007 is closed. No further D007 optimization is authorized.

---

## 2. Scientific Boundaries

D007 used only the frozen development population:

- original train: 3,433 compounds;
- original validation: 736 compounds;
- D007 development universe: 4,169 compounds;
- disclosed internal test: 736 compounds, excluded from D007;
- OpenADMET challenge outcomes: excluded from development and selection.

The retained objective remained stock masked multitask MSE.

The four endpoints remained:

1. `CYP1A2_pIC50_direct_inhibition`
2. `CYP2C9_pIC50_direct_inhibition`
3. `CYP2D6_pIC50_direct_inhibition`
4. `CYP3A4_pIC50_direct_inhibition`

Lower macro ST-RAE remained better.

Tripwire remains characterization/re-ranking only and gains no attrition authority from this work.

---

## 3. D007 Full-Fidelity Design

The original D007 objective used:

- 5 clustered repetitions;
- 10 folds per repetition;
- 50 development folds total;
- 3 frozen training seeds per fold: `20261101`, `20261102`, `20261103`;
- identical folds and seeds for every configuration and C0.

Each full configuration therefore contained 150 fits.

For configuration j, the authoritative objective was the arithmetic mean of the 150 fit-level validation macro ST-RAE values, equivalently the mean of the three seeds within each fold followed by the mean across 50 folds.

The protected 736-compound internal test did not enter D007.

---

## 4. Reduced-Fidelity Search Amendment

The original plan to evaluate all 128 configurations at 150 cells each proved operationally too expensive.

Before continuing the search, D007 was amended to use a frozen 30-cell search objective, O30, while preserving full O150 authority for final selection.

The frozen subset contained 10 split manifests × all 3 training seeds:

- repetition 1: folds 1 and 9;
- repetition 2: folds 2 and 10;
- repetition 3: folds 2 and 7;
- repetition 4: folds 1 and 3;
- repetition 5: folds 2 and 10.

Frozen subset artifact SHA-256:

`6da42fbfb24a3bc3425bddb12ac2fb65c90b999aab3a981a62b4153d094331ea`

Existing-data qualification on trials 1–14 showed:

- Spearman rho = `0.947`;
- Kendall tau = `0.846`;
- the full-objective winner remained reduced-objective rank 1;
- 2 of the full-objective top 3 remained in the reduced-objective top 3.

This was qualification evidence only and did not establish O30/O150 equivalence.

The amended search retained:

- 128 total configurations;
- 32 random-startup configurations;
- 96 sequential TPE configurations;
- the frozen search space;
- the frozen controller RNG;
- all three training seeds;
- no pruning.

O30 had search-ranking authority only.

---

## 5. Completed D007 Search

The amended search completed all 128/128 configurations.

Long-running controller jobs reached their 12-hour wall-time limits during production. Completed scientific evidence and persisted controller/HyperOpt state were preserved and resumed rather than restarting the search or substituting trials.

The frozen O30 top ten were:

| Rank | Trial | O30 |
|---|---:|---:|
| 1 | 121 | 0.698191004621 |
| 2 | 62 | 0.698862474494 |
| 3 | 125 | 0.698893654706 |
| 4 | 114 | 0.699057780597 |
| 5 | 66 | 0.699103911952 |
| 6 | 126 | 0.699364323056 |
| 7 | 119 | 0.699464051093 |
| 8 | 63 | 0.699601167915 |
| 9 | 67 | 0.700460832660 |
| 10 | 124 | 0.700835933811 |

Frozen finalists:

`[121, 62, 125]`

Controller-state SHA-256 at search freeze:

`bc32e72e73a2e276ff8a14863c07a466f3cc4105aad09e786428eed61db99016`

Searcher SHA-256 at search freeze:

`835a87015dc2e911ce54bea7cb7fbea924eb6d1ad713094ce0951eb16734caa9`

No further TPE proposal was authorized.

---

## 6. Authoritative O150 Finalist Evaluation

All three finalists completed the exact frozen 150-cell matrix with zero failed cells.

| Rank | Trial | O30 | O150 | O150 - O30 |
|---|---:|---:|---:|---:|
| 1 | **121** | 0.698191004621 | **0.701212576796** | +0.003021572174 |
| 2 | 125 | 0.698893654706 | 0.702315231168 | +0.003421576463 |
| 3 | 62 | 0.698862474494 | 0.702505363217 | +0.003642888724 |

The frozen D007 search winner is:

**D007-T121**

O30 rank 1 remaining O150 rank 1 is retrospective evidence from this completed search only. It is not a general claim that O30 is equivalent to O150.

---

## 7. C0 Full-Matrix Comparison

C0 was evaluated on the identical 50-fold × 3-seed matrix.

Authoritative comparison:

- D007-T121 O150: `0.701212576796`
- C0 O150: `0.718103374388`
- C0 minus T121: `+0.016890797592`

T121 reduced macro ST-RAE by approximately 2.35% relative to C0 under the D007 development procedure.

The frozen rule was:

`T121 replaces C0 iff O150(T121) < O150(C0)`

Observed:

`0.701212576796 < 0.718103374388`

Result:

**D007-T121 replaces C0 as the preferred Chemprop development configuration.**

Authoritative selection record:

`cyp/artifacts/cyp_007/finalist_completion_v1/C0_COMPARISON.json`

SHA-256:

`48bd619ebd42b30b0c8862f915b7696e2156e822f48e220789b1563eb46817a8`

D007 selection is closed and must not be reopened.

---

## 8. OpenADMET Deadline

The official OpenADMET CYP challenge submission deadline passed before the completed D007 model-selection and prediction workflow was ready.

No official final OpenADMET submission was made from D007-T121.

Accordingly:

- no official prospective OpenADMET challenge result is claimed;
- no final-leaderboard claim is available;
- the missed deadline does not alter the completed D007 development result.

The project instead preserved the possibility of a later post-deadline blind evaluation by freezing predictions before authoritative challenge labels were released or accessed.

---

## 9. Post-Deadline Blind-Inference Amendment

After D007 selection closed, an inference-only amendment froze a 150-checkpoint T121 ensemble.

The ensemble uses exactly all frozen 150 `(repetition, fold, seed)` members.

The fail-closed contract requires:

- exactly 150 members;
- no missing member;
- no checkpoint substitution;
- no checkpoint exclusion;
- no performance weighting;
- no averaging fewer than 150 members;
- every checkpoint matched to its frozen cell identity;
- every checkpoint matched to its training-fitted target transformation;
- original-pIC50 prediction scale;
- authoritative prediction = equal-weight arithmetic mean across exactly 150 checkpoints;
- checkpoint SD with `ddof=1`, descriptive only;
- no fine-tuning;
- no calibration;
- no model adaptation;
- no checkpoint reselection.

Any later labeled evaluation is a **post-deadline blind evaluation, not an official prospective OpenADMET challenge result**.

---

## 10. Ensemble Qualification

The frozen checkpoint inventory established:

- required members: 150;
- inventory members: 150;
- exact canonical identities: true;
- checkpoint/identity failures: 0.

Selection provenance was anchored to `C0_COMPARISON.json` SHA-256:

`48bd619ebd42b30b0c8862f915b7696e2156e822f48e220789b1563eb46817a8`

A second fail-closed qualification reconstructed every member's training-only target transformation from its frozen D007 training partition and required strict checkpoint loading.

Final qualification:

- 150/150 member-specific target transformations reconstructed;
- 150/150 checkpoint hashes unchanged;
- 150/150 strict checkpoint loads passed;
- exact canonical member set preserved;
- zero failures.

Frozen inference artifacts include:

- `cyp/artifacts/cyp_007/t121_blind_inference_v1/INFERENCE_AMENDMENT.json`
- `cyp/artifacts/cyp_007/t121_blind_inference_v1/CHECKPOINT_INVENTORY.json`
- `cyp/artifacts/cyp_007/t121_blind_inference_v1/TARGET_SCALER_STRICT_LOAD_QUALIFICATION.json`
- `cyp/artifacts/cyp_007/t121_blind_inference_v1/INFERENCE_IMPLEMENTATION_FREEZE.json`

Frozen inference implementation:

`cyp/scripts/cyp007_t121_blind_inference.py`

It reuses `cyp006_checkpoint_evaluation.predict_checkpoint` for original-pIC50 checkpoint inference.

The blind-inference path does not invoke challenge scoring.

---

## 11. Blind Input Identity

The organizer-provided blinded input was staged to CoreHPC and byte identity was verified.

Authoritative SHA-256:

`a342f8444a8dcb531ca12f3685293f0bd6c36ae9073f491e44a9bc1cc4b741f9`

CoreHPC inference path:

`cyp/data/cyp-challenge-TEST-BLINDED.csv`

---

## 12. Blind-Inference Attempt History

### Attempt 001 — Slurm job 2886913

Status: **failed before challenge access**

The blinded CSV had not yet been staged to the expected CoreHPC path.

The run failed at file open with `FileNotFoundError`.

Consequences:

- challenge file accessed: no;
- challenge molecules accessed: no;
- prediction output created: no;
- predictions generated: no;
- labels/scores/rank/leaderboard accessed: no.

This was an operational staging failure only.

### Attempt 002 — Slurm job 2887724

Status: **failed after molecular-input access but before prediction**

The initial blind-input adapter reused `_frame_to_molecule_dataset()`, a training/evaluation constructor requiring CYP target columns. The blinded challenge file correctly lacked those columns.

Consequences:

- challenge molecular input accessed: yes;
- predictions generated: no;
- challenge labels accessed: no;
- challenge scores accessed: no;
- challenge rank/leaderboard accessed: no.

A narrowly scoped mechanical amendment replaced the target-requiring dataset construction with target-free Chemprop `MoleculeDatapoint.from_smi(smiles)` construction.

No model, checkpoint, ensemble membership, weighting, aggregation, prediction scale, or scientific selection rule changed.

### Attempt 003 — Slurm job 2889407

Status: **successful**

All 150/150 checkpoint predictions completed for all 750 blinded molecules.

Final provenance recorded:

- configuration: `D007-T121`;
- members: 150;
- challenge rows: 750;
- prediction scale: original pIC50;
- model adaptation after challenge access: false;
- labels accessed: false;
- scores accessed: false;
- rank/leaderboard accessed: false.

---

## 13. Frozen Blind Predictions

Successful prediction root:

`cyp/artifacts/cyp_007/t121_blind_inference_v1/prediction_attempt_003/`

Authoritative ensemble:

`ensemble_predictions.csv`

SHA-256:

`51002c6dbe75041bf4c874f1a4dce4d59f293e82d9d8e4771a2132160da7e7ca`

Complete 150-member matrix:

`member_predictions.csv`

SHA-256:

`f7e683c7ab8573762d19231e092fe7d48e708c9e572d526db12a4c502c02e8cb`

The complete prediction cube represents:

`150 checkpoints × 750 molecules × 4 CYP endpoints`

The authoritative ensemble is the arithmetic mean across exactly 150 checkpoint predictions for each molecule and endpoint.

The prediction artifact was frozen before authoritative challenge-label access in:

`cyp/artifacts/cyp_007/t121_blind_inference_v1/BLIND_PREDICTION_FREEZE.json`

---

## 14. External-Evaluation State at Closure

At Session 003 closure:

- authoritative OpenADMET challenge labels had not been released;
- challenge labels had not been accessed;
- challenge scores had not been accessed;
- challenge ranks/leaderboard results had not been accessed;
- no scoring of the frozen predictions had occurred;
- no model adaptation occurred after challenge-molecule access.

Current state:

**750-molecule blind predictions frozen; external scoring pending authoritative label release.**

If authoritative labels become available later:

1. preserve and hash the untouched authoritative label artifact;
2. freeze the exact scoring contract before opening labels against predictions;
3. score the already-frozen prediction matrix exactly once;
4. describe the result as a post-deadline blind evaluation;
5. do not use the result to reopen D007 or adapt T121.

---

## 15. Scientific Outcome

### D007 development selection

**D007-T121 is the preferred Chemprop development configuration.**

- O150(T121): `0.701212576796`
- O150(C0): `0.718103374388`

### Reduced-fidelity search

The frozen O30 procedure retained the eventual O150 winner in this completed search.

This is a completed-search observation, not a general equivalence claim.

### External blind predictions

A fail-closed 150-checkpoint T121 ensemble generated complete predictions for the 750 blinded OpenADMET molecules.

Those predictions were frozen before authoritative labels became available.

### Challenge claim boundary

Because the official submission deadline was missed:

**the project does not claim an official prospective OpenADMET challenge result.**

A later evaluation against newly released authoritative labels may provide post-deadline blind external evidence.

---

## 16. Artifact Preservation

Preserve all Session 003 artifacts, including:

- D007 split manifests;
- qualification artifacts;
- all 128 search configurations;
- completed full-matrix evidence from early trials;
- reduced-search artifacts;
- superseded pre-amendment controller/searcher state;
- resumed controller states;
- failed/timed-out controller evidence;
- finalist completion artifacts;
- C0 full-matrix evaluation;
- final-selection artifacts;
- `C0_COMPARISON.json`;
- T121 checkpoint inventory;
- target-scaler strict-load qualification;
- inference amendment;
- inference implementation freeze;
- failed blind-inference attempts 001 and 002;
- target-free adapter amendment;
- successful attempt 003;
- full member prediction matrix;
- ensemble prediction matrix;
- blind-prediction freeze;
- hashes, logs, wrappers, manifests, and provenance.

No intermediate artifact is to be deleted merely because it is not expected to appear in a final deliverable.

---

## 17. Scientific Boundaries and Non-Claims

- D007 does not use the disclosed 736-compound internal test for model selection.
- T121 preference is a development-resampling result, not independent external validation.
- O30 is not claimed to be generally equivalent to O150.
- The missed OpenADMET deadline means no official prospective challenge result exists.
- The frozen 750-molecule prediction artifact has not been scored.
- No blind challenge performance claim is currently authorized.
- Checkpoint SD is descriptive only.
- Tripwire remains characterization/re-ranking only.
- D007 may not be reopened based on future challenge labels or performance.

---

## 18. Next CYP Work

The next CYP implementation task is a stable Tripwire inference interface around the selected T121 model:

`molecule -> CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4 direct-inhibition potency predictions + uncertainty + provenance`

This is an integration/inference task, not further D007 optimization.

Blind evaluation remains parked until authoritative OpenADMET labels are released.

---

## 19. Session Closure

**CLOSED — Tripwire-CYP-003**

D007 completed its frozen search and final-selection procedure.

Final selected Chemprop development configuration:

**D007-T121**

- O150(T121): `0.701212576796`
- O150(C0): `0.718103374388`

The selected T121 ensemble subsequently generated and froze complete predictions for the 750-compound OpenADMET blinded challenge set before authoritative labels were released.

No additional D007 training, tuning, checkpoint selection, calibration, or model adaptation is authorized.

Further CYP work proceeds under the next separately scoped implementation task.
