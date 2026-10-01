# Tripwire-CYP-001 — Session Log

**Session:** Tripwire-CYP-001
**Date:** 2026-09-30 to 2026-10-01
**Status:** CLOSED
**Workstream:** Tripwire — CYP inhibition arm
**Governing document:** `Tripwire_Project_Proposal_Rev3.md` (Revision 3, September 30, 2026)
**Challenge repository:** `OpenADMET/CYP-Challenge-Tutorial`
**Tutorial repository commit:** `832ae6d6447b07c2c71059de34d07f1130388247`

---

## 1. Session Objective

Begin the Tripwire CYP inhibition arm under the governing Rev 3 proposal and establish the first defensible independent reference model before moving to the preregistered masked-label multitask challenger and subsequent neural models.

The direct-inhibition CYP arm predicts continuous pIC50 values for:

- CYP3A4
- CYP2C9
- CYP2D6
- CYP1A2

The official challenge primary metric is ST-RAE, with macro-averaged ST-RAE used for the four-endpoint direct-inhibition track. MAE, R2, Spearman rho, and Kendall tau are secondary reported metrics. The challenge evaluator uses the supplied confidence-interval bounds in ST-RAE.

Tripwire remains a characterization/re-ranking layer with no candidate-attrition authority.

---

## 2. Governing Scope and Approved Section 5 Plan

The approved Section 5 scientific decisions were not amended during this session. The implementation plan was clarified as follows.

### 2.1 Independent baseline

- Model family: LightGBM gradient-boosted decision trees.
- One independent regressor per CYP isoform.
- Molecular representation fixed to Morgan/ECFP4-equivalent radius 2, 2048 bits.
- Hyperparameters tuned independently per isoform using validation only.
- Held-out internal test remains untouched until all configurations are frozen.

### 2.2 Analog-cluster holdout

The internal evaluation split is an analog-cluster holdout, motivated by the challenge's analog-expansion blind test.

- Cluster compounds by structural similarity.
- A cluster may not cross train, validation, or test.
- Target allocation: 70% train / 15% validation / 15% test.
- Validation is used only for hyperparameter selection.
- Test is used only once, after all configurations are frozen.

Operational split construction used:

- Morgan radius 2
- 2048 bits
- Butina clustering
- Tanimoto similarity threshold 0.70
- fixed random seed `20261001`

### 2.3 LightGBM tuning implementation

The amended implementation plan was frozen as:

- Randomized search independently per isoform.
- 30 randomized trials per isoform; 120 total fits.
- `learning_rate`: log-uniform 0.005-0.2.
- `num_leaves`: integer 7-63.
- `min_child_weight`: log-uniform 0.001-10.
- `subsample`: uniform 0.5-1.0.
- `n_estimators = 2000` fixed.
- Early stopping determines the effective iteration count per isoform.
- `bagging_freq = 1` so `subsample` is active rather than silently ignored.
- `max_depth` left at LightGBM default as a safety rail and not tuned.
- Selection metric: validation ST-RAE only.
- Exact search-trial count and winning configurations are recorded.

The literature-backed rationale for the search-space families was the 2023 Journal of Cheminformatics molecular-GBM study plus LightGBM documentation. These sources were used to justify the search dimensions/ranges, not to claim known CYP-specific optima.

### 2.4 Preregistered challengers for later sessions

The following remain separate experiments against the tuned independent baseline:

1. masked-label multitask model; missing labels masked during training; complete-case training prohibited;
2. potent-end loss weighting;
3. XGBoost only if a specific reason for an alternate GBM implementation emerges.

The multitask drop condition remains: if it does not beat the tuned independent baseline on a per-isoform basis on the same held-out test set, it is dropped and the comparison is recorded in the decision record.

---

## 3. Challenge Source and Repository Provenance

The tutorial repository was cloned and recorded at:

`832ae6d6447b07c2c71059de34d07f1130388247`

The repository README identifies the Hugging Face dataset `openadmet/cyp-challenge-train-test` as the training/test data source. The direct-inhibition track predicts four continuous pIC50 endpoints for 750 blinded compounds, and the official primary metric is macro-averaged ST-RAE.

The repository's evaluation configuration independently confirmed the endpoint names and metric definitions.

The local challenge files were subsequently downloaded and pinned by their local SHA-256 hashes in `artifacts/cyp_baseline/provenance.json`.

Local inputs:

- `data/cyp_challenge/cyp-challenge-TRAIN_inhibition.csv`
- `data/cyp_challenge/cyp-challenge-TEST-BLINDED.csv`

Training table shape: 4,905 rows × 18 columns.

Blind test table shape: 750 rows × 2 columns.

---

## 4. Data Audit

### 4.1 Training schema

The training table contains:

- `Molecule_Name`
- `SMILES`
- four direct-inhibition pIC50 columns
- four upper confidence bounds
- four lower confidence bounds
- four standard-deviation columns

### 4.2 Label availability

Observed direct-inhibition label counts:

| Isoform | Observed labels |
|---|---:|
| CYP1A2 | 1,412 |
| CYP2C9 | 1,285 |
| CYP2D6 | 1,493 |
| CYP3A4 | 2,335 |

Label-count distribution per compound:

| Number of measured isoforms | Compounds |
|---|---:|
| 1 | 3,596 |
| 2 | 1,039 |
| 3 | 229 |
| 4 | 41 |

Only 41 compounds carry all four labels, establishing that the matrix is strongly and unevenly sparse.

Exact observed label combinations were also characterized and preserved in the session record. Single-isoform combinations dominate, with 1,575 CYP3A4-only, 815 CYP2D6-only, 648 CYP1A2-only, and 558 CYP2C9-only compounds.

### 4.3 Identity and molecular-input integrity

- Training `Molecule_Name` uniqueness: 4,905 / 4,905.
- Training SMILES uniqueness: 4,905 / 4,905.
- Blind-test `Molecule_Name` uniqueness: 750 / 750.
- Blind-test SMILES uniqueness: 750 / 750.
- Train/test `Molecule_Name` overlap: 0.
- Train/test SMILES overlap: 0.
- Invalid training SMILES under RDKit: 0 / 4,905.
- Invalid test SMILES under RDKit: 0 / 750.

The exact-identity separation establishes no exact identifier/SMILES overlap; it does not establish broader chemical-space separation.

### 4.4 Confidence-interval consistency

For all four isoforms, observed rows had:

- `high < low`: 0
- `high < center`: 0
- `low > center`: 0

Thus the released uncertainty bounds were internally ordered for every observed endpoint.

---

## 5. Cluster Construction and Frozen Split

### 5.1 Initial allocation attempts

The first two allocation implementations were rejected because they did not satisfy the intended global 70/15/15 split while preserving balanced observed-label fractions.

- First allocator produced 1,952 train / 1,518 validation / 1,435 test compounds.
- Second allocator produced 3,261 train / 822 validation / 822 test compounds.

Neither was frozen.

These intermediate results are retained as provenance; no scientific conclusion was drawn from them.

### 5.2 Accepted global allocation

The final allocator minimized the joint deviation across split sizes and observed-label counts while preserving cluster integrity.

Final split:

| Split | Compounds |
|---|---:|
| Train | 3,433 |
| Validation | 736 |
| Test | 736 |

Cluster structure at Tanimoto >= 0.70:

- 4,789 clusters total.
- 4,718 singleton clusters.
- Mean cluster size: 1.0242.
- Maximum cluster size: 6.

Final observed-label fractions were essentially exactly 70/15/15 for every isoform:

| Isoform | Train | Validation | Test |
|---|---:|---:|---:|
| CYP1A2 | 70.04% | 15.01% | 14.94% |
| CYP2C9 | 70.12% | 14.94% | 14.94% |
| CYP2D6 | 69.99% | 15.00% | 15.00% |
| CYP3A4 | 70.06% | 14.99% | 14.95% |

Cluster-integrity check passed: every cluster occurs in exactly one split.

Frozen split artifacts:

- `artifacts/cyp_baseline/cluster_assignments.csv`
- `artifacts/cyp_baseline/cluster_split_assignments.csv`
- `artifacts/cyp_baseline/molecule_split_assignments.csv`

The downloaded training molecular universe was later verified to match the frozen split molecular universe exactly: 4,905 training rows, 4,905 split assignments, zero missing assignments, zero extra assignments.

---

## 6. Evaluation Instrument: ST-RAE

The official repository implementation of `rae_soft_threshold_absolute_error` was inspected and reused rather than re-derived.

The metric:

- treats predictions inside `[conf_low, conf_high]` as zero soft error;
- penalizes only distance beyond the nearest interval bound;
- normalizes by the corresponding soft-thresholded error of a constant mean predictor.

A local wrapper was implemented in `scripts/st_rae.py`.

### Validation of the implementation

Initial synthetic invariant test used a degenerate denominator and correctly produced NaN under the official definition. The test was corrected rather than changing the metric.

The corrected invariants established:

- in-band predictions produce zero model numerator error;
- above-upper-bound predictions are penalized by distance above the upper bound;
- below-lower-bound predictions are penalized by distance below the lower bound;
- the wrapper exactly matches the official repository implementation.

A LightGBM sklearn callback adapter was then implemented and tested for exact scalar equivalence with the direct ST-RAE wrapper.

The smoke-test callback passed.

### Callback bug and invalid run

An initial 30-trial CYP3A4 search reported ST-RAE = 0.0 for every trial. A direct validation diagnostic showed that this was impossible given the actual data:

- validation n = 350
- mean-predictor ST-RAE = 1.0
- perfect-prediction ST-RAE = 0.0
- confidence-interval width mean = 0.8394

The issue was traced to the custom evaluator being passed through the LightGBM sklearn API with the wrong callable signature. The callback was corrected to the sklearn `(y_true, y_pred)` interface. The all-zero tuning run is classified as **invalid** and was not used for model selection; its artifacts are retained.

---

## 7. ECFP4 Representation

A fixed Morgan radius-2 / 2048-bit representation was implemented in `scripts/ecfp4.py` using the newer `MorganGenerator` API to avoid deprecation warnings.

Representation tests passed:

- expected 2,048-bit dimension;
- binary feature values;
- deterministic identical output for identical SMILES;
- correct batch matrix shape.

This representation is now fixed for the independent baseline and later ECFP4-based challenger work.

---

## 8. LightGBM Environment and Smoke Test

The repository-defined conda environment was installed and verified, including LightGBM, RDKit, and scikit-learn.

A single-isoform CYP3A4 smoke test passed:

- training rows: 1,636
- validation rows: 350
- train/validation identity overlap: 0
- best iteration: 421 / 2,000
- 350 finite validation predictions

This established the complete path:

`frozen split -> ECFP4 -> LightGBM -> official ST-RAE callback -> early stopping`.

A LightGBM API deprecation warning about `eval_set` was subsequently avoided in the production tuner by using the current `eval_X` / `eval_y` form.

---

## 9. Randomized Baseline Tuning

Thirty randomized validation-only trials were run per isoform.

Search dimensions:

- `learning_rate`: log-uniform 0.005-0.2
- `num_leaves`: integer 7-63
- `min_child_weight`: log-uniform 0.001-10
- `subsample`: uniform 0.5-1.0

Fixed:

- `n_estimators = 2000`
- `bagging_freq = 1`
- `max_depth = default`
- early stopping = 50 rounds
- selection = lowest validation ST-RAE

### Validation-frozen winners

| Isoform | Trials | Winner | Validation ST-RAE | Best iteration | Learning rate | Leaves | Min child weight | Subsample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CYP1A2 | 30 | 15 | 0.8878237 | 195 | 0.0074936 | 41 | 0.160065 | 0.530998 |
| CYP2C9 | 30 | 30 | 0.7991427 | 69 | 0.0792151 | 25 | 0.042775 | 0.798518 |
| CYP2D6 | 30 | 14 | 0.8730698 | 108 | 0.0402396 | 61 | 0.121779 | 0.603701 |
| CYP3A4 | 30 | 24 | 0.6547708 | 570 | 0.0593598 | 24 | 0.076603 | 0.913242 |

Tuning artifacts:

- `artifacts/cyp_baseline/tuning/CYP1A2_pIC50_direct_inhibition_trials.csv`
- `artifacts/cyp_baseline/tuning/CYP1A2_pIC50_direct_inhibition_best.json`
- `artifacts/cyp_baseline/tuning/CYP2C9_pIC50_direct_inhibition_trials.csv`
- `artifacts/cyp_baseline/tuning/CYP2C9_pIC50_direct_inhibition_best.json`
- `artifacts/cyp_baseline/tuning/CYP2D6_pIC50_direct_inhibition_trials.csv`
- `artifacts/cyp_baseline/tuning/CYP2D6_pIC50_direct_inhibition_best.json`
- `artifacts/cyp_baseline/tuning/CYP3A4_pIC50_direct_inhibition_trials.csv`
- `artifacts/cyp_baseline/tuning/CYP3A4_pIC50_direct_inhibition_best.json`
- `artifacts/cyp_baseline/frozen_baseline_config.json`

### CYP3A4 provenance exception

CYP3A4 was tuned before the explicit isoform-seed map was introduced. It was the corrected post-callback-bug 30-trial run. The approved decision was **not to rerun CYP3A4**, because there is no preregistered basis for choosing between multiple valid 30-trial searches after the fact.

The existing corrected CYP3A4 result remains authoritative and is explicitly flagged as having distinct seed provenance. No decision amendment was made.

---

## 10. Final Held-Out Internal Test Evaluation

After all four validation configurations were frozen, the 736-compound internal test split was evaluated once. No test-set peeking occurred during tuning.

Per-isoform results:

| Isoform | Test n | ST-RAE | MAE | R2 | Spearman rho | Kendall tau |
|---|---:|---:|---:|---:|---:|---:|
| CYP1A2 | 211 | 0.7955301 | 0.6710210 | 0.2334249 | 0.5554605 | 0.3924622 |
| CYP2C9 | 192 | 0.9486531 | 0.6163640 | 0.0661452 | 0.3023679 | 0.2051702 |
| CYP2D6 | 224 | 0.9378920 | 0.6088487 | 0.1516672 | 0.3706461 | 0.2506857 |
| CYP3A4 | 349 | 0.6739115 | 0.6340493 | 0.4194745 | 0.6590948 | 0.4771676 |
| **Macro** | **976** | **0.8389967** | **0.6325708** | **0.2176779** | **0.4718923** | **0.3313714** |

Final evaluation artifact:

`artifacts/cyp_baseline/test_evaluation/baseline_test_results.json`

Per-isoform prediction artifacts:

- `artifacts/cyp_baseline/test_evaluation/CYP1A2_pIC50_direct_inhibition_predictions.csv`
- `artifacts/cyp_baseline/test_evaluation/CYP2C9_pIC50_direct_inhibition_predictions.csv`
- `artifacts/cyp_baseline/test_evaluation/CYP2D6_pIC50_direct_inhibition_predictions.csv`
- `artifacts/cyp_baseline/test_evaluation/CYP3A4_pIC50_direct_inhibition_predictions.csv`

The macro internal test reference is **MA-ST-RAE = 0.8389967**.

No external leaderboard result was used in this baseline evaluation.

---

## 11. Session Outcomes

### Completed

- Obtained and pinned the OpenADMET CYP challenge data locally.
- Characterized the sparse four-isoform label matrix.
- Verified molecular-input and exact train/test identity integrity.
- Constructed and froze an analog-cluster 70/15/15 internal split.
- Verified and wrapped the official ST-RAE implementation.
- Implemented and verified fixed Morgan radius-2 / 2048-bit ECFP4.
- Installed and verified the challenge-defined LightGBM environment.
- Passed a single-isoform LightGBM smoke test.
- Completed 30 validation-only randomized trials per isoform.
- Frozen the four independent validation configurations.
- Performed one final held-out internal test evaluation.
- Established the independent reference macro test MA-ST-RAE = 0.8389967.

### Not started

- Masked-label multitask challenger.
- Potent-end loss-weighting challenger.
- XGBoost challenger.
- Chemprop.
- AttentiveFP.
- TDI modeling.
- Challenge leaderboard submission.

---

## 12. No New Decision Record

No new `D###` scientific decision record was created during Tripwire-CYP-001.

The session implemented and executed the already-approved Section 5 baseline plan. The tuning-grid/search and CYP3A4 seed-provenance clarifications were implementation-level decisions, not scientific amendments.

---

## 13. Artifacts to Preserve

All intermediate and final artifacts from this session should be preserved, including:

- pinned challenge input CSVs;
- provenance manifest;
- all cluster/split assignment files;
- invalid initial all-zero CYP3A4 tuning artifacts;
- corrected CYP3A4 tuning artifacts;
- all three deterministic-seed tuning artifacts;
- frozen baseline configuration;
- all held-out prediction CSVs;
- final baseline test JSON;
- scientific-core scripts created during the session.

No intermediate artifact should be deleted merely because it is not part of the final baseline.

---

## 14. Closing Status

**CLOSED — Tripwire-CYP-001**

The session establishes the tuned independent ECFP4 + LightGBM reference needed for all subsequent Tripwire CYP challengers.

The next scientific question is the preregistered masked-label multitask challenger: whether shared supervision across the partially observed CYP label matrix improves per-isoform performance over the frozen independent reference.

---

## 15. Next Session — Tripwire-CYP-002

Start from the frozen baseline artifacts above. Do not retune or reopen the independent baseline.

First objective: implement the masked-label multitask challenger with complete-case training explicitly prohibited.

Compare its per-isoform results against the frozen independent baseline using the same held-out test set.

Drop the multitask approach if it does not beat the independent baseline on a per-isoform basis, and capture that comparison in the appropriate decision record.
