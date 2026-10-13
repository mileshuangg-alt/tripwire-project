# D001 — CYP-002 Two-Variant Masked Multitask Chemprop Experiment

**Decision date:** 2026-10-01  
**Status:** Approved and frozen

## Decision

Tripwire-CYP-002 will proceed as a two-variant experiment within the Chemprop/D-MPNN architecture frozen by Rev. 3 of the Tripwire proposal.

**Variant 1 — stock masked multitask Chemprop**

- Chemprop/D-MPNN trained from scratch.
- Four-output multitask CYP regression.
- Missing labels are masked from the loss using stock Chemprop behavior.
- No custom loss modification.

**Variant 2 — task-balanced masked multitask Chemprop**

- Identical Chemprop/D-MPNN architecture, data, preprocessing, frozen splits, tuning protocol, seed policy, and evaluation procedure as Variant 1.
- The only experimental change is the training objective.
- For each CYP task \(t\) within batch \(b\):

\[
L_{b,t}
=
\operatorname{mean}_{i:m_{it}=1}
\ell_{it}
\]

- Overall batch objective:

\[
L_b
=
\frac{1}{4}
\sum_{t=1}^{4}
L_{b,t}
\]

- A task with zero observed labels in a batch contributes zero loss and zero gradient; the batch loss is **not renormalized over the tasks present**.
- Zero-label task/batches are explicitly detected, counted, and flagged in diagnostics.
- Variant 2 is a **custom loss implementation**, not stock Chemprop behavior.
- README and experiment metadata must explicitly state that the architecture remains Rev. 3 Chemprop while the objective departs from stock Chemprop.

## Configuration

- Chemprop version: **2.3.1**
- Architecture: Chemprop/D-MPNN, trained from scratch.
- Train/validation/test split: frozen CYP-001 split of 3,433 / 736 / 736 compounds.
- Complete-case training is prohibited.
- Both variants use the same dataset, preprocessing, representation, tuning protocol, compute constraints, and primary random seed.
- Validation checkpoint selection is based on macro-averaged ST-RAE across CYP1A2, CYP2C9, CYP2D6, and CYP3A4.
- The single checkpoint with the lowest macro validation ST-RAE is retained.
- Early stopping uses:
  - patience = 20 epochs;
  - min_delta = 0.0.
- The held-out test set is not used for tuning or checkpoint selection.
- Primary A/B experiment uses one fixed shared random seed.
- A second seed is not part of the primary A/B experiment.

## Selection rule

After both variant configurations are frozen, each variant is evaluated once on the untouched held-out test partition.

Primary adjudication metric:

\[
\operatorname{Macro\ ST\text{-}RAE}
=
\frac{
\operatorname{ST\text{-}RAE}_{1A2}
+
\operatorname{ST\text{-}RAE}_{2C9}
+
\operatorname{ST\text{-}RAE}_{2D6}
+
\operatorname{ST\text{-}RAE}_{3A4}
}{4}
\]

Variant 2 replaces Variant 1 only if Variant 2 has the lower macro-averaged held-out test ST-RAE.

No post-hoc cutoff, endpoint-specific exception, qualitative override, or new selection criterion may be introduced after observing the test results.

Per-isoform test ST-RAE remains reported alongside the macro value.

Secondary diagnostics may include MAE, R², Spearman correlation, and Kendall tau, but they do not override the preregistered selection rule.

## Baseline context

The CYP-001 independent baseline produced different per-isoform results, including approximately:

- CYP3A4 ST-RAE = 0.655
- CYP1A2 ST-RAE = 0.888

These values, together with the unequal endpoint label counts, are contextual only. Endpoint difficulty, label availability, and endpoint-specific behavior are confounded, so the CYP-001 results are not evidence for or against task balancing.

The Variant 1 versus Variant 2 experiment is the adjudicator of the task-balancing question.

The independent CYP-001 baseline remains frozen and is not retuned.

## Parked alternatives

The following remain explicitly parked and are not part of CYP-002:

- task-specific heads;
- uncertainty weighting;
- GradNorm;
- gradient surgery;
- IM-GNN-style missing-label imputation.

These may be reconsidered only if the completed A/B experiment provides documented evidence of negative transfer, task-imbalance behavior, or another specific failure mode that directly motivates escalation.

Missing-label imputation remains excluded from the current program.

## Artifact and compute requirements

Compute is metered.

Every scientific training run must preserve sufficient provenance to reproduce and audit the result, including:

- configuration;
- variant;
- random seed;
- software/dependency versions;
- hardware;
- runtime;
- validation history;
- checkpoint selection;
- training outcome.

Failed, rejected, diagnostic, and non-selected CYP-002 runs are preserved until the experiment is finalized.

## Governing principle

CYP-002 changes one scientifically interpretable variable at a time:

**the relative weighting of the four partially observed CYP task losses.**

Architecture, data, preprocessing, representation, splits, tuning protocol, validation criterion, checkpoint-selection rule, seed policy, and final adjudication remain fixed.

## Revisit when

Reconsider the parked multitask methods only if the completed Variant 1 versus Variant 2 experiment provides documented evidence of negative transfer, task-imbalance behavior, or another specific failure mode that a parked method directly addresses.

# D002 — CYP-002 Sensitivity Grid and Variant Retention

**Decision date:** 2026-10-05  
**Status:** Approved and frozen

## Decision

The CYP-002 task-balanced masked multitask Chemprop variant is **not retired** following the completed C0 validation comparison.

The frozen CYP-002 contract remains unchanged:

- Variant 1 is stock masked multitask Chemprop.
- Variant 2 is task-balanced masked multitask Chemprop.
- The full eight-configuration grid, C0-C7, is run for **both variants** under the same seed and controls.
- The sensitivity grid is judged using validation macro ST-RAE only.
- The primary sensitivity statistic is \(k/8\), where \(k\) is the number of configurations for which Variant 2 has lower validation macro ST-RAE than Variant 1.
- The grid does not alter the C0 headline A/B selection rule and does not become an iterative configuration-search procedure.
- After configurations and checkpoints are frozen, each headline variant receives one held-out test evaluation.
- Variant 2 replaces Variant 1 only if Variant 2 has the lower held-out test macro ST-RAE.

The earlier statement that task-balanced was retired is withdrawn as unapproved.

The earlier plan to run a stock-only C1-C7 grid is withdrawn as unapproved.

## C0 Diagnostic Result

The completed C0 validation comparison is recorded as a diagnostic result only:

- Stock masked multitask Chemprop: **0.684556823058**
- Task-balanced masked multitask Chemprop: **0.690857884653**

These validation results do **not** adjudicate the headline Variant 1 versus Variant 2 experiment.

The headline selection rule remains the frozen held-out test rule in D001.

The completed C0 runs and their provenance remain preserved.

## Contract Interpretation

The CYP-002 contract specifies that each of the eight configurations is run for both variants under the same seed and controls.

Accordingly, the remaining sensitivity analysis consists of:

```text
C1-C7 × Variant 1
C1-C7 × Variant 2
```

with validation macro ST-RAE used to calculate the \(k/8\) sensitivity statistic.

No C1-C7 configuration may be run for only one variant on the basis of the C0 validation result.

## Withdrawal of Unapproved Interpretation

The following prior statements are withdrawn because they were not supported by an approved decision record:

1. Task-balanced is retired after C0.
2. C1-C7 should be run using stock MSE only.
3. The C0 validation result establishes the headline Variant 1 versus Variant 2 winner.
4. The grid should exclude Variant 2 after C0.

These statements do not modify D001.

---

# Amendment — CYP Arm Evaluation and Split Robustness

**Amendment date:** 2026-10-10  
**Status:** Approved and frozen  
**Source:** Course staff guidance

## 1. Effect-Size and Confidence-Interval Reporting

All model comparisons in the CYP arm are reported as an effect size with a 95% confidence interval from bootstrapping.

This applies to:

- stock vs task-balanced loss;
- shared vs separate encoders;
- two-arm vs inhibition-only.

This is a fixed decision, not an open methodological choice.

## 2. Split Robustness

Split robustness includes both:

- the existing repeated clustered k-fold analysis; and
- Monte Carlo bootstrapped sampling over split realizations.

For the Monte Carlo split-realization analysis, many random cluster splits are drawn, the model is refit for each split realization, and the distribution of held-out performance is reported rather than committing to one k.

The repeated grouped/clustered k-fold analysis remains.

Whenever a fixed k is used, each fold must contain enough of the minority class.

No numeric minimum for minority-class representation is specified by this guidance. No numeric cutoff is introduced by this amendment.

## 3. Relationship to Existing Preregistered Decisions

These decisions sit alongside the existing preregistered decisions:

- one held-out test difference per headline variant;
- within-seed noise from paired training seeds for judging whether an improvement is real.

These existing decisions remain unchanged.

## 4. Relationship to CYP-002

This amendment does not change the CYP-002 variant-retention decision or the C0-C7 two-variant sensitivity grid established above.

The amendment changes the required evaluation/reporting framework for CYP-arm comparisons and split robustness.

The completed CYP-002 C0 held-out comparison already used a bootstrapped 95% confidence interval and is not reopened by this amendment.

## 5. Conflict with Existing Evaluation Plan

The existing Step-3 plan specifies:

> "Step 3 (only after step 2 is recorded): cluster k-fold cross-validation at C0, as an analysis-stage contract amendment for split-robustness reporting."

The staff guidance adds the Monte Carlo cluster-split/refit analysis alongside the repeated clustered k-fold analysis and therefore **amends the scope of Step 3**.

No other CYP-002 decision is changed by this amendment.

## 6. Step-3 Analysis Universe and Protected Evaluation Data

The Step-3 split-robustness analysis is restricted to the compounds assigned to the frozen CYP-001 **train and validation partitions**:

- frozen train partition: **3,433 compounds**;
- frozen validation partition: **736 compounds**;
- total Step-3 universe: **4,169 compounds**.

The frozen **736-compound internal held-out test partition is excluded from Step 3**. It must not enter Step-3 split construction, feasibility analysis, model fitting, refitting, evaluation, or statistical analysis. Its completed one-shot C0 headline evaluations remain frozen and are not reopened.

The separate **750-compound blinded OpenADMET challenge set is also excluded from Step 3** and remains reserved for prospective external evaluation.

This boundary preserves the independence of the already-spent internal held-out test while allowing split robustness to be measured using only the development data that were available for training and validation.

Within the 4,169-compound Step-3 universe, the frozen scaffold/analog cluster is the splitting and resampling unit. Individual compounds within the same cluster must not cross a training/evaluation boundary within a fold or split realization.

Stock and task-balanced C0 always receive identical outer and inner partitions within each paired comparison.

## 7. Pre-Run Step-3 Data Audit

A zero-training feasibility audit was performed before selecting the fixed number of folds.

Identity alignment was verified using `Molecule_Name` across the CYP training dataset, frozen split assignments, and frozen cluster assignments:

- dataset rows: **4,905**;
- unique molecule names: **4,905**;
- missing molecule names: **0**;
- duplicate molecule names: **0**;
- split/cluster assignment disagreements: **0**;
- eligible Step-3 compounds: **4,169**;
- protected internal-test compounds: **736**.

The 4,169-compound Step-3 universe contains **4,111 scaffold/analog clusters**. The largest eligible cluster contains 6 compounds and the median cluster size is 1.

Observed endpoint support in the eligible universe is:

| CYP endpoint | Observed labels | Clusters with at least one observed label |
|---|---:|---:|
| CYP1A2 | 1,201 | 1,197 |
| CYP2C9 | 1,093 | 1,093 |
| CYP2D6 | 1,269 | 1,264 |
| CYP3A4 | 1,986 | 1,940 |

CYP2C9 is therefore the least-observed enzyme endpoint.

No pIC50 target values were used to construct candidate folds and no enzyme-label balancing or stratification was introduced.

### Fixed-k feasibility audit

A zero-training simulation evaluated unconstrained random grouped assignments for `k = 5` and `k = 10`.

For each candidate `k`, **10,000 random whole-cluster assignments** were generated using audit RNG seed `20261010`. Fold construction balanced only approximate compound count; enzyme-label availability and pIC50 values were not consulted during assignment.

For CYP2C9:

| Quantity | k = 5 | k = 10 |
|---|---:|---:|
| Expected labels per fold | 218.6 | 109.3 |
| Median minimum-fold support | 205 | 96 |
| 2.5th percentile of minimum-fold support | 189 | 86 |
| Worst minimum observed across 10,000 assignments | 172 | 74 |
| Realizations containing a zero-label fold | 0 | 0 |

For `k = 10`, no zero-label fold was observed for **any of the four CYP endpoints** across the 10,000 simulated random assignments.

This audit is a feasibility analysis only. No production fold was selected from the audit, and the audit was not used to optimize enzyme-label balance.

## 8. Repeated Clustered 10-Fold Cross-Validation

Step 3 uses **repeated 10-fold clustered cross-validation with 5 repetitions**.

Whole scaffold/analog clusters are randomly assigned to outer folds. Enzyme-label availability and pIC50 values are not used to optimize, stratify, accept, reject, or select fold assignments.

### Justification for k = 10

`k = 10` is selected because:

1. each outer fit trains against approximately 90% of the eligible development universe before its training-side inner-validation allocation, placing the outer resampling regime closer to a full-development-data fit than 5-fold CV;
2. the principal empirical concern with the smaller 10-fold evaluation sets was tested prospectively rather than assumed;
3. the 10,000-assignment feasibility audit demonstrated substantial natural representation of the least-observed CYP endpoint under random cluster assignment, including a median minimum CYP2C9 support of 96 labels, a worst observed minimum of 74 labels, and zero zero-label realizations;
4. this representation was achieved without label stratification or optimization, preserving the purpose of Step 3 as a measurement of sensitivity to naturally varying cluster splits;
5. the separate Monte Carlo arm provides complementary sampling of alternative cluster-level train/evaluation realizations, so the k-fold arm does not need to maximize the size of each individual held-out partition at the expense of training-set fraction.

No numeric minimum for adequate minority-enzyme representation is retrospectively introduced. The feasibility evidence establishes that the selected design does not exhibit the obvious failure mode of absent or extremely sparse enzyme representation that motivated the staff requirement.

### Justification for five repetitions

The complete 10-fold procedure is repeated **5 times**, with independent randomized whole-cluster fold assignments.

Five repetitions are selected to provide multiple complete out-of-fold prediction passes over the entire 4,169-compound development universe and to expose variation caused by alternative cluster assignments.

Five repetitions are a **project-level design choice rather than a claimed literature-derived optimum**. The methodological literature supports repeated resampling to characterize split-dependent variability but does not establish a universally optimal repetition count for this setting.

Compute capacity is not used to reduce the repetition count.

The repeated-CV arm therefore requires:

`5 repetitions × 10 folds × 2 variants = 100 C0 fits`.

Within each repetition, every eligible compound appears in exactly one outer held-out fold.

## 9. Inner Validation for Checkpoint Selection and Early Stopping

Every Step-3 refit uses a **training-side, cluster-disjoint inner-validation partition**.

The outer held-out evaluation fold or Monte Carlo held-out set must never be used for checkpoint selection, early stopping, architecture selection, hyperparameter selection, or any other model-selection operation.

For each outer refit:

1. freeze the outer held-out clusters;
2. operate only on the remaining outer-training clusters;
3. randomly assign whole outer-training clusters to inner training and inner validation;
4. target **17.65% of the outer-training compounds for inner validation**, subject to preservation of whole clusters;
5. use the remaining clusters for gradient-based training;
6. use inner-validation macro ST-RAE for checkpoint selection and early stopping under the existing frozen C0 procedure;
7. evaluate the selected checkpoint exactly once on the outer held-out data.

Stock and task-balanced receive the identical inner partition.

Neither enzyme-label availability nor pIC50 target values may be used to optimize the inner split.

### Justification for the 17.65% inner-validation fraction

The inner-validation fraction is derived from the already-established development-data allocation:

`736 / 4169 = approximately 0.17654`.

The choice therefore preserves the project's existing relative validation allocation rather than introducing a newly tuned validation fraction after observing Step-3 results.

The 17.65% value is a **project-derived continuity choice, not a claimed literature optimum**.

The procedure preserves the scientific function of the existing C0 validation process—macro ST-RAE checkpoint selection and early stopping—while preventing leakage from the outer evaluation set.

No automatic post-selection refit on combined inner-training plus inner-validation data is introduced. Such a refit would require a new rule for transferring the selected stopping point/checkpoint to a model trained on different data and is not part of the frozen C0 training procedure.

Accordingly, Step-3 absolute performance is interpreted as performance of this nested refitting procedure and not as an estimate from a model trained on all 4,169 development compounds.

## 10. Monte Carlo Cluster-Split/Refit Analysis

The second Step-3 arm uses **30 Monte Carlo split realizations**, each targeting an outer held-out fraction of **10%** of the 4,169-compound Step-3 universe while preserving whole scaffold/analog clusters.

For every realization:

1. whole clusters are randomly assigned to outer development and held-out subsets;
2. no enzyme-label stratification or pIC50-based optimization is permitted;
3. the outer held-out fraction targets approximately 10%, subject to cluster integrity;
4. the remaining outer-development clusters undergo the same training-side inner-validation procedure defined above;
5. stock and task-balanced are fit from scratch on identical partitions;
6. the selected checkpoints are evaluated on the identical outer held-out clusters;
7. absolute macro and per-enzyme ST-RAE and the paired stock-versus-task-balanced effect are preserved.

### Justification for the 10% outer holdout

The 10% Monte Carlo holdout is selected to align the approximate outer training/evaluation fraction with the selected 10-fold CV design.

This limits an avoidable confound: differences between the two Step-3 arms should primarily reflect their resampling structures rather than a deliberate change in training-set size.

The Monte Carlo arm remains complementary to repeated k-fold despite the matched fraction. Repeated k-fold provides complete out-of-fold coverage within every repetition, whereas Monte Carlo independently samples random cluster-level held-out realizations and permits compounds or clusters to appear in the held-out set multiple times or not at all across realizations.

The 10% value is therefore a project-specific design choice that serves comparability between the two robustness arms; it is not asserted to be a universally optimal holdout fraction.

### Justification for 30 realizations

Thirty realizations are selected to provide a substantive empirical distribution of split-dependent held-out performance while preserving independent random split generation.

The methodological literature supports repeated split/refit analysis for characterizing sensitivity to training/evaluation composition but does not establish a universally optimal realization count for this setting.

Thirty is therefore a **predeclared project-level characterization depth**, not a claimed statistical optimum and not a claim that extreme distribution tails are estimated precisely.

Compute capacity is not used to reduce the realization count.

The Monte Carlo arm requires:

`30 realizations × 2 variants = 60 C0 fits`.

## 11. Paired Effect Size

For every Step-3 comparison, the paired effect retains the existing CYP-002 sign convention:

`Delta = ST-RAE_stock - ST-RAE_task-balanced`.

Therefore:

- positive `Delta` favors task-balanced;
- negative `Delta` favors stock.

Stock and task-balanced are always evaluated using identical outer and inner partitions within a comparison.

For repeated 10-fold CV, predictions are concatenated across the ten outer held-out folds within a repetition and macro ST-RAE is calculated once from that complete out-of-fold prediction set. Fold-level ST-RAE ratios are not averaged to construct the repetition-level primary effect.

The five repetition-level effects are individually preserved and their distribution is reported.

For Monte Carlo, each realization produces its own paired effect, and the complete 30-realization effect distribution is preserved and reported.

These distributions characterize **split sensitivity** and do not reopen the completed C0 headline adjudication.

## 12. Effect-Size Confidence Intervals

The D002 requirement that CYP model comparisons report an effect size with a **95% bootstrap confidence interval** remains in force.

Step 3 uses a **paired scaffold/analog-cluster bootstrap** with:

- bootstrap draws: **10,000**;
- confidence level: **95%**;
- interval type: **percentile**;
- resampling unit: frozen scaffold/analog cluster;
- pairing: identical resampled cluster information is applied to stock and task-balanced within each comparison.

Individual compounds are not independently bootstrapped.

### Repeated-CV bootstrap

For each bootstrap draw, eligible scaffold/analog clusters are sampled with replacement.

The corresponding preserved out-of-fold observations are used to recompute stock and task-balanced macro ST-RAE within each of the five completed CV repetitions.

A paired effect is calculated for each repetition and the five repetition-level effects are aggregated to produce the bootstrap statistic for that draw.

The resulting 10,000 bootstrap statistics define the 95% percentile interval for the aggregate repeated-CV paired effect.

### Monte Carlo bootstrap

For the Monte Carlo arm, the preserved held-out membership of each realization is retained.

Cluster-aware paired resampling is applied to the held-out prediction information, and stock and task-balanced remain paired within every realization.

The realization-level paired effects are recomputed and aggregated across the 30 realizations to produce the bootstrap statistic for each draw.

Repeated appearances of the same chemical cluster across Monte Carlo realizations must not be treated as independent molecular observations.

### Claim boundary

The Step-3 bootstrap intervals are explicitly interpreted as **conditional out-of-sample prediction bootstrap intervals given the completed fitted models and realized split structure**.

They are not claimed to be unconditional confidence intervals for the complete learning algorithm under repeated sampling of an entirely new molecular population and complete retraining.

This limitation is necessary because repeated cross-validation and Monte Carlo refitting create dependence through overlapping training and evaluation data. Naively treating folds, realizations, or repeated appearances of compounds as independent would understate uncertainty.

The empirical distributions of the five repeated-CV effects and thirty Monte Carlo effects are therefore reported separately from the bootstrap confidence intervals.

This separation preserves two distinct quantities:

1. observed sensitivity of the stock-versus-task-balanced comparison to alternative cluster splits; and
2. cluster-aware uncertainty in the aggregate paired prediction comparison conditional on the completed Step-3 refits.

The bootstrap procedure is a project-specific extension of the already-established paired scaffold-cluster bootstrap principle. It is not claimed to have demonstrated unconditional 95% frequentist coverage for the complete repeated-refitting process.

## 13. Step-3 Execution Size and Scheduling

The approved Step-3 resampling design requires:

| Arm | Design | C0 fits |
|---|---|---:|
| Repeated clustered CV | 5 × 10 folds × 2 variants | 100 |
| Monte Carlo split/refit | 30 realizations × 2 variants | 60 |
| **Total** | | **160** |

Compute capacity is **not a binding constraint** for Step 3. Available execution resources include UCSF CoreHPC through program-level AICD3 access and NVIDIA-sponsored CS230 course compute.

The scientific resampling design must therefore not be reduced merely to save compute.

Before training begins, a scheduling pass must extract observed runtime-per-fit from the preserved C0 and C0-C7 grid artifacts and place the approved 160-fit execution plan against the **2026-11-03 OpenADMET CYP challenge deadline**.

This scheduling pass controls execution planning only and does not reopen the approved resampling size.

Execution order is:

1. completed C0 headline evaluation — already finished and not rerun;
2. repeated clustered 10-fold Step-3 arm;
3. Monte Carlo cluster-split/refit Step-3 arm.

## 14. Step-3 Non-Claims and Prohibitions

Step 3:

- does not reopen the completed C0 stock-versus-task-balanced headline decision;
- does not access or reuse the protected 736-compound internal held-out test for robustness analysis;
- does not access the 750-compound blinded challenge set;
- does not tune C0 architecture or hyperparameters;
- does not optimize folds for enzyme-label balance;
- does not select favorable random splits after observing model performance;
- does not introduce a retrospective numeric minority-enzyme adequacy cutoff;
- does not treat folds, repeated appearances, or Monte Carlo realizations as independent molecular samples;
- does not grant Tripwire candidate-attrition authority.

Its purpose is strictly to characterize whether the stock-versus-task-balanced C0 comparison is robust to alternative chemically grouped development-data splits.

## 15. Step-3 Design Status

The following Step-3 parameters are now **approved and frozen before production model fitting**:

- eligible universe: 4,169 frozen train+validation compounds;
- protected internal held-out test: excluded;
- blinded challenge: excluded;
- splitting unit: frozen scaffold/analog cluster;
- label stratification: none;
- repeated clustered CV: **5 repetitions of 10 folds**;
- inner validation: cluster-disjoint, training-side only, targeting **17.65% of outer-training compounds**;
- Monte Carlo: **30 random cluster-split/refit realizations**;
- Monte Carlo outer holdout: **10% target**;
- stock/task-balanced partition pairing: required;
- effect direction: `ST-RAE_stock - ST-RAE_task-balanced`;
- effect-size reporting: required;
- bootstrap: paired, cluster-aware, **10,000 draws**, **95% percentile interval**;
- empirical split-performance distributions: reported separately from bootstrap confidence intervals;
- bootstrap inference claim: conditional on completed fitted models and realized split structure.

Production split RNG seeds and execution manifests must be declared and recorded before the corresponding production splits are generated. They may not be selected based on resulting model performance.

### Amendment — Step-3 Aggregate Effect Functional

**Amendment date:** 2026-10-11  
**Status:** Approved and frozen

For the Step-3 stock-versus-task-balanced comparison, the aggregate paired effect is the **arithmetic mean of the unit-level paired effects**.

For repeated clustered cross-validation:

- each of the five completed CV repetitions produces one repetition-level paired effect after concatenating the ten outer-fold prediction sets and recomputing macro ST-RAE on the complete out-of-fold prediction set;
- the aggregate repeated-CV effect is the arithmetic mean of those five repetition-level paired effects;
- the same arithmetic-mean aggregation is applied within every bootstrap draw after the five repetition-level paired effects have been recomputed.

For the Monte Carlo arm:

- each of the thirty completed split realizations produces one realization-level paired effect;
- the aggregate Monte Carlo effect is the arithmetic mean of those thirty realization-level paired effects;
- the same arithmetic-mean aggregation is applied within every bootstrap draw after the thirty realization-level paired effects have been recomputed.

The effect retains the frozen sign convention:

`Delta = ST-RAE_stock - ST-RAE_task-balanced`

so positive Delta favors task-balanced and negative Delta favors stock.

#### Rationale

For repeated clustered cross-validation, each completed repetition provides complete and identical out-of-fold coverage of the 4,169-compound Step-3 universe. Equal weighting of the five repetition-level effects therefore follows from the experimental design rather than from post hoc weighting based on observed performance.

The same arithmetic-mean functional is applied consistently to the thirty Monte Carlo realization-level effects and to every corresponding bootstrap draw. The aggregation rule is fixed before inspection of any Step-3 ST-RAE values.

The empirical Monte Carlo split-sensitivity distribution remains a separate reported quantity. All thirty realization-level paired effects are preserved, with at minimum their **mean, range, and standard deviation** reported separately from the cluster-aware bootstrap confidence interval.

This amendment does not change the frozen split realizations, completed fitted models, paired effect definition, bootstrap resampling unit, bootstrap draw count, confidence level, interval type, or inference claim boundary.


### Amendment — Completed Step-3 Split-Robustness Result

**Amendment date:** 2026-10-13  
**Status:** Approved and frozen

#### Decision

The completed Step-3 split-robustness analysis does **not** provide sufficient evidence to replace the stock masked multitask Chemprop loss with the task-balanced masked multitask loss.

The task-balanced objective is the challenger to the existing stock loss. Across both preregistered Step-3 robustness arms, the aggregate paired point estimates are small, similar in magnitude, and in the direction of stock under the frozen sign convention:

`Delta = ST-RAE_stock - ST-RAE_task-balanced`.

The completed results are:

| Step-3 arm | Aggregate Delta | Conditional 95% cluster-bootstrap CI | Direction |
|---|---:|---:|---|
| Repeated clustered CV | -0.005985558047 | [-0.009363336928, -0.002749970389] | stock |
| Monte Carlo split/refit | -0.003777541737 | [-0.008193401963, 0.000964253581] | stock |

The two arms are therefore directionally concordant and quantitatively similar. The fact that the repeated-CV interval excludes zero while the Monte Carlo interval includes zero is **not interpreted as disagreement between the two arms**.

The repeated-CV arm provides the clearer evidence for the modest stock-leaning effect. The Monte Carlo arm produces a concordant point estimate but a conditional interval that includes zero.

The Step-3 conclusion is:

**Stock leans ahead across both split-robustness arms, and repeated clustered CV alone shows that advantage clearly under its conditional cluster-bootstrap interval. Task balancing does not show sufficient robust performance evidence to justify replacing the stock masked multitask loss.**

This result does not claim a large stock advantage, does not establish equality between the variants in the Monte Carlo arm, and does not convert failure of the Monte Carlo interval to exclude zero into evidence that the two arms disagree.

#### Repeated clustered-CV result

The repeated clustered-CV arm consisted of five complete 10-fold repetitions with paired stock and task-balanced fits.

The frozen aggregate paired effect is:

`Delta_CV = -0.0059855580469182264`.

The paired scaffold/analog-cluster bootstrap used 10,000 draws and produced the frozen 95% percentile interval:

`[-0.009363336928, -0.002749970389]`.

The five repetition-level effects remain preserved and reported separately as the split-sensitivity distribution. Their summary is:

- mean Delta: `-0.005985558047`;
- standard deviation: `0.00723007`;
- range: `[-0.01595089, 0.00388128]`;
- direction: 4 of 5 repetitions favor stock and 1 of 5 favors task-balanced.

The canonical repeated-CV bootstrap production job is CoreHPC Slurm job `2208142`. It completed successfully with 10,000/10,000 bootstrap draws, finite bootstrap statistics, reconstructed percentile bounds identical to the persisted result, and no traceback.

#### Monte Carlo split/refit result

The Monte Carlo arm consisted of thirty independently generated frozen cluster-split/refit realizations with paired stock and task-balanced fits on identical partitions.

The frozen aggregate paired effect is:

`Delta_MC = -0.0037775417371858967`.

The shared-global-multinomial paired scaffold/analog-cluster bootstrap used:

- master seed: `20261011`;
- namespace: `CYP003|bootstrap|monte_carlo`;
- global sampling universe: 4,111 frozen scaffold/analog clusters;
- 4,111 cluster draws with replacement per bootstrap draw;
- 10,000 bootstrap draws;
- identical global cluster multiplicity applied everywhere the same cluster appears across the thirty fixed Monte Carlo held-out realizations.

This construction retains each realization's frozen held-out membership while preventing repeated appearances of the same chemical cluster across realizations from being treated as independent molecular observations.

The frozen 95% percentile interval is:

`[-0.008193401963, 0.000964253581]`.

All thirty realization-level effects remain preserved and reported separately as the Monte Carlo split-sensitivity distribution. The aggregate point estimate remains in the stock-favoring direction, while the conditional bootstrap interval includes zero.

The canonical Monte Carlo bootstrap production job is CoreHPC Slurm job `2229761`. It completed successfully in `00:39:05` with 10,000/10,000 bootstrap draws, no traceback, 10,000 finite bootstrap statistics, and 10,000 finite positive denominator diagnostics.

The frozen Monte Carlo production-manifest SHA-256 is:

`2bc39538d9bb52621cb0dc35aa2a18341ec64c7261afab7f449192b4551ee04e`.

#### Monte Carlo bootstrap feasibility and denominator diagnostics

Before Monte Carlo bootstrap implementation, a zero-bootstrap analytic feasibility audit evaluated all 120 realization-endpoint cells under the frozen shared-global-multinomial resampling mechanism.

A project-owned unified failure-probability cutoff of `1e-6` was frozen before the audit result was read. The unified feasibility gate covered zero-label and nonpositive-denominator failure across all 120 realization-endpoint cells and all 10,000 planned bootstrap draws. The analytic audit passed that gate.

The production implementation remained fail-closed: an empty endpoint, nonpositive or nonfinite ST-RAE denominator, or nonfinite score would terminate the run without redraw, skipped draw, substitution, or altered multiplicity.

Positive denominator magnitude was retained as a reporting-only diagnostic with **no cutoff and no independent decision authority**.

For the unperturbed thirty-realization analysis, the minimum denominator across the 240 realization × variant × endpoint scores was:

`18.720925675393943`.

Across all 10,000 production Monte Carlo bootstrap draws, the minimum authoritative ST-RAE denominator encountered was:

`9.12999913229333`.

The production minimum is reported descriptively as the realized denominator diagnostic across the 10,000 bootstrap draws. No denominator cutoff was defined or applied, and this observation does not introduce a retrospective denominator threshold.

#### Interpretation boundary

The five repeated-CV effects and thirty Monte Carlo effects remain the separate empirical split-sensitivity distributions. They characterize variation in the stock-versus-task-balanced comparison across alternative chemically grouped development-data splits.

The bootstrap intervals answer a different question: they characterize cluster-sampling uncertainty in the aggregate paired prediction comparison **conditional on the completed fitted models and realized split structures**.

The intervals do not quantify:

- variance from generating a new set of randomized Monte Carlo split realizations;
- variance from repeating the complete repeated-CV split-generation procedure;
- training-seed variability;
- uncertainty from retraining the complete learning algorithm on a newly sampled molecular population.

The completed Step-3 result therefore does not reopen the completed C0 headline adjudication. Its role is robustness characterization of the stock-versus-task-balanced comparison.

#### Retention consequence

The stock masked multitask loss remains the retained loss function.

The task-balanced masked multitask loss does not replace it because the completed Step-3 robustness evidence does not demonstrate a sufficiently robust performance advantage for the challenger.

This retention decision does not retire or erase the task-balanced results. All completed task-balanced fits, predictions, split-sensitivity effects, bootstrap artifacts, diagnostics, and provenance remain preserved as experimental evidence.


## Governing Principle

CYP-002 changes one scientifically interpretable variable at a time:

**the relative weighting of the four partially observed CYP task losses.**

The purpose of the sensitivity grid is to characterize whether the task-balanced objective changes performance across the preregistered architectural configurations without converting the grid into an implicit optimization procedure.

Architecture, data, preprocessing, representation, splits, tuning protocol, validation criterion, checkpoint-selection rule, seed policy, and final adjudication remain fixed.

## Revisit when

Reconsider the retained two-variant grid only if a subsequent approved decision explicitly amends D001/CYP-002 or the completed sensitivity analysis provides documented evidence requiring a methodological amendment.