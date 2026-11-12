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

# D003 — OpenADMET CYP Submission Non-Adaptivity, Pre-Upload Validation, and Technical-Submission Contract

**Decision date:** 2026-10-13  
**Status:** Approved and frozen

#### Decision

OpenADMET CYP challenge submissions are external evaluation events. Challenge-derived performance information must not become an additional model-development or model-selection dataset.

An early OpenADMET CYP submission is permitted solely as a **technical pipeline-shakeout**. It may establish that the submission pipeline works, but it has no authority to determine which model should be developed, selected, or submitted finally.

Authority is separated as follows:

- **Authorized internal development and validation evidence** determines model development and model selection under the separately approved modeling procedure.
- **The deterministic pre-upload validator** determines whether a proposed submission file is technically eligible for upload.
- **Challenge-derived performance evidence** has zero model-development, model-selection, or technical-defect-investigation authority.
- **The final challenge result** is prospective external evaluation evidence after the final model and entry have been selected and frozen.

If approved, this amendment supersedes Revision 3 only where Revision 3 permits bounded leaderboard-informed CYP development. Revision 3's authorization of an early technical pipeline-shakeout submission remains in force.

#### Submission track and TDI scope

The project submission track is **Regression Prediction**, covering direct-inhibition pIC50 predictions for:

- CYP1A2;
- CYP2C9;
- CYP2D6;
- CYP3A4.

The optional **time-dependent inhibition (TDI) classification track is deliberately deferred**.

TDI is not required for the early pipeline-shakeout submission or the core November 3 CYP deliverable. It may be reconsidered only as separately approved stretch scope and must not jeopardize the direct-inhibition regression submission or required Method Report Link.

This retains Revision 3's distinction between direct-inhibition pIC50 regression as the core track and TDI classification as stretch scope.

#### Prediction-manifest prerequisite

No submission file may enter pre-upload validation until a **frozen prediction manifest** exists for the exact predictions contained in that file.

For every proposed submission, the prediction manifest must identify at minimum:

- model architecture;
- frozen model configuration;
- training seed;
- producing checkpoint path or immutable checkpoint identifier;
- producing checkpoint SHA-256;
- prediction-generation code/version and relevant provenance;
- frozen official challenge-test input identity and SHA-256;
- molecule-identity mapping;
- CYP endpoint mapping;
- raw prediction artifact identity and SHA-256;
- relationship between the raw prediction artifact and the proposed submission file.

The manifest establishes which authentic model outputs are being submitted.

Every upload record must identify the exact prediction manifest and producing checkpoint SHA-256 associated with that entry.

#### Organizer-facing submission identity and form

The live OpenADMET CYP Space requires:

- an existing Hugging Face username; and
- a prediction file.

The Space checks the supplied Hugging Face username for existence. Login through the submission form is not required.

The project must use **one Hugging Face account for the team** throughout challenge participation. Alternative accounts may not be used to circumvent the one-team-entry rule, submission rate limit, or replacement mechanics.

The track selected for this project is:

**Regression Prediction.**

The Hugging Face username used for each upload must be recorded as submission provenance.

#### Accepted prediction-file contract

For the Regression Prediction track, the live Space accepts `.csv` or `.parquet`.

The proposed upload must contain:

- exactly **750 data rows**;
- exactly **six columns**;
- the two required molecular-identity columns;
- all four direct-inhibition prediction columns;
- finite prediction values for all four direct-inhibition endpoints;
- the organizer-specified **case-sensitive** column names.

The project's required six-column schema and order are:

1. `SMILES`
2. `Molecule_Name`
3. `CYP1A2_pIC50_direct_inhibition`
4. `CYP2C9_pIC50_direct_inhibition`
5. `CYP2D6_pIC50_direct_inhibition`
6. `CYP3A4_pIC50_direct_inhibition`

The organizer's published direct-inhibition submission example and current live-Space/FAQ requirements are the external specification references.

The project intentionally imposes stricter identity and ordering requirements than the live Space handler where necessary to prevent technically accepted but scientifically misaligned submissions.

#### Exact submission identity and alignment contract

Every proposed Regression Prediction upload must be validated against three frozen artifacts:

1. the **official organizer-provided 750-compound challenge test file**, which defines the authoritative submission-row universe and row ordering;
2. the **frozen prediction manifest and referenced raw prediction artifact**, which define which prediction belongs to each molecule and CYP endpoint;
3. the **proposed submission file**, which is the exact file intended for upload.

##### Molecule identity

For every row position \(i=1,\ldots,750\):

- `Molecule_Name` in the proposed submission must exactly equal `Molecule_Name` in row \(i\) of the frozen official challenge test file;
- `SMILES` in the proposed submission must exactly equal `SMILES` in row \(i\) of the frozen official challenge test file as the organizer-provided string.

No canonicalization, normalization, salt handling, fragment selection, neutralization, tautomer conversion, stereochemistry manipulation, or other molecular transformation may be used to make mismatching identity strings agree.

The complete ordered `Molecule_Name` vector and complete ordered `SMILES` vector must each be exactly identical between the proposed submission and the frozen official test input.

The validator must additionally require:

- exactly 750 `Molecule_Name` values;
- no missing `Molecule_Name`;
- no missing challenge row;
- no additional challenge row;
- no substituted challenge identity;
- no duplicated row introduced relative to the official test file;
- no row reordering relative to the official test file.

##### Prediction identity

For every molecule, each submitted prediction must equal the authentic prediction assigned to that same `Molecule_Name` and endpoint in the frozen raw prediction artifact referenced by the prediction manifest.

The endpoint mappings are exactly:

- `CYP1A2_pIC50_direct_inhibition` → CYP1A2 direct-inhibition prediction;
- `CYP2C9_pIC50_direct_inhibition` → CYP2C9 direct-inhibition prediction;
- `CYP2D6_pIC50_direct_inhibition` → CYP2D6 direct-inhibition prediction;
- `CYP3A4_pIC50_direct_inhibition` → CYP3A4 direct-inhibition prediction.

Prediction alignment must first be compared **by molecule identity and endpoint**, not merely by row position.

After identity-based prediction alignment passes, the validator must independently verify that the proposed submission rows occur in the exact official test-file order.

All **3,000 predictions**—750 molecules × four endpoints—must be finite numeric values.

##### Identity/alignment PASS condition

Identity and alignment validation passes only if all required comparisons succeed:

- 750/750 ordered `Molecule_Name` matches;
- 750/750 ordered organizer-provided `SMILES` matches;
- exact six case-sensitive column names;
- exact project-required column order;
- 3,000/3,000 molecule × endpoint prediction associations match the frozen prediction artifact;
- 3,000/3,000 prediction values are finite;
- zero missing challenge rows;
- zero additional challenge rows;
- zero substituted identities;
- zero introduced duplicate rows;
- zero reordered challenge rows.

The validation artifact must record these counts explicitly rather than only emitting a generic PASS.

##### Identity/alignment mismatch behavior

Any mismatch causes **pre-upload validation failure**.

On failure:

- no upload is authorized;
- the proposed submission file is preserved unchanged;
- the mismatch is recorded with row position, field or endpoint, expected identity/value relationship, and observed identity/value relationship as applicable;
- the validator must not repair the file automatically;
- the failure may trigger a technical investigation because it arose from permitted pre-upload technical evidence;
- any correction must satisfy the verified-technical-defect carve-out below;
- a corrected file must undergo a completely new validation run and receive its own PASS artifact before upload.

Authentic prediction values may never be modified merely to obtain challenge acceptance.

#### Mandatory pre-upload validation

Every proposed upload—including an early pipeline-shakeout submission—must pass deterministic pre-upload validation.

The validator must compare the proposed submission against:

1. the current organizer submission specification and live-Space/FAQ requirements;
2. the organizer's published direct-inhibition submission example/schema;
3. the frozen official 750-compound challenge test input;
4. the frozen prediction manifest and referenced raw prediction artifact.

The validator must verify at minimum:

- permitted file type;
- exact row count;
- exact column count;
- exact case-sensitive headers;
- exact project-required column order;
- exact test-set molecule identities;
- exact organizer-provided SMILES strings;
- exact official test-file row ordering;
- no missing, additional, substituted, or introduced duplicate challenge identities;
- correct prediction-to-molecule mapping;
- correct prediction-to-CYP-endpoint mapping;
- 3,000 finite regression predictions;
- consistency with the frozen raw prediction artifact;
- consistency with the producing checkpoint and prediction manifest;
- frozen-input and candidate-output hashes.

The live Space handler's acceptance is **not** a substitute for these checks.

The current Space handler does not enforce exact column order or verify row identity against the official challenge test input. The project pre-upload validator is therefore the mandatory control for both properties.

A submission file that fails validation is not authorized for upload.

The validator must produce and preserve a PASS/FAIL artifact identifying:

- organizer-specification references;
- frozen official test-input identity and hash;
- prediction-manifest identity;
- producing checkpoint hash;
- individual validation results and counts;
- SHA-256 of the exact proposed submission file.

Only the exact file whose hash received PASS may be uploaded.

#### Near-constant prediction rejection

The live Space handler rejects near-constant predictions when prediction standard deviation is below `0.01`.

This challenge-side validation rule must never become an optimization target.

If authentic frozen model outputs fail this requirement:

- stop the submission attempt;
- preserve and report the failure;
- do not add noise;
- do not rescale predictions merely to increase their variance;
- do not perturb predictions;
- do not transform predictions merely to satisfy the handler;
- do not substitute another model on the basis of the challenge-side rejection alone.

Predictions must **never be modified merely to pass the near-constant-output check**.

Any scientific response to genuinely degenerate model output requires an independently justified development decision based on authorized internal evidence.

#### Unified definition of prohibited challenge-performance evidence

For this decision, **prohibited challenge-performance evidence** means any information whose content depends on submitted predictions' performance against blinded challenge outcomes.

This includes:

- live leaderboard score;
- live leaderboard rank;
- relative leaderboard position;
- performance tier;
- per-isoform challenge performance, if exposed;
- aggregate challenge performance;
- changes in score, rank, tier, or relative position after a submission;
- comparisons with another team's challenge performance;
- any other feedback whose information content depends on prediction performance against blinded challenge outcomes.

Prohibited challenge-performance evidence has **zero authority** to:

- select or reject a model;
- choose Chemprop versus AttentiveFP;
- choose model architecture;
- choose hyperparameters;
- choose regularization;
- choose preprocessing;
- choose calibration;
- choose a training seed;
- choose a checkpoint;
- choose ensemble membership;
- choose ensemble weights;
- choose the final-training procedure;
- continue, stop, prioritize, deprioritize, or reopen a modeling branch;
- trigger investigation of a possible submission defect;
- support or strengthen a hypothesis that a submitted file is technically defective;
- determine or influence how a technical defect is corrected.

A suspiciously high, low, improved, worsened, or otherwise unexpected challenge result therefore **cannot even trigger inspection of the submission artifacts for a technical defect**.

This definition applies everywhere in this decision whenever **prohibited challenge-performance evidence** is referenced.

Organizer-generated technical validation/error information is not prohibited challenge-performance evidence when it reports only file validation, schema, account, rate-limit, upload, processing, receipt, or other technical submission state and contains no blinded-performance information.

#### Early-submission non-adaptivity firewall

After any challenge upload, prohibited challenge-performance evidence must not enter:

- model development;
- model selection;
- final-entry selection;
- submission-defect investigation;
- technical correction.

Such information may be preserved as external-evaluation provenance but remains outside development and selection authority.

Technical submission-state information—such as whether an upload was received, whether validation succeeded, or whether the organizer generated a technical error—is governed by the technical-defect provisions below and is not prohibited merely because it originates from the organizers.

#### Narrow verified-technical-defect carve-out

A technical investigation or correction may be triggered only by **permitted technical evidence**:

- the proposed or submitted file itself;
- its frozen prediction manifest;
- its frozen raw prediction artifact;
- the frozen official 750-compound challenge test input;
- the current organizer submission specification;
- the project's preserved pre-upload validator output; or
- an automated organizer technical validation/error message, including technical validation/error logs posted in Discord `#cyp-challenge-submissions`.

Organizer-generated technical error logs are permitted to trigger investigation because they report submission validation or processing failures rather than blinded model performance.

The technical evidence must identify, or permit independent demonstration of, a concrete submission defect.

Permitted defects include, for example:

- incorrect row identity;
- incorrect row ordering;
- incorrect target association;
- incorrect target ordering;
- incorrect headers;
- incorrect serialization;
- incorrect file schema or formatting;
- concrete validation or processing failures identified by the organizer's technical submission system.

**Prohibited challenge-performance evidence cannot trigger an investigation, constitute evidence for a defect, strengthen a defect hypothesis, or influence its correction.**

Any correction must be restricted to repairing the independently verified technical defect.

Authentic scientific prediction values may not be altered. Prediction values may change position only when necessary to restore their already-established association with the correct molecule or endpoint. Serialization may change only as necessary to satisfy the independently verified technical specification.

The following are not technical corrections:

- retraining;
- model reselection;
- checkpoint reselection;
- recalibration;
- ensembling;
- prediction adjustment;
- score-driven transformation;
- replacement by another model.

The original file, permitted technical evidence, any organizer technical error log, original prediction manifest, raw prediction artifact, corrected file, exact correction, validator artifacts, hashes, and complete submission history must all be preserved.

#### Submission rate, account, and replacement contract

The live Space permits **one submission per 12 hours**.

Every upload therefore consumes a 12-hour submission window.

Submission attempts must be planned so that at least 12 hours separate uploads, including the final replacement submission.

The project must use **one Hugging Face account per team**.

Alternative accounts may not be used to bypass the submission rate limit or one-entry rule.

Only the **latest valid submission** counts as the active team entry.

#### Submission receipt and technical acceptance verification

The immediate on-screen **“received”** message is not sufficient evidence that the entry has been accepted as a valid challenge submission.

After upload, the project must verify the automated response in Discord channel `#cyp-challenge-submissions`, expected within approximately two hours.

That response may consist of:

- an automated receipt confirming successful submission processing; or
- an automated technical validation/error log identifying a submission-processing defect.

A technical validation/error log is permitted evidence under the narrow technical-defect carve-out and has no model-performance or model-selection authority.

The submission record must preserve:

- submission timestamp;
- Hugging Face username;
- selected track;
- submitted-file SHA-256;
- prediction-manifest identity;
- producing checkpoint SHA-256;
- pre-upload validator PASS artifact;
- on-screen acknowledgement, if retained;
- automated Discord receipt or technical error log;
- Discord response timestamp.

Failure to obtain the expected automated response is a **submission-state problem**, not evidence about model quality.

Investigation of receipt or transport status may inspect permitted technical evidence but may not use prohibited challenge-performance evidence.

#### Method Report Link and final-leaderboard eligibility

The comprehensive model report is a separate mandatory final-eligibility deliverable.

For the live Space, the report is supplied through a **Method Report Link**: a reachable `https://` URL entered in the submission form.

The final report must therefore be hosted at an accessible HTTPS URL before the challenge deadline.

Revision 3 records:

> “A missing model report excludes the entry from the final leaderboard.”

The final challenge deadline is:

**November 3, 2026, 23:59 UTC = Tuesday, November 3, 2026, 3:59 PM Pacific (PST).**

Revision 3 records:

> “In my America/Los_Angeles timezone, that is **Tuesday, November 3 at 3:59 PM PST**”

For final-leaderboard eligibility, the final active prediction entry and reachable Method Report Link must satisfy the organizer's requirements by that deadline.

The Method Report Link is a separate eligibility requirement from prediction-file validation. A technically valid prediction file alone does not establish final-leaderboard eligibility.

#### First-upload declarations reserved for user decision

Before the first upload, the following live-Space form choices remain explicitly reserved for user decision:

- whether the entry is anonymous;
- if anonymous, the alias to use;
- the proprietary-data declaration/checkbox;
- the open-source declaration/checkbox.

The submission pipeline must not infer or select these values automatically.

The selected values must be recorded as part of submission provenance.

#### Scope of an early pipeline-shakeout submission

An early submission may establish only that:

- a frozen prediction manifest exists;
- the exact candidate submission file passes the project validator;
- the selected Hugging Face username is recognized;
- the Regression Prediction track and required form fields function;
- the Space accepts the technical upload;
- the organizer's automated receipt or technical validation response is generated;
- the complete pipeline can carry an internally selected model's authentic predictions through to a technically valid external entry.

An early submission does **not** establish that the submitted model:

- is preferable to another model;
- is competitive;
- is sufficiently accurate;
- should be retained;
- should replace another model;
- should become the final challenge entry.

Any score, rank, tier, or other prohibited challenge-performance evidence produced by that submission remains outside development and selection authority.

#### Relationship to Revision 3

Revision 3 authorizes an early pipeline-shakeout submission and states that live leaderboard feedback may be used as a bounded development signal.

This amendment, if approved, **narrows that authority**.

The early technical pipeline-shakeout remains authorized, but challenge-derived performance information loses all model-development, model-selection, and technical-defect-investigation authority.

Revision 3's other challenge requirements—including direct-inhibition regression priority, TDI as stretch scope, one team identity, final submission, and comprehensive model report—remain unchanged except where this amendment specifies stricter technical controls.

#### Governing principle

The blind CYP challenge is preserved as prospective external evidence by preventing challenge-derived performance information from becoming another tuning or model-selection dataset.

The project separates three functions:

**Internal evidence chooses the model.**

**The pre-upload validator proves the file.**

**The challenge evaluates the frozen entry.**

Technical defects may be investigated and corrected only from independently permitted technical evidence. Blinded challenge performance cannot create, trigger, strengthen, or resolve a technical-defect hypothesis.

#### Revisit when

Revisit this submission contract only if:

- the organizers change the live-Space submission specification, validation behavior, rate limit, track structure, account requirements, Method Report Link requirements, or deadline;
- the official 750-compound test input or required submission schema changes;
- an organizer-generated technical error reveals a previously undocumented submission requirement; or
- a separately approved decision explicitly amends the non-adaptivity firewall.

Any such change must be documented before it affects a subsequent submission.

# D004 — C5/C6 Stock-Chemprop Seed-Replication Decision

**Decision date:** 2026-10-13  
**Status:** Approved and frozen

## Decision

The project will perform a validation-only seed-replication experiment to determine whether the observed validation advantages of the **C5** and **C6** stock-loss Chemprop configurations over stock C0 persist across training-seed stochasticity.

This experiment does not reopen the completed stock-versus-task-balanced loss decision. Stock masked multitask MSE remains the retained loss function.

The only decision this experiment may change is whether **C5 and/or C6 replaces C0 as the stock-loss Chemprop candidate carried forward for downstream comparison with AttentiveFP and possible final model selection**.

It cannot alter D001, D002, CYP-003/Step-3, the retained stock-loss decision, or D003.

## Configurations under replication

The relevant existing configurations are:

- **C0:** message-passing depth 3; message hidden dimension 300; FFN hidden dimension 300; one FFN layer; dropout 0.0; maximum learning rate 0.001.
- **C5:** message-passing depth 3; message hidden dimension 300; FFN hidden dimension 300; two FFN layers; dropout 0.10; maximum learning rate 0.001.
- **C6:** message-passing depth 3; message hidden dimension 500; FFN hidden dimension 500; two FFN layers; dropout 0.10; maximum learning rate 0.001.

The original seed-`20261001` validation macro ST-RAE results were:

- C0 stock: `0.684556823058`
- C5 stock: `0.672421`
- C6 stock: `0.670208`

These observations motivate replication but do not by themselves establish that C5 or C6 should replace C0.

## Seeds and existing controls

Use exactly the five training seeds previously prespecified for CYP-002:

```text
20261001
20261002
20261003
20261004
20261005
```

These are inherited directly from the frozen CYP-002 five-seed replication protocol.

The preserved C5 and C6 seed-`20261001` runs are reused and are not rerun unless an integrity failure invalidates an existing artifact.

The existing stock-C0 runs at all five seeds are likewise reused.

Therefore, the experiment requires:

- four new C5 stock fits at seeds `20261002`–`20261005`;
- four new C6 stock fits at seeds `20261002`–`20261005`;
- **eight new scientific training fits total**.

No new C0 training is required.

## Same-seed paired comparison

For each candidate \(j\in\{C5,C6\}\) and training seed \(s\), define the validation advantage:

\[
A_{j,s}
=
ST\!-\!RAE_{C0,s}
-
ST\!-\!RAE_{j,s}.
\]

Lower ST-RAE is better. Therefore:

- \(A_{j,s}>0\): candidate \(j\) beats C0 at that seed;
- \(A_{j,s}=0\): tie;
- \(A_{j,s}<0\): C0 beats candidate \(j\).

Every candidate result is compared with the **existing C0 result generated at the same training seed**.

The experiment does not substitute the five-seed C0 mean for these same-seed controls.

The authoritative existing C0 validation values are:

```text
20261001    0.684556823057850
20261002    0.680145541646839
20261003    0.690064499896603
20261004    0.696088668668752
20261005    0.676340834323393
```

## Required reporting

For C5 versus C0 and C6 versus C0 separately, report:

- all five paired advantages \(A_{j,s}\);
- mean paired advantage;
- median paired advantage;
- sample SD of the five paired advantages;
- number of seeds favoring the candidate;
- number of seeds favoring C0;
- five-seed candidate mean validation macro ST-RAE;
- five-seed C0 mean validation macro ST-RAE;
- per-isoform validation ST-RAE;
- paired 95% bootstrap confidence interval for the mean validation advantage.

The approved CYP-arm evaluation amendment requires model comparisons to report an effect size with a 95% bootstrap confidence interval.

## Paired-seed bootstrap

For each candidate separately, the bootstrap unit is the **paired training seed**.

The observed five-element vector is:

\[
(A_{j,1},A_{j,2},A_{j,3},A_{j,4},A_{j,5}).
\]

For each bootstrap draw, sample **five seed indices with replacement** from the fixed five-seed set.

Whenever a seed index is sampled, its already-computed paired advantage \(A_{j,s}\) is carried as one indivisible observation. C0 and candidate scores are never resampled independently.

For bootstrap draw \(b\):

\[
\bar A_j^{*(b)}
=
\frac{1}{5}
\sum_{r=1}^{5}
A_{j,s_r^{*(b)}}.
\]

Use:

```text
bootstrap draws       10,000
sampling              paired seeds with replacement
observations/draw     5
statistic             mean paired validation advantage
confidence interval   95% percentile
```

The analysis RNG seed must be fixed prospectively before the bootstrap is executed, must be distinct from the five training seeds, and must be recorded with the analysis artifacts.

The percentile interval is:

\[
[
Q_{0.025}(\bar A_j^*),
Q_{0.975}(\bar A_j^*)
].
\]

Define:

\[
LCB(A_j)=Q_{0.025}(\bar A_j^*).
\]

The use of 10,000 paired bootstrap draws and a 95% percentile interval follows the already-frozen CYP-002 bootstrap precedent; the experimental unit here is changed appropriately to the five paired training-seed replicates.

## Candidate-specific seed variability

For candidate \(j\), define:

\[
SD_{\mathrm{seed},j}
=
SD(A_{j,1},A_{j,2},A_{j,3},A_{j,4},A_{j,5}).
\]

This is the sample SD of the **C0-versus-candidate paired validation advantages**.

The previously frozen numerical value

```text
0.007094968910757
```

is not reused as the C5/C6 threshold because it was specifically measured from the stock-versus-task-balanced contrast.

The prior CYP-002 preregistration established the methodological principle:

\[
LCB(\Delta_{\mathrm{test}})>SD_{\mathrm{seed}}
\]

for an observed improvement to displace the incumbent.

D004 retains that principle while estimating the stochastic scale from the correct C0-versus-C5 or C0-versus-C6 contrast.

## Frozen decision rule

C5 displaces C0 only if:

\[
\boxed{
LCB(A_{C5})>SD_{\mathrm{seed},C5}
}
\]

C6 displaces C0 only if:

\[
\boxed{
LCB(A_{C6})>SD_{\mathrm{seed},C6}
}
\]

Otherwise that candidate does not displace C0.

Failure to clear this rule means **insufficient evidence to replace C0**. It does not establish equivalence and does not establish that the candidate is intrinsically worse.

C5 and C6 are adjudicated independently.

If neither clears the rule, C0 remains the stock-loss Chemprop candidate.

If exactly one clears the rule, that configuration replaces C0 as the stock-loss Chemprop candidate.

If both clear the rule, D004 does **not** select between C5 and C6. A C5-versus-C6 choice would require a separately prospectively specified selection rule before either can be declared the final optimized Chemprop candidate.

## Conservatism at five seeds

The frozen rule is deliberately conservative at \(n=5\).

Only five paired training-seed effects are observed. Consequently:

- the empirical seed distribution is sparsely sampled;
- the sample SD is itself uncertain;
- the paired-seed bootstrap can resample only the five observed effects and cannot characterize behavior at unobserved training seeds;
- the 2.5th-percentile lower bound must nevertheless exceed an entire sample SD before the incumbent is displaced.

Thus the rule requires substantially more than a positive mean advantage or a majority of favorable seeds. It requires the lower tail of the estimated mean-effect distribution to remain larger than the observed scale of between-seed variation.

The CYP-002 record already limits interpretation of five-seed variability accordingly:

> “The five-seed `SD_seed` is an empirical sample SD across the five prespecified seeds, not an exhaustive characterization of all possible stochastic variation.”

D004 makes the same limitation for the C0-versus-C5 and C0-versus-C6 contrasts.

## Scientific interpretation boundary

D004 evaluates **whole configurations**.

It is not a factorial hyperparameter-ablation experiment.

C5 changes both:

- FFN depth: 1 → 2;
- dropout: 0.0 → 0.10.

Therefore, even if C5 reproducibly beats C0, D004 cannot establish whether the improvement is caused by:

- additional FFN depth;
- dropout;
- their interaction;
- or another consequence of the joint configuration change.

C6 makes the same FFN-depth and dropout changes and additionally changes:

- message hidden dimension: 300 → 500;
- FFN hidden dimension: 300 → 500.

Therefore C6 versus C0 cannot separately identify the effects of width, FFN depth, dropout, or their interactions.

C6 versus C5 also jointly changes both message width and FFN width and therefore cannot isolate either width effect.

The authorized claim is limited to whether **C5 and/or C6 as complete configurations reproducibly improves validation performance over C0 under the five prespecified training seeds**.

## Evaluation boundary

D004 is **validation-only**.

The 736-compound held-out internal-test partition must not be accessed for this experiment.

The 750-compound blinded OpenADMET challenge set and all D003-prohibited challenge-performance evidence must not enter fitting, evaluation, selection, interpretation, or investigation of the D004 scientific result.

## Runtime qualification and Slurm gate

Historical production timings on the original `gpu-dev1` NVIDIA L40S were:

- C5 stock: **82.06 seconds per fit**;
- C6 stock: **66.96 seconds per fit**. 

Using those configuration-specific historical measurements:

- four new C5 fits extrapolate to approximately **5.47 minutes**;
- four new C6 fits extrapolate to approximately **4.46 minutes**;
- eight new fits extrapolate to approximately **9.93 minutes of serial scientific training** under the historical L40S environment.

These historical measurements do not qualify CoreHPC runtime.

Before the Slurm production gate, execution time must be qualified with **one direct interactive benchmark run on the actual target CoreHPC GPU and runtime environment**.

The CoreHPC benchmark must **never be submitted as a Slurm job**.

Its observed wall time is used only to:

- estimate the eight-fit CoreHPC production runtime;
- select an appropriate Slurm wall-time request;
- verify that the intended production execution is operationally reasonable.

The interactive benchmark has no additional model-selection authority.

Its scientific output must not be used as an extra replicate or additional evidence unless it is itself one of the prespecified D004 seed/configuration fits executed under the frozen scientific contract.

Full Slurm production is authorized only after the direct interactive benchmark is recorded and the extrapolated production requirement is reviewed.

## Governing principle

D004 asks one narrow question:

**Do the C5 and/or C6 stock-loss Chemprop configurations retain enough of their observed validation advantage over C0 across the same five training seeds to justify replacing C0 as the downstream Chemprop candidate?**

The experiment measures configuration-level reproducibility, not individual hyperparameter causality.

The incumbent changes only when the candidate's paired validation effect remains large relative to its own observed training-seed variability under the prospectively frozen rule.

## Revisit when

Revisit D004 only if:

- an integrity failure invalidates an existing C0, C5, or C6 artifact used by the experiment;
- the frozen validation split, ST-RAE implementation, Chemprop implementation, or scientific configuration changes before execution;
- the direct CoreHPC benchmark identifies an execution incompatibility requiring a scientific rather than mechanical change;
- both C5 and C6 clear the frozen rule and a separate prospective C5-versus-C6 selection decision is therefore required; or
- a separately approved decision explicitly supersedes this experiment.

# D005 — Bounded Post-D004 Chemprop Search

**Decision date:** October 14, 2026  
**Status:** APPROVED AND FROZEN

## Objective and scope

D005 is a prospective, validation-only search around C5 using **stock masked multitask MSE**.

D004 remains closed and unmodified. C0 remains the baseline of record unless a D005-eligible configuration clears the D005 rule.

Only two factors may vary:

- dropout: `0.00` or `0.10`
- FFN layers: `1` or `2`

Everything else remains frozen at the shared C0/C5 settings: message-passing depth 3, message width 300, FFN width 300, max LR 0.001, LR schedule, batch size, epoch limit, early stopping, preprocessing, validation metric, and other training machinery.

No width, message-passing-depth, learning-rate, or weight-decay search is authorized.

The protected 736-compound internal test and 750-compound blinded challenge remain untouched.

## Grid and seeds

| Configuration | Dropout | FFN layers | Evidence |
|---|---:|---:|---|
| C0 | 0.00 | 1 | existing baseline |
| C5 | 0.10 | 2 | existing five-seed D004 results |
| D005-A | 0.00 | 2 | 5 new fits |
| D005-B | 0.10 | 1 | 5 new fits |

Seeds are exactly:

`20261001, 20261002, 20261003, 20261004, 20261005`

Therefore **10 new fits** are required. C0 and C5 are not rerun.

C5 is eligible to replace C0 under D005. Its D004 five-seed results are **reused under this new prospective D005 rule**. D004 itself remains closed and unmodified, but C5's evidence is therefore **not fresh D005-generated data**. This selection-informed reuse must accompany interpretation of any C5 D005 result.

Before execution, manifests must verify that:

- D005-A differs from C5 in **exactly one** searched setting: dropout `0.10 -> 0.00`;
- D005-B differs from C5 in **exactly one** searched setting: FFN layers `2 -> 1`;
- all other scientific settings equal C5 exactly.

## Runtime and execution

Interactive `gpu-dev1` or interactive CoreHPC only.

Historical C5 fits were approximately 67–91 seconds. Use approximately **90 seconds per new fit** for planning.

Ten new fits therefore require approximately:

`10 × 90 s = 900 s = 15 minutes`

Reserve approximately 20 minutes including manifest and integrity checks. Record actual wall time for every fit.

## D005 selection rule

For candidate `j` and seed `s`:

`A(j,s) = ST-RAE(C0,s) - ST-RAE(j,s)`

Positive values favor the candidate.

The eligible family is exactly:

`{C5 vs C0, D005-A vs C0, D005-B vs C0}`

Use a **0.05 one-sided familywise error budget**. This is the overall Type-I error budget for the three predeclared candidate-vs-C0 claims; using 0.025 familywise would impose additional conservatism for which D005 has no separate justification.

Bonferroni arithmetic:

`per-comparison alpha = 0.05 / 3 = 0.0166666667`

Therefore, for each candidate, use the **1.66666667th percentile** of its paired-seed bootstrap distribution as the multiplicity-adjusted lower confidence bound.

For each candidate:

- bootstrap unit = paired training seed;
- sample five seed indices with replacement;
- preserve C0/candidate pairing;
- statistic = mean paired advantage;
- 10,000 draws;
- analysis RNG seed fixed prospectively before execution.

**Displacement criterion:**

`adjusted LCB(A_j) > 0`

This is deliberately **looser than D004's `LCB > SD_seed,j` rule**. It is acceptable here only because D005 asks the narrower model-selection question of whether a predeclared candidate shows a multiplicity-controlled positive validation advantage over C0. D005 does **not** interpret `LCB > 0` as demonstrating an improvement larger than seed variability or as establishing practical materiality.

If no candidate clears, retain C0.

If exactly one clears, that candidate replaces C0.

If multiple candidates clear, select the candidate with the **largest multiplicity-adjusted LCB**. If the largest adjusted LCB is exactly tied, retain C0 rather than introduce an unregistered tie-breaker.

## Failure rule

For any failed or interrupted new fit:

1. preserve the failed attempt and its artifacts;
2. rerun the **same configuration and same seed** under the frozen D005 settings in a new run directory;
3. use the successful rerun as that seed's evidence.

No alternate seed, configuration, imputation, neighboring run, or other substitution is permitted.

## Interpretation limits

The 2×2 grid can compare dropout at each tested FFN depth, FFN depth at each tested dropout, and describe their interaction within these four settings.

It cannot identify optimal values outside dropout `{0.00, 0.10}` or FFN layers `{1, 2}`, and cannot separate or estimate effects of width, message-passing depth, learning rate, weight decay, or other frozen parameters.

D005 is **selection-informed**: C5 motivated this local search, and C5 itself reuses already-observed D004 data. Multiplicity correction addresses the three formal candidate-vs-C0 comparisons but does not erase that historical selection process. Claims must therefore remain bounded to this post-D004 validation search.

### Bootstrap RNG implementation

Before bootstrap execution, the deterministic implementation is frozen as follows.

NumPy's Generator API is initialized exactly once:

`rng = np.random.default_rng(20261015)`

Candidates consume that single RNG stream in this fixed order:

`C5`, `D005-A`, `D005-B`

For each candidate, exactly one index matrix is generated by:

`rng.integers(0, 5, size=(10000, 5))`

Rows are bootstrap draws and columns are the five sampled paired-training-seed indices. The resulting index matrix is applied to that candidate's five same-seed paired advantages.

The RNG is not reinitialized between candidates, and candidates do not share a common index matrix.

This implementation choice resolves an otherwise unspecified random-stream allocation while preserving D005's frozen scientific design: five paired seed indices sampled with replacement per draw, 10,000 draws per candidate, and the mean paired advantage as the bootstrap statistic. It is frozen before any D005 bootstrap draw is generated and may not be changed in response to results.

## Governing principle

D005 may replace C0 only through the prospectively frozen, multiplicity-controlled validation rule above. It spends no protected-test or blind-challenge evidence.

## Revisit when

Revisit after the ten authorized new fits and frozen D005 analysis are complete, or if an execution/integrity failure prevents completion.

# D006 — Joint Chemprop Hyperparameter Optimization and Disclosed Internal-Test Evaluation

**Decision date:** 2026-10-14  
**Status:** APPROVED — execution remains blocked on the open items identified under *Revisit when*

## Decision

D006 authorizes one joint Chemprop 2.3.1 Bayesian/TPE hyperparameter search using the existing frozen training/validation split, followed by a single preregistered advancement decision against C0 on validation.

Only if the selected search configuration strictly beats C0 at that advancement gate may D006 proceed to a five-seed paired comparison on the existing 736-compound internal test.

The internal test is not fresh. It was previously used under D001/D002 for the C0 stock-versus-task-balanced comparison. That prior exposure must be disclosed in D006 and in any resulting writeup.

D003 remains unchanged. No blinded OpenADMET challenge data, score, rank, tier, leaderboard result, or other challenge-derived performance feedback may enter D006 training, hyperparameter selection, debugging, final internal-test adjudication, or methodological adaptation.

Descriptors, ensembling, and pretrained encoders are outside D006.

## Existing evidence boundary

D006 uses the existing frozen CYP-002 molecular universe and split:

- training: 3,433 compounds;
- validation: 736 compounds;
- internal test: 736 compounds.

The 736 validation compounds drive hyperparameter selection.

The 736 internal-test compounds were already accessed once under D001/D002. The recorded C0 results were:

- stock masked multitask MSE: macro ST-RAE `0.705435453184784`;
- task-balanced masked multitask MSE: macro ST-RAE `0.705530217573275`.

C0 stock was retained.

No fresh labeled evaluation population is available in the current record.

Accordingly, a D006 final evaluation is explicitly a **disclosed reused-internal-test comparison**, not a first-look or untouched internal validation.

The current record contains only one C0 stock internal-test evaluation, not five C0 test evaluations across training seeds. Therefore, if D006 reaches its final phase, the required five-seed C0 test results must be produced alongside the tuned configuration.

No test-result-driven retry, tuning, configuration substitution, or further model selection is permitted after D006 accesses the internal test.

## Search method

D006 uses one joint Bayesian/TPE hyperparameter search.

The search uses stock masked multitask MSE only. D001/D002's task-balanced loss is not reopened.

The search RNG seed is frozen prospectively as:

`20261016`

## Search space

The search space follows the installed Chemprop 2.3.1 hpopt implementation:

| Setting | D006 search space |
|---|---|
| Message-passing depth | integers `2–6` |
| FFN layers | integers `1–2` |
| Dropout | categorical `[0.0] × 8` plus `0.05, 0.10, ..., 0.40` |
| Message hidden width | `300` upward in increments of `100`; production ceiling **NOT APPROVED — pending benchmark** |
| FFN hidden width | `300` upward in increments of `100`; production ceiling **NOT APPROVED — pending benchmark** |
| Batch size | `{16, 32, 64, 128, 256}` |
| Maximum learning rate | log-uniform `1e-4–1e-2` |

Installed-source qualification established that the native Chemprop/Ray width support extends through `2400` inclusive. `2400` is therefore the native boundary available to the compute benchmark; it is not yet the approved production width ceiling.

Weight decay is **not searched: no stock support** in the relevant Chemprop 2.3.1 train/hpopt path. Adding it would require custom scientific implementation outside D006.

The existing CYP-002 configuration and canonical-manifest interfaces already represent all seven searched parameters. D006 requires one generic dynamic trial-configuration adapter so sampled values can enter the existing CYP-002 manifest/model/training pathway. It does not require seven independent scientific model changes.

D006 retains the existing CYP-002 model construction, frozen split routing, stock masked multitask loss, target processing, validation ST-RAE, deterministic training-seed plumbing, checkpointing, and manifest semantics.

## Search-controller and environment contract

The preserved CYP-002 environment does not contain Ray or HyperOpt and must not be modified to add them.

D006 therefore uses a separate search-controller environment for proposal generation and search state. The controller has no authority to train models, process molecular targets, calculate ST-RAE, access internal-test outcomes, or access blinded-challenge performance information.

All scientific model fitting and validation evaluation occur through the preserved CYP-002 pathway.

The controller interaction is sequential:

`controller proposal -> CYP-002 evaluation at both search seeds -> mean validation macro ST-RAE -> controller -> next proposal`

The only performance objective returned to the controller is:

`mean(validation macro ST-RAE at seeds 20261017 and 20261018)`

Lower is better.

No per-isoform result, internal-test result, challenge result, or other auxiliary performance information is supplied to the controller for proposal generation.

### Controller environment

The separate controller environment uses:

- Python `3.12.14`;
- Ray `2.56.1`;
- HyperOpt `0.3.0`;
- setuptools `81.0.0`;
- NumPy `2.5.4`;
- Pydantic `2.14.1`;
- `idna==3.20`;
- `click==8.5.0`.

During qualification, Python user-site visibility initially masked the absence of `idna==3.20` and `click==8.5.0` from the controller environment. Both dependencies were installed into the controller environment with user-site access disabled. Under `PYTHONNOUSERSITE=1`, `pip check` reports no broken requirements and Ray/HyperOpt import and construction succeed.

**Controller status: import and construction qualified under user-site isolation; trial proposal not yet exercised.**

Trial-proposal qualification may be performed as part of implementation before trial one; it has no scientific-performance authority.

### Python user-site isolation

Historical CYP-002 execution did not explicitly set `PYTHONNOUSERSITE=1`. Retrospective audit found user-site visibility enabled, but every examined scientifically consequential package resolved from the intended preserved `tripwire_cyp002` environment:

- Chemprop `2.3.1`;
- PyTorch `2.14.1+cu126`;
- Lightning `2.6.6`;
- NumPy `2.5.3`;
- pandas `3.0.6`;
- RDKit `2026.3.6`;
- SciPy `1.18.1`.

The only detected environment/user-site distribution overlap was `idna==3.20` in both locations. No examined scientific import resolved from user site. This retrospective audit does not independently reconstruct every historical process, and D001–D005 artifacts are not modified retroactively.

For all D006 controller and CYP-002 execution:

`PYTHONNOUSERSITE=1`

is mandatory.

Preflight must assert:

- `site.ENABLE_USER_SITE is False`;
- user site is absent from `sys.path`;
- consequential imports resolve from their approved environment.

D006 manifests must record user-site isolation state and resolved package paths/versions.

## HyperOpt initialization

D006's frozen search RNG seed is:

`20261016`

The controller must explicitly instantiate:

`HyperOptSearch(n_initial_points=N // 2, random_state_seed=20261016)`

where `N` is the frozen D006 production trial count.

The `N // 2` rule reproduces Chemprop 2.3.1's CLI argument-processing behavior. It must be passed explicitly because the rule is applied by Chemprop before `HyperOptSearch` construction; it is not a `HyperOptSearch` default.

`n_initial_points` is derived deterministically from `N` and is not independently tuned.

## Settings remaining fixed

The following remain frozen:

- model family: Chemprop D-MPNN;
- message-passing implementation: BondMessagePassing;
- loss: stock masked multitask MSE;
- four CYP direct-inhibition pIC50 endpoints;
- aggregation: normalized aggregation;
- aggregation norm: `100`;
- optimizer family: Adam;
- LR schedule family: NoamLR;
- initial LR: `0.0001`;
- final LR: `0.0001`;
- warmup: `2` epochs;
- activation: ReLU;
- batch normalization: false;
- maximum epochs: `50`;
- early-stopping patience: `20`;
- early-stopping minimum delta: `0`;
- validation metric: macro ST-RAE, lower is better;
- missing-target handling: masked;
- per-task target scaling fitted on training only;
- existing frozen train/validation/test assignment.

Descriptors, ensembling, pretrained encoders, and transfer learning are excluded.

## Search training seeds

Every search trial is trained at exactly two frozen training seeds:

`20261017`

`20261018`

The same two seeds apply to every trial.

Using two rather than five search seeds is an explicit compute tradeoff. It does not claim that two seeds fully characterize training stochasticity.

Neither search seed may be substituted after results are observed.

## Trial score

For trial `j`:

`V_j = mean(validation macro ST-RAE at seeds 20261017 and 20261018)`

Lower is better.

The Bayesian/TPE optimizer may use validation results from completed trials to choose subsequent hyperparameter configurations.

No internal-test or blinded-challenge information may enter this process.

## Search-trial tie rule

Each trial receives a deterministic trial index before its training results are available.

If multiple search trials have exactly the same lowest two-seed mean validation macro ST-RAE, select the tied trial with the **lowest predetermined trial index**.

No secondary performance metric is used as a tie-breaker.

## C0 validation advancement gate

C0 is the frozen baseline comparator. It is **not** a sampled Bayesian/TPE search candidate.

C0 is evaluated using the same two search training seeds:

`20261017`

`20261018`

and the same frozen training/validation compounds and validation scoring contract.

Let `T*` denote the selected search trial after application of the search-trial tie rule.

`T*` advances to the final internal-test phase if and only if:

`mean validation ST-RAE(T*) < mean validation ST-RAE(C0)`

Lower is better.

An exact tie does not advance.

If `T*` ties or performs worse than C0, **D006 terminates at validation**. The 736-compound internal test is not accessed for D006, C0 remains the scientific baseline, and no search trial receives final evaluation.

Passing this advancement gate does **not** establish that `T*` has demonstrated superiority over C0. It only authorizes the preregistered final comparison.

## Compute benchmark

Before the width ceiling or search trial count is frozen, D006 requires a compute-only benchmark.

The benchmark is performed interactively outside SLURM using the preserved Chemprop 2.3.1 environment.

The benchmark uses a small predetermined set of computationally stressful configurations spanning the prospective upper boundary, including:

- maximum message-passing depth;
- large message hidden width;
- large FFN hidden width;
- maximum FFN layer count;
- relevant batch-size extremes.

The native width boundary `2400` may be benchmarked for feasibility without thereby becoming the approved production ceiling.

Benchmark configuration selection is based solely on computational stress coverage. Validation performance from the benchmark has no search or model-selection authority.

For each benchmark fit, record actual per-fit wall time and relevant resource behavior.

Before search execution, report side by side:

- benchmark-derived per-fit planning time;
- proposed number of trials;
- two fits per trial;
- arithmetic extrapolated total compute time;
- requested execution wall time.

The extrapolated total and requested wall time must be shown separately and must not be conflated.

Historical C5 timing may be reported as prior context but does not substitute for the large-configuration benchmark.

No numerical search budget is authorized from historical timing alone.

## Width ceiling and trial count

**Message-hidden-width ceiling: NOT APPROVED — pending benchmark.**

**FFN-hidden-width ceiling: NOT APPROVED — pending benchmark.**

**Search trial count: NOT APPROVED — pending benchmark.**

After the benchmark:

1. determine the computationally feasible width ceilings;
2. consider the seven-dimensional search space and available compute budget;
3. determine a bounded trial count;
4. freeze both width ceilings and the exact trial count before trial one.

The trial count must not change in response to observed validation performance.

## Width ceiling and trial count

The D006 compute benchmark completed interactively on gpu-dev1 using GPU 1 and the preserved CYP-002 environment with `PYTHONNOUSERSITE=1`. Validation scores from the benchmark had no model-selection authority.

The benchmark results were:

| Fit | Configuration | Batch size | Epochs | Wall time (s) | Seconds/epoch | Peak GPU memory (MiB) |
|---|---|---:|---:|---:|---:|---:|
| B1 | depth 6; message width 2400; FFN width 2400; FFN layers 2 | 16 | 32 | 180.966458345 | 5.655201823 | 1587 |
| B2 | depth 6; message width 2400; FFN width 2400; FFN layers 2 | 256 | 34 | 197.555466600 | 5.810454900 | 9563 |
| B3 | depth 2; message width 300; FFN width 300; reference configuration | 64 | 50 | 85.189836988 | 1.703796740 | 757 |

All three fits completed. B1 and B2 early-stopped; B3 reached the 50-epoch cap. No fit was retried.

GPU 0 had substantial memory allocated by another process during the benchmark, but all three D006 benchmark fits ran on GPU 1. The GPU-0 allocation therefore does not qualify the benchmark timing.

For conservative production planning, B2 is the slowest observed configuration. Its measured rate is:

`5.8104549 seconds/epoch`

Extrapolated to the frozen 50-epoch cap:

`5.8104549 × 50 = 290.522745 seconds/fit`

D006 therefore freezes:

- message-hidden-width search range: `300–2400` in increments of `100`;
- FFN-hidden-width search range: `300–2400` in increments of `100`;
- search trial count: `N = 64`;
- training fits per trial: `2`;
- total planned search fits: `128`;
- conservative planning time per fit: `290.522745 seconds`;
- conservative serial search-time extrapolation: `37,186.91136 seconds = 10.3296976 hours`;
- requested execution wall time: `12 hours`.

The 12-hour requested wall time is an execution allowance, not a search stopping criterion. D006 stops after exactly 64 search trials regardless of unused wall time. Exhaustion of the requested wall time does not authorize additional trials, substitution of incomplete trials, or alteration of the search space.

With `N = 64`, the frozen controller initialization is:

`HyperOptSearch(n_initial_points=32, random_state_seed=20261016)`

The first 32 points are therefore the initial-random portion specified by the frozen Chemprop-compatible `N // 2` rule, followed by the remaining search trials under the HyperOpt/TPE search state.

The benchmark closes the D006 width-ceiling and trial-count decisions.

## Final training seeds

If and only if `T*` passes the validation advancement gate, train both the frozen selected configuration and C0 using the same five fresh training seeds:

`20261019`

`20261020`

`20261021`

`20261022`

`20261023`

These seeds are distinct from:

- prior training seeds `20261001–20261005`;
- D006 search training seeds `20261017–20261018`;
- search RNG seed `20261016`;
- prior relevant analysis RNG identities.

No final seed may be replaced after outcomes are observed.

## Final internal-test comparison

For each final training seed `s`, evaluate the tuned configuration and C0 on the identical frozen 736-compound internal-test partition using the identical scoring contract.

Define:

`A_s = test macro ST-RAE(C0,s) - test macro ST-RAE(tuned,s)`

Positive values favor the tuned configuration.

Define training-seed noise as:

`SD_seed = sample SD(A_1, A_2, A_3, A_4, A_5), ddof=1`

D006 retains the D005 materiality principle:

**the tuned configuration demonstrates improvement over C0 only if the preregistered lower confidence bound of the paired test advantage exceeds `SD_seed`.**

In plain text:

`LCB(paired test advantage) > SD_seed`

Otherwise:

**no demonstrated improvement over C0 under D006.**

Failure to clear the rule does not establish equivalence or intrinsic inferiority.

### Final D006 execution-contract amendment — approved 2026-10-21

#### Confidence-bound construction

The D006 final scientific comparison uses the five frozen paired training seeds:

`20261019, 20261020, 20261021, 20261022, 20261023`.

For each seed `s`:

`A_s = test macro ST-RAE(C0,s) - test macro ST-RAE(tuned,s)`

Positive values favor the tuned configuration.

The training-seed noise scale is:

`SD_seed = sample SD(A_1, A_2, A_3, A_4, A_5), ddof=1`

The final lower confidence bound is constructed by a paired-seed nonparametric bootstrap:

- bootstrap RNG seed: `20261024`;
- bootstrap draws: `10,000`;
- bootstrap unit: paired training seed;
- observations per draw: `5`;
- sampling: with replacement;
- statistic: mean paired test advantage;
- RNG: `np.random.default_rng(20261024)`;
- index generation: `rng.integers(0, 5, size=(10000, 5))`;
- confidence bound: one-sided 95% lower bound;
- LCB percentile: `5.0`;
- percentile implementation: `np.percentile(bootstrap_means, 5.0, method="linear")`;
- seed-noise implementation: `np.std(paired_advantages, ddof=1)`.

The same sampled indices are applied to the paired-advantage vector; C0 and tuned results are never bootstrapped independently.

D006 has exactly one final tuned-versus-C0 comparison, so no multiplicity adjustment is applied.

The final rule is:

`LCB_95_one_sided(mean paired test advantage) > SD_seed`

An exact equality does not pass.

If the rule does not pass, the frozen scientific verdict is:

`no demonstrated improvement over C0 under D006`

Failure does not establish equivalence or intrinsic inferiority.

The `10,000`-draw percentile-bootstrap design follows the established CYP bootstrap precedent. D006 prospectively freezes its own paired-seed implementation, RNG, one-sided confidence level, and percentile calculation.

No alternative confidence level, bootstrap seed, draw count, interpolation method, uncertainty estimator, or post hoc analysis may replace this construction after internal-test outcomes are accessed.

## Artifact and output contract

The already-completed D006 compute benchmarks remain canonical at:

`cyp/artifacts/cyp_002/runs/D006-BENCH-B1/`

`cyp/artifacts/cyp_002/runs/D006-BENCH-B2/`

`cyp/artifacts/cyp_002/runs/D006-BENCH-B3/`

They are not moved, copied, or duplicated into the new D006 namespace.

Their canonical paths, configurations, completion states, timing/resource summaries, and SHA-256 identities of decision-bearing artifacts are recorded in the frozen D006 search specification.

All new post-benchmark D006 artifacts live under:

`cyp/artifacts/cyp_006/`

with:

`cyp/artifacts/cyp_006/search/`
- `search_specification.json`
- `controller_manifest.json`
- `controller_state/`
- `trials/`
- `search_history.jsonl`
- `search_summary.json`
- `SHA256SUMS.txt`

`cyp/artifacts/cyp_006/selection/`
- `selected_configuration.json`
- `selection_manifest.json`

`cyp/artifacts/cyp_006/advancement/`
- `c0/`
- `advancement_result.json`

`cyp/artifacts/cyp_006/final/`
- `c0/`
- `tuned/`
- `analysis/`

### Search-trial artifacts

Each search trial lives at:

`cyp/artifacts/cyp_006/search/trials/trial_NNN/`

and contains:

- `trial_config.json`;
- `seed_20261017/`;
- `seed_20261018/`;
- `trial_summary.json`;
- `SHA256SUMS.txt`.

`trial_config.json` is written before either fit begins and records the controller-proposed seven-parameter configuration and predetermined trial index.

A trial counts as complete only when both frozen seed fits complete successfully and a valid two-seed mean validation objective has been written.

After successful completion, `trial_summary.json` records both validation macro ST-RAE values and their arithmetic mean. That mean is the sole objective returned to the controller.

### Failed or incomplete trial rule

A failed or incomplete trial stops the D006 search immediately.

The failed or incomplete trial and every artifact produced before failure are preserved.

The trial is:

- not substituted;
- not silently rerun;
- not renumbered;
- not deleted;
- not replaced by an additional proposal.

No subsequent trial may be proposed until the failure has been reported and separately adjudicated.

The frozen `N=64` budget does not authorize compensatory trials.

### Mid-search interruption and resume rule

The controller state is persisted after every successfully completed trial/objective update under:

`cyp/artifacts/cyp_006/search/controller_state/`

A controller, process, machine, or connection interruption occurring between completed scientific trial states resumes from the preserved controller state.

The search must never restart from trial 1 or from a fresh HyperOpt state after any proposal or objective has been consumed.

Before resume, the implementation must verify:

- controller-state integrity;
- controller-state hash;
- search-history consistency;
- last completed trial index;
- frozen search specification;
- frozen RNG/search configuration.

If an interruption occurs during an in-progress seed fit and leaves the trial incomplete, the failed/incomplete-trial rule applies: the search stops and the incomplete trial is reported rather than silently rerun.

### Search history and selection

`search_history.jsonl` is append-only and records in trial-index order:

- proposed configuration;
- proposal time;
- frozen training seeds;
- completion state;
- two validation scores when complete;
- mean validation score when valid;
- objective returned to HyperOpt;
- relevant artifact hashes.

After all 64 trials complete under the frozen contract:

`cyp/artifacts/cyp_006/selection/selected_configuration.json`

and

`cyp/artifacts/cyp_006/selection/selection_manifest.json`

freeze `T*`, its seven-parameter configuration, trial index, two-seed validation score, search-history hash, and controller-state hash.

### Advancement artifacts

C0 advancement runs live under:

`cyp/artifacts/cyp_006/advancement/c0/seed_20261017/`

`cyp/artifacts/cyp_006/advancement/c0/seed_20261018/`

The advancement decision is frozen at:

`cyp/artifacts/cyp_006/advancement/advancement_result.json`

It records:

- selected `T*` two-seed mean validation macro ST-RAE;
- C0 two-seed mean validation macro ST-RAE;
- strict comparison result;
- `advance` or `stop`;
- hashes of all decision inputs.

If the result is `stop`, no D006 final directory may be created and D006 terminates at validation.

### Final-evaluation artifacts

If and only if the advancement result is `advance`, final evaluation uses:

`cyp/artifacts/cyp_006/final/c0/seed_20261019/`

through:

`cyp/artifacts/cyp_006/final/c0/seed_20261023/`

and:

`cyp/artifacts/cyp_006/final/tuned/seed_20261019/`

through:

`cyp/artifacts/cyp_006/final/tuned/seed_20261023/`

Each final seed directory preserves:

- run manifest;
- frozen model configuration;
- best checkpoint;
- internal-test metrics;
- per-compound internal-test predictions;
- integrity hashes.

Final statistical analysis lives at:

`cyp/artifacts/cyp_006/final/analysis/`

with:

- `d006_final_analysis.py`;
- `d006_final_results.json`;
- `d006_bootstrap_means.csv`;
- `d006_final_provenance.txt`;
- `SHA256SUMS.txt`.

`d006_final_analysis.py` must be the actual executable analysis source.

`d006_bootstrap_means.csv` preserves all 10,000 bootstrap mean paired advantages.

`d006_final_results.json` records:

- five C0 test macro ST-RAE values;
- five tuned test macro ST-RAE values;
- five paired advantages;
- mean paired advantage;
- median paired advantage;
- `SD_seed`;
- one-sided 95% LCB;
- frozen-rule result;
- scientific verdict.

### Manifest and integrity requirements

Every D006 manifest records:

- `PYTHONNOUSERSITE=1`;
- `site.ENABLE_USER_SITE=False`;
- absence of user site from `sys.path`;
- approved package versions and resolved import paths appropriate to that environment;
- frozen dataset/split identity where applicable;
- exact training seed;
- exact configuration;
- start/end timestamps;
- execution status;
- relevant input/output SHA-256 hashes.

The controller manifest additionally records:

- Python `3.12.14`;
- Ray `2.56.1`;
- HyperOpt `0.3.0`;
- NumPy `2.5.4`;
- Pydantic `2.14.1`;
- setuptools `81.0.0`;
- `idna==3.20`;
- `click==8.5.0`;
- search RNG `20261016`;
- trial count `64`;
- `n_initial_points=32`;
- exact frozen search-space definitions.

Scientific CYP-002 manifests record the approved CYP-002 runtime and resolved paths for at least Chemprop, PyTorch, Lightning, NumPy, pandas, RDKit, and SciPy.

Integrity manifests hash all decision-bearing configuration, source, JSON, metrics, prediction, search-history, controller-state, and analysis artifacts required to reproduce the D006 result.

A file's own SHA-256 is not stored inside that file.

No existing D001-D005 artifact or D006 benchmark artifact may be overwritten, moved, deleted, or repurposed.

Intermediate, incomplete, and failed artifacts are preserved.

## Pre-execution closure

Before trial one, one D006 preflight must assert:

- canonical benchmark paths are exactly `D006-BENCH-B1`, `D006-BENCH-B2`, and `D006-BENCH-B3` under `cyp/artifacts/cyp_002/runs/`;
- benchmark decision-bearing artifacts match their recorded hashes;
- frozen molecular universe and split identity;
- Chemprop `2.3.1`;
- stock masked multitask loss;
- four frozen CYP endpoints;
- frozen seven-parameter search space;
- message and FFN width ranges `300–2400`;
- trial count `64`;
- search fits per trial `2`;
- search seeds `{20261017, 20261018}`;
- search RNG `20261016`;
- `n_initial_points=32`;
- final seeds `{20261019, 20261020, 20261021, 20261022, 20261023}`;
- final-analysis RNG `20261024`;
- final bootstrap draws `10,000`;
- final LCB is one-sided 95%, percentile `5.0`, `method="linear"`;
- seed-noise SD uses `ddof=1`;
- `PYTHONNOUSERSITE=1` for controller and CYP-002 execution;
- frozen D006 artifact paths are collision-free;
- failed/incomplete trials stop rather than substitute;
- controller state is persisted after every completed trial;
- resumed searches use preserved controller state and never restart after search state has been consumed;
- search cannot consume internal-test outcomes;
- final evaluation cannot execute without a frozen `advance` artifact;
- challenge data and challenge feedback are excluded.

Expected endpoint:

`D006 PREFLIGHT PASS — NO TRAINING OR EVALUATION PERFORMED`

Any failed assertion stops execution before trial one.

## Final execution and closure amendment — 2026-10-27

D006 completed the frozen search, validation advancement, five-seed paired final training, disclosed reused-internal-test evaluation, and preregistered final statistical analysis.

This amendment records the executed result. It does not alter the prospectively frozen D006 methodology or retroactively change any selection, advancement, evaluation, or statistical rule.

### Production search result

The frozen 64-trial HyperOpt/TPE search completed all 64 trials.

The selected configuration was trial 12:

- message-passing depth: `5`;
- message hidden dimension: `900`;
- FFN hidden dimension: `1400`;
- FFN layers: `1`;
- dropout: `0.15`;
- batch size: `32`;
- maximum learning rate: `0.004351853828824332`.

Its frozen two-seed validation results were:

- seed `20261017`: `0.6737485938348882`;
- seed `20261018`: `0.6599336141126925`;
- mean validation macro ST-RAE: `0.6668411039737903`.

The selected-configuration artifact SHA-256 is:

`8f9a7a91a9a831ceb5f80ca8fdb72c4271f6c506480212226f91aec1591264a8`

The selection-manifest SHA-256 is:

`010b2fd52531148a38252d681163dc571791c004f7471164a9ff35fd4c88fdec`

The frozen search-history SHA-256 is:

`2587ac31fa85dc63702b423aba143b30dd8f2d33f92787d33a6267dc7e5db294`

The frozen search-summary SHA-256 is:

`c95911398ab6cc58b961c3814e862990d7562ea07fd016d8a7e4d556bbc90a4a`

### Validation advancement result

C0 was evaluated at the same two frozen search seeds.

Its validation macro ST-RAE values were:

- seed `20261017`: `0.694521853753266`;
- seed `20261018`: `0.689300375140654`;
- mean: `0.691911114446960`.

The frozen advancement comparison was therefore:

`0.6668411039737903 < 0.691911114446960`

and trial 12 passed the preregistered validation advancement gate.

The authoritative advancement artifact has SHA-256:

`bb822c75d66cf38b236d1eb535fb0cc98dccade43fc22e8e73c80fbff7f3ac6b`

Passing this gate authorized the final comparison only; it did not establish superiority over C0.

### Final five-seed training

C0 and the frozen trial-12 configuration were each trained at the five prospectively frozen final training seeds:

`20261019, 20261020, 20261021, 20261022, 20261023`

All ten fits completed successfully.

The final-training audit established:

- `5/5` C0 fits complete;
- `5/5` tuned fits complete;
- exact frozen seed identities;
- stock masked multitask objective throughout;
- exact frozen trial-12 configuration for every tuned fit;
- checkpoint and metrics hashes consistent with their manifests;
- no internal-test evaluation during final training.

### Final internal-test execution

The disclosed reused-internal-test comparison was subsequently performed once for all ten frozen final checkpoints.

All ten evaluations completed successfully on the same frozen 736-compound internal-test partition.

The integrity audit established:

- `10/10` evaluations completed;
- `10/10` prediction matrices contained 736 rows;
- exact C0/tuned checkpoint identities;
- prediction hashes verified;
- score hashes verified;
- provenance hashes verified;
- frozen evaluator, dataset, and split identities verified;
- zero incomplete evaluations.

No configuration selection or methodological adaptation occurred from these internal-test results.

### Paired final results

For each final training seed:

`A_s = test macro ST-RAE(C0,s) - test macro ST-RAE(tuned,s)`

with positive values favoring the tuned configuration.

The paired advantages, derived directly from the frozen score artifacts at their stored precision, were:

- seed `20261019`: `-0.030287878144410918`;
- seed `20261020`: `0.0002551160079967696`;
- seed `20261021`: `0.0009163331800771113`;
- seed `20261022`: `-0.030601358415503155`;
- seed `20261023`: `0.01425235100408162`.

The frozen Phase-C analysis produced:

- mean paired advantage: `-0.009093087273551714`;
- `SD_seed`, using `ddof=1`: `0.020275673714827715`;
- bootstrap RNG: `20261024`;
- bootstrap draws: `10,000`;
- bootstrap unit: paired training seed;
- one-sided 95% lower confidence bound;
- percentile: `5.0`;
- percentile method: `linear`;
- bootstrap LCB: `-0.024172427987950208`.

The preregistered final gate was:

`LCB_95_one_sided(mean paired test advantage) > SD_seed`

Observed:

`-0.024172427987950208 > 0.020275673714827715`

Result:

**FALSE**

The frozen D006 scientific verdict is therefore:

**no demonstrated improvement over C0 under D006.**

This result does not establish equivalence between the tuned configuration and C0 and does not establish intrinsic inferiority of Chemprop hyperparameter optimization.

### Final scientific-core provenance

The qualified checkpoint evaluator is:

`cyp/scripts/cyp006_checkpoint_evaluation.py`

SHA-256:

`73d708bb1c0d47a91ecce44dac79e4e2a16efce24581af0f5b2c13e9ac659b48`

Its regression test is:

`cyp/tests/test_cyp006_checkpoint_evaluation.py`

SHA-256:

`b71076f08c80c40f76a98b28f5b42952a4ecd4c1eb735ca348f672b6cfc15634`

The final statistical-analysis implementation is:

`cyp/scripts/cyp006_final_analysis.py`

SHA-256:

`b87fbfbb9d5110ebebe3d3009cea6bd13c0526fe3c1c29070e66389409fe8a66`

Its regression test is:

`cyp/tests/test_cyp006_final_analysis.py`

SHA-256:

`f384a456d4845d3ff71d7c584c6a06e11fb514687900eecea693e8fa5eed359e`

All final-analysis regression tests passed before closure.

### Execution-path reconciliation

The prospective artifact contract declared final artifacts under:

`cyp/artifacts/cyp_006/final/`

During qualified execution, the final workflow was separated into lifecycle-specific roots rather than written to that prospective combined root.

Final training was executed and preserved under:

`cyp/artifacts/cyp_006/final_training/`

The one-time internal-test evaluation was executed and preserved under:

`cyp/artifacts/cyp_006/phase_b_internal_test/attempt_001/`

Qualification and production-wrapper evidence was preserved separately under the corresponding Phase-B qualification and production-wrapper directories.

This is an execution-path deviation from the prospectively named final artifact layout. It does not change the frozen molecular population, split, configurations, seeds, scoring contract, advancement rule, internal-test evaluation procedure, bootstrap procedure, or scientific verdict.

The actual executed paths are retained as canonical provenance. The prospective path declarations above are not rewritten or silently replaced.

One Phase-B submission, Slurm job `2772687`, failed before creation of `attempt_001` because the production wrapper attempted to create the attempt directory before its parent existed. No `STARTED.json` was created and no internal-test evaluator call began. The failure evidence was preserved.

The repaired wrapper retained `attempt_001` as the first scientific attempt. Slurm job `2773318` subsequently completed all ten internal-test evaluations successfully.

### Post-D006 development boundary

D006 is now closed.

Its internal-test outcomes may not be used to:

- rerank the 64 D006 trials;
- substitute another D006 configuration;
- alter the D006 search space;
- change the D006 statistical rule;
- add seeds to rescue the D006 result;
- perform a result-driven D006 retry;
- change the D006 scientific verdict.

Further Chemprop development is permitted only under a separate prospectively defined decision record.

Such development may use the completed D006 training and validation evidence, search history, training diagnostics, computational evidence, and implementation experience.

The revealed D006 internal-test outcomes, including aggregate, per-seed, per-task, and per-compound results, have no model-selection or methodological-adaptation authority for that subsequent development.

Blinded OpenADMET challenge feedback remains excluded from model development under the existing D003 boundary.

### Final analysis artifact closure

The prospectively required final statistical-analysis artifact set was completed under:

`cyp/artifacts/cyp_006/final/analysis/`

The artifacts were generated exclusively from the already-frozen and integrity-audited Phase-B score artifacts. No checkpoint loading, model inference, internal-test re-execution, or new methodological choice occurred during artifact serialization.

Final artifact SHA-256 identities are:

- `d006_final_analysis.py`: `b0c98ef2eead02f60bdc853bffce868068f0c85ae0813c114d6b146eeac2fc26`
- `d006_final_results.json`: `ebdcfd8a5b4aebf8f5b0367bbdf4ec3226e8816b5f180284f726d913dc50478b`
- `d006_bootstrap_means.csv`: `8ef37e0f74f3cf16457a9aa179a5c599332cb1b0856d44518aad1cc5aee782da`
- `d006_final_provenance.txt`: `b1c949dd31d015d936a675a6a69bc054cf5e160e4ad623c17793820b95bf20d9`
- `SHA256SUMS.txt`: `7b610a14439beb8c3337ce7912d8872cb61f6b093754d963889576d292bac1e9`

The final artifact-producing `d006_final_analysis.py` includes the previously qualified Phase-C calculations plus deterministic result serialization. Its final SHA-256 therefore supersedes the earlier pre-serialization source hash for identification of the executable source that produced the canonical final artifact set; the earlier source hash remains part of the historical implementation provenance.

The canonical D006 final artifact set is complete.

## Governing principle

D006 tested whether one prospectively frozen joint Chemprop hyperparameter search could identify a configuration that demonstrated robust improvement over C0 under the preregistered final rule.

The selected configuration passed the validation advancement gate but failed the final frozen scientific gate.

That negative scientific result is preserved independently of subsequent Chemprop development or eventual submission choice.

## Revisit when

D006 is closed.

Do not reopen D006 for additional hyperparameter search, trial substitution, additional training seeds, alternative internal-test analyses, alternative confidence bounds, or result-driven model development.

Revisit this record only to correct a demonstrated provenance, transcription, or artifact-identity error. Any new Chemprop optimization belongs to a separate decision record.

# D007 — Expanded Joint Chemprop Optimization

**Decision date:** 2026-10-27  
**Status:** Approved/Frozen

## Decision

D007 will run a new joint Chemprop 2.3.1 hyperparameter search intended to identify a better development-validation configuration than C0.

D006 remains closed and unchanged. D007 uses training/validation evidence only; the disclosed internal test and blinded challenge have no model-development or selection authority.

## Rationale

D006 searched architecture, dropout, batch size, and `max_lr`, but held the remainder of Chemprop's native NoamLR trajectory fixed.

D007 therefore retains the complete seven-dimensional D006 search space and adds:

- `init_lr_ratio`;
- `final_lr_ratio`;
- `warmup_epochs`.

Because all ten dimensions are searched jointly, D007 can identify a better configuration but cannot attribute any improvement specifically to schedule tuning.

## Search space

Retain the frozen D006 ranges for:

- message-passing depth: `2–6`;
- message hidden dimension: `300–2400`;
- FFN hidden dimension: `300–2400`;
- FFN layers: `{1,2}`;
- dropout: frozen D006 range;
- batch size: `{16,32,64,128,256}`;
- `max_lr`: log-uniform `1e-4–1e-2`.

Add the Chemprop-native NoamLR dimensions:

- `init_lr_ratio`: log-uniform `1e-2–1`;
- `final_lr_ratio`: log-uniform `1e-2–1`;
- `warmup_epochs`: integer `1–25`.

Adam, NoamLR, stock masked multitask MSE, 50 maximum epochs, patience 20, and validation macro ST-RAE remain unchanged.

## Development resampling

D007 reuses the existing frozen molecular `cluster_id` assignments and the CYP-003 repeated clustered 10-fold precedent.

The development universe is exactly the original train + validation population:

`4,169 molecules`

The original 736-molecule internal test is excluded.

Use:

- 5 clustered 10-fold repetitions;
- 50 development folds total;
- 3 prospectively declared training seeds per fold;
- identical folds and seeds for every configuration and C0.

Each configuration therefore receives:

`50 × 3 = 150 fits`.

## Objective

For configuration `j`, let `V(j,r,f,s)` be validation macro ST-RAE for repetition `r`, fold `f`, and training seed `s`.

First average the three seeds within each fold, then average the 50 fold means:

`O(j) = mean_(r,f)[mean_s V(j,r,f,s)]`

Because every fold has three seeds, this equals the arithmetic mean of all 150 fit-level macro ST-RAE values.

The same `O(j)` is used for both TPE optimization and final candidate selection.

### Amendment — reduced-fidelity search objective

**Amendment date:** 2026-10-29  
**Status:** Approved/Frozen

This amendment supersedes only the existing D007 requirement that all 128 configurations receive the full 150-fit objective during search and that the same `O(j)` directly determines selection across all 128 configurations. All other D007 provisions remain unchanged.

During search, each configuration is evaluated on a frozen 30-cell subset comprising 10 split manifests × all 3 training seeds. The 10 manifests were selected outcome-blind before scoring by choosing, within each of the five repetitions, the two frozen manifest files with the lowest SHA-256 hashes. The frozen subset is:

- repetition 1: folds `1, 9`;
- repetition 2: folds `2, 10`;
- repetition 3: folds `2, 7`;
- repetition 4: folds `1, 3`;
- repetition 5: folds `2, 10`.

Frozen subset artifact SHA-256:

`6da42fbfb24a3bc3425bddb12ac2fb65c90b999aab3a981a62b4153d094331ea`

Define the reduced search objective `O30(j)` as the arithmetic mean of the 30 corresponding fit-level validation macro ST-RAE values. This uses the same cell-level measurement and aggregation semantics as `O(j)`; only split coverage is reduced from 50 to 10 manifests. `O30(j)` has search-ranking authority only.

Existing-data qualification on trials 1–14 showed strong but imperfect agreement between `O30(j)` and `O(j)`: Spearman `ρ = 0.947`, Kendall `τ = 0.846`; the full-objective winner remained reduced-objective rank 1, and 2 of the full-objective top 3 remained in the reduced-objective top 3. This explicitly does not establish equivalent ranking among later TPE proposals. Reduced-fidelity screening therefore accepts the risk that a genuinely strong configuration may be excluded from full evaluation and cannot subsequently be rescued.

Existing trials 1–15 retain all completed 150-cell evidence. Their frozen 30-cell subsets supply the amended search observations; no completed scientific work is discarded. A fresh HyperOpt reconstruction using these reduced observations reproduced all 15 already-issued random-startup proposals exactly. The pre-amendment HyperOpt state is preserved as superseded and must not be continued as the amended search state.

The search retains `128` total configurations, consisting of `32` random-startup and `96` sequential TPE configurations, with the existing search space and controller RNG unchanged. Every configuration receives exactly `O30(j)` before the next configuration may be proposed.

After all 128 configurations complete, the three configurations with the lowest `O30(j)` advance to the original full 150-cell objective `O(j)`. Existing valid cells are reused: trials already possessing 150 valid cells require no additional fitting, while later finalists require only their 120 missing cells.

The final D007 configuration is selected solely by the lowest full `O(j)` among these three finalists. Exact full-objective ties retain the existing lower-trial-index rule. `O30(j)` has no authority to override the final 150-cell result.

This amendment preserves the 128-configuration search breadth, sequential TPE behavior, frozen search space, all three training seeds, cell-level scientific procedure, protected-test exclusion, and full 150-cell final-selection authority. Its accepted tradeoff is increased search-ranking uncertainty from reduced split replication.

## Search controller

Use a fresh HyperOpt/TPE search state.

- total configurations: `128`;
- random startup configurations: `32`;
- adaptive TPE configurations: `96`;
- fresh prospectively frozen controller RNG;
- no pruning;
- D006 trials are not imported as TPE observations.

TPE receives only the completed aggregate `O(j)` for each configuration.

## Selection

After all 128 configurations complete:

`T* = argmin_j O(j)`

C0 is evaluated on the identical 50-fold × 3-seed matrix.

T* replaces C0 as the preferred Chemprop development configuration iff:

`O(T*) < O(C0)`.

Exact ties are resolved by lower D007 trial index.

This establishes preference under the D007 development procedure, not independent external generalization.

## Execution contract

Before training:

- freeze the 50 cluster-aware split manifests;
- freeze the three training seeds and controller RNG;
- qualify all ten hyperparameters through the actual Chemprop training path;
- verify internal-test exclusion;
- verify the 150-fit aggregation and TPE interface;
- establish artifact, overwrite, failure/resume, and compute-execution contracts.

No D007 scientific fit is authorized until that qualification is reviewed and approved.

## Implementation qualification

D007 implementation qualification passed before scientific execution.

Frozen prospective RNG identities:

- training seeds: `20261101`, `20261102`, `20261103`;
- HyperOpt/TPE controller RNG: `20261107`.

Qualified implementation:

- `cyp/scripts/cyp007_infrastructure.py` — SHA-256 `521469efc448af749a0849921d903d7c96e8e162ac8efbbc3e278abc367576fd`;
- `cyp/tests/test_cyp007_infrastructure.py` — SHA-256 `7cd510c84a7ad72cb2f1e02cc6f68d3e56ee1880f622d0e9d23a759d89519297`;
- split-manifest `FREEZE_RECORD.json` — SHA-256 `9c6b292b8ca58ae83057cda157a338c0954b18b6f9253f040b0ca7344d6a2a85`.

Qualification result:

`10 passed`

The qualification established the frozen 10-dimensional search space, 50 deterministic common cluster-aware development splits, zero protected internal-test molecules in D007 partitions, three common training seeds, exact 150-cell objective aggregation, fresh 128-trial controller contract with 32 random startup trials and 96 TPE trials, no pruning, no imported D006 observations, and overwrite/failure guards.

No D007 scientific training was performed during qualification.

Production scientific execution remains blocked until the production execution wrapper, resource topology, resume/failure behavior, and preflight are qualified and reviewed.

### Completion amendment — 2026-11-04

**Status:** Approved and frozen — D007 complete

This amendment records the completed D007 execution, final development-selection result, and post-deadline blind-prediction freeze. It does not retroactively change the prospectively frozen D007 methodology, reduced-fidelity amendment, finalist-selection rule, C0 comparison rule, or challenge non-adaptivity boundary.

#### Completed reduced search

The amended D007 search completed all `128/128` configurations under the frozen O30 search objective.

The frozen top three were:

1. trial 121 — `O30 = 0.698191004621`;
2. trial 62 — `O30 = 0.698862474494`;
3. trial 125 — `O30 = 0.698893654706`.

These three configurations advanced to authoritative 150-cell evaluation.

Search-controller state SHA-256 at freeze:

`bc32e72e73a2e276ff8a14863c07a466f3cc4105aad09e786428eed61db99016`

HyperOpt searcher SHA-256 at freeze:

`835a87015dc2e911ce54bea7cb7fbea924eb6d1ad713094ce0951eb16734caa9`

No additional TPE proposal is authorized.

#### Authoritative finalist result

All three finalists completed the exact frozen 150-cell matrix with zero failed cells.

| Rank | Trial | O30 | O150 |
|---|---:|---:|---:|
| 1 | **121** | 0.698191004621 | **0.701212576796** |
| 2 | 125 | 0.698893654706 | 0.702315231168 |
| 3 | 62 | 0.698862474494 | 0.702505363217 |

The frozen D007 search winner is:

**D007-T121**

O30 rank 1 remaining O150 rank 1 is retrospective evidence about this completed search only. It does not establish general equivalence between O30 and O150.

#### C0 comparison and final D007 selection

C0 was evaluated on the identical 50-fold × 3-seed development matrix.

Authoritative comparison:

- O150(T121): `0.701212576796`
- O150(C0): `0.718103374388`
- C0 minus T121: `+0.016890797592`

The frozen rule requires T* to replace C0 iff O150(T*) is strictly lower than O150(C0).

The rule passes.

**D007-T121 replaces C0 as the preferred Chemprop development configuration under D007.**

Authoritative selection record:

`cyp/artifacts/cyp_007/finalist_completion_v1/C0_COMPARISON.json`

SHA-256:

`48bd619ebd42b30b0c8862f915b7696e2156e822f48e220789b1563eb46817a8`

D007 selection is closed. Future internal-test results, challenge labels, challenge scores, ranks, leaderboard results, or post-deadline external evaluation may not reopen or rerank D007.

#### OpenADMET deadline consequence

The official OpenADMET CYP challenge submission deadline passed before the completed D007 model-selection and prediction workflow was ready.

No official final OpenADMET submission was made from the completed D007-T121 model.

Accordingly, the project does not claim an official prospective OpenADMET challenge result from D007.

The missed deadline does not alter the completed D007 development result.

#### Post-deadline blind-inference amendment

After D007 selection closed, a separate inference-only amendment froze an ensemble containing exactly all 150 T121 `(repetition, fold, seed)` checkpoints.

The inference contract is fail-closed:

- exactly 150 members are required;
- no missing checkpoint may be ignored;
- no checkpoint may be substituted, excluded, or reweighted;
- every checkpoint must match its frozen cell identity and training-fitted target transformation;
- predictions are on the original pIC50 scale;
- authoritative prediction = equal-weight arithmetic mean across exactly 150 members;
- checkpoint SD uses `ddof=1` and is descriptive only;
- no training, fine-tuning, calibration, model adaptation, checkpoint reselection, or ensemble-weight adaptation is permitted.

Qualification established:

- 150/150 canonical members present;
- zero checkpoint/identity failures;
- 150/150 training-only target transformations reconstructed;
- all checkpoint hashes unchanged;
- 150/150 strict checkpoint loads passed.

#### Blind-inference attempt history

Authoritative blinded-input SHA-256:

`a342f8444a8dcb531ca12f3685293f0bd6c36ae9073f491e44a9bc1cc4b741f9`

**Attempt 001 — Slurm job `2886913`**

The run failed before challenge access because the blinded CSV had not yet been staged to CoreHPC. No prediction artifact was created.

**Attempt 002 — Slurm job `2887724`**

The blinded molecular input was accessed, but the initial adapter incorrectly reused a target-requiring training/evaluation dataset constructor. The run stopped before the first checkpoint prediction because the correctly blinded file lacked CYP target columns.

No labels, scores, ranks, or leaderboard results were accessed and no predictions were generated.

A narrowly scoped mechanical amendment replaced the target-requiring construction with target-free Chemprop `MoleculeDatapoint.from_smi(smiles)` construction.

No model, checkpoint, ensemble, weighting, aggregation, prediction-scale, or scientific-selection rule changed.

**Attempt 003 — Slurm job `2889407`**

The amended frozen inference implementation completed:

- 150/150 checkpoint predictions;
- all 750 challenge molecules;
- all four direct-inhibition pIC50 endpoints.

No model adaptation occurred after challenge access.

No challenge labels, scores, ranks, or leaderboard results were accessed.

#### Frozen blind predictions

Authoritative ensemble prediction artifact:

`cyp/artifacts/cyp_007/t121_blind_inference_v1/prediction_attempt_003/ensemble_predictions.csv`

SHA-256:

`51002c6dbe75041bf4c874f1a4dce4d59f293e82d9d8e4771a2132160da7e7ca`

Complete 150-member prediction matrix:

`cyp/artifacts/cyp_007/t121_blind_inference_v1/prediction_attempt_003/member_predictions.csv`

SHA-256:

`f7e683c7ab8573762d19231e092fe7d48e708c9e572d526db12a4c502c02e8cb`

The complete prediction artifact contains 150 checkpoints × 750 molecules × 4 CYP endpoints.

The successful prediction artifact was frozen before authoritative challenge-label access in:

`cyp/artifacts/cyp_007/t121_blind_inference_v1/BLIND_PREDICTION_FREEZE.json`

At D007 closure:

- authoritative OpenADMET challenge labels have not been released;
- no blind scoring has occurred;
- no challenge-performance claim is authorized.

If authoritative labels are released later, the scoring contract must be frozen before labels are opened against the predictions.

Any resulting evaluation is a **post-deadline blind evaluation, not an official prospective OpenADMET challenge result**.

#### D007 closure

D007 is complete.

No additional D007 hyperparameter search, TPE proposal, training seed, finalist, C0 comparison, checkpoint selection, calibration, ensemble adaptation, or result-driven retry is authorized.

The next CYP development task is implementation of a stable production inference interface around the already-selected T121 model.

External blind scoring remains parked until authoritative labels are released.

## Governing principle

D007 expands Chemprop optimization while addressing both training-seed variability and dependence on a single molecular validation split. All configurations receive the same development evidence, and previously disclosed test results cannot influence the search.

## Revisit when

Revisit if the frozen Chemprop version, molecular cluster assignments, development universe, search space, split design, objective, or controller implementation changes.

Otherwise, freeze D007 before the first scientific fit.

# D008 — Post-Deadline Blind External Evaluation of Frozen D007-T121

**Decision date:** 2026-11-06

**Status:** Approved and frozen before external label/prediction scoring

## Purpose and claim boundary

D008 governs the one-shot post-deadline blind external evaluation of the already-frozen D007-T121 ensemble against the organizer-released OpenADMET CYP direct-inhibition test labels.

D007 is complete and closed. D008 does not reopen D007 and has zero model-development or model-selection authority.

This evaluation is not an official prospective OpenADMET challenge result. The challenge deadline passed before the completed D007-T121 workflow was ready for submission.

D008 results may not be used to retrain, recalibrate, rerank, select, replace, or otherwise adapt D007-T121; alter the 150-member ensemble; alter preprocessing or target transformations; regenerate predictions; or change any D007 development decision.

The evaluation uses the already-frozen 750 x 4 ensemble prediction artifact. Inference may not be rerun merely because authoritative labels are now available.

## Frozen external evidence

Before any label/prediction scoring, the organizer-released OpenADMET unblinded release was preserved verbatim under:

`cyp/data/openadmet_unblinded_release_2026-11-06/`

The authoritative direct-inhibition label artifact is:

`cyp/data/openadmet_unblinded_release_2026-11-06/cyp-challenge-TEST-UNBLINDED_inhibition.csv`

SHA-256:

`6551eba506c90cf51434bc1764424e928cfa19b45854148d692247b8eb49f74c`

Provenance: organizer-released OpenADMET CYP Challenge unblinded dataset downloaded from the OpenADMET Hugging Face dataset repository and preserved verbatim before label/prediction scoring.

The frozen D007-T121 ensemble prediction artifact is:

`cyp/artifacts/cyp_007/t121_blind_inference_v1/prediction_attempt_003/ensemble_predictions.csv`

SHA-256:

`51002c6dbe75041bf4c874f1a4dce4d59f293e82d9d8e4771a2132160da7e7ca`

The prediction artifact was generated and frozen before authoritative test labels were released.

## Primary external-performance estimand

The primary external evaluation is the absolute performance of the frozen D007-T121 ensemble on the released OpenADMET CYP direct-inhibition test set.

The primary metric is the project's already-qualified ST-RAE implementation, which directly wraps the preserved upstream OpenADMET `rae_soft_threshold_absolute_error` implementation.

No alternative ST-RAE definition is introduced for D008.

ST-RAE is calculated independently for:

- CYP1A2 direct-inhibition pIC50;
- CYP2C9 direct-inhibition pIC50;
- CYP2D6 direct-inhibition pIC50;
- CYP3A4 direct-inhibition pIC50.

For each endpoint, the authoritative released test labels and corresponding released assay confidence bounds are supplied to the frozen ST-RAE implementation together with the already-frozen ensemble predictions.

The primary macro ST-RAE is the unweighted arithmetic mean of the four endpoint ST-RAE values.

Lower ST-RAE is better.

The primary reported external-performance quantities are:

1. CYP1A2 ST-RAE;
2. CYP2C9 ST-RAE;
3. CYP2D6 ST-RAE;
4. CYP3A4 ST-RAE; and
5. arithmetic-mean macro ST-RAE.

These full-test-set values are the project's primary post-deadline blind external-performance estimates.

## Relationship to prior project bootstraps

The project-specific 10,000-draw bootstrap procedures previously used in the CYP arm are not transferred to D008.

Those procedures were prospectively designed for paired model-comparison questions, including scaffold/analog-cluster resampling for stock-versus-task-balanced comparisons and paired-training-seed resampling for tuned-versus-C0 comparisons.

D008 instead measures the absolute external performance of one already-frozen ensemble on one newly released external test cohort.

Accordingly, no prior paired cluster-bootstrap or paired-seed confidence interval is presented as the primary uncertainty procedure for D008.

This does not amend, supersede, or reinterpret any previously frozen bootstrap analysis.

## Organizer-compatible bootstrap characterization

A secondary analysis reproduces the OpenADMET organizer-compatible bootstrap reporting procedure for the same frozen ensemble predictions.

This analysis is descriptive and secondary to the full-test-set primary ST-RAE estimates.

For each of 1,000 bootstrap iterations:

1. test observations are sampled with replacement using the organizer-compatible sampling procedure;
2. ST-RAE is recomputed from the resampled observations using the same preserved OpenADMET scoring implementation;
3. the four endpoint ST-RAE values are calculated; and
4. macro ST-RAE is calculated as the unweighted arithmetic mean of the four endpoint values.

The bootstrap uses:

`np.random.default_rng(seed=0)`

The secondary report preserves the mean and standard deviation across the 1,000 bootstrap iterations for each endpoint and for macro ST-RAE.

These quantities are explicitly reported as organizer-compatible bootstrap characterizations. They do not replace the primary full-test-set ST-RAE estimates and are not interpreted as the project's historical cluster-bootstrap confidence intervals.

## Missing-label and identity handling

Evaluation is endpoint-specific.

Only observations possessing an authoritative released label for a given endpoint may contribute to that endpoint's score.

No missing ground-truth label may be imputed.

Predictions and released labels must be joined using the authoritative molecule identifier shared by the frozen prediction artifact and released test artifact.

Before scoring, the implementation must establish that identifiers are unique where required and that every scored released-label observation has exactly one corresponding frozen prediction.

Unexpected duplicate identifiers, missing required predictions, unexpected endpoint/schema changes, or identity mismatches are fail-closed integrity errors.

Such errors do not authorize ad hoc row removal, prediction regeneration, inference reruns, or altered matching rules.

The number of scored observations for each endpoint must be recorded.

## Additional metrics

No additional metric receives primary or model-selection authority through D008.

MAE, R2, Spearman correlation, Kendall correlation, endpoint residual analyses, rank analyses, subgroup analyses, or other descriptive statistics are not part of the one-shot primary scoring operation unless separately frozen before authoritative labels are opened against predictions.

They may not be added retrospectively in response to the observed ST-RAE result and represented as preregistered evidence.

## Implementation qualification

The D008 evaluator must be qualified before it is permitted to access the authoritative unblinded labels together with the frozen predictions.

Qualification must use synthetic or already-disclosed non-external data and establish at minimum:

- exact agreement with the already-qualified project ST-RAE implementation;
- correct endpoint-wise scoring;
- correct arithmetic macro aggregation;
- correct confidence-bound routing;
- correct identifier alignment independent of input row order;
- correct endpoint-specific missing-label handling;
- fail-closed behavior for duplicate identifiers and missing required predictions;
- deterministic organizer-compatible bootstrap behavior under the frozen RNG contract;
- exactly 1,000 organizer-compatible bootstrap iterations; and
- correct calculation of bootstrap mean and standard deviation.

Qualification must not inspect external performance.

The executable evaluator and qualification tests must be frozen and hashed before the one-shot label/prediction evaluation.

## One-shot execution

After successful qualification and implementation freeze, the evaluator may access the preserved authoritative inhibition-label artifact and already-frozen D007-T121 ensemble prediction artifact together for the first time.

The primary and secondary analyses are executed in the same frozen evaluation run.

No result-driven retry, alternate scorer, changed random seed, changed bootstrap procedure, altered endpoint set, changed identity rule, recalibration, prediction modification, or alternative aggregation is authorized after external performance is observed.

An operational failure before valid performance results are produced may be repaired only if the failure and repair are documented and the scientific scoring contract remains unchanged.

## Reporting

The result must be described as a:

**post-deadline blind external evaluation of the frozen D007-T121 ensemble on the released OpenADMET CYP direct-inhibition test set.**

It must not be described as an official OpenADMET challenge submission, official challenge score, or prospective leaderboard result.

Reporting must distinguish:

- primary full-test-set per-endpoint and macro ST-RAE;
- secondary organizer-compatible 1,000-bootstrap mean +/- standard deviation;
- integrity and sample-count diagnostics; and
- any operational deviations encountered during execution.

External results have no authority to modify D007-T121 or any earlier development decision.

## Governing principle

The released labels are used once to measure external generalization of a model and prediction artifact whose development and inference were already frozen before those labels became available.

Evaluation methodology is frozen before performance is exposed. External performance is evidence about the frozen model, not feedback for adapting it.

## Revisit when

Revisit D008 only if an integrity failure demonstrates that the preserved authoritative label artifact or frozen prediction artifact cannot be evaluated under this contract, or if a factual error is discovered in the documented organizer scoring procedure before external performance is exposed.

Observed external performance is not grounds for revision.

### D008 completion amendment — 2026-11-06

**Status:** Completed and frozen

D008 completed the preregistered post-deadline blind external evaluation of the frozen D007-T121 150-checkpoint ensemble against the organizer-released OpenADMET CYP direct-inhibition test labels.

This amendment records the executed result. It does not alter the prospectively frozen D008 methodology or grant the external result any model-development or model-selection authority.

#### Frozen production inputs

Authoritative released direct-inhibition labels:

`cyp-challenge-TEST-UNBLINDED_inhibition.csv`

SHA-256:

`6551eba506c90cf51434bc1764424e928cfa19b45854148d692247b8eb49f74c`

Frozen D007-T121 ensemble predictions:

`cyp/artifacts/cyp_007/t121_blind_inference_v1/prediction_attempt_003/ensemble_predictions.csv`

SHA-256:

`51002c6dbe75041bf4c874f1a4dce4d59f293e82d9d8e4771a2132160da7e7ca`

The prediction artifact was generated and frozen before authoritative test-label release.

#### Qualified D008 implementation

The original pre-exposure D008 evaluator and test suite were frozen before external scoring at:

- evaluator SHA-256: `ad2f38a60bc8ba620a4db94d23b83e6691201059c2da147d2dee4cad4543ebbb`
- tests SHA-256: `e812481978bb907d3e4d100b56a19b199dbe0fd4573bbcffe072344317ed225a`
- qualification: `58 passed`

Production Attempt 001 completed the frozen scientific calculation in memory but the ad hoc output wrapper failed during JSON serialization because the returned evaluation object contained pandas `DataFrame` objects. No external-performance value was printed or inspected from Attempt 001. The incomplete serialization artifact was preserved and not treated as a valid D008 result.

A serialization-only reporting layer was subsequently added without changing molecular alignment, endpoint routing, ST-RAE calculation, macro aggregation, bootstrap calculation, input artifacts, or any other scientific computation.

The serialization-recovery implementation was frozen at:

- evaluator SHA-256: `bdb934b03a2ece685aea7c021e3d4d1221c4f28c6e51a2acc4e8d408bd52ccde`
- tests SHA-256: `68d4853897a46935090a90021bcf8a3c1f39f4c3c84170a452f655e8f16efbb9`
- local qualification: `59 passed`
- CoreHPC qualification under the production D008 environment: `59 passed`

The recovery execution reran the identical deterministic frozen scientific calculation solely to persist its outputs through the qualified serialization layer.

#### Primary post-deadline blind external result

The preregistered full-test-set ST-RAE results were:

| Endpoint | n scored | n missing | ST-RAE |
|---|---:|---:|---:|
| CYP1A2 | 748 | 2 | `0.7728950609716935` |
| CYP2C9 | 750 | 0 | `0.5558588275060217` |
| CYP2D6 | 748 | 2 | `1.217061477884744` |
| CYP3A4 | 748 | 2 | `0.5078206616488673` |

The preregistered unweighted four-endpoint macro ST-RAE was:

`0.7634090070028317`

Lower ST-RAE is better.

The result demonstrates heterogeneous external performance across endpoints. CYP2C9 and CYP3A4 produced the lowest external ST-RAE values, CYP1A2 was intermediate, and CYP2D6 was the weakest endpoint and exceeded `1.0`.

No causal explanation for the endpoint heterogeneity is assigned by D008.

#### Organizer-compatible bootstrap characterization

The preregistered secondary organizer-compatible characterization used:

- bootstrap iterations: `1000`;
- RNG: `np.random.default_rng(seed=0)`;
- reporting: mean and population standard deviation across bootstrap iterations.

Results:

| Endpoint | Bootstrap mean | Bootstrap SD |
|---|---:|---:|
| CYP1A2 | `0.7745453137726171` | `0.02728147930356748` |
| CYP2C9 | `0.5571939927014358` | `0.020433515126421893` |
| CYP2D6 | `1.2172319746112334` | `0.05332503578021169` |
| CYP3A4 | `0.5084294889718826` | `0.02156381142675119` |
| Macro | `0.7643501925142921` | `0.016520750512073138` |

These bootstrap quantities remain secondary organizer-compatible characterizations. They do not replace the primary full-test-set ST-RAE values and are not interpreted as the project's historical cluster-bootstrap confidence intervals.

#### Frozen result artifact

The authoritative serialized D008 result is:

`d008_external_evaluation_v1/D008_EXTERNAL_EVALUATION.json`

SHA-256:

`b9f9beeaaa36b6e9bedd65740f966b67bd1958e9c09299c05926dbb6558e3b30`

This artifact is the authoritative D008 external-evaluation result.

#### Claim boundary

The D008 result is described as a:

**post-deadline blind external evaluation of the frozen D007-T121 ensemble on the released OpenADMET CYP direct-inhibition test set.**

It is not an official OpenADMET challenge submission, official challenge score, or prospective leaderboard result.

D008 does not reopen D007.

The external result may not be used to retroactively recalibrate, rerank, replace, or otherwise adapt D007-T121, its 150-member ensemble, its frozen predictions, or any D007 development decision.

No residual analysis, subgroup analysis, alternative metric, recalibration, endpoint-specific rescue analysis, or other result-driven post-hoc investigation was performed under D008.

The D007 development objective and D008 external macro ST-RAE are different estimands produced under different evaluation structures and are not interpreted as a paired generalization-gap statistic.

#### D008 closure

D008 is complete.

The frozen external evidence establishes heterogeneous external generalization of D007-T121, including a marked weakness on CYP2D6. That observation is preserved as external evidence rather than used to alter the frozen model.

Any future investigation motivated by the D008 result must be explicitly prospective, separately scoped, and unable to change the D008 result or retrospectively acquire D008 validation authority.

Observed D008 performance is not grounds for revision of D007 or D008.

# D009 — P1 Pairwise CYP Development Candidate

**Decision date:** 2026-11-07

**Status:** Complete / frozen

## Decision and scope

D009 is complete and frozen. P1 was implemented, qualified, benchmarked, run once across the 150 frozen development cells, characterized against the recovered D007-T121 comparator evidence, and closed without advancement.

The frozen final result is:

- P1 O150: `0.7074933757653159`
- T121 O150: `0.701212576796`
- P1 minus T121: `+0.006280798969315882`
- P1 advances: `false`
- preferred configuration: **D007-T121**

No P1 rerun, rescue, endpoint veto, alternative aggregation, recalibration, external rescoring, or result-driven scientific modification is authorized.

## Original decision and scope

Approve P1 as the first prospective pairwise CYP development candidate for Tripwire-CYP.

P1 tests whether a shared four-endpoint Chemprop D-MPNN trained to predict pairwise differences in CYP direct-inhibition pIC50 can improve held-out absolute-pIC50 prediction relative to frozen D007-T121.

This approval covers methodology design and this decision record only.

It does **not** authorize:

- implementation;
- production pair generation;
- the all-reference cost benchmark;
- model training;
- performance evaluation;
- modification of D007, T121, the frozen T121 ensemble, or D008.

D008 may motivate future scientific questions, including new prospective model designs, but D008 has no architecture-selection, hyperparameter-selection, reference-selection, stopping, validation, or advancement authority for P1.

## Scientific question

On the identical frozen D007 molecule-level development matrix, does the approved P1 pairwise package achieve lower held-out absolute-pIC50 macro ST-RAE than D007-T121?

P1 is a controlled **package comparison**, not a parameter-matched ablation.

It does not presume that pairwise learning will repair CYP2D6 or that the cause of D008's endpoint heterogeneity has been established.

## T121 comparator and authoritative development quantity

P1's comparator is the **single-model D007-T121 development procedure**, not the later frozen 150-checkpoint blind-inference ensemble.

For development cell \(c\), T121 trained one model on that cell's frozen training partition. Checkpoint selection and early stopping used held-out **validation macro ST-RAE**, defined as the unweighted macro across CYP1A2, CYP2C9, CYP2D6, and CYP3A4. The retained checkpoint was the checkpoint with the lowest validation macro ST-RAE for that fit.

Let:

\[
S_c(M)
=
\frac{1}{4}
\sum_{t=1}^{4}
\operatorname{ST\!-\!RAE}_{c,t}(M)
\]

denote the retained single model's held-out validation macro ST-RAE in development cell \(c\).

The authoritative full-fidelity D007 objective is:

\[
O_{150}(M)
=
\frac{1}{150}
\sum_{c=1}^{150} S_c(M).
\]

The 150 cells are the frozen:

\[
50\ \text{repetition/fold partitions}
\times
3\ \text{training seeds}.
\]

Equivalently, D007 described this as averaging the three seed-level cell scores within each of 50 folds and then averaging those 50 fold values.

The authoritative frozen comparator is:

\[
O_{150}(\mathrm{T121})
=
0.701212576796.
\]

### Source artifact

The existing D007 selection record is:

`cyp/artifacts/cyp_007/finalist_completion_v1/C0_COMPARISON.json`

SHA-256:

`48bd619ebd42b30b0c8862f915b7696e2156e822f48e220789b1563eb46817a8`

The completed D007 record independently documents T121's full-fidelity O150 as `0.701212576796` and defines O150 as the arithmetic mean of the 150 fit-level validation macro ST-RAE values.

This quantity is **not** the score of the later frozen 150-checkpoint ensemble. The blind ensemble was created only after D007 development selection had closed.

P1 must reproduce the intended comparable development scoring procedure: one P1 fit per corresponding development cell, checkpoint selection using held-out absolute-pIC50 validation macro ST-RAE, and one retained cell-level validation macro score entering O150.

## Frozen T121 configuration inherited where specified

Recovered T121 configuration:

- message-passing depth: `4`
- message hidden dimension: `600`
- FFN hidden dimension: `2000`
- FFN layers: `1`
- dropout: `0.4`
- batch size: `16`
- maximum epochs: `50`
- early-stopping patience: `20` validation checks
- warmup epochs: `10`
- initial LR: `1.4283739994469765e-05`
- maximum LR: `0.0008245400010426866`
- final LR: `0.00020141638331344468`

T121 uses stock Chemprop MSE rather than the previously evaluated task-balanced loss.

The existing pipeline fits per-endpoint target standardization using the training partition only.

D007-T121 remains frozen regardless of P1 outcome.

## P1 architecture

P1 uses a two-component shared Chemprop 2.3.1 D-MPNN.

For molecules \(A\) and \(B\):

\[
h_A=E_\theta(A),
\qquad
h_B=E_\theta(B),
\]

where each embedding has dimension `600` and the same encoder parameters are used for both molecules.

The pair function \(G\) receives ordered concatenation:

\[
[h_A,h_B]\in\mathbb{R}^{1200}.
\]

The approved pair head is:

- input dimension: `1200`;
- one `2000`-unit hidden layer;
- T121 activation;
- dropout `0.4`;
- four endpoint outputs;
- the same \(G\) parameters for both orientations.

The reported delta is:

\[
\widehat{\Delta z}_t(A,B)
=
\frac{
G_t(h_A,h_B)-G_t(h_B,h_A)
}{2}.
\]

No additional width search or parameter-matching experiment is authorized for P1.

### Deterministic symmetry guarantees

In deterministic evaluation mode:

\[
\widehat{\Delta z}_t(A,B)
=
-\widehat{\Delta z}_t(B,A)
\]

and:

\[
\widehat{\Delta z}_t(A,A)=0
\]

are required architecture invariants, subject only to ordinary floating-point arithmetic.

Training-mode stochastic symmetry is not guaranteed. Dropout is `0.4`, and ordinary dropout masks are not assumed to be coupled between ordered evaluations.

Batching both orientations does not imply shared dropout masks.

No coupled-dropout machinery is introduced solely to impose training-time symmetry.

Cycle/transitivity residuals are diagnostic only. The unrestricted antisymmetric head does not guarantee:

\[
\widehat{\Delta}(A,B)
+
\widehat{\Delta}(B,C)
=
\widehat{\Delta}(A,C).
\]

## Parameter accounting

Before training authority is considered, qualification must record:

- T121 total parameter count;
- T121 trainable parameter count;
- P1 total parameter count;
- P1 trainable parameter count.

P1 is expected to have more head parameters because the pair-head input is 1200-dimensional rather than the pointwise model's 600-dimensional molecular representation.

Parameter counts are transparency measures.

P1 is a controlled package comparison, **not a parameter-matched ablation**.

No width search or parameter-matching experiment is required for P1.

## Molecule-level splits

P1 uses the identical frozen D007 molecule-level development partitions.

Pair construction occurs only after molecule assignment.

No molecule outside a cell's training partition may contribute to that cell's pairwise training.

No new split generation is authorized.

## Pair eligibility and missing labels

For endpoint \(t\), pair \((A,B)\) has an eligible delta target only when both training labels are observed.

Define:

\[
M_{ABt}
=
\begin{cases}
1,& y_{A,t}\text{ and }y_{B,t}\text{ are both observed},\\
0,& \text{otherwise}.
\end{cases}
\]

For \(M_{ABt}=1\):

\[
\Delta y_t(A,B)
=
y_{A,t}-y_{B,t}.
\]

No endpoint label is imputed.

No missing target is replaced by a self-pair, prediction, surrogate value, validation label, protected-test label, or D008 label.

If \(n_t<2\) in a training partition, endpoint \(t\) has no valid within-partition pairwise supervision and must be explicitly reported.

Existing-artifact accounting found zero such cases across the 50 frozen D007 partitions.

## Pair sampling

Pairs are sampled on-the-fly from the corresponding training partition.

Sampling is uniform over ordered distinct pairs:

\[
A\ne B.
\]

The full \(N(N-1)\) pair universe is not materialized or represented as independent experimental evidence.

A sampled pair may contain zero to four eligible endpoint targets.

Pairs with zero eligible endpoints are classified as:

`no_supervision`

and:

- produce no optimizer update;
- are rejected and resampled;
- are counted explicitly.

Generated pairs are dependent transformations of the original molecular measurements, not independent new measurements.

## Fold-specific pair weighting

For training-partition size \(N\) and endpoint-specific observed-label count \(n_t\), uniform ordered-distinct pair sampling gives:

\[
q_t
=
\frac{n_t(n_t-1)}{N(N-1)}.
\]

The approved endpoint weight is:

\[
w_t
=
\frac{N-1}{n_t-1},
\qquad n_t\ge2.
\]

Therefore:

\[
q_t w_t
=
\frac{n_t}{N}.
\]

This identity supports matching the **endpoint label mixture in the population limit**.

It does **not** establish equivalence of:

- finite-minibatch weighting;
- stochastic gradients;
- molecule exposure;
- pair dependence;
- optimization trajectories;
- or the complete pointwise and pairwise objectives.

### Recovered ranges across the 50 frozen partitions

Training-label counts:

- CYP1A2: `1062–1101`
- CYP2C9: `965–1008`
- CYP2D6: `1123–1157`
- CYP3A4: `1766–1815`

Approved \(w_t\) ranges:

- CYP1A2: `3.41–3.53628652215`
- CYP2C9: `3.72492552135–3.89107883817`
- CYP2D6: `3.24480968858–3.3431372549`
- CYP3A4: `2.06780595369–2.12521246459`

Uniform-pair eligibility ranges:

- CYP1A2: `0.0800199925348–0.0860538113788`
- CYP2C9: `0.0660989336745–0.072124050502`
- CYP2D6: `0.089528790901–0.0950345011053`
- CYP3A4: `0.221475410395–0.233939748263`

Weights are derived from training-partition membership and training-label availability only.

Performance cannot alter them.

## Approved P1 loss

P1 uses:

\[
L_{\mathrm{P1}}
=
\frac{
\displaystyle
\sum_{j,t}
M_{jt}w_t
\left(
\widehat{\Delta z}_{jt}
-
\Delta z_{jt}
\right)^2
}{
\displaystyle
\sum_{j,t}M_{jt}w_t
}.
\]

The denominator is explicitly:

\[
\sum M w.
\]

This deliberately differs from stock Chemprop MSE's observed-entry-count denominator.

The normalized weighted loss is intended to control relative endpoint contribution without allowing overall loss scale to change merely because a differently weighted endpoint appears in a particular minibatch.

P1 loss and T121 pointwise training loss are different objectives.

Their numerical training-loss values must not be compared as though they measured the same quantity.

## Target transformation and delta inversion

For each training partition and endpoint \(t\), fit the training-only scaler:

\[
z_t
=
\frac{y_t-\mu_t}{s_t}.
\]

The pairwise training target is:

\[
\Delta z_t(A,B)
=
z_{A,t}-z_{B,t}
=
\frac{y_{A,t}-y_{B,t}}{s_t}.
\]

The training mean cancels.

At inference, convert the predicted standardized delta back to original pIC50 difference by:

\[
\widehat{\Delta y}_t(A,B)
=
s_t\widehat{\Delta z}_t(A,B).
\]

No scaler mean is added to a delta.

No validation, protected-test, or D008 label contributes to scaler fitting.

## Optimization and LR budget

Recovered D007 training-partition sizes are:

- `3751`: 1 partition;
- `3752`: 43 partitions;
- `3753`: 6 partitions.

At batch size `16`, every partition produces exactly:

\[
235
\]

T121 optimizer updates per molecule epoch.

P1 therefore defines one **pseudo-epoch** as exactly:

\[
235\ \text{optimizer updates}.
\]

A pseudo-epoch is an optimization-budget unit, not exhaustive traversal of the pair universe.

### Frozen schedule

- pair-row batch size: `16` initially;
- validation cadence: every `235` optimizer updates;
- warmup: `2350` optimizer updates;
- maximum schedule: `11750` optimizer updates;
- maximum validation opportunities: `50`;
- early-stopping patience: `20` validation checks;
- initial LR: `1.4283739994469765e-05`;
- maximum LR: `0.0008245400010426866`;
- final LR: `0.00020141638331344468`.

The Noam-like LR trajectory is mapped to the same optimizer-update breakpoints as T121.

Equal epoch counts are not claimed to establish equal training exposure.

### Required budget reporting

Every P1 cell must preserve:

- \(N\);
- each \(n_t\);
- each \(q_t\);
- each \(w_t\);
- pair attempts;
- accepted pair rows;
- `no_supervision` rejections;
- eligible target contributions by endpoint;
- weighted target contribution by endpoint;
- physical and effective batch size;
- optimizer updates;
- warmup updates;
- maximum scheduled updates;
- actual updates completed;
- validation checks;
- early-stopping state.

These quantities do not convert pairs into independent experimental observations.

## Reference-based absolute reconstruction

For held-out query \(Q\) and endpoint \(t\), references are exactly the endpoint-labeled molecules in that cell's training partition:

\[
R_t^{\mathrm{train}}.
\]

No validation, protected-test, or D008 molecule may serve as a reference for development evaluation.

For training reference \(R_j\):

\[
\widehat{\Delta y}_t(Q,R_j)
=
s_t\widehat{\Delta z}_t(Q,R_j).
\]

Then:

\[
\widehat y_t(Q\mid R_j)
=
y_{j,t}
+
\widehat{\Delta y}_t(Q,R_j).
\]

The approved P1 absolute estimator is:

\[
\widehat y_t(Q)
=
\frac{1}{|R_t^{\mathrm{train}}|}
\sum_{R_j\in R_t^{\mathrm{train}}}
\widehat y_t(Q\mid R_j).
\]

The resulting prediction is in original pIC50 units before qualified ST-RAE scoring.

Reference-derived dispersion may be recorded only as:

**uncalibrated reference-prediction dispersion.**

It has no calibrated-uncertainty, attrition, or independent selection authority.

## All-reference validation-cost benchmark

Before any long P1 development run, a bounded operational benchmark must measure all-reference validation cost on the intended execution hardware.

It must report at minimum:

- held-out molecule count;
- endpoint-specific training-reference counts;
- total pair evaluations;
- wall time;
- peak accelerator memory;
- throughput.

The benchmark is operational only.

It may establish that the approved estimator is computationally infeasible and motivate a separately approved prospective approximation.

It cannot silently alter the estimator.

D009 does **not** authorize or launch this benchmark.

## Comparable validation and checkpoint selection

P1 uses the same scientific validation quantity as T121:

**held-out absolute-pIC50 macro ST-RAE across the four CYP endpoints.**

At each validation check:

1. P1 produces held-out absolute-pIC50 predictions using training references only;
2. those predictions are scored through the existing qualified ST-RAE path;
3. the four endpoint ST-RAEs are averaged without endpoint weighting;
4. the checkpoint with the lowest validation macro ST-RAE is retained;
5. early stopping monitors that same validation macro quantity.

The retained single-model cell score \(S_c(P1)\) enters O150.

The pairwise training loss does not enter O150.

## Prospective advancement rule

The endpoint-veto question is closed:

**P1 has no endpoint-harm veto.**

Per-endpoint results are mandatory reporting but do not override the primary macro rule.

The sole advancement rule is:

\[
P1\text{ advances over T121}
\iff
O_{150}(P1)
<
0.701212576796.
\]

where:

\[
O_{150}(P1)
=
\frac{1}{150}
\sum_{c=1}^{150}S_c(P1).
\]

Any **strictly lower** O150 qualifies.

There is no minimum effect-size requirement.

### Meaning of advancement

This is a **development-selection rule**.

A P1 advancement means only that P1 achieved a lower mean validation macro ST-RAE than T121 under the frozen D007 150-cell development procedure.

It does **not** by itself establish:

- statistical significance;
- a practically meaningful effect size;
- unbiased generalization performance;
- fresh external validation;
- superiority on D008;
- or correction of CYP2D6 external behavior.

The 150 cells are correlated because they reuse the same finite development population across overlapping folds and seeds.

T121 was itself selected using this development matrix.

Therefore the P1-versus-T121 O150 comparison is not presented as fresh unbiased validation.

Per-cell paired differences and per-endpoint results must be preserved and reported as characterization, but they do not replace the frozen advancement rule.

### No endpoint veto

No endpoint-harm veto applies.

The rationale is continuity with D007's established macro development objective.

This rationale does not assert that D008 cannot motivate new prospective scientific designs. D008 may motivate new questions, but its observed labels and performance cannot select, validate, stop, or advance P1.

## Evaluation boundary

The disclosed internal test cannot become fresh validation.

D008 labels are already observed and cannot be used for:

- P1 architecture selection;
- pair sampling;
- loss or weighting selection;
- reference selection;
- hyperparameter selection;
- checkpoint selection;
- early stopping;
- advancement;
- recalibration.

No retrospective D008 rescoring is authorized by D009.

A new prospective external-generalization claim requires new unseen evidence.

## Diagnostic-integrity readiness

The post-D008 Check-1 result remains:

- member-artifact identity: verified;
- ensemble-artifact identity: verified;
- exact 150-member membership: **PASS**;
- molecule coverage across all 150 members and 750 molecules: **PASS**;
- required endpoint schema: **PASS**;
- numerical member-to-ensemble consistency: **UNRESOLVED**.

No numerical tolerance was manufactured.

Therefore:

- Check 1 is a **partial pass**;
- numerical consistency remains unresolved;
- I1 remains **NOT CLEARED**;
- no residual-risk waiver is implied by D009;
- design approval does not constitute acceptance of that unresolved numerical risk for training.

A subsequently demonstrated material technical defect requires correction and adjudication before model interpretation.

## Result-driven changes

Observed P1 results cannot retroactively change:

- the architecture;
- loss;
- sampler;
- weights;
- target transformation;
- update/LR budget;
- reference estimator;
- validation quantity;
- checkpoint-selection rule;
- or advancement rule defined here.

Results may motivate a **separately specified prospective future candidate**.

Such a candidate must receive its own methodology and evaluation boundary and cannot rewrite P1's completed comparison.

A demonstrated technical defect is different from a disappointing scientific result: a genuine defect may require correction, adjudication, and an explicitly documented disposition.

## Historical implementation-readiness gates

D009 initially authorized design only.

Before implementation authority was considered, the record required:

1. preserve the recorded Check-1 partial result and unresolved numerical limitation;
2. specify and qualify exact T121 and P1 total/trainable parameter counting;
3. review the implementation plan for the shared multicomponent encoder, antisymmetric pair head, normalized weighted loss, and sampler;
4. preregister synthetic qualification tests for:
   - molecule-level split isolation;
   - pair eligibility;
   - fold-specific \(n_t,q_t,w_t\);
   - `no_supervision` handling;
   - training-only target scaling;
   - delta inversion;
   - deterministic evaluation antisymmetry;
   - deterministic evaluation self-zero;
   - train-mode finite forward/backward behavior without symmetry claims;
   - training-reference-only reconstruction;
   - all-reference arithmetic averaging;
   - absolute-pIC50 ST-RAE routing;
5. specify the bounded all-reference validation-cost benchmark;
6. verify that no D008 information enters implementation qualification or advancement logic.

The subsequent D009 implementation and compute evidence closed these gates without reopening the scientific design or advancement rule.

Implementation approval did not by itself constitute production-training approval.

Production execution and closure are recorded below as frozen evidence.

### Implementation record — representative P1 runtime benchmark

**Record date:** 2026-11-08

The accepted D009 P1 training-runtime benchmark artifact is:

`/scratch/user/mhuang8/p1_training_runtime_rep01_fold01_seed20261101_warm10_measured100.json`

SHA-256:

`f918775490b054de548f8596cb7c68155b105e91f3c12a9bf05eaf79ebfd32df`

The benchmark used one frozen D007 development cell on one NVIDIA L40S and ran 10 warmup optimizer updates followed continuously by 100 measured optimizer updates. Warmup was excluded from the measured timing and optimizer/scheduler state was not reset between warmup and measured updates.

The complete measured-region training timing was:

`0.05273915853991639` seconds per optimizer update.

The representative extrapolation for the frozen P1 training-update budget is:

`11,750 updates/cell × 150 cells = 1,762,500 training updates`

which projects to approximately:

`25.82` GPU-hours for measured training-update work.

The separately closed D009 all-reference validation-inference projection is approximately:

`4.40` GPU-hours.

The combined approximately `30.22` GPU-hours is recorded only as a **representative measured-component subtotal** for training updates plus validation inference. It is not a complete experiment-runtime measurement, not a proven upper bound, and has no validation-performance, checkpoint-selection, stopping, model-selection, calibration, or scientific advancement authority.

The runtime benchmark excludes loading, caller-side preprocessing, checkpoint/artifact serialization, and validation scoring where those operations were not part of the measured region. The one-cell / one-L40S / 30-minute execution layout remains prospective only and is not itself a production submission authorization.

Implementation gates qualified to date:

- all-reference validation-cost readiness: **PASS**;
- P1 training/orchestration synthetic qualification: **PASS**;
- bounded P1 training-runtime benchmark: **accepted as representative compute evidence**;
- checkpoint save/reload/inference and artifact/failure qualification: **PASS**;
- production cell wrapper, 150-cell preflight/orchestration, and completion aggregation synthetic qualification: **PASS**.

This implementation record did not itself authorize production. The subsequent D009 production execution is recorded below and is now frozen.

## Production execution and integrity closure

D009 P1 production ran on CoreHPC as Slurm array job `2958137`.

The execution covered exactly the 150 frozen D007 development cells:

- 5 repetitions;
- 10 folds per repetition;
- 3 training seeds per repetition/fold cell;
- one P1 cell per task;
- one NVIDIA L40S per cell.

Transferred production log evidence contains 150 `2958137_*.out` task logs with task IDs `0..149`; each records `NVIDIA L40S` and `=== CELL COMPLETE ===`.

The post-production integrity gate is:

`artifacts/d009_closure_workspace/p1_d009_postproduction_preflight_20261108.json`

SHA-256:

`9ea445ecdabf07f13884a89ddb8d2dd9854b2d3ba9657055e123999f421adbd3`

Frozen state:

- 150 expected cells;
- 150 complete;
- 0 absent;
- 0 incomplete/failed;
- protected internal test excluded;
- D008 excluded;
- performance-driven scheduling `false`.

The transferred local closure package intentionally contains lightweight per-cell evidence only: 150 `manifest.json`, 150 `metrics.json`, 150 `run_cell.json`, and 150 `SHA256SUMS.txt`. It contains 0 local `best.ckpt` payloads by design.

## Final P1 versus T121 result

The authoritative final P1 O150 artifact is:

`artifacts/d009_closure_workspace/p1_d009_o150_result_20261108.json`

SHA-256:

`b277f4ccd9138b688ff20bfaf2378ebfcb66bc118ab5943df48027168adc0e2b`

It preserves the frozen advancement result:

| Quantity | Value |
| --- | ---: |
| P1 O150 | `0.7074933757653159` |
| T121 O150 | `0.701212576796` |
| P1 minus T121 | `+0.006280798969315882` |
| P1 advances | `false` |

The final decision is that **P1 does not advance** and **D007-T121 remains preferred**.

Characterization has no advancement authority. It cannot alter the frozen macro O150 rule or create an endpoint-level veto or rescue criterion.

## Required closure characterization

The D009 closure characterization artifact is:

`artifacts/d009_closure_workspace/d009_p1_closure_characterization_20261109.json`

SHA-256:

`611ec29969b675fa46de710ca14ec206c6ca3fff8006f5b199efced1e79a50cd`

The recovered D007-T121 comparator package is:

`artifacts/d009_closure_workspace/d009_t121_comparator_20261109/`

Its transfer archive is:

`artifacts/d009_freeze_transfer/d009_t121_comparator_20261109.tar.gz`

SHA-256:

`580b44955b170a4e30647b537ffe54940374d7e4fc945565f1ec4d5ba002ae60`

The comparator package contains 150 `result.json`, 150 `cell_plan.json`, and 150 `provenance.json` files. Cells were paired by explicit `(repetition, fold, training_seed/seed)` identity, not by directory order. No duplicate, missing, or extra identities were detected.

As a direct internal consistency check, the arithmetic mean of the recovered 150 T121 cell macro ST-RAE values is:

`0.7012125767956957`

The direct difference from the frozen scalar `0.701212576796` is:

`-3.043121310497554e-13`

No new numerical tolerance is introduced and this check is not a new gate.

The identity-matched paired macro difference summary, P1 minus T121, is:

| Statistic | Value |
| --- | ---: |
| n | `150` |
| mean | `0.0062807989696201825` |
| median | `0.007745555153990058` |
| sample SD | `0.017458604390571973` |
| minimum | `-0.03154866274296275` |
| maximum | `0.056337123105406506` |
| P1 lower count | `56` |
| P1 equal count | `0` |
| P1 higher count | `94` |

The four identity-matched per-endpoint paired comparisons are:

| Endpoint | P1 mean | T121 mean | Mean P1 minus T121 | Median diff | P1 lower | P1 higher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CYP1A2 | `0.8155572297028921` | `0.8039443572593442` | `0.01161287244354784` | `0.010018096511007213` | `62` | `88` |
| CYP2C9 | `0.6158902956632285` | `0.6098174041358858` | `0.006072891527342696` | `0.0027106702077795974` | `73` | `77` |
| CYP2D6 | `0.9119082278829614` | `0.9183171939206107` | `-0.006408966037649143` | `-0.00915635314767771` | `83` | `67` |
| CYP3A4 | `0.48661774981218114` | `0.4727713518669417` | `0.013846397945239352` | `0.013565696223148294` | `50` | `100` |

Lower ST-RAE is better. This characterization records that P1 is higher on average for macro ST-RAE and for CYP1A2, CYP2C9, and CYP3A4, while lower on average for CYP2D6. No mechanism is assigned.

## Evidence inventory and provenance

The repository-side D009/P1 evidence inventory is:

`artifacts/d009_closure_workspace/d009_p1_evidence_inventory_20261109.json`

SHA-256:

`116e39fc3289832450c3122f7a38dee0bbe1e6e93d392b387bdafc39b852568c`

The inventory records local SHA-256 hashes for the lightweight closure evidence, including:

- all 150 transferred P1 cell manifests;
- all 150 P1 metrics files;
- all 150 P1 run-cell records;
- all 150 P1 per-cell `SHA256SUMS.txt` files;
- all 150 T121 comparator result files;
- all 150 T121 comparator cell plans;
- all 150 T121 comparator provenance files;
- checkpoint identities recorded by those cell artifacts;
- final production source/test identities;
- qualification evidence preserved in `p1_corehpc_stage_20261107/`;
- validation-cost benchmark;
- training-runtime benchmark;
- initial production preflight;
- production sbatch wrapper;
- transferred production logs/accounting evidence;
- post-production preflight;
- authoritative O150 artifact;
- final characterization artifact;
- lightweight transfer archive SHA-256;
- T121 comparator transfer archive SHA-256;
- full CoreHPC checkpoint-bearing freeze path and SHA-256.

The lightweight P1 transfer archive is:

`artifacts/d009_freeze_transfer/d009_p1_closure_20261108.tar.gz`

SHA-256:

`ffeb387da7f3ab746bf61db7661855b29bf93a3eb97b9cac366f79478e2f3243`

The full checkpoint-bearing CoreHPC freeze remains:

`/scratch/user/mhuang8/d009_p1_freeze_20261108.tar.gz`

SHA-256:

`f77e6815b131da5caf69886e367a538f896e0aa7ee6e42076ae84a1ed3dc8d3e`

The inventory distinguishes local lightweight evidence from checkpoint payloads. It does not claim local checkpoint-byte verification because checkpoint bytes are not present in the local closure package.

## Compute evidence preserved

The accepted bounded training-runtime benchmark remains:

`artifacts/d009_closure_workspace/p1_training_runtime_rep01_fold01_seed20261101_warm10_measured100.json`

SHA-256:

`f918775490b054de548f8596cb7c68155b105e91f3c12a9bf05eaf79ebfd32df`

It preserves:

- complete measured timing: `0.05273915853991639` seconds per optimizer update;
- approximately `25.82` GPU-hours projected training;
- approximately `4.40` GPU-hours separately projected validation inference;
- approximately `30.22` GPU-hours representative measured-component subtotal;
- subtotal is not complete experiment runtime and not a proven upper bound.

The accepted all-reference validation-cost benchmark remains:

`artifacts/d009_closure_workspace/p1_real_benchmark_rep01_fold01_seed20261101_chunk8192.json`

It remains operational compute evidence only and has no validation-performance, checkpoint-selection, stopping, model-selection, calibration, or advancement authority.

## Post-D008 I1 reconciliation

The existing post-D008 diagnostic/I1 record remains:

- member identity/membership: **PASS**;
- molecule coverage: **PASS**;
- endpoint schema: **PASS**;
- numerical member-to-ensemble consistency: **UNRESOLVED**;
- I1: **NOT CLEARED**.

No existing later artifact or record was found that resolves numerical member-to-ensemble consistency. No numerical tolerance was invented. No new external evaluation was performed. D008 labels were not reopened.

## Artifact preservation status

The full checkpoint-bearing freeze currently remains at:

`/scratch/user/mhuang8/d009_p1_freeze_20261108.tar.gz`

SHA-256:

`f77e6815b131da5caf69886e367a538f896e0aa7ee6e42076ae84a1ed3dc8d3e`

The repository/session record does not unambiguously identify a persistent backed-up destination for that full freeze. This is an operational item requiring human confirmation.

No CoreHPC freeze was moved or deleted by this closure record.

## Limitations

P1 does not establish the cause of D008 endpoint heterogeneity.

A P1 development win would establish only that the approved pairwise package achieved the lower frozen D007 development objective.

It would not establish that:

- pairwise learning alone caused the difference;
- generated pairs created additional independent experimental information;
- CYP2D6 external generalization was repaired;
- external distribution shift caused D008;
- D008 would have improved prospectively;
- reference dispersion is calibrated uncertainty;
- or the O150 difference is statistically significant or practically important.

## Governing principle

Change the learning formulation while preserving the strongest available controls: identical molecule-level development partitions, training-only transformations and references, prospectively fixed pair sampling and weighting, matched optimizer-update/LR budget, identical held-out absolute-pIC50 validation quantity, and the existing qualified ST-RAE metric.

Generated pairs increase training representations, not experimental information.

D007 and D008 remain frozen evidence.

## Revisit when

Revisit D009 only if a demonstrated technical or integrity defect requires adjudication, correction, and an explicitly documented disposition.

Do not revisit D009 merely because P1 lost scientifically, because an endpoint-level characterization is disappointing, or because a future candidate design is desired.

The frozen result may not be changed by rerun, rescue, endpoint veto, alternative aggregation, recalibration, result-driven modification, or external rescoring.

Any future model concept must be specified prospectively outside D009 and cannot rewrite this closure.

# D010 — P2 Four-Isoform Exact-ANIL Meta-Regression

**Decision date:** 2026-11-09

**Status:** Complete / frozen

## Purpose

D010 is complete and frozen. P2 was implemented, qualified, benchmarked, run once across the 150 frozen D007 development cells, characterized against the recovered D007-T121 comparator evidence, and closed without advancement.

The frozen final result is:

- P2 O150: `0.7106082085193788`
- T121 O150: `0.701212576796`
- P2 minus T121: `+0.0093956317233788`
- P2 advances: `false`
- preferred configuration: **D007-T121**

No P2 rerun, first-order fallback, endpoint rescue, checkpoint reselection, rescoring, split generation, or post-result modification is authorized.

D010 governs P2, a prospective four-isoform exact-ANIL meta-regression candidate evaluated against the frozen D007-T121 comparator.

P1 is closed under D009 and is not modified, rerun, rescued, or incorporated into P2.

P2 tests whether a shared molecular representation and task-head initialization, meta-learned specifically so that one support-gradient step adapts an isoform's output mapping, improves new-compound prediction across the four known CYP isoforms relative to frozen D007-T121.

The four tasks are exactly:

- CYP1A2 direct-inhibition pIC50;
- CYP2C9 direct-inhibition pIC50;
- CYP2D6 direct-inhibition pIC50;
- CYP3A4 direct-inhibition pIC50.

P2 concerns prediction for new compounds within these four known isoforms. It does not test or claim few-shot generalization to a previously unseen isoform.

## Development evidence and comparator

P2 reuses the exact frozen D007 development procedure:

- 50 frozen repetition/fold molecule partitions;
- three training seeds: `20261101`, `20261102`, and `20261103`;
- exactly 150 development cells.

The sole comparator is frozen D007-T121:

\[
O_{150}(\mathrm{T121}) = 0.701212576796.
\]

The 150 D007 cells are reused development evidence. They are not untouched confirmation data, and their repeated use does not make them independent external evidence.

No additional model baseline or automatic ablation campaign is introduced by D010.

## Data firewall

For every P2 cell, the frozen compound-level training/validation boundary applies across all endpoints.

Validation molecules and validation labels may not enter:

- meta-training;
- episode construction;
- target-scaler fitting;
- support adaptation.

Held-out validation labels may be used only for the specified:

- ST-RAE scoring;
- checkpoint selection;
- early stopping;
- prospective P2 development-advancement rule.

No validation molecule or validation label may contribute to an inner adaptation step.

The protected internal test cannot enter any P2 procedure.

D008 evidence cannot enter any P2 procedure, including model construction, training, adaptation, checkpoint selection, stopping, advancement, or post-result rescue.

New partitions of molecules already used in development do not create independent confirmation. Any future claim of independent external generalization requires genuinely untouched compounds or assays.

## Model architecture

Each P2 cell is trained from scratch. Trained T121 weights are not imported.

P2 reuses the T121 Chemprop 2.3.1 architecture where applicable:

- BondMessagePassing;
- message-passing depth 4;
- molecular representation dimension 600;
- normalized aggregation;
- aggregation norm 100;
- ReLU;
- no batch normalization;
- shared `600 -> 2000` hidden layer;
- ReLU;
- dropout 0.4;
- final `2000 -> 4` linear regression output.

Let:

- \(\theta\) denote the molecular encoder parameters;
- \(\psi\) denote the shared `600 -> 2000` hidden-layer parameters;
- \(w_t,b_t\) denote the final linear output weight vector and scalar bias for isoform \(t\).

The shared encoder and hidden layer are meta-learned across the four isoform tasks.

## Inner-adaptation boundary

For task \(t\), inner adaptation changes only:

- \(w_t\), the selected isoform's final output weight vector;
- \(b_t\), the selected isoform's final output bias.

Exactly one inner gradient step is taken:

\[
(w'_t,b'_t)
=
(w_t,b_t)
-
0.01\nabla_{w_t,b_t}L^{S}_t.
\]

The inner learning rate is therefore fixed at:

\[
\alpha = 0.01.
\]

No inner update is permitted to:

- encoder parameters \(\theta\);
- shared hidden-layer parameters \(\psi\);
- final-output parameters belonging to the other three isoforms.

Final-output-only adaptation and exact-versus-first-order differentiation are separate methodological choices. D010 freezes final-output-only adaptation together with exact differentiation through that adaptation.

## Episode construction

Every outer optimizer update contains exactly four episodes:

1. one CYP1A2 episode;
2. one CYP2C9 episode;
3. one CYP2D6 episode;
4. one CYP3A4 episode.

There is no independent stochastic task selection and no task-frequency weighting.

For each task \(t\), an episode contains:

- 13 support molecules;
- 3 query molecules.

Eligible molecules must:

- belong to that cell's training partition;
- have an observed label for the selected isoform.

Support and query molecules are molecule-disjoint.

Compound identity defines the support/query boundary across endpoints. A compound assigned to one side of an episode cannot supply information across that boundary through another endpoint.

Only the selected isoform's label is consumed by that task episode. Missing task labels are not imputed.

Each outer update therefore contains:

- four task episodes;
- 52 support-molecule appearances;
- 12 query-molecule appearances;
- 64 sampled molecule appearances in total.

The 13-support/3-query construction is the frozen operationalization of the proposed approximately 80:20 support/query ratio at the inherited 16-molecule scale.

## Target transformation and support loss

P2 reuses the existing cell-specific training-only target-scaling machinery.

For task \(t\), the support objective is the arithmetic mean squared error over its 13 observed standardized support labels:

\[
L^{S}_t
=
\frac{1}{13}
\sum_{i\in S_t}
(\hat z_{i,t}-z_{i,t})^2.
\]

No validation information participates in scaler fitting.

Exactly one inner step at learning rate 0.01 is taken from this support objective.

## Meta-training dropout

Dropout 0.4 remains active during both:

- meta-training support forward passes;
- meta-training query forward passes.

Ordinary dropout behavior is used.

D010 introduces no coupled-dropout-mask machinery or other mask-control mechanism.

## Exact-ANIL outer objective

Following the one-step support adaptation for task \(t\), query loss is the arithmetic mean standardized MSE over its three molecule-disjoint query compounds:

\[
L^{Q}_t
=
\frac{1}{3}
\sum_{i\in Q_t}
(\hat z'_{i,t}-z_{i,t})^2.
\]

The four task query losses contribute equally:

\[
L_{\mathrm{outer}}
=
\frac{1}{4}
\left(
L^{Q}_{1A2}
+
L^{Q}_{2C9}
+
L^{Q}_{2D6}
+
L^{Q}_{3A4}
\right).
\]

P2 uses exact ANIL.

Differentiation is retained through the support-gradient calculation. The support gradient must not be detached or replaced with a first-order approximation.

Consequently, the outer gradient includes the dependence of the adapted \(w'_t,b'_t\) on:

- their meta-initialization;
- the support features produced by \(\theta\) and \(\psi\).

Therefore:

- \(\theta\) receives no inner update but receives exact through-adaptation outer query gradients;
- \(\psi\) receives no inner update but receives exact through-adaptation outer query gradients;
- \(w_t,b_t\) receive the one-step inner update and corresponding exact meta-gradient;
- the other three task outputs are not inner-adapted by task \(t\).

## Rationale for exact ANIL

Exact ANIL is selected for fidelity to P2's defined post-support-adaptation query objective, not because exact ANIL has been demonstrated to outperform first-order ANIL for CYP regression.

P2 explicitly defines one support-gradient step as the mechanism by which an isoform conditions its predictor. Exact ANIL retains the derivative through that defined adaptation operation when optimizing subsequent query performance.

Collins et al. provide precedent that both exact and first-order ANIL variants can learn shared representations. That work does not establish superiority of exact ANIL for the present four-isoform CYP regression problem.

First-order ANIL therefore remains an unrejected alternative. Under D010 it has no:

- automatic fallback authority;
- automatic comparison-campaign authority;
- post-result rescue authority.

D010 does not assume that the 13-molecule support size or dropout favors exact over first-order ANIL. It does not assume that an inner learning rate of 0.01 makes the omitted first-order term negligible. It also makes no claim about the incremental runtime or memory burden of exact differentiation before that cost is measured.

## Outer optimization

The four task episodes jointly constitute exactly one outer optimizer update.

The task episodes may execute sequentially for memory efficiency. If they do, each task contributes exactly one quarter of its query objective to accumulated outer gradients.

There is:

- no optimizer step between task episodes;
- no scheduler step between task episodes;
- exactly one Adam optimizer step after all four task episodes;
- exactly one NoamLR scheduler step after all four task episodes.

Sequential execution is a computational implementation of the frozen equal four-task objective and may not change task weighting.

## Training budget and learning-rate schedule

One P2 pseudo-epoch is defined as:

\[
235
\]

outer optimizer updates.

Because each outer update contains four task episodes, one pseudo-epoch contains:

\[
235\times4=940
\]

task episodes.

The frozen schedule is:

- validation every 235 outer updates;
- warmup through outer update 2,350;
- maximum 11,750 outer updates;
- maximum 50 validation opportunities;
- early-stopping patience 20 validation checks;
- `min_delta = 0`.

The outer optimizer is Adam.

The outer NoamLR trajectory is:

- initial learning rate `1.4283739994469765e-05`;
- maximum learning rate `0.0008245400010426866`;
- maximum reached at outer update 2,350;
- final learning rate `0.00020141638331344468`;
- final schedule point at outer update 11,750.

These values preserve T121's optimizer-update and learning-rate breakpoints. They do not assert equivalent molecule exposure between T121 and P2.

The inner learning rate remains fixed at 0.01 and is not controlled by NoamLR.

## Evaluation-time adaptation

Dropout is disabled during:

- evaluation-time support adaptation;
- validation prediction.

At every validation check, each isoform begins from a separate fresh copy or equivalent fresh state of the checkpoint's unadapted meta-parameters.

Adapted parameters may not carry:

- from one isoform to another;
- from one validation check to another.

For task \(t\):

1. use all cell-training molecules having an observed label for \(t\);
2. compute the exact arithmetic-mean standardized MSE over that complete training-support population;
3. take exactly one inner gradient step at learning rate 0.01;
4. update only \(w_t,b_t\);
5. predict endpoint \(t\) for every held-out validation molecule;
6. invert the existing training-only scaler to original pIC50.

If the full support population must be processed in minibatches for computational reasons, the implementation must accumulate the complete support squared-error sum and observation count and divide by the complete support count so that the single adaptation gradient corresponds to the arithmetic mean over the entire support population.

Computational minibatching may not become multiple sequential adaptation steps.

No held-out validation molecule or validation label participates in support adaptation.

A validation molecule's four endpoint predictions therefore arise from four independently task-adapted versions of the same unadapted checkpoint state.

## Validation scoring, checkpoint selection, and stopping

The four original-pIC50 validation predictions pass through the existing qualified ST-RAE evaluation machinery.

For development cell \(c\):

\[
S_c(P2)
=
\frac{1}{4}
\sum_{t=1}^{4}
ST\!-\!RAE_{c,t}(P2).
\]

The retained checkpoint is the checkpoint with the strictly lowest validation macro ST-RAE.

Early stopping monitors exactly the same quantity with:

- patience 20 validation checks;
- `min_delta = 0`.

Validation labels have authority only for this specified scoring, checkpoint-selection, early-stopping, and subsequent prospective development-advancement procedure.

The meta-training objective has no direct advancement authority.

## Prospective advancement rule

After exactly the frozen 150 development cells:

\[
O_{150}(P2)
=
\frac{1}{150}
\sum_{c=1}^{150}S_c(P2).
\]

The sole comparator is:

\[
O_{150}(\mathrm{T121})=0.701212576796.
\]

P2 advances if and only if:

\[
\boxed{
O_{150}(P2)<0.701212576796
}
\]

An exact tie does not advance.

There is no:

- endpoint veto;
- minimum effect-size requirement;
- secondary model comparator;
- automatic ablation campaign;
- first-order fallback;
- D008 criterion;
- post-result rescue.

Per-cell and per-isoform results are characterization only and cannot override the advancement rule.

## Claim boundary

If P2 advances, the supported claim is limited to:

> P2 achieved lower mean validation macro ST-RAE than frozen D007-T121 under the reused frozen D007 150-cell development procedure.

Advancement does not establish:

- independent external generalization;
- independence of the 150 development cells;
- a causal mechanism;
- correction of the previously observed D008 CYP2D6 behavior.

D008 is already-observed evidence and cannot validate P2.

Independent confirmation requires genuinely untouched future compounds or assays.

## Connected implementation and qualification

After D010 is frozen in the repository, one connected P2 implementation/qualification package is authorized.

Qualification is a single connected unit rather than a sequence of helper-by-helper scientific approvals.

It must demonstrate at minimum:

1. exact frozen four-task identity and order;
2. exactly one episode per task per outer update;
3. exact 13-support/3-query accounting;
4. molecule-disjoint support/query membership;
5. compound-level support/query boundary across endpoints;
6. training-partition-only episode eligibility;
7. training-only target-scaler fitting;
8. exclusion of validation data from meta-training, episode construction, scaler fitting, and support adaptation;
9. exclusion of the protected internal test and D008 from all P2 procedures;
10. no missing-label imputation;
11. only the selected task's \(w_t,b_t\) change during inner adaptation;
12. encoder, shared hidden layer, and other task outputs receive no inner update;
13. the support-gradient calculation remains differentiable;
14. outer gradients include the dependence of adapted output parameters on support features and their initialization;
15. encoder and shared hidden-layer parameters receive finite exact outer meta-gradients;
16. all four query losses contribute exactly one quarter to the outer objective;
17. dropout is active during meta-training support and query passes;
18. dropout is disabled during evaluation adaptation and prediction;
19. no dropout-mask coupling is introduced;
20. exactly one optimizer step and one scheduler step occur per four task episodes;
21. the `235 / 2,350 / 11,750` NoamLR update mapping is exact;
22. evaluation starts from fresh unadapted checkpoint state independently for every isoform;
23. full-training-support evaluation adaptation implements the exact arithmetic-mean support objective, including when computationally minibatched;
24. validation labels are used only for authorized scoring, checkpoint selection, early stopping, and development advancement;
25. strict-best macro-ST-RAE checkpoint behavior;
26. save/reload/inference equivalence;
27. deterministic frozen 150-cell routing;
28. strict synthetic O150/T121 advancement arithmetic;
29. malformed, nonfinite, incomplete, or mismatched inputs fail closed.

Qualification does not authorize production training.

## Bounded exact-ANIL runtime and memory benchmark

If the connected qualification package passes, a bounded real-workload exact-ANIL runtime/memory benchmark is authorized on the intended L40S outside SLURM.

The benchmark has compute-planning authority only. It has no model-performance or scientific-selection authority.

It must measure:

- complete four-task exact-ANIL outer-update timing, including support forward passes, differentiable inner-gradient calculations, query forward passes, exact meta-backward, Adam update, and NoamLR update;
- timing stability after warmup;
- peak allocated GPU memory;
- peak reserved GPU memory;
- complete four-isoform evaluation-time full-training-support adaptation plus validation-input prediction cost;
- sufficient accounting to project training through 11,750 outer updates per cell;
- sufficient accounting to project validation through at most 50 validation checks per cell;
- the complete 150-cell measured-component workload.

Warmup and measured updates must form one continuous optimizer/scheduler trajectory; benchmark state may not be reset between them.

The benchmark may use held-out validation molecule inputs solely to measure prediction cost. It may not access validation labels or calculate validation performance.

Timing drift must be reported descriptively. D010 establishes no invented drift threshold.

The benchmark must state which costs are measured and which remain excluded, including as applicable:

- data loading;
- preprocessing;
- checkpoint loading;
- checkpoint serialization;
- artifact writing;
- validation scoring;
- other caller-side or orchestration overhead.

Any extrapolation is a representative measured-component projection, not a measured full experiment or a proven runtime upper bound.

Before any production submission, the following must be shown together:

- projected training workload per cell;
- projected validation workload per cell;
- projected total measured-component workload across 150 cells;
- projected per-job workload;
- proposed requested production wall time;
- measured versus unmeasured cost boundary.

P1's production wall time is not automatically inherited by P2.

## Production authority

### Implementation record — complete-model bounded benchmark and production-readiness integration

**Record date:** 2026-11-09

The accepted corrected D010 complete-model bounded benchmark artifact is:

`cyp/artifacts/cyp_010/p2_d010_complete_model_benchmark_rep01_fold01_seed20261101_warm10_measured100.json`

SHA-256:

`b44457583ab37b14a0f36b38b458cf4874f6fe2da7f742c5f81af4f80bf7b9f4`

Artifact type:

`D010_P2_exact_anil_complete_model_bounded_benchmark_v2`

This corrected benchmark supersedes the earlier head-only diagnostic benchmark for runtime projection. The prior head-only benchmark timed `P2MetaHead` over synthetic 600-dimensional embeddings, did not instantiate or optimize the Chemprop encoder, and has no D010 runtime-projection authority.

The accepted corrected benchmark used one NVIDIA L40S and ran 10 continuous warmup outer updates followed by 100 measured outer updates without resetting optimizer or scheduler state.

Parameter accounting:

- complete P2 model total/trainable parameters: `2,025,404`;
- encoder total/trainable parameters: `815,400`;
- meta-head total/trainable parameters: `1,210,004`.

Measured timing and memory:

- mean complete four-task outer update: `0.04574970336048864` seconds;
- early-window mean: `0.045825388645894374` seconds;
- late-window mean: `0.04577984815731532` seconds;
- late-minus-early drift: `-0.00004554048857905585` seconds;
- peak allocated GPU memory: `9,301,794,304` bytes;
- peak reserved GPU memory: `9,500,098,560` bytes;
- four-isoform validation adaptation/prediction: `2.6993121129926294` seconds per check.

Representative measured-component projections:

- projected training: `537.5590144857415` seconds per cell;
- projected <=50 validation checks: `134.96560564963147` seconds per cell;
- combined measured components: `672.524620135373` seconds per cell, approximately `11.21` minutes;
- 150-cell measured-component projection: `100878.69302030595` seconds, approximately `28.02` GPU-hours.

These are representative measured-component projections only. They are not complete experiment runtime measurements and not proven upper bounds.

Excluded costs include one-time loading, SMILES parsing, static graph featurization/cache construction, checkpoint I/O, artifact writing, validation scoring/ST-RAE, checkpoint-selection/early-stopping overhead, and orchestration.

The prospective production topology is one cell per one NVIDIA L40S. The proposed requested wall time is 20 minutes per cell. This requested wall time is prospective and not a demonstrated upper bound.

The connected local production-readiness integration implements and synthetically qualifies:

- complete P2 production cell runner;
- deterministic 150-cell wrapper/preflight;
- checkpoint/artifact layout with manifest, `best.ckpt`, metrics, and `SHA256SUMS.txt`;
- completed/absent/incomplete-failed cell semantics;
- integrity-verified aggregation to the frozen O150 rule;
- synthetic toy-scale production-wrapper qualification.

This implementation record does not authorize real P2 production training, Slurm submission, or development-performance exposure.

### Closure record — production result and characterization

**Closure date:** 2026-11-09

D010 P2 production evidence is frozen from existing artifacts only. No training, inference, rescoring, checkpoint reselection, split generation, scientific amendment, first-order fallback, endpoint rescue, or post-result modification was performed during closure.

Authoritative local D010 artifacts are under:

`cyp/artifacts/cyp_010/`

Verified SHA-256 values:

- corrected complete-model benchmark: `b44457583ab37b14a0f36b38b458cf4874f6fe2da7f742c5f81af4f80bf7b9f4`
- postproduction preflight: `593b77b9b757c80d58af74d9dbdc45eb20df552a43b3b4468c30b89d08c34fbc`
- frozen 150-cell metrics: `8cf466eff7e123356f085f75b8559b40b6f29154c9fed72c23ebb7a08b70fa55`

The requested separate O150 result SHA-256 `07362bb006e6130de436945493f0ec2479c00fa2ca582671f67ed5db35c1c1be` was not physically present as a local file under `cyp/artifacts/cyp_010/` at closure time. The final O150 value is reconstructed directly from the verified frozen 150-cell metrics artifact.

Production integrity evidence records:

- CoreHPC Slurm job `2968876`;
- `150/150` tasks completed;
- 150 `CELL COMPLETE` markers;
- 0 nonempty error logs;
- postproduction preflight: 150 complete, 0 absent, 0 incomplete/failed;
- protected internal test excluded;
- D008 excluded;
- performance-driven scheduling `false`.

The closure characterization artifact is:

`cyp/artifacts/cyp_010/d010_p2_closure_characterization_20261109.json`

SHA-256:

`0cf25012223d31673e9a82dd89334124c24dcd6400da494aeded0abd53aeaf81`

The evidence inventory is:

`cyp/artifacts/cyp_010/d010_p2_evidence_inventory_20261109.json`

SHA-256:

`432a7012fb70985df91441dc78b5999af739059ec3451615efc19cfd59ec8270`

P2 cells were paired to D007-T121 cells only by explicit `(repetition, fold, training_seed/seed)` identity. The characterization verified 150 P2 identities, 150 T121 identities, exact set equality, and zero duplicate, missing, or extra identities.

The final frozen advancement result is:

| Quantity | Value |
| --- | ---: |
| P2 O150 | `0.7106082085193788` |
| T121 O150 | `0.701212576796` |
| P2 minus T121 | `+0.0093956317233788` |
| P2 advances | `false` |

The final decision is that **P2 does not advance** and **D007-T121 remains preferred**.

Per-endpoint paired characterization, using each model's already-selected macro-ST-RAE checkpoint, is:

| Endpoint | T121 mean | P2 mean | Mean P2 minus T121 | P2 improved | P2 worsened | Exact ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CYP1A2 | `0.8039443572593442` | `0.8099969589372771` | `+0.006052601677932758` | `63` | `87` | `0` |
| CYP2C9 | `0.6098174041358858` | `0.6210036227472333` | `+0.011186218611347542` | `67` | `83` | `0` |
| CYP2D6 | `0.9183171939206107` | `0.9206907011843659` | `+0.002373507263755171` | `66` | `84` | `0` |
| CYP3A4 | `0.4727713518669417` | `0.49074155120863866` | `+0.01797019934169697` | `44` | `106` | `0` |

All four endpoint mean differences favor T121. CYP3A4 has the largest mean degradation, followed by CYP2C9, CYP1A2, and CYP2D6. This is characterization only; no mechanism is assigned and no endpoint criterion or rescue rule is created.

Checkpoint bytes are not preserved locally by this closure. The checkpoint-bearing production root is recorded as:

`/scratch/user/mhuang8/p2_d010_production_20261109`

The repository/session record does not unambiguously identify a persistent backed-up destination for the checkpoint-bearing production tree. This is an operational preservation item requiring human confirmation.

D010 does not authorize production P2 training or submission.

Production training requires separate human approval after:

1. the connected implementation/qualification package passes; and
2. the bounded exact-ANIL L40S runtime/memory benchmark and full workload projection have been reviewed.

A qualification failure or operational benchmark problem does not authorize silent scientific modification.

## Governing principle

P2 must test the prospectively frozen exact-ANIL hypothesis as written.

Implementation may realize the frozen methodology but may not silently alter:

- data authority;
- task definition;
- episode construction;
- adaptation boundary;
- exact through-adaptation differentiation;
- task weighting;
- dropout policy;
- optimization budget;
- evaluation adaptation;
- checkpoint-selection quantity;
- advancement rule;
- evidence or claim boundary.

First-order ANIL remains an unrejected alternative, but D010 gives it no fallback, comparison, or rescue authority.

## Revisit when

Revisit D010 only if:

- a demonstrated technical or integrity defect invalidates an assumption required by the frozen design;
- a required frozen dependency changes before production;
- the bounded exact-ANIL benchmark demonstrates an operational infeasibility requiring explicit human adjudication before any approximation is considered;
- or a future separately approved scientific decision supersedes D010.

A disappointing P2 result is not grounds to alter or rerun D010.

First-order ANIL is not an automatic fallback or post-result rescue.

# D011 — Bounded Exact-ANIL Inner-Adaptation Tuning Study

**Decision date:** 2026-11-09

**Status:** Approved — prospective tuning design frozen; implementation and production not yet authorized.

## Purpose

D011 is a separately declared tuning study downstream of closed D010.

D010 remains closed. P2 did not advance, and D007-T121 remains the preferred configuration.

D011 asks one comparative question:

> Does any configuration in the predeclared exact-ANIL inner-adaptation grid achieve lower O150 than frozen D007-T121?

D011 does not test or establish that D010's original one-step, inner-LR-0.01 adaptation setting caused D010's loss.

## Frozen candidate grid

D011 varies exactly two quantities:

- inner adaptation learning rate: `{0.001, 0.003, 0.01, 0.03}`;
- number of inner adaptation steps: `{1, 2, 3}`.

The full factorial grid therefore contains exactly 12 configurations.

No other D010 scientific parameter changes.

The LR range and spacing are prospective bounded tuning choices around D010's `0.01` setting; they are not claimed CYP-specific optima.

The 1–3-step range is a prospective bounded test of adaptation depth. It does not imply that additional steps are expected to improve performance.

## D010 anchor reuse

The configuration:

- inner LR = `0.01`;
- inner steps = `1`;

is exactly the frozen D010 adaptation configuration.

Its existing frozen 150-cell D010 results are reused as the D011 anchor rather than retrained, provided implementation preparation confirms that the D011 candidate protocol is identical to D010 for this configuration with respect to data partitions, architecture, training procedure, adaptation procedure, checkpoint selection, scoring, and aggregation.

If a concrete protocol incompatibility is identified before execution, it must be documented and adjudicated before any anchor rerun. The anchor may not be silently rerun for symmetry.

Subject to that compatibility check:

- configurations compared = 12;
- newly trained configurations = 11;
- frozen cells per configuration = 150;
- new production cells = 1,650.

## Candidate protocol

For candidate \(c=(\alpha,K)\), the same inner learning rate \(\alpha\) and number of adaptation steps \(K\) are used during both meta-training and evaluation-time adaptation.

During each meta-training task episode, exactly \(K\) sequential exact-ANIL inner steps are taken on the selected isoform's final output weight vector and bias only.

Differentiation is retained exactly through all \(K\) adaptation steps.

During evaluation, each isoform begins from a fresh unadapted selected checkpoint state and receives exactly \(K\) full-training-support adaptation steps at the same candidate-specific inner learning rate.

No adapted state carries between isoforms or validation checks.

All other D010 methodology remains frozen, including:

- four known CYP isoforms as tasks;
- complete Chemprop encoder and shared hidden layer;
- final-output-only inner adaptation;
- exact through-adaptation differentiation;
- 13 support / 3 query molecules per task episode;
- one episode per isoform per outer update;
- equal four-task outer weighting;
- dropout policy;
- Adam and NoamLR outer optimization;
- optimizer-update budget and validation schedule;
- frozen 150 development cells;
- training-only scalers;
- data firewall;
- macro ST-RAE checkpoint selection;
- early-stopping rule;
- T121 comparator.

No architecture change, first-order branch, additional baseline, endpoint veto, or secondary advancement criterion is introduced.

## Candidate scoring and selection

For every candidate \(c\):

\[
O_{150}(c)
=
\frac{1}{150}
\sum_{i=1}^{150} S_i(c),
\]

where each \(S_i(c)\) is the retained macro ST-RAE from that cell's already-defined macro-selected checkpoint.

After all 12 declared configurations have complete eligible results, select the configuration with the strictly lowest O150.

If two or more configurations have exactly equal O150, break the tie deterministically by:

1. fewer inner adaptation steps;
2. then smaller inner learning rate.

No numerical tolerance is introduced for candidate ties.

Candidate evaluation order has no selection authority, and the study does not stop when a candidate first beats T121.

## Final comparison and advancement

The sole comparator remains frozen D007-T121:

\[
O_{150}(\mathrm{T121}) = 0.701212576796.
\]

After candidate selection, the selected D011 configuration advances if and only if:

\[
O_{150}(\mathrm{selected}) < 0.701212576796.
\]

An exact tie does not advance.

There is no:

- endpoint veto;
- minimum effect-size requirement;
- secondary comparator;
- post-result candidate refinement;
- first-order fallback;
- architecture rescue;
- extension of the declared LR or step grid.

## Study endpoint and stopping rule

D011 ends after the complete declared 12-configuration grid has been adjudicated under the frozen selection rule.

If no selected configuration beats T121, D011 closes negative and T121 remains preferred.

If the selected configuration beats T121, D011 also ends. No additional tuning around the winner is authorized by D011.

The search may not be expanded after observing results.

## Evidence and claim boundary

D011 reuses the frozen D007 150-cell development procedure.

The D010 `(0.01, 1)` anchor is already-observed development evidence.

The remaining 11 configurations will also be evaluated on the same reused development cells.

Consequently, the selected configuration's O150 is tuning-selected development performance.

A D011 win supports advancement of that tuned configuration for subsequent work. It does not constitute untouched confirmation or independent external evidence that the selected configuration is superior to T121.

New partitions of already-used molecules do not create independent confirmation.

Protected internal-test evidence and D008 evidence remain excluded from D011 tuning, selection, and advancement.

## Runtime qualification

Before any D011 production authorization, the changed multi-step exact-ANIL workload must receive a bounded complete-model real-graph runtime/memory benchmark on one NVIDIA L40S outside SLURM.

Benchmark the worst planned adaptation depth:

- `K = 3`.

The benchmark must execute the complete D010 molecular graph encoder, aggregation, meta-head, exact through-adaptation computation through all three inner steps, outer backward, optimizer, scheduler, and the corresponding three-step full-support evaluation adaptation/prediction path.

The D010 runtime benchmark may not be automatically inherited because changing \(K\) changes the exact differentiation and evaluation workload.

The benchmark must report:

- synchronized complete outer-update timing after warmup;
- timing stability;
- peak allocated and reserved GPU memory;
- three-step four-isoform validation adaptation/prediction cost;
- projected training and validation workload per cell;
- projected workload for 150 cells per newly trained configuration;
- projected total workload for exactly 1,650 new cells;
- measured versus excluded costs;
- proposed requested wall time per one-cell/one-L40S job.

The production wall-time request is set only after this benchmark.

## Production authority

D011 does not authorize implementation, candidate evaluation, training, or production submission merely by approving this scientific record.

The next authorized preparation step, after this record is frozen, is a connected implementation/qualification package that:

- parameterizes only inner LR and inner-step count;
- verifies exact differentiation through \(K=1,2,3\);
- verifies matched candidate settings during meta-training and evaluation;
- proves D010 `(0.01,1)` protocol compatibility for anchor reuse;
- implements deterministic 12-candidate selection and tie-breaking;
- preserves all D010 firewall and artifact contracts;
- prepares the bounded `K=3` complete-model benchmark.

Production remains separately approval-gated after qualification and runtime projection.

## Governing principle

D011 is a bounded tuning study of the exact-ANIL adaptation operator, not a reopening of D010 or an unrestricted meta-learning search.

Only inner learning rate and inner-step count may vary. Results cannot authorize extension of the search space or modification of the frozen selection and advancement rules.

## Revisit when

Revisit D011 only if:

- a concrete protocol incompatibility prevents valid reuse of the frozen D010 `(0.01,1)` anchor;
- a demonstrated technical or integrity defect invalidates a required assumption;
- the `K=3` runtime qualification demonstrates an operational infeasibility requiring explicit human adjudication;
- or a separately approved future decision supersedes D011.

A disappointing tuning result is not grounds to extend the grid, add candidates, change architecture, introduce first-order ANIL, or otherwise modify D011 after results are observed.
## D012 — Assay-resolved CYP transfer study

**Decision date:** 2026-11-10

**Status:** Approved; preregistration complete. Implementation and qualification may proceed; production training remains separately approval-gated.

### Decision

D012 prospectively replaces D011 as the CYP meta-learning development study. D011 remains superseded/incomplete; its completed artifacts are preserved, its partial candidate scores do not inform D012, and its held remainder is not released or cancelled without explicit human approval.

D012 tests whether assay-episodic ANIL pretraining on routed, assay-resolved CYP IC50 measurements improves prediction of the four fixed OpenADMET direct-inhibition pIC50 endpoints beyond conventional multitask transfer trained with equivalent auxiliary information.

This is a whole-method comparison. It does not isolate feature reuse, make a few-shot target-learning claim, or establish that the four-task D010/D011 formulation caused prior performance. Both transfer methods use all available frozen target-training observations.

### Study arms

D012 has three arms:

1. **A — target-only control:** frozen T121. Existing T121 configuration and results are reused wherever target data, endpoints, partitions, seeds and evaluation protocol match D012. T121 is not automatically retuned or retrained.
2. **B — conventional transfer:** assay-preserving auxiliary multitask Chemprop training followed by target fine-tuning.
3. **C — meta-learning:** assay-episodic ANIL auxiliary training followed by target adaptation/fine-tuning.

The primary comparison is C versus B. B versus A and C versus A are secondary comparisons testing whether auxiliary training improves upon target-only T121.

### Auxiliary endpoint and target relationship

Auxiliary labels are exact ChEMBL IC50 concentrations converted deterministically to pIC50:

`pIC50 = -log10(IC50_M)`.

The four OpenADMET targets are continuous direct-inhibition pIC50 regression endpoints for CYP1A2, CYP2C9, CYP2D6 and CYP3A4.

The numerical quantity is therefore pIC50 in both auxiliary and target data, but the experimental measurements are not treated as interchangeable. Auxiliary measurements retain assay identity because their experimental contexts differ.

### Molecular identity and leakage contract

Cross-source identity uses full Standard InChIKey under:

- Python 3.12.14;
- RDKit 2026.03.6;
- InChI 1.07.3;
- no project-added molecular standardization.

Standard InChI intrinsic normalization is acknowledged. Verbatim source SMILES, canonical-isomeric-SMILES witnesses and full Standard InChI are preserved.

Frozen OpenADMET `Molecule_Name` partitions remain unchanged.

The bounded audit represented all 5,655 OpenADMET structures and 4,821 of 4,822 selected ChEMBL exact records. Activity 7817060 / CHEMBL508338 / CHEMBL1909136 remains `source_structure_unavailable` and identity-unresolved.

Within-assay source molecule-ID grouping and qualified full-InChIKey grouping were exactly concordant across the 4,821 represented records: 4,690 groups under each definition, with zero merges, zero splits and zero same-key/different-InChI cases.

External identities matching OpenADMET validation, internal test or original blinded challenge are protected and unavailable to auxiliary training. OpenADMET-training overlaps remain available but explicitly flagged. Public/unblinded challenge labels remain inaccessible during development and selection.

### Measurement rule

Within a qualified `(isoform, assay task, full Standard InChIKey)` group, exact IC50 measurements are converted to pIC50 and represented by the median derived pIC50.

All contributing source measurements, units, activity IDs and stored ChEMBL pChEMBL values remain provenance. Stored pChEMBL is a witness, not label authority.

### Task provenance and demonstrated duplicate

The routed pre-consolidation population contains 176 candidate assay tasks.

Exactly one pair has affirmative duplicate authority:

- CYP3A4 / CHEMBL2448771
- CYP3A4 / CHEMBL3137501

Both have `assay_src_id=14` and `assay_src_assay_id=1034082`.

They will form one task using the frozen molecular-identity and measurement rules. No other unresolved relationship is consolidated.

The expected post-consolidation count is 175 task units. Final identity/observation counts and task-size distribution must be materialized before the D012 auxiliary population is frozen. The pre-consolidation count of 4,664 available task-identity instances is not the final D012 auxiliary count.

The remaining task units are not claimed to be independent.

### Materialized D012 auxiliary population

The demonstrated duplicate consolidation was materialized before model implementation.

The 176 source assay tasks produce exactly **175 final task units**. The sole consolidated unit is:

`D012_CONSOLIDATED_CYP3A4_CHEMBL2448771_CHEMBL3137501`.

Its two source assays contain 36 and 27 available identities respectively. Their qualified full-Standard-InChIKey intersection is zero, so the consolidated task contains 63 identities. No shared-identity measurement reconciliation was required by this consolidation.

The frozen D012 auxiliary population contains:

- 175 task units;
- 4,664 task-identity instances;
- 4,795 contributing source exact observations;
- 4,615 external-only task-identity instances;
- 49 `overlap_openadmet_train` task-identity instances;
- zero mixed available overlap-class instances;
- zero zero-identity tasks.

Task units by isoform are:

- CYP1A2: 22;
- CYP2C9: 33;
- CYP2D6: 40;
- CYP3A4: 80.

Overall available identities per task range from 15 to 160. Nearest-rank quartiles are 17 / 21 / 29.

The smallest-task counts are:

- size 15: 1 task;
- size 16: 18 tasks;
- size 17: 26 tasks;
- size 18: 15 tasks;
- size 19: 11 tasks;
- size 20: 12 tasks.

No minimum task-size filter was applied. The D010/D011 16-molecule requirement has no D012 authority.

The 175 final task units are not claimed to be mutually independent. All provenance relationships other than the single demonstrated duplicate remain at their previously established status.

The authoritative materialization artifacts are under:

`qualification/cyp_external_task_qualification_20261109/d012_auxiliary_population/`

with frozen hashes recorded in that directory's `SHA256SUMS.txt`.

### Equal-information and exposure contract

Arms B and C receive identical finalized auxiliary task units, molecular structures and labels and the same frozen target-training observations. Neither receives assay descriptions, substrate/probe names, source identifiers or provenance metadata as model inputs.

Fairness is tracked separately as:

- equal input access;
- equal task sampling;
- objective-bearing molecule exposure;
- auxiliary optimization opportunity;
- target-side tuning opportunity.

Tasks are treated uniformly during auxiliary training.

One auxiliary epoch gives every available auxiliary label exactly one objective-bearing exposure.

For conventional multitask training, each identity contributes once to its assay's supervised objective per auxiliary epoch.

For ANIL, each identity contributes exactly once as query per auxiliary epoch. Support identities may be reused.

Both arms record objective-bearing label exposures, total molecule forward passes, optimizer updates, epochs and GPU/wall time. Identical computational operations are not required.

### ANIL episode construction

For a task with `N` identities:

`Q(N) = max(1, ceil(0.20 * N))`

`S(N) = N - Q(N)`.

Support and query identities are disjoint by full Standard InChIKey and sampled without replacement within an episode.

Within an auxiliary epoch, query sets cycle through a seeded shuffled permutation until every identity has appeared exactly once as query. The final query set may contain fewer than `Q(N)` identities. Support reuse is permitted.

Support loss is the mean over support molecules. Query loss is the mean over query molecules. When multiple tasks form a meta-batch, task query losses are averaged equally so larger assays do not receive greater outer-objective weight merely because of task size.

The D010/D011 13+3 / 16-molecule requirement has no D012 authority.

### Conventional multitask transfer

Arm B uses the same Chemprop encoder family and finalized auxiliary evidence.

Auxiliary learning uses assay-specific scalar regression heads with task-balanced exposure. At target transfer, auxiliary heads are discarded, a fresh target regression head is attached, and all frozen target-training observations are available for fine-tuning.

### ANIL target transfer

Arm C uses the episodically trained encoder and learned scalar-head initialization, then adapts/fine-tunes using all frozen target-training observations.

D012 therefore does not make a few-shot target-prediction claim.

### Target-side tuning

Arms B and C receive matched prospectively bounded validation-only target-side tuning budgets.

Shared parameters use the same candidate set where their meanings are equivalent. Algorithm-specific parameters may receive separately declared bounded searches where no meaningful counterpart exists.

No arm receives additional tuning because an initial result is unfavorable.

### T121 control

T121 is the frozen target-only control and is not automatically retrained or retuned.

Existing T121 results are reused wherever target endpoints, molecular universe, train/validation/test assignments, preprocessing/features, labels, seeds and evaluation protocol match D012.

A new target-only run is authorized only if a concrete incompatibility or missing declared seed/evaluation cell is demonstrated. Any such run uses the frozen T121 configuration only; it does not reopen target-only hyperparameter selection.

No new T121 run is currently authorized. Preserved T121 seed coverage must first be inventoried.

### Repeated-seed evaluation

D012 uses five preregistered seeds. Exact seed integers remain pending inspection of the existing project/T121 seed convention.

For each arm, report every seed-level per-isoform and macro ST-RAE result plus arithmetic mean, sample standard deviation, median, minimum and maximum.

The primary statistic is:

`Delta_C-B = mean(O_ANIL) - mean(O_MT)`.

Lower ST-RAE is better.

ANIL outperforms conventional transfer only if its five-seed mean macro ST-RAE is strictly lower than the conventional-transfer mean.

Also report paired seed differences, median paired difference and ANIL seed-win count. No post-hoc significance threshold is introduced.

B-versus-T121 and C-versus-T121 are secondary comparisons and must use matched seed populations.

### Negative-result interpretation

Negative results are authorized outcomes.

D012 distinguishes:

- neither external method improving T121;
- conventional transfer improving T121 without ANIL improving conventional transfer;
- ANIL improving conventional transfer while neither external method improves T121;
- ANIL improving both conventional transfer and T121;
- substantial seed variability.

No favorable seed selection or automatic tuning extension is permitted.

### Prerequisites before implementation or training

Before implementation/training authority is granted:

1. materialize the demonstrated duplicate consolidation and freeze the actual 175-task identity/observation population and distribution;
2. inventory preserved T121 seed/configuration/evaluation coverage and select the five D012 seeds from the established project convention where possible;
3. determine the exact minimum T121 rerun requirement, if any;
4. complete the final D012 preregistration fields and obtain human approval.

### Final preregistration amendment

This section supersedes any conflicting provisional D012 language above concerning repeated seeds, target-transfer optimization, auxiliary scheduling, or target-side tuning.

#### Frozen development matrix and T121 reuse

D012 inherits the exact frozen D007 development matrix:

- 5 repetitions;
- 10 folds per repetition;
- training seeds `20261101`, `20261102`, `20261103`;
- 50 split manifests;
- 150 total development cells.

The three training seeds associated with a split manifest share that manifest's frozen target train/validation membership.

T121 is the frozen target-only control. Its compatible existing 150-cell development evidence is reused without retraining or retuning. Its authoritative development quantity remains:

`O150(T121) = 0.701212576796`.

No new T121 development runs are required by D012.

D012 B and C are evaluated on the identical 150-cell target development matrix. The primary comparison is the cell-paired difference:

`O150(C) - O150(B)`,

with B-versus-T121 and C-versus-T121 secondary.

Report all 150 cell-level paired differences, their arithmetic mean, sample standard deviation, median, range and directional win count. O150 remains the primary aggregate development quantity.

#### Cell-specific auxiliary validation firewall

The finalized global D012 auxiliary population contains 175 task units and 4,664 task-identity instances.

For each frozen D007 split manifest `m`, construct its auxiliary population by removing every D012 task-identity instance whose qualified full Standard InChIKey occurs among that manifest's target-validation identities.

Validation identity is established without accessing validation potency labels:

`validation Molecule_Name -> already-qualified OpenADMET full Standard InChIKey`.

Any missing or ambiguous identity witness fails closed and may not be interpreted as non-overlap.

The same manifest-specific auxiliary mask is used by B and C and by all three training-seed cells associated with that manifest.

The mask is manifest-specific, not global. An identity that is target-validation in one manifest may remain auxiliary-eligible in another manifest where it is target-training, subject to the frozen routing rules.

The characterization found:

- all 5,655 OpenADMET Molecule_Name values mapped uniquely through the qualified identity bridge;
- zero unmapped validation identities;
- 49 of 50 manifests contain at least one auxiliary/validation overlap;
- 33 unique auxiliary molecular identities occur in validation in at least one manifest;
- 245 manifest-specific affected task-identity instances;
- 0 to 14 auxiliary task-identity instances removed per manifest, mean 4.9;
- 4,650 to 4,664 auxiliary task-identity instances retained per manifest;
- zero task units become empty;
- all 175 task units remain present in every manifest;
- minimum post-mask task size across all manifests is 14;
- at most one task is below 15 identities in any manifest;
- at most two tasks are below 16 identities in any manifest.

No minimum task-size criterion is introduced. The historical D010/D011 16-molecule requirement has no D012 authority.

Protected internal-test and original blinded-challenge routing remains governed by the already-qualified global firewall and is not reopened by this manifest-specific rule.

#### Arm B — conventional multitask transfer

B performs assay-preserving conventional multitask auxiliary training followed by conventional target fine-tuning.

Auxiliary training:

- uses the manifest-specific masked auxiliary population;
- uses the common Chemprop encoder architecture;
- uses assay/task-specific scalar regression heads;
- runs for exactly 50 auxiliary epochs;
- has no auxiliary early stopping;
- has no auxiliary checkpoint selection;
- does not access OpenADMET validation labels;
- transfers the final epoch-50 auxiliary state.

One B auxiliary epoch means that every retained task-identity label in that manifest contributes exactly once to the supervised auxiliary objective.

Within each task, identities are deterministically shuffled from the declared run/cell seed and consumed in task-local minibatches of size 16. Loss is mean MSE over the minibatch.

For a manifest with retained task sizes `N_t`, B therefore performs:

`U_B = sum_t ceil(N_t / 16)`

optimizer updates per auxiliary epoch.

Across the 50 frozen manifests, `U_B` is either 387 or 388:

- 387 updates/epoch in 10 manifests;
- 388 updates/epoch in 40 manifests.

At target transfer:

1. initialize the encoder from B's final auxiliary checkpoint;
2. discard all auxiliary heads;
3. attach a fresh target head;
4. fine-tune the full target network on the cell's complete target-training partition;
5. use the cell's validation partition for early stopping and macro ST-RAE checkpoint selection.

B inherits the applicable frozen T121 target-training schedule:

- message-passing depth 4;
- message hidden dimension 600;
- FFN hidden dimension 2000;
- one FFN layer;
- dropout 0.4;
- batch size 16;
- maximum 50 target epochs;
- patience 20 validation checks;
- warmup 10 epochs;
- Adam/NoamLR optimization;
- initial LR `1.4283739994469765e-05`;
- maximum LR `0.0008245400010426866`;
- final LR `0.00020141638331344468`;
- training-partition-only target standardization.

B therefore receives target-validation-driven optimization and checkpoint selection as part of the conventional-transfer method.

#### Arm C — assay-episodic ANIL transfer

C performs assay-episodic ANIL auxiliary training followed by exactly one full-support head-only target adaptation step.

Auxiliary meta-training:

- uses the same manifest-specific masked auxiliary population as B;
- runs for exactly 50 auxiliary epochs;
- has no auxiliary early stopping;
- has no auxiliary checkpoint selection;
- does not access OpenADMET validation labels;
- transfers the final epoch-50 encoder and learned scalar-head initialization;
- uses exact through-adaptation differentiation;
- adapts only the scalar head weights and bias in the inner loop;
- uses inner LR `0.01`;
- uses exactly one inner step, `K=1`.

For a task containing `N` retained identities:

`Q(N) = max(1, ceil(0.20*N))`

and query sets cycle through one deterministic seeded permutation until every retained identity appears exactly once as query during the auxiliary epoch.

Support and query identities are disjoint by qualified full Standard InChIKey within every episode. Support identities may be reused across episodes.

Support loss is mean MSE over support identities. Query loss is mean MSE over query identities.

With one task per meta-step, every episode produces one outer optimizer update.

For task `t`:

`E_t = ceil(N_t / Q(N_t))`.

Therefore:

`U_C = sum_t E_t`

outer optimizer updates per auxiliary epoch.

Across the 50 frozen manifests, `U_C` is:

- 856 updates/epoch in 9 manifests;
- 857 updates/epoch in 32 manifests;
- 858 updates/epoch in 9 manifests.

At target transfer for each development cell:

1. load C's final epoch-50 encoder and learned scalar-head initialization;
2. freeze the encoder;
3. use the complete target-training partition as support;
4. perform exactly one head-only gradient update at inner LR `0.01`;
5. evaluate the resulting adapted state.

C performs no subsequent generic target fine-tuning, no target early stopping and no target checkpoint selection.

The cell validation partition is evaluation evidence for C, not an optimization or selection signal.

#### Auxiliary exposure and optimization fairness

D012 equalizes auxiliary information access and objective-bearing label exposure, not computational operations.

For every manifest and auxiliary epoch:

- B and C receive the same retained auxiliary task-identity population;
- every retained auxiliary label contributes exactly once to B's supervised objective;
- every retained auxiliary label contributes exactly once as a C query label.

Support reuse in C is permitted and recorded.

D012 does not require equality of:

- optimizer-step counts;
- support exposures;
- total molecular forward passes;
- gradient computations;
- wall time.

These are method-dependent quantities and are recorded as audit statistics.

Fifty auxiliary epochs therefore means 50 objective-bearing opportunities for every retained auxiliary label in both B and C.

#### Auxiliary optimizer and scheduler

No new learning-rate search is introduced.

Both auxiliary arms use Adam/NoamLR as the established outer/supervised optimizer family and inherit the T121 LR anchors:

- initial LR `1.4283739994469765e-05`;
- maximum LR `0.0008245400010426866`;
- final LR `0.00020141638331344468`.

The scheduler is defined relative to each arm's own declared optimizer-step topology rather than by assuming identical step counts.

For each manifest:

- B total auxiliary optimizer steps = `50 * U_B`;
- C total auxiliary outer optimizer steps = `50 * U_C`.

The warmup phase occupies the first 10 auxiliary epochs of the respective arm:

- B warmup steps = `10 * U_B`;
- C warmup steps = `10 * U_C`.

The remaining schedule proceeds to the declared final LR over that arm's remaining auxiliary optimizer steps.

Thus the LR anchors and epoch-relative schedule are shared while optimizer-step counts remain method-specific.

#### Whole-method claim boundary

D012 is a whole-method comparison.

B receives conventional full-network target fine-tuning with validation-driven early stopping and checkpoint selection.

C receives one frozen-encoder, head-only full-support adaptation step and no target-side optimization or checkpoint selection thereafter.

D012 therefore does not claim:

- matched target optimization opportunity;
- isolation of feature reuse;
- isolation of the adaptation operator;
- a few-shot target-learning experiment.

Both methods use all available target-training observations according to their declared target-transfer procedures.

No additional tuning branch, MAML arm, metric-learning arm or post-result rescue rule is authorized.

### First-cell production and operational qualification

The D012 implementation/correctness qualification and first real production cells completed successfully before release of the remaining development matrix.

#### Qualified implementation workload

For `rep_01_fold_01` / training seed `20261101`, the qualified manifest-specific firewall removes all auxiliary instances whose qualified full Standard InChIKey occurs in that manifest's target-validation identities.

The resulting first-cell auxiliary population contains:

- 4,660 task-identity instances;
- all 175 frozen task units;
- zero unresolved validation identity witnesses.

Arm B's frozen auxiliary workload is:

- 388 task-local optimizer updates per auxiliary epoch;
- 50 auxiliary epochs;
- 19,400 auxiliary optimizer updates total.

Arm C's frozen auxiliary workload is:

- 857 exact-ANIL episodes/outer updates per auxiliary epoch;
- 50 auxiliary epochs;
- 42,850 outer updates total.

These counts do not imply equal molecular forward-pass exposure. B gives every retained auxiliary label one supervised objective-bearing exposure per epoch. C gives every retained auxiliary label exactly one query exposure per epoch while support identities may be reused.

#### Preserved complete-model runtime benchmark

Production planning uses the preserved gpu-dev1 complete-model benchmark:

`qualification/cyp_external_task_qualification_20261109/d012_arm_qualification/gpu_dev1/d012_chemprop_dmpnn_complete_model_runtime_benchmark_rep01_fold01_seed20261101.json`

SHA-256:

`e06f593ff5ba0090e655f0402c2bc69919b9fcafb390ef84c4cb5929a9f1d600`

The benchmark ran on `n4pu04` using one NVIDIA L40S, Python 3.12.14, Torch 2.14.1+cu126, Chemprop 2.3.1 and the actual Chemprop D-MPNN molecular-graph path.

It used 10 continuous warmup complete updates followed by 100 measured complete updates without optimizer/scheduler reset.

The benchmark projected representative first-cell measured-region runtimes of approximately:

- Arm B: 35.66 minutes;
- Arm C: 12.08 minutes.

The requested first-cell planning envelopes were therefore 60 minutes for B and 30 minutes for C.

The benchmark excluded file loading, SMILES preprocessing and dataset-object construction, model construction, artifact serialization and queue time. Its GPU-memory measurement did not establish host-RAM sufficiency.

No separate CoreHPC runtime witness is required for D012 production planning.

#### Arm B first real production cell

Arm B `rep_01_fold_01` / seed `20261101` completed successfully under the frozen D012 procedure.

The cell executed assay-preserving auxiliary multitask training followed by conventional full-network target fine-tuning with validation-driven checkpoint selection and the frozen early-stopping rule.

Observed end-to-end runtime:

`336.19944089394994 s` = approximately `5.60 min`.

Recorded phase timings:

- auxiliary training: `244.65247676102445 s`;
- model construction: `1.2096986830001697 s`;
- target preparation: `0.45308310096152127 s`;
- target training/checkpoint selection: `87.79631654091645 s`.

Observed resource use:

- peak host RSS: `1815.76953125 MB`;
- peak CUDA allocated: `2628.216796875 MB`;
- peak CUDA reserved: `2986.0 MB`.

The target procedure stopped after 35 target epochs under the frozen validation/early-stopping procedure rather than changing the scientific stopping rule to fit wall time.

The complete first-cell artifact contract passed SHA-256 verification, including:

- final auxiliary checkpoint;
- retained best target checkpoint;
- target metrics;
- startup configuration;
- progress log;
- completion record.

#### Arm C first real production cell

Arm C `rep_01_fold_01` / seed `20261101` completed successfully under the frozen D012 procedure.

The cell executed 50 epochs of exact-ANIL auxiliary training followed by the frozen target procedure: encoder frozen, complete target-training support, exactly one head-only update at inner LR `0.01`, `K=1`, and validation scoring with no target checkpoint selection or generic target fine-tuning.

Observed end-to-end runtime:

`1111.704668656108 s` = approximately `18.53 min`.

Recorded phase timings:

- auxiliary training: `1105.5337280919775 s`;
- model construction: `0.13799810397904366 s`;
- target preparation: `0.3711214130744338 s`;
- one-step target adaptation and validation: `3.1582950410665944 s`.

Observed resource use:

- peak host RSS: `1712.05859375 MB`;
- peak CUDA allocated: `4812.59814453125 MB`;
- peak CUDA reserved: `4978.0 MB`.

The complete first-cell artifact contract passed SHA-256 verification, including:

- final auxiliary checkpoint;
- final transfer state;
- validation metrics;
- startup configuration;
- progress log;
- completion record.

#### Operational disposition

The first-cell execution gate passes for both arms.

The existing requested production envelopes remain:

- Arm B: one NVIDIA L40S, 4 CPUs, 32 GB host RAM, 60-minute wall time per cell;
- Arm C: one NVIDIA L40S, 4 CPUs, 32 GB host RAM, 30-minute wall time per cell.

The requests are operational safety envelopes and have no scientific stopping authority.

The first-cell observations establish successful execution and observed resource use for these representative cells. They are not universal upper bounds for all remaining cells. Every production cell must retain runtime/resource measurement, progress/failure preservation and the frozen scientific stopping rules.

The first-cell validation results have no authority to redesign D012, extend tuning, change stopping rules or select a different methodology.

The remaining 149 B cells and 149 C cells are not released by this record. Their array mapping, artifact-storage plan and wrappers require review before separate production authorization.

Large checkpoint payloads remain preserved on server-side storage. The local project preserves lightweight provenance, metrics, logs and hashes rather than duplicating all checkpoint payloads onto the Mac.

### Governing principle

D012 compares two methods for exploiting the same qualified assay-resolved auxiliary evidence while preserving the frozen OpenADMET target and evaluation framework.

Algorithmic computation may differ. Information access, protected-data access and undeclared tuning opportunity may not.

### Revisit when

Revisit D012 methodology only if duplicate materialization exposes an inconsistency in the frozen data rules, the qualified molecular-identity contract fails reproducibility, or a demonstrated incompatibility prevents use of the frozen OpenADMET evaluation framework.

The D012 auxiliary population, frozen 150-cell development matrix, training seeds, T121 reuse contract, cell-specific validation firewall, and B/C transfer procedures are now preregistered. Implementation and qualification findings that do not alter these scientific rules do not reopen D012; any required methodological change returns for explicit human adjudication before production training.
