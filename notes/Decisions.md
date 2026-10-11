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

## 6. Step-3 Analysis Universe

For the Step-3 split-robustness analyses established above, the eligible analysis universe is restricted to the compounds assigned to the frozen CYP-001 **train and validation partitions**.

The Step-3 universe therefore contains:

- 3,433 compounds from the frozen train partition; and
- 736 compounds from the frozen validation partition;

for a total of **4,169 compounds**.

The frozen **736-compound internal held-out test partition is excluded from Step 3**. It must not enter Step-3 split construction, feasibility analysis, model fitting, refitting, evaluation, or statistical analysis. Its completed one-shot C0 headline evaluations remain frozen and are not reopened.

The separate **750-compound blinded OpenADMET challenge set is also excluded from Step 3** and remains reserved for prospective external evaluation.

Within the 4,169-compound Step-3 universe, split construction for both repeated grouped/clustered k-fold and Monte Carlo split-realization analysis occurs at the frozen scaffold/analog-cluster level rather than at the individual-compound level. A scaffold/analog cluster must not cross the training/evaluation boundary within a fold or split realization.

Stock and task-balanced C0 must receive the **same cluster partition** within each paired comparison, and each model is refit as required by the Step-3 analysis defined above.

This amendment defines the Step-3 analysis universe and protected-data boundary only. It does not reopen the completed C0 headline decision or alter the existing CYP-002 sensitivity-grid results.

The fixed value of \(k\), number of repeated k-fold realizations, Monte Carlo split proportion, number of Monte Carlo realizations, randomization seeds, and final repeated-refit aggregation procedure are **not fixed by this amendment**. They must be declared before the corresponding Step-3 model runs. The existing requirement that each fixed-\(k\) fold contain enough of the minority class remains in force, without introducing a numeric minimum not specified by the governing guidance.

## Governing Principle

CYP-002 changes one scientifically interpretable variable at a time:

**the relative weighting of the four partially observed CYP task losses.**

The purpose of the sensitivity grid is to characterize whether the task-balanced objective changes performance across the preregistered architectural configurations without converting the grid into an implicit optimization procedure.

Architecture, data, preprocessing, representation, splits, tuning protocol, validation criterion, checkpoint-selection rule, seed policy, and final adjudication remain fixed.

## Revisit when

Reconsider the retained two-variant grid only if a subsequent approved decision explicitly amends D001/CYP-002 or the completed sensitivity analysis provides documented evidence requiring a methodological amendment.