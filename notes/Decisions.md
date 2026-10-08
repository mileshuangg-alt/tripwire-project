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

## Governing Principle

CYP-002 changes one scientifically interpretable variable at a time:

**the relative weighting of the four partially observed CYP task losses.**

The purpose of the sensitivity grid is to characterize whether the task-balanced objective changes performance across the preregistered architectural configurations without converting the grid into an implicit optimization procedure.

Architecture, data, preprocessing, representation, splits, tuning protocol, validation criterion, checkpoint-selection rule, seed policy, and final adjudication remain fixed.

## Revisit when

Reconsider the retained two-variant grid only if a subsequent approved decision explicitly amends D001/CYP-002 or the completed sensitivity analysis provides documented evidence requiring a methodological amendment.