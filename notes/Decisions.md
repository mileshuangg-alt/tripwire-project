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