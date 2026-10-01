# Tripwire

Two-arm drug-drug interaction liability characterization and re-ranking module: PXR induction + CYP inhibition. Built for the OpenADMET blind challenges and Stanford CS230.

A compound that induces PXR can increase clearance of co-administered drugs; one that inhibits CYPs can decrease it. Both can contribute to drug-drug interaction liability. Tripwire characterizes both arms so a candidate's DDI profile can be assessed before it becomes a clinical surprise.

Tripwire is a **characterization and re-ranking layer, not an attrition gate**. Its outputs do not independently eliminate candidates from the upstream SBDD cascade.

## Arms

**CYP inhibition (lead arm).** Direct-inhibition pIC50 regression across four isoforms: CYP3A4, CYP2C9, CYP2D6, and CYP1A2. Entry in the OpenADMET CYP inhibition blind challenge: 750-compound blind test set built by hit expansion. Primary metric: MA-ST-RAE. Submission deadline: November 3, 2026.

**PXR induction (quarter pace).** pEC50 regression on OpenADMET PXR dose-response data. Retrospective internal evaluation only, with a predeclared scaffold holdout as the primary evaluation.

## Baseline

The CYP independent reference baseline is four independently tuned LightGBM GBDT regressors, one per isoform, using fixed Morgan fingerprints (radius 2, 2048 bits).

- **Split:** analog-cluster holdout chosen to mirror the challenge's hit-expansion blind set. Whole analog clusters remain in exactly one of train, validation, or held-out test.
- **Tuning:** bounded randomized search per isoform using validation rows only, with validation ST-RAE as the selection metric. `n_estimators` is fixed at 2000 with early stopping; `bagging_freq=1` ensures that `subsample` is active.
- **Test evaluation:** the held-out test set is evaluated once, after all four validation configurations are frozen.

The frozen independent baseline is the reference against which subsequent CYP challengers are compared.

## Preregistered Challengers

CYP challengers are run as separate experiments against the frozen independent baseline and evaluated on the same held-out test set.

- **Masked-label multitask model:** shared molecular supervision across isoforms while masking missing labels; complete-case training is not permitted.
- **Potent-end loss weighting:** a separate experiment testing whether emphasizing potent compounds improves alignment with the soft-thresholded evaluation objective.
- **Chemprop:** headline neural model, trained from scratch.
- **AttentiveFP:** second headline neural model, trained from scratch.
- **GROVER:** backup model if the primary neural approaches require it.

Challenger disposition follows the preregistered experiment-specific criteria recorded in the decision record. Tripwire outputs retain characterization/re-ranking authority only; no challenger result independently creates candidate-attrition authority.

## Evaluation

Primary CYP direct-inhibition metric: **MA-ST-RAE** (macro-averaged soft-threshold relative absolute error), the official challenge metric.

Reported alongside:

- MAE
- R²
- Spearman ρ
- Kendall τ

Metrics are reported per isoform and macro-averaged across the four direct-inhibition endpoints.

## Provenance

Scientific decisions are preregistered in the decision records under `notes/` before implementation.

Frozen configurations, randomized-search trial tables, split assignments, and evaluation artifacts are committed so the Git history forms part of the audit trail.

Dataset revisions, upstream challenge materials, literature references, and pinned tutorial commits are recorded under `references/`.

## Layout

    cyp/        CYP inhibition arm:
                scripts/, data/, artifacts/

    pxr/        PXR induction arm (quarter pace)

    shared/     Featurization, analog-cluster splitting,
                ST-RAE evaluation, and other components
                explicitly shared across both arms

    notes/      Decision records, project log, and
                session logs

    references/ Challenge documentation, papers,
                dataset provenance, and pinned
                upstream tutorial information

## Current CYP Baseline

Tripwire-CYP-001 established the first independent CYP reference baseline.

Final internal held-out macro results:

    ST-RAE:       0.838997
    MAE:          0.632571
    R²:            0.217678
    Spearman ρ:   0.471892
    Kendall τ:    0.331371

Detailed provenance, tuning records, frozen configurations, and test predictions are recorded in the CYP artifacts and in:

`notes/Tripwire_CYP_001_Session_Log.md`

The next CYP experiment is the preregistered masked-label multitask challenger.
