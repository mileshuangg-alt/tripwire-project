# Tripwire-CYP-006 Session Log

**Date:** 2026-11-17  
**Status:** Open. Screening-validation weighting decided (D016) and implemented locally; no GPU check, run, training or submission.

## 1. Session purpose

Session 006 continues the ActFound-CYP whole-package study after Session 005 closed. Its first task was the carried question of whether evaluation-phase adaptation should be weighted (`notes/Tripwire_CYP_005.md`, §8–§9).

Governing decisions: D013 and Amendments 1–4, D015 (comparison plan; authorizes no run) and D016.

## 2. Weighting at evaluation

Inspection found that `run_task` in `cyp/scripts/actfound_production_train_v2.py` built observation weights only when `is_training_phase=True`. Screening-validation adaptation and the pIC50 evaluator therefore both ran unweighted. No weight source for the pIC50 evaluator exists in the records, and the training-only gate has no recorded approval. Details: supporting record S2.

**D016** (`notes/Decisions.md`) fixes the intent: weighted adaptation for screening validation only; the pIC50 evaluator stays unweighted. D015's open-items line on pIC50 support weights is therefore not required for the evaluator; whether to annotate it is undecided.

## 3. Screening-validation weights: edit and local test

`run_task` gained a `use_observation_weights` keyword, and the screening-validation call passes `True`. The foundation-validation call is unchanged; whether it should also be weighted is not decided. The edited runner has SHA-256 `af92e152…` (backup and details: supporting record S1).

Local mock-level tests of the five relevant files give 39 passed, 0 failed (S3). One earlier failure was a test written against a keyword that the runner passes positionally; the test was corrected and the runner was not changed. The tests use a stubbed forward, so they establish neither real-model behavior nor launch readiness.

No GPU check has been done. The stated position is that a GPU check is not required for this change unless it leaves GPU-specific behavior untested; this authorizes no run, training or submission and is not entered in `notes/Decisions.md`. Whether any GPU-specific behavior is left untested, for example device and dtype handling of the weights tensor, has not been determined. The edited runner is not bound to any staged package or qualification evidence.

## 4. Stopping point and next step

Session 006 is open with no run authorized. Carried from Session 005 and still unresolved: the wiring list, T121 connection to the controller, comparator provenance (`comparison_is_final=false`), the `d014_lr_penalty` folder name colliding with PXR D014, and the job 3103473 hash binding (S4). D015 Amendment 1 (`notes/Decisions.md`) settles the D013 question: the fresh corrected-penalty initialization writes new, separate artifacts, and the D013 epoch-51 checkpoint, job 3076848 and results stay frozen; it authorizes no run. The next step is to determine whether any GPU-specific behavior of the weights change is untested (which decides whether a GPU check is needed), then continue the wiring inspection.
