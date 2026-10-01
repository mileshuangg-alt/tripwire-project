# Tripwire Project Log

| Session | Date | Objective | Outcome |
|---|---|---|---|
| Tripwire-CYP-001 | 2026-09-30 to 2026-10-01 | Establish the OpenADMET CYP direct-inhibition baseline: audit the 4,905-compound sparse four-isoform training set; freeze the 70/15/15 analog-cluster holdout; verify the official ST-RAE evaluator; build the fixed Morgan radius-2/2048-bit ECFP4 representation; tune one LightGBM regressor per isoform with 30 validation-only randomized trials and ST-RAE early stopping; freeze the four validation configurations; and evaluate once on the untouched held-out test set | ✅ Complete — independent ECFP4/LightGBM baseline frozen; 30 trials per isoform; final macro held-out ST-RAE 0.838997 (CYP1A2 0.795530, CYP2C9 0.948653, CYP2D6 0.937892, CYP3A4 0.673911); CYP3A4 search retained with distinct provenance and was not rerun; next: masked-label multitask challenger in Tripwire-CYP-002 |
