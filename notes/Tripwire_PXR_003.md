# Tripwire-PXR-003 Session Log

**Date:** 2026-11-18
**Status:** CLOSED on declaration, 2026-11-18.

**Summary (final).**
- **Decision.** D019 (Approved as a design) adds ECFP4 + LightGBM and ECFP4 + random forest baselines. It reverses the D017 skip of the tree-baseline ladder for these two baselines only and does not restore a baseline-first ordering.
- **Cohort.** Baselines train on 2,897 primary-train molecules (OADMET-0003727 included); the neural runs use 2,896. Validation and holdout sets are identical (621 each).
- **Environment and code.** Conda env `tripwire_pxr003`; code package `pxr/scripts/pxr003_*.py` and `pxr/tests/test_pxr003.py`. Tests 22 passed; smoke PASS; local benchmark typical 24.2 min, upper bound 82.6 min.
- **Selection (validation only).** LightGBM trial 53 (confirmation mean validation MAE 0.60562) and random forest trial 1 (0.67615), neither clearly separable from the runner-up. Frozen configuration `pxr/artifacts/pxr003/frozen/frozen_config.json` SHA-256 38238eb784a514289b04d65addebffe2ca1953e23af54157ce9f85cff4090030.
- **Final models.** Five-seed ensembles (seeds 20261220-24) per method, trained on train only.
- **Holdout (single, reporting-only, internal-retrospective; official test not used).** Ensemble MAE / Spearman / RMSE: multitask 0.48427 / 0.73567 / 0.67936; aux-off 0.48469 / 0.73675 / 0.68374; LightGBM 0.54461 / 0.72780 / 0.71982; random forest 0.61269 / 0.68224 / 0.80864 (§8). Descriptive only; the holdout is consumed for these baselines.
- **Open observation (uninvestigated).** Holdout MAE is below validation MAE for every system, while the constant baseline is about equal.
- **Open items (none decided, none authorized or started).** Neural validation RMSE and Spearman were not computed. The Rev 3 line 129 numerical bar and any statistical test remain undefined. Standardization is undeclared. Auxiliary-learning improvement is a separate follow-up. Official-test evaluation and retraining on train plus validation remain unauthorized.
- **Artifacts.** Hashes in `notes/Tripwire_PXR_003_supporting.md`; decision in `notes/Decisions.md` (D019).

**Workstream:** Tripwire — PXR induction arm; continuation after Tripwire-PXR-002 (closed 2026-11-18; `notes/Tripwire_PXR_002.md`).

## 1. Session purpose and starting state

Build the two classical PXR baselines required by the proposal (Rev 3 line 90) as comparators to the completed neural work. Starting state is the PXR-002 close: multitask five-seed holdout MAE 0.4843 / Spearman 0.7357; auxiliary-off MAE 0.4847 / Spearman 0.7367; constant training-mean floor MAE 0.9237 (`notes/Tripwire_PXR_002.md`). The frozen D014 split is unchanged.

## 2. Classical baselines (D019)

D019 (`notes/Decisions.md`, Approved as a design) specifies ECFP4 + LightGBM and ECFP4 + random forest: features, two-stage seeded selection on validation MAE, five-model final ensembles trained on train only, a frozen configuration recorded before any holdout scoring, and a single reporting-only holdout scoring that also computes neural holdout RMSE. It supersedes the D017 skip of the tree-baseline ladder for these two baselines only and does not restore a baseline-first ordering.

Cohort difference, disclosed: baselines train on 2,897 primary-train molecules (OADMET-0003727 included) against 2,896 for the neural runs; training-mean pEC50 4.32761 against 4.32822. Validation and holdout sets are identical (621 each).

## 3. Implementation, tests and benchmark

New conda env `tripwire_pxr003` (Python 3.12.16, RDKit 2026.03.6, LightGBM 4.7.0, scikit-learn 1.9.1, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, pytest 9.2.0). Code package: `pxr/scripts/pxr003_{config.json,common,models,tune,final,holdout_report,benchmark,smoke}.py` and `pxr/tests/test_pxr003.py`. One post-benchmark edit: the LightGBM fit in `pxr003_models.py` passes `eval_X`/`eval_y` in place of the deprecated `eval_set` (deprecation warning only); tests and smoke were re-run after it.

- Tests (`test_pxr003.py` with `test_pxr001_baseline.py`): 22 passed, 1 expected warning.
- Smoke (`pxr003_smoke`): PASS on cohort 2897/621/621 through features, search, confirmation, selection, freeze assembly and holdout-structure prediction; no holdout labels read.
- Benchmark (local interactive, 8 threads, before the `eval_X` edit, which does not change fit work): featurization 0.7 s; LightGBM 74 fits typical 22.3 min, upper bound 69.4 min (slowest fit 56.3 s); random forest 54 fits typical 1.9 min, upper bound 13.3 min; total typical 24.2 min, upper bound 82.6 min. Runs locally; no Slurm wall-time comparison applies.

Pre-run SHA-256 of code and inputs, recorded before tuning, are in `notes/Tripwire_PXR_003_supporting.md`.

## 4. LightGBM validation-only tuning

Run locally (`python -m pxr.scripts.pxr003_tune --model lgbm --out-dir pxr/artifacts/pxr003/tune_lgbm --threads 8`): 60 search trials (single seed 20261201+trial), then top-3 confirmation over 3 seeds (20261211-13). Runtime 791 s (13.2 min) for 69 fits, against the benchmark typical 22.3 min for 74 fits.

- Search validation MAE ranged about 0.6026-0.6357. Top three by search MAE: trial 30 (0.60256), trial 53 (0.60317), trial 2 (0.60318).
- Confirmation mean validation MAE: trial 53 0.60562 (selected), trial 2 0.60635, trial 30 0.60713. Differences are about 0.001, comparable to the seed-to-seed spread (per-seed confirmation MAEs 0.6042-0.6102), so the three are not clearly separable.
- Selected trial 53: learning_rate 0.015387, num_leaves 51, min_child_samples 29, colsample_bytree 0.15843, subsample 0.67151, reg_lambda 0.62672. Best iterations over the three confirmation seeds: 1816, 1310, 1965.
- No trial reached the 3000-tree cap (search best_iteration maximum 2632; trials_at_iteration_cap = 0).

Artifacts are in `pxr/artifacts/pxr003/tune_lgbm/` (`selection.json`, `trials.csv`, `confirm.csv`); hashes are in the supporting record.

## 5. Random forest validation-only tuning

Run locally (`python -m pxr.scripts.pxr003_tune --model rf --out-dir pxr/artifacts/pxr003/tune_rf --threads 8`): 40 search trials (single seed 20261201+trial), then top-3 confirmation over 3 seeds (20261211-13), 500 trees fixed. Runtime 132 s.

- Search validation MAE ranged about 0.676-0.757. Top three by search MAE: trial 1 (0.67523), trial 37 (0.67608), trial 27 (0.67703).
- Confirmation mean validation MAE: trial 1 0.67615 (selected), trial 37 0.67661, trial 27 0.67873. The top two differ by about 0.0005, within the seed spread (per-seed confirmation MAEs 0.6742-0.6795), so they are not clearly separable.
- Selected trial 1: max_features 0.17099, min_samples_leaf 2, max_samples 0.69428, all interior to the searched ranges.
- `selection.json` records config_sha256 36e00a857d43e7493df66d628ec15e309eab2cc7191a40386937e78063a25628, matching the recorded `pxr003_config.json` hash.

Artifacts are in `pxr/artifacts/pxr003/tune_rf/` (`selection.json`, `trials.csv`, `confirm.csv`); hashes are in the supporting record.

## 6. Frozen baseline configuration

Run locally: `python -m pxr.scripts.pxr003_final freeze` with the two `selection.json` files, output to `pxr/artifacts/pxr003/frozen/`. Both baseline configurations are frozen before any holdout scoring.

- `frozen_config.json` SHA-256 38238eb784a514289b04d65addebffe2ca1953e23af54157ce9f85cff4090030 (re-verified locally). It records config_sha256 36e00a857d43e7493df66d628ec15e309eab2cc7191a40386937e78063a25628 (matches `pxr003_config.json`), train CSV and assignments input hashes (fdc2252bb5763ab5742e78cb1ba692bebe131a972cf83aa1bd45e43d8c5bc223 and c6028f1b7216ab2ea4bab96616fc02b95ce011be20b1b96f807d1e7d582ece46; both match the D014-recorded values), LightGBM trial 53 (confirmation mean validation MAE 0.60562), random forest trial 1 (0.67615) with the parameters in sections 4 and 5, final seeds 20261220-24, and selection metric validation MAE.
- The freeze's config-hash and input-hash mismatch checks passed (run completed without rejection).

## 7. Final training (train only)

Run locally (`python -m pxr.scripts.pxr003_final train --frozen-config pxr/artifacts/pxr003/frozen/frozen_config.json --frozen-sha256 38238eb784a514289b04d65addebffe2ca1953e23af54157ce9f85cff4090030 --out-dir pxr/artifacts/pxr003/final --threads 8`): 5 seeds (20261220-24) per method, trained on the 2,897 train molecules only. The manifest records holdout_labels_read=false and a frozen_config_sha256 matching the frozen configuration. Runtime 114.5 s.

| Method | Per-seed validation MAE | Ensemble MAE | Ensemble Spearman | Ensemble RMSE |
|---|---|---|---|---|
| LightGBM | 0.60535, 0.61094, 0.60488, 0.60635, 0.60566 | 0.60487 | 0.66949 | 0.79518 |
| Random forest | 0.67769, 0.67381, 0.67445, 0.67590, 0.67382 | 0.67451 | 0.58994 | 0.88192 |
| Constant train mean (4.327606) | n/a | 0.92176 | undefined (constant input) | 1.13570 |

- LightGBM best iterations 1830, 1332, 1531, 1818, 1786; none at the 3000 cap. Random forest used 500 trees per seed.
- Ensembles are five-model means on the 621 validation molecules.
- For context only: PXR-002 neural five-seed mean validation MAE was about 0.546 (0.5458 multitask, 0.5499 aux-off, as recorded). Neural validation Spearman and RMSE were not computed; no comparison is drawn.
- `predictions.csv` has 12,420 rows (2 methods x 5 seeds x (621 validation + 621 holdout)), no missing values; holdout rows are structure-only predictions.

Artifacts are in `pxr/artifacts/pxr003/final/`; hashes are in the supporting record.

## 8. Single reporting-only holdout scoring

Run locally: `python -m pxr.scripts.pxr003_holdout_report ... --out-dir pxr/artifacts/pxr003/holdout_report --authorize-holdout-evaluation`. One scoring of 621 holdout molecules. Evidence class: internal-retrospective; the official test set was not used. The neural holdout results were known before the baselines were configured. No holdout score changed any configuration or chose between random forest and LightGBM.

In-run checks passed: frozen config hash, baseline predictions hash and neural predictions hash match the recorded values; input hashes match; prediction coverage is exactly the 621 holdout ids; predictions are finite; re-derived neural MAE and Spearman equal PXR-002 `final_evaluation.json` to 1e-9, and the neural constant MAE (0.9237158726788007) matches. Hashes are in the supporting record.

Holdout ensemble metrics (five-model means):

| System | MAE | Spearman | RMSE |
|---|---|---|---|
| Chemprop multitask (primary) | 0.48427 | 0.73567 | 0.67936 |
| Chemprop aux-off | 0.48469 | 0.73675 | 0.68374 |
| ECFP4 + LightGBM | 0.54461 | 0.72780 | 0.71982 |
| ECFP4 + random forest | 0.61269 | 0.68224 | 0.80864 |

Mean per-model metrics: multitask 0.51357 / 0.71562 / 0.71049; aux-off 0.51034 / 0.71933 / 0.70948; LightGBM 0.54675 / 0.72549 / 0.72194; random forest 0.61316 / 0.68022 / 0.80930 (MAE / Spearman / RMSE).

Constant training-mean baseline on holdout: the 2,897-molecule mean 4.327606 gives MAE 0.92387, RMSE 1.13727; the 2,896-molecule mean 4.328223 gives MAE 0.92372, RMSE 1.13729. Spearman is undefined for both.

Results are descriptive only; no numeric threshold and no statistical test exist or were applied.

Observation (not a diagnosis; uninvestigated): holdout MAE is lower than validation MAE for every system (LightGBM 0.6049 to 0.5446, random forest 0.6745 to 0.6127, neural about 0.546 to 0.484), while the constant-baseline MAE is about equal (validation 0.9218, holdout 0.9239). This carries over the PXR-002 open observation and shows it is not specific to the neural models.

The holdout scoring is consumed for these baselines and must not be re-run or used for tuning or selection.

## 9. Stopping point and next step

Tuning, freezing, final training and the single holdout scoring are complete; D019 is unchanged. The session was declared closed on 2026-11-18. No next step is authorized or started; the open items are listed in the summary at the top.
