# Tripwire-PXR-003 Supporting Record

**Date:** 2026-11-18
**Report:** `notes/Tripwire_PXR_003.md`

## Environment (local, conda env `tripwire_pxr003`)

Python 3.12.16, RDKit 2026.03.6, LightGBM 4.7.0, scikit-learn 1.9.1, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, pytest 9.2.0. Source: local command output as reported.

## Pre-run SHA-256 (recorded before tuning; local checkout; as reported, not recomputed by the bookkeeper)

| File | SHA-256 |
|---|---|
| pxr/scripts/pxr003_benchmark.py | a6d3835ee73043228a0615d0c5e8c04e8dcbddbcb40379c6dbadbd0a71f66da8 |
| pxr/scripts/pxr003_common.py | f3bb9609e73a625abb07446ee82993a912c2e8a7cf03f161fa65497b10b07f9a |
| pxr/scripts/pxr003_final.py | 6dac1643d23b77afef49f9b7ee2962181db7dcd14d371b22ca0225dc51b5088a |
| pxr/scripts/pxr003_holdout_report.py | 0f2a61d7159c32feb55cfb9d5905190ed254f30076f64a498bfa8aa3ef3824d7 |
| pxr/scripts/pxr003_models.py | 25768d8b5309d0ddd1703a3e8d4346d184e6e0963b16cf5847fdfb15a7aab5bb |
| pxr/scripts/pxr003_smoke.py | efee441196fa706e7ae2180d3b6b97dd4c7ecd2adc34bba59e826084e33f3542 |
| pxr/scripts/pxr003_tune.py | 7b5aa09cb7391a4af4ee1e6e3a0d77611977665c1b8d29117c2f59af396bcf91 |
| pxr/scripts/pxr003_config.json | 36e00a857d43e7493df66d628ec15e309eab2cc7191a40386937e78063a25628 |
| pxr/scripts/pxr001_baseline.py | 6342058080217e0b970e94211e1247e4514ba43cbced26e3de6771aa6083a4b7 |
| cyp/scripts/ecfp4.py | 0b094c5a68df83b459c4435d618dcd7d50b5149074fb2176c56a3c0f2a1157b6 |

The `pxr003_models.py` hash above is the post-edit file (containing the LightGBM `eval_X`/`eval_y` change), the version the tests, smoke test and tuning ran against: a local re-hash of the current file gives the same SHA-256 as the recorded pre-run hash, and the current file contains the `eval_X`/`eval_y` change.

| File | SHA-256 |
|---|---|
| pxr/tests/test_pxr003.py | b921b3dde7c84452d8a08c4de77eec64e7e75b3749df478f9064bb124e8a70b7 |

## Checks (local)

- `pytest pxr/tests/test_pxr003.py pxr/tests/test_pxr001_baseline.py`: 22 passed, 1 warning (pandas ConstantInputWarning from a constant-predictor Spearman in a unit test; expected). Re-run after the `eval_X`/`eval_y` edit.
- `pxr003_smoke`: PASS (cohort 2897/621/621; features; search; confirmation; selection; freeze assembly; holdout-structure prediction). No holdout labels read.

## Benchmark (local interactive, 8 threads; run before the `eval_X` edit)

| Item | Result |
|---|---|
| Featurization | 0.7 s |
| LightGBM, 74 fits | typical 22.3 min; upper bound 69.4 min; slowest fit 56.3 s (lr 0.01, 63 leaves, best_iteration 2968 of cap 3000) |
| Random forest, 54 fits | typical 1.9 min; upper bound 13.3 min |
| Total | typical 24.2 min; upper bound 82.6 min |

Local run; no Slurm and no wall-time comparison applies.

## LightGBM tuning outputs (local, `pxr/artifacts/pxr003/tune_lgbm/`; SHA-256 computed locally, as reported)

| File | SHA-256 |
|---|---|
| selection.json | 718078b9759ed6bdc744c04308312bf148826debcb1e5750fc17c1618f6ddf91 |
| trials.csv | 7b0e88c665f4cf912b2584564443346e5f5b864ea64c3e253c1cf28aea03df12 |
| confirm.csv | 8077d424ebc0fd6aa87bab1a53baf4e5eba92fda93fb6f38a4344309cc4cf3e5 |

Run: 60 search trials, top-3 x 3-seed confirmation, 791 s, 8 threads; output and files read locally. Trials at iteration cap: 0.

## Random forest tuning outputs (local, `pxr/artifacts/pxr003/tune_rf/`; SHA-256 computed locally, as reported)

| File | SHA-256 |
|---|---|
| selection.json | 0c97e490fb61255da616dceac70c6e9cf5f5e3ccc517a7abb57906f6b0f7dce7 |
| trials.csv | 6af68359ceeded5b7ec9f0a0c4bd8e231117d206e25fdb70934802add31d05eb |
| confirm.csv | 5add1290f074c7987fe3f8aab2a14de37b1a57c8f73f2047717274f415b50d08 |

Run: 40 search trials, top-3 x 3-seed confirmation, 500 trees, 132 s, 8 threads; output and files read locally. `selection.json` config_sha256 36e00a857d43e7493df66d628ec15e309eab2cc7191a40386937e78063a25628 matches the recorded `pxr003_config.json` hash.

## Frozen configuration (local, `pxr/artifacts/pxr003/frozen/`; SHA-256 re-verified locally, as reported)

| File | SHA-256 |
|---|---|
| frozen_config.json | 38238eb784a514289b04d65addebffe2ca1953e23af54157ce9f85cff4090030 |

Command (freeze): `python -m pxr.scripts.pxr003_final freeze --lgbm-selection pxr/artifacts/pxr003/tune_lgbm/selection.json --rf-selection pxr/artifacts/pxr003/tune_rf/selection.json --out-dir pxr/artifacts/pxr003/frozen`. Contents: config_sha256 36e00a857d43e7493df66d628ec15e309eab2cc7191a40386937e78063a25628; input hashes train CSV fdc2252bb5763ab5742e78cb1ba692bebe131a972cf83aa1bd45e43d8c5bc223 and assignments c6028f1b7216ab2ea4bab96616fc02b95ce011be20b1b96f807d1e7d582ece46 (both match D014-recorded values); LightGBM trial 53, random forest trial 1; final seeds 20261220-24; selection metric validation MAE.

## Final training outputs (local, `pxr/artifacts/pxr003/final/`; SHA-256 computed locally, as reported)

| File | SHA-256 |
|---|---|
| predictions.csv | 85231ba487521d3c8aa4bfb2affc9cace29ce482ff7713ecf8f28c966591f48f |
| validation_summary.json | 059d43d666ab469de26ab559239ca5965b60b4819cdae15ae19cd73703fec2ff |
| manifest.json | 672fbc66ed18666140de4e192e0291a972bd9070bab16589f36ada9d171e44f5 |

Command: `python -m pxr.scripts.pxr003_final train --frozen-config pxr/artifacts/pxr003/frozen/frozen_config.json --frozen-sha256 38238eb784a514289b04d65addebffe2ca1953e23af54157ce9f85cff4090030 --out-dir pxr/artifacts/pxr003/final --threads 8`; runtime 114.5 s; output pasted and files read locally. Manifest: holdout_labels_read=false; frozen_config_sha256 matches 38238eb784a514289b04d65addebffe2ca1953e23af54157ce9f85cff4090030. predictions.csv: 12,420 rows, no missing values. Constant-baseline Spearman is undefined (constant input; ConstantInputWarning expected).

## Holdout report (local, `pxr/artifacts/pxr003/holdout_report/`; SHA-256 computed locally, as reported)

| File | SHA-256 |
|---|---|
| holdout_report.json | b7cb8b8c9b2e9bbdc124724e15bda4e42daf3600f8a83f180b7607d50a919d51 |

Command (environment tripwire_pxr003): `python -m pxr.scripts.pxr003_holdout_report --baseline-predictions pxr/artifacts/pxr003/final/predictions.csv --baseline-predictions-sha256 85231ba487521d3c8aa4bfb2affc9cace29ce482ff7713ecf8f28c966591f48f --frozen-config pxr/artifacts/pxr003/frozen/frozen_config.json --frozen-sha256 38238eb784a514289b04d65addebffe2ca1953e23af54157ce9f85cff4090030 --out-dir pxr/artifacts/pxr003/holdout_report --authorize-holdout-evaluation`; output pasted and file hashed locally. Hashes verified within the run:

| Item | SHA-256 |
|---|---|
| Frozen config | 38238eb784a514289b04d65addebffe2ca1953e23af54157ce9f85cff4090030 |
| Baseline predictions (final/predictions.csv) | 85231ba487521d3c8aa4bfb2affc9cace29ce482ff7713ecf8f28c966591f48f |
| Neural predictions | 45c84c925b9dfc1c5c959e3a95c6b01961667dd19e2338470e8ae26655aa07aa |

Other in-run checks passed: input hashes; coverage exactly the 621 holdout ids; finite predictions; re-derived neural MAE/Spearman equal PXR-002 `final_evaluation.json` to 1e-9; neural constant MAE 0.9237158726788007.
