# Phase-C Prediction Matrix Provenance

Read-only provenance extraction for the already-generated Phase-C prediction matrices.

No inference, evaluation, bootstrap, training, or tests were rerun. The prediction matrix files were not modified. `cyp/data/cyp-challenge-TEST-BLINDED.csv` was not accessed.

## File Provenance

| Matrix | SHA-256 | Rows |
|---|---|---:|
| `cyp/artifacts/cyp_002/heldout_internal_test_eval/CYP002-C0-stock-seed20261001-r2_heldout_test_predictions.csv` | `1ebfbe5c226ec968445231e7dc7f13f25485f1767c1a4964605760f0c3b2cbba` | 736 |
| `cyp/artifacts/cyp_002/heldout_internal_test_eval/CYP002-C0-task-balanced-seed20261001-r1_heldout_test_predictions.csv` | `18d0665de15b50784c621fadf282a499e1a78d0277a8f1558df5a79a1fb587bf` | 736 |

Both files contain the same 736 `molecule_name` values in the same order: confirmed.

Both files contain the required `cluster_id`, four observed internal-test targets, four predictions, and confidence bounds used for ST-RAE: confirmed.

## Headers

Both files have this exact header list:

```text
variant
run_id
checkpoint_path
molecule_name
cluster_id
observed_CYP1A2_pIC50_direct_inhibition
predicted_CYP1A2_pIC50_direct_inhibition
CYP1A2_pIC50_direct_inhibition_conf_low
CYP1A2_pIC50_direct_inhibition_conf_high
observed_CYP2C9_pIC50_direct_inhibition
predicted_CYP2C9_pIC50_direct_inhibition
CYP2C9_pIC50_direct_inhibition_conf_low
CYP2C9_pIC50_direct_inhibition_conf_high
observed_CYP2D6_pIC50_direct_inhibition
predicted_CYP2D6_pIC50_direct_inhibition
CYP2D6_pIC50_direct_inhibition_conf_low
CYP2D6_pIC50_direct_inhibition_conf_high
observed_CYP3A4_pIC50_direct_inhibition
predicted_CYP3A4_pIC50_direct_inhibition
CYP3A4_pIC50_direct_inhibition_conf_low
CYP3A4_pIC50_direct_inhibition_conf_high
```

## Stock First 3 Rows

```csv
variant,run_id,checkpoint_path,molecule_name,cluster_id,observed_CYP1A2_pIC50_direct_inhibition,predicted_CYP1A2_pIC50_direct_inhibition,CYP1A2_pIC50_direct_inhibition_conf_low,CYP1A2_pIC50_direct_inhibition_conf_high,observed_CYP2C9_pIC50_direct_inhibition,predicted_CYP2C9_pIC50_direct_inhibition,CYP2C9_pIC50_direct_inhibition_conf_low,CYP2C9_pIC50_direct_inhibition_conf_high,observed_CYP2D6_pIC50_direct_inhibition,predicted_CYP2D6_pIC50_direct_inhibition,CYP2D6_pIC50_direct_inhibition_conf_low,CYP2D6_pIC50_direct_inhibition_conf_high,observed_CYP3A4_pIC50_direct_inhibition,predicted_CYP3A4_pIC50_direct_inhibition,CYP3A4_pIC50_direct_inhibition_conf_low,CYP3A4_pIC50_direct_inhibition_conf_high
stock,CYP002-C0-stock-seed20261001-r2,cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001-r2/best.ckpt,OCNT-0007477,4786,,3.267076253890991,,,,3.0129570960998535,,,,4.321588516235352,,,2.577795577,1.8612208366394043,1.140199,3.71784025
stock,CYP002-C0-stock-seed20261001-r2,cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001-r2/best.ckpt,OCNT-0022125,4774,,4.3031511306762695,,,,3.6755356788635254,,,,5.044539451599121,,,3.853962615,3.0641510486602783,3.4793095,4.15196175
stock,CYP002-C0-stock-seed20261001-r2,cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001-r2/best.ckpt,OCNT-0022126,4773,,4.513634204864502,,,4.573013837,4.394776344299316,4.317132,4.88299625,4.470200998,5.135263919830322,4.39338525,4.580512,,4.448963165283203,,
```

## Stock Last 3 Rows

```csv
variant,run_id,checkpoint_path,molecule_name,cluster_id,observed_CYP1A2_pIC50_direct_inhibition,predicted_CYP1A2_pIC50_direct_inhibition,CYP1A2_pIC50_direct_inhibition_conf_low,CYP1A2_pIC50_direct_inhibition_conf_high,observed_CYP2C9_pIC50_direct_inhibition,predicted_CYP2C9_pIC50_direct_inhibition,CYP2C9_pIC50_direct_inhibition_conf_low,CYP2C9_pIC50_direct_inhibition_conf_high,observed_CYP2D6_pIC50_direct_inhibition,predicted_CYP2D6_pIC50_direct_inhibition,CYP2D6_pIC50_direct_inhibition_conf_low,CYP2D6_pIC50_direct_inhibition_conf_high,observed_CYP3A4_pIC50_direct_inhibition,predicted_CYP3A4_pIC50_direct_inhibition,CYP3A4_pIC50_direct_inhibition_conf_low,CYP3A4_pIC50_direct_inhibition_conf_high
stock,CYP002-C0-stock-seed20261001-r2,cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001-r2/best.ckpt,OCNT-2395790,78,,4.058732986450195,,,,4.262812614440918,,,,4.593358993530273,,,4.13460412,4.129591941833496,3.9396765,4.33215175
stock,CYP002-C0-stock-seed20261001-r2,cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001-r2/best.ckpt,OCNT-2395791,77,,5.125585556030273,,,,5.227950096130371,,,,4.82181978225708,,,5.45690557,5.437744617462158,5.38223725,5.5345205
stock,CYP002-C0-stock-seed20261001-r2,cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001-r2/best.ckpt,OCNT-2395792,9,,5.376035213470459,,,,5.523257732391357,,,,4.3491644859313965,,,4.702887592,5.364975929260254,4.62826,4.81535075
```

## Task-Balanced First 3 Rows

```csv
variant,run_id,checkpoint_path,molecule_name,cluster_id,observed_CYP1A2_pIC50_direct_inhibition,predicted_CYP1A2_pIC50_direct_inhibition,CYP1A2_pIC50_direct_inhibition_conf_low,CYP1A2_pIC50_direct_inhibition_conf_high,observed_CYP2C9_pIC50_direct_inhibition,predicted_CYP2C9_pIC50_direct_inhibition,CYP2C9_pIC50_direct_inhibition_conf_low,CYP2C9_pIC50_direct_inhibition_conf_high,observed_CYP2D6_pIC50_direct_inhibition,predicted_CYP2D6_pIC50_direct_inhibition,CYP2D6_pIC50_direct_inhibition_conf_low,CYP2D6_pIC50_direct_inhibition_conf_high,observed_CYP3A4_pIC50_direct_inhibition,predicted_CYP3A4_pIC50_direct_inhibition,CYP3A4_pIC50_direct_inhibition_conf_low,CYP3A4_pIC50_direct_inhibition_conf_high
task_balanced,CYP002-C0-task-balanced-seed20261001-r1,cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261001-r1/best.ckpt,OCNT-0007477,4786,,3.4290432929992676,,,,3.1320858001708984,,,,4.344618320465088,,,2.577795577,2.0833401679992676,1.140199,3.71784025
task_balanced,CYP002-C0-task-balanced-seed20261001-r1,cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261001-r1/best.ckpt,OCNT-0022125,4774,,4.482628345489502,,,,3.638485908508301,,,,4.9583940505981445,,,3.853962615,3.0212063789367676,3.4793095,4.15196175
task_balanced,CYP002-C0-task-balanced-seed20261001-r1,cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261001-r1/best.ckpt,OCNT-0022126,4773,,4.492647171020508,,,4.573013837,4.480323314666748,4.317132,4.88299625,4.470200998,5.070425033569336,4.39338525,4.580512,,4.537765979766846,,
```

## Task-Balanced Last 3 Rows

```csv
variant,run_id,checkpoint_path,molecule_name,cluster_id,observed_CYP1A2_pIC50_direct_inhibition,predicted_CYP1A2_pIC50_direct_inhibition,CYP1A2_pIC50_direct_inhibition_conf_low,CYP1A2_pIC50_direct_inhibition_conf_high,observed_CYP2C9_pIC50_direct_inhibition,predicted_CYP2C9_pIC50_direct_inhibition,CYP2C9_pIC50_direct_inhibition_conf_low,CYP2C9_pIC50_direct_inhibition_conf_high,observed_CYP2D6_pIC50_direct_inhibition,predicted_CYP2D6_pIC50_direct_inhibition,CYP2D6_pIC50_direct_inhibition_conf_low,CYP2D6_pIC50_direct_inhibition_conf_high,observed_CYP3A4_pIC50_direct_inhibition,predicted_CYP3A4_pIC50_direct_inhibition,CYP3A4_pIC50_direct_inhibition_conf_low,CYP3A4_pIC50_direct_inhibition_conf_high
task_balanced,CYP002-C0-task-balanced-seed20261001-r1,cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261001-r1/best.ckpt,OCNT-2395790,78,,4.128278732299805,,,,4.392084121704102,,,,4.552784442901611,,,4.13460412,4.261448860168457,3.9396765,4.33215175
task_balanced,CYP002-C0-task-balanced-seed20261001-r1,cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261001-r1/best.ckpt,OCNT-2395791,77,,5.127291679382324,,,,5.201905727386475,,,,4.8442063331604,,,5.45690557,5.367127418518066,5.38223725,5.5345205
task_balanced,CYP002-C0-task-balanced-seed20261001-r1,cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261001-r1/best.ckpt,OCNT-2395792,9,,5.412856578826904,,,,5.5564045906066895,,,,4.4223527908325195,,,4.702887592,5.410326957702637,4.62826,4.81535075
```
