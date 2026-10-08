# Tripwire-CYP-002 — Session Log

**Session:** Tripwire-CYP-002  
**Continuation:** Tripwire-CYP-002.5 was a direct chat continuation of Session 002 and is recorded here as part of the same session.  
**Date:** 2026-10-05 to 2026-10-08  
**Status:** **CLOSED**  
**Workstream:** Tripwire — CYP inhibition arm  
**Governing document:** `Tripwire_Project_Proposal_Rev3.md` (Revision 3, September 30, 2026)  
**Execution target:** `gpu-dev1` / `n4pu04`  
**Execution environment:** `/mnt/nfs/CX900004_DS1/conda-envs/tripwire_cyp002`

---

## 1. Session Objective

Session 002 implemented, debugged, completed, and ultimately evaluated the CYP-002 masked-label multitask Chemprop experiment against the approved CYP inhibition contract.

The scientific experiment compared two variants across the frozen C0-C7 validation sensitivity grid:

1. **Variant 1 — stock masked multitask MSE**
2. **Variant 2 — task-balanced masked multitask MSE**

C0 is the headline configuration. The grid is validation-only and reports the preregistered sensitivity statistic `k/8`.

The final C0 headline comparison was performed only after the validation grid, Step-1 diagnostic, Phase-B five-seed noise characterization, and Phase-B freeze were complete.

The session ended after exactly one C0 stock held-out internal-test evaluation and one C0 task-balanced held-out internal-test evaluation, followed by the frozen paired scaffold-cluster bootstrap decision rule.

---

## 2. Governing CYP-002 Contract

The confirmed CYP-002 contract specifies:

- four CYP direct-inhibition pIC50 endpoints;
- C0 as the headline configuration;
- stock masked multitask Chemprop and task-balanced masked multitask Chemprop as the two variants;
- the full C0-C7 x two-variant validation sensitivity grid;
- validation-only comparison using macro ST-RAE;
- lower ST-RAE is better;
- the grid sensitivity statistic `k/8`;
- no feedback from the grid into the C0 headline rule;
- exactly one held-out evaluation per headline variant;
- the held-out test set is not used for model or configuration selection before the headline evaluations.

The task endpoints and order used by the implementation are:

```text
0 CYP1A2_pIC50_direct_inhibition
1 CYP2C9_pIC50_direct_inhibition
2 CYP2D6_pIC50_direct_inhibition
3 CYP3A4_pIC50_direct_inhibition
```

Task-balanced loss remained the frozen D001-style objective:

- missing labels masked;
- uniform task weights of 0.25;
- `renormalize_over_present_tasks = false`;
- zero-label task contribution/gradient = zero under the frozen implementation.

Task-balanced was **not retired** during this session.

---

## 3. Frozen C0 Configuration

The C0 architecture/configuration remained fixed throughout the scientific comparison:

```text
name                    C0
message_passing_depth   3
message_hidden_dim      300
ffn_hidden_dim          300
ffn_num_layers          1
dropout                  0.0
max_lr                   0.001
aggregation              norm
aggregation_norm         100
batch_size               64
max_epochs               50
early_stopping_patience   20
early_stopping_min_delta 0.0
initial_lr               0.0001
final_lr                 0.0001
warmup_epochs             2
activation                relu
batch_norm                false
num_tasks                 4
gradient_clipping         None
data_loader_workers       0
```

The original C0 scientific seed was `20261001`. The approved five-seed replication amendment later made the training seed a run-level parameter, using:

```text
20261001
20261002
20261003
20261004
20261005
```

Only the random training seed changed across those replicate runs. All other C0 scientific settings remained frozen.

---

## 4. Authoritative Dataset and Split

Training dataset:

`cyp/data/cyp-challenge-TRAIN_inhibition.csv`

Frozen internal split:

`cyp/artifacts/cyp_baseline/molecule_split_assignments.csv`

Frozen split SHA-256:

`4cea719334a0836e4b9c0d50dd093872d472142b33337eb98362f95aa10c1902`

Verified split counts:

```text
train         3433
validation     736
test            736
```

The internal held-out test in Phase C was therefore the 736 compounds assigned `split=test` in the frozen training table.

The separate blinded OpenADMET challenge file:

`cyp/data/cyp-challenge-TEST-BLINDED.csv`

contains the 750-compound prospective blind set and was **not accessed during this session**.

No blind-set labels were used for model selection, validation, Step-1 diagnostics, seed-noise characterization, or Phase-C evaluation.

---

## 5. Execution Environment

Official scientific execution occurred on:

- logical target: **gpu-dev1**;
- physical host: **n4pu04**;
- project directory on GPU host: `/mnt/nfs/CX900004_DS117/tripwire-project`;
- conda environment: `/mnt/nfs/CX900004_DS1/conda-envs/tripwire_cyp002`.

Relevant runtime versions recorded during the session:

```text
Python       3.12.14
PyTorch      2.14.1+cu126
Chemprop     2.3.1
Lightning    2.6.6
CUDA         12.6
cuDNN        91002
GPU          NVIDIA L40S
GPU count    4
```

The four additional seed runs were executed serially using `CUDA_VISIBLE_DEVICES=1`.

The project did not use BKS.

---

## 6. CYP-002 C0 Debugging and Validation Corrections

### 6.1 Initial stock C0 run — invalidated

Initial run:

`cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001/`

This run is preserved as historical invalid-run evidence and is not a scientific result.

Its saved validation macro ST-RAE was:

`9.315000257342257`

Independent checkpoint evaluation showed the official original-scale macro ST-RAE was:

`0.8620893873605819`

The root cause was conclusively identified as a prediction-scale mismatch:

- true targets were on original pIC50 scale;
- confidence bounds were on original pIC50 scale;
- collected model predictions remained on standardized target scale;
- the project path allowed those standardized predictions to reach the ST-RAE scorer.

A mixed-scale reproduction produced:

`9.315000685651672`

confirming the defect.

The first run also exposed a validation lifecycle defect: Lightning's pre-training sanity validation used only the first 128 validation rows while the scorer expected the complete 736-row confidence arrays.

### 6.2 Validation lifecycle correction

`CYP002ValidationMPNN.on_validation_epoch_end()` was corrected to skip scoring while `self.trainer.sanity_checking` is true.

Real validation epochs continue to concatenate all validation batches and score once.

### 6.3 Prediction-scale correction

`cyp/scripts/cyp002_validation.py` was corrected so collected standardized predictions are explicitly restored to original pIC50 scale before ST-RAE scoring.

The official ST-RAE implementation itself was not changed:

`references/cyp_upstream/custom_scoring_functions.py`

The correction was accompanied by the regression test:

`cyp/tests/test_cyp002_scoring_path_regression.py`

The corrected implementation passed:

```text
39 passed
```

The corrected-scoring code provenance recorded for the earlier fix was:

`fbb27f0a65a1d536154f4e43616bb5208eb62895`

No C0 architecture, loss semantics, split, masking policy, ST-RAE definition, checkpoint criterion, patience, or other scientific methodology was changed by the scoring correction.

---

## 7. Authoritative C0 Seed-20261001 Runs

### 7.1 Stock

Run:

`cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001-r2/`

Checkpoint SHA-256:

`89a3f37657e439258d3d6a8c650aceae4d4014951084baffaa4e536449d8b8f4`

Metrics SHA-256:

`9a18a5bd8e5422defebf9d7f9f0b33441f7a557ee0dd98029b47fd6c2b8772bb`

Best validation macro ST-RAE:

`0.684556823058`

Per-task best validation ST-RAE:

```text
CYP1A2  0.933527926232
CYP2C9  0.562665455225
CYP2D6  0.824360512224
CYP3A4  0.417673398550
```

### 7.2 Task-balanced

Run:

`cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261001-r1/`

Checkpoint SHA-256:

`f082d67d3d7e2d3e243a148fa1b8a34326faecc239ec4c1b78b9c0dd39b3ef97`

Metrics SHA-256:

`d823331fe0e3d4db6fa5e09c7a3756434bc4e56de40646929f0e45f0c8d08c6e`

Best validation macro ST-RAE:

`0.690857884653`

Per-task best validation ST-RAE:

```text
CYP1A2  0.929560578113
CYP2C9  0.563400759611
CYP2D6  0.831836783368
CYP3A4  0.438633417519
```

### 7.3 Single-seed comparison

Lower ST-RAE is better.

```text
stock            0.684556823058
task-balanced    0.690857884653
```

Thus the single-seed validation difference was:

```text
stock - task_balanced = -0.006301061595
```

approximately 0.92% in favor of stock.

This single-seed result was **not** treated as sufficient evidence of material superiority; the five-seed replication was subsequently run specifically to characterize seed-level stochastic noise before spending the one-shot held-out test budget.

Both C0 variants were independently reproduced from their saved checkpoints.

---

## 8. Validation-Only C0-C7 x 2 Sensitivity Grid

The completed validation sensitivity grid contains all 16 cells:

```text
C0 stock / task-balanced
C1 stock / task-balanced
C2 stock / task-balanced
C3 stock / task-balanced
C4 stock / task-balanced
C5 stock / task-balanced
C6 stock / task-balanced
C7 stock / task-balanced
```

All 16 validation cells were completed.

The preregistered sensitivity statistic is `k/8`, where `k` is the number of configurations for which Variant 2 has lower validation macro ST-RAE.

The completed grid yielded:

```text
k = 1
k/8 = 0.125
```

The grid was validation-only and did not modify the C0 headline configuration or the held-out testing rule.

### 8.1 Grid artifact provenance

| Cell | Run path | Checkpoint SHA-256 | Metrics SHA-256 |
|---|---|---|---|
| C0 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001-r2/` | `89a3f37657e439258d3d6a8c650aceae4d4014951084baffaa4e536449d8b8f4` | `9a18a5bd8e5422defebf9d7f9f0b33441f7a557ee0dd98029b47fd6c2b8772bb` |
| C0 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261001-r1/` | `f082d67d3d7e2d3e243a148fa1b8a34326faecc239ec4c1b78b9c0dd39b3ef97` | `d823331fe0e3d4db6fa5e09c7a3756434bc4e56de40646929f0e45f0c8d08c6e` |
| C1 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C1-stock-seed20261001-r1/` | `4ac8f9749828da0df09a33f791806619957859a570e19482f23cf8fe53cfcd7f` | `3357c7f714c66b2fefbce110b33e485fc3c101a9212b0268ec9f5f889d5999ee` |
| C1 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C1-task-balanced-seed20261001-r1/` | `427ee45786750761cab437cdf00a5ab50d92a1699d8364b49868650e135fb412` | `86109a251964dbe6fc76e219ce47b5541cf33454ae7c878749fd6ee5ebba9fc5` |
| C2 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C2-stock-seed20261001-r1/` | `294fb2ee625559df473e8b95d5f1644b442b810c135ddbb8c36e90bd483a78bf` | `2e5f3446a64ca2e3f508848b91e58c763ff54a833923d94ddd83caf7548c0556` |
| C2 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C2-task-balanced-seed20261001-r1/` | **UNKNOWN** | **UNKNOWN** |
| C3 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C3-stock-seed20261001-r1/` | `92e2941a6863ac097de0aec870362ef59cf1a7e215a0266e1099e4cffa57498c` | `2c60a699df5fc2803c9bbff35fc62abc95f2eb9a4fb152d6ec62cfa2f3e899f2` |
| C3 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C3-task-balanced-seed20261001-r1/` | `2ac5d158da421310bb9d4bd797b16bbdb01bdae1329e38dd4cc0004899d736aa` | `f8aac1f304fc5a8fbd452794b7c4fbbf17316633f5d708b64b2e5c0e363ee43f` |
| C4 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C4-stock-seed20261001-r1/` | `8f04144035a2928e9c1ea44706ef74afcf5d893daa380f9d445a95aeaf37ddb6` | `916566a7d92e7b8518e8c880d54f4dc600a5c2aa9501607407451177ad7943fb` |
| C4 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C4-task-balanced-seed20261001-r1/` | `009f85b7c7f860b325c6f8739902caa8cd8361e190f2ea5af3408e089a30d2ee` | `f52455b435ff1bc7b7b6230c44eb6c3d6ddf09d738b51f8400d1473c5554a13b` |
| C5 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C5-stock-seed20261001-r1/` | `2057a211f4b63def828bd8e46ee6f3b82af0a32964c181d87309b8d037ef10a4` | `4235d876c274ddf57267abf161d595569fb31fccc31d8f8981f4803b247bfee5` |
| C5 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C5-task-balanced-seed20261001-r1/` | `246b48e776f2a22631f1fdd7b373fe73715bb81963ac8ebdcae1980ade927a74` | `6803e2af7605a3d363454984e58cb8d0bd3bf41c4118e7f578610e00f9fd31df` |
| C6 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C6-stock-seed20261001-r1/` | `3ed8bc60f1fd3cc13b8170915d314c47a8b8f9f0baf826c05a4ea643775e2117` | `b8dd17c94e02a44f490f4cc83ab9a8edc80dc1396e71d81175f66a59eb55bc01` |
| C6 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C6-task-balanced-seed20261001-r1/` | `288310353a77de7326f78168e257eab95ecd1180fbc3067af473efa7f09063f5` | `65c4f20facf123ecb01462efb8323d57b944cc456a42de39a5f0a3a604d59b0a` |
| C7 / Variant 1 | `cyp/artifacts/cyp_002/runs/CYP002-C7-stock-seed20261001-r1/` | `2a2ed527cdb34cbee633cd29ffb209f670ed700796d90896dba2a243087a50d7` | `c033638b601190dd772b0dbec5094c18d08941e6af1b357516b95e3fadb62d6d` |
| C7 / Variant 2 | `cyp/artifacts/cyp_002/runs/CYP002-C7-task-balanced-seed20261001-r1/` | `788bff18c199a85151554e8b3929e8c73f68fdcfd04eb31c1ad5ae833b63e740` | `81676743fa6fbec9a760d13021ad5d28a74fffb30e3b3d74578c6ae1cd2f438e` |

The C2 / Variant 2 checkpoint and metrics hashes remain explicitly UNKNOWN; no inferred hashes are substituted.

---

## 9. Step-1 Error-Structure Analysis

The permanent C0 stock validation prediction matrix was:

`cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261001-r2/step1_c0_stock_validation_predictions_original_pIC50.tsv`

SHA-256:

`4ebe312bd62575c71f757bc829e61f129011ee5db206eab88951992537708c73`

The already-frozen ECFP4 implementation was:

`cyp/scripts/ecfp4.py`

using Morgan radius 2 / 2048 bits. No new molecular representation was introduced.

### 9.1 Task-performance direction

C0 validation ordering:

```text
CYP3A4  0.4177  strongest
CYP2C9  0.5627  intermediate
CYP2D6  0.8244  weak
CYP1A2  0.9335  weakest
```

The earlier project interpretation that CYP3A4 was the weak task was corrected and preserved as a dated correction rather than deleted.

### 9.2 Uncertainty-band diagnostic

CYP1A2:

```text
n = 212
median band width = 0.346060750

Tight:
  n = 106
  MAE = 0.602665666
  soft-thresholded error = 0.486998973

Wide:
  n = 106
  MAE = 0.736621396
  soft-thresholded error = 0.348092104

Wide - tight MAE = +0.133955730
Wide - tight soft-thresholded error = -0.138906868
```

CYP2D6:

```text
n = 224
median band width = 0.261560000

Tight:
  n = 112
  MAE = 0.548373140
  soft-thresholded error = 0.453370430

Wide:
  n = 112
  MAE = 0.606492224
  soft-thresholded error = 0.300883179

Wide - tight MAE = +0.058119084
Wide - tight soft-thresholded error = -0.152487250
```

The lower soft-thresholded error in the wide-band group is expected mechanically from the ST-RAE construction because wider uncertainty bands absorb more absolute error by design. It is descriptive and is not evidence for either hypothesis.

### 9.3 Chemical-distance diagnostic

CYP1A2:

```text
Pearson r = 0.020871114
Pearson p = 0.762557342
Spearman rho = 0.048806031
Spearman p = 0.479661877
nearest quartile MAE = 0.661611583
farthest quartile MAE = 0.612540103
far - near MAE = -0.049071480
```

CYP2D6:

```text
Pearson r = 0.025172946
Pearson p = 0.707879352
Spearman rho = -0.016523221
Spearman p = 0.805736302
nearest quartile MAE = 0.561280975
farthest quartile MAE = 0.577104626
far - near MAE = +0.015823651
```

The nearest-training-neighbor chemical-distance relationship was effectively null for both tasks.

### 9.4 Step-1 classification

> **Weak CYP1A2/CYP2D6 performance is more consistent with broad assay/label uncertainty than with insufficient chemical coverage under the tested ECFP4 nearest-neighbor diagnostic; the classification rests on higher raw MAE in the wide-band group (+0.134 for CYP1A2 and +0.058 for CYP2D6) and the null nearest-neighbor chemical-distance relationship, while the lower wide-band soft-thresholded error is mechanical under ST-RAE's interval construction and is not evidence for either hypothesis.**

This is a comparative classification, not a claim that assay/label uncertainty is the sole cause.

The existing CYP3A4 error analysis and the existing 1A2/2D6 analyses remain preserved.

---

## 10. Phase-B Design Audit and Freeze

The Phase-B audit was explicitly frozen before the held-out test spend.

The finalized audit fixed:

- C0 as the headline configuration;
- the completed C0-C7 x 2 validation grid;
- `k/8 = 0.125`;
- five paired C0 training seeds;
- seed-noise definition;
- one-shot held-out test spend;
- paired scaffold-cluster bootstrap;
- `B = 10,000`;
- 95% percentile interval;
- dedicated analysis RNG seed `20261008`;
- frozen materiality rule.

### 10.1 Five training seeds

The five validation replicate seeds were fixed as:

```text
Seed 1  20261001
Seed 2  20261002
Seed 3  20261003
Seed 4  20261004
Seed 5  20261005
```

Seed `20261001` was the existing authoritative C0 replicate and was **not rerun**.

### 10.2 Seed-noise definition

For seed `s`:

$$
\Delta_{\mathrm{seed},s}
=
\mathrm{ST\!\!-\!RAE}_{\mathrm{stock},s}
-
\mathrm{ST\!\!-\!RAE}_{\mathrm{TB},s}
$$

The noise scale is the sample standard deviation:

$$
SD_{\mathrm{seed}}
=
\operatorname{SD}
\left(
\Delta_{\mathrm{seed},1},
\ldots,
\Delta_{\mathrm{seed},5}
\right)
$$

Frozen result:

```text
SD_seed = 0.007094968910757
```

### 10.3 Paired scaffold-cluster bootstrap definition

Frozen cluster-assignment artifact:

`cyp/artifacts/cyp_baseline/cluster_assignments.csv`

SHA-256:

`4923d31401cc4f775e89377665ba0b15102245cca252911fd02e50879d10ffb8`

The clustering unit is the frozen scaffold/analog cluster, not the individual compound.

For each bootstrap draw, clusters are sampled with replacement. Both stock and task-balanced are evaluated on the exact same resampled cluster multiset within the draw.

Individual compounds are never independently resampled because doing so would treat correlated compounds within scaffold clusters as independent observations and understate interval width.

The bootstrap statistic is:

$$
\Delta_{\mathrm{test}}^{*(b)}
=
\mathrm{ST\!\!-\!RAE}_{\mathrm{stock}}^{*(b)}
-
\mathrm{ST\!\!-\!RAE}_{\mathrm{TB}}^{*(b)}
$$

with:

```text
B = 10,000
sampling = with replacement
pairing = same resampled clusters for both variants
confidence interval = 95% percentile
analysis RNG seed = 20261008
```

The percentile interval is:

$$
\left[
Q_{0.025}\!\left(\Delta_{\mathrm{test}}^*\right),
Q_{0.975}\!\left(\Delta_{\mathrm{test}}^*\right)
\right]
$$

and:

$$
LCB(\Delta_{\mathrm{test}})
=
Q_{0.025}\!\left(\Delta_{\mathrm{test}}^*\right)
$$

### 10.4 Frozen headline decision rule

The C0 test gain is:

$$
\Delta_{\mathrm{test}}
=
\mathrm{ST\!\!-\!RAE}_{\mathrm{stock,test}}
-
\mathrm{ST\!\!-\!RAE}_{\mathrm{TB,test}}
$$

Positive values favor task-balanced.

Task-balanced takes the C0 headline if and only if:

$$
LCB(\Delta_{\mathrm{test}})
>
SD_{\mathrm{seed}}
$$

Otherwise stock remains the headline.

This criterion was frozen before the held-out test numbers were observed.

---

## 11. Phase-B Five-Seed Validation Results

The four additional seed pairs were run under the frozen C0 protocol. All eight new runs completed successfully, serially on `gpu-dev1` using `CUDA_VISIBLE_DEVICES=1`.

Execution failures:

`none`

Seed `20261001` was not rerun.

### 11.1 Five-seed paired validation table

| Training seed | Stock macro ST-RAE | Task-balanced macro ST-RAE | Delta_seed = Stock - TB |
|---|---:|---:|---:|
| `20261001` | 0.684556823057850 | 0.690857884652704 | -0.006301061594854 |
| `20261002` | 0.680145541646839 | 0.675366787440831 | +0.004778754206008 |
| `20261003` | 0.690064499896603 | 0.688951856296707 | +0.001112643599896 |
| `20261004` | 0.696088668668752 | 0.694874571038753 | +0.001214097629999 |
| `20261005` | 0.676340834323393 | 0.689190790762711 | -0.012849956439318 |

Frozen sample SD:

`SD_seed = 0.007094968910757`

The sign of the paired gain changes across seeds:

- task-balanced better on `20261002`, `20261003`, `20261004`;
- stock better on `20261001`, `20261005`.

This establishes measurable seed-level variation in the stock-vs-task-balanced comparison.

### 11.2 New run paths and validation metrics

| Seed | Variant | Status | Best macro validation ST-RAE | Run path |
|---|---|---|---:|---|
| `20261002` | stock | completed | 0.680145541647 | `cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261002/` |
| `20261002` | task_balanced | completed | 0.675366787441 | `cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261002/` |
| `20261003` | stock | completed | 0.690064499897 | `cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261003/` |
| `20261003` | task_balanced | completed | 0.688951856297 | `cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261003/` |
| `20261004` | stock | completed | 0.696088668669 | `cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261004/` |
| `20261004` | task_balanced | completed | 0.694874571039 | `cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261004/` |
| `20261005` | stock | completed | 0.676340834323 | `cyp/artifacts/cyp_002/runs/CYP002-C0-stock-seed20261005/` |
| `20261005` | task_balanced | completed | 0.689190790763 | `cyp/artifacts/cyp_002/runs/CYP002-C0-task-balanced-seed20261005/` |

Checkpoint and metrics SHA-256 values for the eight new runs were not returned in the session execution report and are therefore not invented in this log. The persistent run artifacts remain preserved in their run directories.

### 11.3 Per-task validation results for the four new seeds

| Seed | Variant | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---:|---:|---:|---:|
| `20261002` | stock | 0.905050631700 | 0.555882626589 | 0.830613702209 | 0.429035206090 |
| `20261002` | task_balanced | 0.890828204574 | 0.550430029009 | 0.820181591308 | 0.440027324873 |
| `20261003` | stock | 0.916873370772 | 0.579676327809 | 0.832668153434 | 0.431040147571 |
| `20261003` | task_balanced | 0.883726339391 | 0.583277210580 | 0.836059427696 | 0.452744447520 |
| `20261004` | stock | 0.910284020153 | 0.585101397921 | 0.851626573629 | 0.437342682972 |
| `20261004` | task_balanced | 0.930348530520 | 0.565640931859 | 0.861675537523 | 0.421833284254 |
| `20261005` | stock | 0.910927556994 | 0.559914725880 | 0.819222725101 | 0.415298329319 |
| `20261005` | task_balanced | 0.907715439610 | 0.581635265002 | 0.837372937555 | 0.430039520884 |

---

## 12. Seed-Plumbing Implementation for Replication

The existing C0 implementation originally fixed the training seed as `CYP002_SEED`, which prevented direct execution of the approved additional seed replicates.

The implementation was minimally patched so seed became a manifest-supplied runtime parameter while preserving the scientific C0 configuration.

Changed files:

```text
cyp/scripts/cyp002_manifest.py
cyp/scripts/cyp002_train.py
cyp/tests/test_cyp002_manifest.py
cyp/tests/test_cyp002_train.py
cyp/tests/test_cyp002_train_config_selection.py
```

The patch:

- allows `canonical_experiment_config(..., seed=...)` while retaining `20261001` as the backward-compatible default;
- writes the run seed into the top-level manifest;
- writes the same seed into `canonical_experiment_config.seed`;
- writes the same seed into `canonical_experiment_config.data_loading.shuffle_seed`;
- writes the same seed into determinism seed fields;
- uses the manifest seed for Lightning/PyTorch determinism;
- uses the manifest seed for Chemprop DataLoader seeding;
- validates seed consistency across manifest, canonical configuration, data loading, and determinism fields.

Focused tests after the patch:

`17 passed in 12.62s`

The patch was an implementation change enabling the already-approved five-seed validation protocol. It did not alter C0 architecture, loss, split, metric, or other scientific methodology.

The patched source was transferred to the GPU execution tree before the four additional seeds were run.

---

## 13. Phase-C One-Shot Held-Out Internal-Test Evaluation

Phase C began only after Phase B was frozen and explicitly authorized.

The evaluated held-out set was the frozen 736-compound internal test partition of:

`cyp/data/cyp-challenge-TRAIN_inhibition.csv`

as identified by:

`cyp/artifacts/cyp_baseline/molecule_split_assignments.csv`

No blind challenge labels were accessed.

Exactly one stock inference/evaluation and exactly one task-balanced inference/evaluation were performed.

No training, fine-tuning, validation rerun, or second test evaluation was performed.

### 13.1 Test ST-RAE results

| Variant | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | Macro |
|---|---:|---:|---:|---:|---:|
| Stock | 0.824663483004 | 0.585091883988 | 0.904094491271 | 0.507891954477 | **0.705435453185** |
| Task-balanced | 0.815458848090 | 0.598786250423 | 0.891108892231 | 0.516766879549 | **0.705530217573** |

The exact macro values were:

```text
stock            0.705435453184784
task-balanced    0.705530217573275
```

The test gain is:

$$
\Delta_{\mathrm{test}}
=
0.705435453184784
-
0.705530217573275
=
-0.000094764388491
$$

Negative values favor stock.

### 13.2 Paired cluster-bootstrap result

The frozen bootstrap specification was applied:

```text
B = 10,000
sampling unit = frozen scaffold/analog cluster
sampling = with replacement
pairing = identical resampled cluster multiset for stock and TB
interval = 95% percentile
analysis RNG seed = 20261008
```

Observed interval:

```text
95% bootstrap interval
[-0.011603421562066, 0.011236177474477]

LCB(Delta_test)
-0.011603421562066
```

Seed-noise threshold:

```text
SD_seed = 0.007094968910757
```

Decision comparison:

```text
LCB > SD_seed?
-0.011603421562066 > 0.007094968910757
FALSE
```

### 13.3 Frozen-rule verdict

**Stock remains the C0 headline.**

Task-balanced does not satisfy the frozen criterion and therefore does not displace stock.

This is not a post hoc decision: the materiality rule and bootstrap procedure were frozen before the test numbers were observed.

---

## 14. Required Phase-C Prediction Matrices

The two complete per-compound internal-test prediction matrices were saved and preserved.

### Stock matrix

`cyp/artifacts/cyp_002/heldout_internal_test_eval/CYP002-C0-stock-seed20261001-r2_heldout_test_predictions.csv`

SHA-256:

`1ebfbe5c226ec968445231e7dc7f13f25485f1767c1a4964605768f0c3b2cbba`

Rows:

`736`

### Task-balanced matrix

`cyp/artifacts/cyp_002/heldout_internal_test_eval/CYP002-C0-task-balanced-seed20261001-r1_heldout_test_predictions.csv`

SHA-256:

`18d0665de15b50784c621fadf282a499e1a78d0277a8f1558df5a79a1fb587bf`

Rows:

`736`

The read-only provenance extraction confirmed:

- both files contain the same 736 `molecule_name` values in the same order;
- both contain `cluster_id`;
- both contain four observed internal-test targets;
- both contain four predictions;
- both contain the confidence bounds used for ST-RAE;
- the prediction matrix files were not modified during provenance extraction;
- the blinded 750-compound file was not accessed.

The provenance extraction artifact was:

`phase_c_prediction_matrix_provenance.md`

---

## 15. Complete Scientific Outcome

Session 002 establishes the following:

### Validation sensitivity grid

- C0-C7 x two variants complete;
- 16 validation cells complete;
- `k = 1`;
- `k/8 = 0.125`.

### Step-1 error characterization

The weakest validated tasks were CYP1A2 and CYP2D6. The uncertainty-band diagnostic showed higher raw MAE in wide uncertainty bands, while lower wide-band ST-RAE was recognized as mechanically expected from the ST-RAE construction. The nearest-training-neighbor ECFP4 chemical-distance relationship was effectively null for both tasks.

The resulting classification was that weak CYP1A2/CYP2D6 performance is more consistent with broad assay/label uncertainty than insufficient chemical coverage under the tested diagnostic.

### Seed noise

Five paired C0 validation seeds were characterized:

`20261001` through `20261005`.

Frozen noise scale:

`SD_seed = 0.007094968910757`

### Held-out internal test

Exactly one stock and one task-balanced C0 internal-test evaluation were performed.

Macro test ST-RAE:

```text
stock            0.705435453184784
task-balanced    0.705530217573275
```

Paired test gain:

`Delta_test = -0.000094764388491`

95% paired scaffold-cluster percentile-bootstrap interval:

`[-0.011603421562066, 0.011236177474477]`

Headline decision:

**Stock remains the C0 headline under the frozen rule.**

---

## 16. Artifact Preservation

All artifacts from the session are to be preserved, including:

- completed C0-C7 x 2 run directories;
- the invalidated first stock C0 run;
- corrected C0 stock and task-balanced runs;
- Step-1 prediction and diagnostic artifacts;
- all five-seed validation run directories;
- Phase-B audit record;
- Phase-C prediction matrices;
- Phase-C prediction-matrix provenance report;
- bootstrap outputs;
- manifests, metrics, checkpoints, and hashes generated by the existing run machinery;
- implementation and regression-test changes.

No intermediate artifact is to be deleted merely because it is not expected to appear in the final model deliverable.

---

## 17. Scientific Boundaries and Non-Claims

- Tripwire remains a DDI-liability characterization/re-ranking module with **no attrition authority**.
- The CYP model's result does not create candidate-attrition authority.
- The 750-compound blind challenge remains prospective external validation and was not accessed in this session.
- The Step-1 label-uncertainty classification is comparative and does not establish assay uncertainty as the sole cause of weak performance.
- The five-seed `SD_seed` is an empirical sample SD across the five prespecified seeds, not an exhaustive characterization of all possible stochastic variation.
- The C0 held-out result applies to the frozen 736-compound internal analog-cluster test partition, not to the 750-compound blind challenge set.

---

## 18. Session Closure

**CLOSED — Tripwire-CYP-002**

The session completed the approved CYP-002 validation and headline-test protocol.

The final C0 headline is:

> **Stock masked multitask MSE.**

It remains the headline because the task-balanced variant's 95% paired scaffold-cluster bootstrap lower bound did not exceed the measured five-seed stochastic noise scale.

No additional C0 training or test evaluation is warranted under the frozen protocol.

CYP-002 is closed. Further CYP work proceeds only under the next separately authorized project step.
