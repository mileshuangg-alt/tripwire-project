# Tripwire-CYP-002 Handoff — Build and evaluate the masked-label multitask challenger

**Date prepared:** 2026-10-01  
**Purpose:** Start Tripwire-CYP-002 at the exact point where Tripwire-CYP-001 ended, without recreating the CYP environment, rediscovering the challenge dataset, reconstructing the analog-cluster split, repeating the baseline tuning, or searching for completed artifacts.

## 0. Standing rule

**This handoff is the audit.**

Start at the first unresolved scientific/execution step in Section 10.

Do **not** begin Tripwire-CYP-002 with:

- repository archaeology;
- broad `find`, `grep`, or filesystem discovery;
- rediscovering the OpenADMET dataset;
- rebuilding the 4,905-row training matrix audit;
- reclustering the molecular universe;
- recreating the 70/15/15 split;
- rerunning the four 30-trial LightGBM searches;
- rerunning the ST-RAE diagnostic or callback debugging;
- rerunning the held-out independent baseline evaluation;
- asking the user to relocate artifacts already named here;
- recreating the OpenADMET tutorial environment unless a concrete dependency problem is demonstrated.

Tripwire-CYP-001 is closed. Its artifacts are authoritative for the independent reference baseline.

If execution exposes a specific missing or corrupt artifact, identify the exact artifact and the exact frozen work package it blocks. Do not turn a concrete missing-file problem into a general discovery phase.

---

## 1. Critical boundary

Tripwire-CYP-001 established and froze the first independent CYP reference baseline.

The next session is **not baseline development**. It is the preregistered masked-label multitask challenger against that frozen reference.

The scientific question is:

> Does shared molecular supervision across the four partially observed CYP tasks improve per-isoform prediction relative to the frozen independent ECFP4 + LightGBM reference?

This is a separate experiment. The frozen independent baseline must not be retuned to accommodate the challenger.

Tripwire remains a **characterization/re-ranking module, not an attrition gate**.

---

## 2. What Tripwire-CYP-001 achieved

### 2.1 Dataset audit — frozen factual structure

Authoritative direct-inhibition training data:

```text
4,905 compounds
18 columns
```

Blind test:

```text
750 compounds
2 columns: Molecule_Name, SMILES
```

Observed direct-inhibition labels:

```text
CYP1A2 = 1,412
CYP2C9 = 1,285
CYP2D6 = 1,493
CYP3A4 = 2,335
```

Exact label-count distribution:

```text
1 measured isoform: 3,596
2 measured isoforms: 1,039
3 measured isoforms:   229
4 measured isoforms:    41
```

Train/test identity checks:

```text
Molecule_Name overlap = 0
SMILES overlap        = 0
```

SMILES parseability:

```text
training: 0 invalid / 4,905
test:     0 invalid /   750
```

Confidence-interval ordering was valid for all four isoforms.

Do not redo this audit unless a specific integrity problem is demonstrated.

### 2.2 Frozen analog-cluster split

Clustering representation:

```text
Morgan radius = 2
nBits         = 2048
```

Clustering:

```text
Butina
Tanimoto threshold = 0.70
```

Final split:

```text
train      = 3,433
validation =   736
test       =   736
```

Cluster integrity:

```text
every cluster occurs in exactly one split
```

Per-isoform label fractions were approximately 70/15/15 in every endpoint.

Authoritative split artifact:

```text
cyp/artifacts/cyp_baseline/molecule_split_assignments.csv
```

Do not recluster or regenerate this split in Tripwire-CYP-002.

### 2.3 Official ST-RAE implementation

The challenge's official `rae_soft_threshold_absolute_error` implementation was preserved from the pinned upstream tutorial commit.

Upstream repository:

```text
OpenADMET/CYP-Challenge-Tutorial
```

Captured commit:

```text
832ae6d6447b07c2c71059de34d07f1130388247
```

Tripwire provenance copy:

```text
references/cyp_upstream/custom_scoring_functions.py
```

The ST-RAE adapter and LightGBM callback were tested and passed.

Do not replace the official ST-RAE definition with a newly derived metric.

### 2.4 Frozen independent baseline

Reference model:

```text
four independent LightGBM GBDT regressors
one per CYP isoform
Morgan radius 2 / 2048 bits
```

Tuning:

```text
30 randomized trials per isoform
validation ST-RAE only
n_estimators = 2000
bagging_freq = 1
max_depth = default
50-round early stopping
```

Search dimensions:

```text
learning_rate:    log-uniform 0.005–0.2
num_leaves:       integer 7–63
min_child_weight: log-uniform 0.001–10
subsample:        uniform 0.5–1.0
```

Frozen validation winners:

```text
CYP1A2
  trial = 15
  validation ST-RAE = 0.8878237148303819
  best iteration    = 195
  learning_rate     = 0.007493581432955413
  num_leaves        = 41
  min_child_weight  = 0.16006543580783064
  subsample         = 0.5309983063408124

CYP2C9
  trial = 30
  validation ST-RAE = 0.7991426892733124
  best iteration    = 69
  learning_rate     = 0.07921505531874473
  num_leaves        = 25
  min_child_weight  = 0.04277500477345989
  subsample         = 0.7985177920408508

CYP2D6
  trial = 14
  validation ST-RAE = 0.8730698116139428
  best iteration    = 108
  learning_rate     = 0.04023960497447025
  num_leaves        = 61
  min_child_weight  = 0.12177920971919333
  subsample         = 0.6037007057447841

CYP3A4
  trial = 24
  validation ST-RAE = 0.6547707974523649
  best iteration    = 570
  learning_rate     = 0.0593598469086799
  num_leaves        = 24
  min_child_weight  = 0.07660312692852367
  subsample         = 0.9132423504906277
```

### 2.5 CYP3A4 provenance exception

CYP3A4 was **not rerun** after deterministic explicit isoform seeds were introduced.

Its existing corrected 30-trial run is authoritative because there is no preregistered basis for choosing between two candidate searches.

This exception is already recorded in:

```text
cyp/artifacts/cyp_baseline/frozen_baseline_config.json
```

Do not reopen or rerun CYP3A4 tuning merely to make the seeds cosmetically symmetric.

### 2.6 Final held-out independent baseline

The held-out test was evaluated exactly once after all four configurations were frozen.

Per-isoform results:

```text
CYP1A2
  n_test       = 211
  ST-RAE       = 0.795530124263526
  MAE          = 0.6710209790014506
  R2           = 0.23342485714420458
  Spearman_R   = 0.5554604565603404
  Kendall_Tau  = 0.392462198149402

CYP2C9
  n_test       = 192
  ST-RAE       = 0.9486531366401376
  MAE          = 0.6163640449522414
  R2           = 0.06614515953648092
  Spearman_R   = 0.30236788921140445
  Kendall_Tau  = 0.20517015706806285

CYP2D6
  n_test       = 224
  ST-RAE       = 0.9378919703293982
  MAE          = 0.6088487140783577
  R2           = 0.15166724628304762
  Spearman_R   = 0.37064605238258114
  Kendall_Tau  = 0.2506856720087549

CYP3A4
  n_test       = 349
  ST-RAE       = 0.6739114677520612
  MAE          = 0.6340492801732581
  R2           = 0.4194744641346977
  Spearman_R   = 0.6590947618240446
  Kendall_Tau  = 0.477167558730818
```

Macro-held-out reference:

```text
ST-RAE       = 0.8389966747462807
MAE          = 0.6325707545513269
R2           = 0.2176779317746077
Spearman_R   = 0.47189228999459265
Kendall_Tau  = 0.33137139648925945
```

This is the frozen independent reference. Do not recompute it as part of Tripwire-CYP-002 unless a concrete artifact-integrity failure is demonstrated.

---

## 3. Exact standalone Tripwire repository state

Tripwire is now a standalone repository, separate from `sbdd-project` and separate from the upstream OpenADMET tutorial repository.

Local repository:

```text
~/Desktop/tripwire-project
```

Expected top-level layout:

```text
README.md
.gitignore
cyp/
notes/
references/
pxr/
shared/
```

CYP baseline code:

```text
cyp/scripts/ecfp4.py
cyp/scripts/st_rae.py
cyp/scripts/lgbm_smoke_test.py
cyp/scripts/lgbm_tuner.py
cyp/scripts/freeze_baseline_config.py
cyp/scripts/evaluate_baseline_test.py
```

CYP data:

```text
cyp/data/cyp-challenge-TRAIN_inhibition.csv
cyp/data/cyp-challenge-TEST-BLINDED.csv
```

CYP artifacts:

```text
cyp/artifacts/cyp_baseline/
```

This directory contains the frozen split, provenance, tuning trial tables, winner JSONs, frozen baseline config, final test predictions, and `baseline_test_results.json`.

Session record:

```text
notes/Tripwire_CYP_001_Session_Log.md
```

Project log:

```text
notes/project_log.md
```

Upstream scorer/provenance:

```text
references/cyp_upstream/
```

Do not ask the user to rediscover these locations.

---

## 4. Exact CYP artifacts to use

Authoritative frozen baseline config:

```text
cyp/artifacts/cyp_baseline/frozen_baseline_config.json
```

Authoritative molecule-to-split mapping:

```text
cyp/artifacts/cyp_baseline/molecule_split_assignments.csv
```

Tuning records:

```text
cyp/artifacts/cyp_baseline/tuning/
```

Final baseline test results:

```text
cyp/artifacts/cyp_baseline/test_evaluation/baseline_test_results.json
```

Final baseline predictions:

```text
cyp/artifacts/cyp_baseline/test_evaluation/CYP1A2_pIC50_direct_inhibition_predictions.csv
cyp/artifacts/cyp_baseline/test_evaluation/CYP2C9_pIC50_direct_inhibition_predictions.csv
cyp/artifacts/cyp_baseline/test_evaluation/CYP2D6_pIC50_direct_inhibition_predictions.csv
cyp/artifacts/cyp_baseline/test_evaluation/CYP3A4_pIC50_direct_inhibition_predictions.csv
```

Dataset provenance:

```text
cyp/artifacts/cyp_baseline/provenance.json
```

The upstream tutorial provenance is recorded in:

```text
references/cyp_upstream/README.md
```

Do not rebuild any of these artifacts merely to verify that they exist.

---

## 5. What Tripwire-CYP-002 is for

The preregistered next experiment is the **masked-label multitask challenger**.

The scientific question is whether a shared molecular representation can exploit the sparse cross-isoform supervision without requiring complete-case examples.

The frozen facts driving this experiment are:

```text
4,905 training compounds
3,596 compounds have exactly one measured isoform
1,039 have exactly two
229 have exactly three
41 have all four
```

Therefore:

```text
complete-case training is prohibited
```

The multitask model must train with missing labels **masked**, so compounds measured for one, two, or three isoforms remain usable for the corresponding observed tasks.

The challenger is evaluated per isoform on the **same 736-compound held-out test partition** used by the independent baseline.

The independent baseline is frozen and cannot be changed after seeing multitask results.

---

## 6. Do NOT reopen the baseline

Closed baseline work:

- dataset source selection;
- dataset schema audit;
- label-overlap audit;
- train/test identity check;
- SMILES parseability check;
- confidence-interval consistency check;
- analog-cluster threshold = 0.70;
- 70/15/15 split proportions;
- frozen molecule-level split artifact;
- fixed Morgan radius-2 / 2048-bit representation;
- official ST-RAE definition;
- ST-RAE callback implementation;
- independent LightGBM model family;
- randomized search dimensions/ranges;
- 30-trial budget per isoform;
- 2,000 maximum estimators;
- 50-round early stopping;
- `bagging_freq=1`;
- four independent validation winners;
- frozen CYP3A4 provenance exception;
- held-out independent baseline evaluation.

If a later challenger needs a new implementation of a shared primitive, reuse the existing scientifically validated primitive instead of creating a parallel definition.

---

## 7. First unresolved scientific/execution step

Do **not** begin by writing the multitask model immediately.

The first work package is a focused **multitask-model design and alternatives pass**.

Before implementation:

1. enumerate serious masked-label multitask alternatives for molecular regression;
2. identify what each enables that the others do not;
3. assess compute, dependencies, and scope;
4. verify that each candidate can consume the fixed Morgan representation or establish why a different representation is necessary;
5. verify how missing labels are masked without dropping partial-label compounds;
6. define the shared output structure for the four regression tasks;
7. define the exact training loss and masking semantics;
8. confirm how validation ST-RAE will be computed per isoform;
9. obtain user approval before freezing scientifically consequential architecture choices.

This alternatives pass is required by the project's standing model-selection rule. Do not import a neural architecture merely because it is familiar or convenient.

---

## 8. Evaluation protocol for the multitask challenger

The comparison must use the same frozen split:

```text
train = 3,433 compounds
validation = 736 compounds
test = 736 compounds
```

For each target, training loss is computed only where that target is observed.

Conceptually:

```text
compound i
    |
    +--> CYP1A2 label present?   mask / train
    +--> CYP2C9 label present?   mask / train
    +--> CYP2D6 label present?   mask / train
    +--> CYP3A4 label present?   mask / train
```

Complete-case restriction to the 41 four-label compounds is explicitly prohibited.

Hyperparameter selection for the multitask challenger must not use the held-out test set.

Final comparison:

```text
masked multitask
       vs
frozen independent baseline
```

Compare per isoform using the same test-set metrics:

```text
ST-RAE
MAE
R2
Spearman_R
Kendall_Tau
```

The macro values may be reported alongside the per-isoform results, but the preregistered comparison is per isoform.

---

## 9. Challenger disposition

The approved Section 5 drop condition is:

> If the masked-label multitask challenger does not beat the tuned independent baseline on a per-isoform basis, drop the multitask approach and record the comparison in the decision record.

Do not silently weaken this criterion after seeing results.

Do not average away a per-isoform failure.

Do not retune the independent baseline after observing the multitask result.

Do not use the blind OpenADMET submission outcome to retroactively alter the internal test comparison.

Tripwire remains characterization/re-ranking only.

---

## 10. Environment and execution state

The completed CYP-001 development environment was established from the OpenADMET tutorial `environment.yaml`.

Relevant environment used for the baseline:

```text
oadmet_cyp_tutorial
```

Verified packages included LightGBM, RDKit, and scikit-learn.

Do not recreate the environment if the existing one is still available.

First verify only what execution actually requires:

```bash
conda activate oadmet_cyp_tutorial
cd ~/Desktop/tripwire-project
```

Then run the targeted dependency/import check needed by the selected multitask implementation.

Do not run broad environment archaeology.

---

## 11. Git / repository handling

Tripwire is independent from `sbdd-project`.

The intended remotes follow the existing SBDD convention:

```text
origin  = git.ucsf.edu:Miles-Huang/tripwire-project.git
github  = https://github.com/mileshuangg-alt/tripwire-project.git
```

Stage only intended files.

Do not:

```text
git add .
git clean
```

The CYP-001 standalone migration included the completed baseline artifacts and session record so that the Git history is part of the audit trail.

Before making Tripwire-CYP-002 changes, inspect only the specific repository state needed for the next implementation step.

---

## 12. Artifact preservation rules

Preserve all CYP-001 artifacts, including:

- all 30-trial tuning tables for all four isoforms;
- all four winner JSONs;
- the invalid initial all-zero CYP3A4 tuning attempt, if present in the migrated artifact history;
- the final frozen baseline config;
- the frozen split mapping;
- final held-out predictions;
- final test metrics;
- the ST-RAE diagnostic evidence;
- the upstream scorer provenance copy;
- the completed CYP-001 session log.

The invalid all-zero tuning run is historical evidence of a corrected callback-signature failure and must not be used for selection or comparison. Do not delete it merely because it is invalid.

New Tripwire-CYP-002 experiments should likewise preserve trial tables, failed configurations, diagnostics, and rejected approaches until the challenger is finalized.

---

## 13. Claim boundaries

### Authorized

Tripwire-CYP-001 established:

- a fixed analog-cluster internal evaluation protocol;
- a tuned independent ECFP4 + LightGBM reference;
- a frozen per-isoform validation configuration;
- a one-time held-out internal baseline evaluation;
- a macro held-out baseline ST-RAE of 0.8389966747462807.

### Not established

Tripwire-CYP-001 does **not** establish:

- performance on the hidden OpenADMET leaderboard;
- external prospective generalization beyond the internal analog-cluster holdout;
- superiority of any neural challenger;
- that the blind challenge's evaluation is identical to the internal held-out test distribution in every respect;
- DDI liability decisions with attrition authority.

Do not turn the internal baseline into a claim about the final external leaderboard before an actual submission and external result exist.

---

## 14. Minimal resume procedure

Start Tripwire-CYP-002 with:

```bash
cd ~/Desktop/tripwire-project
conda activate oadmet_cyp_tutorial
```

Then read this handoff and go directly to Section 7.

The baseline is already frozen.

The data are already present.

The split is already present.

The official ST-RAE implementation is already present.

The baseline artifacts are already present.

The final held-out comparison is already complete.

No cluster archaeology, dataset rediscovery, baseline rerun, or environment reconstruction is required to begin the next scientific step.

---

## 15. Intended Tripwire-CYP-002 completion state

Minimum target:

```text
multitask alternatives pass completed
+
scientific architecture frozen
+
masked-label loss semantics frozen
+
implementation/tests complete
+
validation-only training/tuning complete
+
multitask configuration frozen
```

Then:

```text
same held-out test partition evaluated
+
per-isoform comparison against frozen independent baseline
+
disposition recorded in decision record
```

If the multitask model fails the preregistered per-isoform comparison:

```text
drop multitask approach
+
record result
+
move to next preregistered challenger
```

If it passes:

```text
retain multitask approach as an evaluated challenger
+
record result
+
continue according to the frozen Tripwire model order
```
