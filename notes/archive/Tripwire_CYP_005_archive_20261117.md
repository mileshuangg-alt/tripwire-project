# Tripwire-CYP-005 — Session Log

**Session:** Tripwire-CYP-005
**Date:** start UNKNOWN (recovered history imported 2026-11-14); in progress through 2026-11-17 (Miles calls the 2026-11-17 work "Session 005.5", a continuation of this session, recorded here by topic; there is no separate 005.5 file)
**Status:** CLOSED — session close declared by Miles 2026-11-17 ("this will close session 5, after this i will start a new session for session 6"); closes with the commit that follows this edit. Open items are carried into Session 006 unresolved (§12).
**Workstream:** Tripwire — CYP inhibition arm; ActFound-CYP whole-package study
**Governing decision:** D013 and D013 Amendments 1–4 (`notes/Decisions.md`), all Approved; D015 (2026-11-17 comparison plan; form approved, authorizes no run; whether it supersedes any D013 term is an open question)
**Execution:** CoreHPC project `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project`; pinned ActFound env `/scratch/user/mhuang8/actfound/envs/actfound_c96f055` (Python 3.9.18, torch 1.12.1, RDKit 2023.03.1). Local frozen identity env `tripwire_cyp002` (Python 3.12.14, RDKit 2026.03.6, InChI 1.07.3); scoring in a Python ≥ 3.10 env.
**Detailed records (never edited):**
- **A1** = `notes/archive/Tripwire_CYP_005_archive_20261115.md`: the original chronological log, §1–§55. Where `notes/Decisions.md` cites "Session 005 §N", that means A1 §N.
- **A2** = `notes/archive/Tripwire_CYP_005_archive_20261116.md`: the compact entry log, entries 1–51. Where `notes/Decisions.md` cites "Session 005 entry N", that means A2 entry N.

Exact approval wording, commands, pasted outputs and full hash inventories are in these two files. This report consolidates them by topic.

---

## 1. Session Objective

The session began by reconstructing ActFound and running one production training of it on a ChEMBL32 corpus regenerated from the SQLite release (job 3066823). An identity audit then showed that corpus was not firewalled against OpenADMET structures. The session's objective became the D013 study:

1. build a ChEMBL32 foundation corpus firewalled against all OpenADMET structures;
2. retrain the ActFound foundation model on it once and freeze the checkpoint;
3. qualify target adaptation on the frozen D007 cells;
4. evaluate with the unchanged 150-cell comparison against frozen T121.

Only whole-package claims are made.

Comparator: O150(T121) = 0.701212576796 (D013 Amendment 4 item 6). The ST-RAE scorer is the frozen upstream implementation, unchanged.

D012 is project background only. It is not this study's arm definition or interpretation contract (A1 §3, §26 E).

---

## 2. Governing Scope

| Record | Status | What it fixes | What it does not authorize |
|---|---|---|---|
| D013 | Approved 2026-11-15 | Firewall: all 5,655 OpenADMET structures, by full Standard InChIKey. Base corpus: the authors' released `chembl_processed_chembl32.csv`. Molecule-level removal, with assays below 20 usable molecules dropped and logged in every split. Retrain once; checkpoint freeze rule. Adaptation on each D007 cell's training molecules; 150-cell comparison against T121. Job 3066823 is not usable as the study checkpoint. | Production training or evaluation by itself |
| Amendment 1 | Approved 2026-11-15 | Route-C graph re-qualification behind an exact reproduction gate. Conservative pretraining exclusion of largest-fragment full-key matches, applied to train and validation. Corpus identity acceptance check. | Build, reproduction run, benchmark, training |
| Amendment 2 | Approved 2026-11-15 | Gate comparison scope: `record_id_unused` and `graph_pair_count` declared non-comparable but reported. Item 2 (`standard_value` at 15 significant figures) was later replaced by Amendment 3. | Build, benchmark, training |
| Amendment 3 | Approved 2026-11-15 | Replaces Amendment 2 item 2: `standard_value` agrees within a relative tolerance of 1e-14; the tolerance's effect is reported in the manifest | Build, benchmark, training |
| Amendment 4 | Approved 2026-11-16 | Settles D013 open items 1–4. O150 is computed from 200 unique cell-endpoint adaptations, each recorded into its three seed slots as a replicate. pIC50 with no conversion; complete support, no cap. Thin target adapter as specified by the approved brief. Adaptation and scoring run in separate processes. Comparator citation kept. | Any run; implementation of the thin adapter |

D013 adopts the study Miles stated on 2026-11-15 as established on 2026-11-12 (A1 §26 E). He also said not to reopen the architecture or adaptation-path menu, not to add a published-fidelity pass/fail gate, and not to reselect epochs through extra validation draws.

---

## 3. Starting Point: Job 3066823 and the Leakage Finding (historical)

Before D013, the pipeline was reconstructed from recovered GPT and handoff records (A1 §1–§20):
- a ChEMBL32 graph-qualified population, 54,732 assays and 2,773,995 rows, merged TSV `24d9fd29…`;
- the ActFound model side and production runner;
- a per-ID split reconciliation, 33,296 train + 500 validation assays.

Several early jobs failed on path, header, split-mapping and GPU-architecture problems. The production request was then constrained to L40S (A1 §7, §11).

Job 3066823 completed `0:0` in 12:16:08: 60 epochs, best epoch 58, validation score 0.9770. Checkpoint selection and the reported metrics use the same 500 validation assays, and no held-out evaluation exists (A1 §21–§26).

The training corpus had no identity firewall at any stage; the projection and pregraph summaries both record `openadmet_firewall_applied: false` (A1 §26–§29). The full-corpus audit in the frozen identity env found these OpenADMET molecules in loader-kept training rows (A1 §30–§31):

| Protected tier | Molecules | Training rows |
|---|---:|---:|
| D007 development | 203 | 2,139 |
| internal test | 44 | 444 |
| blinded challenge | 7 | 25 |

Miles chose Option A, retraining on a firewalled corpus (A1 §32). Job 3066823's outputs are preserved unchanged and are not the study checkpoint.

---

## 4. Methods

### 4.1 Firewall and pretraining exclusion

Identities were computed in the frozen identity env from the corpus SMILES, matched against the 5,655 protected OpenADMET structures (A1 §36, §39–§42):

| List | Molregnos | Basis |
|---|---:|---|
| Firewall list v1 | 337 | full Standard InChIKey: development 263, internal test 61, blinded 13 |
| Largest-fragment additions | 180 | largest-fragment full key: development 140, internal test 40 |
| **Exclusion list v2** | **517** | v1 ∪ the 180, each with its `exclusion_basis` |

First-block and largest-fragment first-block near-matches are reported only; they are not removed and not declared safe. The identity rule remains the full Standard InChIKey. The largest-fragment exclusion is protective, not an identity claim.

### 4.2 Route C and the reproduction gate

Upstream graph eligibility depends on pre-graph assay size, so it cannot be recomputed from the authors' post-graph rows (A1 §36 E). Route C therefore works as follows:
1. Take the authors' assay universe.
2. Rebuild the pre-graph rows (pregraph `43dfd672…`).
3. Apply the firewall.
4. Rerun the unchanged, hash-checked graph stage.

Before firewall mode may run, a reproduction-mode run (no firewall) must reproduce the authors' file inside the pinned upstream train/valid split. The output must also give a PASS manifest under the current comparison scope. Outside-split differences are reported but do not block (A1 §38–§39, §43).

### 4.3 Acceptance

Identities are recomputed from the built corpus file in the frozen env. The check passes only if no train or validation row has a full-key or largest-fragment full-key match, and no train or validation row has an unverifiable identity. It fails closed (A2 entries 32, 35).

### 4.4 Adaptation and scoring

Adaptation runs in the pinned env with the frozen checkpoint and writes a hash-pinned predictions artifact. The frozen scorer runs separately in a Python ≥ 3.10 env (Amendment 4 item 5).

---

## 5. Results

### 5.1 Reproduction gate

| Job | Builder / comparison scope | Gate | Blocking in-split differences |
|---|---|---|---|
| 3073155 | `c73d6a7c…`, exact | FAIL, exit 3 | 1,525,375 rows |
| 3075157 | `096951eb…`, `a358c320693929d9` (Amendment 2) | FAIL, exit 3 | 4 `standard_value` rows |
| 3075299 | `9c5d449b…`, `0ca8f477310cb7f3` (Amendment 3) | **PASS, exit 0** | 0 |

What the diagnosis of all 1,959,398 compared rows found (A1 §47–§53):
- **Fields that define the training data:** no row differs in any of them.
- **`record_id_unused`:** differs in every mismatching row. It is a provenance column that upstream does not consume.
- **`standard_value`:** differs by at most 4.35e-15 relative, a text-precision difference. 8,657 of the authors' values carry more than 15 significant digits, so Amendment 2's 15-significant-figure premise was false. Job 3075157's 4 blocking rows sat on the 15-significant-figure rounding boundary; Amendment 3 replaced that rule with a relative tolerance.
- **`graph_pair_count`:** differs in 69 assays, by 1–47 pairs and almost always upward. It is metadata only. Exact-equality pair exclusion was ruled out as the cause. The remaining explanation, Morgan-fingerprint similarity under a different RDKit build, was reached by elimination; similarity was not measured. Byte-identity with the authors' file is not achievable from the pinned env.

Job 3075299's PASS manifest is `/scratch/user/mhuang8/actfound/chembl32/d013_reproduction_20261115c/corpus_manifest.json`, SHA-256 `2e8a6d184380a3c74796b884db8d30ffb5792f12be908385cadaa597599b9306`. Within tolerance but not equal as text: 600 rows in split and 203 outside (A2 entry 27).

### 5.2 Firewalled corpus build (job 3075410)

| Quantity | Reproduction (3075299) | Firewalled (3075410) |
|---|---:|---:|
| Pre-graph rows kept | 2,027,275 | 2,018,649 |
| Rows removed by firewall | 0 | 8,626 (7,870 full-key, 756 largest-fragment) |
| Graph-qualified assays | 47,324 | 47,187 |
| Assays rejected by graph valid-count gate | 0 | 137 |
| Final rows | 1,959,399 | 1,946,000 |

The firewalled corpus is `actfound_chembl32_d013_firewalled.tsv`, SHA-256 `8b00f349726e7cc0d84d25195c3bdf09f76089159a93eed58401932ad0cbdecd`. Job 3075410 completed `0:0` in 00:11:42 (A2 entry 30).

The 137 rejected assays (A2 entry 31):
- By upstream split: 77 `train`, 1 `valid`, 2 `test`, 1 `test_external`, 56 in no split.
- Cause: removing molecules also removes their graph neighbours, so an assay can fall below 20 valid molecules even with many rows left.
- These drops are the ineligible assays D013 says are to be logged.
- Final-row accounting: the loss of 13,399 rows = 9,221 rows in the dropped assays + 4,178 rows lost inside 2,459 kept assays.

Per-assay valid-molecule counts were not measured.

### 5.3 Acceptance check

The first run failed on one `train` row (A2 entry 33): molregno 2336618, a hexafluorophosphate salt whose SMILES the frozen RDKit rejects, so its identity could not be recomputed. Two keys were checked against the protected sets: ChEMBL's standard InChIKey `NASMLRHAGIXRIG-UHFFFAOYSA-N` and the organic-cation key `JELBKKMJVYHXBP-UHFFFAOYSA-N`. Neither is a protected key or a first-block match.

Miles adjudicated that row as not a protected structure (§6). The adjudication was added as a hash-pinned input, `acceptance_adjudications_v1.tsv` `bc4647a5…`.

The rerun PASSED: zero full-key and zero largest-fragment full-key matches in every scope, and no exclusion-list molregno in the corpus. All 1,946,000 rows were checked, covering 846,299 distinct SMILES (A2 entry 36).

Caveat: that one row passes by adjudication, not by a recomputed identity.

### 5.4 Split reconciliation and config v2 (job 3076199)

Train: 33,137 of 33,296 upstream IDs retained (159 dropped). Validation: 498 of 500 (2 dropped). Config v2 is `dad741d0b6a3e1f014d47579acec29e9bacdad3ac7b1fb7b0c8b2f69a1984af6`; it was validated by runner v2 and points at the firewalled corpus (A2 entry 37).

The loader drops 82 train and 1 validation assays beyond those the graph stage removed. Its gates are: percent units skipped, `=` relation, 20–10,000 rows, RDKit parse. Per-assay reasons were not measured.

### 5.5 Benchmark and runtime extrapolation

The benchmark was interactive job 3076227: one L40S, running the unchanged production wrapper on a config v2 copy that differs only in `output_dir` (A2 entry 38).
- Measured: load 506.5 s; epoch 0 932.7 s; max GPU memory 1,747 MiB.
- Planning estimate: 15.7 h (an epoch-0 lower bound). The estimate scaled from job 3066823 was 12.2 h.
- The 72 h request was adequate. The actual run took 12:17:02.

### 5.6 Production foundation training (job 3076848)

| | 3066823 (historical, unfirewalled) | 3076848 (D013) |
|---|---|---|
| Corpus | `24d9fd29…` | `8b00f349…` |
| Validation assays | 500 | 498 |
| Batches per epoch | 2,410 | 2,398 |
| Elapsed | 12:16:08 | 12:17:02 |
| Initial validation score | 0.3485 | 0.3546 |
| Best epoch / score | 58 / 0.9770 | 51 / 0.98231 |

The two runs are not like-for-like: different corpus and validation split. The integrity check passed (A2 entries 39–41):
- 60 contiguous epochs, no non-finite losses;
- best epoch equals `completion.json`;
- `model_best.pt` equals `model_51.pt`, `upstream_commit` `c96f0559…`;
- the startup source hashes equal the deployed files;
- the deployed runner v2 and model side equal the reviewed hashes.

**Frozen foundation checkpoint (D013 checkpoint freeze rule):** CoreHPC `/scratch/user/mhuang8/actfound/chembl32/production_training_d013/actfound_chembl32_d013_v2/checkpoints/model_best.pt`, epoch 51, SHA-256 `4db29a04f69ecc1ef6ec2f058aa0335e132394a2a0bbe36c14bd27d54a49de2b`. It was selected on the firewalled upstream `valid` split before any CYP data was used.

### 5.7 Target-adaptation qualification

Cell counts were taken from the frozen D007 manifests, 50 partitions × 4 endpoints. No protected internal-test or blinded molecule is in any cell (A2 entry 43).

| Endpoint | Support per cell | Target per cell |
|---|---|---|
| CYP1A2 | 1,062–1,101 | 100–139 |
| CYP2C9 | 965–1,008 | 85–128 |
| CYP2D6 | 1,123–1,157 | 112–146 |
| CYP3A4 | 1,766–1,815 | 171–220 |

**Stage 1, adaptation (CoreHPC, interactive L40S, pinned env; three fixed CYP3A4 partitions; A2 entry 47).** Smoke used synthetic molecules only and passed.

| Partition | Support / target | adapt_total (s) | Peak GPU allocated (MiB) | Peak host RSS (MiB) | Offset max \|Δpred\| |
|---|---|---:|---:|---:|---:|
| `rep_01_fold_01` | 1,799 / 187 | 0.782 | 443.9 | 2,660 | 1.19e-5 |
| `rep_03_fold_05` | 1,793 / 193 | 0.026 | 442.6 | 2,692 | 1.34e-5 |
| `rep_05_fold_10` | 1,789 / 197 | 0.024 | 442.3 | 2,692 | 1.31e-5 |

- **All checks passed:**
  - Repeat across two seed values: max |Δpred| exactly 0.0.
  - Scrambled target labels: exactly 0.0.
  - pIC50 vs the pretraining scale: 1.19–1.34e-5, against a declared threshold of 1e-3.
- **Cost:** steady state is about 0.34 s per cell, mostly CPU featurization. The first cell's extra time is attributed to CUDA warm-up; that was not measured directly.
- **Projection for all 200 adaptations:** 68–228 s, assuming cells run independently and sequentially. Cost scaling with support size was not characterized.

**Stage 2, scoring (local, Python 3.12).** `score_predictions.py` `88e2d400…` ran with the frozen scorer on the checks artifact `adaptation_checks.json` `1be10d32…` (local copy equal to the CoreHPC hash). `all_bounds_hold` is true (A2 entry 49). Output: `qualification/actfound/d013_target_qualification_20261116/strae/strae_scores.json` `2240436d…` (A2 entry 50).

| Partition | ST-RAE, pIC50 (both seeds and scrambled targets identical) | pretraining-scale arm, corrected | uncorrected (expected ~15) |
|---|---:|---:|---:|
| `rep_01_fold_01` | 0.72253 | 0.72253 | 15.26 |
| `rep_03_fold_05` | 0.72195 | 0.72195 | 15.22 |
| `rep_05_fold_10` | 0.77181 | 0.77181 | 15.98 |

The corrected-offset arm differs from the reference by 1.7e-7 to 8.6e-7 ST-RAE, inside the derived bounds of 5.9e-6 to 6.2e-6.

These are CYP3A4-only diagnostics on 3 of the 50 partitions. They are not O150, are not predictive of it, and select no setting.

### 5.8 Evaluation code (thin target adapter), written for review

The code follows the evaluation and adapter brief Miles approved on 2026-11-16, the brief Amendment 4 item 4 cites. It was written on his instruction to bring the code and test results for review. Location in this checkout: `qualification/actfound/d013_target_qualification_20261116/` (untracked). At the time of review it had not been run or uploaded (A2 entries 50–51); it was subsequently deployed to CoreHPC and run there (§5.9).

| File | Role | SHA-256 |
|---|---|---|
| `d013_evaluation_config_v1.json` | Frozen settings: 50 partitions × 4 endpoints × 3 seed slots; pIC50; complete support; comparator block | `c7d9638f…` |
| `run_evaluation.py` | Stage 1, pinned env: 200 unique adaptations via unchanged `run_task`; writes predictions, labels and bands, no score | `1f0a9408…` |
| `score_evaluation.py` | Stage 2, Python ≥ 3.10: per-endpoint ST-RAE from the frozen scorer, macro via the frozen `compute_validation_macro_st_rae`, 150 cell values → O150. Emits `comparison_is_final` false while the cited comparator artifact is absent | `bd334499…` |
| `test_evaluation.py` | 20 tests; directory suite 55 passed in `tripwire_cyp002` | `fdcd27eb…` |

Mutation checks: 14 run, 12 caught at once. The strict comparison rule's `<` vs `<=` gap was closed with a tie-boundary test. The remaining survivor is judged equivalent.

Defect fixed: a cell whose labels all lie inside their bands gives a zero RAE denominator. Stage 2 now refuses such a cell rather than folding NaN into O150.

Two suggestions are recorded and not acted on:
- a plausible-pIC50-range assertion in stage 1;
- `cyp/scripts/` as an alternative code location.

### 5.9 Full target evaluation (job 3083113) — complete

Reported by Miles on 2026-11-16 as completed work from another session. He directed that it be recorded as current project state, not as a request to rerun anything.

The evaluation scripts and the necessary artifacts were put onto CoreHPC and the full target evaluation ran as job 3083113: COMPLETED, exit `0:0`, 18 s batch elapsed; the adaptation loop reported 9.9 s. Earlier deployment/path and local dependency problems were superseded by this successful HPC run. No evaluation job is still running, and the Mac scoring environment does not need to be recreated.

Configuration as run: 200 unique endpoint adaptations = 50 partitions × 4 CYP endpoints, complete labelled support per endpoint, five nominal inner steps, pIC50 labels with no conversion. The 50 partition macros were replicated into three seed slots to populate the 150 O150 rows. **These are replicas, not three independent adaptation runs.**

| Quantity | Value |
|---|---|
| ActFound-CYP O150 | 0.8699544509650656 |
| Supplied T121 comparator | 0.701212576796 |
| Direction | lower is better |
| `favours_actfound_cyp` | **false** |
| `comparison_is_final` | **false** (preserved) |

The originally cited `C0_COMPARISON.json` was absent; scoring used the corroborating D009 artifact. The numerical comparison was therefore calculated, but the original comparator-provenance requirement (Amendment 4 item 6) remains unresolved and the comparison is not to be labelled final (§10).

Locations (CoreHPC project root `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project`):
- predictions: `.../qualification/actfound/d013_target_qualification_20261116/eval_predictions_20261116/cyp_adaptation_predictions.json`
- scores: `.../qualification/actfound/d013_target_qualification_20261116/eval_scores_20261116/cyp_o150_result.json`
- comparator used: `.../artifacts/d009_closure_workspace/p1_d009_o150_result_20261108.json`

Mean endpoint ST-RAE across the 50 unique partitions, from the saved predictions:

| Endpoint | Mean ST-RAE | Approx. mean within-cell Spearman | Approx. prediction SD | Approx. label SD |
|---|---:|---:|---:|---:|
| CYP1A2 | 0.93589602428044 | 0.323 | 0.350 | 1.015 |
| CYP2C9 | 0.8071864107051999 | 0.476 | 0.348 | 0.767 |
| CYP2D6 | 1.0087652883375327 | 0.199 | 0.272 | 0.915 |
| CYP3A4 | 0.7279700805370899 | 0.597 | 0.565 | 1.095 |

### 5.10 Saved-results audit (no retraining)

The audit used the existing saved predictions and the source code only; nothing was retrained.

Verified:
- the exact supplied frozen ST-RAE reference function reproduced all 200 endpoint scores with zero difference;
- aggregation and seed-slot replication reproduced O150 exactly;
- the supplied artifacts matched their recorded hashes;
- the inspected code preserved support/query membership and target prediction order; tasks start with fresh adaptation weights; updates do not modify the checkpoint head in place; evaluation restores batch-normalization running statistics.

No scoring-arithmetic defect and no cross-cell adaptation-carryover defect was found.

**Characterized behaviour.** The main behaviour is compressed predictions and weak ranking, worst on CYP2D6 (§5.9 table). Within held-out CYP2D6 cells, by actual-potency quartile: lowest quarter actual pIC50 3.747 vs predicted 4.725; highest quarter actual 5.833 vs predicted 4.865. The actual gap of about 2.085 log units becomes only 0.139 in predictions. Near-zero overall bias hides opposing errors at the extremes. A monotone calibration can change scale but cannot repair the weak ranking.

**Stability across splits.** Each labelled molecule appeared in validation five times. Median within-molecule prediction SD across those splits was approximately 0.014–0.020 pIC50 depending on endpoint, so the behaviour is consistent across splits rather than a few failed folds. These overlapping observations are not independent samples.

### 5.11 Training-parity discrepancy (verified omission, not a cause)

Upstream `external/ActFound/learning_system/system_actfound.py` adds `sum(lr_param ** 2) * args.lrloss_weight` from `epoch >= args.begin_lrloss_epoch`. The deployed `cyp/scripts/actfound_production_train_v2.py` carries `begin_lrloss_epoch=50` and `lrloss_weight=35.0` in its configuration but never adds that penalty to its training objective; no equivalent term was found in the called loss or optimizer. The selected epoch-51 checkpoint is after the point where the upstream penalty should have started.

The penalty discourages large learned inner-loop learning rates. **The omission is verified; its effect on accuracy is not.** It is not established that it caused the poor result, nor that restoring it would fix the model. Equally, this is not a clean verdict on a faithful upstream reproduction.

**Hypothesis (not established).** Native target adaptation updates only the final linear head, not the whole encoder; predictions use weighted support-label-anchored pairwise estimates over the most fingerprint-similar half of the support molecules. That averaging could contribute to the compression, but the necessary internal weights/differences were not saved, so the mechanism remains a hypothesis. Five nominal inner steps do not prove five applied updates, because the implementation can skip an update.

### 5.12 Data scope of this study

The ActFound study used the broad authors' curated ChEMBL32 corpus after the OpenADMET identity firewall. It did **not** use only the earlier 175-assay CYP auxiliary collection. The result therefore does not isolate architecture from pretraining-data scope. The corpus and firewall decisions already settled under D013 and its amendments remain settled.

### 5.13 Winning-model research (external review; no method adopted)

An external review of the challenge's public results and reports. Nothing here is an adopted method and nothing here is a result of this project.

- Official direct-inhibition results put whippoorwill/Inductive Bio and preheat-to-450 in the top tier, followed by NVIDIA and Recursion. Public method details were incomplete. Inductive describes consortium ADMET data and Beacon models, but not enough architecture/training detail to reconstruct its winner or identify the decisive component. No shared winning recipe is established.
- The verified eighth-place rasayan-labs team published a detailed self-report: multiple molecular representations including protonation/tautomer states and pretrained embeddings; boosted trees, kernel models and neural networks; out-of-fold ensembling; measurement-uncertainty weighting; use of single-concentration screening information; and explicit calibration to restore a compressed high-potency tail.
- Their own internal ablation attributes their largest gain to assay-aware learning that connected screening data to the prediction tasks. That is team-reported evidence on their study, not proof of what would help this project. Their calibration work matches our compression symptom but cannot establish a fix for our weak ranking.
- Their official blind-test results are not directly comparable to our internal O150. The organizers' reported blind CYP2D6 distribution shift does not explain our internal cross-validation result.

Sources: `https://openadmet.ghost.io/take-me-heme-binding-modes-wrapping-up-the-cyp-inhibition-challenge/`; `https://github.com/whis9/rasayan-cyp`; `https://www.inductive.bio/news/inductive-places-first-in-openadmet-cyp-inhibition-challenge`.

### 5.14 Session 005.5 (2026-11-17): staged ActFound/T121 comparison package — component tests only

Recorded from Miles's report and local verification on 2026-11-17. This section records existing work and the plan Miles stated (§6). It authorizes no run, and no launch readiness is established.

**Plan Miles stated (see §6 for his wording):** compare frozen T121 against updated T121 and ActFound. ActFound is to be started from a fresh seeded foundation initialization with the corrected learning-rate penalty (§5.11), then run as selected foundation, screening, pIC50 adaptation. T121 is screened for 50 epochs.

**Component tests (Miles's report; run on the Mac):**

| Suite | File | Result | Nature |
|---|---|---|---|
| ActFound stage controller | `cyp/tests/test_actfound_stage_controller.py` | 11 passed | mock executors |
| ActFound stage adapters | `cyp/tests/test_actfound_stage_adapters.py` | 8 passed | training and subprocesses mocked |

The package's `PACKAGE_SCOPE.txt` states: not a production launch bundle; no datasets, checkpoints, environments or launch config. These tests do not establish launch readiness. No resume support is established.

**Staged-integration archive** `/Users/mileshuang/Desktop/tripwire-project/actfound_staged_integration_20261117.tar.gz`, SHA-256 `79c4ace8b862d84b8b52be009a082f942ff7d694813e977ec1f2a1d8c33c793b` (uncommitted). Local check on 2026-11-17: the archive's `shasum` matched; the internal `SHA256SUMS.txt` verified with no failures; all 78 packaged files are byte-identical to the current working tree (no drift since packaging).

**CoreHPC transfer and tests (reported done by Miles, 2026-11-17; run by him on the CoreHPC login host `chpc-ucsf-login-vm1.corehpc.ucsf.edu` via the bastion ProxyJump; the output was reported in summary, not pasted):**

| Step | Reported result |
|---|---|
| Pre-transfer destination check | "as expected": nothing existed at the destination beforehand |
| Transfer | "as expected": archive copied to `/scratch/user/mhuang8/tripwire_staging/` |
| Archive hash on CoreHPC | "hash check okay" against `79c4ace8b862d84b8b52be009a082f942ff7d694813e977ec1f2a1d8c33c793b` |
| Extraction | into the new isolated directory `/scratch/user/mhuang8/tripwire_staging/actfound_staged_integration_20261117`; its output was not reported separately |
| Package inner `SHA256SUMS.txt` check (step 5) | "step 5 printed 0" (0 non-OK lines); this check could only run after extraction |
| Plumbing tests, 19 | "all tests pass", run with `/scratch/user/mhuang8/actfound/envs/actfound_c96f055/bin/python -I`: `cyp/tests/test_actfound_stage_controller.py` (11 tests) and `cyp/tests/test_actfound_stage_adapters.py` (8 tests) |

The "Ran N tests" lines were not pasted; 11 and 8 are the files' defined test counts, not pasted output. These are mock/component tests under the Python 3.9 environment. A pass does not establish launch readiness. The CoreHPC output itself is not in this checkout; the evidence is Miles's report.

**GPU correctness job 3103473 (result pasted by Miles in chat, 2026-11-17):** earlier recorded as pending with result UNKNOWN. The JSON Miles pasted carries no job id; it is the content for the intended result path `/scratch/user/mhuang8/tripwire_staging/actfound_stage_transfer_20261117/results/screen_validation_gpu_check/result.json` (CoreHPC). Miles has since run a read-only check on CoreHPC (2026-11-17, output pasted; see "Job record and hash check" below): the job is COMPLETED 0:0 and the result file's SHA-256 is recorded. The result content in the table below was pasted earlier; its bytes were not received in this checkout.

| Field | Reported value |
|---|---|
| status | PASS |
| scope | GPU fixed-episode validation and checkpoint correctness |
| optimizer_steps | 0 |
| model_initialization | random qualification model |
| validation_episodes | 8 |
| metric | native_log2fc_query_inverse_SE_squared_weighted_MSE |
| validation score | 0.9987984361996203 |
| per-enzyme | 1A2 0.41869687173151704; 2C9 0.3785024889483063; 2D6 0.12972015504120488; 3A4 3.0682742290774527 |

The numbers come from a random, untrained model; they are not performance claims. Checks reported PASS: `shared_state_modes_and_rng_restored` (true), `no_accumulated_parameter_gradients` (true), repeat-then-equal-enzyme aggregation, manual metric and tie rule, checkpoint-writer round trip, authorization gate. Reported not executed: the screening training loop, foundation pretraining, production staged execution.

**Hashes.** `runner_sha256` in the result is `9627ff17c2d0132dea2c8d34eae6cf38a5921479898e91a2eb1f465450534303`. Local check, 2026-11-17: it equals the current working-tree `cyp/scripts/actfound_production_train_v2.py` and that file's copy in the staged-integration archive. This corrects the earlier note that the runner in the stage_transfer archive was the job's runner: that archive's copy is `87d3766a…`, but the job's own result reports `9627ff17…`, so the job ran the current runner. The job result, not the archive, is the evidence. The discrepancy with the stage_transfer archive is unexplained; the staging folder now holds `9627ff17…` (see below), so it was evidently updated after that archive was made (inferred, not verified). `checkpoint_sha256` `095121d8f23ecb861316ff43796c0af1bf3004a3548de2d5326028a4ab5bbd5e` is the test checkpoint; it was not verified locally.

**Job record and hash check (read-only, run by Miles on CoreHPC, output pasted, 2026-11-17):**

| Item | Value |
|---|---|
| `sacct` job 3103473 | State COMPLETED, ExitCode 0:0 (also the `.batch` and `.extern` steps); End 2026-11-17T18:57:58 |
| `results/screen_validation_gpu_check/result.json` (CoreHPC staging folder `/scratch/user/mhuang8/tripwire_staging/actfound_stage_transfer_20261117`) | `4e7ae5520ce764bb2b3ed85fb9c50edede96afa0ce625acdc497f3c23521f0fd` |
| `cyp/scripts/actfound_model_side.py` (same staging folder) | `86ccf4b135a0388c147d2838e5d12ead2b5aaf5968f7140ec341e7950d3a62fd` |
| `cyp/scripts/actfound_production_train_v2.py` (same staging folder) | `9627ff17c2d0132dea2c8d34eae6cf38a5921479898e91a2eb1f465450534303` |

Local check (`shasum -a 256`, this checkout, 2026-11-17): current `cyp/scripts/actfound_model_side.py` is `86ccf4b1…` and current `cyp/scripts/actfound_production_train_v2.py` is `9627ff17…`; both full values are identical to the CoreHPC staging values. The result's own `runner_sha256` is also `9627ff17…`. The result file's bytes were not received locally; the `4e7ae552…` hash is for the CoreHPC file only.

Caveats on the binding:
- The staging hashes were taken after the job ended (18:57:58), so they show the folder's present state. Nothing records that the sources were unchanged during the job, except that the runner hash agrees with the job's own `runner_sha256`. For `actfound_model_side.py` there is no in-job record, so only the post-job hash binds it.
- The controller still checks only status, scope and `optimizer_steps`; it does not enforce this binding.

**Still open:** the wiring list, support weights at evaluation, and T121 connection to the controller (§10). The job 3103473 "read result and hash source" item is closed.

**Decision record.** The plan above is recorded as D015 in `notes/Decisions.md`. D015 authorizes no run.

---

## 6. Decisions and Approvals

All of the following are Miles's. The exact wording is in the archives.

| Date | Decision or approval | Scope and limits | Record |
|---|---|---|---|
| 11-14 | Import Session 005's recovered history | This session only; no backfill, no decision edits | A1 §2 |
| 11-15 | Option A: retrain on a firewalled corpus | Job 3066823 preserved | A1 §32 |
| 11-15 | Sub-choices: all 5,655 structures; authors' corpus; molecule-level removal; draft D013 | — | A1 §33 |
| 11-15 | D013 approved; implementation authorized | Training and evaluation still gated | D013; A1 §34 |
| 11-15 | Q1–Q9 implementation answers accepted; no live monitor agent; no Slurm alerts; overnight-failure rule withdrawn | Implementation detail | A1 §35 |
| 11-15 | Route C as the proposed repair path; split-aware stop-rule fix and read-only impact count | Amendment 1 stayed Proposed; no build, run, benchmark or training | A1 §38 |
| 11-15 | Widen the exclusion to the 180 largest-fragment matches, validation included | Identity rule unchanged | A1 §40 |
| 11-15 | Firewall-v2 plan and the Amendment 1 exclusion and acceptance wording | Amendment 1 as a whole still Proposed | A1 §41 |
| 11-15 | Amendment 1 approved for recording | No upload, build, run, benchmark or training | Amendment 1; A1 §43 |
| 11-15 | Fork session's work made canonical | — | A1 §44 |
| 11-15 | CoreHPC upload, pinned-env tests, reproduction run | Miles then ran CoreHPC jobs himself, for credit cost | A1 §45 |
| 11-15 | Record decisions and progress when they happen, not retrospectively | Working practice | A1 §46 |
| 11-15 | Investigate the `graph_pair_count` cause, then amend the gate scope; Amendment 2 approved | Gate scope only | Amendment 2; A1 §50 |
| 11-15 | Implement Amendment 2; Amendment 3 approved; implement Amendment 3 | Code changes; no run | Amendments 2–3; A1 §51, §54–§55 |
| 11-15 | Reproduction run 3075299: Miles submitted it in answer to the authorization question | That run | A2 entry 26 |
| 11-15 | Firewalled corpus build | Build only | A2 entry 28 |
| 11-15 | Build the acceptance check | Check only | A2 entry 32 |
| 11-15 | Adjudicate molregno 2336618 as not a protected structure | That one row's identity failure only | A2 entry 34 |
| 11-15 | Benchmark plan | Not training | A2 entry 38 |
| 11-16 | Production foundation training | Not evaluation | A2 entry 39 |
| 11-16 | Open items 1–4 choices; consolidated qualification brief and harness, including the checks' own ST-RAE diagnostics | Not the full evaluation or the thin adapter | A2 entries 42, 49 |
| 11-16 | Split adaptation from scoring | How scoring runs only | A2 entry 44 |
| 11-16 | Consolidated evaluation and thin-adapter brief approved (the brief Amendment 4 item 4 cites); then: bring the evaluation code and test results for review | The brief only; not running the evaluation | A2 entries 50, 51 |
| 11-16 | Record D013 Amendment 4 | Authorizes no run. The second-seed sweep was dropped, and the original comparator citation was kept | Amendment 4; A2 entry 48 |
| 11-16 | Miles deployed the evaluation to CoreHPC and ran the full target evaluation himself (job 3083113), then a saved-results audit and the winner research | Those runs and reviews only. He states that no retraining, tuning, architecture change or new experiment was approved by his audit or winner-research requests | Reported by Miles 2026-11-16 in another session; relayed into this session's conversation |
| 11-16 | Preserve `comparison_is_final=false`: the comparison was computed against the corroborating D009 artifact, and the original comparator-provenance requirement stays unresolved; it is not to be labelled final | Status of the comparison only | Same report; §5.9, §10 |
| 11-16 | Treat the reported completed work as current project state, not a request to rerun anything or a new implementation plan; settled decisions stay settled | Recording only; authorizes no run | Same report |
| 11-17 | Comparison: "Frozen T121 vs updated T121 and ActFound; preserve full O150, protected identities and existing scoring/comparator-provenance checks, including nonfinal comparison status if unresolved." | Plan only; runs nothing | Miles, 2026-11-17 conversation (Session 005.5); recorded in `notes/Decisions.md` D015 item 1 (originally written before D015 existed; "not yet in Decisions.md" was true then) |
| 11-17 | ActFound: "fresh original seeded foundation initialization with corrected penalty, then selected foundation → screening → pIC50. No historical penalty-omitting checkpoint or epoch-49 hunt." | Plan only; runs nothing | Same |
| 11-17 | ActFound screening: "80 epochs, existing schedule, eight fixed validation episodes, minimum native query WMSE checkpoint, no full-screen refit." | Plan only | Same |
| 11-17 | T121 screening: "50 epochs; trial-specific warmup 10 verified. Preserve approved validation/selection and transfer rules." | Plan only | Same |
| 11-17 | "Preserve exact approved uncertainty and harmonic pair weighting from qualified code and session evidence. Do not recreate formulas from defaults." | Plan only | Same |
| 11-17 | Execution: "I run execution commands. No training, model runs or SLURM submissions are authorized here. Runtime benchmarking is waived only for this CYP comparison, not correctness checks." "No commit/push." | The waiver covers runtime benchmarking only, for this comparison. It does not authorize any run | Same |
| 11-17 | Bookkeeping: update the session report and decisions log, recording existing decisions and work and keeping unresolved items open; preserve history | Does not authorize implementation edits, execution, commits or pushes | Miles, 2026-11-17 |
| 11-17 | Form of the decision record: "New D015" (chosen over D013 Amendment 5) | Form only. D015 records the six 2026-11-17 statements above as items 1-6 and authorizes no run. Whether it supersedes D013's foundation-training and checkpoint-freeze terms is not decided (open question in D015) | `notes/Decisions.md` D015; Miles, 2026-11-17 |

Standing operational instruction: run everything through the terminal. For CoreHPC, use the established SSH commands and wait for Miles to authenticate (password + Duo) (A1 §34).

---

## 7. Artifacts

| Artifact | Location | SHA-256 |
|---|---|---|
| Exclusion list v2 (517) | `qualification/actfound/d013_firewall_20261115/firewall_list_v2/` | `exclusion_molregnos.tsv` `b3f70e56…` |
| Amendment 3 builder and tests | `qualification/actfound/d013_firewall_20261115/` (committed; equal on CoreHPC) | `9c5d449b…`, `d2f6c8c3…` |
| PASS reproduction manifest | CoreHPC `.../chembl32/d013_reproduction_20261115c/` | `2e8a6d18…` |
| Firewalled corpus and manifest | CoreHPC `.../chembl32/d013_firewalled_20261115/` | corpus `8b00f349…`, manifest `c2963ad2…` |
| Acceptance check, adjudication file, runs | `qualification/actfound/d013_firewall_20261115/` (`acceptance_20261115/` FAIL, `acceptance_20261115b/` PASS; committed) | check `537b57e7…`, adjudication `bc4647a5…` |
| Config v2 | CoreHPC `.../tripwire-project/cyp/artifacts/actfound/d013_split_config_v2_20261115/` | `dad741d0…` |
| Benchmark helpers | `qualification/actfound/d013_benchmark_20261115/` (uncommitted) | in A2 entry 38 |
| Frozen foundation checkpoint and run outputs | CoreHPC `.../production_training_d013/actfound_chembl32_d013_v2/` | `model_best.pt` `4db29a04…` |
| Target-qualification harness and counts | `qualification/actfound/d013_target_qualification_20261116/` (uncommitted) | in A2 Current state |
| Evaluation code (for review) | `qualification/actfound/d013_target_qualification_20261116/` (uncommitted) | §5.8 |
| Full evaluation predictions (job 3083113) | CoreHPC `.../qualification/actfound/d013_target_qualification_20261116/eval_predictions_20261116/cyp_adaptation_predictions.json` | `c6fbb0b1…` (§7.1) |
| Full evaluation scores / O150 result | CoreHPC `.../qualification/actfound/d013_target_qualification_20261116/eval_scores_20261116/cyp_o150_result.json` | `dd1e43df…` (§7.1) |
| Comparator artifact actually used (corroborating D009 result) | CoreHPC `.../artifacts/d009_closure_workspace/p1_d009_o150_result_20261108.json`; local copy `artifacts/d009_closure_workspace/` | `b277f4cc…` (§7.1) |
| Target-qualification outputs | CoreHPC `/scratch/user/mhuang8/actfound/qualification/d013_target_20261116/`; local copy `~/Downloads/d013_checks/` | `adaptation_checks.json` `1be10d32…` |
| Frozen ST-RAE scorer (unchanged) | `cyp/scripts/st_rae.py`, `references/cyp_upstream/custom_scoring_functions.py` | `f533a6ef…`, `c0ee287c…` |

**2026-11-17 (Session 005.5) artifacts, all untracked/uncommitted. Hashes were not supplied for these, except the staged-integration archive, the job 3103473 result file and the two staging sources listed in the last rows, and the two modified scripts' current local hashes (§5.14); this list claims nothing about what any of them proves.**

| Artifact | Location in this checkout |
|---|---|
| Staged-integration archive (78 files; SHA-256 `79c4ace8…`, locally verified §5.14) | `actfound_staged_integration_20261117.tar.gz` |
| Other archives | `actfound_adapter_sources_`, `actfound_current_source_`, `actfound_harmonic_complete_`, `actfound_local_upstream_`, `actfound_minimal_architecture_`, `actfound_screen_loader_`, `actfound_screen_partition_`, `actfound_stage_transfer_`, `t121_screen_dependencies_`, `t121_weighting_primitives_` (each `…20261117.tar.gz`) |
| Patch | `actfound_harmonic_weights_patch.py` |
| New scripts | `cyp/scripts/actfound_stage_adapters.py`, `actfound_stage_controller.py`, `t121_screen_stage.py`, `t121_weighted_training.py` |
| Modified (uncommitted; `.bak` copies beside them) | `cyp/scripts/actfound_model_side.py`, `cyp/scripts/actfound_production_train_v2.py`. Note: §7.1 hashes for these two were matched on 2026-11-16; whether the working tree still equals the §7.1 values was not re-compared. On 2026-11-17 the current local hashes were taken: `actfound_model_side.py` `86ccf4b1…`, `actfound_production_train_v2.py` `9627ff17…`, both equal to the CoreHPC staging copies (§5.14) |
| New tests | under `cyp/tests/`: `test_actfound_stage_controller.py`, `test_actfound_stage_adapters.py`, `test_actfound_stage_transfer.py`, `test_actfound_lr_penalty.py`, `test_actfound_weighting.py`, `test_t121_screen_stage.py`, `test_t121_training_weights.py`, `check_actfound_screen_loader.py` |
| Qualification folders | `qualification/actfound/d014_lr_penalty_20261117/` (naming collision, §10), `qualification/actfound/screen_loader_20261117/`, `qualification/t121_executor_source_20261117/` |
| Job 3103473 result file (CoreHPC, not in this checkout) | `/scratch/user/mhuang8/tripwire_staging/actfound_stage_transfer_20261117/results/screen_validation_gpu_check/result.json`; SHA-256 `4e7ae5520ce764bb2b3ed85fb9c50edede96afa0ce625acdc497f3c23521f0fd` (§5.14) |
| `actfound_model_side.py` in the CoreHPC staging folder (same folder as above, `cyp/scripts/`) | SHA-256 `86ccf4b135a0388c147d2838e5d12ead2b5aaf5968f7140ec341e7950d3a62fd`; equals the current local file (§5.14) |

Further untracked 2026-11-17 items beyond the truncated `git status` snapshot, if any, were not enumerated.

Full 64-character hashes for the completed ActFound-CYP work are in §7.1. The wider inventory, including superseded versions, is in A2's Current state block and the A1 sections cited above. Files in this checkout are not evidence of what a CoreHPC job loaded or produced.

The following are preserved and not to be deleted:
- job 3066823's outputs;
- the failed reproduction outputs (3073155, 3075157);
- the FAIL acceptance run;
- the superseded converter-route folder `qualification/actfound/firewall_corpus_20261115/`, which Amendment 1 cites;
- the overlap audit `qualification/actfound/overlap_audit_20261115/`.

### 7.1 SHA-256 inventory for the completed ActFound-CYP work

Supplied by Miles on 2026-11-16 with the instruction to keep each hash paired with its filename. These identify existing artifacts; they are not new runs and not changed files, and the handoff authorizes no repository edit, experiment or retraining. Full 64-character values are below; truncated citations elsewhere in this report are unchanged.

"Local check" = `shasum -a 256` run in this checkout in this session (2026-11-16); that is a local verification, not Miles's claim. Four entries are CoreHPC-side and are recorded from his handoff only.

| File | Location checked | SHA-256 | Local check |
|---|---|---|---|
| `model_best.pt` (epoch 51) | CoreHPC `.../production_training_d013/actfound_chembl32_d013_v2/checkpoints/` | `4db29a04f69ecc1ef6ec2f058aa0335e132394a2a0bbe36c14bd27d54a49de2b` | not in this checkout; handoff only (agrees with §5.6) |
| `eval_predictions_20261116/cyp_adaptation_predictions.json` | CoreHPC (§5.9) | `c6fbb0b13890af91cb6f14fa5d9cda45346a2b35415b0040da9861fd98362271` | not in this checkout; handoff only |
| `eval_scores_20261116/cyp_o150_result.json` | CoreHPC (§5.9) | `dd1e43df022888b1f53af9b0840c9a8a6db1359efddb6b6dc77b0de1e614465c` | not in this checkout; handoff only |
| `actfound_chembl32_d013_config_v2.json` | CoreHPC `.../cyp/artifacts/actfound/d013_split_config_v2_20261115/` | `dad741d0b6a3e1f014d47579acec29e9bacdad3ac7b1fb7b0c8b2f69a1984af6` | not in this checkout; handoff only (agrees with §5.4) |
| `d013_evaluation_config_v1.json` | `qualification/actfound/d013_target_qualification_20261116/` | `c7d9638f2bb1673c10127a9b82513bf5168bc19d1c7014f70faec5dc518b97c4` | matched |
| `run_evaluation.py` | same directory | `1f0a9408e009db3b7c5cf69c42fb7e14b4ca347ea79d50f94ac752ecea9453cc` | matched |
| `score_evaluation.py` | same directory | `bd3344995c8713ca69d6b4ad3e9ab477596039f1eb09aab2fcce5ef970143361` | matched |
| `qualify_adaptation.py` (current reviewed version) | same directory | `a49d0cd2392e0b818d6ce3efc3ae7fdc999e2fa6091d06b611d88cef0e52dc9f` | matched; no earlier version is in the checkout |
| `count_cells.py` | same directory | `953d8d799ae4c0dc5589ef2c26209c47b06897b2b0be955e12e09b72edebf50b` | matched |
| `test_evaluation.py` | same directory | `fdcd27ebb76240984ddf1bafe8d47c535261e0b5570c51b67a19cb9a64775bf9` | matched |
| `d013_target_qualification_config_v1.json` | same directory | `d09239e269ba26b20832ef1f544335acbcfaa28d1d0d00537ccb7207939b6050` | matched |
| `strae_scores.json` (earlier qualification scoring output, §5.7) | same directory, `strae/` | `2240436d56d632cf9137f4ca608e5f18784cc2a9dc8e2ccadcce52bbc07c3303` | matched |
| `cell_counts.json` | same directory, `counts/` | `0ebad26f7d66dc020c3b1080da2a8ef0ddaaab3071ae593ce458ddf2a460e92f` | matched |
| `actfound_production_train_v2.py` | `cyp/scripts/` | `ed3e1055b1a9f5a76fc6b0ddad0df7e4a2df95254e513aca90e7359615db1865` | matched |
| `actfound_model_side.py` | `cyp/scripts/` | `97c1753512d102e1198bfad2ab4b601749cb1d57b6a1a3a1ad847be83214ee45` | matched |
| `st_rae.py` (frozen scorer) | `cyp/scripts/` | `f533a6ef1e279bee922e03c3d6a69bae9313d3d1cf14a448041221bc5b274edf` | matched |
| `cyp002_validation.py` | `cyp/scripts/` | `f0693b9294be0570988f08cc28256bf8ae995ea0cb82b421c45d40acb8bbc81e` | matched |
| `custom_scoring_functions.py` (frozen upstream scorer) | `references/cyp_upstream/` | `c0ee287c06c0301d4b3bfd74559b90eb1bd85db68b75b7d018ae92280c0e1dd3` | matched |
| `cyp-challenge-TRAIN_inhibition.csv` | `cyp/data/` | `b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278` | matched |
| `FREEZE_RECORD.json` | `cyp/artifacts/cyp_007/split_manifests/` | `9c6b292b8ca58ae83057cda157a338c0954b18b6f9253f040b0ca7344d6a2a85` | matched |
| `p1_d009_o150_result_20261108.json` (corroborating comparator) | `artifacts/d009_closure_workspace/` | `b277f4ccd9138b688ff20bfaf2378ebfcb66bc118ab5943df48027168adc0e2b` | matched |

All 17 locally present files matched the supplied values exactly, byte for byte.

**What these hashes do and do not prove** (Miles's stated source scope): the result, predictions, supplied evaluation scripts/config, tests and qualification scoring output were hashed from the original supplied bytes. The remaining input/comparator hashes are recorded in the completed run artifacts; the training config, runner, model-side code and frozen scorer sources were also checked against their supplied audit copies. The final result JSON hash was computed from that existing audit copy, not a regenerated score. Hashing `test_evaluation.py` does not imply its tests were independently run. The local check establishes byte identity in this checkout only; files in this checkout are not evidence of what a CoreHPC job loaded or produced.

The original cited `C0_COMPARISON.json` is still absent; the D009 comparator is corroborating only, so `comparison_is_final` remains false (§5.9, §10).

---

## 8. Corrections

Each correction keeps its original text in the archive.

- **`graph_pair_count` magnitude:** first recorded as "both directions, 1–2 pairs". It is 1–47 pairs and almost always upward (A1 §49 B annotation, §50 A).
- **Digit count in job 3075157's analysis:** only one of the two authors' values said to carry 16 digits does (A1 §53 C).
- **Activity 2091754:** first recorded as outside the split; it is inside (A2 entry 27).
- **§40 sentence:** corrected in place twice on Miles's approval. Its corrected wording was inferred because the fork session holding the original was deleted (A1 §40). This is what prompted the working-practice instruction in A1 §46.
- **Thin-adapter brief:** first recorded as not known to have preceded the evaluation code. It did; it is the brief Amendment 4 item 4 cites (A2 entry 51).
- **Evaluation run status:** the report previously stated that the evaluation code had not been run or uploaded and that the full evaluation was not authorized. That was true as recorded; Miles subsequently ran the full evaluation on CoreHPC as job 3083113 (reported 2026-11-16), so the status lines in §5.8, §9 and §11 were updated. The earlier deployment/path and local dependency problems are superseded by that successful HPC run.
- **Withdrawn overnight-failure rule:** its log entry was removed at Miles's request before either archive was made (A1 §35).

---

## 9. Current Status (as of 2026-11-17; items dated 2026-11-16 are historical unless re-stated)

- **Session 005.5 (2026-11-17), current:** Miles's settled comparison plan is in §6 (T121 vs updated T121 and ActFound; fresh corrected-penalty ActFound foundation initialization; ActFound 80-epoch and T121 50-epoch screening). Staged package built and archived (`79c4ace8…`, locally verified, uncommitted); 11 controller tests and 8 adapter tests passed on the Mac as component/mock tests (§5.14). Transfer, archive hash check (`79c4ace8…` okay), extraction, package `SHA256SUMS.txt` check (0 non-OK lines) and the 19 plumbing tests on CoreHPC (ActFound env, `-I`) are reported done by Miles on 2026-11-17 (summary report, no pasted output; counts are the files' defined counts); they are mock/component tests under the Python 3.9 environment and do not establish launch readiness (§5.14). GPU correctness job 3103473: COMPLETED 0:0 (`sacct`, End 2026-11-17T18:57:58); result PASS (random-initialized model, 0 optimizer steps; not a performance claim); result file hash `4e7ae552…` recorded; model-side (`86ccf4b1…`) and runner (`9627ff17…`) hashes in the CoreHPC staging folder equal the current local files, taken after the job ended (the runner also matches the result's `runner_sha256`; model-side has no in-job record); the controller does not enforce this binding (§5.14). The plan is recorded as D015 in `notes/Decisions.md` (form chosen by Miles: "New D015"; authorizes no run). Launch readiness not established; no resume support established. **No training, model run or SLURM submission is authorized; no commit or push.**

- **Firewalled foundation corpus:** built, acceptance PASSED.
- **Foundation checkpoint:** retrained and frozen (epoch 51, `4db29a04…`).
- **Target-adaptation qualification:** complete. Runtime, memory, all three correctness checks and their stage-2 ST-RAE bound checks all passed.
- **D013 open items 1–4:** settled and recorded in Amendment 4.
- **Evaluation code (thin target adapter):** reviewed, then deployed to CoreHPC and run. The earlier status line "not run, not uploaded" and the earlier "Not authorized: running the full evaluation" were true before Miles ran it; both are superseded by job 3083113 (§5.9). No new authorization beyond that run is implied.
- **Full target evaluation:** complete (job 3083113, exit `0:0`). **ActFound-CYP O150 = 0.8699544509650656 vs supplied T121 = 0.701212576796, lower is better, `favours_actfound_cyp=false`.** This implementation produced weak held-out CYP predictions and did not beat the supplied T121 value.
- **`comparison_is_final=false`,** preserved: the comparison was computed against the corroborating D009 artifact because the cited `C0_COMPARISON.json` was absent, so the comparator-provenance requirement is still open (§10).
- **Saved-results audit:** complete. Scoring, aggregation and seed-slot replication reproduce exactly; no scoring-arithmetic or adaptation-carryover defect found. Characterized behaviour: compressed predictions and weak ranking, worst on CYP2D6, consistent across splits (§5.10).
- **Training parity:** the upstream inner-loop learning-rate penalty is verifiably absent from the deployed trainer's objective while its configuration carries `begin_lrloss_epoch=50`/`lrloss_weight=35.0` (§5.11). No established causal explanation for the weak result and no demonstrated fix.
- **Data scope:** the study used the broad firewalled ChEMBL32 corpus, not the 175-assay CYP auxiliary collection, so architecture is not isolated from pretraining-data scope (§5.12).
- **Winner research:** external review done; no method adopted and no decisive component identified (§5.13).
- **Artifact hashes:** the SHA-256 inventory for the completed ActFound-CYP work is recorded in §7.1. 17 of 21 files matched by a local check in this checkout; the four CoreHPC-side entries (checkpoint, predictions, scores, config v2) rest on Miles's handoff.
- **No retraining, tuning, architecture change or new experiment is approved.** *(2026-11-16 line; superseded on 2026-11-17 in respect of the plan only: Miles has settled a fresh corrected-penalty ActFound foundation initialization and a T121-vs-updated-T121-and-ActFound comparison as the decided plan (§6). That plan authorizes no execution: no training, model run or SLURM submission is authorized in this session. The earlier line's history is preserved.)*
- **Repository (per `git status`, 2026-11-16):**
  - D013 Amendment 4 in `notes/Decisions.md`: uncommitted.
  - Benchmark-helper and target-qualification folders, including the evaluation code: untracked.
  - Whether and how the harness was uploaded to CoreHPC before the qualification runs: not recorded, although it was run there. The evaluation itself was deployed and run on CoreHPC (§5.9).
  - The restructuring into this thematic report, and the matching update to the bookkeeper and `CLAUDE.md` rules, were applied on 2026-11-16. Before the swap, the prior live log was archived byte-identical as A2 (`57cca182…`, verified); A1 (`b70517b8…`) is unchanged. Both archives and the rule changes are uncommitted.
- **Working setup: model routing (2026-11-17; tooling only, no scientific change).** Project-scoped and uncommitted:
  - `.claude/agents/bookkeeper.md` gains `model: sonnet`;
  - new read-only `.claude/agents/lookup.md` (`model: haiku`; Read, Grep and Glob only) for narrow lookups and summaries of existing logs;
  - new `.claude/settings.json` sets `CLAUDE_CODE_SUBAGENT_MODEL=sonnet` for subagents with no explicit model;
  - `CLAUDE.md` gains short routing and delegation rules (Opus only on request, with the reason explained first).

  Agent roles, tools, permissions, hooks and the Bedrock setup are unchanged. Checked: the frontmatter of both agents parses, the settings JSON is valid, and installed Claude Code 2.1.325 recognizes `CLAUDE_CODE_SUBAGENT_MODEL`. Not yet confirmed:
  - that the new models are used, which needs a new session;
  - that the `sonnet`/`haiku` aliases resolve to models enabled in the Bedrock account;
  - the main-session switch to Sonnet, which Miles makes in the model picker.

---

## 10. Open Items and Caveats

- **Comparator artifact (open):** `cyp/artifacts/cyp_007/finalist_completion_v1/C0_COMPARISON.json` (`48bd619e…`) is not in this checkout and was absent when the evaluation ran. Amendment 4 item 6 requires a targeted check to locate it or account for it before the final comparison. The D009 O150 result file carries the same value but only corroborates it. The O150 comparison has now been computed against that corroborating artifact, so the number exists but `comparison_is_final` stays false (§5.9).
- **No causal explanation:** the verified training-parity omission (§5.11) is not established as the cause of the weak result, and no fix is demonstrated. The head-only-adaptation/pairwise-averaging compression mechanism is a hypothesis; the internal weights needed to test it were not saved.
- **Replica seed slots:** the 150 O150 rows come from 50 unique partition macros replicated into three seed slots, so they carry no seed-variance information.
- **Non-independent observations:** each labelled molecule appears in validation five times, so the per-molecule stability figures in §5.10 are not independent samples.
- **Adjudication record:** whether to also record the molregno 2336618 adjudication in `notes/Decisions.md` was raised and not decided (A2 entry 34).
- **Pre-submission checks:** for jobs 3076199 and 3076848, only runner v2 and the model side are confirmed by later hashing. The other check outputs were not recorded (A2 entries 37, 39, 41).
- **`graph_pair_count` cause:** attributed by elimination; similarity was not measured. Upstream of the pregraph `43dfd672…`, the sort step from the projection output has no recorded summary (A1 §29). The reproduction gate compares the output against the authors' file instead.
- **Who submits later CoreHPC jobs:** not stated. Miles has submitted all of them so far. (2026-11-17: Miles states "I run execution commands.")
- **Session 005.5 wiring still to inspect (open, none done; Miles's list):** pIC50 support weights reaching adaptation; normalization/reset semantics; qualification evidence bound to deployed hashes; real adapter/config/input binding; T121 production/O150 routing and epoch-end checkpoint behavior. Component/mock tests do not establish launch readiness. No resume support established.
- **Transfer and CoreHPC plumbing tests (reported done 2026-11-17, §5.14):** reported by Miles in summary only; the "Ran N tests" lines were not pasted. They are mock/component tests under the Python 3.9 environment and a pass does not establish launch readiness. Still open alongside them: the wiring list above, support weights at evaluation, and T121 connection to the controller (the job 3103473 read-and-hash item is closed, §5.14).
- **Job 3103473 (closed item):** COMPLETED 0:0; result PASS; result file hash and the staging model-side and runner hashes recorded 2026-11-17 (§5.14). The "read result and hash source" item is closed. Earlier lines "final result UNKNOWN" and "file not read / hash not supplied" are superseded.
- **Decisions log (updated 2026-11-17):** the settled 2026-11-17 statements (§6) are now recorded as D015 in `notes/Decisions.md` (appended after the D014 records, in numeric order); Miles chose "New D015" as the form. The earlier "lag / returned for approval" status is superseded. **Open question for Miles:** D015 does not state that it supersedes D013's "retrain once; checkpoint freeze" terms. The epoch-51 checkpoint `4db29a04…` and job 3076848 are preserved unchanged, and whether D015 item 2 supersedes those D013 terms for this comparison is not decided.
- **Observation from code inspection (not an established defect; nothing edited or run):** `run_task` in `cyp/scripts/actfound_production_train_v2.py` (about line 574) builds observation weights only when `is_training_phase=True`. Screening validation (about line 1363 of the same file) and the pIC50 evaluator (`qualification/actfound/d013_target_qualification_20261116/run_evaluation.py`, about line 155, legacy 3-field arrays) call it with `is_training_phase=False`, so support-loss adaptation at evaluation would be unweighted. Whether that matches the approved weighting rule is not recorded.
- **Observation from code inspection (not an established defect):** the stage controller and adapters cover only ActFound foundation → screening → pIC50. The T121 production, O150 routing and epoch-end checkpoint wiring is not connected to them (matches the last item of the wiring list above).
- **Job 3103473 binding (partly addressed):** model-side and runner hashes now bound to the current local files (§5.14), but the staging hashes were taken after the job ended; only the runner hash is corroborated in-job (`runner_sha256`), and model-side has no in-job record. The result records no other source hashes, and the controller checks only status, scope and `optimizer_steps`, so it does not enforce the binding. The earlier note that the stage_transfer archive's runner copy was `87d3766a…` stays unexplained (staging folder evidently updated after the archive; inferred, not verified).
- **Naming collision:** the folder `qualification/actfound/d014_lr_penalty_20261117/` uses "d014", but D014 in `notes/Decisions.md` is the PXR holdout decision. The ActFound penalty work is not PXR D014 and has no D-number. Nothing renamed.
- **Comparator provenance:** `comparison_is_final` stays false if unresolved (Miles's 2026-11-17 comparison decision preserves this).
- **Hashes for the 2026-11-17 artifacts (corrected at close, 2026-11-17):** the earlier line "hashes not supplied for the 2026-11-17 artifacts other than the staged-integration archive" is stale. Hashes were later recorded for the staged-integration archive (`79c4ace8…`), the job 3103473 result file (`4e7ae552…`), the CoreHPC staging `actfound_model_side.py` (`86ccf4b1…`) and `actfound_production_train_v2.py` (`9627ff17…`), and the two modified scripts' current local hashes (§5.14, §7). Hashes were still not supplied for the other 2026-11-17 archives, patch, new scripts, new tests and qualification folders listed in §7.

---

## 11. Next Steps

1. Locate or account for the comparator artifact before the comparison can be called final (Amendment 4 item 6); until then `comparison_is_final` stays false.
2. (Done 2026-11-16) The job 3083113 predictions and score hashes and the comparator hash are recorded in §7.1, with 17 of the 21 inventory files confirmed by a local check in this checkout.

3. (Session 005.5, 2026-11-17; reported done by Miles, §5.14) Transfer of the staged-integration archive to `/scratch/user/mhuang8/tripwire_staging/`, hash verification, extraction to `actfound_staged_integration_20261117`, package checksum check and the 19 plumbing tests (11 controller + 8 adapter) in the ActFound env. (Done 2026-11-17, §5.14) Job 3103473 COMPLETED 0:0; its result file and the staging model-side and runner sources were hashed on CoreHPC, and the model-side and runner hashes equal the current local files.
4. Complete the wiring inspection listed in §10 before any launch is considered.

Session closed 2026-11-17; remaining steps are carried into Session 006 (§12). Section 12 follows the training-parity note below.

Training-parity omission (§5.11): Miles has since settled the plan to restart ActFound from a fresh corrected-penalty initialization (§6, 2026-11-17). *(Earlier line, 2026-11-16: "Open questions that follow from the findings but that Miles has not directed any work on: what to do about the verified training-parity omission. No retraining, tuning, architecture change or new experiment is authorized." Superseded in respect of the plan only; see §9. No training, model run or SLURM submission is authorized.)*

---

## 12. Carried into Session 006 (unresolved at close, 2026-11-17)

Session 005 closed on Miles's declaration of 2026-11-17. None of the following was resolved in this session; each is carried forward as it stands in §5.14, §9 and §10. This list adds no Session 006 content and no decision.

1. **Wiring list (Miles's list, none done):** pIC50 support weights reaching adaptation; normalization/reset semantics; qualification evidence bound to deployed hashes; real adapter/config/input binding; T121 production/O150 routing and epoch-end checkpoint behavior. Component/mock tests do not establish launch readiness; no resume support established.
2. **Support weights at evaluation:** `run_task` builds observation weights only when `is_training_phase=True`; screening validation and the pIC50 evaluator call it with `False`. Whether that matches the approved weighting rule is not recorded (§10).
3. **T121 connection to the controller:** the stage controller and adapters cover only ActFound foundation → screening → pIC50; T121 is not connected (§10).
4. **Comparator provenance:** `C0_COMPARISON.json` still absent; the O150 comparison (ActFound-CYP 0.8699544509650656 vs T121 0.701212576796) was computed against the corroborating D009 artifact, so `comparison_is_final=false` (§5.9, §10).
5. **Open question on D015:** whether D015 supersedes D013's "retrain once; checkpoint freeze" terms is not decided; epoch-51 checkpoint `4db29a04…` and job 3076848 are preserved unchanged (§10, `notes/Decisions.md` D015).
6. **Naming note:** `qualification/actfound/d014_lr_penalty_20261117/` uses "d014", but D014 is the PXR holdout decision; the ActFound penalty work has no D-number; nothing renamed (§10).
7. **Job 3103473 binding:** the controller does not enforce the hash binding; model-side has no in-job hash record (§5.14, §10).

Standing at close: launch readiness is not established. No training, model run or SLURM submission is authorized. Corrected at close: the §10 "hashes not supplied" line (see §10).
