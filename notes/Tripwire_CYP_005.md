# Tripwire-CYP-005 Session Log

**Dates:** 2026-11-14 (recovered history imported; start date UNKNOWN) to 2026-11-17. The 2026-11-17 work, which Miles calls "Session 005.5", is a continuation recorded here by topic.  
**Status:** Closed on Miles's declaration, 2026-11-17.  
**Thread:** Tripwire CYP, ActFound-CYP whole-package study  
**Governing decisions:** D013 and Amendments 1–4 (Approved); D015 (2026-11-17 comparison plan; form approved, authorizes no run)

**Execution environments:** CoreHPC project `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project`; pinned ActFound env `/scratch/user/mhuang8/actfound/envs/actfound_c96f055` (Python 3.9.18, torch 1.12.1, RDKit 2023.03.1). Local frozen identity env `tripwire_cyp002` (Python 3.12.14, RDKit 2026.03.6); scoring in a Python ≥ 3.10 env.

**Related records:**

- Supporting record: `notes/Tripwire_CYP_005_supporting.md` (S1–S5: dated approvals, artifact and SHA-256 inventories, 2026-11-17 transfer and job checks, correction history).
- Archives, never edited: A1 `notes/archive/Tripwire_CYP_005_archive_20261115.md` (chronological log, §1–§55; `notes/Decisions.md` "Session 005 §N" means A1 §N); A2 `..._20261116.md` (entry log, entries 1–51; "Session 005 entry N" means A2 entry N); A3 `..._20261117.md` (the previous version of this report, SHA-256 `c2c9e5ed…`). Other records that cite "Session 005 §N" of the live report, such as `notes/Tripwire_CYP_006.md`, refer to A3's numbering.

## 1. Session purpose

Session 005 began by reconstructing ActFound and training it once on a ChEMBL32 corpus regenerated from the SQLite release (job 3066823). An identity audit showed that corpus was not firewalled against OpenADMET structures. The session's purpose became the D013 study:

1. build a ChEMBL32 foundation corpus firewalled against all OpenADMET structures;
2. retrain the ActFound foundation model on it once and freeze the checkpoint;
3. qualify target adaptation on the frozen D007 cells;
4. evaluate with the unchanged 150-cell comparison against frozen T121;
5. (2026-11-17) stage a package for a further ActFound/T121 comparison under D015.

Only whole-package claims are made. The comparator is O150(T121) = `0.701212576796` (Amendment 4 item 6), lower is better, scored by the unchanged frozen upstream ST-RAE implementation.

## 2. Starting point: job 3066823 and the leakage finding

Job 3066823 completed `0:0` in 12:16:08: 60 epochs, best epoch 58, validation score 0.9770, with checkpoint selection and reported metrics on the same 500 validation assays and no held-out evaluation. The corpus had no identity firewall (`openadmet_firewall_applied: false`). The full-corpus audit found OpenADMET molecules in loader-kept training rows:

| Protected tier | Molecules | Training rows |
|---|---:|---:|
| D007 development | 203 | 2,139 |
| internal test | 44 | 444 |
| blinded challenge | 7 | 25 |

Miles chose Option A, retraining on a firewalled corpus (A1 §32). Job 3066823's outputs are preserved unchanged and are not the study checkpoint.

## 3. D013 and its amendments

D013 (approved 2026-11-15) fixes the study: firewall on all 5,655 OpenADMET structures by full Standard InChIKey; base corpus the authors' released `chembl_processed_chembl32.csv`; molecule-level removal, with assays below 20 usable molecules dropped and logged; retrain once; checkpoint freeze rule; adaptation on each D007 cell's training molecules; 150-cell comparison against T121. It authorizes no run by itself.

| Amendment | Approved | Effect |
|---|---|---|
| 1 | 2026-11-15 | Route-C graph re-qualification behind an exact reproduction gate; conservative pretraining exclusion of largest-fragment full-key matches (train and validation); corpus identity acceptance check |
| 2 | 2026-11-15 | Gate comparison scope: `record_id_unused` and `graph_pair_count` non-comparable but reported (item 2 later replaced) |
| 3 | 2026-11-15 | `standard_value` compared within relative tolerance 1e-14, effect reported in the manifest |
| 4 | 2026-11-16 | Settles D013 open items 1–4: O150 from 200 unique cell-endpoint adaptations, each replicated into three seed slots; pIC50, no conversion; complete support, no cap; thin target adapter per the approved brief; adaptation and scoring in separate processes; comparator citation kept |

Miles also said not to reopen the architecture or adaptation-path menu, not to add a published-fidelity pass/fail gate, and not to reselect epochs through extra validation draws. The dated approvals and their archive locations are in S1.

## 4. Firewalled corpus

**Exclusion list.** Identities were computed in the frozen identity env and matched against the 5,655 protected structures. Firewall list v1 (337 molregnos, full Standard InChIKey) plus 180 largest-fragment full-key matches gives exclusion list **v2, 517 molregnos**. First-block near-matches are reported only. The identity rule remains the full Standard InChIKey; the largest-fragment exclusion is protective, not an identity claim.

**Route C and reproduction gate.** Upstream graph eligibility depends on pre-graph assay size, so Route C takes the authors' assay universe, rebuilds the pre-graph rows (pregraph `43dfd672…`), applies the firewall and reruns the unchanged graph stage. Firewall mode may run only after a reproduction-mode run reproduces the authors' file inside the pinned upstream train/valid split.

| Job | Comparison scope | Gate | Blocking in-split differences |
|---|---|---|---|
| 3073155 | exact | FAIL | 1,525,375 rows |
| 3075157 | Amendment 2 | FAIL | 4 `standard_value` rows |
| 3075299 | Amendment 3 | **PASS** | 0 |

Diagnosis of all 1,959,398 compared rows: no row differs in any field that defines training data. `record_id_unused` is an unconsumed provenance column. `standard_value` differs by at most 4.35e-15 relative (text precision); 8,657 authors' values exceed 15 significant digits, so Amendment 2's premise was false and Amendment 3 replaced it. `graph_pair_count` differs in 69 assays by 1–47 pairs, almost always upward; it is metadata only. Exact-equality pair exclusion was ruled out; Morgan-fingerprint similarity under a different RDKit build is the remaining explanation, reached by elimination and not measured. Byte-identity with the authors' file is not achievable from the pinned env.

PASS manifest (job 3075299): CoreHPC `.../chembl32/d013_reproduction_20261115c/corpus_manifest.json`, SHA-256 `2e8a6d184380a3c74796b884db8d30ffb5792f12be908385cadaa597599b9306`.

**Build (job 3075410, `0:0`, 00:11:42).**

| Quantity | Reproduction (3075299) | Firewalled (3075410) |
|---|---:|---:|
| Pre-graph rows kept | 2,027,275 | 2,018,649 |
| Rows removed by firewall | 0 | 8,626 (7,870 full-key, 756 largest-fragment) |
| Graph-qualified assays | 47,324 | 47,187 |
| Assays rejected by valid-count gate | 0 | 137 |
| Final rows | 1,959,399 | 1,946,000 |

The corpus is `actfound_chembl32_d013_firewalled.tsv`, SHA-256 `8b00f349726e7cc0d84d25195c3bdf09f76089159a93eed58401932ad0cbdecd`. The 137 rejected assays (77 train, 1 valid, 2 test, 1 test_external, 56 in no split) fell below 20 valid molecules after removal also removed graph neighbours; these are the ineligible assays D013 says to log. The row loss of 13,399 is 9,221 rows in dropped assays plus 4,178 inside 2,459 kept assays. Per-assay valid-molecule counts were not measured.

**Acceptance check.** Identities are recomputed from the built corpus in the frozen env; the check fails closed. The first run failed on one train row, molregno 2336618, a hexafluorophosphate salt whose SMILES the frozen RDKit rejects. Neither its ChEMBL standard InChIKey nor its organic-cation key is a protected key or first-block match, and Miles adjudicated it as not a protected structure (hash-pinned input `acceptance_adjudications_v1.tsv`, `bc4647a5…`). The rerun PASSED over all 1,946,000 rows (846,299 distinct SMILES): zero full-key and zero largest-fragment matches, no exclusion-list molregno present. One row therefore passes by adjudication, not by a recomputed identity. Whether to record that adjudication in `notes/Decisions.md` was raised and not decided.

## 5. Foundation training (job 3076848)

Split reconciliation (job 3076199): train 33,137 of 33,296 upstream IDs retained, validation 498 of 500; config v2 `dad741d0b6a3e1f014d47579acec29e9bacdad3ac7b1fb7b0c8b2f69a1984af6`. The loader drops a further 82 train and 1 validation assays under its gates (percent units, `=` relation, 20–10,000 rows, RDKit parse); per-assay reasons were not measured.

Interactive benchmark (job 3076227, one L40S): load 506.5 s, epoch 0 932.7 s, max GPU memory 1,747 MiB; the epoch-0 lower-bound estimate was 15.7 h and the 72 h request was adequate.

| | 3066823 (historical, unfirewalled) | 3076848 (D013) |
|---|---|---|
| Corpus | `24d9fd29…` | `8b00f349…` |
| Validation assays | 500 | 498 |
| Elapsed | 12:16:08 | 12:17:02 |
| Best epoch / score | 58 / 0.9770 | 51 / 0.98231 |

The runs are not like-for-like (different corpus and validation split). Integrity checks passed: 60 contiguous epochs, no non-finite losses, best epoch equals `completion.json`, `model_best.pt` equals `model_51.pt`, startup source hashes equal the deployed files, and runner v2 and the model side equal the reviewed hashes.

**Frozen foundation checkpoint:** CoreHPC `.../production_training_d013/actfound_chembl32_d013_v2/checkpoints/model_best.pt`, epoch 51, SHA-256 `4db29a04f69ecc1ef6ec2f058aa0335e132394a2a0bbe36c14bd27d54a49de2b`, selected on the firewalled upstream `valid` split before any CYP data was used.

## 6. Target-adaptation qualification

Cell counts come from the frozen D007 manifests (50 partitions × 4 endpoints); no protected internal-test or blinded molecule is in any cell.

| Endpoint | Support per cell | Target per cell |
|---|---|---|
| CYP1A2 | 1,062–1,101 | 100–139 |
| CYP2C9 | 965–1,008 | 85–128 |
| CYP2D6 | 1,123–1,157 | 112–146 |
| CYP3A4 | 1,766–1,815 | 171–220 |

**Stage 1 (CoreHPC, interactive L40S, three fixed CYP3A4 partitions).** All correctness checks passed: repeat across two seed values max |Δpred| 0.0; scrambled target labels 0.0; pIC50 vs pretraining scale 1.19–1.34e-5 against a 1e-3 threshold. Steady state is about 0.34 s per cell, mostly CPU featurization; peak GPU 443.9 MiB, host RSS about 2.7 GiB. The projection for all 200 adaptations is 68–228 s (cells independent and sequential; scaling with support size not characterized).

**Stage 2 (local scoring, Python 3.12).** `all_bounds_hold` is true on `adaptation_checks.json` `1be10d32…`; output `strae_scores.json` `2240436d…`.

| Partition | ST-RAE, pIC50 (both seeds and scrambled targets identical) | Pretraining-scale arm, corrected | Uncorrected (expected ~15) |
|---|---:|---:|---:|
| `rep_01_fold_01` | 0.72253 | 0.72253 | 15.26 |
| `rep_03_fold_05` | 0.72195 | 0.72195 | 15.22 |
| `rep_05_fold_10` | 0.77181 | 0.77181 | 15.98 |

The corrected arm differs from the reference by 1.7e-7 to 8.6e-7 ST-RAE, inside derived bounds of 5.9e-6 to 6.2e-6. These are CYP3A4-only diagnostics on 3 of 50 partitions; they are not O150 and select no setting.

**Evaluation code (thin target adapter).** Written to the brief Miles approved on 2026-11-16 (the brief Amendment 4 item 4 cites) in `qualification/actfound/d013_target_qualification_20261116/`: a frozen config, `run_evaluation.py` (stage 1, 200 adaptations via unchanged `run_task`), `score_evaluation.py` (stage 2, per-endpoint ST-RAE, macro, 150 cell values to O150) and `test_evaluation.py` (20 tests; directory suite 55 passed). Mutation checks: 14 run, 12 caught at once; the `<` vs `<=` gap was closed with a tie-boundary test and the remaining survivor is judged equivalent. A cell whose labels all lie inside their bands gives a zero RAE denominator; stage 2 now refuses it rather than folding NaN into O150. Hashes are in S3.

## 7. Full target evaluation (job 3083113) and audit

Miles ran the evaluation on CoreHPC and reported it on 2026-11-16 as completed work, to be recorded as current state and not as a request to rerun anything. Job 3083113 COMPLETED `0:0` (18 s batch; adaptation loop 9.9 s). Configuration: 200 unique endpoint adaptations, complete labelled support, five nominal inner steps, pIC50 labels with no conversion. The 50 partition macros were replicated into three seed slots to populate 150 O150 rows; **these are replicas, not three independent runs**, and carry no seed-variance information.

| Quantity | Value |
|---|---|
| ActFound-CYP O150 | `0.8699544509650656` |
| Supplied T121 comparator | `0.701212576796` |
| Direction | lower is better |
| `favours_actfound_cyp` | **false** |
| `comparison_is_final` | **false** |

The cited `C0_COMPARISON.json` (`cyp/artifacts/cyp_007/finalist_completion_v1/`, `48bd619e…`) was absent, so scoring used the corroborating D009 artifact `p1_d009_o150_result_20261108.json` (`b277f4cc…`). The number exists, but the comparator-provenance requirement of Amendment 4 item 6 is unresolved and the comparison is not to be labelled final. Predictions and scores are on CoreHPC (`.../eval_predictions_20261116/cyp_adaptation_predictions.json` `c6fbb0b1…`; `.../eval_scores_20261116/cyp_o150_result.json` `dd1e43df…`).

| Endpoint | Mean ST-RAE (50 partitions) | Approx. within-cell Spearman | Approx. prediction SD | Approx. label SD |
|---|---:|---:|---:|---:|
| CYP1A2 | 0.93589602428044 | 0.323 | 0.350 | 1.015 |
| CYP2C9 | 0.8071864107051999 | 0.476 | 0.348 | 0.767 |
| CYP2D6 | 1.0087652883375327 | 0.199 | 0.272 | 0.915 |
| CYP3A4 | 0.7279700805370899 | 0.597 | 0.565 | 1.095 |

**Saved-results audit (no retraining).** The frozen ST-RAE reference function reproduced all 200 endpoint scores with zero difference, aggregation and seed-slot replication reproduced O150 exactly, supplied artifacts matched their hashes, and the inspected code preserved support/query membership and prediction order, started each task with fresh adaptation weights, left the checkpoint head unmodified in place and restored batch-norm statistics. No scoring-arithmetic or cross-cell carryover defect was found.

The characteristic behaviour is compressed predictions and weak ranking, worst on CYP2D6. In held-out CYP2D6 cells, the lowest actual-potency quartile has actual pIC50 3.747 vs predicted 4.725 and the highest has 5.833 vs 4.865: an actual gap of about 2.085 log units becomes 0.139 in predictions. Near-zero overall bias hides opposing errors at the extremes, and a monotone calibration can change scale but cannot repair weak ranking. Median within-molecule prediction SD across the five validation appearances is about 0.014–0.020 pIC50, so the behaviour is consistent across splits rather than a few failed folds (these observations are not independent samples).

**Training-parity omission (verified; not established as a cause).** Upstream `system_actfound.py` adds `sum(lr_param ** 2) * lrloss_weight` from `begin_lrloss_epoch`. The deployed `cyp/scripts/actfound_production_train_v2.py` carries `begin_lrloss_epoch=50` and `lrloss_weight=35.0` but never adds the penalty to its objective, and the epoch-51 checkpoint is past the point where it should have started. The omission is verified; its effect on accuracy is not, and this is not a clean verdict on a faithful upstream reproduction. A hypothesis, not established: head-only adaptation with support-label-anchored pairwise averaging over the most fingerprint-similar half of the support contributes to the compression; the internal weights needed to test it were not saved.

**Data scope.** The study used the broad firewalled ChEMBL32 corpus, not the earlier 175-assay CYP auxiliary collection, so it does not isolate architecture from pretraining-data scope. D013's settled corpus and firewall decisions remain settled.

**External review of the challenge winners (nothing adopted).** Public details for the top teams (whippoorwill/Inductive Bio, preheat-to-450, NVIDIA, Recursion) were incomplete and no shared winning recipe is established. The eighth-place rasayan-labs team self-reports multiple representations, tree/kernel/neural ensembles, out-of-fold ensembling, uncertainty weighting, use of single-concentration screening data and explicit calibration of a compressed high-potency tail; its internal ablation attributes the largest gain to assay-aware learning connecting screening data to the tasks. That is team-reported evidence, not proof for this project, and its blind-test results are not comparable to our internal O150. Sources: openadmet.ghost.io (CYP challenge wrap-up), github.com/whis9/rasayan-cyp, inductive.bio news item.

## 8. Session 005.5 (2026-11-17): D015 and the staged comparison package

**Decision.** Miles stated a comparison plan, recorded as **D015** in `notes/Decisions.md` (form chosen by Miles: "New D015" over D013 Amendment 5): frozen T121 vs updated T121 and ActFound, preserving full O150, protected identities and the existing scoring and comparator-provenance checks including non-final status if unresolved; ActFound from a fresh seeded foundation initialization with the corrected learning-rate penalty, then selected foundation → screening → pIC50 (80 screening epochs, existing schedule, eight fixed validation episodes, minimum native query WMSE checkpoint, no full-screen refit; no historical penalty-omitting checkpoint); T121 screened for 50 epochs with trial-specific warmup 10; approved uncertainty and harmonic pair weighting preserved from qualified code, not recreated from defaults. Execution: Miles runs all execution commands; no training, model runs or SLURM submissions are authorized; runtime benchmarking is waived only for this CYP comparison, not correctness checks; no commit or push. Wording and dates are in S1. D015 authorizes no run.

**Staged package.** A staged ActFound/T121 package was built and archived (`actfound_staged_integration_20261117.tar.gz`, 78 files, SHA-256 `79c4ace8b862d84b8b52be009a082f942ff7d694813e977ec1f2a1d8c33c793b`, uncommitted). Local check: archive hash and internal `SHA256SUMS.txt` verified, and all 78 files are byte-identical to the working tree. Its `PACKAGE_SCOPE.txt` states it is not a production launch bundle. New and modified scripts, tests, other archives and qualification folders are listed in S2; the modified `actfound_model_side.py` (`86ccf4b1…`) and `actfound_production_train_v2.py` (`9627ff17…`) are the sources the job below ran.

**Tests.**

| Suite | Where | Result | Nature |
|---|---|---|---|
| Stage controller (11) and stage adapters (8) | Mac | 11 and 8 passed | mock executors; training and subprocesses mocked |
| Same 19 tests plus archive and package checksum checks | CoreHPC, ActFound env (Python 3.9), reported by Miles in summary | passed; package checksum check 0 non-OK lines | mock/component; "Ran N tests" lines not pasted |

**GPU correctness job 3103473.** COMPLETED `0:0` (`sacct`, End 2026-11-17T18:57:58). Result PASS on a random-initialized qualification model with 0 optimizer steps and 8 validation episodes (validation score 0.9987984361996203): shared state modes and RNG restored, no accumulated parameter gradients, repeat-then-equal-enzyme aggregation, manual metric and tie rule, checkpoint-writer round trip and authorization gate. These numbers are not performance claims. Not executed: the screening training loop, foundation pretraining, production staged execution. The result file (CoreHPC, not in this checkout) has SHA-256 `4e7ae5520ce764bb2b3ed85fb9c50edede96afa0ce625acdc497f3c23521f0fd`; the result's `runner_sha256` equals the current local runner. The model-side and runner hashes in the CoreHPC staging folder equal the current local files, but were taken after the job ended, so only the runner is corroborated in-job, and the controller checks only status, scope and `optimizer_steps` and does not enforce this binding. An earlier note that the stage_transfer archive held the job's runner was wrong: that archive's copy is `87d3766a…`; why it differs is unexplained. Transfer and hash details are in S4.

**Claim boundary.** Launch readiness is not established and no resume support is established. Component and mock tests do not show otherwise.

## 9. Stopping point and next step

Session 005 closed 2026-11-17 with the D013 study complete (ActFound-CYP O150 `0.8699544509650656` vs T121 `0.701212576796`, non-final) and the D015 comparison package staged but not launched. No training, model run or SLURM submission is authorized.

Unresolved items carried into Session 006 are listed in `notes/Tripwire_CYP_006.md` §1: the wiring inspection (support weights reaching adaptation, normalization/reset semantics, hash-bound qualification evidence, real adapter/config/input binding, T121 routing and epoch-end checkpoint behavior), the observation that `run_task` builds observation weights only when `is_training_phase=True` so evaluation-time adaptation would be unweighted, T121 not yet connected to the controller, comparator provenance, whether D015 supersedes D013's retrain-once and checkpoint-freeze terms, the `d014_lr_penalty` folder name colliding with PXR D014, and the job 3103473 hash binding. The next step is Session 006's wiring inspection.
