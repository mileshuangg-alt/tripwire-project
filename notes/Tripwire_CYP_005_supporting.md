# Tripwire-CYP-005 — Supporting Record

**Purpose:** exhaustive evidence behind `notes/Tripwire_CYP_005.md`: the dated decision/approval table, the artifact and SHA-256 inventories, the 2026-11-17 CoreHPC transfer and job-record checks, and the correction history. The session report references this file rather than copying it.

**Provenance:** the sections below are extracted unchanged from the pre-restructuring session log, preserved byte-identical as **A3** = `notes/archive/Tripwire_CYP_005_archive_20261117.md` (SHA-256 `c2c9e5ed55334fc8615bbfd1f1550ab84dceabb752edb06640b1196fe961dd63`). Section numbers inside the extracts (§5.14, §7.1, §10, …) are A3 section numbers. A1 (`..._20261115.md`) and A2 (`..._20261116.md`) are unchanged and still hold the original wording and commands.

---

## S1. Decisions and approvals (A3 §6)

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

## S2. Artifacts (A3 §7)

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

## S3. SHA-256 inventory for the completed ActFound-CYP work (A3 §7.1)

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

## S4. 2026-11-17 staged package: transfer, tests and job 3103473 (A3 §5.14)

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

## S5. Correction history (A3 §8)

Each correction keeps its original text in the archive.

- **`graph_pair_count` magnitude:** first recorded as "both directions, 1–2 pairs". It is 1–47 pairs and almost always upward (A1 §49 B annotation, §50 A).
- **Digit count in job 3075157's analysis:** only one of the two authors' values said to carry 16 digits does (A1 §53 C).
- **Activity 2091754:** first recorded as outside the split; it is inside (A2 entry 27).
- **§40 sentence:** corrected in place twice on Miles's approval. Its corrected wording was inferred because the fork session holding the original was deleted (A1 §40). This is what prompted the working-practice instruction in A1 §46.
- **Thin-adapter brief:** first recorded as not known to have preceded the evaluation code. It did; it is the brief Amendment 4 item 4 cites (A2 entry 51).
- **Evaluation run status:** the report previously stated that the evaluation code had not been run or uploaded and that the full evaluation was not authorized. That was true as recorded; Miles subsequently ran the full evaluation on CoreHPC as job 3083113 (reported 2026-11-16), so the status lines in §5.8, §9 and §11 were updated. The earlier deployment/path and local dependency problems are superseded by that successful HPC run.
- **Withdrawn overnight-failure rule:** its log entry was removed at Miles's request before either archive was made (A1 §35).
