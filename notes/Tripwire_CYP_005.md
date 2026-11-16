# Tripwire-CYP-005 — Session Log

**Session:** Tripwire-CYP-005 · **Status:** OPEN — in progress · **Start date:** UNKNOWN (recovered history imported 2026-11-14)
**Workstream:** Tripwire — CYP inhibition arm; ActFound ChEMBL32 foundation model under D013
**Execution target:** CoreHPC, project `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project`; pinned ActFound env `/scratch/user/mhuang8/actfound/envs/actfound_c96f055`
**Archive:** `notes/archive/Tripwire_CYP_005_archive_20261115.md`, a byte-identical copy of this log before compaction (SHA-256 `b70517b8f70614ae6215044a184db93cd870afc1691b6015e02e551da005ce87`, 3,046 lines). `A§N` below means section N of the archive. It holds the exact wording of every approval, decision quote, command and output summarized here. References of the form "Session 005 §N" in `notes/Decisions.md` (D013 and its amendments) mean `A§N`.

---

## Current state

*Mutable summary, updated in place. Everything below the next `---` rule is append-only history. Last updated: 2026-11-15 (after entry 27; log compacted the same day; reconciled the same day with commit `877f332` and the observed 3075299 stdout).*

- **Objective:** D013 — ActFound-CYP whole-package study on a firewalled ChEMBL32 foundation corpus (D013 status "Approved design; foundation corpus, retraining and evaluation not yet executed"). The route-C reproduction gate passed under D013 Amendment 2 as amended by Amendment 3 (entries 21–25, 27). The firewalled corpus build is next in D013 but is not authorized (entry 24).
- **Last completed step:** reproduction job 3075299 COMPLETED `0:0`; reproduction gate PASS under D013 Amendment 3, scope `0ca8f477310cb7f3`, 0 blocking differences (entry 27). Reproduction jobs 3073155 and 3075157 both stay FAILED as recorded (entries 20, 23). Amendment 3 builder on CoreHPC matches the checkout; CoreHPC test run UNKNOWN (entry 26).
- **Next step:** UNKNOWN. The firewalled corpus build is not authorized and has not been run (entries 24, 27).
- **Pending decisions or approvals:**
  - Firewalled corpus build, benchmark and training: not authorized (D013 Amendments 1–3; entries 24, 27).
  - D013 open items 1–4 (seed meaning for a single frozen checkpoint, 150 vs 50 cells; label-scale conversion; large-support handling; the thin target adapter and its tests): open; each needs approval before evaluation (entry 8).
- **Active constraints:**
  - Governing: D013 and Amendments 1, 2 and 3, all Approved. Amendment 3 replaces Amendment 2 item 2 only. Firewall mode refuses to run without a PASS reproduction manifest made under the current comparison scope (Amendment 1 gate; entry 22).
  - Production training and the 150-cell evaluation need D013's qualification, interactive benchmark and runtime extrapolation gates plus Miles's go-ahead (entry 8).
  - Decisions and progress are recorded when they happen, not reconstructed later (entry 19). This compaction is a reorganization Miles approved on 2026-11-15; the archive keeps the original text.
  - CoreHPC work goes through the terminal with the established SSH commands, and Miles authenticates (password + Duo) himself (entry 8).
  - Files in this checkout are not evidence of what a CoreHPC job loaded or produced.
  - Who runs later CoreHPC jobs: UNKNOWN. Miles paused assistant CoreHPC work for credit cost and ran 3073155, 3075157 and 3075299 himself; the assistant ran read-only probes and checks (entries 18, 20–23, 26).
- **Artifact references:**
  - Reproduction inputs pinned in Amendment 1, CoreHPC paths and SHA-256: A§45 D; as used by 3075157: A§52 B.
  - Builder on CoreHPC and in this checkout (checkout copy committed in `877f332`): `qualification/actfound/d013_firewall_20261115/build_firewalled_corpus.py` `9c5d449b37e840e4b49cdb5451c12fe799d21070e36e2f58f262857fbb562457`, scope `0ca8f477310cb7f3` (Amendment 3); tests `d2f6c8c3713721389dc56bf2168535e6a009414fca21465d7836bda9d760ebe8`. Deployed and checkout hashes match (entries 25, 26). The Amendment 2 builder `096951eb…` (scope `a358c320693929d9`) is committed in `3bc0d28` and was what 3075157 ran (entries 22, 23).
  - Job 3073155 outputs: CoreHPC `/scratch/user/mhuang8/actfound/chembl32/d013_reproduction_20261115` (file list A§47 B). Job 3075157 outputs: `/scratch/user/mhuang8/actfound/chembl32/d013_reproduction_20261115b`, stdout `/scratch/user/mhuang8/d013_repro2_3075157.out`. Output SHA-256 values for both: not reported, UNKNOWN. Job 3075299 outputs: CoreHPC `/scratch/user/mhuang8/actfound/chembl32/d013_reproduction_20261115c`, stdout CoreHPC `/scratch/user/mhuang8/d013_repro3_3075299.out`, 7,245 bytes, modified Nov 15 16:32, SHA-256 `975550f41c78307d08f6f0bafc6025ba67f5172a0cf357ea46a2dbc2284ac281` (assistant read-only check, 2026-11-15; `--output` pattern in entry 26); PASS manifest `corpus_manifest.json` `2e8a6d184380a3c74796b884db8d30ffb5792f12be908385cadaa597599b9306` (the value firewall mode's `--reproduction-manifest-sha256` would require); other output hashes in entry 27.
  - Committed on `main` as `877f332` "Implement D013 Amendment 3 and record the job 3075299 reproduction PASS": Amendment 3 builder `9c5d449b…` and tests `d2f6c8c3…`, `notes/Decisions.md` Amendment 3, this log through entry 27, the archive `notes/archive/Tripwire_CYP_005_archive_20261115.md`, and `.claude/agents/bookkeeper.md` (git history). Push to the `origin` and `github` remotes: not yet recorded.
  - Firewall exclusion list v2 (517 molregnos): `qualification/actfound/d013_firewall_20261115/firewall_list_v2/` (entry 15).
  - Job 3066823 (not the study checkpoint; preserved unchanged): outputs under CoreHPC `/scratch/user/mhuang8/actfound/chembl32/production_training_20261113/actfound_chembl32_production_v1` (entry 3).

---

## History

### 1. Recovered history imported (2026-11-14)
- **What:** Session 005's earlier work was imported from two sources: GPT's exported record [GPT-R] and the corrected mid-session handoff [HO-C]. Miles approved importing this session's history only, with no backfill of other sessions or edits to decisions (A§2).
- **Why it matters:** entries 2–3 rest partly on these sources. Most of their implementation details are OpenCode reports that neither source re-verified. The retained 2026-11-12 handoff and the decision copy they cite were not accessible (A§2, A§3).
- **Unresolved:** session start date UNKNOWN; the 2026-11-12 handoff date is source-reported only (A§3).

### 2. Pre-D013 ActFound production pipeline, as recovered (2026-11-14)
- **What:** a ChEMBL32 graph-qualified population was built (54,732 assays, 2,773,995 rows; merged TSV `24d9fd297b735843284d82e7d31b85ccd46038f279931db8d5997fb9a096149e`, hash confirmed in A§28). The model side and production runner were reconstructed. Upstream split IDs were bridged to frozen assay keys through `assay_chembl_id` plus the descriptor (per-ID reconciliation 33,296 train + 500 validation, `7ea5ae080ecc52fb95dd293230bd6225f5e09a73622a9ec0e6dec46a02f74645`), replacing an earlier CHEMBL-prefix mapping that was shown to be wrong (A§5–§10).
- **Failures:** jobs 3048559 (wrapper path), 3048570 (TSV header `final_assay_size` vs `count_scratch`), 3048576 and 3052606 (no train assays loaded, the latter under the obsolete prefix mapping) failed; 3048574 was cancelled by Miles; an H100 attempt failed on sm_90, so jobs were constrained to L40S (A§7, A§11).
- **Other:** a gpu-dev1 route was considered and not used. Miles cancelled the obsolete D011 array `2979324_[57-1649]` (A§12, A§13).
- **Evidence:** A§5–§17. D012 is background only, not this run's interpretation contract (A§3).

### 3. Job 3066823 completed and verified (2026-11-15)
- **What:** production job 3066823 (one L40S, `ggpu1-11`) COMPLETED `0:0` in 12:16:08. 60 epochs; best epoch 58, score `0.977011134486706` (`score = mean_r2 − mean_rmse + 1` on the 500 validation assays). `model_best.pt` equals `model_58.pt`. It ran with the current config's content (`64ea4af5…`). The startup source hashes match the CoreHPC live files for six source files. Runtime: Python 3.9.18, torch 1.12.1, RDKit 2023.03.1 (A§21, A§22, A§25, A§26).
- **Why it matters:** checkpoint selection and reported metrics use the same validation set, so they are optimistic; no held-out evaluation exists (A§25).
- **Evidence:** CoreHPC output directory in Current state; `completion.json` `b077c590…`, `progress.jsonl` `92bdd843…`, `model_best.pt` `9eedd74e…` (A§22). Validation trajectory: A§25.
- **Unresolved:** the executed Slurm wrapper cannot be recovered (`sacct -B` returned `NONE`). The cause of overlap step `3066823.5` (nvidia-smi) `255:0` is UNKNOWN; the batch step was unaffected (A§22).

### 4. Local sync from CoreHPC and test-file fixes (2026-11-15)
- **What:** 10 CoreHPC files were copied byte-identical into this checkout (bundle `ed2c8c8f…`; Miles approved the copy). Two test files had defects, not the code 3066823 ran. Miles approved fixing them: `test_actfound_production_train.py` → `ef99704a…`, `test_actfound_model_side.py` → `6f5fa3d9…`. Miles reported "all tests pass" on CoreHPC (A§23, A§24).
- **Unresolved:** CoreHPC pass count and post-upload hashes were not reported; they were later seen matching in A§36 B.
- **Corrections in passing:** the production config hash `64ea4af5…` and `actfound_reported_unresolved_31_resolution.tsv` `dced5472…`, UNKNOWN in A§8/§9/§18, are now recorded (A§23).

### 5. Miles's statement of the intended study (2026-11-15)
- **What:** Miles stated the study established on 2026-11-12: leakage-clean ChEMBL pretraining; upstream ActFound adaptation using each frozen D007 cell's training molecules only; evaluation on its validation molecules; the unchanged 150-cell comparison against frozen T121; whole-package claims only; D012 as background. He also instructed: don't reopen the architecture or adaptation-path menu, add a published-fidelity gate, or reselect epochs through extra validation draws (A§26 E).
- **Why it matters:** it became the D013 contract (entry 8). The original 2026-11-12 record is not accessible (A§26 E).

### 6. Job 3066823's corpus was not firewalled: lineage and identity overlap (2026-11-15)
- **What:** no stage from the ChEMBL32 SQLite projection to the merged TSV `24d9fd29…` applies an identity firewall; the projection and pregraph summaries record `openadmet_firewall_applied: false` (A§26–§29). Two molecules were confirmed by full Standard InChIKey and Standard InChI in the frozen identity env: molregno 1370 (`OCNT-0018684`) and 88100 (`OCNT-0495340`) (A§30; Miles approved that check). A full-corpus audit, approved by Miles, found these in loader-kept training rows: 203 D007 development molecules (2,139 rows / 1,515 assays), 44 internal-test, 7 blinded-challenge (A§31).
- **Why it matters:** 3066823's checkpoint is not leakage-clean under the 2026-11-12 firewall requirement, which led to entry 7.
- **Evidence:** this checkout, `qualification/actfound/overlap_audit_20261115/` (scripts; `full_run/`, `SHA256SUMS.txt` verified; `molecule_identities.tsv` is git-ignored). Export bundle `30073d53…` in `~/Downloads/actfound_overlap_export_20261115/` (A§31).
- **Unresolved:** the sort step from projection output `bd804a6d…` to pregraph input `6d8fe4ac…` has no recorded summary. The projection script bytes that ran at 23:02 are unrecovered (recorded `71161e52…` ≠ deployed `22fca408…`). Bundle-level hash equality for the export is unverified (A§29 C, A§31 B, A§36 B).

### 7. Route decided: retrain on a firewalled corpus; sub-choices approved (2026-11-15)
- **Decision:** Miles chose Option A, retraining the foundation model on a firewalled corpus. Using 3066823 as-is (B) and keeping it as a leaky-vs-clean diagnostic (C) were not chosen (A§32).
- **Approval:** Miles approved the proposed sub-choices and asked for a D013 draft. Firewall scope is all 5,655 OpenADMET structures (development 4,169 + internal test 736 + blinded 750) by full Standard InChIKey. Base corpus is the authors' released `chembl_processed_chembl32.csv` (`a5b562d4…`). Removal is molecule-level, then assays below 20 usable molecules are dropped and logged, validation included (A§33).
- **Evidence:** authors' corpus qualification: 1,959,398 rows, 47,324 assays; every (descriptor, molregno) pair is present in the rebuilt corpus (A§33 A).

### 8. D013 approved; implementation authorized (2026-11-15)
- **Decision:** D013 is Approved and recorded in `notes/Decisions.md` (heading at line 6135). Miles authorized implementation. Production training and the 150-cell evaluation are not authorized by D013 itself. They need its gates and his go-ahead, and D013 open items 1–4 need separate approval (A§34).
- **Instruction:** run everything through the terminal; for CoreHPC use the established SSH commands and wait for Miles to authenticate (A§34 D).

### 9. Implementation answers Q1–Q9 accepted; monitor and alerts declined (2026-11-15)
- **Approval:** Miles accepted the assistant's answers to Q1–Q9. Those cover: split = upstream train/valid minus firewall-emptied assays, in a v2 config; config-driven runner v2; firewall then graph re-qualification; salt/parent near-matches reported, not removed; no resume or requeue; output-directory guard; pre-submission record; scratch cache paths; value-by-value converter check (A§35 D).
- **Also:** no live monitor agent; no Slurm mail alerts. An overnight-failure rule was withdrawn and its log entry removed at Miles's request; that removal predates the archive (A§35 B).

### 10. Firewall list v1, runner v2 and builders written and tested (2026-11-15)
- **What:** firewall list v1 has 337 molregnos (development 263, internal test 61, blinded 13), with near-matches reported only (A§36 C). Runner v2 and wrapper v2 (`ed3e1055…`, `439c8ad1…`) and the route-C builder and split/config builder were written, uploaded with verified hashes, and passed 41 tests in the pinned env (A§36 F, A§37).
- **Why it matters:** upstream graph eligibility depends on pre-graph assay size, so graph qualification cannot be rerun from the authors' post-graph rows alone. This forced the route-C question (A§36 E).

### 11. Route C as proposed repair path; narrow fix and impact count authorized (2026-11-15)
- **Decision:** Miles said to proceed with route C as the proposed repair path. Amendment 1 stayed Proposed. He authorized only a split-aware stop-rule fix (in-split differences block with no tolerance or normalization, outside-split differences are reported, both behaviours tested) and a read-only impact count of the 180 largest-fragment matches with no graph rebuild. Not authorized: corpus build, reproduction run, benchmark, training. The runner's manifest check is defense-in-depth, not proof of zero overlap. The identity rule was not widened (A§38).

### 12. Stop-rule fix and 180-match impact count (2026-11-15)
- **What:** the fix passed locally, with mutation checks (A§39 A). The impact count found 180 molecules (development 140, internal test 40) with no overlap with the 337, and 2 more training assays falling below 20. Blinded identities are already in the exclusion list (A§39 B). The count script and output were in the fork session's scratchpad (paths and short hashes in A§39 B); the fork was later deleted (A§46), so whether those files still exist is UNKNOWN.

### 13. Exclusion widened to the 180 largest-fragment matches, validation included (2026-11-15)
- **Decision:** Miles kept full Standard InChIKey as the identity rule and added largest-fragment full-key matches as a conservative pretraining exclusion. It applies to pretraining validation rows too, because validation selects the checkpoint (A§40 A).
- **Correction:** one A§40 sentence about the assistant's earlier recommendation was corrected in place twice on Miles's approval. The original fork session was deleted, so the corrected wording was inferred rather than quoted (A§40 Correction).

### 14. Firewall-v2 plan and Amendment 1 exclusion/acceptance wording approved (2026-11-15)
- **Approval:** Miles approved the proposed plan: list v2, builder manifest counts by exclusion basis, a corpus identity acceptance check, and tests. He also approved the Amendment 1 pretraining-exclusion and acceptance wording. Amendment 1 as a whole stayed Proposed (A§41).

### 15. Firewall-v2 implemented locally (2026-11-15)
- **What:** `firewall_list_v2/` has 517 molregnos (337 full-key + 180 largest-fragment), verified to equal v1 ∪ the 180. `check_corpus_identity.py` fails on any full-key or largest-fragment match in train/valid rows. 24 local tests passed (A§42).

### 16. D013 Amendment 1 approved and recorded (2026-11-15)
- **Approval:** Miles approved the corrected Amendment 1 for recording: route-C graph re-qualification with the exact qualification gate, plus the conservative exclusion. It is recorded in `notes/Decisions.md` (heading at line 6246). It does not authorize the corpus build, reproduction run, benchmark or training, nor the CoreHPC upload or tests (A§43).

### 17. Fork merged; fork work canonical (2026-11-15)
- **What:** on Miles's instruction, the fork session's `d013_firewall_20261115/` work and runner v2 files became canonical. 51 local tests passed in the merged state (A§44). The main session's earlier runner v2 versions were overwritten and are unrecoverable. The superseded `qualification/actfound/firewall_corpus_20261115/` is retained because Amendment 1 cites it.

### 18. D013 deployed; pinned-env tests; reproduction job 3073155 submitted (2026-11-15)
- **Approval:** Miles authorized the CoreHPC upload, pinned-env tests and reproduction run. He then paused assistant CoreHPC work for credit cost and submitted the job himself (A§45).
- **What:** deploy bundle `ab2eb4aec5885e0a01b41ab768befb682f37483e5f0d6d6c604d5172ffd24450`, all 18 file hashes equal local. Tests: 49 passed, 9 failed, all 9 from the frozen-identity-env gate (`GetInchiVersion` absent in RDKit 2023.03.1); these are not execution-path defects. Job 3073155: partition `cpu`, 64G, 8 h, reproduction mode (A§45 B–E).

### 19. Working practice: record at the time, no retrospective amendment (2026-11-15)
- **Instruction:** Miles instructed that decisions and session progress must not be amended or logged retrospectively (A§46).

### 20. Job 3073155: reproduction gate FAILED; differences characterized (2026-11-15)
- **What:** exit 3 with 1,525,375 in-split differing rows. A field-level diagnosis of all 1,959,398 rows found three classes. `record_id_unused` differs in every mismatching row; it is a provenance column upstream does not consume. `standard_value` differs in 600 in-split rows, all within about one unit in the last place (15 vs 17 significant-digit text). `graph_pair_count` differs in 69 assays, metadata that reaches no training path. No training-defining field differs (A§47–§49).
- **Superseded explanation:** a duplicate-row/dedup hypothesis for the `standard_value` rows was refuted (A§49 A).
- **Correction:** A§49 B's "both directions, 1–2 pairs" for `graph_pair_count` was wrong. The deltas are 1–47 pairs and almost all upward. A§49 B was annotated on Miles's approval (A§49 B annotation, A§50 A).
- **Evidence:** CoreHPC outputs in Current state; probe scripts were in the assistant's scratchpad, run read-only.

### 21. `graph_pair_count` cause narrowed; D013 Amendment 2 approved (2026-11-15)
- **What:** the exact-equality pair rule was refuted as the cause. 5 of the 69 assays have byte-identical inputs and still differ, which leaves Morgan-fingerprint similarity (a different RDKit build) by elimination; similarity was not measured. Byte-identity with the authors' file is not achievable from the pinned env (A§50 A).
- **Approval:** Miles chose to investigate the cause first (option C), then amend the gate scope (option A). He approved the Amendment 2 wording, recorded in `notes/Decisions.md` (heading at line 6300). It changes only the gate's comparison scope and does not authorize the corpus build, benchmark or training (A§50 B).

### 22. Amendment 2 implemented in the builder (2026-11-15)
- **What:** on Miles's authorization, `build_firewalled_corpus.py` `c73d6a7c…` → `096951eb…` and its tests `ebb5ed97…` → `617248d9…`. The builder now uses a declared comparison scope `a358c320693929d9`, reports exempt-column differences, and refuses a manifest from another scope. 31 local tests passed, and three mutation checks fail the tests that pin them (A§51). Miles said he would upload and run it himself (A§51).

### 23. Job 3075157: reproduction gate FAILED on 4 rounding-boundary rows (2026-11-15)
- **What:** 3075157 ran the deployed Amendment 2 builder (`096951eb…`, scope `a358c320693929d9`) and exited 3 after 00:12:57 on `gcpu2-05`. Exactly 4 in-split `standard_value` rows blocked; they straddle a 15-significant-figure rounding boundary. This is a correct FAIL under Amendment 2; no code defect (A§52).
- **Probes (read-only):** max relative `standard_value` difference is 4.35e-15, with none above 5e-15. The 826 outside-split mismatches are 825 `count_scratch` + 1 `standard_value`. 8,657 of the authors' values carry more than 15 significant digits, contradicting Amendment 2's premise (A§53).
- **Correction:** A§52 E said two of the four authors' values had 16 digits; only `9003570138859.371` does (A§53 C).

### 24. D013 Amendment 3 approved and recorded (2026-11-15)
- **Approval:** Miles approved the Amendment 3 wording for recording, in `notes/Decisions.md` lines 6343–6387. It replaces Amendment 2 item 2 with a numerical relative tolerance of 1e-14 on `standard_value` and adds manifest reporting of the tolerance's effect. It does not authorize the builder/test change, a rerun, the corpus build, the benchmark or training (A§54).

### 25. Amendment 3 implemented in the builder, local only (2026-11-15)
- **What:** on Miles's authorization to implement, `build_firewalled_corpus.py` `096951eb…` → `9c5d449b…` and its tests `617248d9…` → `d2f6c8c3…`. `standard_value` now agrees when the texts are equal or both parse as finite floats within 1e-14 relative; otherwise it is compared exactly as text. The manifest reports the tolerance's effect per scope. The new scope ID `0ca8f477310cb7f3` makes the builder reject Amendment 2 manifests, including 3075157's. 47 local tests passed. Mutation checks: five mutations fail tests; one, removing the finiteness guard, survives, and the assistant judges it an equivalent mutation (A§55).
- **Why it matters:** a PASS manifest now needs a new reproduction run under the new scope. That run is not authorized and the code is not deployed (A§55).

### 26. Amendment 3 builder on CoreHPC; reproduction job 3075299 submitted (2026-11-15)
- **What:** the assistant asked "Do you authorize the new reproduction-mode run?". Miles submitted the run himself and reported: "Submitted batch job 3075299".
- **Deployment check (assistant, read-only over the shared SSH connection; nothing written on CoreHPC):** CoreHPC `.../tripwire-project/qualification/actfound/d013_firewall_20261115/build_firewalled_corpus.py` `9c5d449b37e840e4b49cdb5451c12fe799d21070e36e2f58f262857fbb562457` and `test_build_firewalled_corpus.py` `d2f6c8c3713721389dc56bf2168535e6a009414fca21465d7836bda9d760ebe8`, both equal to entry 25's local hashes. Deployed and checkout code match again (Amendment 3, scope `0ca8f477310cb7f3`). Who uploaded them and how: not reported, UNKNOWN. Whether the tests were run on CoreHPC: not reported, UNKNOWN.
- **Job (sacct, same check):** 3075299 RUNNING on `gcpu2-05`, submitted 2026-11-15T16:19:48, elapsed 00:00:12 at check. Submit line: `sbatch --partition=cpu --cpus-per-task=4 --mem=16G --time=2:00:00 --job-name=d013-repro3 --output=/scratch/user/mhuang8/d013_repro3_%j.out`. Reproduction mode (no firewall list), with the same inputs as 3075157: pregraph `43dfd672…db076`, authors CSV `a5b562d4…5e20`, graph-code dir `/scratch/user/mhuang8`, upstream split `7f3cd598…e3e1`. `--out-dir /scratch/user/mhuang8/actfound/chembl32/d013_reproduction_20261115c`.
- **Status:** running; result not yet known. Job 3075157 stays FAILED (entry 23). Corpus build, benchmark and training remain unauthorized (entry 24).

### 27. Job 3075299: reproduction gate PASS under Amendment 3; correction to A§53 (2026-11-15)
- **What (Miles, manifest read on the CoreHPC login node):** `/scratch/user/mhuang8/actfound/chembl32/d013_reproduction_20261115c/corpus_manifest.json`, `reproduction_vs_authors`: gate `PASS`, `blocking_differences` 0, `comparison_scope_id` `0ca8f477310cb7f3`. In split: 1,525,375 identical rows. Outside split: 433,198 identical, 825 `field_mismatch`, 1 `extra_in_rebuilt`. `standard_value_tolerance` (rule "D013 Amendment 3", relative tolerance 1e-14): max relative difference in split `4.351811221356154e-15` (activity 2091754), outside split `1.4095890945413776e-15` (activity 1877390); rows agreeing within tolerance but not as text: 600 in split, 203 outside.
- **Job and hashes (assistant, read-only over the shared SSH connection; nothing written on CoreHPC):** sacct 3075299 COMPLETED, ExitCode `0:0`, Elapsed 00:13:00; batch step MaxRSS 2184328K (requested 16G, 2:00:00). Out-dir `SHA256SUMS.txt`: `reproduction_corpus.tsv` `ad44ccce633db3cea5dd86b2450d52a3c0a3314dd7a7b13565405b20eac02114`; `corpus_manifest.json` `2e8a6d184380a3c74796b884db8d30ffb5792f12be908385cadaa597599b9306` (direct `sha256sum` agrees); `graph_stage_summary.json` `04c7e6fcfa52204f0996b4ec84ef1cff2e07aea4da88066cfb6800dd576e4305`; `reproduction_differences.tsv` `d416cd92ae1adf00708184a1b51d22a2c9bca7c6a1a92be39ca13a10093f198e`. `exempt_differences_by_assay.tsv` (not in `SHA256SUMS.txt`) `4c23b49287226c60fddd0da847b4b2bb0183f50ee1d5a177069f4a313e76f424`.
- **Why it matters:** this is the PASS reproduction manifest under the current scope that firewall mode requires (entry 22). Its SHA-256 `2e8a6d18…` is the value firewall mode's `--reproduction-manifest-sha256` would require (assistant). The firewalled build has not been run, and the build, benchmark and training remain unauthorized (entry 24). The Amendment 3 code and these records are still uncommitted in this checkout.
- **Consistency (assistant analysis):** the 600/203 within-tolerance rows equal Amendment 2 Reason 2's 600/203 `standard_value` counts. The 825 outside-split mismatches match the 825 `count_scratch` rows (entry 23). The in-split maximum `4.351811221356154e-15` equals the overall maximum in the 3075157 probe (A§53).
- **Correction to A§53 (summarized in entry 23, which does not repeat the error):** A§53 (archive line 2917) recorded activity 2091754's 4.35e-15 difference as "outside split". That was the assistant's error: the probe's `-` meant the row was not in the differences file (it passed at 15 significant figures), not that it was outside the split. 3075299's manifest shows 2091754 is in split. Amendment 3's text does not state that row's scope, so the decision record is unaffected. A§53 and entry 23 are left as written.
