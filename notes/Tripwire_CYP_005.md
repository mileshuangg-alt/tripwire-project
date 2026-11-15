# Tripwire-CYP-005 — Session Log

**Session:** Tripwire-CYP-005
**Date:** Start date UNKNOWN; session ongoing. Recovered history imported 2026-11-14.
**Status:** **OPEN — in progress** (Miles, 2026-11-14: "Session 005 is ongoing.")
**Workstream:** Tripwire — CYP inhibition arm; ActFound ChEMBL32 model-side reconstruction and production training
**Governing decision for the ActFound run:** UNKNOWN — no D-number has been established for this run
**Primary execution target:** CoreHPC
**CoreHPC project:** `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project` [HO-C]
**Pinned ActFound environment:** `/scratch/user/mhuang8/actfound/envs/actfound_c96f055` [GPT-R, HO-C]

---

## 1. Session Objective

Recovered from the sources below. The work is the ActFound ChEMBL32 model-side reconstruction and production training run [GPT-R, HO-C]. The operational objective was to:

1. reconstruct the ActFound model/data path;
2. preserve upstream dataset filtering, sampling, pair construction, loss and adaptation semantics;
3. use the frozen ChEMBL32 graph-qualified population;
4. reconcile the upstream assay split membership to the frozen population explicitly;
5. qualify the production runner and obtain a real production ActFound training run on CoreHPC in the pinned environment.

[GPT-R, HO-C]

The immediate operational objective at the time of the handoff was to let the already-submitted production job continue running, then collect its training artifacts and results [HO-C].

A formal ActFound-specific hypothesis, baseline comparator, score threshold, acceptance criterion or advancement rule is UNKNOWN. None is established in either source [GPT-R, HO-C].

---

## 2. Sources and Provenance

This log imports recovered history for the current Session 005 only, under Miles's approval of 2026-11-14:

> "Have the actual bookkeeper create and populate notes/Tripwire_CYP_005.md using the existing session-log structure, not the proposed Live log redesign. Use the attached GPT record-so-far as source-labelled recovered history. Use the corrected handoff too if accessible in this session; don't pretend to have it if it isn't. Don't invent a start date: use UNKNOWN unless the supplied sources establish it."

> "This authorizes importing the recovered history of the current Session 005 only. Don't backfill other sessions, add a project-log advance row, edit existing decisions, sync checkouts or touch scientific files."

Source labels used throughout:

- **[GPT-R]**: recovered history from GPT's conversation, as exported in `/Users/mileshuang/Downloads/Tripwire_CYP_005_Record_So_Far.md` (record date 2026-11-14). It is GPT's account of what it could recover. Its own text says the history is partial and was subject to compaction.
- **[HO-C]**: the corrected mid-session handoff, `/Users/mileshuang/Downloads/Tripwire_CYP_mid_session_handoff_CORRECTED_20261114.md` (dated 2026-11-14, mid-session transfer, not a close-out).
- **[Miles 2026-11-14]**: Miles's own updates in the Claude Code conversation on 2026-11-14. These are authoritative for the current state and take precedence over [GPT-R] and [HO-C] on current state.

Sources not used:

- `/Users/mileshuang/Downloads/Tripwire_CYP_mid_session_handoff_20261114.md`: an older handoff. The CORRECTED file supersedes it. It exists but was not used as a source.
- `Pasted text(20261112-214104).txt`: the retained handoff into Session 005, cited by both sources. Not accessible in this session and not read.
- `Decisions(20261112-224142).md`: the decision-record copy cited by both sources. Not accessible in this session and not read.

Most implementation, qualification and reconciliation details below come from OpenCode reports and earlier chat output. Neither source independently re-verified them [GPT-R, HO-C]. None of them were re-verified for this log.

### Evidence location

- The production work and its artifacts are on **CoreHPC**. The hashes recorded here are CoreHPC live-file hashes as reported by OpenCode.
- Miles, 2026-11-14: "This morning's fixes were made directly on CoreHPC and haven't been transferred to the local repo."
- So the ActFound files in this local checkout (`/Users/mileshuang/Desktop/tripwire-project`) are not evidence of the CoreHPC live state or of what job `3066823` loaded. No local file was hashed or compared for this log.
- The gpu-dev1 extracted bundle is not evidence of the CoreHPC code or environment used by `3066823` [GPT-R].

---

## 3. Starting State and Governing Context

### Session start

The session start date is UNKNOWN. [GPT-R] reports that the retained handoff `Pasted text(20261112-214104).txt` identifies itself as `# HANDOFF: 2026-11-12 — Tripwire-CYP Session 005`. That file is not accessible here. The 2026-11-12 handoff date is source-reported only and does not establish the session start date.

[GPT-R] and [HO-C] report that this retained handoff carries D012 corrective-production context and the operational map and open items from Session 004.

### D012: background only

Both sources quote D012, "Assay-resolved CYP transfer study", with this status:

> "Status: Approved; preregistration complete. Implementation and qualification may proceed; production training remains separately approval-gated."

They quote its arms ("D012 has three arms:" A, target-only frozen T121; B, conventional transfer; C, assay-episodic ANIL) and:

> "The primary comparison is C versus B. B versus A and C versus A are secondary comparisons testing whether auxiliary training improves upon target-only T121."

[GPT-R, HO-C]. [HO-C] also states that D012 was approved 2026-11-10 and prospectively replaces D011.

The local working copy of `notes/Decisions.md`, read on 2026-11-14, has these same three quoted passages under `## D012 — Assay-resolved CYP transfer study`.

Neither source and no accessible record links this ActFound run to D012 Arm B, Arm C or any other arm. D012 is recorded here as project background only. It is not the ActFound run's interpretation contract [GPT-R, HO-C].

[HO-C] also reports that the retained Session 005 handoff records a D012 artifact-integrity contract for the corrected production matrix: per-cell `SHA256SUMS.txt` verification and exact frozen cell-key equality. Whether that contract governs this ActFound run is UNKNOWN [HO-C].

---

## 4. Reporting Metrics Identified

Both sources report that Miles quoted this Tripwire-CYP README language in the GPT conversation [GPT-R, HO-C]:

- primary CYP direct-inhibition metric: MA-ST-RAE (macro-averaged soft-threshold relative absolute error), the official challenge metric;
- also reported: MAE, R², Spearman ρ, Kendall τ;
- reported per isoform and macro-averaged across the four direct-inhibition endpoints.

Whether these are the intended interpretation metrics for this ActFound run is UNKNOWN unless a governing artifact links them [HO-C]. They are not an ActFound acceptance rule.

---

## 5. ChEMBL32 Graph-Qualified Population

The graph-qualified population was produced before the ActFound training work [GPT-R, HO-C]. [HO-C] describes it as a 16-shard graph production that completed successfully.

Recorded aggregate results [GPT-R, HO-C]:

| Quantity | Value |
|---|---:|
| processed assays | `67,985` |
| pregraph rows [GPT-R] / processed rows [HO-C] | `9,873,276` |
| graph-size-rejected assays | `169` |
| graph-valid-count-rejected assays | `13,084` |
| graph-qualified assays | `54,732` |
| final graph-qualified rows | `2,773,995` |
| graph-qualified pair count total | `66,612,505` |

Frozen merged input (CoreHPC):

`/scratch/user/mhuang8/actfound/chembl32/graph_production_20261113/actfound_chembl32_graph_qualified.tsv`

Reported size: about `459M` [GPT-R]; "Latest reported size: 459M" [HO-C].

Manifest (CoreHPC):

`/scratch/user/mhuang8/actfound/chembl32/graph_production_20261113/actfound_chembl32_graph_qualified.manifest.json`

[HO-C] reports that the manifest recorded the hashes of the merged output and the shard summaries. The exact final merged-output SHA-256 in the current live state is UNKNOWN. It is not reconstructed from earlier merge or materialization hashes [GPT-R, HO-C].

---

## 6. ActFound Model-Side Reconstruction

Project files created and qualified (CoreHPC live copies are the relevant ones):

- `cyp/scripts/actfound_model_side.py`
- `cyp/tests/test_actfound_model_side.py`

[GPT-R, HO-C]

Components reported by OpenCode [GPT-R]: `ActFoundAssay`, `ActFoundFrozenMetaDataset`, `iter_frozen_graph_qualified_assays`, `upstream_count_morgan_features`, `make_episode_split`, `construct_pair_matrices`, `robust_square_error`, `actfound_loss_and_preds`, `adapt_final_linear_weight`, `adapt_final_linear_params`, `continuous_inference`, `PROVENANCE`.

Reported qualification coverage [GPT-R]:

- dataset filtering and representation;
- exact pair counts and labels on a fixture;
- pairwise-loss behavior;
- sampling determinism;
- upstream mixed RNG;
- target adaptation and differentiable adaptation;
- continuous inference semantics;
- count-fingerprint Jaccard;
- complete forward topology;
- real upstream-network compatibility when the source tree was present;
- provenance coverage.

Implementation findings reported by OpenCode [GPT-R, HO-C]:

- Upstream mixed RNG behavior was preserved. Python `random.uniform()` is used for support-size sampling. NumPy `RandomState(rand_seed)` is used for the N>64 cap shuffle and the final split shuffle [GPT-R].
- The downstream invalid-SMILES drop (`preprocess_assay()`-style [HO-C]) and the `len(lines) > 10000` rejection were added.
- The production ActFound QSAR final `linear` layer uses `use_bias=False`, so production adaptation of that layer is weight-only.
- An earlier adaptation implementation detached parameters incorrectly. The corrected dict-based adaptation preserves differentiable dependence and `create_graph` behavior.
- Full forward topology was reconstructed and guarded [HO-C].
- Python 3.9-compatible typing/annotations were restored.

### Early test/debug issues [GPT-R]

- **Pairwise-loss fixture.** A fixture expected `2.5` for `robust_square_error`, but direct execution gave `actual_loss: 0.5`. The implementation was then recognized as correct for the actual gather semantics, and the test expectation was corrected. A later manual qualification reported `PASS: pairwise_loss`.
- **Manual harness errors.** The pinned environment did not initially expose `pytest` in the expected way, so a custom pytest stub was used. Several guessed test function names were wrong, and the actual definitions were later inspected with `grep -n '^def test_'`. These were harness/operator errors, not scientific failures.
- **Missing upstream source in one CoreHPC checkout.** One real upstream-network qualification failed because `external/ActFound` was absent from that CoreHPC checkout at the time. The source tree was later bundled separately for deployment.

### CoreHPC live-file hashes reported by OpenCode after job 3052606

| File (CoreHPC live) | SHA-256 |
|---|---|
| `cyp/scripts/actfound_model_side.py` | `97c1753512d102e1198bfad2ab4b601749cb1d57b6a1a3a1ad847be83214ee45` |
| `cyp/scripts/actfound_production_train.py` | `8a160427afad728348b572b9e9e432ad40493f2ad5e129594e4ee45c41258dc4` |

[GPT-R, HO-C]

These are current/post-fix live-file hashes. They are not evidence of the code path used by `3052606`, and they are not job-specific provenance for what `3066823` loaded [GPT-R, HO-C]. They are not hashes of the local checkout files.

---

## 7. Production Runner and Configuration

- Production runner: `cyp/scripts/actfound_production_train.py`
- Slurm wrapper: `cyp/scripts/actfound_production_train.slurm`
- Production config: `cyp/artifacts/actfound/actfound_chembl32_production_config_v1.json`

[GPT-R, HO-C]

Reported runner design [GPT-R]:

- fail closed unless a reviewed config is provided;
- verify pinned data/config hashes and runtime expectations;
- require explicit train/validation assay membership;
- instantiate the upstream `FCNReLUNormNetworkQSAR`;
- use the project-side ActFound primitives;
- write `startup_config.json`, `progress.jsonl`, checkpoints (under `checkpoints/` [HO-C]) and `completion.json`.

The Slurm wrapper was corrected to request an NVIDIA L40S [GPT-R, HO-C]:

`#SBATCH --gres=gpu:nvidia_l40s:1`

`sinfo` identified CoreHPC nodes `ggpu1-06` through `ggpu1-12` as L40S-capable [GPT-R]. The eventual production jobs were submitted to the CoreHPC `gpu` partition under `hpc_core` [HO-C].

Per OpenCode's later direct inspection of the live CoreHPC file, the live runner now consumes [HO-C]:

- `split_reconciliation_per_id_path`
- `split_reconciliation_per_id_sha256`
- `validate_split_reconciliation_per_id()`
- `resolved["mapped_train_assays"]`
- `resolved["mapped_val_assays"]`

It no longer uses the obsolete CHEMBL-prefix crosswalk [GPT-R, HO-C].

### GPU architecture failure (superseded state)

An early production attempt reached an H100 NVL and emitted [GPT-R]:

```text
NVIDIA H100 NVL with CUDA capability sm_90 is not compatible with the current PyTorch installation.
```

The pinned ActFound PyTorch stack does not support that compute capability. The production request was then constrained to L40S. The job ID of this attempt is UNKNOWN.

### Frozen graph header mismatch (superseded state)

The actual frozen graph-qualified TSV header, as reported by Miles in the GPT conversation [GPT-R]:

```text
assay_key	activity_id	assay_id	record_id_unused	molregno	bao_endpoint	standard_relation	standard_value	standard_units	standard_type	assay_type	bao_format	assay_chembl_id	count_scratch	canonical_smiles	graph_pair_count
```

The code initially expected `final_assay_size` and failed with `ValueError: unexpected graph-qualified TSV header`. The live model-side source was corrected to use `count_scratch`. Miles verified the correction by inspecting `GRAPH_QUALIFIED_COLUMNS` (`... "assay_chembl_id", "count_scratch", "canonical_smiles", "graph_pair_count"`) [GPT-R].

---

## 8. Split Reconciliation

### CHEMBL-prefix assumption (superseded state)

An early assumption was that an upstream split ID such as `CHEMBL644949_nM_Ki` could be mapped to the frozen `assay_key` by removing `CHEMBL`. A cheap full scan disproved this [GPT-R]:

```text
train_config: 33296
validation_config: 500
train_found: 104
validation_found: 2
```

The upstream split IDs and the frozen `assay_key` values are not related by stripping `CHEMBL` [GPT-R, HO-C].

### Actual identity bridge

The bridge is the frozen `assay_chembl_id` plus the assay-key descriptor. This includes the documented descriptor normalization/resolution of the 31 initially unresolved records [GPT-R, HO-C].

Example [GPT-R, HO-C]:

```text
CHEMBL644949_nM_Ki
  assay_chembl_id: CHEMBL644949
  frozen_assay_keys: ['31705_nM_Ki']
```

Further recorded examples [GPT-R]:

```text
CHEMBL830878_nM_IC50        -> 306103_nM_IC50
CHEMBL666759_nM_Ki          -> 48033_nM_Ki
CHEMBL763914_ug.mL-1_MIC    -> 153171_ug.mL-1_MIC
CHEMBL4264872_nM_MIC90      -> 1792953_nM_MIC90
CHEMBL3887051_nM_IC50       -> 1640328_nM_IC50
CHEMBL3061245_ug.mL-1_MIC   -> 902755_ug.mL-1_MIC
CHEMBL1639082_nM_IC50       -> 695412_nM_IC50
CHEMBL3995339_nM_IC50       -> 1646283_nM_IC50
CHEMBL4180939_ug.mL-1_MIC   -> 1746429_ug.mL-1_MIC
```

### Final per-ID reconciliation (CoreHPC live, rebuilt by OpenCode)

`cyp/artifacts/actfound/actfound_split_reconciliation_per_id.tsv`

Reported final verification [GPT-R, HO-C]:

```text
rows: 33796
train: 33296
validation: 500
unresolved: 0
ambiguous: 0
duplicates: 0
train/validation overlap: 0
31-resolution conflicts: 0
```

| Artifact (CoreHPC live) | SHA-256 |
|---|---|
| `cyp/artifacts/actfound/actfound_split_reconciliation_per_id.tsv` | `7ea5ae080ecc52fb95dd293230bd6225f5e09a73622a9ec0e6dec46a02f74645` |
| `cyp/artifacts/actfound/actfound_split_reconciliation_summary.json` | `86c73c4f5361de8a6fd238892129f7927fceb37e13950955af1397a5550e5fc5` |
| `cyp/artifacts/actfound/actfound_reported_unresolved_31_resolution.tsv` | UNKNOWN |

These results rest on OpenCode reports, later corroborated by the live runner's fail-closed logic. They are not independent handoff verification [HO-C].

---

## 9. Production Config Hash Correction and Validation

The production config initially contained the old per-ID reconciliation SHA-256 `e22fc261c961551128eae3749c3eec50ff798120d2fda4d97eadd549256c2116` [GPT-R]. This is a superseded state.

It was updated to the completed TSV SHA-256 `7ea5ae080ecc52fb95dd293230bd6225f5e09a73622a9ec0e6dec46a02f74645` [GPT-R, HO-C].

After the update, the exact `--validate-only` command returned [GPT-R]:

```json
{"config_path": "cyp/artifacts/actfound/actfound_chembl32_production_config_v1.json", "status": "validated"}
```

The final production-config SHA-256 after this edit is UNKNOWN [GPT-R, HO-C].

---

## 10. Dataset-Construction Preflight

OpenCode reported a successful bounded dataset-construction preflight in the pinned environment, using a temporary SHA-correct production config [GPT-R, HO-C]:

```text
data_length_train: 38565
data_length_val: 500
mapped_train_assays: 33296
mapped_val_assays: 500
train_membership_missing: 0
val_membership_missing: 0
```

Later manual harness attempts [GPT-R, HO-C]:

- `load_dataset()` called on the raw config, without the resolver, failed with `KeyError: 'mapped_train_assays'`.
- A corrected attempt reached the runner's exact resolution path. It failed only because `cyp/scripts` / `actfound_model_side` was not on the ad hoc harness's `PYTHONPATH`/`sys.path`: `ModuleNotFoundError: No module named 'actfound_model_side'`.

These were harness errors and do not count against the OpenCode-reported preflight. The preflight remains a reported result, not an independent verification. No new dataset result was generated afterwards [GPT-R, HO-C].

---

## 11. Production-Job Attempt History

All jobs ran on CoreHPC. Exact submission commands are not preserved for any job and are not reconstructed here [GPT-R].

### Job 3048559 [GPT-R]

Status: **failed** after about two seconds.

The recorded problem was the initial project-root/path issue in the Slurm wrapper. Full stderr and path details are only partially recoverable.

### Job 3048570 [GPT-R]

Status: **failed** on the frozen graph-qualified TSV header mismatch (`final_assay_size` vs `count_scratch`; see §7).

### Job 3048574 [GPT-R]

Status: **cancelled by Miles**. It had been submitted with an L40S-oriented correction. No training result.

### Job 3048576 [GPT-R]

Status: **failed**. Miles recorded the final error as:

```text
RuntimeError: no train assays were loaded from frozen dataset
```

No scientific training result.

### Job 3052606 [GPT-R, HO-C]

Status: **FAILED**

It waited for CoreHPC GPU priority before starting. Accounting, as provided by Miles:

```text
JobID           3052606
State           FAILED
ExitCode        1:0
Elapsed         00:11:01
Start           2026-11-14T04:46:22
End             2026-11-14T04:57:23
NodeList        ggpu1-09
```

Stderr, as provided by Miles:

```text
RuntimeError: no train assays were loaded from frozen dataset after CHEMBL-prefix assay-ID crosswalk
```

The code path that 3052606 actually executed still used the obsolete CHEMBL-prefix logic. The hash of the source it loaded is UNKNOWN [GPT-R, HO-C].

### Fixes after 3052606 [GPT-R, HO-C]

OpenCode reported modifying the live CoreHPC runner:

`/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project/cyp/scripts/actfound_production_train.py`

The reported changes:

- consume `split_reconciliation_per_id_path` / `split_reconciliation_per_id_sha256`;
- validate the per-ID reconciliation fail-closed;
- map to frozen `assay_key` values;
- remove the obsolete CHEMBL-prefix stripping.

OpenCode then rebuilt the per-ID reconciliation TSV (§8). The config's reconciliation SHA was updated and the real `--validate-only` passed (§9).

Miles, 2026-11-14: "This morning's fixes were made directly on CoreHPC and haven't been transferred to the local repo."

No job-specific hash artifact proves the step from this fixed runner state to the source bytes that 3066823 actually loaded [GPT-R, HO-C].

### Job 3066823: current production job [GPT-R, HO-C]

Status: **last reported RUNNING; current status UNKNOWN**

It was submitted after the live-runner and reconciliation fixes were reported. The exact submission command is not preserved.

It runs:

- the runner `cyp/scripts/actfound_production_train.py`;
- the wrapper `cyp/scripts/actfound_production_train.slurm`;
- the pinned environment `/scratch/user/mhuang8/actfound/envs/actfound_c96f055`;
- the config `cyp/artifacts/actfound/actfound_chembl32_production_config_v1.json`;
- one NVIDIA L40S on CoreHPC `gpu`.

[HO-C]

Last explicitly reported status, from `scontrol show job 3066823` output provided by Miles:

```text
JobState=RUNNING
RunTime=00:49:01
StartTime=2026-11-14T11:20:06
NodeList=ggpu1-11
```

GPU evidence provided by Miles at about 2026-11-14 12:16:58 PST, from `srun --jobid=3066823 --overlap nvidia-smi`:

```text
GPU: NVIDIA L40S
GPU memory: 1742 MiB
GPU utilization: 22%
Power: 118W
Process: .../actfound_c96f055/bin/python
```

Batch stderr observed while the job was running:

- `RuntimeWarning: invalid value encountered in divide`. [HO-C] attributes it to a NumPy correlation calculation.
- PyTorch LR-scheduler warnings, including ``UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.`` [HO-C]. [GPT-R] describes these as warnings about `lr_scheduler.step()` ordering and an explicit epoch parameter. [HO-C] calls them ordering/deprecation warnings.

These were warnings, not a reported termination. The job was still RUNNING at that check.

A check for `startup_config.json`, `progress.jsonl` and `completion.json` came back blank while 3066823 was running. That was an intermediate state and does not imply failure [HO-C]. The exact output directory of 3066823 is UNKNOWN [HO-C].

Unknown for 3066823:

- the exact source hashes it loaded;
- its output directory;
- its completion and result;
- any later status.

No status refresh has been made for this log.

---

## 12. Queue and Scheduler Work [GPT-R]

Job 3052606 spent hours pending on CoreHPC `gpu` priority.

- `sprio -j 3052606` showed approximately `PRIORITY 9355`, `AGE 8`, `FAIRSHARE 9348`, `QOS 0`.
- `sshare -u "$USER" -l` showed `hpc_core / mhuang8 FairShare 0.093478`.
- `sacctmgr show user "$USER" withassoc format=Account,User,QOS,DefaultQOS` showed account `hpc_core`, with a QOS set including `cmnt`, `cmnt_short` and `cpu...`, and default QOS `hpc_core`.

The output established no alternate authorized account or QOS.

Miles explicitly cancelled the obsolete D011 ANIL job array `2979324_[57-1649]` because D011 was no longer being worked on.

---

## 13. gpu-dev1 Diversion — Not Used for Training [GPT-R]

A gpu-dev1 path was considered only because the CoreHPC queue wait became long. It was a proposal, not a decision (§15).

- A bundle was created at `/scratch/user/mhuang8/actfound_gpu_dev1_20261113.tar.gz`. One bundle hash reported during the transfer sequence was `f4d00aea19c407f585268716af8e1429492f5be536f77715bee2afddbf404d8e`.
- The bundle was later repackaged to include the reconciliation artifacts. The hash of the repackaged bundle is UNKNOWN. It transferred and extracted successfully after one incomplete-transfer attempt.
- gpu-dev1 has L40S hardware, but the pinned environment path `/scratch/user/mhuang8/actfound/envs/actfound_c96f055` was not present there at the time. The shell reported Python `3.12.3`.
- Miles explicitly chose not to audit unrelated conda environments.
- No gpu-dev1 ActFound training was started.

The current ActFound run is CoreHPC SLURM job 3066823, not gpu-dev1.

---

## 14. Current State (as of 2026-11-14)

Miles, 2026-11-14 (authoritative on current state):

> "I am already using the live monitor Claude Chat provided for SLURM job 3066823, and the scientific work is waiting."

> "This morning's fixes were made directly on CoreHPC and haven't been transferred to the local repo."

Recorded:

- **Monitor:** a live monitor for job 3066823 is in use, per Miles. This log does not record which command it runs.
- **Work status:** the scientific work is waiting on job 3066823.
- **Job status:** the last reported state is RUNNING on `ggpu1-11` (§11). The current status is UNKNOWN and has not been refreshed.
- **Local checkout:** it does not contain this morning's CoreHPC fixes, per Miles.

[GPT-R] separately describes a `watch -n 10`-based monitor for 3066823 that was established in GPT's conversation. It covered:

- `scontrol show job` fields;
- `sprio` and approximate queue position while pending;
- `sstat -j 3066823.batch`;
- `srun --jobid=3066823 --overlap nvidia-smi` and GPU process detail;
- discovery of `startup_config.json`, `progress.jsonl`, `completion.json`, stdout and stderr;
- stdout/stderr tails;
- `sacct`.

A later simplified check used `srun --jobid=3066823 --overlap nvidia-smi`. [GPT-R] says that no monitor command is asserted to be currently running. Whether the monitor Miles is using now is the one [GPT-R] describes is UNKNOWN.

Recorded next steps when the job is next checked [HO-C]:

- check the actual live state with Slurm accounting/status;
- if it is still running, leave it untouched and inspect existing output/progress artifacts;
- if it completed, collect and verify the frozen production artifacts before any scientific interpretation;
- if it failed, inspect the exact stderr/output and determine whether the failure is operational or implementation-related before making any change.

[HO-C] lists as settled and not to be reopened:

- the frozen ChEMBL32 graph-qualified population;
- the completed 31-case reconciliation;
- the final 33,796-row per-ID reconciliation;
- the live runner's reconciliation mechanism;
- D007-T121 selection;
- D009 P1 closure;
- D010/D011 settled or superseded methodology.

---

## 15. Decisions and Approvals Reported

The items below are reported in GPT's recovered record [GPT-R]. They are recorded as reported, not as newly verified approval evidence. None is a new D-numbered decision, and none has been entered in `notes/Decisions.md` by this log.

1. Use the final frozen graph-qualified ChEMBL32 dataset as the ActFound production input. [GPT-R] calls this an established operational state.
2. Use the completed 33,796-row per-ID reconciliation instead of the obsolete CHEMBL-prefix string mapping. [GPT-R] calls this an implementation correction reported by OpenCode.
3. Constrain the CoreHPC production Slurm job to NVIDIA L40S because the pinned PyTorch stack does not support H100 NVL. [GPT-R] calls this an operational decision.
4. Cancel the obsolete D011 ANIL array `2979324_[57-1649]`. [GPT-R] calls this an explicit decision by Miles.
5. Leave job 3066823 untouched once running. [GPT-R] calls this an explicit operational instruction. [HO-C] likewise says: "Do not modify, cancel, or resubmit it." No later cancellation or resubmission authorization is recorded.

Proposals and recommendations that were not decisions [GPT-R]:

- gpu-dev1 + tmux as an escape hatch from the CoreHPC queue wait;
- creating a dedicated gpu-dev1 ActFound runtime;
- several proposed preflight/test harnesses.

[GPT-R] states that no ActFound scientific acceptance threshold was approved in its conversation.

The absence of other approvals from these sources does not show that anything was unapproved.

---

## 16. Superseded States

These states were replaced during the session. They are not unresolved conflicts.

| Superseded state | Replaced by | Source |
|---|---|---|
| CHEMBL-prefix-stripping assay-ID crosswalk (scan: `train_found 104`, `validation_found 2`; executed by 3052606) | `assay_chembl_id` + descriptor bridge; 33,796-row per-ID reconciliation; live runner consumes per-ID path/SHA | GPT-R, HO-C |
| TSV header expectation `final_assay_size` (job 3048570 failure) | `count_scratch` in `GRAPH_QUALIFIED_COLUMNS` | GPT-R |
| Config per-ID reconciliation SHA `e22fc261c961551128eae3749c3eec50ff798120d2fda4d97eadd549256c2116` | `7ea5ae080ecc52fb95dd293230bd6225f5e09a73622a9ec0e6dec46a02f74645`; `--validate-only` passed | GPT-R (old SHA); GPT-R, HO-C (new SHA) |
| H100 NVL attempt (sm_90 incompatible with pinned PyTorch) | `#SBATCH --gres=gpu:nvidia_l40s:1` | GPT-R (H100 attempt); GPT-R, HO-C (L40S request) |
| `robust_square_error` fixture expectation `2.5` | Corrected expectation; `PASS: pairwise_loss` | GPT-R |
| Adaptation that detached parameters | Dict-based differentiable adaptation preserving `create_graph` | GPT-R, HO-C |
| gpu-dev1 bundle hash `f4d00aea19c407f585268716af8e1429492f5be536f77715bee2afddbf404d8e` | Repackaged bundle including reconciliation artifacts (hash UNKNOWN); gpu-dev1 path not used | GPT-R |

---

## 17. Unresolved Conflicts

The two sources did not genuinely disagree on any fact. The shared history (graph population, reconciliation, hashes, jobs 3052606 and 3066823, D012 quotations, README metrics) was deduplicated and is recorded once with both labels.

Differences in wording or coverage that were preserved and not treated as conflicts:

- The `9,873,276` count is labelled "pregraph rows" [GPT-R] and "processed rows" [HO-C].
- Jobs 3048559, 3048570, 3048574 and 3048576, the queue/scheduler details and the gpu-dev1 diversion appear only in [GPT-R].
- The 16-shard description, the manifest's recorded hashes, D012's approval date, the blank output-file check and the next-step plan appear only in [HO-C].
- The LR-scheduler warning is described as ordering plus explicit-epoch [GPT-R] and as ordering/deprecation [HO-C].

---

## 18. Open Items and Gaps

1. Session start date: UNKNOWN. The 2026-11-12 handoff header is source-reported only.
2. Current live status of job 3066823: UNKNOWN beyond last reported RUNNING on `ggpu1-11`.
3. Whether 3066823 has finished, and whether it produced `startup_config.json`, `progress.jsonl`, `completion.json` or checkpoints: UNKNOWN.
4. Exact output directory of 3066823: UNKNOWN.
5. Exact source-file hashes loaded by 3066823: UNKNOWN.
6. Exact source hashes loaded by 3052606 and earlier jobs: UNKNOWN.
7. Final production-config SHA-256 after the reconciliation-SHA update: UNKNOWN.
8. SHA-256 of `actfound_reported_unresolved_31_resolution.tsv`: UNKNOWN.
9. Final SHA-256 of the graph-qualified merged TSV in the current live state: UNKNOWN.
10. Hash of the repackaged gpu-dev1 bundle: UNKNOWN.
11. Job ID of the H100 NVL attempt: UNKNOWN.
12. D-number authority for the ActFound run: UNKNOWN. No ActFound entry exists in the local working copy of `notes/Decisions.md` as read 2026-11-14.
13. Formal ActFound interpretation/acceptance/advancement rule: UNKNOWN. D012 A/B/C logic is not imported and no rule is invented.
14. Whether the README metric language is the intended interpretation metric set for this run: UNKNOWN.
15. ActFound production training/validation scores and final CYP metrics: none recorded.
16. Exact submission commands and some OpenCode edit/transfer timestamps: not recoverable [GPT-R].
17. Retained handoff `Pasted text(20261112-214104).txt` and decision copy `Decisions(20261112-224142).md`: cited but not accessible in this session.
18. Transfer of this morning's CoreHPC fixes to the local repo: not done, per Miles.

---

## 19. Scientific Boundaries and Non-Claims

- No ActFound production training result, validation score or CYP metric has been recorded.
- This ActFound run is not linked to any D012 arm, and D012 comparison logic is not its interpretation contract.
- CoreHPC live-file hashes are not job-specific provenance for 3066823 or for earlier jobs.
- Local checkout files are not evidence of the CoreHPC state or of what any job loaded.
- The bounded dataset preflight and the reconciliation verification are OpenCode-reported results, not independent verification.
- The gpu-dev1 bundle is not evidence of the code or environment used by 3066823.

---

## 20. Session Status

**OPEN — Tripwire-CYP-005 in progress.**

The scientific work is waiting on CoreHPC SLURM job 3066823. No session close or advance has been declared.

---

## 21. Job 3066823 Completion Report (2026-11-15)

Source: outputs reported by Miles in the Claude Code conversation on 2026-11-15 [Miles 2026-11-15]. Miles confirmed this entry belongs to Session 005: "Yes, log to 005". All evidence is on CoreHPC (`chpc-ucsf-login-vm1.corehpc.ucsf.edu`), not in this local checkout. This section records a later state than §11, §14, §18, §19 and §20; those sections are left as written at 2026-11-14.

No D-number is linked to this run (see §18 item 12). This entry is not a decision and not a session advance.

### Live monitor output

Command: `bash /tmp/am.sh` on the CoreHPC login node. The script's contents are not recorded here. Monitor timestamp: `Sun Nov 15 00:01:59 2026`. Output as reported by Miles, verbatim:

```text
slurm_load_jobs error: Socket timed out on send/recv operation

EPOCH 59/59   batch 2409/2410   completed=60/60
sec/epoch=731   remaining=0h0m   ETA=Sun 00:02

  ep      rmse       r2     R2os    score      train
  52    0.4816   0.4473   0.3257   0.9657     0.3784
  53    0.4789   0.4529   0.3301   0.9740     0.3763
  54    0.4787   0.4517   0.3269   0.9730     0.3735
  55    0.4777   0.4515   0.3319   0.9738     0.3689
  56    0.4767   0.4522   0.3298   0.9755     0.3678
  57    0.4791   0.4541   0.3185   0.9749     0.3654
  58    0.4783   0.4553   0.3297   0.9770     0.3642
  59    0.4795   0.4527   0.3310   0.9732     0.3646

BEST epoch 58   score 0.9770  rmse 0.4783  r2 0.4553   | since best: 1 epoch(s)
delta vs epoch 0:  rmse -0.0555   r2 +0.1121   R2os +0.1316
```

These are monitor-displayed values. They have not been checked against the run's own output artifacts (`progress.jsonl`, `completion.json` or checkpoints). The definitions of the `score`, `R2os` and `train` columns, and which split the `rmse`/`r2`/`R2os` columns refer to, are not recorded here. No acceptance threshold exists for these values (§15, §18 item 13), and none is applied.

### Slurm accounting

Run by Miles on the CoreHPC login node. The time of the query is not recorded. Output verbatim:

```text
$ sacct -j 3066823 --format=JobID,JobName,State,ExitCode,Elapsed
JobID           JobName      State ExitCode    Elapsed
3066823      actfound-+  COMPLETED      0:0   12:16:08
3066823.bat+      batch  COMPLETED      0:0   12:16:08
3066823.ext+     extern  COMPLETED      0:0   12:16:08
3066823.0    nvidia-smi  COMPLETED      0:0   00:00:00
3066823.1    nvidia-smi  COMPLETED      0:0   00:00:01
3066823.2    nvidia-smi  COMPLETED      0:0   00:00:00
3066823.3    nvidia-smi  COMPLETED      0:0   00:00:00
3066823.4    nvidia-smi  COMPLETED      0:0   00:00:00
3066823.5    nvidia-smi     FAILED    255:0   00:00:00
3066823.6    nvidia-smi  COMPLETED      0:0   00:00:29
```

Recorded:

- Job `3066823` and its batch step: **COMPLETED**, exit code `0:0`, elapsed `12:16:08`.
- Step `3066823.5` (`nvidia-smi`): **FAILED**, exit code `255:0`. Cause: **UNKNOWN**, under investigation.
- Assistant hypothesis, not a finding: steps `.0`–`.6` may be overlap steps launched from the login node, for example by `srun --jobid=3066823 --overlap nvidia-smi` (see §11 and §14), rather than steps of the batch script. This has not been verified.

### Provenance

Miles, 2026-11-15:

> "The local script does not match because I had opencode made direct changes on corehpc before I migrated to claude code."

The submitted CoreHPC script has not been diffed against the local script. The local patch has not been made.

### Status and unknowns (as of 2026-11-15)

- CoreHPC output and checkpoint paths for 3066823: NOT YET RECORDED.
- Output integrity checks: NOT YET DONE.
- Cause of the step `3066823.5` FAILED `255:0`: UNKNOWN.
- Diff of the submitted CoreHPC script against the local script: NOT DONE.
- Exact source hashes loaded by 3066823: still UNKNOWN (§18 item 5).

### Planned next steps (not completed)

Miles, 2026-11-15: "examine the run, do integrity checks, review the outputs, resolve what the failure was, and then patch the local script".

Work stopped at: job 3066823 complete per `sacct`; outputs not yet examined or verified.

---

## 22. Job 3066823 Integrity Checks (2026-11-15)

Source: output of read-only checks that Miles ran on the CoreHPC login node (`chpc-ucsf-login-vm1`) and pasted into the Claude Code conversation on 2026-11-15 [Miles 2026-11-15]. All evidence is on CoreHPC, not in this local checkout. This section records a later state than §11, §14, §18, §19, §20 and §21; those sections are left as written.

No D-number is linked to this run (see §18 item 12). This entry is not a decision and not a session advance.

### Slurm step timing

`sacct` for job `3066823` on `ggpu1-11`, all times 2026-11-14:

| Step | Time(s) as reported | State |
|---|---|---|
| job / batch / extern | Start 11:20:06, End 23:36:14 | COMPLETED `0:0` |
| `.0` nvidia-smi | 12:16:58 | COMPLETED |
| `.1` nvidia-smi | 13:44:10–13:44:11 | COMPLETED |
| `.2` nvidia-smi | 13:54:26 | COMPLETED |
| `.3` nvidia-smi | 13:55:27 | COMPLETED |
| `.4` nvidia-smi | 13:56:28 | COMPLETED |
| `.5` nvidia-smi | 13:57:24 | FAILED `255:0` (zero duration) |
| `.6` nvidia-smi | 13:58:36–13:59:05 | COMPLETED |

Observed:

- Step `.0`'s start time, 12:16:58, matches the timestamp of the GPU evidence from `srun --jobid=3066823 --overlap nvidia-smi` recorded in §11.
- Steps `.2`–`.6` are about one minute apart.
- The batch step continued after `.5` failed and completed `0:0`.

Assistant interpretation, not a finding: the nvidia-smi steps are overlap steps launched from the login node by monitoring, not by the batch script. Step `.5` was a transient failure of one such `srun` call and did not affect training. The `slurm_load_jobs ... Socket timed out` message in the 00:01 monitor output (§21) is unrelated to `.5`, because it occurred after the job ended.

The exact cause of the `.5` exit `255` remains **UNKNOWN**. `srun`'s own stderr went to the login terminal and was not captured.

### Stored batch script

`sacct -j 3066823 -B` returned `NONE`. The executed Slurm wrapper cannot be recovered from Slurm accounting.

### Production config (CoreHPC)

| File | SHA-256 |
|---|---|
| `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project/cyp/artifacts/actfound/actfound_chembl32_production_config_v1.json` | `64ea4af51239e2d68838988d933ccd521b3b4b0738f12d8047a41530eb273c94` |

This is the current live file, hashed on 2026-11-15 after the run. The `config_path` in `startup_config.json` is this same path.

### Output directory (CoreHPC)

`/scratch/user/mhuang8/actfound/chembl32/production_training_20261113/actfound_chembl32_production_v1`

- `checkpoints/`: 61 files, `model_0` through `model_59` plus `model_best.pt`. The listing tail was `model_58.pt`, `model_59.pt`, `model_best.pt`.

| File | Size (bytes) | Modified | SHA-256 |
|---|---:|---|---|
| `startup_config.json` | 3248446 | Nov 14 11:20 | `e54f8951856a4e9126e078c88461b0ea15fd5b1d6fe196b779b41480ba41daa5` |
| `completion.json` | 448 | Nov 14 23:36 | `b077c5903ffa0f8bfa07cc8de12bd072ba96d3527f2b3f2ca7e5a5c16991b58b` |
| `progress.jsonl` | 11678402 | Nov 14 23:36 | `92bdd843dc0237da44dee69f519caaa974a84477cb22f20384464193e35d431c` |
| `checkpoints/model_best.pt` | not reported | not reported | `9eedd74ef8de5ba13ccff4ee527e5f361a1e991b2589cb54c48223261dbef2c1` |

`completion.json` contents:

- `status`: `"completed"`
- `best_score`: `0.977011134486706`
- `best_checkpoint`: `.../checkpoints/model_best.pt`
- `completed_at_unix`: `1794728173.7193623`

### Job stdout and stderr (CoreHPC)

Slurm WorkDir is the `/mnt/scratch/...` path.

| File | Size (bytes) | Modified |
|---|---:|---|
| `/mnt/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project/actfound_production_3066823.err` | 1557 | Nov 14 11:31 |
| `/mnt/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project/actfound_production_3066823.out` | 80 | Nov 14 23:36 |

- A grep of `.err` for `traceback|error|nan|killed|oom` returned no matches.
- `.out` tail:

```text
torch.Size([2, 2048])
torch.Size([2, 2048])
FCNNetwork build torch.Size([2, 1])
```

### `progress.jsonl` checks

| Check | Result |
|---|---|
| `dataset_loaded` events | 1 |
| `validation` events | 1 |
| `train_batch` events | 144600 |
| `epoch_complete` events | 60 |
| `epoch_complete` epochs | contiguous 0..59 |
| train batches per epoch | 2410 every epoch |
| non-finite train losses | 0 |
| best by score | epoch 58, score `0.977011134486706` |
| `model_best.pt` stored epoch | 58 |
| `model_best.pt` `upstream_commit` | `c96f0559ea7a2504e797b9d65b9f1af1aa899ddd` |
| pre-training validation (phase `"initial"`) score | `0.34852629081543296` |

The best epoch and score agree with `completion.json` and with the monitor output in §21.

Not cross-checked:

- The `progress.jsonl` validation rows had no keys named `rmse`, `r2` or `R2os` (those lookups printed `None`). The monitor's `rmse`/`r2`/`R2os` column values (§21) have therefore not been cross-checked against `progress.jsonl`. The mapping from monitor columns to `progress.jsonl` keys is **UNKNOWN**.
- The baseline used by the monitor's "delta vs epoch 0" line (pre-training validation or the first `epoch_complete`) is **UNKNOWN**.

### Source provenance

`startup_config.json` `frozen_input_sha256`: `24d9fd297b735843284d82e7d31b85ccd46038f279931db8d5997fb9a096149e`.

The `source_hashes` recorded in `startup_config.json` at job startup **match** the current CoreHPC live files for all six files below. The comparison was run on 2026-11-15. The script compared full hashes but printed only 16-hex-character prefixes. All paths are under `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project/`.

| File | SHA-256 prefix |
|---|---|
| `cyp/scripts/actfound_model_side.py` | `97c1753512d102e1` |
| `cyp/scripts/actfound_production_train.py` | `8a160427afad7283` |
| `external/ActFound/inner_loop_optimizers.py` | `3c9d1eaa7655c260` |
| `external/ActFound/learning_system/system_actfound.py` | `c0d3232760ecbbc3` |
| `external/ActFound/learning_system/system_base.py` | `14544b8173a66279` |
| `external/ActFound/models/meta_neural_network_architectures.py` | `e88de14e09f354e3` |

This establishes which source bytes job 3066823 loaded for these six files. The Slurm wrapper (`.slurm`) and the config are not in `source_hashes`, so the executed wrapper remains unverified.

### Status after these checks (as of 2026-11-15)

- Output directory and checkpoints of 3066823: recorded above.
- Source hashes loaded by 3066823: established for the six files above. Wrapper and config are not covered.
- Executed Slurm wrapper: not recoverable from accounting, and unverified.
- Cause of step `.5` FAILED `255:0`: UNKNOWN.
- Monitor `rmse`/`r2`/`R2os` mapping and "delta vs epoch 0" baseline: UNKNOWN.
- Diff of the CoreHPC scripts against the local scripts: NOT DONE.

### Planned next step (not completed)

Miles, 2026-11-15: "lets compare scripts and patch the differences locally."

Work stopped at: read-only integrity checks of 3066823 done. The script comparison and local patch have not been started.

---

## 23. Local Repo Sync and Test Runs (2026-11-15)

Source: work done and outputs reported in the Claude Code conversation on 2026-11-15 [Miles 2026-11-15]. Subsections A and B describe this local checkout. Subsections C and D describe CoreHPC runs by Miles, and that evidence is on CoreHPC, not in this checkout. This section records a later state than §6, §8, §9, §18 and §22; those sections are left as written.

No D-number is linked to this work (see §18 item 12). This entry is not a decision and not a session advance.

### A. Local repo sync from CoreHPC

Approved by Miles: "Copy and remove the temp staging folder from downloads".

- Bundle (CoreHPC): `/scratch/user/mhuang8/actfound_local_sync_20261115.tar.gz`, SHA-256 `ed2c8c8f646b4c7f8d40aab84b737336768f06b0037ebc67c1d279a582c86da6`.
- Downloaded by the established route: `scp -o ProxyJump=mhuang8@chpc-ucsf-bastion-vm1.corehpc.ucsf.edu`.
- 10 files were copied byte-identical into the local repo. `shasum -a 256 -c` against the bundle's `SHA256SUMS.txt` returned OK for all 10.

| File (local repo, as synced) | SHA-256 |
|---|---|
| `cyp/scripts/actfound_model_side.py` | `97c1753512d102e1198bfad2ab4b601749cb1d57b6a1a3a1ad847be83214ee45` |
| `cyp/scripts/actfound_production_train.py` | `8a160427afad728348b572b9e9e432ad40493f2ad5e129594e4ee45c41258dc4` |
| `cyp/scripts/actfound_production_train.slurm` | `70f76c0ed92e5afcb2fd516c36c78221771b373768fcd201c6e4981b5075af14` |
| `cyp/tests/test_actfound_production_train.py` | `792868b68266d6b22ae999b9ee0736ca93bfd14c3f3d55c912dc8bed88f83a09` (later modified locally, see B) |
| `cyp/artifacts/actfound/actfound_chembl32_production_config_v1.json` | `64ea4af51239e2d68838988d933ccd521b3b4b0738f12d8047a41530eb273c94` |
| `cyp/artifacts/actfound/actfound_split_reconciliation_per_id.tsv` | `7ea5ae080ecc52fb95dd293230bd6225f5e09a73622a9ec0e6dec46a02f74645` |
| `cyp/artifacts/actfound/actfound_split_reconciliation_summary.json` | `86c73c4f5361de8a6fd238892129f7927fceb37e13950955af1397a5550e5fc5` |
| `cyp/artifacts/actfound/actfound_reported_unresolved_31_resolution.tsv` | `dced5472fb55d8724e4faab92d01cc9fa86bc06c168aab3893b7032df534f59b` |
| `cyp/artifacts/actfound/_tmp_recon.py` | `057e0b47aea2714028fcbfbaea0fe42440b57ccbdbf3978ca8c70c34a8327879` |
| `cyp/artifacts/actfound/_tmp_recon_fast.py` | `89b33aa868d852f1f8b57de38f2147cea5b2e5428bab7ea6cabf8c2754440dce` |

- The two script hashes (`actfound_model_side.py`, `actfound_production_train.py`) equal the job 3066823 startup source hashes (§22).
- The `.slurm` wrapper hash is the current CoreHPC file. It is not evidence of the wrapper 3066823 executed (§22).
- The local Downloads staging folder was moved to `~/.Trash`.
- Nothing was committed.

### B. Local test fix: `cyp/tests/test_actfound_production_train.py`

Approved by Miles: "Apply it".

Changes:

- adds a per-ID reconciliation fixture;
- adds `train_mapped`/`validation_mapped` to the summary fixture, with mapped-key assertions;
- changes the wrapper assertion to `--gres=gpu:nvidia_l40s:1`.

New local SHA-256: `ef99704a2a04773b7179bd3d2d1507e16a52164d54e17bdbd8f142ce5fc88eca`.

Local run (miniforge base Python on the Mac):

```text
python -m pytest -q cyp/tests/test_actfound_production_train.py cyp/tests/test_actfound_model_side.py
```

| State | Result |
|---|---|
| Before the fix | 5 failed, 2 passed, 1 skipped |
| After the fix | 7 passed, 1 skipped |

The skip is the model-side test file, because torch is not installed locally. The CoreHPC copy of this test file is still the old version.

### C. CoreHPC model-side test run

Run by Miles on the CoreHPC login node in the pinned env `/scratch/user/mhuang8/actfound/envs/actfound_c96f055` (Python 3.9). pytest 8.3.5 was installed with `pip install --target /scratch/user/mhuang8/tools/pytest_actfound_py39` and loaded through `PYTHONPATH` only. The pinned env was not modified.

```text
python -m pytest -q -p no:cacheprovider cyp/tests/test_actfound_model_side.py
```

Result: **4 failed, 10 passed** in 2.72s.

| Test | Failure |
|---|---|
| `test_dataset_adapter_preserves_upstream_representation_and_filters` | `assert '20_nM_IC50' == 'CHEMBLASSAY_nM_IC50'` |
| `test_pairwise_loss_behavior_matches_formula` | `robust_square_error` returned `0.5`; the test expected `1.0` |
| `test_target_adaptation_inputs_outputs` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` (in `adapt_final_linear_params` → `torch.autograd.grad`) |
| `test_tiny_end_to_end_qualification` | same `RuntimeError` as above |

### D. Read-only diagnostic probe (CoreHPC)

Run by Miles on CoreHPC. Output verbatim. The first three lines are `head -3 | cut -f1,3,9,10,13,14` of the frozen graph-qualified TSV.

```text
assay_key       assay_id        standard_units  standard_type   assay_chembl_id count_scratch
37_nM_Kd        37      nM      Kd      CHEMBL615149    26
37_nM_Kd        37      nM      Kd      CHEMBL615149    26
H2 robust_square_error: 0.5
H3 weight.requires_grad: False
H3 with requires_grad: (1, 3) (1.6142135858535767, 1.6042135953903198)
H1 swapped fixture assay_id: ['CHEMBLASSAY_nM_IC50']
```

### E. Diagnosis

This is the assistant's diagnosis, supported by the probe in D. It is not a decision.

- **Failure 1:** the test fixture has its `assay_chembl_id` and `count_scratch` values swapped (`"20"` vs `"CHEMBLASSAY"`). The real frozen TSV holds CHEMBL IDs in `assay_chembl_id`. Swapping the fixture values gives the expected ID.
- **Failure 2:** `robust_square_error` is line-for-line identical to upstream `external/ActFound/learning_system/system_base.py` `robust_square_error`. `0.5` is the correct value for the test inputs, and the test's expected `1.0` is wrong.
- **Failures 3–4:** the weight tensor from `_tiny_episode()` lacks `requires_grad`. With `requires_grad_(True)` the adaptation runs. The production runner does not call `adapt_final_linear_weight`.
- **Conclusion:** all four are test-file defects. There is no evidence of a defect in the code that job 3066823 ran.

Record note: §6 reports, from [GPT-R], that an earlier `robust_square_error` fixture expected `2.5`, that the expectation was corrected and that a later manual qualification reported `PASS: pairwise_loss`. The CoreHPC test file run in C expects `1.0`. How the earlier corrected expectation relates to the current `1.0` is not recorded and is **UNKNOWN**.

A test-only fix for `cyp/tests/test_actfound_model_side.py` has been proposed and is pending Miles's approval. It has **not** been applied.

### Status (as of 2026-11-15)

- Hashes now recorded that §8, §9 and §18 list as UNKNOWN: the production config, `64ea4af5…` (also §22), and `actfound_reported_unresolved_31_resolution.tsv`, `dced5472…`. These are the current CoreHPC files as synced on 2026-11-15.
- Local `test_actfound_production_train.py`: fixed and passing locally. The CoreHPC copy is not updated.
- `test_actfound_model_side.py`: 4 failures on CoreHPC, diagnosed as test defects. The fix is not approved and not applied.
- Nothing committed.

Work stopped at: the model-side test-only fix is awaiting Miles's approval.

---

## 24. Model-Side Test Fix, CoreHPC Upload and Test Run (2026-11-15)

Source: work done and outputs reported in the Claude Code conversation on 2026-11-15 [Miles 2026-11-15]. Subsections A and B describe this local checkout. Subsection C describes a CoreHPC run by Miles; that evidence is on CoreHPC, not in this checkout. This section records a later state than §23, which is left as written. It resolves the §23 status item that the model-side test-only fix was pending approval and not applied.

No D-number is linked to this work. This entry is not a decision and not a session advance.

### A. Local test fix: `cyp/tests/test_actfound_model_side.py`

Approved by Miles: "Yes, apply".

Test file only. `cyp/scripts/actfound_model_side.py` is untouched and still `97c1753512d102e1198bfad2ab4b601749cb1d57b6a1a3a1ad847be83214ee45`, the job 3066823 startup hash.

Changes:

- `_write_graph_fixture`: default `assay_key` changed from `"CHEMBLASSAY_nM_IC50"` to `"11_nM_IC50"`; the `assay_chembl_id`/`count_scratch` values swapped back to `"CHEMBLASSAY"`/`"20"`;
- `_tiny_episode`: weight created with `requires_grad=True`;
- pairwise-loss test: expected value changed to `mean([0.0, 1.0]) = 0.5`, with an arithmetic comment.

New local SHA-256: `6f5fa3d912dc7b1673c5822f5318e55b641574c6d138afe5e42acb34bb7401ff`.

### B. Upload bundle (local)

- Bundle: `~/Downloads/actfound_tests_upload_20261115/actfound_tests_20261115.tar.gz`, SHA-256 `8d2491f3d7fdd7dba0b8b8d900346e5a2cc2e7e7af2f692826b2f6f141e050ba`.

| File in bundle | SHA-256 |
|---|---|
| `cyp/tests/test_actfound_model_side.py` | `6f5fa3d912dc7b1673c5822f5318e55b641574c6d138afe5e42acb34bb7401ff` |
| `cyp/tests/test_actfound_production_train.py` | `ef99704a2a04773b7179bd3d2d1507e16a52164d54e17bdbd8f142ce5fc88eca` |

Commands were provided for: `scp` upload via ProxyJump bastion to `/scratch/user/mhuang8/`; on CoreHPC, a bundle hash check, backup of the prior CoreHPC test files as `cyp/tests/*.orig_20261115`, extraction into `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project`, and a hash print.

### C. CoreHPC test run

Command provided:

```text
PYTHONPATH=/scratch/user/mhuang8/tools/pytest_actfound_py39 /scratch/user/mhuang8/actfound/envs/actfound_c96f055/bin/python -m pytest -q -p no:cacheprovider cyp/tests/test_actfound_model_side.py cyp/tests/test_actfound_production_train.py
```

Result, Miles's words: "all tests pass". The output was not pasted.

Not reported (**UNKNOWN**):

- the pass count;
- the hashes printed by the CoreHPC hash check, so the CoreHPC test-file hashes after upload are not independently confirmed in this record;
- whether the `.orig_20261115` backups were created.

### Status (as of 2026-11-15)

- `test_actfound_model_side.py` fix applied locally (`6f5fa3d9…`); `actfound_model_side.py` unchanged (`97c17535…`).
- Both test files reported passing on CoreHPC by Miles ("all tests pass"); counts and post-upload CoreHPC hashes not reported.
- Nothing committed in the local repo.

Work stopped at: CoreHPC test run reported passing; nothing committed.

---

## 25. Job 3066823 Validation Trajectory from `progress.jsonl` (2026-11-15)

Source: output of a read-only extraction that Miles ran on CoreHPC and pasted into the Claude Code conversation on 2026-11-15 [Miles 2026-11-15]. All evidence is on CoreHPC, not in this local checkout. This section records a later state than §21 and §22; those sections are left as written. It resolves two §22 UNKNOWNs: the monitor column mapping and the "delta vs epoch 0" baseline.

No D-number is linked to this run (see §18 item 12). No acceptance rule exists for these values (§1, §18), and none is applied. This entry is not a decision and not a session advance.

### Extraction

- File (CoreHPC): `/scratch/user/mhuang8/actfound/chembl32/production_training_20261113/actfound_chembl32_production_v1/progress.jsonl`, SHA-256 `92bdd843dc0237da44dee69f519caaa974a84477cb22f20384464193e35d431c` (§22).
- Rows printed: every `validation` and `epoch_complete` event.
- Fields printed: `epoch`, phase, `mean_rmse`, `mean_r2`, `mean_R2os`, `score`, `assay_count`, `mean_train_loss`.

Output verbatim:

```text
ep phase rmse r2 R2os score n train
0 initial 0.7831 0.1316 -0.8355 0.3485 500 None
0 end 0.5349 0.3406 0.1993 0.8057 500 0.8983
1 end 0.5232 0.3639 0.2225 0.8406 500 0.7796
2 end 0.515 0.3761 0.2426 0.8611 500 0.7443
3 end 0.5105 0.3848 0.2573 0.8742 500 0.7172
4 end 0.5068 0.3915 0.2644 0.8847 500 0.6922
5 end 0.5031 0.4041 0.2726 0.9011 500 0.6676
6 end 0.5028 0.406 0.267 0.9032 500 0.6507
7 end 0.5 0.4179 0.2853 0.918 500 0.6352
8 end 0.4983 0.4194 0.2767 0.9211 500 0.6233
9 end 0.4994 0.4163 0.2761 0.9169 500 0.6119
10 end 0.4974 0.4167 0.2794 0.9193 500 0.6
11 end 0.4942 0.4233 0.294 0.9291 500 0.5965
12 end 0.4937 0.4206 0.2918 0.9269 500 0.5824
13 end 0.4961 0.4218 0.2855 0.9257 500 0.5696
14 end 0.4902 0.4331 0.3021 0.9429 500 0.5673
15 end 0.4913 0.4343 0.3005 0.943 500 0.557
16 end 0.4926 0.4267 0.2953 0.9341 500 0.545
17 end 0.4919 0.4311 0.2972 0.9391 500 0.5394
18 end 0.4919 0.4307 0.305 0.9389 500 0.5304
19 end 0.4876 0.4351 0.3111 0.9475 500 0.523
20 end 0.4887 0.4354 0.3109 0.9468 500 0.5218
21 end 0.4893 0.4349 0.3015 0.9456 500 0.5121
22 end 0.4861 0.437 0.3056 0.9509 500 0.5028
23 end 0.4847 0.446 0.3184 0.9612 500 0.4992
24 end 0.4861 0.4391 0.3102 0.953 500 0.4895
25 end 0.4853 0.4415 0.3103 0.9561 500 0.4854
26 end 0.4862 0.4361 0.3045 0.95 500 0.4804
27 end 0.4865 0.439 0.3069 0.9525 500 0.4752
28 end 0.4863 0.4402 0.3075 0.954 500 0.4695
29 end 0.486 0.44 0.3109 0.954 500 0.4645
30 end 0.4872 0.4352 0.3079 0.948 500 0.4583
31 end 0.4854 0.4422 0.3126 0.9568 500 0.4557
32 end 0.4833 0.4437 0.3178 0.9604 500 0.4501
33 end 0.4846 0.4457 0.3128 0.9611 500 0.446
34 end 0.4856 0.4453 0.3096 0.9597 500 0.4416
35 end 0.4834 0.4472 0.3169 0.9638 500 0.4355
36 end 0.485 0.444 0.3119 0.959 500 0.4343
37 end 0.485 0.4413 0.313 0.9563 500 0.4274
38 end 0.4841 0.4438 0.3133 0.9597 500 0.4232
39 end 0.4828 0.4428 0.3222 0.9601 500 0.422
40 end 0.4812 0.4452 0.322 0.964 500 0.4157
41 end 0.4823 0.4497 0.3237 0.9674 500 0.4138
42 end 0.4844 0.4487 0.3197 0.9643 500 0.4079
43 end 0.4805 0.4525 0.3308 0.972 500 0.4068
44 end 0.482 0.45 0.3262 0.968 500 0.4007
45 end 0.4827 0.4513 0.3223 0.9686 500 0.3961
46 end 0.4806 0.4534 0.3297 0.9728 500 0.3953
47 end 0.4811 0.4492 0.3274 0.9682 500 0.3926
48 end 0.4791 0.4551 0.3268 0.976 500 0.3878
49 end 0.4799 0.4525 0.3294 0.9726 500 0.386
50 end 0.4787 0.4511 0.3286 0.9724 500 0.383
51 end 0.4789 0.4472 0.3268 0.9684 500 0.3799
52 end 0.4816 0.4473 0.3257 0.9657 500 0.3784
53 end 0.4789 0.4529 0.3301 0.974 500 0.3763
54 end 0.4787 0.4517 0.3269 0.973 500 0.3735
55 end 0.4777 0.4515 0.3319 0.9738 500 0.3689
56 end 0.4767 0.4522 0.3298 0.9755 500 0.3678
57 end 0.4791 0.4541 0.3185 0.9749 500 0.3654
58 end 0.4783 0.4553 0.3297 0.977 500 0.3642
59 end 0.4795 0.4527 0.331 0.9732 500 0.3646
```

### Cross-checks (verified arithmetic)

- **Column mapping (resolves the §22 UNKNOWN).** The monitor's `rmse`/`r2`/`R2os` columns (§21) are `progress.jsonl` `mean_rmse`/`mean_r2`/`mean_R2os`. The monitor rows for epochs 52–59 (§21) match the rows above exactly.
- **"Delta vs epoch 0" baseline (resolves the §22 UNKNOWN).** The §21 monitor line (`rmse -0.0555`, `r2 +0.1121`, `R2os +0.1316`) equals epoch 59 minus the epoch-0 end-of-epoch row, within rounding: `0.4795 - 0.5349`, `0.4527 - 0.3406`, `0.331 - 0.1993`. The baseline is therefore the end of epoch 0, not the pre-training `initial` row. The comparison epoch is the last epoch (59), not the best (58).
- `assay_count` is 500 on every row.
- `mean_train_loss` decreases monotonically from `0.8983` (epoch 0) to `0.3642` (epoch 58), then is `0.3646` at epoch 59.

### Metric definitions (runner code)

From `cyp/scripts/actfound_production_train.py` `metric_from_task` and `evaluate_validation`, the version job 3066823 loaded (startup source hash `8a160427afad7283…`, §22; local synced copy `8a160427…`, §23 A):

- Each validation assay is split into support and query sets. Metrics are computed on the query set.
- `rmse`: root mean squared error.
- `r2`: squared Pearson correlation. Set to 0 if the correlation is NaN or below 0.
- `R2os`: `1 - SSE/SS`, where SS is taken relative to the support-set mean.
- Each metric is averaged over the 500 validation assays.
- `score = mean_r2 - mean_rmse + 1`.
- One support/query draw per assay (`repeat_cnt=0`).
- The best checkpoint is selected by `score` on these same 500 validation assays.

### Assistant observations (not findings or decisions)

- Validation improved rapidly over epochs 0–7, more slowly up to about epoch 23, and gradually after that.
- Over epochs 43–59 the score ranges from `0.9657` to `0.9770`, with epoch-to-epoch swings of up to about 0.01. Epoch 58's best score is therefore within that variation of epochs 48 and 53–56.
- Validation metrics plateau while training loss keeps decreasing slowly. No validation decline is observed.
- The best-epoch metrics are selected and reported on the same validation set, so they are optimistic. No held-out evaluation exists.

### Status (as of 2026-11-15)

- Monitor column mapping and "delta vs epoch 0" baseline: resolved above.
- Still open from §22: the executed Slurm wrapper is unverified, and the cause of the step `.5` FAILED `255:0` is UNKNOWN.
- No held-out evaluation of 3066823 exists.

Work stopped at: the validation trajectory was extracted from `progress.jsonl` and cross-checked against the monitor output.

---

## 26. Job 3066823 Checkpoint, Config, Runtime and Manifest Probe (2026-11-15)

Source: output of a read-only probe that Miles ran on the CoreHPC login node with the pinned Python `/scratch/user/mhuang8/actfound/envs/actfound_c96f055` and pasted into the Claude Code conversation on 2026-11-15 [Miles 2026-11-15]. The probed artifacts are on CoreHPC, not in this local checkout. Earlier sections are left as written.

No D-number is linked to this run (see §18 item 12). This entry is not a decision and not a session advance.

### A. Probe

The probe inspected:

- the graph-qualified manifest (CoreHPC) `/scratch/user/mhuang8/actfound/chembl32/graph_production_20261113/actfound_chembl32_graph_qualified.manifest.json`, searching its keys for `firewall`/`openadmet`/`leak`/`exclu`/`protect`;
- the config stored in `startup_config.json` against the current config file;
- the startup runtime recorded for the job;
- `model_best.pt` against `checkpoints/model_58.pt`.

Output verbatim:

```text
manifest top-level keys: ['artifact_type', 'graph_qualified_assays', 'graph_qualified_rows', 'graph_size_rejected_assays', 'graph_valid_count_rejected_assays', 'merged_output', 'merged_output_sha256', 'pair_count_total', 'shard_count', 'shard_summary_sha256', 'source_pregraph_assays', 'source_pregraph_rows']
startup config vs current config file, differing keys: ['config_path', 'mapped_train_assays', 'mapped_val_assays', 'split_reconciliation', 'split_reconciliation_per_id']
runtime: {'cuda_available': True, 'cuda_device_count': 1, 'cuda_device_name': 'NVIDIA L40S', 'python': '3.9.18', 'rdkit': '2023.03.1', 'torch': '1.12.1'}
epochs: 58 58 | upstream: c96f0559ea7a2504e797b9d65b9f1af1aa899ddd
best == model_58 weights: True | inner_lrs equal: True
```

### B. Findings (verified by inspection of the output above)

- **Best checkpoint.** `model_best.pt` and `model_58.pt` have identical `regressor_state_dict` tensors and identical `inner_lrs`. Both record epoch 58 and upstream commit `c96f0559ea7a2504e797b9d65b9f1af1aa899ddd`. The best-validation checkpoint is epoch 58.
- **Config.** The only keys that differ between the `startup_config.json` config and the current config file (SHA-256 `64ea4af5…`, §22) are keys the runner adds during resolution: `config_path`, `mapped_train_assays`, `mapped_val_assays`, `split_reconciliation`, `split_reconciliation_per_id` (see `validate_and_resolve_config` in `cyp/scripts/actfound_production_train.py`). Job 3066823 therefore ran with the current config file's content.
- **Runtime.** Python 3.9.18, torch 1.12.1, RDKit 2023.03.1, one NVIDIA L40S (`cuda_device_count` 1).
- **Manifest.** No top-level key of the graph-qualified manifest mentions firewall, OpenADMET, leakage, exclusion or protection.

### C. Related code facts (local copies)

These local files are byte-identical to the CoreHPC copies per §23.

- `cyp/scripts/actfound_chembl32_projection.py` line 952 docstring: "The project's OpenADMET identity firewall is deliberately NOT applied here."
- `cyp/scripts/actfound_chembl32_projection.py` and `cyp/scripts/actfound_pregraph_stream.py` write `"openadmet_firewall_applied": False` in their summaries.
- The production config (`cyp/artifacts/actfound/actfound_chembl32_production_config_v1.json`) and the runner (`cyp/scripts/actfound_production_train.py`) contain no exclusion step.

### D. Assistant conclusion (not a finding or decision)

No OpenADMET or leakage exclusion is recorded anywhere in the chain from ChEMBL32 to job 3066823's training data. Whether the training corpus overlaps protected or D007 validation identities is **UNKNOWN**. It has not been measured.

### E. Miles's statement of the intended study (2026-11-15)

Miles, 2026-11-15, verbatim:

> "The Nov 12 conversation already established the intended ActFound study: leakage-clean ChEMBL pretraining, upstream ActFound adaptation using each frozen D007 cell's training molecules only, evaluation on its validation molecules, and the unchanged 150-cell comparison against frozen T121. Whole-package claims only. D012 remains background, not this run's arm definition."

Miles, 2026-11-15, verbatim instructions:

> "Don't reopen the architecture or adaptation-path menu, introduce a published-fidelity pass/fail gate, or reselect epochs through extra validation draws."

This is recorded as Miles's statement. The original 2026-11-12 record is not accessible: §3 notes that the 2026-11-12 handoff file is inaccessible, and no accessible record contains this contract. No decision record was created for it.

### F. Unresolved tension

The term "leakage-clean" in Miles's statement (E) is in apparent tension with the code facts (C) and the assistant conclusion (D): no leakage exclusion is recorded in the chain that produced job 3066823's training data. **Unresolved; resolution is pending Miles's decision.**

### Status (as of 2026-11-15)

- Best checkpoint = epoch 58; job 3066823 ran with the current config file's content; runtime recorded above.
- No leakage or OpenADMET exclusion is recorded for the training chain. Overlap with protected or D007 validation identities: UNKNOWN.
- Tension between "leakage-clean" and the recorded chain: pending Miles's decision.
- Still open from §22 and §25: the executed Slurm wrapper is unverified; the cause of the step `.5` FAILED `255:0` is UNKNOWN; no held-out evaluation of 3066823 exists.

Work stopped at: the read-only checkpoint/config/runtime/manifest probe was recorded, and Miles stated the intended study contract.

---

## 27. Training-Corpus Overlap Checks: D012 Identity Audit vs ActFound Split, Corpus-Chain Trace, CoreHPC Assay Rows (2026-11-15)

Sources: (A) and (B) are read-only checks the assistant ran in this local checkout on 2026-11-15. (C) is the output of a read-only trace Miles ran on CoreHPC on 2026-11-15 and pasted into the Claude Code conversation [Miles 2026-11-15]. The CoreHPC artifacts in (C) are on CoreHPC, not in this local checkout. Earlier sections are left as written.

This entry records read-only checks only. It is not a decision, has no D-number, and is not a session advance.

### A. Local check: D012 identity-audit matches vs ActFound split (assistant, 2026-11-15)

Inputs (local checkout):

- `qualification/cyp_external_task_qualification_20261109/identity_audit/cross_source_matches.tsv` (SHA-256 `6613d182fb4492b4e79f71dcd29d15c7c73f689fe15087d7b1a66aeb7967a8c5`). This is the D012 bounded identity audit: 55 full-Standard-InChIKey cross-source match groups over 176 ChEMBL CYP candidate assays / 4,822 ChEMBL records. Frozen identity runtime: Python 3.12.14, RDKit 2026.03.6, InChI 1.07.3.
- `cyp/artifacts/actfound/actfound_split_reconciliation_per_id.tsv` (SHA-256 `7ea5ae08…`; 33,296 train + 500 validation).

Each match was checked by `assay_chembl_id` against the split file. Result, by the OpenADMET partition of the matched molecule:

| OpenADMET partition | ChEMBL assay in ActFound split | Count |
|---|---|---|
| train | train split | 2 |
| train | not in split | 31 |
| validation | not in split | 8 |
| internal_test | not in split | 14 |

The two train-partition groups whose ChEMBL assay is in ActFound's train split:

- `AQHHHDLHHXJYJD-UHFFFAOYSA-N` via `CHEMBL832148`
- `NCDNCNXCDXHOMX-XGKFQTDJSA-N` via `CHEMBL1100026` and `CHEMBL3118928`

Scope limit: the audit covers only the 176 D012 CYP assays. Overlap across the full training corpus is **UNKNOWN**.

### B. Local code trace: corpus chain (assistant, 2026-11-15)

- No stage of the corpus chain reads any InChIKey, OpenADMET or firewall input: `actfound_chembl32_projection.py` → `actfound_pregraph_stream.py` → `actfound_graph_stage.py` → merged TSV → `actfound_production_train.py`.
- The projection and pregraph summaries write `"openadmet_firewall_applied": False`.
- No firewall artifact exists in this repo or among the CoreHPC project files changed since 2026-11-12 (the §23 listing).
- The corpus was regenerated from the ChEMBL32 SQLite release, not from an authors' released processed artifact. The reason for this is not recorded.

### C. CoreHPC read-only trace (Miles, 2026-11-15)

Miles ran:

1. `find` plus a JSON summary scan under `/scratch/user/mhuang8/actfound/chembl32`;
2. `awk` on `graph_production_20261113/actfound_chembl32_graph_qualified.tsv` for assays `CHEMBL832148`, `CHEMBL1100026` and `CHEMBL3118928`.

The pasted output contained only:

- two JSON summary lines: `production_training_20261113/.../completion.json` and `startup_config.json` (startup `frozen_input_sha256` `24d9fd297b735843284d82e7d31b85ccd46038f279931db8d5997fb9a096149e`);
- the `awk` rows.

The `find` listing and any graph-production or pregraph manifest lines were not in the paste. They are **NOT YET REVIEWED** (the paste may have been truncated). This does not mean they are absent.

`awk` result: the frozen training TSV (`24d9fd29…`) contains rows for all three assays.

| Assay | `assay_key` | Rows shown |
|---|---|---|
| `CHEMBL1100026` | `621465_nM_IC50` | 38 |
| `CHEMBL3118928` | `1291572_nM_IC50` | 31 |
| `CHEMBL832148` | `305983_nM_IC50` | 11 |

The output was capped by `head -80`.

Relevant rows, verbatim:

```text
621465_nM_IC50 88100 CHEMBL1100026 CC(C)c1nc(CN(C)C(=O)N[C@H](C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1
1291572_nM_IC50 88100 CHEMBL3118928 CC(C)c1nc(CN(C)C(=O)N[C@H](C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1
305983_nM_IC50 1370 CHEMBL832148 CC(C)NCC(O)COc1cccc2ccccc12
```

### D. Assistant observation (not an identity finding)

- The SMILES for molregno 1370 appears to be propranolol (formula C16H21NO2). This is consistent with the InChI of `AQHHHDLHHXJYJD-UHFFFAOYSA-N` in the audit.
- The SMILES for molregno 88100 appears to be ritonavir, a candidate for `NCDNCNXCDXHOMX-XGKFQTDJSA-N`.
- Identity adjudication in the frozen identity environment has **NOT** been performed. How the SMILES look is suggestive only.
- Load-time row filters (for example `standard_relation`) were not checked for these rows.

### Status (as of 2026-11-15)

- The firewall flag is `False` at projection and pregraph. No later stage has a firewall step.
- Two audited OpenADMET training-partition molecules are plausibly present in training-split assay rows. Their identity is unadjudicated.
- Full-corpus overlap: UNKNOWN.
- The stage-to-stage hash chain has not yet been reviewed.
- No retraining, rescoring, CYP evaluation or decision entry.

Work stopped at: the D012 audit-vs-split check, the corpus-chain code trace and the CoreHPC assay-row trace were recorded. The CoreHPC `find` listing and the manifest lines are not yet reviewed.

---

## 28. Training-Corpus Lineage: CoreHPC `find` Listing and Manifest Chain (2026-11-15)

Source: the top portion of the same CoreHPC read-only trace recorded in §27 C, pasted by Miles into the Claude Code conversation on 2026-11-15 [Miles 2026-11-15]. The artifacts are on CoreHPC, not in this local checkout. This is a later state than §27 C: the `find` listing and the manifests that §27 C marked **NOT YET REVIEWED** are reviewed here, except the pregraph summary contents. Earlier sections are left as written.

This entry records read-only checks only. It is not a decision, has no D-number, and is not a session advance.

### A. Files found (CoreHPC)

`find` under `/scratch/user/mhuang8/actfound/chembl32` (maxdepth 3; `*.json` or names containing firewall/inchikey/exclu/prohibit), with timestamps:

| Timestamp | File |
|---|---|
| 2026-11-12 23:02 | `chembl32_actfound_unit_eligible.tsv.summary.json` |
| 2026-11-13 00:23 | `pregraph_production_20261113/chembl32_actfound_pregraph.tsv.summary.json` |
| 2026-11-13 00:34 | `graph_shards_20261113/shard_manifest.json` |
| 2026-11-13 00:40–00:42 | `graph_production_20261113/shard_000..015_summary.json` |
| 2026-11-13 00:46 | `graph_production_20261113/actfound_chembl32_graph_qualified.manifest.json` |

Also present: `production_training` `completion.json` and `startup_config.json`.

No file name contains firewall, inchikey, exclu or prohibit.

### B. Recorded lineage (self-reported hashes in summaries; not re-hashed)

- **Projection.** `chembl32_actfound_unit_eligible.tsv.summary.json`: artifact_type `ActFound_ChEMBL32_unit_eligible_projection_v1`; `"openadmet_firewall_applied": false`; source_database `/scratch/user/mhuang8/actfound/chembl32/extracted/chembl_32/chembl_32_sqlite/chembl_32.db` (24,137,826,304 bytes); output `/scratch/user/mhuang8/actfound/chembl32/chembl32_actfound_unit_eligible.tsv` SHA-256 `bd804a6dabd68c39221e459b9f61d7c3039ffc75b156cd2d141a0c59ea9fe4c0`; script_sha256 `71161e52a44278e1666b968d52e827c94b23682f4b1c9fa8a8801ed450fd0c54`.
- **Pregraph.** The summary file exists, but its contents were not in either pasted portion. **NOT YET REVIEWED.** The link `bd804a6d…` → pregraph TSV and the pregraph firewall flag are therefore not yet confirmed from CoreHPC output.
- **Shard split.** `graph_shards_20261113/shard_manifest.json`: artifact_type `ActFound_graph_stage_shards_v1`; input `/scratch/user/mhuang8/actfound/chembl32/pregraph_production_20261113/chembl32_actfound_pregraph.tsv` SHA-256 `43dfd672e790ba3f92bd1e17b5902a44aec88911a5bbf0db26f92d0e192db076`; input_assays 67,985; input_rows 9,873,276. Per-shard output hashes were not in the printed (key-filtered) output.
- **Graph stage.** 16 shard summaries (`ActFound_graph_stage_shard_result_v1`), each with input `graph_shards_20261113/shard_NNN.tsv` + input_sha256 and output `graph_production_20261113/shard_NNN_result.tsv` + output_sha256. Summed across shards (assistant arithmetic, verified): processed_assays 67,985; processed_rows 9,873,276; graph_qualified_assays 54,732; final_rows 2,773,995. These exactly match the shard manifest input and the merged manifest.
- **Merge.** `actfound_chembl32_graph_qualified.manifest.json`: artifact_type `ActFound_ChEMBL32_graph_qualified_population_v1`; graph_qualified_assays 54,732; graph_qualified_rows 2,773,995; graph_size_rejected 169; graph_valid_count_rejected 13,084; merged_output_sha256 `24d9fd297b735843284d82e7d31b85ccd46038f279931db8d5997fb9a096149e`; shard_summary_sha256 per shard.
- **Training.** `startup_config.json` frozen_input_sha256 `24d9fd29…` = merged_output_sha256.

### C. Assistant conclusion (assistant analysis, not a decision)

The merged TSV that job 3066823 trained on descends directly from the ChEMBL32 SQLite projection, whose summary records `openadmet_firewall_applied` false. Every row and assay removed after the pregraph stage is accounted for by the graph-size and graph-valid-count assay rejections (169 + 13,084 = 13,253 = 67,985 − 54,732). There is no identity-based removal stage. The false flag therefore describes the actual training-corpus lineage, not an abandoned intermediate.

### Remaining gaps

- (a) Pregraph summary contents: the `bd804a6d…` → `43dfd672…` link and the pregraph firewall flag.
- (b) Shard TSV hashes vs the shard manifest (not printed).
- (c) All hashes are as recorded in the summaries, not independently re-hashed.
- (d) Molecule-level overlap across the corpus is unmeasured; the propranolol/ritonavir identities are unadjudicated (§27).

### Status (as of 2026-11-15)

- The CoreHPC `find` listing and the projection, shard, graph-stage, merge and training-startup records are reviewed. The pregraph summary contents are not.
- Recorded lineage: ChEMBL32 SQLite → projection (`bd804a6d…`, firewall flag false) → [pregraph, link unconfirmed] → pregraph TSV (`43dfd672…`) → 16 shards → merged TSV (`24d9fd29…`) → job 3066823.
- No retraining, rescoring, CYP evaluation or decision entry.

Work stopped at: the CoreHPC `find` listing and the manifest chain were recorded. The pregraph summary contents and the shard TSV hashes are not yet reviewed.

---

## 29. Pregraph Summary Contents and Lineage Cross-Checks (2026-11-15)

Source: Miles ran `cat /scratch/user/mhuang8/actfound/chembl32/pregraph_production_20261113/chembl32_actfound_pregraph.tsv.summary.json` on CoreHPC and pasted the output into the Claude Code conversation on 2026-11-15 [Miles 2026-11-15]. The artifact is on CoreHPC, not in this local checkout. This is a later state than §28: it covers the pregraph summary that §28 B marked **NOT YET REVIEWED**. Earlier sections are left as written.

This entry records read-only checks only. It is not a decision, has no D-number, and is not a session advance.

### A. Pregraph summary (CoreHPC, verbatim)

```json
{"artifact_type": "ActFound_ChEMBL32_streamed_pregraph_population_v1", "assay_tasks": 573656, "dedup_ge20_assays": 67985, "dedup_ge20_rejected_assays": 1382, "dedup_ge20_rows": 9873276, "discarded_molecules": 5288, "duplicate_groups": 114296, "duplicate_rows": 210604, "elapsed_seconds": 69.17183827375993, "input_path": "/tmp/actfound_unit_eligible_sorted_3048100.tsv", "input_rows": 12377324, "input_sha256": "6d8fe4acad9e1c50c3b2913b76ccd0f53cac848f399f848d4942822df114411c", "openadmet_firewall_applied": false, "output_path": "/scratch/user/mhuang8/actfound/chembl32/pregraph_production_20261113/chembl32_actfound_pregraph.tsv", "output_sha256": "43dfd672e790ba3f92bd1e17b5902a44aec88911a5bbf0db26f92d0e192db076", "raw_ge20_assays": 69367, "raw_ge20_rejected_assays": 504289, "raw_ge20_rows": 10109406, "rdkit_graph_applied": false, "rows_per_second": 178935.88357467853, "runtime": {"python": "3.9.25", "rdkit": "UNAVAILABLE", "rdkit_import_error": "ModuleNotFoundError: No module named 'rdkit'", "sqlite": "3.34.1"}, "script_sha256": "d67370ac54eab35d47079c9083d16b397b6d5728c82846639976ed99ec2f1713"}
```

### B. Cross-checks (assistant, verified by comparison)

- Pregraph `"openadmet_firewall_applied": false`. Pregraph output_sha256 `43dfd672…` equals the shard manifest input_sha256 (§28 B). This resolves §28 gap (a) for the pregraph → shard link and the pregraph firewall flag.
- RDKit was unavailable in this stage's runtime, so no structure-based identity step could have run here. discarded_molecules 5,288 is not identity-based; its cause per code is not adjudicated here.
- Pregraph script_sha256 `d67370ac…` equals the SHA-256 of local `cyp/scripts/actfound_pregraph_stream.py` (computed 2026-11-15).

### C. New gaps

- The pregraph input is `/tmp/actfound_unit_eligible_sorted_3048100.tsv` (SHA-256 `6d8fe4acad9e1c50c3b2913b76ccd0f53cac848f399f848d4942822df114411c`, 12,377,324 rows), a sorted temporary file. It is not the projection output `chembl32_actfound_unit_eligible.tsv` (SHA-256 `bd804a6d…`). The sort step `bd804a6d…` → `6d8fe4ac…` has no recorded summary. Whether the temporary file still exists: UNKNOWN.
- The projection summary's script_sha256 `71161e52a44278e1666b968d52e827c94b23682f4b1c9fa8a8801ed450fd0c54` does NOT equal the SHA-256 of local `cyp/scripts/actfound_chembl32_projection.py` (`22fca4084849fd58e791f80c94d24e9b282e1288900a37ddbf31c0872dcf8f3b`). The local projection script is therefore not byte-identical to the script that produced `bd804a6d…`. Which script version ran: UNKNOWN. The local file was not part of the 2026-11-15 sync (§23 A).

### D. Assistant conclusion (assistant analysis, not a decision)

The firewall flags are false at both stages that record one (projection and pregraph), and no later stage removes rows by identity. The conclusion that the training corpus was not OpenADMET-firewalled is unchanged by the gaps in C.

### Status (as of 2026-11-15)

- §28 gap (a) is resolved for the pregraph → shard link and the pregraph firewall flag. The projection → pregraph-input link remains unconfirmed because of the unrecorded sort step.
- §28 gaps (b), (c) and (d) are unchanged.
- No retraining, rescoring, CYP evaluation or decision entry.

Work stopped at: the pregraph summary and its cross-checks were recorded. The sort step `bd804a6d…` → `6d8fe4ac…` and the projection script version are unresolved.

---

## 30. Two-Molecule Identity Adjudication in the Frozen D012 Identity Environment (2026-11-15)

Source: a read-only, in-memory check the assistant ran locally in this checkout on 2026-11-15, after Miles approved it ("Approve the check") [Miles 2026-11-15]. No files were written. This is a later state than §27 D and §28 gap (d): it adjudicates the two identities that those sections marked unadjudicated. Earlier sections are left as written.

This entry records a read-only check only. It is not a decision, has no D-number, and is not a session advance.

### A. Environment

- Conda env `tripwire_cyp002` (`/opt/homebrew/Caskroom/miniforge/base/envs/tripwire_cyp002`).
- Printed runtime: Python 3.12.14, RDKit 2026.03.6, InChI 1.07.3. This is the frozen D012 identity environment.
- No project-added standardization. The ActFound runtime was not used.

### B. Method

- RDKit `MolFromSmiles` on the training-corpus SMILES as printed from the frozen training TSV (`24d9fd29…`, §27 C), then Standard InChI (`MolToInchi`) and the full Standard InChIKey.
- Compared with `full_standard_inchikey` and `standard_inchi` in `qualification/cyp_external_task_qualification_20261109/identity_audit/cross_source_matches.tsv` (SHA-256 `6613d182fb4492b4e79f71dcd29d15c7c73f689fe15087d7b1a66aeb7967a8c5`).

### C. Output (verbatim)

```text
env: 3.12.14 RDKit 2026.03.6 InChI 1.07.3
molregno 1370 / CHEMBL832148
  computed key : AQHHHDLHHXJYJD-UHFFFAOYSA-N
  audit key    : AQHHHDLHHXJYJD-UHFFFAOYSA-N
  key match    : True
  InChI match  : True
  OpenADMET    : partitions=train names=OCNT-0018684 chembl_ids=CHEMBL27
molregno 88100 / CHEMBL1100026+CHEMBL3118928
  computed key : NCDNCNXCDXHOMX-XGKFQTDJSA-N
  audit key    : NCDNCNXCDXHOMX-XGKFQTDJSA-N
  key match    : True
  InChI match  : True
  OpenADMET    : partitions=train names=OCNT-0495340 chembl_ids=CHEMBL163
```

### D. Finding (verified)

Under the frozen identity contract, two training-corpus molecules have the same full Standard InChIKey and identical Standard InChI as OpenADMET development-universe molecules:

| Training-corpus molregno | SMILES (frozen training TSV) | Assay(s) / `assay_key` | OpenADMET molecule | OpenADMET partition | ChEMBL ID |
|---|---|---|---|---|---|
| 1370 | `CC(C)NCC(O)COc1cccc2ccccc12` | `CHEMBL832148` / `305983_nM_IC50` | `OCNT-0018684` | train | `CHEMBL27` |
| 88100 | as printed in §27 C | `CHEMBL1100026` / `621465_nM_IC50`; `CHEMBL3118928` / `1291572_nM_IC50` | `OCNT-0495340` | train | `CHEMBL163` |

All three assays are in job 3066823's training split (§27 A).

This supersedes §27 D's "identity unadjudicated" and the matching part of §28 gap (d) (later state).

### E. Limits

- Load-time row filters for these rows were not checked.
- Full-corpus overlap remains unmeasured.
- This is a bounded 2-molecule check, not an overlap census.

### F. Assistant note (assistant analysis, not a finding or decision)

This confirms that job 3066823's foundation corpus contained at least two OpenADMET development-universe molecules, contrary to the Nov 12 proposal's firewall requirement. What this means for how the checkpoint is used is Miles's decision.

### Status (as of 2026-11-15)

- Identity of the two §27 molecules: adjudicated, matched, in the frozen D012 identity environment.
- Load-time row filters for these rows: not checked. Full-corpus overlap: UNKNOWN.
- §29 gaps (the sort step `bd804a6d…` → `6d8fe4ac…` and the projection script version) are unchanged.
- No retraining, rescoring, CYP evaluation or decision entry.

Work stopped at: the two-molecule identity adjudication was recorded. Consequences for the checkpoint's use are pending Miles's decision.

---

## 31. Full-Corpus Identity Overlap Audit: CoreHPC Membership Export and Local Frozen-Environment Run (2026-11-15)

Sources: Miles approved the overlap calculation spec ("Approve as recommended"), the scripts ("Approve, give commands") and the full run ("Start full run") [Miles 2026-11-15]. Miles ran the CoreHPC export (steps 1–3) and downloaded the bundle. The sample qualification and the full run were run locally in this checkout on 2026-11-15. The CoreHPC export directory and bundle are on CoreHPC; the downloaded copy is in `~/Downloads/actfound_overlap_export_20261115/` on the Mac, outside this checkout. The full-run outputs are in this checkout. This is a later state than §27–§30: it measures the full-corpus overlap those sections marked UNKNOWN or unmeasured. Earlier sections are left as written.

This entry records a result only. It is not a decision, has no D-number, and is not a session advance.

### A. Scripts (new, uncommitted)

In `qualification/actfound/overlap_audit_20261115/`:

| Script | SHA-256 | Where it runs |
|---|---|---|
| `export_training_membership.py` | `4cd92c5058fc432e75ce547da1dea4475bf0e210e81905f5a9ba2754794dbd59` | CoreHPC, pinned ActFound env; membership export only |
| `identity_overlap.py` | `12b0d68b578eb9438d4be8b94afaa68e8941535807df3035255ad68e231534a7` | Mac, frozen identity env |
| `test_overlap_audit.py` | `19cca9a3136b16461c74a744e29ad27d76dd627819ed07484c56fe881b34f691` | Mac |

Tests: 4 passed in `tripwire_cyp002` (`pytest -q`). The export script's loader alignment was tested locally only with a stand-in loader, because torch is absent on the Mac.

### B. CoreHPC membership export (Miles, steps 1–3)

- Output directory (CoreHPC): `/scratch/user/mhuang8/actfound/chembl32/overlap_audit_20261115/export`.
- Bundle (CoreHPC): `/scratch/user/mhuang8/actfound_overlap_export_20261115.tar.gz`. Downloaded with ProxyJump `scp` to `~/Downloads/actfound_overlap_export_20261115/`. Local bundle SHA-256 `30073d53806305baa91f56031c205584abbbc5da01f8fa72886d67a6aa8e52d4`.
- The step-3 CoreHPC bundle hash was not pasted, so bundle-level equality is **UNVERIFIED**. The per-file `SHA256SUMS.txt` inside the bundle verified OK:
  - `membership.tsv.gz` `1ccfbc0ac9ceeba9b5260efb2a294b3791ff2f9a37c6f741d1c76c270cef8949`
  - `molecules.tsv.gz` `e7f756f7f4396e0286affdc156965e8afbef8e4003155cd62448d6ded4ff2904`
  - `export_summary.json` `69022d6b83ffd9cde889037d2225d3345af80bfd7eb42d9ffc2839939904540b`

Export summary:

- script_sha256 `4cd92c50…`; loader_module_sha256 `97c1753512d102e1198bfad2ab4b601749cb1d57b6a1a3a1ad847be83214ee45` (job 3066823's loader); tsv_sha256 `24d9fd29…`; per_id_sha256 `7ea5ae08…`.
- Runtime 359.4 s; Python 3.9.18.
- unique_molregnos 1,076,934; molregnos_missing_chembl32_structure 0.
- Row counts by split | status:

| Split | Status | Rows |
|---|---|---|
| train | kept | 1,321,726 |
| train | relation_not_eq | 181,869 |
| validation | kept | 19,442 |
| validation | relation_not_eq | 2,338 |
| none | kept | 541,617 |
| none | relation_not_eq | 485,796 |
| none | assay_not_emitted_lt20_or_std | 221,207 |
| **Total** | | **2,773,995** (= corpus rows; verified) |

The `/usr/bin/time` wall-clock and RSS lines were not pasted.

### C. Sample qualification (local, frozen env)

- 10,000 molecules. Scratch outputs were not kept.
- 1 worker vs 8 workers: byte-identical `molecule_identities.tsv` and `failures.tsv`.
- 0 parse failures; 0 source-vs-witness disagreements.
- Identity time 7.4 s (1 worker) / 3.6 s (8 workers).

### D. Full run (local)

- Env `tripwire_cyp002`: Python 3.12.14, RDKit 2026.03.6, InChI 1.07.3. 8 workers. Exit 0. Identity 115.6 s; total 123.3 s.
- Outputs in `qualification/actfound/overlap_audit_20261115/full_run/` (`SHA256SUMS.txt` verified OK): `molecule_identities.tsv` (125,031,703 bytes), `failures.tsv`, `summary.json`.
- The run's built-in check reproduced both §30 matches (molregnos 1370 and 88100).
- Molecules processed 1,076,934. 1 parse failure in both source and witness (molregno 2336618, hexafluorophosphate salt SMILES). 0 source-vs-witness key disagreements.

Prohibited sets:

- D007 development universe: 4,169 names = OpenADMET train 3,433 + validation 736 (verified against the 50 D007 split manifests).
- internal_test: 736.
- challenge_blinded_test: 750.

Matched molecules (unique molregnos) by tier and membership. A molecule can carry several flags.

| Tier | train_kept | train_dropped | validation_kept | validation_dropped | none_kept | none_dropped |
|---|---|---|---|---|---|---|
| development | 203 | 37 | 14 | — | 166 | 240 |
| internal_test | 44 | 10 | 3 | — | 44 | 57 |
| blinded | 7 | 3 | — | 3 | 7 | 6 |

(— = not reported.)

Matched rows / assays, loader-kept:

| Tier | Training split | Validation split |
|---|---|---|
| development | 2,139 rows in 1,515 assays | 22 rows / 18 assays |
| internal_test | 444 rows in 332 assays | 3 rows / 3 assays |
| blinded | 25 rows in 23 assays | not reported |

Follow-up computations from `molecule_identities.tsv` (assistant):

- D007 development names seen in loader-kept training rows: 203 of 4,169 (4.9%). By OpenADMET partition: 163 train, 40 validation.
- Per D007 partition (50 rep/fold manifests), share of that partition's validation molecules present in loader-kept training rows: min 0.026, median 0.046, max 0.072.

### E. Assistant note (assistant analysis, not a finding or decision)

This establishes that job 3066823's foundation corpus was not leakage-clean under the Nov 12 proposal's firewall definition: 203 development molecules (plus 44 internal-test and 7 blinded-challenge molecules) entered training rows. Consequences for the checkpoint and any retraining are Miles's decision. No decision is recorded.

This supersedes the "full-corpus overlap UNKNOWN / unmeasured" state in §27–§30 (later state).

### Status (as of 2026-11-15)

- Full-corpus identity overlap: measured (D above).
- Bundle-level hash equality between CoreHPC and the local download: UNVERIFIED. Per-file hashes verified.
- §29 gaps (the sort step `bd804a6d…` → `6d8fe4ac…` and the projection script version) are unchanged.
- Not yet decided or recorded: how `full_run/molecule_identities.tsv` (125 MB) is kept out of git; whether the local export staging is preserved.
- No retraining, rescoring, CYP evaluation or decision entry.

Work stopped at: the full-corpus overlap audit was recorded. Consequences for the checkpoint and any retraining are pending Miles's decision.

---

## 32. Miles's Decision on the Route After the Overlap Result: Option A (2026-11-15)

Sources: Miles's reply to the assistant's options message [Miles 2026-11-15]. The assistant sent the options message on 2026-11-15, after the §31 overlap result. Miles confirmed this entry belongs to Session 005 ("Yes, log to 005").

This entry records Miles's decision as he stated it. No D-number record has been created for it: the governing record (likely D013) is pending Miles's approval of a draft. This is not a session advance.

### A. Decision (verbatim)

Miles, 2026-11-15:

> "A"

### B. What "A" refers to

Option A in the assistant's options message (2026-11-15), verbatim:

> "A. Retrain the foundation model on a firewalled corpus (recommended). This is the proposal's own route: build the firewall, freeze the corpus, train once, freeze the checkpoint, then evaluate."

The alternatives were not chosen:

- B: use job 3066823's checkpoint, labelled not leakage-clean.
- C: keep 3066823 as a leaky-vs-clean diagnostic. This would only be noted and would need separate approval.

In the same message the assistant stated that job 3066823's artifacts are preserved unchanged either way.

### C. Open sub-choices (not answered by Miles; not resolved here)

1. **Firewall scope.** The Nov 12 proposal requires the development universe (4,169). The assistant recommended also removing internal_test (736) and challenge_blinded_test (750), which covers all 5,655 OpenADMET structures. **Open.**
2. **Base corpus.** The candidates are the authors' released `chembl_processed_chembl32.zip` (inside `external/ActFound/upstream_data/datas.zip`, never extracted) and the SQLite-regenerated corpus `24d9fd29…`. The assistant recommended a read-only qualification of the authors' file first. **Open.**
3. **Removal level.** Molecule-level row removal, with attrition of assays at the ≥20-molecule threshold logged. **Open.**
4. **Governing decision record.** This would be a record (likely D013) covering the Nov 12 contract with correct authorship, the firewall definition, the corpus choice and job 3066823's status. The assistant offered to draft it for Miles's approval. **Not drafted or approved.**

### Status (as of 2026-11-15)

- Route: Option A (retrain on a firewalled corpus), decided by Miles.
- Sub-choices 1–4: open.
- Job 3066823 artifacts: preserved unchanged.
- No firewall built, no corpus frozen, no retraining, no evaluation, no D-number record.

Work stopped at: Miles chose Option A. Next is answering the sub-choices and the D013 draft for his approval.

---

## 33. Authors' Processed ChEMBL32 Corpus Qualification and Miles's Approval of the Option A Sub-Choices (2026-11-15)

Sources: the assistant's read-only check, 2026-11-15. Files were extracted only into the assistant's scratch directory, outside this checkout, and nothing in the repo changed. Miles's reply to the assistant's sub-choices message [Miles 2026-11-15]. Miles confirmed this entry belongs to Session 005 ("Yes, log to 005"). This is a later state than §32 C: it records Miles's answers to sub-choices 1–3 and his instruction on sub-choice 4. §32 is left as written.

This entry records a result and Miles's decision as he stated it. No D-number record has been created: the D013 wording is pending Miles's approval. This is not a session advance.

### A. Result: authors' processed ChEMBL32 corpus (assistant, read-only, 2026-11-15)

Source chain (SHA-256):

| File | SHA-256 |
|---|---|
| `external/ActFound/upstream_data/datas.zip` | `74b1b30cf2460ee68db89e97299a3029f9460e991a5e6646e035e3c31640b2f1` |
| → `datas/chembl/chembl_processed_chembl32.zip` | `29389247e1c743e9bf0598b4e820dc1cd589d2819e886f5bf8ceb7441d493e16` |
| → `chembl_processed_chembl32.csv` (file date 2023-03-10, 309,056,950 bytes) | `a5b562d4cd237f94f80b1d6a85031825bc1b0b1157e734f76bc3315f256c5e20` |
| → `chembl_processed_chembl32_percent.csv` | `e503a2888e9e3133bbe1047415bad78992aed07a8e3652382b2abc7c1a6bb553` |

The extracted copies are in the assistant's scratch directory only, not in this checkout.

Main CSV:

- 1,959,398 rows, all with 15 columns, no header.
- 47,324 assay descriptors (`assay_chembl_id_units_type`); 847,866 molregnos.

Upstream split (`external/ActFound/datas/chembl/chembl_split.json`):

- Keys train / valid / test / test_external = 33,296 / 500 / 500 / 1,348.
- Every split ID is present directly as a descriptor in the authors' CSV.

Comparison with the rebuilt corpus export (§31):

- All 1,959,398 authors' (descriptor, molregno) pairs are present in the rebuilt corpus export (2,773,995 pairs). 1,402,497 of them are in loader-kept rows.
- 0 authors-only pairs; 0 authors-only molregnos.
- Values were **not compared**.

Job 3066823 config: `val_assays` == upstream "valid" (not "test"); `train_assays` == upstream "train".

Overlap in the authors' training split (rows with relation "=" and units not "%"), using the §31 identities:

| Tier | Molecules | Rows |
|---|---|---|
| development | 203 | 2,139 |
| internal_test | 44 | 444 |
| blinded | 7 | 25 |

These are the same counts as the rebuilt corpus (§31 D).

Training-assay attrition (assays with ≥20 such rows):

- 33,296 before removal.
- 33,205 after removing development-universe molecules.
- 33,184 after removing all 5,655 OpenADMET structures.
- Validation-split attrition: **not yet computed**.

### B. Decision (verbatim)

Miles, 2026-11-15:

> "Approve, draft D013"

### C. What was approved

Miles was answering the assistant's message proposing the following. These are the approved choices, as given in the assistant's message:

1. Firewall scope: remove all 5,655 OpenADMET structures (D007 development universe 4,169 + internal_test 736 + challenge_blinded_test 750), by full Standard InChIKey in the frozen identity environment.
2. Base corpus: the authors' released `chembl_processed_chembl32.csv` (Nov 12 proposal step 2), requiring a small tested converter to the loader's graph-qualified format with the same graph-qualification checks.
3. Removal level: molecule-level row removal, then drop and log assays left with fewer than 20 usable molecules, including the validation split.
4. The assistant drafts D013 (Nov 12 contract with authorship as Miles specified — GPT proposed, assistant summarized, Miles approved source pinning; option A; firewall definition; base corpus; job 3066823 preserved but not usable as the study checkpoint; frozen-checkpoint rule) for Miles's approval of wording before anything is written.

### Status (as of 2026-11-15)

- §32 sub-choices 1–3: answered by Miles (C above). Sub-choice 4: Miles instructed the assistant to draft D013.
- D013 wording: **not yet approved**. No `notes/Decisions.md` entry exists for D013.
- Not started: firewall, corpus build, converter, training, evaluation.
- Validation-split attrition: not yet computed.
- Job 3066823 artifacts: preserved unchanged.

Work stopped at: Miles approved the sub-choices and asked for the D013 draft. Next is the D013 wording for his approval.

---

## 34. Miles's Approval of D013 and Operational Instruction (2026-11-15)

Sources: Miles's reply to the assistant's D013 draft [Miles 2026-11-15]. Miles confirmed this entry belongs to Session 005 ("Yes, log to 005"). This is a later state than §33: §33 recorded D013 wording as not yet approved. §33 is left as written.

This entry records Miles's decision and instruction as he stated them. This is not a session advance.

### A. Decision (verbatim)

Miles, 2026-11-15:

> "D013 is approved, move forward with implementation."

Miles approved the complete D013 draft as written.

### B. Decision record written

D013 was appended to `notes/Decisions.md` verbatim from the approved draft, as:

- Heading: `# D013 — ActFound-CYP Whole-Package Study on a Firewalled ChEMBL32 Foundation Corpus`
- Location: `notes/Decisions.md` lines 6135–6244, after the D012 post-result forensic addendum.

### C. Scope of the approval

- Implementation is authorized, per Miles's words ("move forward with implementation").
- Production training and the 150-cell evaluation still follow D013's own gates. D013 states that it "does **not** by itself authorize production training or the 150-cell evaluation; each still requires the qualification, interactive benchmark and runtime extrapolation below, and Miles's go-ahead."
- Each D013 open item (1. seed meaning for a single frozen checkpoint, 150 vs 50 cells; 2. label-scale conversion; 3. large-support handling; 4. the thin target adapter and its scientific unit tests) still requires separate approval before evaluation, as D013 states.

### D. Operational instruction (verbatim)

Miles, 2026-11-15:

> "run everything trhough the terminal. for core hpc use the establish ssh commands in the terminal and wait for me to authenticate the password and duo mobile push"

### Status (as of 2026-11-15)

- D013: approved by Miles and recorded in `notes/Decisions.md`.
- Implementation: authorized; not started as of this entry.
- Not started: firewall, corpus build, converter, training, evaluation.
- Production training and 150-cell evaluation: not authorized until D013's gates are met and Miles gives the go-ahead.
- D013 open items 1–4: open.
- Validation-split attrition: not yet computed.
- Job 3066823 artifacts: preserved unchanged.

Work stopped at: Miles approved D013 and authorized implementation. Next is D013 implementation, run through the terminal per Miles's instruction.

---

## 35. Miles's Acceptance of the Grilling-Round Answers (Q1–Q9) and Withdrawal of the Overnight-Failure Rule (2026-11-15)

Sources: Miles's reply to the assistant's grilling round Q1–Q9 [Miles 2026-11-15]. Miles confirmed this entry belongs to Session 005 ("Yes, log to 005").

This entry records Miles's decisions as he stated them. No D-number record has been created for it, and D013 is unchanged by this entry. This is not a session advance.

Correction: this entry was corrected on 2026-11-15 at Miles's approval ("Approve the fix") because it had misquoted him.

### A. Decision (verbatim)

Miles, 2026-11-15:

> "ok lets just make our submission is bullet tight, disregard the reuqest for a live monitor agent. i accept your answers to each question, remove the overnight-failure rule from the log"

### B. Live monitor and overnight-failure rule

- "disregard the reuqest for a live monitor agent": no live or periodic monitor agent.
- "remove the overnight-failure rule from the log": a withdrawn operational rule is withdrawn. Its log entry (the earlier §35 of this file) was removed at Miles's request, with his explicit approval.

### D. Accepted answers (grilling round Q1–Q9, 2026-11-15)

"i accept your answers to each question": Miles accepted the assistant's recommended answers to Q1–Q9:

1. Q1 Split: the study split is the upstream train/valid split minus assays the firewall empties out (below ≥20 usable molecules), frozen in new hashed reconciliation artifacts and a new config version (v2); the runner's fail-closed checks are not loosened.
2. Q2 Runner v2: the input path and its SHA-256 move from runner constants into the frozen config; all else byte-identical; diff and new hash shown before use; tests updated.
3. Q3 Graph re-qualification: apply the firewall first, then rerun the existing graph-qualification code (which reproduces upstream chembl_preprocess.py lines 103–174), then the ≥20 rule. Fact established: the authors' chembl_processed_chembl32.csv is the final post-graph output of upstream chembl_preprocess.py.
4. Q4 Salt/parent forms: keep the full-Standard-InChIKey firewall per the contract; additionally report, without removing, InChIKey-first-block and largest-fragment near-matches as characterization; any widening is a separate decision.
5. Q5 Failure handling: accept no resume; no --requeue; a failed run is resubmitted from scratch into a new output directory. (Slurm mail alerts were later declined by Miles: "no need for slurm alerts, I'll be asleep anyways" — record this.)
6. Q6 Output guard: runner v2 refuses to start if its output directory already exists; each run has its own directory recorded in its config.
7. Q7 Submission record: before submission, record the exact sbatch line, the hashes of wrapper, runner, config and corpus, and explicit PROJECT_ROOT/ACTFOUND_CONFIG values; the wrapper prints its own hash and resolved environment at start.
8. Q8 Home quota: the wrapper sets TMPDIR, XDG_CACHE_HOME, TORCH_HOME and CUDA_CACHE_PATH under /scratch/user/mhuang8.
9. Q9 Converter check: convert the unfirewalled authors' file and compare it value by value with the rebuilt corpus on shared rows; every difference reported, none silently resolved.

### Status (as of 2026-11-15)

- Q1–Q9: the assistant's answers were accepted by Miles (D).
- Live monitor agent: none (B).
- Overnight-failure rule: withdrawn, and its log entry removed (B).
- Implementation of D013: proceeding; nothing built or submitted yet.
- D013 and its gates: unchanged by this entry. Production training and the 150-cell evaluation still need D013's gates and Miles's go-ahead.
- None of the Q1–Q9 items are recorded here as implemented. No config v2, runner v2, reconciliation artifacts, converter qualification or submission is recorded.

Work stopped at: Miles accepted the Q1–Q9 answers and withdrew the overnight-failure rule. Next is implementing per the accepted answers.

---

## 36. D013 Implementation Progress: CoreHPC Connection and Discovery, Firewall List, Per-Assay Comparison, Runner v2 and Wrapper v2 (2026-11-15)

Sources: implementation results reported in the main conversation [Miles/assistant 2026-11-15]. Miles confirmed this entry belongs to Session 005 ("Yes, log to 005"). Work authorized by Miles's D013 approval ("D013 is approved, move forward with implementation.", §34) and his acceptance of Q1–Q9 (§35).

This is a routine factual results entry. It is not a decision and not a session advance. `notes/Decisions.md` is unchanged by this entry.

### A. Shared CoreHPC SSH connection

- OpenSSH ControlMaster opened with Miles's password + Duo at about 01:49 PST (master PID 15100, ControlPersist 10h).
- Confirmed 01:53 PST: `Master running`; a command ran on chpc-ucsf-login-vm1 as mhuang8.

### B. Read-only CoreHPC discovery (01:50 PST)

Evidence location: CoreHPC, `/scratch/user/mhuang8/` (hashes and listings as reported from CoreHPC).

Corpus-pipeline scripts in `/scratch/user/mhuang8/`, all equal to the local copies:

| Script | SHA-256 | mtime |
|---|---|---|
| `actfound_chembl32_projection.py` | `22fca4084849fd58e791f80c94d24e9b282e1288900a37ddbf31c0872dcf8f3b` | 2026-11-12 23:23:13 |
| `actfound_graph_stage.py` | `1783c5fe8f237261e7f5a246910a8f2bad1cf990a8a6ac11a07af04b645a9995` | 00:33:21 |
| `actfound_pregraph_stream.py` | `d67370ac54eab35d47079c9083d16b397b6d5728c82846639976ed99ec2f1713` | 00:12:16 |

- The projection script's mtime (23:23:13) is after the 23:02 projection run. Per the assistant, this explains the recorded projection script_sha256 `71161e52…` mismatch (§29 C). The graph stage imports its graph functions from this file. (Interpretation reported by the assistant: the bytes that ran at 23:02 were not recovered, so they are still unverified.)
- `graph_production_20261113` contains `run_graph_array.sbatch`, so the graph stage ran as a Slurm array.
- Slurm MailProg `/bin/mail`, MailDomain (null).
- `/scratch`: 1.5P free; `/scratch/user/mhuang8/actfound` uses 48G.
- Pinned env: Python 3.9.18, numpy 1.26.4, RDKit 2023.03.1, torch 1.12.1.
- L40S nodes ggpu1-[06-12] (2 GPUs each, mixed state).
- CoreHPC project-tree test files `6f5fa3d9…` (model-side) and `ef99704a…` (production-train) confirm the earlier upload (§24). Runner `8a160427…`, model side `97c17535…` and wrapper `70f76c0e…` are unchanged.
- Slurm mail alerts: declined by Miles, verbatim: "no need for slurm alerts, I'll be asleep anyways" (also noted in §35 D Q5).

### C. Firewall list (local, frozen identity env)

Evidence location: this checkout.

- Environment: frozen identity env `tripwire_cyp002` (Python 3.12.14 / RDKit 2026.03.6 / InChI 1.07.3).
- Script: `qualification/actfound/d013_firewall_20261115/build_firewall_list.py` (`b1280e093632877309403aeb3403b090c5825a0cc0d5a20ea50dce6b5954f401`).
- Test: `test_build_firewall_list.py` (`f904cff3a682ca0ac700e8c5bd091e8e229b56476dcb6eb35bfdf6c92d0cc838`): 1 passed.
- Output: `qualification/actfound/d013_firewall_20261115/firewall_list/`; `SHA256SUMS.txt` verified OK.
- Result: 337 firewall molregnos out of 1,076,934 considered, by full Standard InChIKey match to the 5,655 protected OpenADMET structures. By tier: development 263, internal_test 61, blinded 13.
- Near-matches NOT removed (Q4 characterization; counts are molregno-kind rows): first_block 219, largest_fragment_full 180, largest_fragment_first_block 461.
- largest_fragment_full by tier: development 140, internal_test 40. These are, for example, salt forms whose parent structure equals a protected OpenADMET structure.
- Widening the firewall would be a separate Miles decision (Q4). Not decided.

### D. Per-assay comparison (assistant, read-only)

- The rebuilt graph-qualified corpus (`24d9fd29…`) has the identical molecule set to the authors' `chembl_processed_chembl32.csv` in 47,323 of 47,324 authors' assays.
- The exception, `CHEMBL4651403_nM_IC50`, has 825 molecules in the authors' file and 826 in the rebuilt corpus. It is in none of the upstream splits (train/valid/test/test_external).

### E. Graph-qualification constraint (assistant analysis, not a decision)

- Upstream eligibility uses degree ≥ sqrt(sqrt(N))·2, where N is the pre-graph assay size including degree-0 molecules (`actfound_chembl32_projection.py` `postgraph_assay_filter_exact_actfound`).
- So graph qualification cannot be rerun exactly from the authors' post-graph rows alone.
- Resolution pending Miles. This affects D013's "converter" wording. Unresolved.

### F. Runner v2 and wrapper v2 (local, uncommitted, not yet uploaded)

Evidence location: this checkout. These files are new; nothing has been uploaded to CoreHPC.

- `cyp/scripts/actfound_production_train_v2.py` (`ed3e1055b1a9f5a76fc6b0ddad0df7e4a2df95254e513aca90e7359615db1865`), derived from v1 (`8a160427…`):
  - input path and SHA-256 come from the config (Q2);
  - requires artifact_type `ActFoundProductionConfigV2`;
  - new required `corpus_manifest_path`/`corpus_manifest_sha256`, with fail-closed checks that the manifest records `openadmet_firewall_applied` true and that its output path/SHA-256 equal the config's frozen input. This is an assistant addition beyond Q2, flagged for Miles's review. Not yet approved;
  - refuses an existing output directory before any work (Q6);
  - startup source hashes include the v2 runner.
- `cyp/scripts/actfound_production_train_v2.slurm` (`439c8ad1bb43e5bcb4535e28e8eeb69646de18f10e30b47d7fe4c50d4db123e9`):
  - L40S gres, `--no-requeue`, no mail;
  - PROJECT_ROOT, ACTFOUND_ENV and ACTFOUND_CONFIG are required, with no defaults;
  - TMPDIR, XDG_CACHE_HOME, TORCH_HOME and CUDA_CACHE_PATH are set under `/scratch/user/mhuang8/actfound/runtime/<jobid>` (Q8);
  - logs the wrapper, runner, model-side and config hashes and the resolved environment at start (Q7).
- `cyp/tests/test_actfound_production_train_v2.py` (`e88e61e0ed6dcd14c308f77b053423114b52c6241adf45e16e69b902757593e6`).
- Local test run: `python -m pytest -q -p no:cacheprovider cyp/tests/test_actfound_production_train_v2.py cyp/tests/test_actfound_production_train.py`: 20 passed.
- `bash -n` on the wrapper: OK.
- The v1 runner and wrapper are unchanged.
- These are local edits with passing local tests. They are not deployed or verified on CoreHPC.

### Status (as of 2026-11-15)

- Firewall list: built and hash-verified locally (C).
- Firewalled corpus: not built.
- Config v2 and reconciliation v2: not written.
- Runner v2 / wrapper v2: written locally, 20 local tests pass; not uploaded.
- Uploads and submissions: none.
- Open for Miles: graph-qualification constraint and D013 "converter" wording (E); the runner v2 corpus-manifest check added beyond Q2 (F); any firewall widening (C).
- D013 and its gates: unchanged. Production training and the 150-cell evaluation still need D013's gates and Miles's go-ahead.

Work stopped at: firewall list built; runner v2 and wrapper v2 written and tested locally. Next is building the firewalled corpus, config v2 and reconciliation v2, pending Miles on items E and F.

---

## 37. D013 Implementation Progress: Split/Config v2 and Firewalled-Corpus Builders, CoreHPC Upload and Test Runs (2026-11-15)

Sources: implementation results reported in the main conversation [Miles/assistant 2026-11-15]. Miles confirmed this entry belongs to Session 005 ("Yes, log to 005"). Work authorized by Miles's D013 approval (§34) and his acceptance of Q1–Q9 (§35).

This is a routine factual results entry. It is not a decision and not a session advance. `notes/Decisions.md` is unchanged by this entry.

### A. New builder scripts (local, uncommitted)

Evidence location: this checkout, `qualification/actfound/d013_firewall_20261115/`.

- `build_split_and_config_v2.py` (`95bc039090ead03abb00769663adf2e184b3f847caf55f44deacc98ba0440c2c`):
  - runs on CoreHPC in the pinned env;
  - implements Q1: upstream train/valid minus assays not emitted by the project loader from the firewalled corpus, with drop-outs logged;
  - writes reconciliation v2 (per-ID TSV, summary, dropouts) and config v2 = config v1 with only the listed keys replaced (asserted);
  - then validates the config with runner v2.
  - Test: `test_build_split_and_config_v2.py` (`c672327f3cbfbc17c87fe47e578d01c7fb1a369dfbca099e7377cd5922e693ef`).
- `build_firewalled_corpus.py` (`0ac0a620c24dae58f468a84b42b47ae09e67e091350584b4217cc725b2b263d6`):
  - graph route C: the authors' assay universe; rebuilt pre-graph rows (`43dfd672…`); firewall removal; the unchanged graph stage with a code-hash check (`1783c5fe…` / `22fca408…`);
  - reproduction mode compares every output row, field by field, with the authors' file.
  - Test: `test_build_firewalled_corpus.py`, now `53a6738251d5c6b4e1b41c4f4d4d8462312ed4bca94540ab5962b7e04010c8cf` after a fixture fix adding `ACTFOUND_GRAPH_CODE_DIR` (earlier version `7c1c58d6…`).
- Route C is NOT approved by Miles. The D013 amendment for it (Amendment 1) is drafted, not approved. Nothing has been run on real data with this script.

### B. Upload to CoreHPC (shared connection)

Evidence location: local bundle and CoreHPC paths as reported.

- Bundle: `~/Downloads/d013_upload_20261115/d013_code_20261115.tar.gz` (SHA-256 `dcd0feed5208f10c71e035adb1659289eb4cac64aa7eda3fd24a583245a6a89c`) → `/scratch/user/mhuang8/`; hash verified on CoreHPC.
- Extracted into `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project` with `--keep-old-files`. Target paths were verified absent beforehand, and no existing file was overwritten.
- Remote hashes equal local for runner v2 (`ed3e1055…`), wrapper v2 (`439c8ad1…`), and all builder scripts and tests. firewall_list `SHA256SUMS` verified OK on CoreHPC.
- The fixed `test_build_firewalled_corpus.py` (`53a67382…`) was later copied over its earlier CoreHPC copy and hash-verified.
- This updates §36 F/Status ("not uploaded"): runner v2 and wrapper v2 are now on CoreHPC with verified hashes.

### C. CoreHPC facts (as reported from CoreHPC)

- `external/ActFound/upstream_data/datas.zip` on CoreHPC: SHA-256 `74b1b30c…`, the same as the local copy.
- Pre-graph TSV present: `/scratch/user/mhuang8/actfound/chembl32/pregraph_production_20261113/chembl32_actfound_pregraph.tsv` (1,446,956,134 bytes), alongside `run_pregraph.sbatch` and `slurm-3048100` logs.
- Partitions include cpu (4 days), community, gpu, large_gpu, pod.

### D. Test runs

CoreHPC, pinned env (Python 3.9.18, RDKit 2023.03.1, torch 1.12.1; pytest via `PYTHONPATH` `/scratch/user/mhuang8/tools/pytest_actfound_py39`; `ACTFOUND_GRAPH_CODE_DIR=/scratch/user/mhuang8`), on `test_actfound_production_train_v2.py`, `test_actfound_production_train.py`, `test_actfound_model_side.py`, `test_build_split_and_config_v2.py` and `test_build_firewalled_corpus.py`:

- First run: 3 failed / 38 passed. Cause: the fixture could not find the graph code outside the project tree.
- After the fixture fix: 41 passed.

Local (miniforge base): route-C tests 3 passed; split/config tests 4 passed.

These are passing tests of the code. No builder has been run on real data, and nothing is deployed for production.

### Status (as of 2026-11-15)

- Firewalled corpus: not built.
- Config v2 and reconciliation v2: not generated.
- Benchmark: not run.
- Submissions: none.
- Code uploaded to CoreHPC and hash-verified (B); 41 tests pass on CoreHPC (D).
- Open for Miles: route C and the D013 Amendment 1 wording (A; drafted, not approved); review of runner v2's extra corpus-manifest guard (§36 F); whether to widen the firewall for the 180 largest-fragment exact matches (§36 C).
- D013 and its gates: unchanged. Production training and the 150-cell evaluation still need D013's gates and Miles's go-ahead.

Work stopped at: builders written, uploaded and tested on CoreHPC; nothing run on real data. Next is Miles's decisions on route C / Amendment 1, the corpus-manifest guard and firewall widening.

---

## 38. Miles's Decision on the §37 Open Items: Route C as Proposed Repair Path, Split-Aware Stop-Rule Fix and Read-Only Impact Count (2026-11-15)

Sources: Miles's reply in the main conversation [Miles 2026-11-15]. Thread and session (Tripwire CYP, Session 005) as identified by the main conversation.

This entry records Miles's decision as he stated it. It is not a D-entry: no D-number record has been created, and `notes/Decisions.md` is unchanged by this entry. D013 Amendment 1 remains Proposed (not approved) and is not recorded in any record file. This is not a session advance.

### A. Decision (verbatim)

Miles, 2026-11-15:

> "Proceed with route C as the proposed repair path. I authorize the narrow split-aware stop-rule fix and the read-only impact count, not the corpus build, reproduction run, benchmark or training yet.
>
> Make the reproduction check consume the hash-pinned upstream train/valid split. Fail closed on a missing or mismatched split input. Any field difference or missing/extra assay or molecule inside those splits must produce failure and block continuation. Formatting-only differences inside those splits also stop for review; don't introduce a tolerance or silently normalize them away. Differences outside those splits are reported, not blocking. Test both the blocking and report-only behavior.
>
> Keep the runner's matching-file-hash firewall-manifest check as defense-in-depth, not as proof of zero overlap. The actual built corpus must also pass the identity-overlap acceptance check under the approved exclusion rule.
>
> Count the incremental impact of the 180 largest-fragment matches, deduplicated against existing exclusions: molecules, rows and affected assays by train/valid split and protected tier. Distinguish directly countable loader attrition from any graph-stage effect that can't be established without rebuilding. No graph rebuild for this count.
>
> Don't widen the identity rule yet. Describe these as largest-fragment matches, not proven identical active compounds or activities. Leave first-block-only matches unchanged for now, without calling them safe. State whether the blinded-challenge identities are included in the current exclusion list.
>
> Keep Amendment 1 marked Proposed. Return the fix/test result, impact count and corrected amendment together so I can settle the remaining scope and authorize the next run in one step. Don't reopen settled architecture or adaptation choices."

### B. How this answers the §37 open items

1. Route C (§37 A): Miles said to proceed with route C "as the proposed repair path". D013 Amendment 1 stays Proposed; it is not approved.
2. Runner v2 corpus-manifest guard (§36 F, §37 Status): kept as defense-in-depth, "not as proof of zero overlap". The built corpus must also pass the identity-overlap acceptance check under the approved exclusion rule.
3. The 180 largest-fragment matches (§36 C): the identity rule is not widened. A read-only incremental impact count is authorized, with no graph rebuild.

### C. Authorized and not authorized

- Authorized: the narrow split-aware stop-rule fix (with tests of both blocking and report-only behavior) and the read-only impact count.
- Not authorized: corpus build, reproduction run, benchmark, training.

### D. Amendment 1 draft status correction

The assistant's scratchpad draft of D013 Amendment 1 had wrongly carried "Status: Approved". It is being corrected to "Proposed". No record file was affected: Amendment 1 is not written in `notes/Decisions.md` or any other record.

### Status (as of 2026-11-15)

- Route C: proceed as the proposed repair path; D013 Amendment 1: Proposed, not approved, not recorded.
- Split-aware stop-rule fix: authorized; not yet implemented or tested as of this entry.
- Impact count of the 180 largest-fragment matches: authorized (read-only, no graph rebuild); not yet run as of this entry.
- Identity rule: not widened. First-block-only matches: unchanged, not declared safe.
- Whether the blinded-challenge identities are in the current exclusion list: to be stated in the return to Miles; not established by this entry.
- Corpus build, reproduction run, benchmark, training: not authorized.
- D013 and its gates: unchanged.

Work stopped at: Miles decided the §37 open items. Next is the split-aware stop-rule fix with tests, the read-only impact count and the corrected Amendment 1 (Proposed), returned to Miles together.

---

## 39. Split-Aware Stop-Rule Fix (Local Test), Read-Only Impact Count of the 180 Largest-Fragment Matches, and Amendment 1 Draft Correction (2026-11-15)

Sources: results reported in the main conversation [assistant 2026-11-15]. Thread and session (Tripwire CYP, Session 005) as identified by the main conversation. Work authorized by Miles's decision in §38.

This is a routine factual results entry. It is not a decision and not a session advance. `notes/Decisions.md` is unchanged by this entry. D013 Amendment 1 remains Proposed (not approved).

### A. Split-aware stop-rule fix (edit plus passing local test; not uploaded to or tested on CoreHPC)

Evidence location: this checkout, `qualification/actfound/d013_firewall_20261115/`. Nothing has been run on real data.

- `build_firewalled_corpus.py`: was `0ac0a620…`, now `c017a650b604295b797331b23bce76c6531516d7ed5d0925cd329aa5ec7214a7`.
- `test_build_firewalled_corpus.py`: was `53a67382…`, now `885658c5eba117f688992074db22a69a18b8017a528c091684d2b37a514092a8`.
- Behaviour:
  - Reproduction mode requires `--upstream-split` and `--upstream-split-sha256`. A missing split or a hash mismatch stops the run before any output is written.
  - Split IDs are matched by the descriptor `assay_chembl_id_units_type` with "/" replaced by "_".
  - The comparison is exact, with no tolerance and no normalization.
  - Any field, missing-row, extra-row or assay-key difference in a train/valid assay sets gate FAIL. The outputs are kept and the exit code is 3.
  - Differences outside the splits are reported and do not block.
  - The differences TSV gains a `scope` column.
  - Firewall mode now refuses to run without a PASS reproduction manifest whose pregraph, authors-CSV, graph-code and script hashes match the current run.
- Tests, local env `tripwire_cyp002` (Python 3.12):
  - Command: `/opt/homebrew/Caskroom/miniforge/base/envs/tripwire_cyp002/bin/python -m pytest -q qualification/actfound/d013_firewall_20261115/` reported "15 passed"; `test_build_firewalled_corpus.py` contributes 10.
  - Covered: a formatting-only in-split difference blocks; a missing or extra in-split row blocks; a missing in-split assay blocks; an out-of-split difference is report-only (exit 0); a missing or mismatched split fails closed; firewall mode is blocked without a PASS manifest, with a FAIL manifest, or with a manifest made from different inputs.
  - Mutation check: with the gate forced to PASS, 4 blocking tests failed.

This is an edit with a passing local test. It is not uploaded to CoreHPC, not tested in the pinned env, and not deployed.

### B. Read-only impact count of the 180 largest_fragment_full matches (no graph rebuild)

Evidence location: assistant scratchpad on this machine (not this checkout, not CoreHPC):
- Script: `/private/tmp/claude-501/-Users-mileshuang-Desktop-tripwire-project/a2da1db5-40ca-4e93-9733-19876e392d12/scratchpad/lf180_impact.py` (`8d349ce9…`)
- Output: `/private/tmp/claude-501/-Users-mileshuang-Desktop-tripwire-project/a2da1db5-40ca-4e93-9733-19876e392d12/scratchpad/lf180_impact.json` (`6979f580…`)

Inputs: `firewall_list/` and the job-3066823 membership export (`membership.tsv.gz` `1ccfbc0a…`, rebuilt corpus `24d9fd29…`, v1 split labels). 2,773,995 rows were read, matching the export total.

- Molecules: 180 unique, 0 overlap with the 337-molregno firewall. Development 140, internal_test 40, blinded 0. All 180 are also largest_fragment_first_block matches.
- Rows (tier | split | loader status):
  - development: train kept 358, train relation_not_eq 23, validation kept 3, validation relation_not_eq 1.
  - internal_test: train kept 98, train relation_not_eq 3.
  - Outside the splits: development kept 534, relation 832, not-emitted 19; internal_test kept 223, relation 285, not-emitted 11.
- Molecules in train-kept rows: development 77, internal_test 27. In validation-kept rows: development 3.
- Distinct assays touched: train 134, validation 4, none 239.
- Loader attrition (≥20 kept-row rule only, after removing firewall plus 180, compared with firewall only):
  - train: 2 more assays fall below 20 (`566883_nM_IC50`, `969732_nM_IC50`) out of 33,184; 456 kept rows removed.
  - validation: 0 assays; 3 kept rows removed.
  - Not established by this count: the loader's std gate, and any graph-stage effect (√√N gate and degree changes after removal), which needs a rebuild.
- Wording: these are largest-fragment matches, not proven identical active compounds or activities. The identity rule is not widened. First-block-only matches are unchanged and not called safe.
- Blinded identities: the current exclusion list includes them (13 of 337 molregnos are tier blinded, from the 750 challenge_blinded_test structures among the 5,655 protected).

### C. D013 Amendment 1 draft correction (scratchpad only)

The scratchpad draft was corrected to "Status: Proposed" and updated with the stop rule, the requirement that the built corpus pass the overlap acceptance check, and the runner check as defense-in-depth. Path: `/private/tmp/claude-501/-Users-mileshuang-Desktop-tripwire-project/a2da1db5-40ca-4e93-9733-19876e392d12/scratchpad/D013_amendment1_draft.md` (`e35c8cd0…`). Amendment 1 is not approved and is not written in `notes/Decisions.md` or any other record.

### Status (as of 2026-11-15)

- Split-aware stop-rule fix: edited and passing locally (A); not uploaded to or tested on CoreHPC.
- Impact count of the 180 largest-fragment matches: done, read-only, no graph rebuild (B). Graph-stage effects and the loader's std gate not established.
- Blinded-challenge identities: included in the current exclusion list (B).
- Amendment 1 draft: corrected in scratchpad; Proposed, not approved, not recorded.
- Still not authorized: corpus build, reproduction run, benchmark, training.
- D013 and its gates: unchanged.

Work stopped at: fix tested locally, impact count and corrected draft ready for Miles. Next step, pending Miles: upload the fixed builder and tests to CoreHPC and test them in the pinned env.

---

## 40. Miles's Decision on the Exclusion Scope: Widen the Pretraining Exclusion to the 180 Largest-Fragment Matches, Including Validation (2026-11-15)

Sources: Miles's reply in the main conversation [Miles 2026-11-15]. Thread and session (Tripwire CYP, Session 005) as identified by the main conversation.

This entry records Miles's decision as he stated it. It belongs to D013 Amendment 1, which remains Proposed: its wording is not approved, and it is not written in `notes/Decisions.md` or any other record. `notes/Decisions.md` is unchanged by this entry. This is not a session advance.

### A. Decision (verbatim)

Miles, 2026-11-15:

> "Kets widen the exclusion to the 180. Keep full Standard InChIKey as the identity rule, but add largest-fragment full-key matches as a conservative pretraining exclusion. The reported direct data loss is small enough to justify that protection; the final graph/loader impact still needs checking after the build.
>
> One more thing: Apply the exclusion to the pretraining validation data too, since that data chooses the checkpoint. Don't leave the three validation rows just because they don't update weights."

("Kets" is quoted as given.)

### B. Context

- This settles the scope question left open in §39.
- "The 180" are the largest_fragment_full near-matches from the §39 B impact count: development 140, internal_test 40, blinded 0, none already on the 337-molregno full-key firewall.
- Identity rule: full Standard InChIKey stays the identity rule. The 180 are added as a conservative pretraining exclusion, not as identity matches. The §38 instruction "Don't widen the identity rule yet" is not contradicted: the identity rule is unchanged; the exclusion is widened.
- Before this decision, the assistant compared the options. It noted that ActFound featurizes the whole SMILES with a Morgan count fingerprint and does no salt stripping (`external/ActFound/dataset/data_base.py:93`), so a salt form gives a near-duplicate input. It recommended exclusion on the training side only. Miles's decision goes further: the exclusion also applies to the pretraining validation data (the 3 validation-kept rows from §39 B).

### C. Still open

- Implementation: firewall list v2, corpus manifest, acceptance check.
- The Amendment 1 wording.
- The reproduction-manifest requirement.
- Authorization of the CoreHPC upload, tests and reproduction run.
- Final graph/loader impact, to be checked after the build.

### Status (as of 2026-11-15)

- Exclusion scope: decided by Miles (A). Widened to the 180 largest_fragment_full matches, applied to pretraining train and validation data.
- Identity rule: full Standard InChIKey, unchanged.
- D013 Amendment 1: Proposed, not approved, not recorded outside this session log.
- Nothing is built or run.
- Still not authorized: CoreHPC upload, tests and reproduction run; corpus build, benchmark, training.
- D013 and its gates: unchanged.

Work stopped at: Miles decided the exclusion scope. Next, pending Miles: implementation of the firewall list v2, corpus manifest and acceptance check; Amendment 1 wording; the reproduction-manifest requirement; and authorization of the CoreHPC upload, tests and reproduction run.

---

## 41. Miles's Approval of the Firewall-v2 Implementation Plan and the Amendment 1 Exclusion and Acceptance Wording (2026-11-15)

Sources: Miles's reply in the main conversation [Miles 2026-11-15]. Thread and session (Tripwire CYP, Session 005) as identified by the main conversation.

This entry records Miles's approval as he stated it. It is not a D-entry: `notes/Decisions.md` is unchanged by this entry. D013 Amendment 1 as a whole remains Proposed: the reproduction-manifest requirement and the authorization of the CoreHPC run are still open, and the amendment is not recorded as a D-entry. This is not a session advance.

### A. Approval (verbatim)

Question asked: "Approve the implementation and amendment wording as proposed?"

Miles, 2026-11-15:

> "Approve as proposed"

### B. What was approved (the assistant's proposal following §40)

1. Firewall list v2 in a new folder `firewall_list_v2/`, with v1 (`firewall_list/`) kept unchanged.
   - It holds 517 molregnos: 337 full-key matches plus 180 largest-fragment full-key matches.
   - An `exclusion_basis` column records which rule matched each one.
   - First-block matches and largest-fragment first-block matches not already excluded stay report-only.
   - It is generated in the frozen `tripwire_cyp002` environment on the Mac from the existing hash-checked inputs.
2. A small change to the corpus builder: its manifest records the v2 list hash and a count for each exclusion basis. The filtering logic is unchanged.
3. A new acceptance check on the built corpus. It recomputes identities from the built TSV in the frozen identity environment, and passes only if no train or validation row has a full-key or largest-fragment full-key match. First-block matches are reported but do not fail it. It runs after the build, together with the graph/loader impact measurement.
4. Tests for each part, including that a largest-fragment match in a validation row fails acceptance.
5. The following wording, to be added to Amendment 1:

> "**Pretraining exclusion (conservative).** The identity rule remains the full Standard InChIKey. In addition, the pretraining corpus excludes every ChEMBL32 molregno whose largest-fragment full Standard InChIKey equals that of a protected OpenADMET structure (largest fragment = most heavy atoms, ties broken by canonical SMILES, computed in the frozen identity environment). The exclusion applies to all pretraining rows, train and validation alike, because validation selects the checkpoint. These are largest-fragment matches, not proven identical active compounds or activities; the exclusion is protective, not an identity claim. First-block-only and largest-fragment first-block-only matches remain report-only and are not deemed safe.
>
> **Acceptance.** The built corpus must contain no row, in train or validation, with a full-key or largest-fragment full-key match to a protected structure, verified by recomputing identities from the built file. The final graph-stage and loader impact is measured after the build and reported."

### C. Still open

- The reproduction-manifest requirement.
- Authorization of the CoreHPC upload, the tests there, and the reproduction run.
- A proposed correction to one sentence of §40 that Miles has not answered. It is not applied; §40 is unchanged.

### Status (as of 2026-11-15)

- Implementation plan (B 1–4): approved by Miles; not yet implemented or tested as of this entry.
- Amendment 1 exclusion and acceptance wording (B 5): approved by Miles. D013 Amendment 1 as a whole: Proposed, not recorded as a D-entry, not written in `notes/Decisions.md`.
- Identity rule: full Standard InChIKey, unchanged.
- Nothing is built or run.
- Still not authorized: CoreHPC upload, tests there and reproduction run.
- D013 and its gates: unchanged.

Work stopped at: Miles approved the implementation plan and amendment wording. Next: local implementation of firewall list v2, the builder manifest change, the acceptance check and tests. Pending Miles: the reproduction-manifest requirement, authorization of the CoreHPC upload, tests and reproduction run, and the proposed §40 sentence correction.

---

## 42. Firewall-v2 Implementation: Firewall List v2, Builder Manifest Change, Corpus Identity Acceptance Check and Local Tests (2026-11-15)

Sources: results reported in the main conversation [assistant 2026-11-15]. Thread and session (Tripwire CYP, Session 005) as identified by the main conversation. Work authorized by Miles's approval in §41.

This is a routine factual results entry. It is not a decision and not a session advance. `notes/Decisions.md` is unchanged by this entry. D013 Amendment 1 as a whole remains Proposed.

Evidence location: this checkout, `qualification/actfound/d013_firewall_20261115/`, except the Amendment 1 draft (assistant scratchpad, E). All of this is code plus local passing tests and one local artifact generation. Nothing has been uploaded to CoreHPC, built or run on the corpus.

### A. Firewall list v2 (local artifact generation)

- New script `build_firewall_list_v2.py` (`625fb140…`), with test `test_build_firewall_list_v2.py` (`c9e405aa…`). It is a copy of v1 with changed output logic.
- v1 is unchanged: `build_firewall_list.py` (`b1280e09…`) and `firewall_list/`.
- Generated in the frozen `tripwire_cyp002` env (Python 3.12.14, RDKit 2026.03.6, InChI 1.07.3) from the hash-checked inputs: export molecules `e7f756f7…`, audit `c0881d8f…`, full_run `SHA256SUMS`. Runtime 31 s.
- Output `firewall_list_v2/`:
  - `exclusion_molregnos.tsv` `b3f70e56…`
  - `near_matches_not_removed.tsv` `9099c12a…`
  - `summary.json` `2dbe1114…`
- Contents:
  - 517 molregnos excluded.
  - `full_standard_inchikey` basis: blinded 13, development 263, internal_test 61.
  - `largest_fragment_full_inchikey` basis: development 140, internal_test 40.
  - Report-only: first_block 219, largest_fragment_first_block 281.
- Cross-check: the v2 set equals exactly the v1 337 ∪ the 180 v1 largest_fragment_full near-matches, with the basis assigned correctly.

### B. Corpus builder change

- `build_firewalled_corpus.py`: was `c017a650…` (§39), now `c73d6a7c…`.
- `test_build_firewalled_corpus.py`: was `885658c5…` (§39), now `ebb5ed97…`.
- The exclusion list must have an `exclusion_basis` column with known values and no duplicate molregnos; otherwise the build stops (fail closed).
- The manifest records `firewall_molregnos_by_exclusion_basis` and `filter.rows_removed_by_exclusion_basis`.
- The filtering logic is unchanged: rows are removed for all assays regardless of split, so validation is covered.

### C. New acceptance check: `check_corpus_identity.py`

- `check_corpus_identity.py` (`0ac187e4…`), with test `test_check_corpus_identity.py` (`639d9e8e…`).
- Runs on the Mac in the frozen env, on the built corpus TSV copied from CoreHPC.
- Recomputes full and largest-fragment keys from the built file's SMILES.
- FAILs (exit 3, outputs kept) on any of the following in a train or valid row:
  - a full-key or largest-fragment full-key match;
  - an uncomputable identity;
  - a molregno that is on the exclusion list.
- Report-only: findings outside the splits, and first-block or largest-fragment first-block matches.
- Hash-checks every input and refuses to overwrite its output.

### D. Tests (local, `tripwire_cyp002`)

- Command: `/opt/homebrew/Caskroom/miniforge/base/envs/tripwire_cyp002/bin/python -m pytest -q qualification/actfound/d013_firewall_20261115/` reported "24 passed". The `overlap_audit_20261115` tests reported "4 passed".
- Covered: a largest-fragment match in a validation row fails; a full-key match in a train row fails; outside-split and first-block matches are report-only; uncomputable identities and excluded molregnos fail closed; an input hash mismatch stops the check; firewall mode refuses a v1-format list.
- One test expectation was corrected during development: a one-fragment molecule also matches on largest-fragment first block. The code was right; the test was wrong.

These are edits with passing local tests. They are not uploaded to CoreHPC, not tested in the pinned env, and not deployed.

### E. Amendment 1 draft (scratchpad only, still Proposed)

The approved §41 wording has been added as item 6 (pretraining exclusion and acceptance), and the closing line now says "full Standard InChIKey identity rule". Path: `/private/tmp/claude-501/-Users-mileshuang-Desktop-tripwire-project/a2da1db5-40ca-4e93-9733-19876e392d12/scratchpad/D013_amendment1_draft.md`. Amendment 1 is not recorded as a D-entry and is not written in `notes/Decisions.md`.

### Still open

- The reproduction-manifest requirement.
- Authorization of the CoreHPC upload, the pinned-env tests and the reproduction run.
- The proposed §40 sentence correction, which Miles has not answered. It is not applied; §40 is unchanged.

### Status (as of 2026-11-15)

- Implementation plan (§41 B 1–4): implemented, with passing local tests (A–D). Not uploaded to CoreHPC, not tested in the pinned env, not run on the corpus.
- Firewall list v2: generated locally (A).
- Amendment 1 draft: §41 wording added in scratchpad (E). D013 Amendment 1 as a whole: Proposed.
- Identity rule: full Standard InChIKey, unchanged.
- Still not authorized: CoreHPC upload, pinned-env tests and reproduction run; corpus build, benchmark, training.
- D013 and its gates: unchanged.

Work stopped at: local implementation and tests done. Pending Miles: the reproduction-manifest requirement, authorization of the CoreHPC upload, pinned-env tests and reproduction run, and the proposed §40 sentence correction.

---

## 43. Miles's Approval of the Corrected D013 Amendment 1 for Recording (2026-11-15)

Sources: Miles's reply in the main conversation [Miles 2026-11-15]. Thread and session (Tripwire CYP, Session 005) as identified by the main conversation ("Yes, log to 005").

This entry records Miles's approval and the resulting decision-record write. It is not a session advance.

### A. Approval (verbatim)

Question asked: "Approve the corrected Amendment 1 for recording?"

Miles, 2026-11-15:

> "Approve for recording"

### B. Decision record written

- File: `notes/Decisions.md`, appended at the end of the D013 record (D013 was the last record in the file).
- Heading: "## D013 Amendment 1 — Graph re-qualification route and conservative pretraining exclusion for the firewalled corpus (2026-11-15)", a subsection of D013, at lines 6246–6298.
- Source text: the corrected draft at `/private/tmp/claude-501/-Users-mileshuang-Desktop-tripwire-project/9f222e17-2a4f-422a-b294-b8dd31356156/scratchpad/D013_amendment1_corrected.md`, recorded verbatim. The one change was the status line: "**Status:** Proposed (not approved; not recorded in Decisions.md)" became "**Status:** Approved (2026-11-15). Recorded on Miles's approval: \"Approve for recording\"."
- Existing content of `notes/Decisions.md`, including the uncommitted D012 addendum and the D013 text, was not changed.

### C. What the corrected amendment contains

- It supersedes the earlier scratchpad draft, `/private/tmp/claude-501/-Users-mileshuang-Desktop-tripwire-project/a2da1db5-40ca-4e93-9733-19876e392d12/scratchpad/D013_amendment1_draft.md`. That draft's hash was `e35c8cd0…` when §39 C recorded it, before the §42 E update. The main conversation described this earlier draft as the fork session's draft.
- Change 1: graph re-qualification (route C), with the exact qualification gate. The gate text includes the requirement that the excluded build refuses to run without a PASS reproduction manifest.
- Change 2: the §41-approved pretraining exclusion and acceptance wording. The Acceptance paragraph adds one sentence: the training runner's corpus-manifest check is defense-in-depth, not proof of zero overlap.
- A scope table reconciling the near-match counts:
  - full-key matches: 337;
  - largest-fragment full-key matches: 180, under the heavy-atom rule;
  - first-block matches: 219;
  - largest-fragment first-block matches: 461 before Change 2, 281 after;
  - exclusion list v2: 517.
- A reconciliation of the earlier census count of 184. It equals the 180 plus 4 molregnos (674418, 834692, 996164, 1248748), which the RDKit default `LargestFragmentChooser` selected. Those 4 are not excluded under the approved heavy-atom rule.

### D. Not authorized

The amendment states: "This amendment does not authorize the corpus build, the reproduction run, the benchmark or training." The corpus build, the reproduction run, the benchmark and training remain unauthorized. The CoreHPC upload and pinned-env tests are not authorized by this approval.

### Status (as of 2026-11-15)

- D013 Amendment 1: approved and recorded in `notes/Decisions.md` (B).
- Identity rule: full Standard InChIKey, unchanged.
- Nothing is built or run.
- Still not authorized: CoreHPC upload, pinned-env tests, corpus build, reproduction run, benchmark, training.
- Still open: the proposed §40 sentence correction, which Miles has not answered. §40 is unchanged.

Work stopped at: Miles approved the corrected Amendment 1, and it is recorded in D013. Pending Miles: authorization of the CoreHPC upload, pinned-env tests and reproduction run, and the proposed §40 sentence correction.

---

## 44. Fork/Main Merge: Fork Work Made Canonical and Merged-State Verification (2026-11-15)

Sources: supersession and verification results reported in the main conversation [assistant 2026-11-15]; Miles's merge instruction [Miles 2026-11-15]. Thread and session (Tripwire CYP, Session 005) as identified by the main conversation ("Yes, log to 005").

This is a routine factual results entry. It is not a decision, carries no D-number, and is not a session advance. `notes/Decisions.md` is unchanged by this entry.

Context: a fork of the main conversation ("Tripwire-CYP-005 (fork)", session `local_a2da1db5-40ca-4e93-9733-19876e392d12`) ran in parallel in this same checkout and produced §36–§42. Miles's instruction, verbatim:

> "if the fork supercedes your work, use the fork's work. once you finish the merge I'll delete the fork"

The fork's work is canonical from here, and the main conversation continues the implementation.

### A. Supersession (verified in the main session on 2026-11-15)

- **Canonical:** `qualification/actfound/d013_firewall_20261115/` — the fork's route-C builders, firewall list v1 and v2, and acceptance check. All 10 scripts match the SHA-256 values recorded in §42. `firewall_list_v2/SHA256SUMS.txt` verified OK (`exclusion_molregnos.tsv`, `near_matches_not_removed.tsv`, `summary.json`).
- **Canonical:** the fork's versions of the production runner v2 files in this checkout:

| File (this checkout) | SHA-256 |
|---|---|
| `cyp/scripts/actfound_production_train_v2.py` | `ed3e1055b1a9f5a76fc6b0ddad0df7e4a2df95254e513aca90e7359615db1865` |
| `cyp/scripts/actfound_production_train_v2.slurm` | `439c8ad1bb43e5bcb4535e28e8eeb69646de18f10e30b47d7fe4c50d4db123e9` |
| `cyp/tests/test_actfound_production_train_v2.py` | `e88e61e0ed6dcd14c308f77b053423114b52c6241adf45e16e69b902757593e6` |

The main session's earlier versions of these three files were overwritten on disk by the fork and are not recoverable from this checkout.
- **Superseded but RETAINED:** `qualification/actfound/firewall_corpus_20261115/` — the main session's earlier converter-route scripts `prohibited_identities.py`, `build_firewalled_corpus.py`, `test_build_firewalled_corpus.py`, and the outputs under `prohibited/`. Reason for retention: `prohibited/near_matches_report_only.tsv` and `prohibited/` are cited twice as evidence in the approved D013 Amendment 1 in `notes/Decisions.md` (the 184 = 180 + 4 census reconciliation, and the 219 first-block set), and Miles's standing rule is to preserve intermediate and validation artifacts until the relevant stage is frozen. It is not used going forward. Nothing in it was deleted or moved.
- **Shared dependency, not superseded:** `qualification/actfound/overlap_audit_20261115/` — the fork's `build_firewall_list.py` and `build_firewall_list_v2.py` read its `full_run` outputs.

### B. Merged-state verification (main session, local, frozen env `tripwire_cyp002`)

Environment: Python 3.12.14 / RDKit 2026.03.6 / InChI 1.07.3 where applicable.

- `cyp/tests/test_actfound_production_train_v2.py` + `cyp/tests/test_actfound_production_train.py`: 20 passed.
- `qualification/actfound/d013_firewall_20261115/`: 24 passed.
- `qualification/actfound/overlap_audit_20261115/`: 4 passed.
- `qualification/actfound/firewall_corpus_20261115/` (superseded, run for the record): 3 passed.
- `bash -n` on `cyp/scripts/actfound_production_train_v2.slurm`: OK.
- Total: 51 passing tests.

All of these are local tests of code. Nothing was uploaded, built or run on the corpus in this merge.

### C. State unchanged by the merge

- Nothing uploaded to CoreHPC; no corpus build, no reproduction run, no benchmark, no training.
- D013 and its gates: unchanged.
- D013 Amendment 1: recorded and Approved (§43).

### D. Still open for Miles (carried from §42, unchanged by the merge)

- The reproduction-manifest requirement.
- Authorization of the CoreHPC upload, the pinned-env tests and the reproduction run.
- The proposed §40 sentence correction. It is not applied; §40 is unchanged.

Work stopped at: the fork/main merge is complete, with the fork's work canonical and 51 local tests passing in the merged state. Pending Miles: the reproduction-manifest requirement, authorization of the CoreHPC upload, pinned-env tests and reproduction run, and the proposed §40 sentence correction.

---

## 45. D013 CoreHPC Deploy, Pinned-Env Tests and Reproduction Job 3073155 Submitted (2026-11-15)

Sources: upload, hash and test results reported in the main conversation [assistant 2026-11-15]; Miles's authorization, pause instruction and submission report [Miles 2026-11-15]. Thread and session (Tripwire CYP, Session 005) as identified by the main conversation ("Yes, log to 005").

This is a routine factual results entry. It is not a decision, carries no D-number, and is not a session advance. `notes/Decisions.md` is unchanged by this entry.

### Authorization (verbatim)

Question asked: "Authorize the CoreHPC upload, pinned-env tests and reproduction run?"

Miles, 2026-11-15:

> "Authorize it"

The assistant then paused CoreHPC work at Miles's request:

> "pause where you are now, your work on corehpc is burning through credits too quickly. I'll run the commands"

Miles submitted the reproduction job himself and reported:

> "Submitted batch job 3073155"

### A. Shared connection

OpenSSH ControlMaster to `mhuang8@chpc-ucsf-login-vm1.corehpc.ucsf.edu` via ProxyJump bastion `chpc-ucsf-bastion-vm1`, socket `~/.ssh/cm-corehpc.sock`, `ControlPersist 10h`, `ServerAliveInterval 60`. Miles authenticated password + Duo; the master reported `MASTER_READY`.

### B. Upload and deploy (assistant, authorized)

- Bundle `~/Downloads/d013_upload2_20261115/d013_code2_20261115.tar.gz`, SHA-256 `ab2eb4aec5885e0a01b41ab768befb682f37483e5f0d6d6c604d5172ffd24450`, 18 files of `qualification/actfound/d013_firewall_20261115/`. Copied to `/scratch/user/mhuang8/` and extracted to staging `/scratch/user/mhuang8/d013_staging_b`. Every staged file's SHA-256 equalled the local copy.
- Copied from staging into `/scratch/user/mhuang8/d012_runtime_qualification_20261110/tripwire-project/qualification/actfound/d013_firewall_20261115/`. All 18 post-copy hashes equal local.
- Changed on CoreHPC by this deploy:

| File | Before | After |
|---|---|---|
| `build_firewalled_corpus.py` | `0ac0a620…` | `c73d6a7c48cb7547f7d284ac732df6302cb4511d4e729c542ab7e3b0122bbd54` |
| `test_build_firewalled_corpus.py` | `53a67382…` | `ebb5ed9701bf5a16884f312567df332b15ab7bb608c4f086d5995329d35aae92` |

  CoreHPC had been at the pre-§39 versions; the §39 and §42 updates had never been uploaded.
- Newly added on CoreHPC: `build_firewall_list_v2.py` (`625fb140…`), `check_corpus_identity.py` (`0ac187e4…`), `test_build_firewall_list_v2.py` (`c9e405aa…`), `test_check_corpus_identity.py` (`639d9e8e…`), and `firewall_list_v2/` (`exclusion_molregnos.tsv` `b3f70e56…`, `near_matches_not_removed.tsv` `9099c12a…`, `SHA256SUMS.txt` `cc567f11…`, `summary.json` `2dbe1114…`).
- Unchanged: `build_firewall_list.py` (`b1280e09…`), `build_split_and_config_v2.py` (`95bc0390…`), `test_build_firewall_list.py` (`f904cff3…`), `test_build_split_and_config_v2.py` (`c672327f…`), `firewall_list/` (4 files).

### C. Tests in the pinned ActFound env on CoreHPC

Environment: Python 3.9.18, RDKit 2023.03.1, torch 1.12.1; pytest via `PYTHONPATH /scratch/user/mhuang8/tools/pytest_actfound_py39`; `ACTFOUND_GRAPH_CODE_DIR=/scratch/user/mhuang8`. Scope: `qualification/actfound/d013_firewall_20261115/`, `cyp/tests/test_actfound_production_train_v2.py`, `test_actfound_production_train.py`, `test_actfound_model_side.py`.

Result: **49 passed, 9 failed.**

- All 9 failures have one identical cause, confirmed by collecting the error lines: `AttributeError: module 'rdkit.Chem.inchi' has no attribute 'GetInchiVersion'` (9 occurrences, no other E-line).
- The 9 are all in the three frozen-identity-environment scripts: `test_build_firewall_list.py` (1), `test_build_firewall_list_v2.py` (2), `test_check_corpus_identity.py` (6). Those scripts are designed to run only in the frozen identity env (RDKit 2026.03.6) and refuse any other; they pass there (24 passed locally, §42/§44).
- Assistant assessment, labelled as such: these are environment-gate failures, not defects in the CoreHPC execution path. The CoreHPC-side scripts (`build_firewalled_corpus.py`, `build_split_and_config_v2.py`) and the runner/model-side tests are among the 49 passed.

### D. Reproduction-run inputs (extracted/verified on CoreHPC immediately before submission)

All equal the values pinned in D013 Amendment 1.

| Input (CoreHPC) | SHA-256 |
|---|---|
| pregraph `/scratch/user/mhuang8/actfound/chembl32/pregraph_production_20261113/chembl32_actfound_pregraph.tsv` | `43dfd672e790ba3f92bd1e17b5902a44aec88911a5bbf0db26f92d0e192db076` |
| authors' CSV extracted from `datas.zip` to `/scratch/user/mhuang8/actfound/chembl32/authors_20261115/chembl_processed_chembl32.csv` | `a5b562d4cd237f94f80b1d6a85031825bc1b0b1157e734f76bc3315f256c5e20` |
| upstream split `.../external/ActFound/datas/chembl/chembl_split.json` | `7f3cd5988bfd3d608a8f6c120b54e19d932e0b3c026ec5e43b9798e0d2a6e3e1` |
| graph code `/scratch/user/mhuang8/actfound_graph_stage.py` | `1783c5fe8f237261e7f5a246910a8f2bad1cf990a8a6ac11a07af04b645a9995` |
| graph code `/scratch/user/mhuang8/actfound_chembl32_projection.py` | `22fca4084849fd58e791f80c94d24e9b282e1288900a37ddbf31c0872dcf8f3b` |

### E. Submission (pre-submission record, Q7)

Miles ran the assistant-provided `sbatch` command himself. Slurm job **3073155**.

- Partition `cpu`, `--cpus-per-task=4`, `--mem=64G`, `--time=8:00:00`, `--job-name=d013-repro`, `--output=/scratch/user/mhuang8/d013_repro_%j.out`, submitted with `--wrap`.
- Runner: the pinned-env `python -B` on `.../d013_firewall_20261115/build_firewalled_corpus.py` with the four inputs above, `--graph-code-dir /scratch/user/mhuang8`, `--out-dir /scratch/user/mhuang8/actfound/chembl32/d013_reproduction_20261115`, and **no** `--firewall-list` (reproduction mode).
- The resource request is an assistant estimate, not a measurement. `MaxRSS` and `Elapsed` are to be read from `sacct` afterwards.
- Expected semantics: exit 0 = gate PASS; exit 3 = an in-split difference, outputs kept, stop for review.

### F. Result

**UNKNOWN.** The job had only just been submitted as of this entry. No `sacct` state, exit code, runtime or output has been reported.

### Status (as of 2026-11-15)

- Code and both firewall lists: deployed and hash-verified on CoreHPC (B).
- Pinned-env tests: 49 passed, 9 failed, all 9 from the frozen-identity environment gate (C).
- Reproduction job 3073155: submitted by Miles; result UNKNOWN (E, F).
- Still not authorized: corpus build, benchmark, training.
- D013 and its gates: unchanged.
- Still open: the proposed §40 sentence correction, which Miles has not answered. It is not applied; §40 is unchanged.

Work stopped at: reproduction job 3073155 submitted on CoreHPC, with no result yet. Pending: the `sacct` state, exit code, runtime and outputs of 3073155; and from Miles, the proposed §40 sentence correction.
