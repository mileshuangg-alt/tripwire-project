# Tripwire-PXR-004 Session Log

**Date:** 2026-11-19
**Status:** OPEN.

**Workstream:** Tripwire — PXR induction arm; continuation after Tripwire-PXR-003 (closed 2026-11-18; `notes/Tripwire_PXR_003.md`).

## 1. Session purpose and starting state

Study sequential auxiliary pretraining on pEC50 with two arms from the same released CheMeleon encoder: direct pEC50 fine-tuning (arm 1) against first training on the eligible single-concentration PXR responses and then fine-tuning on pEC50 (arm 2), with the downstream setup matched. Starting state is the PXR-003 close (D019; `notes/Tripwire_PXR_003.md`). Continuity reference: `notes/Handoffs/Tripwire_PXR_003_Handoff_20261119.md`, which is not decision authority.

## 2. Scope decision (D020)

D020 (`notes/Decisions.md`, scope direction only) records the two-arm scope, the carried-forward data exclusions and evaluation boundaries (D014 split; D017-D019 eligibility; holdout consumed and closed; official test unused), and the next action: a compatibility review followed by a proposal for a bounded pilot, before any implementation or training. No code, installation, training or submission is authorized.

Decided within D020 (2026-11-19): both arms use descriptor-free inputs for the pilot. This is a design choice, not a CheMeleon requirement: the Chemprop 2.3.1 foundation-model CLI (`chemprop/cli/train.py`) rejects atom-features, atom-descriptors and bond-features paths with CheMeleon and requires the V2 multi-hot atom featurizer, but does not reject molecule-level descriptors downstream of the encoder. Added molecule-level descriptors may be tested separately afterward (not decided, not scheduled).

Decided within D020 (2026-11-19): cohort. Both arms use the full 2,897 primary-train molecules; OADMET-0003727 (auranofin) is restored, and the v1 frame (`pxr/artifacts/pxr002_aux/`: 2,897 primary-train plus 7,922 auxiliary-only, 10,819 training molecules, 621 validation) is used with descriptor-free inputs. This follows a read-only inspection run in `tripwire_cyp002` (Chemprop 2.3.1, RDKit 2026.03.6; scratchpad script `inspect_auranofin.py`, not in the repository). Findings: the SMILES parses as one fragment (32 atoms, 32 bonds), stereocentres retained, canonical SMILES reproduces the input; the V2 graph is finite (node 32x72, edge 64x14). The molecule is primary_train, eligible, D014 group 1620 (scaffold C1CCOCC1), train split, pEC50 2.54, no auxiliary labels; its only prior exclusion was the descriptor-based amendment (D018 record), which does not apply to descriptor-free inputs, and no other eligibility rule excludes it. The consumed holdout was not used for the choice. Neural PXR-002 used 2,896; PXR-003 baselines used 2,897; PXR-004 uses 2,897 in both arms.

Disclosed caveat (not an exclusion): the metal centre is encoded P(triethyl)=Au-S-(acetylated thioglucose) and RDKit treats P=Au as a double bond (Au valence 3, P valence 5), a formal-bond depiction rather than a physically accurate bond order. P and S have explicit V2 element bins (index 14, 15); Au (Z=79) is outside the V2 explicit element list (Z 1-36 and 53) and maps to the generic unknown-element bin (index 37; hybridization SP3D2). Au is the only element outside the V2 list in the 10,819-molecule training frame, so only this molecule is affected. CheMeleon pretraining coverage of that bin was not checked.

## 3. Pilot design (D020, amended in place 2026-11-19)

The pilot design is approved and recorded in D020 (Pilot design sections; full text and basis labels there). Approval covers the design and its documentation only; it does not authorize implementation, installation, downloads (including the CheMeleon weights), training, timing runs or submission. All execution commands are run by the user. Preserved: descriptor-free two-arm design, three paired seeds, and the evaluation firewall (621 holdout consumed and closed, official test unused, no train+validation retraining).

Summary of what was fixed:
- **Arms and encoder.** Both arms fully fine-tune the CheMeleon encoder with a fresh FFN head (300 hidden units, 1 layer, dropout 0; library default, not CheMeleon-derived). Arm 2 adds stage 1 on the three-head log2FC auxiliary labels, then stage 2 as arm 1. Frame: v1 (2,897 primary-train plus 7,922 auxiliary-only; 10,819 training; 621 validation); 9,847 training molecules have an auxiliary label (head counts 9,737 / 8,540 / 606).
- **Stage-1 monitor.** Fixed D014-group-held-out draw (seed 20261119, about 10% of auxiliary-labelled molecules), shared across seeds, to be saved as a hashed artifact; group 417 stays in training. Trial draw during review (not an artifact): 951 groups, 997 molecules, 985 labelled, monitor labels 968 / 837 / 61, 179 primary-train. Selection by pooled masked MSE in standardized units (scalers fit on stage-1 training labels only). Limitations are listed in D020 (large heads dominate; 99 uM head experiment-confounded; near molecule-level separation as 9,712 of 10,000 training-pool groups are singletons; one draw, noise unquantified; about 10% of auxiliary labels lost to stage 1; the monitor does not assess group 417).
- **Stopping and optimisation.** Maximum 100 epochs, patience 20, min_delta 0 in both stages; stage 2 follows the executed PXR-002 protocol (`pxr/scripts/pxr002_train.py`), stage 1 is a disclosed pilot convention. Batch 64 and stock Chemprop LR schedule (executed project precedent, also Chemprop CLI defaults). Stage 2 restarts optimizer state and scheduler/warmup in both arms; head initialization and downstream randomness are paired per seed.
- **External sources.** CheMeleon paper main text (arXiv 2506.15792 v2) gives full fine-tuning with a fresh head and no hyperparameter optimization, but not the downstream epoch, LR, batch or FFN settings; its Supporting Information was not read. The user reports that the authors' paper-era fine-tuning code uses 50 epochs, patience 5 on validation loss and a 256-unit head; this is not verified here, and the pilot deliberately differs.
- **Reporting.** Validation MAE, Spearman and RMSE per seed and mean, with paired per-seed differences, descriptive only (no threshold, no test). RMSE needs primary-head validation predictions saved at the best epoch (new relative to PXR-002). Context points, not matched comparators: constant mean 0.9218; ECFP4 + LightGBM 0.6049; PXR-002 single-model mean about 0.546 (descriptor-based, 2,896 cohort).
- **Timing.** Feasibility only, never evidence of learning, and nothing is chosen from it; route not chosen. Standing pre-Slurm rules apply.

Resolved by this design: encoder frozen versus fine-tuned, stage-1 head/loss/budget, comparison criterion, RMSE as a reported metric.

Still open and unapproved (the design-time statement; implementation and the weights download were subsequently approved, see section 4): creation of the hashed monitor artifact; timing route; any production training or submission; check of the CheMeleon Supporting Information or repository fine-tuning configuration; any numeric threshold or statistical test.

## 4. Implementation and CheMeleon weights (2026-11-19)

**Approval and scope.** Implementation of the pilot code and tests per D020, and the download of the CheMeleon weights, were approved. Not approved: running tests, training, timing or Slurm; the user runs all execution commands.

**Weights.** Downloaded with `curl --fail` from `https://zenodo.org/records/15460715/files/chemeleon_mp.pt` (the URL Chemprop 2.3.1 uses for `--from-foundation CHEMELEON`) to `pxr/artifacts/pxr004_weights/chemeleon_mp.pt`: HTTP 200, 34,859,448 bytes, SHA-256 `c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651`. `SHA256SUMS` (sha256 `7e0f7dc0ffa49433ed9399e82c68716f8bddc954bacfd6dea43917ab9df8f764`) and `PROVENANCE.txt` sit beside it. The file was inspected only with `torch.load(weights_only=True)`: keys `hyper_parameters` and `state_dict`; four tensors (W_i.weight 2048x86, W_h.weight 2048x2048, W_o.weight 2048x2120, W_o.bias 2048); 8,714,240 parameters; d_v 72, d_e 14, d_h 2048, depth 6, dropout 0.0, bias False, activation relu, undirected False. This matches the paper's 2048 width, depth 6 and 8.7M-parameter encoder. The weights are not yet copied to CoreHPC.

**Code (edit made, syntax-checked only).** Three new files were written; only a `py_compile` syntax check has passed. They have not been executed or tested: there is no passing test and no deployment verification.
- `pxr/scripts/pxr004_data.py` (sha256 `430c7e968f4a3baaec310e175903f787d255dec9415baf2d1e383d277cfa1fbd`): hash-verified v1 frame loading, D014 group attach, seeded monitor draw (seed 20261119, 10%, group 417 excluded), hashed monitor-artifact builder, descriptor-free V2 datasets, stage-1 scaler fit on post-split training labels only.
- `pxr/scripts/pxr004_train.py` (sha256 `afb5d1e9d94e9ebb6d106f76409b771c6bbd5f7ccd5b652768999654e0e44385`): two-arm trainer (direct, sequential); stage-1 pooled masked MSE monitor in standardized units with per-head MAE and Spearman logged; stage-2 primary validation MAE, RMSE and Spearman with best-epoch validation predictions saved; only encoder weights carried from stage 1 to stage 2 (fresh optimizer and scheduler); stage 2 seeded identically in both arms for paired head initialization.
- `pxr/tests/test_pxr004.py` (sha256 `29f5569c30d62580d95705eb77046b656f4bdbabb1fa841d9a416d02a2e659ce`).

PXR-002 helpers are reused only through their frozen artifact hashes; no PXR-002 file was edited. No Slurm script exists.

**Not yet done (status at the implementation entry; superseded by section 5).** The monitor artifact is not built. The trial draw figures quoted in section 3 (951 groups, 997 molecules) came from a draw that did not exclude group 417, so the built artifact's counts may differ; the artifact is authoritative once built. Tests have not been run; no timing; no Slurm.

## 5. Tests and monitor artifact (2026-11-19, later entry)

Dated note: the trial-draw figures in section 3 and in D020 (951 groups, 997 molecules, 985 labelled, monitor labels 968 / 837 / 61, 179 primary-train) remain the labelled trial figures. The built artifact below is the authoritative monitor and differs from them, as anticipated in section 4.

**Tests.** The user ran `pxr/tests/test_pxr004.py` with the `tripwire_cyp002` interpreter and reports that all tests pass; the pytest summary line with counts was not supplied. The tests exercise `pxr004_data.py` (sha256 `430c7e96...1fbd`), `pxr004_train.py` (sha256 `afb5d1e9...4385`) and the test file (sha256 `29f5569c...6ce`) as hashed at the implementation entry (section 4); these files were not re-hashed after the run. An earlier run in the wrong interpreter (`.venv`, Python 3.15, no Chemprop, no scipy) gave 2 failed, 5 passed, 9 skipped; the failures were the missing scipy in that interpreter, not a code defect, and that run is superseded by the correct one.

**Monitor artifact.** Built by the user with `pxr/scripts/pxr004_data.py build-monitor` into `pxr/artifacts/pxr004_monitor/` (one draw, seed 20261119, 10%, group 417 excluded). Console output and a file check agree:

| Quantity | Artifact value |
|---|---|
| Monitor groups | 940 |
| Monitor molecules with auxiliary labels | 985 (10% target of 9,847 is 984.7) |
| Monitor labels per head (8.25 / 33 / 99 uM) | 974 / 848 / 67 |
| Monitor primary-train molecules | 177 |
| Labelled training molecules | 9,847 |
| Stage-1 training molecules | 8,862 |

Files: `pxr004_monitor_assignments.csv` (538,921 bytes; 10,819 rows, 1,002 in the monitor, of which 985 carry auxiliary labels; sha256 `3ad2d8bf9f766d9283798216b68ffc25b96f7d9ddef802dd13c56a2738f2b030`); `SHA256SUMS` verified with `shasum -c`; `PROVENANCE.txt` (sha256 `c57d84732d75b1f01bf70428a3c85fb2777d4871f1a6ffe72ff4919511f1c391`) records the script hash `430c7e96...`. Group 417 is not in the monitor (checked).

## 6. Timing script and test-run hash check (2026-11-19, later entry)

**Re-hash.** `pxr004_data.py` (`430c7e96...1fbd`), `pxr004_train.py` (`afb5d1e9...4385`) and `test_pxr004.py` (`29f5569c...6ce`) were re-hashed and are unchanged from the section 4 values. The passing test run in section 5 was therefore against these exact files.

**Timing script (edit made, syntax-checked only).** At the go-ahead for a timing script, two files were written; only `py_compile` has passed. Neither has been run or tested.
- `pxr/scripts/pxr004_timing.py` (sha256 `63a7933d542d90b17c5894c7dcb3eebc191d66c0f9563ae662ac1de61c41f208`).
- `pxr/tests/test_pxr004_timing.py` (sha256 `e2087c1a3ce75d5a154cfbf895cac37d8b1f0c87198aad9e596fe06cd578d036`); arithmetic tests only (batches per epoch, summaries, worst-case run seconds, totals).

The script is a feasibility check only, per D020. For each stage it builds the real datasets and the CheMeleon-initialised model, times a few forward/backward/optimizer steps on a throwaway model after warmup, one full validation pass and one checkpoint write. It never reads, prints or stores a loss, writes nothing under `pxr/artifacts`, and refuses to overwrite its `--out-json`. It extrapolates a worst case (every run to the 100-epoch maximum, validation and checkpoint write every epoch; 3 stage-1 runs and 6 stage-2 runs) and reports the serial total, one sequential-arm seed and one direct-arm seed, compared with a user-supplied requested wall time if given. The accelerator (cpu, gpu or mps) is the user's choice and is recorded in the output; no route has been chosen.

2026-11-19: all tests were reported to have passed locally. The report did not state which test files were run, the pytest counts or the interpreter; that it refers to `pxr/tests/test_pxr004_timing.py` (sha256 `e2087c1a...d036`) is an inference and is unconfirmed. The timing run itself has not been done; the route (local vs CoreHPC GPU, interactive vs batch) is not yet chosen.

## 7. Staging, interactive failure and timing route (2026-11-19, later entry)

**Staging.** A 16-file package was built by the assistant: `pxr004_stage_20261119.tar.gz` (hash list `pxr004_stage_hashes_20261119.txt`, which has its own sha256 `7efe0c1f98aca2c3e8e08f1975db5ca9823f3679ee37d5e28c11d670da6ec17a`; file list `pxr004_stage_files_20261119.txt`). Contents: `pxr004_data.py`, `pxr004_train.py`, `pxr004_timing.py`, the frozen split assignments, the v1 artifact (6 files), the monitor artifact (3 files) and the weights directory (3 files). Per the user's pasted shell prompt, it is extracted on CoreHPC at `/scratch/user/mhuang8/tripwire_staging/pxr004_stage_20261119`. The archive hash comparison and `sha256sum -c` results were not pasted, so those checks are unreported; the wrapper itself performs a per-file hash check at run time (below).

**Interactive attempt failed.** The user pasted the output of `srun --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:nvidia_l40s:1 --cpus-per-task=8 --mem=96G --time=1:00:00 --pty bash` from the CoreHPC login host (the line was derived by the assistant from the PXR-002 wrapper resource values; the interactive form is not recorded in the repository). It queued jobs 3132497 and 3132498 and printed the socket errors "Insane message length" and "Srun communication socket apparently being written to by something other than Slurm"; no interactive shell was obtained. The same error was recorded in PXR-002 on 2026-11-17 (`notes/Tripwire_PXR_002.md` section 5a), and the user states it occurs whenever a gpu-partition interactive allocation is requested. The two stray jobs are to be cancelled by the user; their current state is UNKNOWN until reported.

**Route decision (D020, Timing route section).** The feasibility timing is run as a batch job on CoreHPC for now, a deviation from the interactive-benchmark practice, for the same reason and in the same manner as PXR-002 section 5a. Scope is timing only: no learning evidence, no hyperparameter choice, no training run, and no other submission.

**Wrapper (written, syntax-checked with `bash -n` only; not run).** `pxr/scripts/pxr004_timing.slurm`, sha256 `0a255a12e5054561b767beaca2b6117fe0820f3e26b5808d4378e7e6c92a1cf6`. Resources: 1 h, 1 L40S, 8 CPUs, 96G, partition gpu, no requeue. It verifies every staged file against the hash list before running; requires `STAGE_ROOT`, `PXR_ENV`, `HASH_LIST` and `OUT_JSON`; refuses an existing `OUT_JSON`; accepts optional `REQUESTED_WALL_HOURS`; and is to be submitted from the staging root. It is not in the 16-file package; it is transferred separately and checked against the sha256 above (that check is not yet reported).

**Timing job submitted (2026-11-19, reported by the user).** The timing wrapper `pxr/scripts/pxr004_timing.slurm` (sha256 `0a255a12e5054561b767beaca2b6117fe0820f3e26b5808d4378e7e6c92a1cf6`, the version supplied to the user) was reported submitted on CoreHPC as Slurm batch job 3132558. The submission output, the wrapper hash check, the cancellation of stray jobs 3132497 and 3132498, and the job state were not pasted and are unreported. No result yet; timing numbers are pending. This is the D020 feasibility timing, not a training run.

## 8. Stopping point and next step

Tests pass in the correct interpreter (counts not supplied), against the unchanged hashed files (section 6). The monitor artifact is built and hashed locally; the CheMeleon weights are downloaded and hashed locally; the timing script and its tests, and the Slurm wrapper, are written and syntax-checked locally. The package is staged on CoreHPC (integrity checks unreported). The timing route is batch, decided in D020; the timing was reported submitted as job 3132558 and no result is yet available (section 7). Next: the user reports the job state and output of 3132558 and the status of stray jobs 3132497 and 3132498. Training and any other submission remain unauthorized.
