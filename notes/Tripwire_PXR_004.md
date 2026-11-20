# Tripwire-PXR-004 Session Log

**Date:** 2026-11-19
**Status:** CLOSED on declaration, 2026-11-19.

**Summary (final).**
- **Objective.** Sequential auxiliary pretraining on PXR: two arms (direct versus sequential) from the same released CheMeleon encoder, matched stage 2, descriptor-free, full 2,897-molecule primary-train cohort (D020).
- **Work.** Implementation, stage-1 monitor artifact, feasibility timing (job 3132558; extrapolated worst case about 0.47 h per seed), and a pilot of three paired seeds (20261125-27) as three Slurm jobs, one per seed, run 2026-11-19. Reporting script `pxr/scripts/pxr004_report.py`; its integrity checks against the predictions passed on the real outputs.
- **Result (validation only, descriptive; no threshold or test per D020).** Direct versus sequential validation MAE means 0.5258 vs 0.5280 (paired sequential-minus-direct +0.0022; per seed +0.0000, +0.0001, +0.0063); RMSE means 0.7269 vs 0.7226; Spearman means 0.7453 vs 0.7369 (sequential lower on all three seeds). Differences are small relative to the seed-to-seed spread (SD about 0.005 MAE). No conclusion beyond the descriptive numbers (section 10).
- **Open or carried forward (none decided or authorized).** The CheMeleon Supporting Information check remains unverified (not located; public repositories do not state downstream settings; section 3). Molecule-level descriptors may be tested separately later (not scheduled, not decided). No further experiment is decided.
- **Evaluation boundaries.** No holdout or official-test use (holdout consumed and closed; official test unused).
- **Commits.** 406259d (wrapper, report script, D020 packaging and seeds, results) and 52ee063 (CLAUDE.md and bookkeeper rule); the later section 3 external-source edits are uncommitted.
- **Artifacts.** Full-precision values in `notes/Tripwire_PXR_004_supporting.md`; decision in `notes/Decisions.md` (D020).

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
- **External sources.** CheMeleon paper main text (arXiv 2506.15792 v2) gives full fine-tuning with a fresh head and no hyperparameter optimization, but not the downstream epoch, LR, batch or FFN settings; its Supporting Information was not read. The user reports that the authors' paper-era fine-tuning code uses 50 epochs, patience 5 on validation loss and a 256-unit head; the pilot deliberately differs. Read-only check of the external sources on 2026-11-19 (web-fetch subagent output, not independently re-verified): (a) arXiv 2506.15792 v2 (revised 2026-02-09; v1 not compared), HTML read in full, confirms end-to-end fine-tuning with a randomly initialized task FFN, no hyperparameter optimization, 5 repetitions varying a seed that sets the early-stopping validation split, and defers full hyperparameters to the Supporting Information; max epochs, patience, monitored metric, LR/schedule, batch size and FFN size/layers/dropout are not in the main text; the Supporting Information could not be located or read (PDF text not extractable, no SI link found). (b) Authors' repository github.com/JacksonBurns/chemeleon (main): no dedicated CheMeleon benchmark fine-tuning script or config found. CheMeleon appears only in `finetuning_demo.ipynb` (Chemprop lipophilicity demo; encoder trainable; max_epochs 20; ModelCheckpoint on val_loss, no EarlyStopping; default head 2048->300, one hidden layer, dropout 0.0, not set explicitly; batch size, LR, schedule not set). `models/chemprop_mpnn/evaluate.py` (Chemprop baseline script, can start from a pretrained checkpoint; does not load CheMeleon in the opened files) uses max_epochs 50, EarlyStopping on val_loss patience 5 and pretrained-arm hidden_dim 256, which matches the reported settings but is not shown to be the CheMeleon benchmark configuration. (c) Public repository github.com/JacksonBurns/how-to-train-your-chemeleon (README, `pretraining/` listing and `pretraining/config.py` read at Miles's request by a web-fetch subagent; subagent report, not independently re-verified): tied to the paper by its README but described there as a re-written, not faithful, reproduction of the pre-training script; no downstream benchmark fine-tuning code or config and no Supporting Information file; the README states that fine-tuning needs only `chemprop train` with `--from-foundation CheMeleon`. Its `config.py` holds pre-training values only (100 epochs, patience 5, LR 1e-4 to 1e-3 to 1e-4 with 5 warmup epochs, head 2048 hidden / 1 layer / dropout 0.30, message passing 2048 hidden / depth 6), which must not be read as downstream settings. Every downstream item remains not found in the public code. Not checked: the paper's Supporting Information (location unknown) and the `analysis/` directory of the chemeleon repository. Status: public repositories do not settle the downstream settings; the reported code-precedent statement remains unverified and the D020 open item (Supporting Information / repository configuration check) stays unverified, with the Supporting Information the only remaining source. No effect on the pilot, whose settings are a disclosed convention not derived from these sources.
- **Reporting.** Validation MAE, Spearman and RMSE per seed and mean, with paired per-seed differences, descriptive only (no threshold, no test). RMSE needs primary-head validation predictions saved at the best epoch (new relative to PXR-002). Context points, not matched comparators: constant mean 0.9218; ECFP4 + LightGBM 0.6049; PXR-002 single-model mean about 0.546 (descriptor-based, 2,896 cohort).
- **Timing.** Feasibility only, never evidence of learning, and nothing is chosen from it; route not chosen. Standing pre-Slurm rules apply.

Resolved by this design: encoder frozen versus fine-tuned, stage-1 head/loss/budget, comparison criterion, RMSE as a reported metric.

Still open and unapproved (the design-time statement; implementation and the weights download were subsequently approved, see section 4): creation of the hashed monitor artifact; timing route; any production training or submission; check of the CheMeleon Supporting Information or repository fine-tuning configuration; any numeric threshold or statistical test.

## 4. Implementation and CheMeleon weights (2026-11-19)

**Approval and scope.** Implementation of the pilot code and tests per D020, and the download of the CheMeleon weights, were approved. Not approved: running tests, training, timing or Slurm; the user runs all execution commands.

**Weights.** Downloaded with `curl --fail` from `https://zenodo.org/records/15460715/files/chemeleon_mp.pt` (the URL Chemprop 2.3.1 uses for `--from-foundation CHEMELEON`) to `pxr/artifacts/pxr004_weights/chemeleon_mp.pt`: HTTP 200, 34,859,448 bytes, SHA-256 `c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651`. `SHA256SUMS` (sha256 `7e0f7dc0ffa49433ed9399e82c68716f8bddc954bacfd6dea43917ab9df8f764`) and `PROVENANCE.txt` sit beside it. The file was inspected only with `torch.load(weights_only=True)`: keys `hyper_parameters` and `state_dict`; four tensors (W_i.weight 2048x86, W_h.weight 2048x2048, W_o.weight 2048x2120, W_o.bias 2048); 8,714,240 parameters; d_v 72, d_e 14, d_h 2048, depth 6, dropout 0.0, bias False, activation relu, undirected False. This matches the paper's 2048 width, depth 6 and 8.7M-parameter encoder. (Status at this entry; the weights were later staged on CoreHPC, section 7.)

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

2026-11-19: all tests were reported to have passed locally. The report did not state which test files were run, the pytest counts or the interpreter; that it refers to `pxr/tests/test_pxr004_timing.py` (sha256 `e2087c1a...d036`) is an inference and is unconfirmed. Status at this entry (superseded): the timing run had not been done and the route was not chosen; the route was later decided and the run completed (sections 7 and 8).

## 7. Staging, interactive failure and timing route (2026-11-19, later entry)

**Staging.** A 16-file package was built by the assistant: `pxr004_stage_20261119.tar.gz` (hash list `pxr004_stage_hashes_20261119.txt`, which has its own sha256 `7efe0c1f98aca2c3e8e08f1975db5ca9823f3679ee37d5e28c11d670da6ec17a`; file list `pxr004_stage_files_20261119.txt`). Contents: `pxr004_data.py`, `pxr004_train.py`, `pxr004_timing.py`, the frozen split assignments, the v1 artifact (6 files), the monitor artifact (3 files) and the weights directory (3 files). Per the user's pasted shell prompt, it is extracted on CoreHPC at `/scratch/user/mhuang8/tripwire_staging/pxr004_stage_20261119`. The archive hash comparison and `sha256sum -c` results were not pasted, so those checks are unreported; the wrapper itself performs a per-file hash check at run time (below).

**Interactive attempt failed.** The user pasted the output of `srun --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:nvidia_l40s:1 --cpus-per-task=8 --mem=96G --time=1:00:00 --pty bash` from the CoreHPC login host (the line was derived by the assistant from the PXR-002 wrapper resource values; the interactive form is not recorded in the repository). It queued jobs 3132497 and 3132498 and printed the socket errors "Insane message length" and "Srun communication socket apparently being written to by something other than Slurm"; no interactive shell was obtained. The same error was recorded in PXR-002 on 2026-11-17 (`notes/Tripwire_PXR_002.md` section 5a), and the user states it occurs whenever a gpu-partition interactive allocation is requested.

**Route decision (D020, Timing route section).** The feasibility timing is run as a batch job on CoreHPC for now, a deviation from the interactive-benchmark practice, for the same reason and in the same manner as PXR-002 section 5a. Scope is timing only: no learning evidence, no hyperparameter choice, no training run, and no other submission.

**Wrapper (written, syntax-checked with `bash -n` only; not run).** `pxr/scripts/pxr004_timing.slurm`, sha256 `0a255a12e5054561b767beaca2b6117fe0820f3e26b5808d4378e7e6c92a1cf6`. Resources: 1 h, 1 L40S, 8 CPUs, 96G, partition gpu, no requeue. It verifies every staged file against the hash list before running; requires `STAGE_ROOT`, `PXR_ENV`, `HASH_LIST` and `OUT_JSON`; refuses an existing `OUT_JSON`; accepts optional `REQUESTED_WALL_HOURS`; and is to be submitted from the staging root. It is not in the 16-file package; it is transferred separately and checked against the sha256 above (that check is not yet reported).

**Timing job submitted (2026-11-19, reported by the user).** The timing wrapper `pxr/scripts/pxr004_timing.slurm` (sha256 `0a255a12e5054561b767beaca2b6117fe0820f3e26b5808d4378e7e6c92a1cf6`, the version supplied to the user) was submitted on CoreHPC as Slurm batch job 3132558; its result is in section 8. This is the D020 feasibility timing, not a training run.

## 8. Timing result, job 3132558 (2026-11-19)

**Source.** The user pasted the stdout of `/scratch/user/mhuang8/tripwire_staging/pxr004_timing_3132558.out` from the CoreHPC login host. The evidence lives on CoreHPC and in that paste; the JSON `/scratch/user/mhuang8/pxr004_timing_20261119/timing_gpu.json` has not been downloaded or reviewed locally.

**Run.** Job 3132558 ran on node ggpu1-07 (NVIDIA L40S, 46,068 MiB, driver 595.91.07), 2026-11-19 14:38:01 to 14:38:21 (-08:00), about 20 s, exit 0. The wrapper's per-file hash check printed OK for all 16 staged files (three `pxr004_*.py` scripts, split assignments, `pxr002_aux` x6, `pxr004_monitor` x3, `pxr004_weights` x3). CoreHPC environment: chemprop 2.3.1, torch 2.14.1+cu126, lightning 2.6.6, pandas 3.0.6, scipy 1.18.1, CUDA available. This resolves the earlier unknown on CoreHPC-side versions; the script's GPU path ran without error. No requested-wall-time comparison line appeared in the output, so `REQUESTED_WALL_HOURS` appears not to have been set (inference from the absent line).

**Measurements (throwaway model, GPU).**

| Stage | Train batches/epoch | Step median (s) | Fetch median (s) | Validation batches | Worst-case run, median / p90 (h) |
|---|---|---|---|---|---|
| 1 | 139 | 0.045 | 0.020 | 16 | 0.27 / 0.28 |
| 2 | 46 | 0.044 | 0.019 | 10 | 0.09 / 0.10 |

Extrapolated worst case (every run to 100 epochs, validation and checkpoint every epoch; 3 stage-1 and 6 stage-2 runs):

| Quantity | Median (h) | p90 (h) |
|---|---|---|
| Direct arm, one seed | 0.0939 | 0.0960 |
| Sequential arm, one seed | 0.3620 | 0.3747 |
| Serial, all runs | 1.3678 | 1.4123 |

**Status.** Feasibility only, per D020: no cap, patience or hyperparameter was chosen from it, and it is not evidence of learning. The figures are an extrapolated worst case from a throwaway model, not a measured production run. Packaging, wall-time request and seeds were subsequently decided (section 9); no production run or submission is approved.

## 9. Production packaging and seeds (D020, 2026-11-19)

**Decision (recorded in D020, Production packaging and seeds).** Three Slurm jobs, one per seed, each running the direct arm then the sequential arm for its seed on the same node, with a 1:00:00 request. Basis: worst-case p90 from job 3132558 of 0.096 h (direct) plus 0.375 h (sequential) is 0.47 h per seed; the request is about 2.1 times that. The timing is feasibility only; no cap or patience derives from it. Seeds 20261125, 20261126 and 20261127, shared by both arms for pairing, fixed before any result and not searched; distinct from the PXR-002 seeds (20261117-19, 20261120-24) and the monitor draw seed 20261119. The approval covers packaging, seeds, preparation of the production wrapper and hash list, and records. It does not authorize training or Slurm submission; a separate launch approval is required.

**Wrapper (written locally, syntax-checked with `bash -n` only; not staged, not submitted; uncommitted).** `pxr/scripts/pxr004_pilot.slurm`, sha256 `86d40878bfd324e0407b55451769a55c84e7c69e8371720df9868b9f87355a52`. It requires `STAGE_ROOT`, `PXR_ENV`, `HASH_LIST`, `SEED` (restricted to the three approved seeds), a new `OUT_ROOT` per seed and `AUTHORIZE_PXR004_PILOT=yes`; hash-checks the staged files; loops direct then sequential through `pxr004_train.py`; no requeue; L40S, 8 CPUs, 96G. It had not been run when written (see Launch below).

**Launch (2026-11-19).** The three pilot jobs were submitted on CoreHPC from the staging root `/scratch/user/mhuang8/tripwire_staging` with the wrapper above (sha256 as listed), `AUTHORIZE_PXR004_PILOT=yes` and `OUT_ROOT=/scratch/user/mhuang8/pxr004_pilot_20261119/seed<SEED>`, one job per seed (20261125, 20261126, 20261127; direct then sequential arm; 1:00:00). The submission itself is the launch approval, within the packaging and seeds decided above. Source: pasted CoreHPC submission shell transcript (2026-11-19). Job-to-seed mapping: seed 20261125 is job 3132953 (OUT_ROOT `/scratch/user/mhuang8/pxr004_pilot_20261119/seed20261125`); seed 20261126 is job 3132954 (`.../seed20261126`); seed 20261127 is job 3132955 (`.../seed20261127`). Each was submitted from `/scratch/user/mhuang8/tripwire_staging` via `pxr004_pilot.slurm` with `STAGE_ROOT` `.../pxr004_stage_20261119`, `PXR_ENV` `/scratch/user/mhuang8/envs/tripwire_cyp002`, `HASH_LIST` `.../pxr004_stage_hashes_20261119.txt` and `AUTHORIZE_PXR004_PILOT=yes`. The first `squeue` showed only 3132953 pending (state PD, reason Priority); an earlier pasted `squeue` had shown all three (`pxr004-p...`) pending on the gpu partition, state PD, reason Priority. UNKNOWN: whether the wrapper-hash and staged-file checks printed OK before submission (output not pasted). No job has started; no output or result exists. No holdout/test use and no reruns are authorized.

**Checks (local).** The 16 files in `pxr004_stage_hashes_20261119.txt` still match their hashes (`shasum -c`, no mismatches); the three `pxr004_*.py` scripts are unchanged since the timing job. From `pxr004_train.py`, `pl.seed_everything(seed)` is called in both stage 1 and stage 2, so the seed governs head initialization, shuffling and dropout streams, and both arms seed stage 2 identically.

## 10. Pilot results, jobs 3132953-3132955 (2026-11-19)

**Source.** The user pasted CoreHPC login-host output reading `stage2/metrics.json` (both arms) and the sequential arm's `stage1/metrics.json` under `/scratch/user/mhuang8/pxr004_pilot_20261119/seed<SEED>/` (seed 20261125 is job 3132953, 20261126 is 3132954, 20261127 is 3132955). The evidence lives on CoreHPC and in that paste; nothing has been downloaded or reviewed locally. All values are on the 621-compound D014 validation set at the best epoch chosen by primary-head validation MAE. Full-precision values are in `notes/Tripwire_PXR_004_supporting.md`. Reporting is descriptive only per D020: no threshold, no statistical test.

**Stage 2, validation (lower MAE/RMSE and higher Spearman are better).**

| Seed | Arm | MAE | RMSE | Spearman | Best epoch | Epochs run |
|---|---|---|---|---|---|---|
| 20261125 | direct | 0.52129 | 0.71946 | 0.74402 | 23 | 44 |
| 20261125 | sequential | 0.52134 | 0.71792 | 0.74290 | 15 | 36 |
| 20261126 | direct | 0.53029 | 0.72749 | 0.73712 | 28 | 49 |
| 20261126 | sequential | 0.53040 | 0.72784 | 0.73416 | 34 | 55 |
| 20261127 | direct | 0.52583 | 0.73371 | 0.75472 | 2 | 23 |
| 20261127 | sequential | 0.53213 | 0.72215 | 0.73371 | 2 | 23 |

Paired differences (sequential minus direct; arithmetic by the assistant, descriptive): MAE +0.00005, +0.00011, +0.00630 (arm means: direct 0.52580, sequential 0.52796, difference +0.00215); RMSE -0.00154, +0.00035, -0.01155 (means 0.72689 and 0.72264, difference -0.00425); Spearman -0.00112, -0.00297, -0.02101 (means 0.74529 and 0.73692, difference -0.00837). The per-seed MAE differences are small relative to the seed-to-seed spread in MAE of about 0.009. In seed 20261127 both arms' stage-2 best epoch was 2.

**Stage 1 (sequential arm), monitor set.**

| Seed | Best pooled masked standardized MSE | Best epoch | Epochs run | MAE 8.25 / 33 / 99 uM | Spearman 8.25 / 33 / 99 uM |
|---|---|---|---|---|---|
| 20261125 | 0.5607 | 2 | 23 | 0.1988 / 0.2997 / 0.3792 | 0.571 / 0.631 / 0.579 |
| 20261126 | 0.5589 | 4 | 25 | 0.1945 / 0.3035 / 0.3897 | 0.570 / 0.640 / 0.574 |
| 20261127 | 0.5815 | 20 | 41 | 0.2070 / 0.3103 / 0.3955 | 0.544 / 0.629 / 0.549 |

Monitor labels per head are 974 / 848 / 67. Stage-1 best epochs were early for two seeds (2 and 4).

**Context (not matched comparators; earlier records).** Constant training mean 0.9218 validation MAE; ECFP4 + LightGBM 0.6049; PXR-002 single model about 0.546 (descriptor-based, 2,896 cohort).

**Reporting script.** `pxr/scripts/pxr004_report.py` was written after these results; its integrity checks against the predictions passed on the real outputs. Commits: 406259d (see the header summary).

**Not done.** No holdout or official-test use. This entry records no conclusion about whether auxiliary pretraining helps.

## 11. Stopping point and next step

Tests pass in the correct interpreter (counts not supplied), against the unchanged hashed files (section 6). The monitor artifact and CheMeleon weights are built and hashed locally and were staged on CoreHPC, where the wrapper verified all 16 staged files. Timing job 3132558 completed (exit 0, about 20 s; section 8). Packaging and seeds are decided and the production wrapper written (section 9). The three pilot jobs were run and their validation results recorded (section 10). The reporting script `pxr/scripts/pxr004_report.py` was run and its integrity checks passed. The session was declared closed on 2026-11-19. No next step is authorized or started; no further experiment is decided. Carried forward, none decided: the CheMeleon Supporting Information check (unverified, source not located) and a possible separate test of molecule-level descriptors (not scheduled). No holdout or official-test use, no reruns.
