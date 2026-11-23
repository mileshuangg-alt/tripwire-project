# Tripwire-CYP-008 → Tripwire-CYP-009 Handoff

Written 2026-11-21 from the Tripwire-CYP session 008 chat. Next session: **Tripwire-CYP-009**, which follows implementation of the CheMeleon encoder comparison.

This is a continuity reference, not decision authority. It does not authorize implementation, runs, submissions, branches or edits. The receiving session confirms the position and next action briefly, then waits for direction.

Source note: everything below is from the 008 chat record and files that chat wrote or read. Items not established there are marked UNKNOWN. Nothing was re-verified for this document (no commands run, no files read back).

---

## 1. Objective (as established in 008)

Session 008 implemented and ran **D021**: updated T121 plus 196 fixed RDKit descriptors (arm D, Design A: descriptors in both the screening and pIC50 stages) versus updated T121 (arm R), over 3 screening seeds per arm, using the O150 rule (item 6 of D021).

Decision record: `notes/Decisions.md`, D021 ("Updated T121 plus Molecule-Level Descriptors versus Updated T121", from about line 5837). Session log: `notes/Tripwire_CYP_008.md`; supporting record `notes/Tripwire_CYP_008_supporting.md`. The 008 session began from `notes/Handoffs/Tripwire_CYP_007_Handoff_20261119.md`.

**Session 009 scope (stated at hand-off time):** it follows implementation of the CheMeleon encoder comparison. That implementation is **not done in 008**; nothing has been audited, redesigned, implemented or launched for it. The design was agreed in chat after the first version of this handoff and is stated in the next subsection. 008's log still only says CheMeleon was "queued from 007 §7" (`notes/Tripwire_CYP_007.md`, §7). Related but different thread: a PXR-004 CheMeleon two-arm pilot exists (commit 671856d, D020); whether it bears on the CYP comparison is UNKNOWN and is not assumed here.

### 1a. Agreed CheMeleon design (stated in chat; supersedes the earlier undecided-design wording)

Source: the user's chat statement after the first handoff. Recorded as **D022** in `notes/Decisions.md` (see §3).

- **Encoder:** the pretrained CheMeleon encoder replaces the current encoder and is **fully fine-tuned** through CYP screening and then potency (pIC50).
- **Recipe carried over:** updated T121's remaining recipe, **not frozen T121's** (the instruction writes "updated T121*"; the asterisk's meaning is not defined in this chat): head width, depth and dropout; supervision; weighting; batch size; stage-transfer behavior. The head input is adapted to CheMeleon's 2048-dimensional output.
- **Excluded:** no appended descriptors (D021 arm D features are not part of this) and no frozen-encoder comparison.
- **Learning rates, both stages:**
  - Head: updated T121's exact Noam schedule: init 1.4283739994469765e-5, peak 8.245400010426866e-4, final 2.0141638331344468e-4, warmup 10 epochs.
  - Encoder: exactly 0.1 times the head rate throughout that schedule, including init, peak and final, with the same schedule shape and warmup.
- **Stage rules unchanged:** screening 50 epochs, no early stopping, existing validation checkpoint-selection rule; potency maximum 50 epochs, patience 20, existing checkpoint selection. **Optimizer and scheduler are reset at the screening-to-potency transition.**
- **Evaluation unchanged:** existing evaluation, comparator artifacts and O150 rules.
- **Claim boundary:** the 0.1 ratio has experimental precedent, not evidence of superiority; using it across this two-stage schedule is the chosen design. A result tests the complete CheMeleon-based setup, not pretraining alone.

## 2. Approvals versus proposals

**Explicitly approved (with scope):**
- D021 itself was already approved before this chat (it is in `Decisions.md`); 008 did not re-derive or reopen it.
- Launch preparation under D021 using the recommended layout: preserve the existing reference package, build a separate hash-bound descriptor package. Keep the existing qualification standard; no custom evidence rule, no bypass of verification.
- A **one-off batch-Slurm timing deviation, for the D021 descriptor cell timing only** (interactive `srun` failed with "Insane message length"). Not a standing permission. Recorded in the 008 log.
- Building the O150/Δ̄ aggregation script: it was proposed with its design in chat; the cell-output format and R replicate 1 path were then supplied on request, and the script was built and run. A separate explicit "approved" statement for it was not given in words; the go-ahead is inferred from those supplied inputs.
- Session identification: this work is **Tripwire-CYP session 008** (stated in chat), and the 008 log files were created on that basis.
- Writing style for records: shared point of view, no personal name and no "the user"; a pending decision is written as "an explicit decision entry is required" (saved to memory as `records-shared-point-of-view`).
- Committing the working tree (see §4); the push was not completed.
- The CheMeleon design in §1a, stated as agreed. The instruction was to update this handoff and then wait; no audit, redesign, implementation or launch was approved with it.

**Not approved / proposals and parked items:**
- Per-step run approvals (screening runs, the five cell arrays) are **not itemized** anywhere. D021's status line says it authorizes no training, screening, cell run, GPU job or Slurm submission without separate approval; those runs were submitted by the user. The 008 log says per-step approvals are not itemized. No approval statement was supplied to record.
- The "Governing principle" sentence in D021: awaiting a decision, not settled.
- No next step after the D021 outcome is decided. D021 states no follow-up for a "not favoured" result.
- Descriptor-aware inference and diagnostics scripts: not built.
- Optional read-only `ls` of `t121_screen_complete_20261117` to recover the three older evidence scripts: offered, not requested.

## 3. Records lag

- **The CheMeleon design in §1a is recorded as D022** (`notes/Decisions.md`, appended after D021 from about line 5878; "Approved as a design only"; it authorizes no audit, redesign, implementation, training, screening, GPU job or Slurm submission; it supersedes nothing). The 008 log carries it in §8a and §7. Source: the bookkeeper's report after this handoff was first updated; the entries were not read back by the session that wrote this handoff.
- Two lines in `notes/Tripwire_CYP_008.md` are stale and were left unchanged: the §1 Purpose line still says "CheMeleon remains queued after the descriptor comparison", and the top Status line does not mention D022. Correcting them is a routine report update not yet requested.
- D021 **execution and result** are recorded in `notes/Tripwire_CYP_008.md` and `_supporting.md` (created by the bookkeeper in 008). Whether `notes/Decisions.md` D021 carries any result or status update is UNKNOWN; 008 did not edit Decisions.md.
- `notes/Tripwire_CYP_007.md` has no entries for D021 execution (they belong to 008). The bookkeeper judged no change to 007 necessary.
- In the supporting record, full hashes were **not supplied** to the bookkeeper for the pinned reference tarball (`9ecb6ccf…`), the five screening record SHAs (abbreviated below) and the smoke `check.py` hash (`09abee9d…`); they are kept abbreviated and marked. The per-run mapping of screening jobs 3161795–3161799 to runs is not in the record. The CoreHPC report JSON's own hash was never computed.
- The three older evidence scripts (screen_complete and related) are not in the repository; the descriptor smoke covers equivalent ground under the same evidence standard. Gap not closed.
- The log2fc assay definition (what is compared to what) was not found or confirmed in the repository; `cyp/data/openadmet_unblinded_release_2026-11-06/PROVENANCE.txt` was not read.

## 4. Work and artifacts

**Recorded outputs (pasted from CoreHPC in chat):**

| Item | Evidence |
|---|---|
| Descriptor smoke, job 3155222 | COMPLETED 0:0, PASS, 32 s, descriptor_dim 196, predictor widths [[2000, 796]]; result.json SHA-256 `dad22845c87c8486df2dd5c0c59252d842cfbb8fab29dbf4d5cb7925391df1b2` |
| Timing, job 3158201 (L40S, batch) | screening 4.90 s/epoch (about 251 s per 50 epochs); cell 0 147.6 s (about 22,137 s serial per 150 cells). Job 3158166 failed first (SCREEN_EPOCHS left at 3), fixed. The cell score 0.7015 came from a 3-epoch checkpoint and is not a result. |
| 5 screening runs, jobs 3161795–3161799 | all COMPLETED 0:0, 50 epochs, no full-screen refit. Seeds/epochs: 20261102 epoch 31 (R), 20261103 epoch 29 (R), 20261101 epoch 29 (D), 20261102 epoch 46 (D), 20261103 epoch 28 (D). Record SHA prefixes: `0bab1c8a`, `c7c71844`, `a03d270c`, `ce1666a8`, `9a415bd1`. R replicate 1 (existing) selected epoch 16. Job-to-run mapping not recorded. |
| 5 cell arrays | r_s2 3161888, r_s3 3161889, d_s1 3161890, d_s2 3161892, d_s3 3161893; each 150 `COMPLETED.json`, 0 `INCOMPLETE.json`, all tasks COMPLETED 0:0 (last reported 2026-11-21). Layout: 5 repetitions × 10 folds × 3 cell seeds. |
| O150 aggregation | stdout pasted; report `/scratch/user/mhuang8/tripwire_staging/t121_d021_o150_report_20261121.json` (CoreHPC; hash not computed) |

**D021 result (item 6; lower ST-RAE is better; Δ_s = D − R):**
- O150 R: r1 0.676954037 (reproduces the recorded 0.6769540372619428 within 1e-12), r2 0.668499342, r3 0.669626167. O150 D: d1 0.682479984, d2 0.680816837, d3 0.686278215.
- Δ_s: +0.005525946, +0.012317495, +0.016652049. **Δ̄ = +0.011498497; D is not favoured** under the strict Δ̄ < 0 rule. Ō(R) 0.671693182, Ō(D) 0.683191679.
- Per-isoform Δ̄: CYP1A2 +0.012091654, CYP2C9 +0.003409566, CYP2D6 +0.009012727, CYP3A4 +0.021480040.
- No significance claim: replicates share partitions and are not independent (D021).

**Code and tests (local, committed as `69dc80b` on `main`):** `cyp/scripts/t121_d021_layout.py`, `t121_d021_build_package.py`, `t121_d021_aggregate.py` (SHA-256 `55ade624c4cd67a9c4fbc03430e29274e179f20f44517851457a9a2037d42c39`), `t121_descriptor_cache.py`, `t121_screen_chain_run.py`, `t121_production_cell.py` (descriptors attach after fold routing; a dry run caught the pre-routing bug), `t121_cells.slurm`; tests `cyp/tests/test_t121_d021_layout.py`, `test_t121_d021_aggregate.py` (11 passed in `tripwire_cyp002`), others. Earlier local runs: 61 passed, 26 passed inside the built package. The full suite was not re-run before the commit.

**Package and evidence:** `t121_d021_package_20261119.tar.gz` (repo root, untracked), SHA-256 `bd73e8c3a6f1257e24587eb78a8102a03e3eb1b534ebe68b5b6e2005601a0978`; CoreHPC deployment `/scratch/user/mhuang8/tripwire_staging/t121_d021_package_20261119`; pins `d021_launch/t121_d021_package_20261119/d021_pins.json`; manifests `d021_launch/manifests/`; qualification `qualification/t121/d021_descriptor_smoke_20261119/`, `qualification/t121/d021_timing_20261119/`; frozen descriptor cache `cyp/artifacts/d021_descriptors/` (descriptor manifest `870188de…`, descriptors.npz `03734bbe…`, abbreviated).

**Git:** commit `69dc80b` ("Add D021 descriptor comparator: code, package layout, evidence, O150 aggregation and session 008 log"), 43 files. **Push failed** (`ssh: connect to host git.ucsf.edu port 22: Operation timed out`); the commit exists locally only. Cause not confirmed (likely off the UCSF network). 51 entries remain untracked: transfer tarballs, `.bak` files, hash lists, `t121_valinf_*` output directories, `cyp_adaptation_predictions.json`, `pxr/artifacts/pxr004_weights/chemeleon_mp.pt`, the root-level CYP-007 handoff copy, and the PXR-002/PXR-004 handoffs. Which of these should be committed or ignored is undecided.

## 5. Current position

- D021 is executed and its result computed and recorded in the 008 log. Nothing is running or queued on CoreHPC as far as this chat knows (last reported status above; live status UNKNOWN).
- Awaiting an explicit decision entry for: the outcome's consequence (none stated), the "Governing principle" sentence, and handling of the leftover untracked files.
- Awaiting network access to complete `git push origin`.
- The CheMeleon encoder comparison implementation, which 009 follows, has not started in 008. Its design is agreed (§1a); nothing is audited, implemented or launched.

## 6. Next action

Established: the design in §1a, and that the user wants the handoff updated and then to give the next instruction. **The next action after that is UNKNOWN**; no audit, implementation, run or submission is authorized by this handoff. Existing evaluation and O150 rules are stated to remain unchanged; the exact acceptance and interpretation criteria for the CheMeleon comparison (for example how its O150 is to be read against which comparator) are not restated in the instruction and are UNKNOWN here. D021's item 6 rule (strict Δ̄ < 0, no effect floor, no bootstrap, no significance claim) is not assumed to carry over unless the Decisions record says so.

## 7. Unknowns

- The comparison's acceptance and interpretation criteria (§6). The CheMeleon checkpoint's location and how it is loaded into this repo's code are not established in this chat.
- Whether `Decisions.md` D021 has a result or status update; live CoreHPC job state.
- Full hashes listed as abbreviated or "not supplied" (§3, §4); report JSON hash.
- Why the push timed out; whether the three older evidence scripts exist on CoreHPC.
- The log2fc assay definition (§3).
- Whether the PXR-004 CheMeleon pilot or the untracked `chemeleon_mp.pt` is relevant to the CYP comparison (belongs, if at all, to a different thread: uncertain).

## 8. Access the receiving session needs

- **Repository checkout** at `/Users/mileshuang/Desktop/tripwire-project` (local Mac); the assistant has no direct CoreHPC access.
- **CoreHPC**: login `chpc-ucsf-login-vm1.corehpc.ucsf.edu` via bastion `chpc-ucsf-bastion-vm1.corehpc.ucsf.edu`, user `mhuang8`; staging `/scratch/user/mhuang8/tripwire_staging/`; T121 environment `/scratch/user/mhuang8/envs/tripwire_cyp002`. Commands, transfers and submissions are run by the user; interactive GPU `srun` fails with "Insane message length", so any batch use needs its own recorded approval.
- Local test environment: `tripwire_cyp002` (conda, miniforge).
- Subagents: `bookkeeper` (records, per the project CLAUDE.md) and `lookup`. Briefs and records use the shared point of view described in §2.
