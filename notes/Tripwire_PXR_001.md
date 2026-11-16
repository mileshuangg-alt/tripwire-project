# Tripwire-PXR-001 — Session Log

**Session:** Tripwire-PXR-001 · **Status:** CLOSED — session close declared by Miles 2026-11-15 · **Date:** 2026-11-15
**Workstream:** Tripwire — PXR induction arm; first PXR working session
**Governing document:** `Tripwire_Project_Proposal_Rev3.md` ("Revision 3, September 30, 2026"), supplied by Miles on 2026-11-15 as a file in his Downloads folder. It is **not** in this repository and not in its git history; line numbers cited here are those of the file as he provided it.
**Execution target:** local checkout only. No remote host, no Slurm, no GPU was used in this session.
**Origin:** mid-session migration handoff from a prior chat (Miles supplied `Tripwire_PXR_001_mid_session_handoff.md`).

---

## Current state

*Mutable summary, updated in place. Everything below the next `---` rule is append-only history. Last updated: 2026-11-15 (after entry 8).*

- **Objective:** start the PXR induction arm — retrieve and pin the official dataset, audit it, declare and instantiate the predeclared internal scaffold holdout (entry 1). No model work was authorized at any point in this session.
- **Last completed step:** Miles declared the session close (entry 8). Before that, the predeclared scaffold holdout was implemented, regenerated under D014 Amendment 4 and frozen, and independently verified: `pxr/artifacts/pxr001_split/pxr001_split_assignments.csv`, SHA-256 `c6028f1b7216ab2ea4bab96616fc02b95ce011be20b1b96f807d1e7d582ece46`; train 2,897 (69.99%) / validation 621 (15.00%) / holdout 621 (15.00%) over 3,699 groups; committed as `fabec38` (entry 7). Before that, D014 and its four amendments plus an artifact record were written to `notes/Decisions.md` (entry 6). Before that, the split-design diagnostics were measured (entry 5), the CYP code-reuse assessment was done (entry 4), the compact data audit was built and committed as `fdd5573` (entry 3), and the official release was retrieved, hash-verified and preserved as `5cca55c` (entry 2).
- **Next step:** the ECFP4 + gradient boosting classical baseline, which Rev 3 line 90 makes mandatory and first ("Always, first"), with RF as an additional classical comparison; scored as MAE of predicted pEC50 with Spearman for ranking quality (Rev 3 line 117) on the frozen holdout (entries 6, 8). Not started.
- **Pending decisions or approvals:**
  - Molecule standardization: undeclared. Rev 3 line 107 names it as a shared component but specifies none; the frozen split used the official SMILES as released, so any standardization decision would change scaffold assignment (entry 8).
  - The numerical improvement bar required by Rev 3 line 129: no value yet (entry 8).
  - Eligibility of crude and semi-pure HT-chem material for a pEC50 target: undecided (entries 3, 8).
  - The semi-pure/blinded structural match `OCNT-2395416` == `OADMET-0006234`: unresolved (entry 3).
  - Whether a further grouping rule is wanted for the 30 ringless molecules, whose 19 train / 3 validation / 8 holdout distribution is not proportional to 70/15/15 (entries 7, 8).
  - The 3 TRAIN molecules without DRC curves, the 348 wide-CI rows, and the absence of inactives in the primary table: observed data properties, no exclusion decided (entries 3, 8).
  - PXR model training, tuning and evaluation: not authorized (D014 and all four amendments).
- **Active constraints:**
  - Governing: `notes/Decisions.md` D014 — Approved; D014 Amendments 1, 2, 3 and 4 — all Approved (entry 6). Amendment 4 is the only one that authorizes an action: regenerating the split (entry 7).
  - The holdout specification is closed: Bemis-Murcko scaffold with chirality excluded; whole groups indivisible; InChIKey connectivity-block groups kept intact; each ringless molecule its own group; stratification on group-median pEC50; three-way 70/15/15; seed 20261115; assignment frozen as a CSV with its own SHA-256 keyed on `Molecule Name` (D014 Amendments 1–4).
  - Rev 3 line 109 forbids the internal holdout being an iterative tuning target, and requires explicit near-duplicate and analog-leakage checking (entry 6).
  - Rev 3 line 53 settles the PXR evidence class as "Internal-retrospective evaluation, not a prospective blind challenge result" (entry 8).
  - PXR identity key is `Molecule Name` (OADMET ids); CYP's `Molecule_Name` (OCNT ids) is a different key, and `OCNT_ID` is a separate PXR column (entry 4).
  - The governing proposal is not version-controlled with the work it governs (entry 8).
- **Artifact references** (all in this checkout; all three commits are local, nothing pushed — entry 8):
  - Preserved release: `pxr/data/openadmet_pxr_release_2026-11-15/` and the firewalled `pxr/data/openadmet_pxr_heldout_labels_2026-11-15/`, each with `PROVENANCE.txt` and `SHA256SUMS`; Hugging Face `openadmet/pxr-challenge-train-test` revision `063e723841063c2dfa86231f58fb59a6a738d113`; TRAIN `pxr-challenge_TRAIN.csv` SHA-256 `fdc2252bb5763ab5742e78cb1ba692bebe131a972cf83aa1bd45e43d8c5bc223`. Commit `5cca55c` (entry 2).
  - Data audit: `pxr/scripts/pxr001_data_audit.py`; `pxr/artifacts/pxr001_data_audit/`, `audit_report.txt` SHA-256 `6bd2eb8aad12c26baee14d3646b11b95b3a1973c5c5546daa90cb95660168bb1`. Commit `fdd5573` (entry 3).
  - Split: `pxr/scripts/pxr001_split.py`, tests `pxr/tests/test_pxr001_split.py` (19 passed); `pxr/artifacts/pxr001_split/pxr001_split_assignments.csv` SHA-256 `c6028f1b7216ab2ea4bab96616fc02b95ce011be20b1b96f807d1e7d582ece46`, with `PROVENANCE.txt` and `SHA256SUMS`. Commit `fabec38` (entry 7).
  - Decision records: `notes/Decisions.md` — D014 at line 6389, Amendments at 6453, 6512, 6550, 6585, artifact record at 6614, commit follow-up at 6657 (entry 6).
  - Split-design diagnostics (entry 5) and the independent split verification (entry 7) were scratchpad scripts, not committed; only their reported outputs are recorded here.
  - `Tripwire_PXR_001_mid_session_handoff.md` and `Tripwire_Project_Proposal_Rev3.md` live in Miles's Downloads folder, not in this repository; no hash recorded, UNKNOWN.

---

## History

### 1. Session origin and starting point (2026-11-15)
- **What:** this session began as a mid-session migration handoff from a prior chat; Miles supplied `Tripwire_PXR_001_mid_session_handoff.md`. It is the first PXR working session. The handoff placed the stopping point before the first official dataset metadata inspection.
- **Verification:** read-only inspection confirmed the starting point — `pxr/` and `shared/` were empty directories with no commit ever touching them, and "pxr" appeared zero times in `notes/Decisions.md` and `notes/project_log.md`.
- **Objective set:** retrieve and pin the official dataset, audit it, declare and instantiate the predeclared scaffold holdout. No model work authorized.
- **Evidence:** the handoff file is in Miles's Downloads folder, not in this repository; its hash is UNKNOWN.

### 2. Official PXR release retrieved, hash-verified and preserved — commit `5cca55c` (2026-11-15)
- **What:** read-only metadata inspection through the Hugging Face API of `openadmet/pxr-challenge-train-test`: HEAD revision `063e723841063c2dfa86231f58fb59a6a738d113`, upstream `lastModified` 2026-09-28, 8 declared configs, 14 root files plus 184 ground-truth PDBs. The dataset card documents "over 11,000 compounds", a challenge period of April 1 – July 1 2026 (closed), CC-BY-4.0, DOI 10.57967/hf/9731. All 10 in-scope files were then fetched one per request from `resolve/<revision>/` and preserved verbatim under two directories, each with `PROVENANCE.txt` and `SHA256SUMS`:
  - `pxr/data/openadmet_pxr_release_2026-11-15/` — development-eligible: primary TRAIN, counter-assay, single-concentration, HT-chem crudes, the 96-compound semi-pure set, the 513 TEST_BLINDED structures, the TRAIN DRC parquet, and `DATASET_CARD.md`;
  - `pxr/data/openadmet_pxr_heldout_labels_2026-11-15/` — firewalled: the Phase 1 and Phase 2 unblinded labels and the TEST DRC parquet, which carries test label information despite its TRAIN-like filename.
- **Why it matters:** TRAIN SHA-256 `fdc2252bb5763ab5742e78cb1ba692bebe131a972cf83aa1bd45e43d8c5bc223` is the input every later PXR artifact is keyed to.
- **Provenance limits:** only the parquet files are LFS-tracked upstream and carry a published SHA-256; both reproduced their upstream oid exactly. The CSVs are plain git blobs with no upstream hash, so their provenance rests on the revision pin plus locally computed hashes.
- **Scope:** the structure sub-challenge config and its 184 PDBs were deliberately not retrieved, being outside the pEC50 scope.

### 3. Compact data audit built and run — commit `fdd5573` (2026-11-15)
- **What:** generator `pxr/scripts/pxr001_data_audit.py`, artifact `pxr/artifacts/pxr001_data_audit/` with `audit_report.txt` SHA-256 `6bd2eb8aad12c26baee14d3646b11b95b3a1973c5c5546daa90cb95660168bb1`. Re-running reproduced the report byte-identically. The label firewall is enforced in code: the Phase 1/2 files are read with `usecols` restricted to identity columns, so no label value is materialized.
- **Findings:**
  - TRAIN 4,139 rows, one row per molecule, all identity fields unique, no nulls; zero RDKit parse failures in every table.
  - pEC50 spans 1.61–7.55, median 4.65, mean 4.32; 4,064 of 4,139 values are exactly 2-dp; 348 rows have a 95% CI wider than 2 log units; no censoring floor apparent.
  - Every TRAIN row carries a fitted pEC50 and a positive Emax: the primary table contains no inactives or non-responders.
  - The TRAIN `Split` column is constant ("Train"), so the release supplies no internal partition.
  - Phase 1 (253) and Phase 2 (260) are disjoint and together cover all 513 blinded compounds: the full test set has released labels.
  - Train/test overlap is zero at every level tested: `Molecule Name`, raw SMILES, canonical SMILES, InChIKey, InChIKey connectivity.
  - Four InChIKey connectivity-block groups inside TRAIN hold more than one molecule, with pEC50 spreads of 1.22, 0.94, 0.57 and 0.005.
  - Semi-pure HT-chem compound `OCNT-2395416` is structurally identical to blinded test molecule `OADMET-0006234`; a name-based identity rule does not catch it. The crudes table has no such match.
  - counter-assay molecules are a strict subset of TRAIN (2,859), pEC50 non-null in 2,647 of 2,859. DRC curves cover 4,136 of 4,139 TRAIN molecules, so 3 have no curve; 1,255 molecules have one curve, 2,753 two, 128 three. single-concentration holds 10,870 molecules of which only 2,744 are in TRAIN (8,126 appear nowhere else), four concentrations, endpoint `log2_fc_estimate`. crudes: 456 molecules, none in TRAIN. semi-pure: 96, only 5 in TRAIN.
- **Why it matters:** the absence of an internal partition in the release is what makes a predeclared holdout necessary (entry 6); the four connectivity-block groups are the near-duplicates Rev 3 line 109 requires to be checked.

### 4. CYP code reuse assessed, nothing copied (2026-11-15)
- **What:** read-only assessment of what the CYP arm offers the PXR arm. Only `cyp/scripts/ecfp4.py` is genuinely endpoint-agnostic. No scaffold or analog-clustering implementation exists anywhere in the repository: grep for `Butina`, `MurckoScaffold`, `ClusterData` and `taylor.butina` across all Python files returned zero matches, and the CYP analog clusters survive only as frozen CSVs under `cyp/artifacts/cyp_baseline/`.
- **Not runnable as committed:** `cyp/scripts/lgbm_tuner.py` and `evaluate_baseline_test.py` are reusable only as templates — their hardcoded data path `cyp/data/cyp_challenge/cyp-challenge-TRAIN_inhibition.csv` does not exist (the file is at `cyp/data/cyp-challenge-TRAIN_inhibition.csv`), and their objective is ST-RAE, which is not a PXR metric. `st_rae.py` and `multitask_targets.py` should not be carried over.
- **Identity trap recorded:** CYP's `Molecule_Name` values are OCNT ids, while PXR's `Molecule Name` values are OADMET ids and `OCNT_ID` is a separate column. This is why D014 constraint 4 keys the split artifact on `Molecule Name`.

### 5. Split-design diagnostics (2026-11-15)
- **What:** read-only diagnostics on the preserved data, by scratchpad scripts that were not committed; the numbers below are their reported outputs.
  - PXR train is structurally sparse: 3,670 Bemis-Murcko scaffolds across 4,139 molecules, 85% of molecules in singleton scaffolds, largest group 148; median nearest-neighbour Tanimoto (ECFP4, 2048 bits) within train 0.391; only 18% of molecules have any neighbour at ≥ 0.5.
  - Butina clustering is near-degenerate: 4,103 clusters at Tanimoto ≥ 0.7, and 2,977 even at ≥ 0.4.
  - The 513 blinded test molecules sit closer to train (median nearest-neighbour 0.523; 98% have a train neighbour ≥ 0.4, 69% ≥ 0.5) than train molecules sit to each other, while only 19% share a Murcko scaffold with train.
- **Why it matters:** the Butina degeneracy is why the CYP-shaped analog-cluster option was not recommended (D014 Amendment 1 Change 3). Recorded consequence: any internal holdout is a conservative, pessimistic proxy for the official test set.

### 6. D014 declared, with four amendments and an artifact record (2026-11-15)
- **What:** four split options were drafted (Bemis-Murcko scaffold; Butina analog-cluster; generic-framework/graph scaffold; difficulty-matched in the manner of SIMPD). Miles chose A, the Bemis-Murcko scaffold holdout ("just go with A"). The record then developed through four amendments and a factual artifact record, all in `notes/Decisions.md`: D014 at line 6389, Amendments 1–4 at 6453, 6512, 6550 and 6585, the artifact record at 6614 and its commit follow-up at 6657.
- **Approvals, in plain language:** Amendment 1 recorded the Rev 3 reconciliation ("sounds good, move forward with your proposed amendment"); Amendment 2 recorded that constraints 1, 3 and 4 bind ("Yes, all three bind"); Amendment 3 fixed 70/15/15 and seed 20261115 (question as put: "Use 70/15/15 with seed 20261115 for the PXR holdout?", answer "Yes, 70/15/15" — his words named the proportions only, and he was told the record will be amended if he meant a different seed); Amendment 4 fixed the ringless-molecule grouping and directed regeneration. Only Amendment 4 authorizes an action; none authorizes training, tuning or evaluation.
- **Rev 3 reconciliation:** the proposal commits to the type of split (lines 52, 117) and to explicit near-duplicate and analog-leakage checking (line 109), and fixes MAE with Spearman (line 117), but commits to **no algorithm** — no Murcko, Butina, threshold or proportions appear anywhere in it. Option A is the operational instantiation Rev 3 requires to be recorded but leaves unspecified.
- **Evidence:** `notes/Decisions.md` at the lines above; the full approval wording lives in those records and in the main conversation of 2026-11-15.

### 7. Split implemented, regenerated, frozen and independently verified — commit `fabec38` (2026-11-15)
- **What:** `pxr/scripts/pxr001_split.py` with `pxr/tests/test_pxr001_split.py`, 19 tests all passing.
- **Bug caught by a test before any artifact existed:** choosing each group's partition by absolute shortfall against target counts sent every early potency stratum to train, giving synthetic partition medians 6.48 / 10.45 / 10.73. The fix compares filled **ratios** (placed / proportion), so each partition fills at the same relative rate.
- **First generation superseded:** it lumped all 30 ringless molecules into validation through the empty Bemis-Murcko scaffold string (train 2,897 / validation 621 / holdout 621; 3,670 groups). Miles amended the rule (D014 Amendment 4) and the split was regenerated on the same seed. `check_invariants` was narrowed to exclude only the empty scaffold from the scaffold-spanning assertion, with that exclusion stated in the artifact's `PROVENANCE.txt` and covered by tests confirming the guard still catches a ringless near-duplicate pair being separated.
- **Frozen artifact:** `pxr/artifacts/pxr001_split/pxr001_split_assignments.csv`, SHA-256 `c6028f1b7216ab2ea4bab96616fc02b95ce011be20b1b96f807d1e7d582ece46`. Realized: 4,139 molecules in 3,699 groups; train 2,897 (69.99%, 2,584 groups), validation 621 (15.00%, 557), holdout 621 (15.00%, 558). Ringless molecules distributed 19 train / 3 validation / 8 holdout.
- **Independent verification:** a separate script recomputed scaffolds and connectivity blocks from the preserved raw CSV rather than trusting the artifact's own columns. Both columns reproduce; 0 ringed scaffolds, 0 connectivity blocks and 0 groups span a partition; all four near-duplicate pairs are intact and all in train; 0 duplicate identities, 0 unassigned rows, identity set equal to TRAIN; 0 names shared with TEST_BLINDED; regeneration is byte-identical; partition pEC50 medians 4.65 / 4.64 / 4.66, means within 0.03. The verification script was a scratchpad script and is not committed.

### 8. Session close declared by Miles (2026-11-15)
- **What:** Miles approved creating this session log and declared the session close: "did we create Tripwire_PXR_001.md yet, if not please do that work. then we'll note this as a session close and I'll push". The project-log row for Tripwire-PXR-001 was written on that declaration.
- **Commits this session, all local, nothing pushed:** `5cca55c` (preserved release), `fdd5573` (audit), `fabec38` (split freeze). `d1ccb7a`, a CYP D013 commit of Miles's, landed between `fdd5573` and `fabec38`. CYP and ActFound work was not modified at any point in this session.
- **Not done / not authorized:** no PXR model training, tuning or evaluation; no Slurm submission; no GPU allocation; no classical baseline built.
- **Open items and unknowns recorded at close:**
  - Rev 3 is not in the repository, so the governing document is not version-controlled with the work it governs.
  - Molecule standardization is undeclared: Rev 3 line 107 names it as a shared component but specifies none; the split used the official SMILES as released, and any standardization decision would change scaffold assignment.
  - The numerical improvement bar required by Rev 3 line 129 has no value yet.
  - Rev 3 line 90 makes the ECFP4 + gradient boosting baseline mandatory and first ("Always, first"), with RF as an additional classical comparison. It is the next piece of work and now has a holdout to be scored on. **Correction to the handoff:** the earlier handoff wrongly described this baseline as conditional on a 4–5 hour effort cap.
  - Eligibility of crude and semi-pure HT-chem material for a pEC50 target is undecided, and the semi-pure/blinded structural match (`OCNT-2395416` == `OADMET-0006234`) is unresolved.
  - The ringless distribution 19 / 3 / 8 is not proportional to 70/15/15; expected for 30 indivisible singleton groups across potency strata, recorded in case Miles wants a further grouping rule.
  - The 3 TRAIN molecules without DRC curves, the 348 wide-CI rows, and the absence of inactives in the primary table are observed data properties with no exclusion decided.
  - **Correction to the handoff:** Rev 3 line 53 settles the PXR evidence class as "Internal-retrospective evaluation, not a prospective blind challenge result", correcting the handoff's description of the internal holdout as a prospective evaluation.
  - At the time of writing, `notes/Decisions.md` carried 17 uncommitted lines (the `fabec38` commit follow-up), and this log plus the project-log entry add further uncommitted changes.
