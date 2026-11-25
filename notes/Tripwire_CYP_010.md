# Tripwire-CYP-010 Session Log

**Date:** 2026-11-24 (opened)  
**Status (current):** Open. Implementation and local tests are reported complete, the local weight preview and the baseline tail cohorts are done, D024 Amendment 3 is recorded, the CYP3A4 coverage grouping is built and the penalty-comparison package is built locally, and cell 0 of each penalty arm has run on CoreHPC and passed the integrity gate (section 10). Cells 1-149 of each arm are not yet submitted. Nothing has been committed.

## 1. Session purpose and starting state

Session 010 takes up the three-arm comparison of the updated-T121 recipe with and without a screening-informed extra loss term: updated T121 unchanged, a Q3-cutoff penalty and a response-rank penalty. It commenced on 2026-11-24 from `notes/Handoffs/Tripwire_CYP_009_Handoff_20261124.md`.

Starting state:
- The handoff was read. D024 and its Amendment 1 were confirmed to be present in `notes/Decisions.md` and consistent with the handoff.
- Commit `9bcf355` holds the session 009 record edits, which the handoff listed as uncommitted.

## 2. Implementation plan and approval (2026-11-24)

A consolidated plan was drafted from code reading, revised, and approved. It is recorded as D024 Amendment 2 in `notes/Decisions.md` (Approved), which holds the design, the settled definitions, the one-off runtime-sizing deviation and the execution order. It lifts the bar of D024 and Amendment 1 on implementation of code, configurations and tests, closes the items Amendment 1 left open on adoption readings, per-compound predictions and runtime sizing, and states that planned items are not re-submitted for approval. The bar it names as lifted is the implementation bar; it states no commit authorization. Execution commands are run by the project owner, as before.

The CYP2C9 and CYP3A4 coverage and direction check is not closed by the amendment, and the r2 and r3 confirmation stage remains separately authorized.

**Runtime estimate.** About 5.8 GPU-hours per arm if a cell takes the D022-array mean of 139.867 s (`notes/Tripwire_CYP_009.md` section 9). That mean comes from a different model; this is an estimate only.

## 3. Source-reading finding: prediction scale at validation (unverified)

- Finding from reading source, not verified by execution: installed Chemprop 2.3.1 keeps the output transform in train mode during validation. This implies that the scorer receives standardized predictions and unscales them.
- It is contrary to the statement in the CYP-009 handoff (section 4.2: validation in eval mode returns unscaled predictions to the frozen scorer) and to the keep-raw branch inference in `notes/Tripwire_CYP_009_supporting.md` (the same inference appears in `notes/Tripwire_CYP_009.md` section 12). Those records mark the keep-raw branch as an inference.
- Saved post-scorer predictions are expected to be unaffected.
- The Chemprop version on CoreHPC is unchecked.
- Status: unresolved. A planned test is to check the finding before any record correction; no record has been corrected.

## 4. Local tests

The focused CYP-010 test files and the full local suite (excluding `test_t121_training_weights.py`) were reported as all passing (console output of local commands). An earlier attempt used the project `.venv` Python, which lacks torch and chemprop, so torch-based tests could not run; the passing run used the `tripwire_cyp002` environment's own Python. Test counts were not captured and are not recorded. Equivalence at lambda = 0 against the pre-edit module is part of the planned tests (D024 Amendment 2 item 1).

## 5. Local weight preview

Run over all 50 folds x 4 enzymes x 2 arms; output directory `analysis/cyp_010_penalty_preview_20261124`, report `preview.json` (SHA-256 `06c4d49c9c4ffad1e616cbaf231e5b0cb224c2cd37d58bfa755759c72b77741e`). Screen input hash `4f61be8ffdec767a61981ab89e3f06d28850603c3fa4ad0715a2183d4261a706`; identity input hash `2f3045124dbf138ec535abfd8b697a6e2e2cd605d2bccd9f30a58f4ba5464f54`.

Result: no disabled enzyme-fold-arm, no training-SMILES differences, no automatic stop. No missing-value, unverified-identity or conflicting records in any enzyme.

| Enzyme | Labelled training molecules per fold (range over 50 folds) | Population per fold |
|---|---|---|
| CYP1A2 | 1062-1101 | 1062-1101 |
| CYP2C9 | 965-1008 | 965-1008 |
| CYP2D6 | 1123-1157 | 1123-1157 |
| CYP3A4 | 1766-1815 | 1388-1432 |

CYP3A4: 366-391 labelled molecules per fold have no screening row; none of the other enzymes have any. Flagged fraction: cutoff arm 0.250 (0.250-0.251 for CYP2C9); rank arm 0.999 (all but the lowest-ranked cell receive a positive weight).

Decision on the CYP3A4 gap: handled by the D024 missing-value rule (no extra penalty; ordinary supervised loss and denominator kept); the plan continues to packaging (D024 Amendment 3 item 1).

## 6. Frozen configurations

Under `cyp/artifacts/d024_penalty/` (SHA-256):
- `penalty_q3.json` `ddfe1f01e853e990fabd3ac60e0e7fec5b4bb4e998800b46440b50a4d372af7b`
- `penalty_rank.json` `a20552a33392891dcd1e9eac4af178825c7fc476a7b596004e964f9fc58edefe`
- `evaluation.json` `8e7aabecdaa364f33faf0f59a9292e97a6455d9fcaa9c94dfec34863fd669c77`

## 7. Tail cohorts

Built once from the saved baseline predictions (`t121_valinf_full_20261118/full_cells_000_149`, checked against `full_cells_000_149_SHA256SUMS.txt`) into `analysis/cyp_010_tail_cohorts_20261124.json` (SHA-256 `cc336d4ab3de85a5c8da0ccfd2408b43a27e02c7b5ea2ee41b407388c0d73e7e`). The file is frozen before any penalty-arm result exists.

| Enzyme | Molecules | Cohort size per tail |
|---|---|---|
| CYP1A2 | 1201 | 120 |
| CYP2C9 | 1093 | 109 |
| CYP2D6 | 1269 | 127 |
| CYP3A4 | 1986 | 199 |

## 8. CYP3A4 coverage split (report-only)

D024 Amendment 3 item 2 adds a descriptive report of CYP3A4 results separately for molecules with and without screening coverage, alongside the full-enzyme and fixed-tail results, from the same saved predictions with no new training. It is not an adoption clause and does not change Amendments 1 or 2. "Covered" means a verified CYP3A4 screening value exists under the verified-value rule (Amendment 2 item 2); the reading is open to correction.

The grouping was built once from the screening inputs (screen files hash-checked against the contract) and the baseline predictions into `analysis/cyp_010_cyp3a4_coverage_20261124.json` (SHA-256 `3640127b4fb5b4af135bf27fe9e31c29be6de89138ffa5525ea729fdc2342fb2`), before any penalty-arm result exists. Of 1986 molecules, 1565 are covered and 421 are not covered; all 421 have no CYP3A4 screening row and no other status occurred. The report-only split therefore uses fixed groups of 1565 and 421 molecules.

## 9. Package, baseline provenance and timing

**Package (local only).** Built from the unchanged reference package plus the listed overlay files: `t121_d024_build_20261124/t121_d024_package_20261124.tar.gz`, SHA-256 `83b61e6ae961ad62115f0e2335622bb0c45ee07463a5a5171e7a4b977ae4edf3`, 115 files listed in its `SHA256SUMS.txt`. It contains the pins file, the three frozen configs (section 6), the coverage config and the new modules and tests. It was built after the coverage addition. The focused test rerun after that addition has not been reported, so its result is not recorded. Nothing has been transferred or run.

**Baseline selected record (CoreHPC).** `/scratch/user/mhuang8/tripwire_staging/t121_production_chain_20261117_screen_run/selected_screen_record.json`, SHA-256 `4da7a07de2926a62894441da3cd0183b0e6c38f87902266fba40fc3195b1a0fc`. It records epoch 16 (zero-based), score 0.2029570272781568, and checkpoint `preparation/epoch_checkpoints/epoch_016.ckpt` with SHA-256 `96b75ceb4ddd629373e2356ba5cff3ae8ba58d27a4de1c48db0091646bcd5e7c`. This agrees with the checkpoint hash prefix recorded for the baseline cells (`96b75ceb4ddd6293`).

**Measured same-recipe runtime.** Baseline array job 3105880 (sacct, cells 1-149, all COMPLETED): per-cell elapsed 00:01:29 to 00:02:56, typically about two minutes (cell 0 was an earlier pilot and is not in that list). This is the measured basis for the Amendment 2 sizing, about 5 GPU-hours per 150-cell arm, with the 12 h per-cell request retained. It supersedes the section 2 estimate, which came from a different model.

## 10. Cell-0 integrity gate (CoreHPC)

Cell 0 (repetition 1, fold 1, seed 20261101) is a production cell of each arm, run first as the planned integrity gate. Source: console output of the integrity check over `/scratch/user/mhuang8/tripwire_staging/t121_d024_{q3,rank}_cells_run/trials/trial_121/cells/cell_000`; Slurm jobs 3254977 (cutoff arm, array task 0) and 3254978 (rank arm, array task 0), both COMPLETED. Scores were not printed and change nothing.

**Gate result: PASS for both arms.**
- `COMPLETED.json` present, no INCOMPLETE marker.
- Arm labels `updated_t121_q3_penalty_from_screen_checkpoint` and `updated_t121_rank_penalty_from_screen_checkpoint`.
- Screen checkpoint hash prefix `96b75ceb4ddd6293` (the r1 checkpoint, matching section 9).
- Penalty weights hash equals the local preview value for that fold: cutoff `3346f1d25608bc3237599a0649a4e92071548ea092330e94cdaaf861a331a584`, rank `3f2a9adb332294e2ab0286901c44f3a19c9eb5c45c4617f08837f72bf649363e`.
- Epoch-log lines equal completed epochs (cutoff 36 and 36, rank 35 and 35); lambda 1.0 in every record shown.

**Loss components (cell-count-weighted; first and last logged epoch).** This is the first measurement of the lambda-scaled penalty relative to the base loss. It is a one-cell observation, descriptive only.

| Arm | Epoch 0 base | Epoch 0 unscaled penalty | Last epoch | Last base | Last penalty | Penalty / base at epoch 0 | Penalty / base at last epoch |
|---|---|---|---|---|---|---|---|
| Cutoff | 0.8726 | 0.3595 | 35 | 0.2604 | 0.0271 | about 41% | about 10% |
| Rank | 0.8544 | 0.1929 | 34 | 0.2502 | 0.0378 | about 23% | about 15% |

Flagged (positive-weight) observed cells at the last epoch, by enzyme (CYP1A2, CYP2C9, CYP2D6, CYP3A4): cutoff 267, 245, 285, 354; rank 1068, 979, 1140, 1413.

**Not seen.** Elapsed times of the two jobs; the outputs of the archive/checksum check, the loaded-module check and the cell-planning step (the cells ran from the package paths, so the code ran, but those printed checks were not returned); the cluster library versions remain unrecorded.

## 11. Claim boundary

The comparison is development evaluation under D024 and its amendments; protected test data are untouched by it. The section 3 finding is a source reading only. The only penalty-arm result is the cell-0 integrity observation (section 10); no comparison result exists.

## 12. Stopping point and next step

Cell 0 of each arm has run and passed the integrity gate. Next: cells 1-149 of each arm, not yet submitted.
