# Tripwire-CYP-010 Supporting Record

Supports `notes/Tripwire_CYP_010.md`. Identifiers and hashes only.

## Screen-head loading verification (2026-11-24; source: output of read-only checks reported from the CoreHPC login host `chpc-ucsf-login-vm1.corehpc.ucsf.edu`)

### CYP r1 screening run

Root `/scratch/user/mhuang8/tripwire_staging/t121_production_chain_20261117_screen_run`.

| Item | SHA-256 / value | Status |
|---|---|---|
| `selected_screen_record.json` | 4da7a07de2926a62894441da3cd0183b0e6c38f87902266fba40fc3195b1a0fc | matches recorded value |
| `preparation/epoch_checkpoints/epoch_016.ckpt` | 96b75ceb4ddd629373e2356ba5cff3ae8ba58d27a4de1c48db0091646bcd5e7c; 24,322,347 bytes; 50 epoch checkpoints present | matches recorded value |
| `preparation/preparation.json` | f3cb45708a4e1f0cb8386fcec66afd3343396207d74dbed8c500672a18599e20 | newly recorded |
| `preparation/split_assignments.csv` | d26da9ce522653816dab0876cee082e63f7a49c7fe486e863c1d77c5c7a27544 | newly recorded |
| `preparation/epoch_checkpoints/selection.json` | 61c87012752cdfb722e83ceb1d13c4416bf96b08dca03ba1b59f6d9e213b69f1 | newly recorded |
| `actfound_model_side.py` | 86ccf4b135a0388c147d2838e5d12ead2b5aaf5968f7140ec341e7950d3a62fd | equals value in `notes/Tripwire_CYP_006.md` and the D024 package SHA256SUMS |
| D024 evaluation report `/scratch/user/mhuang8/tripwire_staging/t121_d024_report_20261124.json` (CoreHPC) | 7cefd735d92c61f4b18822bde0ca18afcd15aeba9ec23d2ff448b628cdf462b1 | previously missing, recovered 2026-11-26 from CoreHPC `sha256sum` output; report produced 2026-11-25 |

- `sha256sum -c` of the 16 pinned sources and inputs from the reviewed r1 manifest (`screen_manifest_v2.json`): 16 of 16 OK.
- Selection record: zero-based epoch 16 of 50, score 0.2029570272781568. Per-enzyme native weighted MSE at epoch 16 (CYP1A2, CYP2C9, CYP2D6, CYP3A4): 0.10042278332384545, 0.10268352820917172, 0.07421386418979108, 0.5345079333898188.
- `completion.json`: status completed, 50 epochs, seed 20261101, no full-screen refit.
- `preparation.json`: train 6,252, validation 695 (6,947 total), split seed 20261001; message_hidden_dim 600, depth 4, FFN 2000 x 1, dropout 0.4, batch size 16, native log2fc, no target normalization. The split file has CRLF line endings, which garbled the awk partition count; counts agree with `preparation.json`.
- Environment `/scratch/user/mhuang8/envs/tripwire_cyp002`: chemprop 2.3.1, torch 2.14.1+cu126, lightning 2.6.6.
- Local (not CoreHPC) source comparison: `cyp/scripts/t121_screen_stage.py` (b9b3c013...) and `t121_weighted_training.py` (3f4d6c32...) differ from the r1-pinned versions (def6ddb6..., cf8f0321...).

### PXR-004 stage-1 (sequential arm)

Under `/scratch/user/mhuang8/pxr004_pilot_20261119/seed<S>/sequential_seed<S>/stage1/`. In all three seeds `best.ckpt` SHA-256 equals metrics.json `best_checkpoint_sha256` and the stage-2 manifest `encoder_source`; each about 112 MB.

| Seed | best.ckpt SHA-256 | Best epoch | Epochs run | 8.25 uM MAE (n 974) | 8.25 uM Spearman | Pooled masked standardized MSE |
|---|---|---|---|---|---|---|
| 20261125 | 56ba8157de3285b7c7c94e3f3242ed1dbdda171e8c3b745a17b3935e4a3cff1d | 2 | 23 | 0.19876082436789758 | 0.5713548514033054 | 0.5606804132150986 |
| 20261126 | 15eabba65475cbfad748d8a902cf8f8066dcbc1270614d32f6e262a11b9e17b2 | 4 | 25 | 0.19447852186423983 | 0.5704787434803834 | 0.5588789975435186 |
| 20261127 | 6fabe68ba493c53798defc7f006832b22aeb07663566921f5148223d15acbf98 | 20 | 41 | 0.2069821233249088 | 0.5440995976826861 | 0.5814922265129593 |

- Scaler (identical across seeds): mean [0.301972839854651, 0.5814180428191748, 1.16459734781899], scale [0.37739070416368464, 0.5379514342062427, 0.6199903188725561].
- Stage-1 train label counts 8763 / 7692 / 539; monitor 974 / 848 / 67 (8.25 / 33 / 99 uM).
- CheMeleon weights SHA-256 c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651 in all three manifests.
- Staged package `/scratch/user/mhuang8/tripwire_staging/pxr004_stage_20261119`: `sha256sum -c ../pxr004_stage_hashes_20261119.txt` 16 of 16 OK.

## Smoke-test files (local, uncommitted; not executed, not run under pytest; syntax parse only)

| File | SHA-256 |
|---|---|
| `cyp/artifacts/screen_heads_smoke/cyp_r1_pins.json` | 0e5160782f3289924d77acaa78d33798e3916a30583bb45d73cd4405d2e147f9 |
| `pxr/artifacts/screen_heads_smoke/pxr_stage1_pins.json` | 144695415cb940d52f47717d26d7eae07a4297850b7a0f838886398c58d7257d |
| `cyp/scripts/screen_head_smoke_cyp.py` | 55639bb7000b620b58b3b829b6ab4a80db9ac4ee1f146b9d23d7e4e107ad55f5 |
| `pxr/scripts/screen_head_smoke_pxr.py` | 65ebd11f62317156d081a73f89a3ac94caccfdb31304ba06518202b8d086c833 |
| `cyp/tests/test_screen_head_smoke_cyp.py` | c5f00978fb7c10b9f6c404d6a9025502e32fa1e2e6b52fe745f10e311ce6d6e6 |
| `pxr/tests/test_screen_head_smoke_pxr.py` | b0ea3d23252a08cce5f2c71e6c1cbab064b31815062841785f9c59355d403d8b |

Hashes computed 2026-11-24 on the local checkout; files uncommitted. These files were later run on CoreHPC (next section); the local files themselves were not run under pytest.

## Screen-prediction exposure files (D025 Amendment 1; local, uncommitted, not run, syntax parse only; SHA-256)

| File | SHA-256 |
|---|---|
| `cyp/scripts/screen_predict.py` | 40c27c155bf5bd87395ffed22184b571bf4a0e7dbd2d16e1597509e649ef4208 |
| `cyp/scripts/screen_predict_reference.py` | ed9e8d6113c5edf8141726b51c2c09064dbba98441366d78f4c3a1daa9a7bc5d |
| `cyp/tests/test_screen_predict.py` | bdc62a8c4afc3e758ffe2782989dde795c7051f5b327f6eb7ddf5128615f53ed |
| `cyp/tests/test_screen_predict_reference.py` | 16caf3f852094db434a74308c2b89cfd65b92b828960dac385f57dfd28890dbe |
| `cyp/artifacts/screen_predict_smoke/synthetic_input.csv` | 97434fae707e19ba33b132cc7d5569ca2356c1bfd737a4160b8196705688d67d |
| `cyp/artifacts/screen_predict/protected_inputs.json` | d40fe73ade3fe53b1c34bed0e38ca37ca72d9113315c447cf57b4ea872cb5c2c |

`protected_inputs.json` holds 13 protected-file hashes computed locally on 2026-11-24 (hash-only reads) from the CYP challenge blinded and unblinded files, the CYP training file containing the internal-test partition, the screening file, and the PXR training, single-concentration, counter-assay, blinded test and unblinded label files; cross-checked against recorded SHA256SUMS where listed. The earlier loading-smoke scripts and pins are unchanged.

## Screen-prediction CoreHPC runs (reported 2026-11-25; source: pasted output from the CoreHPC login host `chpc-ucsf-login-vm1`, CPU, 2 threads, directory `/scratch/user/mhuang8/tripwire_staging/screen_predict_20261124/`)

Script `cyp/scripts/screen_predict.py`, SHA-256 40c27c155bf5bd87395ffed22184b571bf4a0e7dbd2d16e1597509e649ef4208; synthetic input SHA-256 97434fae707e19ba33b132cc7d5569ca2356c1bfd737a4160b8196705688d67d.

### Script and input identity (confirmed 2026-11-25; source: pasted output of a read-only command on the CoreHPC login host in the same directory)

- `sha256sum` on CoreHPC: `screen_predict.py` 40c27c155bf5bd87395ffed22184b571bf4a0e7dbd2d16e1597509e649ef4208 and `screen_predict_reference.py` ed9e8d6113c5edf8141726b51c2c09064dbba98441366d78f4c3a1daa9a7bc5d, equal to the local hashes.
- The `script_sha256` recorded in the manifests of `synthetic_b3`, `synthetic_b64` and `pxr_reference_b16` is 40c27c15... in all three, so those runs used this script.
- The script verifies the synthetic input and the protected-hash list at run time against the declared `--input-sha256` and `--protected-hashes-sha256`, so their CoreHPC copies matched the declared hashes.

### Synthetic input, runs A (batch size 3) and B (batch size 64, with the separate PXR transform check)

- Both completed. 14 input rows; status counts: ok 10, invalid_missing_id 1, invalid_empty_smiles 1, invalid_structure 2 (invalid SMILES `not_a_smiles` and `C1CC`; RDKit parse errors were printed to the log as expected and are not run failures).
- Duplicate id recorded: `00042` at rows 0 and 8.
- Manifest names exactly four checkpoints: `cyp_r1_epoch_016`, `pxr_stage1_seed20261125`, `pxr_stage1_seed20261126`, `pxr_stage1_seed20261127`.
- Read-only comparison of A and B: same ids, same statuses, all prediction columns identical within 1e-6 (batch size does not change values or row order).
- PXR transform check on a deep copy (run B, 10 molecules): `inference_model_untouched` true; max absolute difference between eval output and standardized output x scale + mean 2.73e-08.

### Targeted comparison for the changed PXR loading path (D025 Amendment 1 item 6)

- Reference input `pxr_reference_input.csv` (16 rows), SHA-256 c6399d5e40137bacfe12516a57b16d579ffcc15939f1455277e6500ee707be67, built from the three PXR loading-smoke `result.json` files and the staged `pxr002_train_frame.csv` (SHA-256 024ed1072a399fc13df0fac5dc26b0d913416c0fb62fef1c8b66547907feec06; read only to look up the SMILES of the 16 ids).
- All 16 ids carry the `OADMET-` prefix, i.e. primary-train (DRC subset) molecules, which confirms the earlier inference that the smoke molecules are hit-enriched and not representative.
- `screen_predict.py` on that input (batch size 16): 16 of 16 ok.
- Comparison against the saved smoke predictions, tolerance 1e-5 absolute: PASS. Max absolute difference 0.0 for all three seeds (n 16 each); the three-seed mean equals the arithmetic mean of the retained seed columns (max difference 0.0). Predictions file SHA-256 04647fa89cdef664109361432d359fdd4e952eb8f72567e8a1f632c1c346d7ec. Labels not used; nothing scored.
- Assistant note: the comparison used the same batch size and molecule order as the smoke (16), so exact equality is expected; batch-size invariance is covered by the synthetic runs. The change tested is that the PXR scaler is built from pinned constants rather than recomputed from training frames.

### Synthetic output CSV (`synthetic_b3`, 14 rows; read 2026-11-25)

- `input_row` 0 to 13 contiguous in input order.
- Ids preserved literally: `00042`, `1E5`, `NA`, `nan`, `id with, comma` (CSV-quoted), `trail-id ` with its trailing space kept and warning `id_has_surrounding_whitespace`, `0x1F`.
- Duplicate id `00042` flagged `duplicate_id` on rows 0 and 8.
- Row 9 (empty id) kept as `invalid_missing_id`; row 10 as `invalid_empty_smiles`; rows 6 and 7 `invalid_structure` with reasons naming both the CYP and PXR featurizer checks. All prediction cells are empty on invalid rows. `C[Hg]Cl` (row 5) is ok.
- `mean3` equals the arithmetic mean of the three seed columns on spot-checked rows 0 and 13 (assistant arithmetic).

Descriptive observations on the 10 ok rows (public, mostly small molecules chosen for input-handling coverage; not a test of predictive performance; not representative; out-of-domain inputs; nothing scored; no interpretation established):
- CYP outputs lie between -0.47 and +0.34 native log2fc. CYP2D6 is positive on all 10 rows (0.16 to 0.34), as in the 16-molecule loading smoke (0.23 to 0.36). CYP1A2 -0.47 to -0.007; CYP2C9 -0.19 to -0.016; CYP3A4 -0.23 to -0.029.
- PXR 8.25 uM seed means range -0.09 to 0.54. Seeds disagree most on tiny out-of-domain molecules: propane (row 12) 0.316, 0.559, 0.732; 1-chloropropane (row 13) 0.260, 0.417, 0.738; methylmercury chloride (row 5) 0.225, 0.246, 0.599.

### Not yet seen or reported

Timing. Local pytest, 2026-11-25 (source: pasted console output): pytest 9.1.1, Python 3.12.14, interpreter `/opt/homebrew/Caskroom/miniforge/base/envs/tripwire_cyp002/bin/python`, rootdir `/Users/mileshuang/Desktop/tripwire-project`; 38 items collected, 38 passed in 7.79 s, none skipped, across `cyp/tests/test_screen_predict.py` (including its torch/chemprop model-behaviour tests) and `cyp/tests/test_screen_predict_reference.py`. (The transfer hash-check output is superseded by the CoreHPC `sha256sum` and manifest confirmation above.)

## Execution-location decision (D025 Amendment 2, 2026-11-25)

Evidence behind the earlier open item, as recorded on 2026-11-24: no CoreHPC usage-policy or login-node document exists in the repository records (notes, handoffs, CLAUDE.md, references, external and qualification searched); a web search did not locate a CoreHPC-specific policy (the older Wynton cluster's guidance applies to a different cluster and is not authoritative here); interactive `salloc` and `srun` fail on the login VM; earlier batch runs were one-off approved deviations. D025 Amendment 2 settles the location as the established CoreHPC CPU route (environment `/scratch/user/mhuang8/envs/tripwire_cyp002`) without a new policy-search prerequisite. Decision text is in `notes/Decisions.md`, not repeated here.

## Real input set definition (D025 Amendment 3, 2026-11-25)

Counts carried from the earlier read-only analysis (not re-derived here; the builder fixes the exact final count): 449 base; 213 after removing CYP internal-test and challenge-blinded tiers (77) and PXR holdout-role molecules (190); 209 after removing the 4 CYP `bridges_protected` molecules; about 207 after the canonical isomeric SMILES agreement rule (2 expected). Named reads for the builder: screening file; `adjudication.tsv` and `cluster_adjudication.tsv` from `actfound_screen_loader_20261117.tar.gz` (member hashes checked); `pxr/artifacts/pxr002_aux/pxr002_eligibility.csv`; PXR single-concentration file; PXR `TRAIN.csv`. 
### Input builder files (2026-11-25; local, uncommitted, syntax parse only, not run)

| File | SHA-256 |
|---|---|
| `cyp/scripts/build_screen_input_set.py` | 6d0682a18477596726ecd7ae2481b3b4fd87b22d3630e3edb6162755d6ba2429 |
| `cyp/tests/test_build_screen_input_set.py` | 413a0fde9dd5cb6b12079425adb3d6ebb6a94699d6e674fbf35de5340d036911 |

Input hashes confirmed locally by hashing, all equal to earlier recorded values (full values are in the earlier records where they appear; prefixes shown here):
- `actfound_screen_loader_20261117.tar.gz` 9238985ee22fa9de365349707f37659dea159b689b974760fad48319530bdba8 (first recorded here);
- members `screen_full.csv` 4f61be8f..., `adjudication.tsv` 2f304512..., `cluster_adjudication.tsv` eadd679f...;
- `pxr002_eligibility.csv` 0d5f2d00...;
- PXR single-concentration file e781f840...;
- PXR `TRAIN.csv` fdc2252b....

Design notes:
- The builder reads from the tarball members with member hashes checked.
- The PXR comparison SMILES is `TRAIN.csv` for primary roles and the single-concentration string for aux-only molecules. This role-appropriate comparison may change the earlier count of about 207, which used the single-concentration string for all roles; the exact final count comes from the builder report.
- It writes `screen_input_set.csv` (OCNT_ID, SMILES), `screen_input_mapping.csv`, `screen_input_exclusions.csv` and `screen_input_set_report.json` into a new directory, and exits nonzero if the step counts differ from the recorded 449, 213 and 209 (outputs are still written).
- Nothing has been run.

## First real screen-prediction run (reported 2026-11-25; source: pasted output from the CoreHPC login host)

- Run: `screen_predict.py` on `real_input_20261125/screen_input_set.csv` (built by the approved input builder, D025 Amendment 3; id column `OCNT_ID`, SMILES column `SMILES`), directory `/scratch/user/mhuang8/tripwire_staging/screen_predict_20261124/`, output `results/real_set_20261125/`.
- Settings: device CPU, 2 threads, batch size 64; eval and inference_mode on every batch with no training-mode toggling; PXR seeds first, then the CYP builder; PXR transform verification not requested.
- Rows: 208 input rows, 208 ok, 0 `invalid_missing_id`, 0 `invalid_empty_smiles`, 0 `invalid_structure`, no duplicate ids; `screen_predictions.csv` has 209 lines (header plus 208 rows).
- Manifest `script_sha256` 40c27c155bf5bd87395ffed22184b571bf4a0e7dbd2d16e1597509e649ef4208 (equal to the verified script); the manifest names exactly four checkpoints (`cyp_r1_epoch_016`, `pxr_stage1_seed20261125`, `pxr_stage1_seed20261126`, `pxr_stage1_seed20261127`).
- Timing: total 34.29 s on the login host, including model loading; a single run, not a benchmark and not a production estimate.
- The final set size of 208 is the builder's result and replaces the earlier estimate of about 207.
- Hashes (source: pasted `sha256sum` output on the CoreHPC login host, same directory; the paste re-showed the manifest facts above unchanged):

| File (under `/scratch/user/mhuang8/tripwire_staging/screen_predict_20261124/`) | SHA-256 |
|---|---|
| `real_input_20261125/screen_input_set.csv` | ec85098928d21928f59002b32985ae7db530fbe0dc9f15bfe9405e7c90f91537 |
| `results/real_set_20261125/screen_predictions.csv` | 0a04fe735ea5d55e2fbd0ad72ab62e7f696d137f2865cec1213211a48eeb5d1f |
| `results/real_set_20261125/screen_predictions_manifest.json` | 214d805715dcb54510238f232bf6fa75a2e5a46fee8239208d24d96548ad533f |

- Builder report (2026-11-25; source: owner-pasted output of a read-only command on the local machine reading `analysis/cyp_010_screen_input_set_20261125/screen_input_set_report.json`; an earlier attempt to read it in the CoreHPC session returned nothing):
  - Step counts: base 449; after_protected_and_holdout 213; after_bridges 209; after_smiles_rule 208. `counts_match_records` True (equal to the recorded 449, 213, 209).
  - Removal reasons (a molecule can carry several): cyp_protected_tier 77; pxr_holdout_role 190; cyp_bridges_protected 5 (4 unique after the first two removals); smiles_structure_mismatch_or_unparseable 1. The earlier estimate of 2 used the single-concentration string for all roles; the role-appropriate comparison gives 1.
  - Selected 208: CYP tier train 178 and validation 30; PXR role primary_validation 160 and aux_only 48; PXR eligibility reason primary_validation_aux_excluded 160, aux_only_group_contains_validation_or_holdout 37, aux_only_group_contains_test_structure 11. SMILES string equal for 201 and canonical-equal with a different string for 7.
  - The 11 aux-only molecules whose group touches official-test structures were kept under the approved rule. They are identifiable in the mapping file (`pxr_eligibility_reason`), so they can be set aside at analysis time without a new run. Whether to set them aside is open (an explicit decision entry is required).
  - The local `screen_input_set.csv` hash in the report is ec85098928d21928f59002b32985ae7db530fbe0dc9f15bfe9405e7c90f91537, identical to the CoreHPC copy hash recorded above, so the transfer is confirmed intact.
- Not yet seen or reported: the builder test results, and the builder's exit code as such (`counts_match_records` True implies exit 0 by the script's logic, an inference).
- No predictions were reviewed, scored or evaluated; no label was used.

## Descriptive evaluation set (D026, 2026-11-25)

Derived from the builder report counts above: 208 selected, minus the 11 aux-only molecules with eligibility reason `aux_only_group_contains_test_structure`, gives 197 (160 PXR primary-validation, 37 aux-only; arithmetic from the recorded counts). The exclusion is by the mapping file column `pxr_eligibility_reason`. No evaluation result exists yet. Decision text is in `notes/Decisions.md`.

Baseline inputs (D026 Amendment 1): CYP baseline source `preparation/split_assignments.csv` SHA-256 d26da9ce522653816dab0876cee082e63f7a49c7fe486e863c1d77c5c7a27544 (CoreHPC artifact under the r1 screening run, hash first recorded in this file's screen-head verification section; 6,252 train rows, to be transferred, not rebuilt). PXR baseline source `pxr/artifacts/screen_heads_smoke/pxr_stage1_pins.json` SHA-256 144695415cb940d52f47717d26d7eae07a4297850b7a0f838886398c58d7257d, `expected.scaler_mean[0]` = 0.301972839854651 (8,763 stage-1 training labels at 8.25 uM, monitor excluded).

### Evaluation files and builder-output hashes (2026-11-25; local, uncommitted; script and tests syntax-parsed only, not run)

| File | SHA-256 |
|---|---|
| `cyp/scripts/evaluate_screen_predictions.py` | 5eaafb5f3338737d48422902bf01c55ecb75bc54b9d608735fbec4e9f73a1dcb |
| `cyp/tests/test_evaluate_screen_predictions.py` | 252625c22275dae430d905004cc6b65f338be5473a8b000aa35f3cd022b58351 |
| `screen_input_mapping.csv` (builder output, computed locally) | fd9ab197b736039f497c62085ce0617e8ce3d4f0631729f8c93fa761887111e3 |
| `screen_input_set.csv` | ec85098928d21928f59002b32985ae7db530fbe0dc9f15bfe9405e7c90f91537 |
| `screen_input_exclusions.csv` | 7934255e9b9ac4b310d8cc78b597f176284d23f213790dad459227a879dd0815 |
| `screen_input_set_report.json` | 0d52993a4d3a3467388d08f859d752881766261393289f4652de3e77a8652dd5 |

Read-only check of the builder's mapping file (local):
- 208 rows, with identical ids and order in the input and the mapping; no duplicate `OCNT_ID` or `pxr_row_id`.
- `cyp_molecule_name` equals `OCNT_ID` for 208 of 208; aux-only `pxr_row_id` equals `OCNT_ID`, and primary-role `pxr_row_id` differs.
- All 208 SMILES pairs are canonically identical (201 string-identical, 7 string-different).
- 241 exclusion rows, none among the kept molecules.
- After setting aside the 11 test-structure aux-only molecules, the evaluated set is 197 (primary_validation 160, aux_only 37). CYP tier by PXR role: train/primary_validation 137, train/aux_only 32, validation/primary_validation 23, validation/aux_only 5.
- Label presence (not values) for the 197: all four CYP enzymes and the PXR 8.25 uM value are present for all 197 (160 and 37 by stratum); no duplicate label rows.

Script design: labels are read only for the 197 after the set-aside step; CYP constants come from the existing train-partition split file and the PXR constant from the pins; MAE and Spearman are computed in native units for strata all, primary_validation and aux_only; it writes `evaluation_report.json`, `evaluation_metrics.csv`, `evaluation_rows.csv` and `crosswalk_table.csv` into a new directory. The script was subsequently run locally (see results below).

Assumption: the approval of option A was taken as approval of the plan with option A and of writing the script (as in D026 Amendment 1). The D025 Amendment 3 label-values bullet remains unreconciled.

### D026 descriptive evaluation results (2026-11-25; source: owner-pasted output of a read-only command on the local machine reading `analysis/cyp_010_screen_eval_20261125/report/`)

`evaluate_screen_predictions.py` (SHA-256 5eaafb5f3338737d48422902bf01c55ecb75bc54b9d608735fbec4e9f73a1dcb) ran locally on the 208-row predictions (SHA-256 0a04fe735ea5d55e2fbd0ad72ab62e7f696d137f2865cec1213211a48eeb5d1f) and the builder mapping. Its own assertions on hashes, crosswalk and recorded counts passed (inferred from completion).

| Report file | SHA-256 |
|---|---|
| `evaluation_report.json` | 2097f1445b1d417bbe2211085348597520d0acb7460051ed1adf411089127a65 |
| `evaluation_metrics.csv` | 27b00b3a925212b25a8fb07fb740c535acc909ef9b1b15aeadd510f5d2960156 |
| `evaluation_rows.csv` | 919f86799938dbe8d58491b355db8e2936e1a57bdcaca40b1431e68811c7945a |
| `crosswalk_table.csv` | 8cc613a22fbe84b2200d2e0515123b43075843ca33d342e95e8f39e48667303d |

CYP tier by PXR role after setting aside the 11: train/aux_only 32, train/primary_validation 137, validation/aux_only 5, validation/primary_validation 23 (equal to the recorded counts). Label counts n: 197 for every output in the all stratum, 160 primary_validation, 37 aux_only.

Baseline constants: CYP unweighted native-log2fc means over the 6,252 screening-training molecules (CYP1A2 -0.1932, CYP2C9 -0.3715, CYP2D6 +0.3457, CYP3A4 -0.9739); PXR the recorded stage-1 training-label mean 0.301972839854651 (reported as 0.3020).

Metrics in native log2fc. Ratio is MAE divided by baseline MAE. Spearman is of the prediction against the label; the baseline has none.

| Stratum | Output | n | MAE | Spearman | Baseline MAE | Ratio |
|---|---|---|---|---|---|---|
| all | CYP1A2 | 197 | 0.5090 | 0.5675 | 0.6334 | 0.8036 |
| all | CYP2C9 | 197 | 0.2829 | 0.6814 | 0.5142 | 0.5501 |
| all | CYP2D6 | 197 | 0.7448 | 0.4777 | 0.7943 | 0.9376 |
| all | CYP3A4 | 197 | 0.9028 | 0.7008 | 1.5425 | 0.5853 |
| all | PXR 8.25 uM three-seed mean | 197 | 0.3568 | 0.6405 | 0.5646 | 0.6318 |
| primary_validation | CYP1A2 | 160 | 0.5152 | 0.4971 | 0.6454 | 0.7982 |
| primary_validation | CYP2C9 | 160 | 0.2908 | 0.5857 | 0.5438 | 0.5348 |
| primary_validation | CYP2D6 | 160 | 0.7762 | 0.4287 | 0.8270 | 0.9385 |
| primary_validation | CYP3A4 | 160 | 0.9668 | 0.6114 | 1.6511 | 0.5855 |
| primary_validation | PXR 8.25 uM three-seed mean | 160 | 0.4001 | 0.4936 | 0.6522 | 0.6134 |
| aux_only | CYP1A2 | 37 | 0.4819 | 0.5955 | 0.5811 | 0.8293 |
| aux_only | CYP2C9 | 37 | 0.2487 | 0.7956 | 0.3863 | 0.6438 |
| aux_only | CYP2D6 | 37 | 0.6089 | 0.6351 | 0.6529 | 0.9326 |
| aux_only | CYP3A4 | 37 | 0.6260 | 0.8525 | 1.0728 | 0.5835 |
| aux_only | PXR 8.25 uM three-seed mean | 37 | 0.1693 | 0.3544 | 0.1858 | 0.9113 |

Descriptive reading (corrected 2026-11-25; not a decision): over all 197 molecules CYP2C9 has the lowest MAE/baseline ratio (0.550), ahead of CYP3A4 (0.585) and PXR (0.632); CYP3A4 has the highest overall Spearman (0.701); CYP2D6 has the weakest ratio (0.938) and the weakest Spearman (0.478) among the five outputs. Every output has MAE below its constant baseline and a positive Spearman. Per-stratum figures are as tabulated; in the aux_only stratum PXR has the lowest Spearman (0.354).

Removed, not established: (1) any explanation of the CYP3A4 baseline MAE by the training mean (-0.974), since the baseline MAE depends on the distance of the evaluation labels from the constant and nothing about label spread is inferred from the constant's value; (2) the statement that the positive CYP2D6 training mean is consistent with the head's earlier outputs near +0.2 to +0.35, since no prediction distribution has been inspected; (3) the word "floor" and the statement that a training-only median would give a lower baseline MAE: the training-mean constant is a reference point and not a bound, and a training-only median minimizes training absolute error but is not guaranteed to give a lower or higher MAE on these 197 evaluation molecules; no median baseline was computed.

CYP caveat: 169 of the 197 molecules are CYP training rows and 28 are CYP validation rows (train/primary_validation 137 + train/aux_only 32 = 169; validation/primary_validation 23 + validation/aux_only 5 = 28). The primary_validation and aux_only strata describe PXR roles, not CYP holdout status. The pooled CYP metrics therefore cannot establish held-out CYP performance.

Other limits: descriptive only; molecules share scaffold groups and are not independent; no confidence intervals; not challenge performance, not validation of the full potency-plus-screen report, not evidence of confirmed PXR-specific activity; no threshold chosen.

Unresolved: `evaluation_report.json` lists 6,251 training labels per CYP enzyme (`n_train_labels`) while the records describe 6,252 screening-training molecules; the one-molecule gap is not yet explained.

### CYP-tier-split diagnostic (D026 Amendment 2; run locally, reported 2026-11-26; source: output of `cyp/scripts/evaluate_screen_cyp_tier_split.py` pasted by the project owner)

`cyp/scripts/evaluate_screen_cyp_tier_split.py` and `cyp/tests/test_evaluate_screen_cyp_tier_split.py` written locally (uncommitted). Unit tests run locally 2026-11-26 by the project owner with the tripwire_cyp002 interpreter: all passed (individual output and count not pasted). SHA-256 of the files as they now exist, computed 2026-11-26 from the working tree after the run (the run-time identity of the script was not hashed at run time): script f916846846b97c32511f453c1dd59e21a933fc80ef990c9cdf30766b6c8e43f7; tests 96dfa23b6d666497bdf00089ffcdd77f41a3ded2178a0037b7f515132d023a04. Inputs, hash-checked by the script: `evaluation_rows.csv` 919f86799938dbe8d58491b355db8e2936e1a57bdcaca40b1431e68811c7945a and `evaluation_metrics.csv` 27b00b3a925212b25a8fb07fb740c535acc909ef9b1b15aeadd510f5d2960156 (equal to the hashes recorded above). The pooled all-stratum consistency check passed (inferred from completion).

| Output file (`analysis/cyp_010_screen_eval_20261125/cyp_tier_diagnostic/`) | SHA-256 |
|---|---|
| `cyp_tier_split.csv` | 80a2f11e4bacc5c49b62446bf788a04380f678b819250fed4b265837f423f2c5 |
| `cyp_tier_split_report.json` | e6952e1ea2f827f10dad9cec80ac307caa14897ff33e8e837cb3fd63010606e6 |

Tier counts: train 169, validation 28. Native log2fc; ratio is MAE over in-tier baseline MAE; mean residual is prediction minus label.

| Output | Tier | n | MAE | Spearman | Baseline const | Baseline MAE | Ratio | Mean residual |
|---|---|---|---|---|---|---|---|---|
| CYP1A2 | train | 169 | 0.4931 | 0.5947 | -0.1932 | 0.6206 | 0.7945 | +0.3181 |
| CYP1A2 | validation | 28 | 0.6046 | 0.3892 | -0.1932 | 0.7101 | 0.8514 | +0.3801 |
| CYP2C9 | train | 169 | 0.2821 | 0.6971 | -0.3715 | 0.5090 | 0.5543 | +0.0881 |
| CYP2C9 | validation | 28 | 0.2875 | 0.5977 | -0.3715 | 0.5457 | 0.5267 | +0.0915 |
| CYP2D6 | train | 169 | 0.7480 | 0.5100 | +0.3457 | 0.7977 | 0.9377 | +0.5843 |
| CYP2D6 | validation | 28 | 0.7252 | 0.2742 | +0.3457 | 0.7737 | 0.9374 | +0.5508 |
| CYP3A4 | train | 169 | 0.8889 | 0.7177 | -0.9739 | 1.5409 | 0.5769 | +0.6432 |
| CYP3A4 | validation | 28 | 0.9866 | 0.5807 | -0.9739 | 1.5521 | 0.6356 | +0.4970 |

Descriptive reading and limits are as in section 11 of the session report (no cause explained; n 28, no confidence intervals; residual is aggregate offset only; no new baseline, bootstrap, threshold or inference).

Not seen: the pytest result for the evaluation tests and the output of the fetch and verify steps (the script verified its input hashes at run time).

## CoreHPC smoke results (2026-11-24; source: output pasted back from the CoreHPC login host `chpc-ucsf-login-vm1.corehpc.ucsf.edu`)

Run from `/scratch/user/mhuang8/tripwire_staging/screen_heads_smoke_20261124/`, CPU only, 2 threads; script and pins copies hash-verified before the run. No labels used; nothing scored.

### CYP

- Script SHA-256 55639bb7000b620b58b3b829b6ab4a80db9ac4ee1f146b9d23d7e4e107ad55f5; pins SHA-256 0e5160782f3289924d77acaa78d33798e3916a30583bb45d73cd4405d2e147f9.
- Status PASS; `result.json` SHA-256 5c7569b09f5786ba9d5e177f0b87ce3401ea961aec1679108863c7c0dd7736a6; 13 of 13 checks true:
  1. pins hash;
  2. 17 pinned sources and inputs hashed (16 manifest roles plus `actfound_model_side`);
  3. 5 run files hashed (checkpoint, `preparation.json`, selected record, `selection.json`, split assignments);
  4. selection records consistent (epoch 16, score 0.2029570272781568);
  5. split counts train 6252, validation 695;
  6. preparation matches split;
  7. imported `t121_weighted_training` is the pinned file (resolved path `/mnt/scratch/user/mhuang8/tripwire_staging/t121_production_chain_20261117/cyp/scripts/t121_weighted_training.py`; `/scratch` resolves under `/mnt/scratch`);
  8. task names CYP1A2, CYP2C9, CYP2D6, CYP3A4 (CYP3A4 index 3);
  9. dataset population 6947 equals the split;
  10. head is Identity output transform, final projection `ffn.1.2`, 4 outputs;
  11. strict `state_dict` load succeeded (11 keys; checkpoint epoch field 16; global_step 6647, which equals 17 epochs x 391 batches of size 16, arithmetic by the assistant);
  12. forward shape [16, 4], finite, within the sanity bound;
  13. two eval-mode forwards identical.
- Molecules: 8 train plus 8 validation by sorted id (OCNT-0001883 ... OCNT-0494072; full list in `result.json`).
- Native log2fc prediction column min/max on those 16 molecules: CYP1A2 -0.206 to 0.093; CYP2C9 -0.247 to -0.041; CYP2D6 0.228 to 0.359; CYP3A4 -0.264 to -0.026. Descriptive only; not representative and not comparable with the development-compound label ranges in `notes/Tripwire_CYP_009.md` section 12.
- Login-node timing, not a benchmark: build dataset and model 1.27 s, two forwards 0.136 s, total 9.46 s.
- Environment: chemprop 2.3.1, torch 2.14.1+cu126, python 3.12.14.

### PXR stage-1

- Script SHA-256 65ebd11f62317156d081a73f89a3ac94caccfdb31304ba06518202b8d086c833; pins SHA-256 144695415cb940d52f47717d26d7eae07a4297850b7a0f838886398c58d7257d.
- Status PASS for seeds 20261125, 20261126, 20261127.
- Run by seed 20261125, 20261126 and 20261127 on the login host, CPU only, 2 threads; each reported status PASS, meaning every recorded check in that run was true (hash checks, strict load, scaler three-way agreement, eval-mode unscale identity, output sanity). The stage-2 potency checkpoint was not opened by design.
- Result files in `/scratch/user/mhuang8/tripwire_staging/screen_heads_smoke_20261124/results/` (SHA-256 as listed by `sha256sum` on CoreHPC):

| File | SHA-256 |
|---|---|
| `cyp_r1/result.json` | 5c7569b09f5786ba9d5e177f0b87ce3401ea961aec1679108863c7c0dd7736a6 |
| `pxr_seed20261125/result.json` | 17a0364e41860ef72401c953753c00e9dd79de1e4d52f9480be8658685cb0d05 |
| `pxr_seed20261126/result.json` | 9988a5e20ec4a3a56eaeebd6c315d09961e17c332ba4aa1beb97673cbe27c16a |
| `pxr_seed20261127/result.json` | 50afaf12bb337b395f44e9254086c44f4821917b6148ee9c9be3313016691a7d |


Per-check detail (source: pasted output of a read-only command on the CoreHPC login host reading the three `result.json` files, 2026-11-24). All three seeds: PASS, 13 of 13 checks true:
1. pins hash;
2. staged package, 16 files hashed;
3. stage-1 checkpoint and records: best epochs 2, 4, 20; checkpoint hashes 56ba8157..., 15eabba6..., 6fabe68b... equal the pinned values (full values in the PXR-004 stage-1 table above);
4. imported `pxr004_data` and `pxr004_train` are the staged files (resolved under `/mnt/scratch`);
5. head order `log2fc_8p25uM`, `log2fc_33uM`, `log2fc_99uM`; index 0 is the 8.25 uM head;
6. stage-1 cohort 8,862 training and 985 monitor molecules;
7. strict `state_dict` load, 13 keys; global_step 417, 695, 2919 for the three seeds;
8. encoder differs from the foundation weights;
9. scaler three-way agreement: recomputed vs stage-1 manifest max difference 0.0; recomputed vs checkpoint buffer 4.43e-08 (mean) and 2.97e-08 (scale), identical across seeds, float32 rounding level;
10. scaler equals the pinned values;
11. eval-mode output equals standardized output times scale plus mean; max absolute difference 8.74e-08, 8.31e-08, 1.21e-07 for the three seeds (tolerance 1e-5);
12. output shape [16, 3], finite, within the sanity bound;
13. repeated eval forwards identical.

- Login-node CPU timing, not a benchmark: three forward passes about 0.93 to 0.98 s; total about 8.2 to 8.9 s per seed.
- Assistant arithmetic: global_step equals (best epoch + 1) x 139, where 139 is the stage-1 batches per epoch (8,862 molecules at batch size 64, ceiling of 138.47, matching the 139 train batches in the PXR-004 timing record): 3 x 139 = 417, 5 x 139 = 695, 21 x 139 = 2919. This indicates each stage-1 `best.ckpt` was saved at its recorded best epoch.
- Native predictions on the 16 monitor molecules with the lowest sorted row ids (descriptive only; no labels used; nothing scored):

| Seed | 8.25 uM (index 0) range | 8.25 uM mean |
|---|---|---|
| 20261125 | -0.0045 to 1.2264 | 0.7324 |
| 20261126 | -0.0392 to 1.2798 | 0.8172 |
| 20261127 | -0.0402 to 1.2556 | 0.8221 |

  The 33 uM and 99 uM columns ranged 0.003 to 2.0007 and 0.472 to 1.88, with means 1.11 to 1.18 and 1.44 to 1.58.
- Seed-to-seed agreement on those 16 molecules (assistant arithmetic from the pasted values), for the three seed pairs: Spearman 0.844, 0.882, 0.859; Pearson 0.930, 0.936, 0.947; mean per-molecule range across seeds 0.172, maximum 0.425.
- Caveat: the 16 molecules are the first by sorted row id in the monitor set. The row-id prefix ordering suggests they are primary-train (DRC) molecules, which are hit-enriched (about 95% FDR below 0.05 at 8.25 uM in the monitor, median label about 0.8), so their predictions are not representative of the screen-only population. This is an inference, to be confirmed from the molecule list in `result.json`. Sixteen molecules support no statement about discrimination or about the three-seed-mean decision.
