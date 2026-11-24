# Tripwire-CYP-009 Supporting Record

Supports `notes/Tripwire_CYP_009.md`. Identifiers and hashes only.

## Package and config

| Item | Value |
|---|---|
| D022 package `t121_d022_package_20261121.tar.gz` (v2) | SHA-256 781ba3f327e53fd7f624bfe529b0c133da0a95d66b6c85331a1d962ea814900f; 32,857,437 bytes; 113 files; 22 pinned roles |
| Superseded v1 package tarball | SHA-256 4202a97ae6248f85bbea9aea060b6dc98df679a391e1d287a36a8879a4befb39 |
| CheMeleon weights `chemeleon_mp.pt` | c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651 |
| `cyp/artifacts/d022_chemeleon/chemeleon_config.json` | 444d37bc3f2501237e5734893a48522bd60b93b8cbed5ece3a3324830eb67e4b |
| CoreHPC deployment | `/scratch/user/mhuang8/tripwire_staging/t121_d022_package_20261121` |

## Smoke jobs

| Job | Outcome |
|---|---|
| 3191973 | v1 FAILED 1:0, 32 s; check.py head-width filter defect; no result.json |
| 3202933 | v2 passed; gpu, ggpu1-08, COMPLETED 0:0, 00:00:42 |
| 3203300 | approved one-off batch timing; gpu, ggpu1-11, NVIDIA L40S, COMPLETED 0:0, 00:02:49; output `/scratch/user/mhuang8/tripwire_staging/t121_d022_timing_20261121`; package tarball SHA-256 781ba3f3...814900f; `timing.json` hash not computed |

## Screening runs (source: pasted CoreHPC output; sacct state lines not in the excerpt)

| Job (assumed submission order, confirmed by completion seeds) | Run | Seed | Selected epoch (zero-based) | Manifest SHA-256 | selected_record_sha256 |
|---|---|---|---|---|---|
| 3203564 | c_s1 | 20261101 | 4 | 3503dccf900c8537961eca2e2d5104da689868b2d6707189e12ee34f832ae734 | 330c4706a31bfc86c4326ecb49966e36eca261c3fb82454350d82cca7399825a |
| 3203565 | c_s2 | 20261102 | 45 | 2e9e47181392e052bfcaa9a6b575dc287ef98bd293964a3ca1f639223cb22e30 | 1242c5772ca7844a3ccb5a0fa84c22488aec746fe9d897c4adf4c2808c0dadc9 |
| 3203566 | c_s3 | 20261103 | 4 | a57f4c1f9c1a4a313e8f4475796576075bf696cb2a26ab9890f76b756922a6f3 | d59633bc694c7f360aae8f74bdb38f7861e125c5c5223703904157ed1aa6d84e |

Manifests: `t121_d022_manifests/screen_manifest_t121_d022_c_s{1,2,3}.json`, checked with `sha256sum -c` before submission. Each job `--time=00:30:00`. Completion: status completed, epochs_recorded 50, full_screen_refit false. Run roots `/scratch/user/mhuang8/tripwire_staging/t121_d022_c_s{1,2,3}_screen_run`.

## Cell arrays and O150 aggregation (2026-11-23; source: pasted CoreHPC stdout)

| Job (assumed submission order) | Run | Roots under `/scratch/user/mhuang8/tripwire_staging/` | Outcome |
|---|---|---|---|
| 3207090 | c_s1 | `t121_d022_c_s1_cells_run`, `t121_d022_c_s1_cells_logs` | 150/150 completed, 0 incomplete; `--time=00:30:00` per task |
| 3207091 | c_s2 | `t121_d022_c_s2_cells_run`, `t121_d022_c_s2_cells_logs` | same |
| 3207092 | c_s3 | `t121_d022_c_s3_cells_run`, `t121_d022_c_s3_cells_logs` | same |

Mean elapsed of the first array's tasks 139.867 s (sacct as pasted).

Scripts (verified by `sha256sum -c` on CoreHPC, both OK): `t121_d022_aggregate.py` 7a2be5f7bfcf0c64103ef26f5838f0e03bd009a923fce70b57d622fc578d34c9; `t121_d021_aggregate.py` 55ade624c4cd67a9c4fbc03430e29274e179f20f44517851457a9a2037d42c39.

Report: `/scratch/user/mhuang8/tripwire_staging/t121_d022_o150_report_20261123.json`, SHA-256 3f70eb45a630548fff3b24460b201d9cd546c9e9245950c06f39dc4637a288ab.

| Run | Roots (each under `trials/trial_121/cells`) | O150 | Screening checkpoint sha (prefix) |
|---|---|---|---|
| r1 | `t121_production_chain_20261117_cells_run` | 0.676954037 | 96b75ceb4ddd6293 |
| r2 | `t121_d021_r_s2_cells_run` | 0.668499342 | 6d8bdf79018e352e |
| r3 | `t121_d021_r_s3_cells_run` | 0.669626167 | a3bb987338d65fe2 |
| c1 | `t121_d022_c_s1_cells_run` | 0.706368862 | 4116a463f8675b02 |
| c2 | `t121_d022_c_s2_cells_run` | 0.693407594 | 5121c5674960dcb9 |
| c3 | `t121_d022_c_s3_cells_run` | 0.710411351 | 9b2e7e7d7d8fccad |

Aggregator reproduction check: r1 matches the recorded 0.6769540372619428 (the aggregator stops otherwise; the report was written).

## D023 frozen-artifact set (2026-11-23)

Frozen by D023; identifiers as stated in the decision: report 3f70eb45a630548fff3b24460b201d9cd546c9e9245950c06f39dc4637a288ab; package tarball 781ba3f327e53fd7f624bfe529b0c133da0a95d66b6c85331a1d962ea814900f; smoke `result.json` 62c034709ccabe284c69eaa60b7eb41cf2b93a402f23ed01332b685ec6af4ed6; config 444d37bc3f2501237e5734893a48522bd60b93b8cbed5ece3a3324830eb67e4b; screening selected-record hashes and manifest hashes are in the screening table above.

## Reopened error diagnostic (2026-11-24; session log section 12)

- Script: `cyp/scripts/t121_error_diagnostic.py`, SHA-256 fdc23a4ba0e2df796d4d98c0b5ba1878f970024b1317cfa0f31cb296060ba214 (uncommitted). Written under a bounded approval (implement and `--selftest`; no execution by the assistant).
- Inputs (read only): `t121_valinf_full_20261118/full_cells_000_149` (150 saved replicate-1 validation-prediction CSVs; optional `full_cells_000_149_SHA256SUMS.txt`), `cyp/artifacts/cyp_baseline/cluster_assignments.csv`, `cyp/artifacts/cyp_007/split_manifests`, reproduction JSONs.
- Output: `analysis/cyp_009_error_diagnostic_20261124/full_150/` (mode full, 150 cells); output-file hashes not computed.
- Environment `tripwire_cyp002`: pandas 3.0.6, numpy 2.5.3.
- Run outputs: the full run's console output was reported; the self-test and three-cell smoke were run by the user, with output not reported.
- Checks: range-check anomaly flags 0 of 600; recomputed vs recorded cell ST-RAE max abs diff 2.858e-09 over 600 comparisons.
- Scorer scaling-heuristic branch (`cyp002_validation.py` lines 195-218): not recorded in any saved artifact; keep-raw branch is an inference from code-path analysis, not a record.

## Reopened screening-versus-error analysis (2026-11-24; session log section 12, follow-on)

- Script: `cyp/scripts/t121_screening_error_analysis.py`, SHA-256 0760bbfde003a027fc4b370ba073a884a410a597975481a3bf41319a04d6035f (uncommitted; full run reported 2026-11-24). Reuses `cyp/scripts/t121_error_diagnostic.py`, SHA-256 fdc23a4ba0e2df796d4d98c0b5ba1878f970024b1317cfa0f31cb296060ba214 (unchanged).
- Screening source: `actfound_screen_loader_20261117.tar.gz` (read in place), members `inputs/screen_full.csv` (SHA-256 4f61be8ffdec767a61981ab89e3f06d28850603c3fa4ad0715a2183d4261a706, matches contract), `inputs/adjudication.tsv` (identity file SHA-256 2f3045124dbf138ec535abfd8b697a6e2e2cd605d2bccd9f30a58f4ba5464f54, matches contract), `qualification/actfound/screen_loader_20261117/contract.json`.
- Screening columns: Molecule_Name, SMILES, OCNT_Batch, enzyme, plate_id, concentration_M, log2fc_estimate, log2fc_std_error, p_value, log2fc_fdr, log2fc_median, cohens_d.
- Other inputs (read only): `t121_valinf_full_20261118/full_cells_000_149`, cluster assignments, split manifests, training CSV (SMILES check).
- External web sources (fetched 2026-11-24; not repository facts): https://huggingface.co/datasets/openadmet/cyp-challenge-train-test (README); https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/.
- Output directory: `analysis/cyp_009_screening_error_20261124/full_150/` (mode full, 150 cells): screening_join_audit.csv, molecule_screening_error.csv, prediction_rows_with_screening.csv, relationship_summary.csv, log2fc_quintile_summary.csv, prediction_strata_summary.csv, overprediction_tail_comparison.csv, partition_relationship.csv, run_metadata.json. Output-file hashes not computed. Source of figures: console output pasted from the full run plus these tables read locally; self-test and smoke output not reported.
- Join audit: CYP1A2 1,201 of 1,201 and CYP2D6 1,269 of 1,269 ok; 0 missing, duplicate, identity-unverified or missing-value rows; one concentration value (4.95049505e-05 M); no warnings.

| Quantity | CYP1A2 | CYP2D6 |
|---|---|---|
| log2fc range (labelled) | -2.905 to +2.703 | -3.273 to -0.592 |
| Spearman log2fc vs measured pIC50 (per-partition range) | -0.871 (-0.936..-0.797) | -0.825 (-0.881..-0.771) |
| Spearman log2fc vs predicted pIC50 | -0.486 | -0.396 |
| Spearman log2fc vs signed error (Pearson) | +0.758 (+0.808) | +0.688 (+0.615) |
| Rank partial corr. with signed error given predicted | +0.811 | +0.748 |
| Within predicted-pIC50 quintiles, Spearman range | +0.594..+0.927 | +0.711..+0.812 |
| Per-partition Spearman with signed error (50 partitions) | +0.624..+0.832 | +0.528..+0.772 |
| Spearman log2fc vs absolute error (per-partition range) | +0.326 (+0.121..+0.514) | -0.086 (-0.274..+0.138) |
| MAE by log2fc quintile (strongest to weakest) | 0.622, 0.396, 0.320, 0.432, 1.451 | 0.757, 0.381, 0.432, 0.603, 0.707 |
| Mean signed error by log2fc quintile | -0.519, -0.238, +0.033, +0.257, +1.438 | -0.684, -0.173, +0.192, +0.548, +0.690 |
| Share overpredicted by quintile | 16.2, 30.8, 55.4, 75.8, 98.8% | 11.8, 30.7, 59.3, 84.3, 93.3% |
| Weakest vs strongest quintile mean measured / predicted | 3.35 / 4.79 (weakest); 5.90 / 5.38 (strongest) | 4.05 / 4.74 (weakest); 5.81 / 5.13 (strongest) |
| Top-10% signed-error tail n (tail vs rest) | 120 vs 1,081 | 127 vs 1,142 |
| Tail vs rest median log2fc | -0.195 vs -1.601 | -1.068 vs -1.892 |
| Rank AUC, top-10% tail / signed error > +1.0 (n) | 0.980 / 0.976 (183) | 0.787 / 0.788 (106) |

Quintiles are equal-count (about 240 molecules per bin), exploratory.

Pointer (2026-11-24): the three-arm penalty comparison definitions are recorded as D024 in `notes/Decisions.md` (design definitions only; no run authorized). The tables above are motivating context only; no new hashes or evidence were generated for D024.

Pointer (2026-11-24): lambda, adoption rule, replicate scope, lambda = 0 checks, condition key and per-enzyme tails are recorded as Amendment 1 to D024 in `notes/Decisions.md`; no new hashes or evidence were generated.

## Smoke result

Smoke v2 `result.json` SHA-256 62c034709ccabe284c69eaa60b7eb41cf2b93a402f23ed01332b685ec6af4ed6 (CoreHPC `results/d022_chemeleon_smoke/result.json`).

Fields from the pasted readout:

| Field | SHA-256 |
|---|---|
| check_py | 723017572535f2f6612d29c0c82529d4342a76f0c36ab2aaa0a964ec154ff090 |
| screen_stage | b9b3c0133342f21a1d3fbd546e369ff2ceb226f587efc56c8de980242f46a5d1 |
| weighted_training | 50414196e14dd1846480749d328b7fcb9c680ca08cfed4f7cf47f4b6fa67c49e |
| screen_chain_run | 3a2d38346c62c4b3a3bce492598a690b0b6a39d6edb0c42f0b937e5fa5d89b56 |
| split_manifest | 16aa90c786039b3f6f3ffeb071035742ca1641dbc9232c73eaa780a7f92e18df |
