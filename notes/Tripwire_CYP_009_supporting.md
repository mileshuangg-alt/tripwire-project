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
