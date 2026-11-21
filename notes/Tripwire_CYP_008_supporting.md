# Tripwire-CYP-008 Supporting Record

Supports `notes/Tripwire_CYP_008.md`. Identifiers and hashes only; where a full hash was not supplied to the bookkeeper, the abbreviated form is recorded and the full value is UNKNOWN in this record.

## Package and code

| Item | Value |
|---|---|
| Descriptor package | `t121_d021_package_20261119`, tarball SHA-256 bd73e8c3a6f1257e24587eb78a8102a03e3eb1b534ebe68b5b6e2005601a0978 |
| Reference production-chain tarball (pinned) | 9ecb6ccf... (full hash not supplied) |
| `cyp/scripts/t121_d021_aggregate.py` | 55ade624c4cd67a9c4fbc03430e29274e179f20f44517851457a9a2037d42c39 |
| Other D021 code (layout, build_package, descriptor_cache, screen_chain_run, production_cell, cells.slurm) | local, uncommitted; hashes not supplied |

## Qualification (arm D smoke)

Job 3155222, PASS, COMPLETED 0:0, 32 s, ggpu1-09; descriptor_dim 196; predictor widths [[2000, 796]]; `result.json` SHA-256 dad22845c87c8486df2dd5c0c59252d842cfbb8fab29dbf4d5cb7925391df1b2; `check.py` 09abee9d... (full hash not supplied).

## Timing jobs

| Job | Outcome |
|---|---|
| 3158166 | Failed: timing.py left SCREEN_EPOCHS=3; trial warmup contradicted the 50-epoch schedule |
| 3158201 | Rerun after restoring 50; L40S, ggpu1-07, 2:56; 4.90 s/epoch screening, cell 0 147.6 s |

## Screening runs

| Job | Run | Seed | Epoch | Record SHA-256 (abbreviated) |
|---|---|---|---|---|
| 3161795-3161799 (per-run job mapping not supplied) | r_s2 | 20261102 | 31 | 0bab1c8a...eeb3 |
| | r_s3 | 20261103 | 29 | c7c71844...513f |
| | d_s1 | 20261101 | 29 | a03d270c...d5bc |
| | d_s2 | 20261102 | 46 | ce1666a8...c40b |
| | d_s3 | 20261103 | 28 | 9a415bd1...fd68 |

## Cell arrays

| Run | Array job | COMPLETED.json | INCOMPLETE.json |
|---|---|---|---|
| r_s2 | 3161888 | 150 | 0 |
| r_s3 | 3161889 | 150 | 0 |
| d_s1 | 3161890 | 150 | 0 |
| d_s2 | 3161892 | 150 | 0 |
| d_s3 | 3161893 | 150 | 0 |

All submitted with `--time=00:30:00`; every task COMPLETED 0:0.

## Aggregation

Tests: `cyp/tests/test_t121_d021_aggregate.py`, 11 passed locally. Report on CoreHPC: `/scratch/user/mhuang8/tripwire_staging/t121_d021_o150_report_20261121.json`; hash not computed. Source of the result numbers: as reported to the bookkeeper by the main conversation on 2026-11-21 from that report.
