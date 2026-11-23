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

Smoke v2 `result.json` SHA-256 62c034709ccabe284c69eaa60b7eb41cf2b93a402f23ed01332b685ec6af4ed6 (CoreHPC `results/d022_chemeleon_smoke/result.json`).

Fields from the pasted readout:

| Field | SHA-256 |
|---|---|
| check_py | 723017572535f2f6612d29c0c82529d4342a76f0c36ab2aaa0a964ec154ff090 |
| screen_stage | b9b3c0133342f21a1d3fbd546e369ff2ceb226f587efc56c8de980242f46a5d1 |
| weighted_training | 50414196e14dd1846480749d328b7fcb9c680ca08cfed4f7cf47f4b6fa67c49e |
| screen_chain_run | 3a2d38346c62c4b3a3bce492598a690b0b6a39d6edb0c42f0b937e5fa5d89b56 |
| split_manifest | 16aa90c786039b3f6f3ffeb071035742ca1641dbc9232c73eaa780a7f92e18df |
