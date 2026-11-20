# Tripwire-PXR-004 Supporting Record

Supports `notes/Tripwire_PXR_004.md`. Evidence for section 10 lives on CoreHPC under `/scratch/user/mhuang8/pxr004_pilot_20261119/seed<SEED>/` (stage2/metrics.json; sequential stage1/metrics.json), transcribed from a pasted login-host output on 2026-11-19. Not reviewed locally.

## Stage 2 validation, full precision (seeds 20261125 / 20261126 / 20261127)

| Metric | Direct | Sequential |
|---|---|---|
| MAE | 0.5212885567531493 / 0.5302900705552523 / 0.525830623416317 | 0.5213383854683279 / 0.5303964313485581 / 0.5321343104428523 |
| RMSE | 0.7194605867805192 / 0.7274881871571496 / 0.7337074205466382 | 0.717918443590824 / 0.7278396387934111 / 0.7221536976347064 |
| Spearman | 0.7440156201911423 / 0.7371245844071725 / 0.7547241439327474 | 0.74289639028694 / 0.7341557138404134 / 0.7337132782906665 |

