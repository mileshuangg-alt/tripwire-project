"""Tripwire PXR-002 tuning grid and selection (D018). Authorizes no run.

Stage A: the full 36-configuration grid, one common seed.
Stage B: the three best configurations by primary-head validation MAE, compared
across three seeds (the common seed counts as the first), ranked by mean MAE.
The seed values are a proposal following the project's date-shaped convention.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts.pxr002_train import PXR002Config  # noqa: E402

HIDDEN_SIZES = (300, 600)
DEPTHS = (3, 4, 5)
FFN_LAYERS = (1, 2)
DROPOUTS = (0.0, 0.1, 0.2)

COMMON_SEED = 20261117
STAGE_B_SEEDS = (20261117, 20261118, 20261119)
TOP_K = 3


def build_grid() -> list[PXR002Config]:
    grid = [PXR002Config(h, d, f, p) for h, d, f, p in itertools.product(HIDDEN_SIZES, DEPTHS, FFN_LAYERS, DROPOUTS)]
    if len(grid) != 36 or len({c.name for c in grid}) != 36:
        raise AssertionError("grid must be 36 distinct configurations")
    return grid


def rank_by_mae(results: dict[str, list[float]]) -> list[tuple[str, float]]:
    """Mean primary-head validation MAE per configuration, best first; ties by name."""
    if any(len(v) == 0 for v in results.values()):
        raise ValueError("every configuration needs at least one MAE")
    return sorted(((n, sum(v) / len(v)) for n, v in results.items()), key=lambda t: (t[1], t[0]))


def select_top(stage_a: dict[str, float], k: int = TOP_K) -> list[str]:
    return [name for name, _ in rank_by_mae({n: [m] for n, m in stage_a.items()})[:k]]


def stage_b_plan(top: list[str], done_seed: int = COMMON_SEED) -> list[dict]:
    """Extra runs needed: seeds other than the already-completed common seed."""
    return [{"configuration": n, "seed": s} for n in top for s in STAGE_B_SEEDS if s != done_seed]


def read_stage_a(root: Path) -> dict[str, float]:
    out = {}
    for path in sorted(Path(root).glob("*/metrics.json")):
        m = json.loads(path.read_text())
        if m["seed"] == COMMON_SEED:
            out[m["configuration"]] = m["best_primary_validation_mae"]
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "list":
        for c in build_grid():
            print(json.dumps({"name": c.name, "hidden_size": c.hidden_size, "depth": c.depth,
                              "ffn_layers": c.ffn_layers, "dropout": c.dropout, "seed": COMMON_SEED}))
    elif cmd == "stage-b" and len(sys.argv) == 3:
        stage_a = read_stage_a(Path(sys.argv[2]))
        if len(stage_a) != 36:
            raise SystemExit(f"expected 36 stage-A results, found {len(stage_a)}")
        top = select_top(stage_a)
        print(json.dumps({"top": top, "extra_runs": stage_b_plan(top)}, indent=2))
    else:
        raise SystemExit("usage: pxr002_grid.py list | stage-b <stage_a_results_root>")


if __name__ == "__main__":
    main()
