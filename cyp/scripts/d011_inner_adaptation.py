from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from p2_anil_training import D011Candidate, T121_O150


D010_ANCHOR_O150 = 0.7106082085193788
D011_INNER_LRS = (0.001, 0.003, 0.01, 0.03)
D011_INNER_STEPS = (1, 2, 3)
D011_CELLS_PER_CONFIG = 150


@dataclass(frozen=True)
class D011CandidateResult:
    candidate: D011Candidate
    o150: float
    source: str


def d011_candidate_grid() -> tuple[D011Candidate, ...]:
    return tuple(D011Candidate(lr, steps) for lr in D011_INNER_LRS for steps in D011_INNER_STEPS)


def d010_anchor_candidate() -> D011Candidate:
    return D011Candidate(0.01, 1)


def d011_new_candidate_grid() -> tuple[D011Candidate, ...]:
    anchor = d010_anchor_candidate()
    return tuple(candidate for candidate in d011_candidate_grid() if candidate != anchor)


def d011_anchor_compatibility_record() -> dict[str, object]:
    return {
        "compatible": True,
        "anchor_candidate": asdict(d010_anchor_candidate()),
        "anchor_o150": D010_ANCHOR_O150,
        "evidence": {
            "architecture_model_construction": "D011 default candidate uses D010 build_p2_message_passing/build_p2_aggregation/build_p2_meta_head.",
            "data_partitions_and_cell_identities": "D011 reuses D007 150-cell routing from D010/P2 production utilities.",
            "episode_construction": "D011 default candidate uses D010 13 support / 3 query, four task episodes per update.",
            "losses_and_task_weighting": "D011 default candidate uses D010 standardized support/query MSE and exact 1/4 query weighting.",
            "outer_optimizer_and_schedule": "D011 default candidate uses D010 Adam and NoamLR constants.",
            "dropout_policy": "D011 default candidate uses D010 train/eval mode policy.",
            "validation_adaptation": "D011 default candidate uses inner_lr=0.01 and inner_steps=1, identical to D010.",
            "checkpoint_selection_and_scoring": "D011 default candidate uses macro ST-RAE strict-best semantics inherited from D010.",
            "firewall": "D011 default candidate inherits D010 training/validation/protected-test/D008 boundaries.",
        },
    }


def d011_workload_plan(anchor_reusable: bool = True) -> dict[str, object]:
    grid = d011_candidate_grid()
    new_candidates = d011_new_candidate_grid() if anchor_reusable else grid
    return {
        "configuration_count": len(grid),
        "anchor_reused": bool(anchor_reusable),
        "new_candidate_count": len(new_candidates),
        "cells_per_new_candidate": D011_CELLS_PER_CONFIG,
        "new_cell_count": len(new_candidates) * D011_CELLS_PER_CONFIG,
        "new_candidates": [asdict(candidate) for candidate in new_candidates],
    }


def aggregate_candidate_o150(cell_macro_scores: Sequence[float]) -> float:
    if len(cell_macro_scores) != D011_CELLS_PER_CONFIG:
        raise ValueError("D011 candidate O150 requires exactly 150 cells.")
    values = np.asarray(cell_macro_scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("D011 candidate cell scores must be finite.")
    return float(values.mean())


def select_d011_candidate(results: Iterable[D011CandidateResult]) -> dict[str, object]:
    rows = list(results)
    if len(rows) != 12:
        raise ValueError("D011 selection requires exactly 12 candidate results.")
    if len({row.candidate for row in rows}) != 12:
        raise ValueError("D011 candidate results contain duplicate or missing configurations.")
    selected = min(rows, key=lambda row: (float(row.o150), row.candidate.inner_steps, row.candidate.inner_lr))
    return {
        "selected_candidate": asdict(selected.candidate),
        "selected_o150": float(selected.o150),
        "selected_source": selected.source,
        "t121_o150": T121_O150,
        "advances": bool(float(selected.o150) < T121_O150),
        "selection_rule": "lowest O150; exact tie -> fewer inner_steps; remaining exact tie -> smaller inner_lr",
        "advancement_rule": "selected D011 candidate advances iff O150(selected) < 0.701212576796",
        "endpoint_veto": False,
    }


def results_with_d010_anchor(new_results: Iterable[D011CandidateResult]) -> tuple[D011CandidateResult, ...]:
    rows = list(new_results)
    if len(rows) != 11:
        raise ValueError("D011 anchor reuse expects exactly 11 new candidate results.")
    anchor = D011CandidateResult(
        candidate=d010_anchor_candidate(),
        o150=D010_ANCHOR_O150,
        source="frozen D010 anchor",
    )
    combined = [anchor, *rows]
    if len({row.candidate for row in combined}) != 12:
        raise ValueError("D011 combined results do not cover exactly 12 configurations.")
    return tuple(combined)
