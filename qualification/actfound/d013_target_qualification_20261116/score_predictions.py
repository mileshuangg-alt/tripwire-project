"""Score a D013 adaptation predictions artifact with the frozen ST-RAE scorer.

Second stage of the split Miles approved on 2026-11-16. The frozen scorer
(cyp/scripts/st_rae.py -> references/cyp_upstream/custom_scoring_functions.py)
declares PEP 604 annotations, so it requires Python >= 3.10 and cannot be
imported in the pinned ActFound environment (Python 3.9.18) where adaptation
must run. Adaptation therefore writes predictions, and this script scores them
here, in a >= 3.10 environment, with the scorer unchanged.

It reads a hash-pinned adaptation artifact and writes scores. It performs no
adaptation, selects nothing, and never alters the predictions it is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
SCORED_ARTIFACTS = {"D013_target_adaptation_checks_v1"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def assert_scoring_python(version_info=None) -> dict:
    v = tuple(sys.version_info[:2]) if version_info is None else tuple(version_info[:2])
    if v < MIN_PYTHON:
        raise SystemExit(f"the frozen ST-RAE scorer needs Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}; "
                         f"this is {v[0]}.{v[1]}")
    return {"python": ".".join(str(x) for x in v)}


def load_artifact(path: Path, expected_sha: str) -> dict:
    if not path.is_file() or sha256(path) != expected_sha:
        raise SystemExit(f"predictions artifact missing or SHA-256 mismatch: {path}")
    art = json.loads(path.read_text())
    if art.get("artifact_type") not in SCORED_ARTIFACTS:
        raise SystemExit(f"not a scorable D013 adaptation artifact: {art.get('artifact_type')}")
    return art


def score_cell(cell: dict, metric_fn) -> dict:
    """One ST-RAE per prediction arm, against that cell's own labels and bands.

    The artifact carries JSON lists; the frozen scorer takes arrays, so the
    values are converted without being altered. Every arm's length is validated
    before any scoring, so a malformed artifact fails before the scorer runs.
    """
    import numpy as np

    y = np.asarray(cell["target_y"], dtype=np.float64)
    lo = np.asarray(cell["target_conf_low"], dtype=np.float64)
    hi = np.asarray(cell["target_conf_high"], dtype=np.float64)
    n = len(y)
    if not (len(lo) == len(hi) == n) or n == 0:
        raise SystemExit(f"{cell.get('partition')}: labels and bands disagree or are empty")
    arms = sorted(cell["target_predictions"].items())
    for arm, preds in arms:
        if len(preds) != n:
            raise SystemExit(f"{cell.get('partition')}/{arm}: {len(preds)} predictions for {n} targets")
    return {arm: float(metric_fn(y, np.asarray(preds, dtype=np.float64), lo, hi))
            for arm, preds in arms}


def check_bounds(cell: dict, scores: dict, reference_arm: str) -> list:
    """Each derived bound must hold against the ST-RAE differences just measured."""
    verdicts = []
    pairs = {"offset": "pretraining_scale_corrected"}
    for seed_arm in sorted(k for k in scores if k.startswith("pic50_seed_")):
        if seed_arm != reference_arm:
            pairs["repeat"] = seed_arm
    for name, arm in sorted(pairs.items()):
        bound = cell["strae_derived_bounds"].get(name, {}).get("strae_difference_upper_bound")
        if arm not in scores:
            raise SystemExit(f"{cell.get('partition')}: arm {arm} absent; cannot check the {name} bound")
        observed = abs(scores[arm] - scores[reference_arm])
        verdicts.append({
            "pair": name, "arm": arm, "reference_arm": reference_arm,
            "observed_strae_difference": observed,
            "derived_upper_bound": bound,
            "bound_holds": None if bound is None else bool(observed <= bound + 1e-12),
        })
    return verdicts


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--predictions-sha256", required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)

    env = assert_scoring_python()
    art = load_artifact(a.predictions, a.predictions_sha256)
    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")

    sys.path.insert(0, str(a.repo_root))
    sys.path.insert(0, str(a.repo_root / "cyp" / "scripts"))
    from st_rae import st_rae_metric

    scorer = {
        "st_rae_py_sha256": sha256(a.repo_root / "cyp" / "scripts" / "st_rae.py"),
        "custom_scoring_functions_sha256": sha256(
            a.repo_root / "references" / "cyp_upstream" / "custom_scoring_functions.py"),
    }
    cells = []
    for cell in art["cells"]:
        reference_arm = f"pic50_seed_{min(int(k.rsplit('_', 1)[1]) for k in cell['target_predictions'] if k.startswith('pic50_seed_'))}"
        scores = score_cell(cell, st_rae_metric)
        cells.append({
            "partition": cell["partition"], "endpoint": cell["endpoint"],
            "support_count": cell["support_count"], "target_count": cell["target_count"],
            "strae_by_arm": scores,
            "reference_arm": reference_arm,
            "bound_checks": check_bounds(cell, scores, reference_arm),
        })
    out = {
        "artifact_type": "D013_target_adaptation_strae_v1",
        "scored_artifact": str(a.predictions),
        "scored_artifact_sha256": a.predictions_sha256,
        "scored_artifact_type": art["artifact_type"],
        "endpoint": art["endpoint"],
        "scoring_environment": env,
        "frozen_scorer": scorer,
        "script_sha256": sha256(Path(__file__)),
        "all_bounds_hold": all(v["bound_holds"] is not False for c in cells for v in c["bound_checks"]),
        "cells": cells,
    }
    a.out_dir.mkdir(parents=True)
    path = a.out_dir / "strae_scores.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (a.out_dir / "SHA256SUMS.txt").write_text(f"{sha256(path)}  {path.name}\n")
    print(json.dumps({k: out[k] for k in ("endpoint", "scoring_environment", "frozen_scorer",
                                          "all_bounds_hold", "cells")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
