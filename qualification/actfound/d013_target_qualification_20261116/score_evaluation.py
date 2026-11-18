"""D013 stage 2: score the adaptation predictions and compute O150.

Runs in a Python >= 3.10 environment, because the frozen ST-RAE scorer declares
PEP 604 annotations and cannot be imported in the pinned ActFound environment
where adaptation runs (D013 Amendment 4 item 5). The scorer is unchanged; only
where it runs changes.

Per partition it scores the four endpoints with the frozen per-isoform scorer
and aggregates them with the frozen macro aggregator
(cyp/scripts/cyp002_validation.compute_validation_macro_st_rae), never a
hand-rolled mean. Each partition's macro value is then recorded into its three
seed slots, giving the 150 fit-level values whose arithmetic mean is O150, as
D009 defines it. The seed slots are replicates of one unique computation, not
independent adaptations (D013 Amendment 4 item 1).

The comparison against O150(T121) is reported as NOT final while D013's cited
comparator artifact is absent from the checkout: the corroborating D009 file
carries the same value but a matching number does not establish equivalent
provenance (D013 Amendment 4 item 6).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))   # `python -I` implies -P: the script dir is not added
import qualify_adaptation as qa  # noqa: E402
import run_evaluation as re_  # noqa: E402
import score_predictions as sp  # noqa: E402

PREDICTIONS_ARTIFACT = "D013_cyp_adaptation_predictions_v1"


def load_predictions(path: Path, expected_sha: str) -> dict:
    if not path.is_file() or qa.sha256(path) != expected_sha:
        raise SystemExit(f"predictions artifact missing or SHA-256 mismatch: {path}")
    art = json.loads(path.read_text())
    if art.get("artifact_type") != PREDICTIONS_ARTIFACT:
        raise SystemExit(f"not a {PREDICTIONS_ARTIFACT}: {art.get('artifact_type')}")
    if not art.get("contains_no_scores"):
        raise SystemExit("predictions artifact claims to contain scores; refusing to rescore it")
    return art


def read_comparator(path: Path, expected_sha: str, declared: float) -> dict:
    """The comparator value is read from its artifact, never taken from code."""
    if not path.is_file() or qa.sha256(path) != expected_sha:
        raise SystemExit(f"comparator artifact missing or SHA-256 mismatch: {path}")
    art = json.loads(path.read_text())
    value = art.get("t121_o150")
    if value is None:
        raise SystemExit("comparator artifact has no t121_o150")
    if float(value) != float(declared):
        raise SystemExit(f"comparator artifact gives {value}; the config declares {declared}")
    return {"artifact": str(path), "sha256": expected_sha, "o150_t121": float(value),
            "strict_rule": art.get("strict_advancement_rule")}


def group_by_partition(cells: list, task_order: list) -> dict:
    """Partition -> the four endpoints' labels, predictions and bands."""
    out: dict = {}
    for c in cells:
        part = out.setdefault(c["partition"], {"repetition": c["repetition"], "fold": c["fold"],
                                               "training_seeds": list(c["training_seeds"]),
                                               "endpoints": {}})
        label = c["endpoint_label"]
        if label in part["endpoints"]:
            raise SystemExit(f"{c['partition']}: endpoint {label} appears twice")
        if not (len(c["target_predictions"]) == len(c["target_y"]) ==
                len(c["target_conf_low"]) == len(c["target_conf_high"])):
            raise SystemExit(f"{c['partition']}/{label}: predictions, labels and bands disagree")
        part["endpoints"][label] = c
    for name, part in sorted(out.items()):
        missing = [t for t in task_order if t not in part["endpoints"]]
        if missing:
            raise SystemExit(f"{name}: missing endpoints {missing}; the macro needs all of them")
        if len(part["endpoints"]) != len(task_order):
            raise SystemExit(f"{name}: {len(part['endpoints'])} endpoints; expected {len(task_order)}")
    return out


def score_partition(part: dict, task_order: list, macro_fn) -> dict:
    import numpy as np
    y_true, y_pred, lo, hi = {}, {}, {}, {}
    for label in task_order:
        c = part["endpoints"][label]
        y_true[label] = np.asarray(c["target_y"], dtype=np.float64)
        y_pred[label] = np.asarray(c["target_predictions"], dtype=np.float64)
        lo[label] = np.asarray(c["target_conf_low"], dtype=np.float64)
        hi[label] = np.asarray(c["target_conf_high"], dtype=np.float64)
    got = macro_fn(y_true, y_pred, lo, hi, task_order)
    import math
    for label in task_order:
        v = got["per_task"][label]
        if not math.isfinite(v):
            raise SystemExit(
                f"non-finite ST-RAE for {label}: the frozen scorer's RAE denominator is the "
                "soft-thresholded error of a constant mean predictor, which is zero when every "
                "label lies inside its own band. Refusing to fold a non-finite value into O150.")
    if not math.isfinite(got["macro_st_rae"]):
        raise SystemExit("non-finite macro ST-RAE")
    return got


def expand_seed_slots(partitions: dict, macros: dict, expected_slots: int) -> list:
    """One row per (partition, seed slot); each is a replicate of one computation."""
    rows = []
    for name in sorted(partitions):
        part = partitions[name]
        seeds = list(part["training_seeds"])
        if len(seeds) != expected_slots or len(set(seeds)) != expected_slots:
            raise SystemExit(f"{name}: {seeds} seed slots; expected {expected_slots} distinct")
        for seed in seeds:
            rows.append({
                "cell": {"fold": part["fold"], "repetition": part["repetition"],
                         "training_seed": int(seed)},
                "macro_st_rae": float(macros[name]["macro_st_rae"]),
                "per_endpoint_st_rae": {k: float(v) for k, v in macros[name]["per_task"].items()},
                "replicate_of": {"partition": name, "unique_computation": True},
            })
    return rows


def favours_actfound(value: float, comparator: float) -> bool:
    """D013's prospective comparison rule: STRICTLY lower favours ActFound-CYP.

    An exact tie does not favour it, and there is no minimum effect size.
    """
    return bool(value < comparator)


def o150(rows: list, expected_cells: int) -> float:
    if len(rows) != expected_cells:
        raise SystemExit(f"{len(rows)} cell values; O150 requires exactly {expected_cells}")
    return sum(r["macro_st_rae"] for r in rows) / len(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--predictions-sha256", required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)

    env = sp.assert_scoring_python()
    cfg = re_.read_eval_config(a.config, a.config_sha256)
    art = load_predictions(a.predictions, a.predictions_sha256)
    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")
    task_order = list(cfg["endpoint_task_order"])
    if list(art.get("endpoint_task_order", [])) != task_order:
        raise SystemExit("the predictions artifact was produced under a different task order")

    comp_cfg = cfg["comparator"]
    comparator = read_comparator(a.repo_root / comp_cfg["corroborating_artifact"],
                                 comp_cfg["corroborating_artifact_sha256"],
                                 comp_cfg["o150_t121"])
    cited = a.repo_root / comp_cfg["cited_artifact"]
    cited_present = cited.is_file() and qa.sha256(cited) == comp_cfg["cited_artifact_sha256"]

    sys.path.insert(0, str(a.repo_root))
    sys.path.insert(0, str(a.repo_root / "cyp" / "scripts"))
    from cyp002_validation import compute_validation_macro_st_rae

    partitions = group_by_partition(art["cells"], task_order)
    if len(partitions) != int(cfg["expected_partitions"]):
        raise SystemExit(f"{len(partitions)} partitions; expected {cfg['expected_partitions']}")
    macros = {name: score_partition(part, task_order, compute_validation_macro_st_rae)
              for name, part in sorted(partitions.items())}
    rows = expand_seed_slots(partitions, macros, int(cfg["expected_seed_slots"]))
    value = o150(rows, int(cfg["o150_cells"]))

    out = {
        "artifact_type": "D013_cyp_o150_result_v1",
        "o150": value,
        "t121_o150": comparator["o150_t121"],
        "strict_rule": comp_cfg["strict_rule"],
        "favours_actfound_cyp": favours_actfound(value, comparator["o150_t121"]),
        "cell_count": len(rows),
        "unique_adaptations": int(art["unique_adaptations"]),
        "seed_slots_are_replicates": cfg["seed_slots_are_replicates"],
        "comparison_is_final": bool(cited_present),
        "comparison_provenance_note": (
            "final" if cited_present else
            "NOT final: D013's cited comparator artifact is absent from this checkout; the "
            "corroborating D009 artifact carries the same value but a matching number does not "
            "establish equivalent provenance. A targeted check must locate or account for the "
            "cited artifact before the final comparison (D013 Amendment 4 item 6)."),
        "comparator": {**comparator, "cited_artifact": comp_cfg["cited_artifact"],
                       "cited_artifact_present": cited_present,
                       "corroborating_only": True},
        "scoring_environment": env,
        "frozen_scorer": {
            "st_rae_py_sha256": qa.sha256(a.repo_root / "cyp" / "scripts" / "st_rae.py"),
            "cyp002_validation_sha256": qa.sha256(a.repo_root / "cyp" / "scripts" / "cyp002_validation.py"),
            "custom_scoring_functions_sha256": qa.sha256(
                a.repo_root / "references" / "cyp_upstream" / "custom_scoring_functions.py"),
        },
        "endpoint_task_order": task_order,
        "inputs": {"config_sha256": a.config_sha256,
                   "predictions_sha256": a.predictions_sha256,
                   "script_sha256": qa.sha256(Path(__file__))},
        "cells": rows,
    }
    a.out_dir.mkdir(parents=True)
    path = a.out_dir / "cyp_o150_result.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (a.out_dir / "SHA256SUMS.txt").write_text(f"{qa.sha256(path)}  {path.name}\n")
    print(json.dumps({k: out[k] for k in ("o150", "t121_o150", "strict_rule", "favours_actfound_cyp",
                                          "cell_count", "unique_adaptations", "comparison_is_final",
                                          "comparison_provenance_note", "scoring_environment")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
