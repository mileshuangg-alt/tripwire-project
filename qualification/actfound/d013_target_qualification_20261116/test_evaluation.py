"""Tests for the D013 evaluation stages (thin target adapter, open item 4).

Stdlib plus numpy/pandas only; no CUDA and no upstream model. The GPU half of
stage 1 is covered through its injectable pieces, and stage 2 end to end against
the real frozen scorer and the real comparator artifact.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import qualify_adaptation as qa  # noqa: E402
import run_evaluation as re_  # noqa: E402
import score_evaluation as se  # noqa: E402

EVAL_CONFIG = HERE / "d013_evaluation_config_v1.json"
QUAL_CONFIG = HERE / "d013_target_qualification_config_v1.json"
LABELS = ["CYP1A2_pIC50_direct_inhibition", "CYP2C9_pIC50_direct_inhibition",
          "CYP2D6_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _cfg() -> dict:
    return json.loads(EVAL_CONFIG.read_text())


# ---------------------------------------------------------------- config

def test_evaluation_config_is_frozen_and_consistent():
    cfg = re_.read_eval_config(EVAL_CONFIG, _sha(EVAL_CONFIG))
    assert cfg["artifact_type"] == re_.CONFIG_ARTIFACT
    assert cfg["endpoint_task_order"] == LABELS
    assert cfg["expected_partitions"] == 50 and cfg["expected_endpoints"] == 4
    assert cfg["expected_seed_slots"] == 3 and cfg["o150_cells"] == 150
    assert cfg["expected_partitions"] * cfg["expected_seed_slots"] == cfg["o150_cells"]
    assert cfg["comparator"]["o150_t121"] == 0.701212576796
    assert cfg["comparator"]["cited_artifact_present_in_checkout"] is False
    assert cfg["qualification_config_sha256"] == _sha(QUAL_CONFIG)
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        re_.read_eval_config(EVAL_CONFIG, "0" * 64)


def test_endpoint_order_follows_the_frozen_task_order():
    cfg, qual = _cfg(), json.loads(QUAL_CONFIG.read_text())
    order = re_.endpoint_order(cfg, qual)
    assert [e["label"] for e in order] == LABELS
    assert [e["name"] for e in order] == ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
    bad = {**cfg, "endpoint_task_order": LABELS[:3] + ["CYP9Z9_pIC50_direct_inhibition"]}
    with pytest.raises(SystemExit, match="absent from the qualification config"):
        re_.endpoint_order(bad, qual)
    with pytest.raises(SystemExit, match="expected 4"):
        re_.endpoint_order({**cfg, "endpoint_task_order": LABELS[:3]}, qual)


# ---------------------------------------------------------------- feature cache

def test_feature_cache_is_deterministic_and_reuses_rows():
    calls = []

    def featurizer(smiles):
        calls.append(list(smiles))
        return np.array([[float(len(s)), 1.0] for s in smiles], dtype=np.float32)

    cache = re_.FeatureCache(featurizer)
    a = cache.matrix(["C", "CC", "C"])
    assert a.shape == (3, 2) and a[0].tolist() == a[2].tolist()
    assert calls == [["C", "CC"]]                  # distinct molecules only, once
    assert (cache.misses, cache.hits) == (2, 1)
    b = cache.matrix(["CC", "C"])
    assert calls == [["C", "CC"]]                  # nothing refeaturised
    assert b[0].tolist() == a[1].tolist() and b[1].tolist() == a[0].tolist()
    assert (cache.misses, cache.hits) == (2, 3)


def test_feature_cache_rejects_a_molecule_the_featuriser_drops():
    cache = re_.FeatureCache(lambda s: np.zeros((len(s) - 1, 2), dtype=np.float32))
    with pytest.raises(SystemExit, match="failed to parse"):
        cache.matrix(["C", "bad"])


# ---------------------------------------------------------------- predictions artifact

def _cell(partition: str, rep: int, fold: int, label: str, *, n=5, exact=True, shift=0.0) -> dict:
    # Spread wide enough that a constant mean predictor lies outside most bands,
    # so the frozen scorer's RAE denominator is non-zero (see check_st_rae_invariants).
    y = [4.0 + 1.0 * i + 0.01 * LABELS.index(label) for i in range(n)]
    preds = list(y) if exact else [v + shift for v in y]
    return {
        "partition": partition, "repetition": rep, "fold": fold,
        "endpoint": label.split("_")[0], "endpoint_label": label,
        "training_seeds": [20261101, 20261102, 20261103],
        "support_count": 1000, "target_count": n,
        "target_names": [f"{partition}_V{i}" for i in range(n)],
        "target_y": y,
        "target_conf_low": [v - 0.3 for v in y],
        "target_conf_high": [v + 0.3 for v in y],
        "target_predictions": preds,
        "inner_support_loss_steps": 5,
        "seconds": {"featurize": 0.1, "adapt": 0.02},
    }


def _artifact(tmp: Path, *, partitions=50, labels=None, exact=True, shift=0.0, **over) -> tuple[Path, dict]:
    labels = labels or LABELS
    cells = []
    for k in range(partitions):
        rep, fold = k // 10 + 1, k % 10 + 1
        name = f"rep_{rep:02d}_fold_{fold:02d}"
        for label in labels:
            cells.append(_cell(name, rep, fold, label, exact=exact, shift=shift))
    art = {
        "artifact_type": se.PREDICTIONS_ARTIFACT,
        "contains_no_scores": True,
        "unique_adaptations": len(cells),
        "endpoint_task_order": LABELS,
        "cells": cells,
        **over,
    }
    p = tmp / "preds.json"
    p.write_text(json.dumps(art))
    return p, art


def test_grouping_requires_all_four_endpoints_exactly_once(tmp_path):
    _, art = _artifact(tmp_path, partitions=2)
    parts = se.group_by_partition(art["cells"], LABELS)
    assert len(parts) == 2 and all(len(p["endpoints"]) == 4 for p in parts.values())
    _, missing = _artifact(tmp_path, partitions=1, labels=LABELS[:3])
    with pytest.raises(SystemExit, match="missing endpoints"):
        se.group_by_partition(missing["cells"], LABELS)
    _, dup = _artifact(tmp_path, partitions=1, labels=LABELS[:1] + LABELS[:1])
    with pytest.raises(SystemExit, match="appears twice"):
        se.group_by_partition(dup["cells"], LABELS)


def test_grouping_rejects_ragged_predictions(tmp_path):
    _, art = _artifact(tmp_path, partitions=1)
    art["cells"][0]["target_predictions"] = art["cells"][0]["target_predictions"][:-1]
    with pytest.raises(SystemExit, match="predictions, labels and bands disagree"):
        se.group_by_partition(art["cells"], LABELS)


# ---------------------------------------------------------------- macro and O150

def test_macro_uses_the_frozen_aggregator_over_all_four_endpoints(tmp_path):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "cyp" / "scripts"))
    from cyp002_validation import compute_validation_macro_st_rae
    _, art = _artifact(tmp_path, partitions=1, exact=False, shift=1.0)
    part = se.group_by_partition(art["cells"], LABELS)["rep_01_fold_01"]
    got = se.score_partition(part, LABELS, compute_validation_macro_st_rae)
    assert set(got["per_task"]) == set(LABELS)
    assert got["macro_st_rae"] == pytest.approx(np.mean([got["per_task"][t] for t in LABELS]))
    assert got["macro_st_rae"] > 0.0
    assert got["task_order"] == LABELS


def test_seed_slots_expand_to_three_replicates_per_partition(tmp_path):
    _, art = _artifact(tmp_path, partitions=2)
    parts = se.group_by_partition(art["cells"], LABELS)
    macros = {n: {"macro_st_rae": 0.5 + 0.1 * i, "per_task": {t: 0.5 for t in LABELS}}
              for i, n in enumerate(sorted(parts))}
    rows = se.expand_seed_slots(parts, macros, 3)
    assert len(rows) == 6
    assert [r["cell"]["training_seed"] for r in rows[:3]] == [20261101, 20261102, 20261103]
    assert {r["macro_st_rae"] for r in rows[:3]} == {0.5}
    assert all(r["replicate_of"]["unique_computation"] for r in rows)
    assert rows[0]["cell"]["repetition"] == 1 and rows[0]["cell"]["fold"] == 1


def test_seed_slot_expansion_rejects_a_wrong_or_duplicated_slot_set(tmp_path):
    _, art = _artifact(tmp_path, partitions=1)
    parts = se.group_by_partition(art["cells"], LABELS)
    macros = {"rep_01_fold_01": {"macro_st_rae": 0.5, "per_task": {t: 0.5 for t in LABELS}}}
    parts["rep_01_fold_01"]["training_seeds"] = [1, 2]
    with pytest.raises(SystemExit, match="seed slots; expected 3"):
        se.expand_seed_slots(parts, macros, 3)
    parts["rep_01_fold_01"]["training_seeds"] = [1, 1, 2]
    with pytest.raises(SystemExit, match="expected 3 distinct"):
        se.expand_seed_slots(parts, macros, 3)


def test_o150_is_the_mean_of_exactly_150_values():
    rows = [{"macro_st_rae": 0.6} for _ in range(150)]
    assert se.o150(rows, 150) == pytest.approx(0.6)
    rows = [{"macro_st_rae": float(i)} for i in range(150)]
    assert se.o150(rows, 150) == pytest.approx(sum(range(150)) / 150)
    with pytest.raises(SystemExit, match="requires exactly 150"):
        se.o150(rows[:149], 150)
    with pytest.raises(SystemExit, match="requires exactly 150"):
        se.o150(rows + rows[:1], 150)


# ---------------------------------------------------------------- comparator

def test_comparator_is_read_from_its_artifact(tmp_path):
    cfg = _cfg()["comparator"]
    path = ROOT / cfg["corroborating_artifact"]
    got = se.read_comparator(path, cfg["corroborating_artifact_sha256"], cfg["o150_t121"])
    assert got["o150_t121"] == 0.701212576796
    assert "0.701212576796" in got["strict_rule"]
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        se.read_comparator(path, "0" * 64, cfg["o150_t121"])
    with pytest.raises(SystemExit, match="the config declares"):
        se.read_comparator(path, cfg["corroborating_artifact_sha256"], 0.5)
    empty = tmp_path / "c.json"
    empty.write_text("{}")
    with pytest.raises(SystemExit, match="no t121_o150"):
        se.read_comparator(empty, _sha(empty), cfg["o150_t121"])


def test_cited_comparator_artifact_is_absent_so_the_config_says_so():
    cfg = _cfg()["comparator"]
    assert not (ROOT / cfg["cited_artifact"]).is_file()
    assert cfg["cited_artifact_present_in_checkout"] is False


# ---------------------------------------------------------------- stage 2 end to end

def _run_stage2(tmp_path, preds: Path, out: str = "scores"):
    o = tmp_path / out
    rc = se.main(["--config", str(EVAL_CONFIG), "--config-sha256", _sha(EVAL_CONFIG),
                  "--predictions", str(preds), "--predictions-sha256", _sha(preds),
                  "--repo-root", str(ROOT), "--out-dir", str(o)])
    return rc, json.loads((o / "cyp_o150_result.json").read_text()), o


def test_stage2_end_to_end_exact_predictions_give_zero_o150(tmp_path):
    pytest.importorskip("pandas")
    preds, _ = _artifact(tmp_path, exact=True)
    rc, s, out = _run_stage2(tmp_path, preds)
    assert rc == 0
    assert s["cell_count"] == 150 and s["unique_adaptations"] == 200
    assert s["o150"] == pytest.approx(0.0)
    assert s["t121_o150"] == 0.701212576796
    assert s["favours_actfound_cyp"] is True
    assert s["comparison_is_final"] is False and "NOT final" in s["comparison_provenance_note"]
    assert s["comparator"]["corroborating_only"] is True
    assert s["comparator"]["cited_artifact_present"] is False
    assert s["frozen_scorer"]["custom_scoring_functions_sha256"] == \
        _sha(ROOT / "references" / "cyp_upstream" / "custom_scoring_functions.py")
    assert len({(r["cell"]["repetition"], r["cell"]["fold"], r["cell"]["training_seed"])
                for r in s["cells"]}) == 150
    assert (out / "SHA256SUMS.txt").exists()


def test_stage2_reports_a_nonzero_o150_and_the_strict_rule(tmp_path):
    pytest.importorskip("pandas")
    preds, _ = _artifact(tmp_path, exact=False, shift=5.0)   # far outside every band
    _, s, _ = _run_stage2(tmp_path, preds)
    assert s["o150"] > 0.701212576796
    assert s["favours_actfound_cyp"] is False
    # every partition is identical here, so O150 equals any cell's macro
    assert s["o150"] == pytest.approx(s["cells"][0]["macro_st_rae"])


def test_strict_rule_excludes_an_exact_tie():
    """D013: a strictly lower O150 favours ActFound-CYP; an exact tie does not."""
    t121 = 0.701212576796
    assert se.favours_actfound(t121 - 1e-12, t121) is True
    assert se.favours_actfound(t121, t121) is False
    assert se.favours_actfound(t121 + 1e-12, t121) is False
    assert se.favours_actfound(0.0, t121) is True


def test_stage2_refusals(tmp_path):
    pytest.importorskip("pandas")
    preds, art = _artifact(tmp_path)
    args = ["--config", str(EVAL_CONFIG), "--config-sha256", _sha(EVAL_CONFIG),
            "--predictions", str(preds), "--predictions-sha256", _sha(preds),
            "--repo-root", str(ROOT)]
    (tmp_path / "exists").mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        se.main(args + ["--out-dir", str(tmp_path / "exists")])
    bad = list(args)
    bad[bad.index("--predictions-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        se.main(bad + ["--out-dir", str(tmp_path / "o1")])
    scored = tmp_path / "scored.json"
    scored.write_text(json.dumps({**art, "contains_no_scores": False}))
    with pytest.raises(SystemExit, match="refusing to rescore"):
        se.main(["--config", str(EVAL_CONFIG), "--config-sha256", _sha(EVAL_CONFIG),
                 "--predictions", str(scored), "--predictions-sha256", _sha(scored),
                 "--repo-root", str(ROOT), "--out-dir", str(tmp_path / "o2")])
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({**art, "artifact_type": "something_else"}))
    with pytest.raises(SystemExit, match="not a D013_cyp_adaptation_predictions_v1"):
        se.main(["--config", str(EVAL_CONFIG), "--config-sha256", _sha(EVAL_CONFIG),
                 "--predictions", str(wrong), "--predictions-sha256", _sha(wrong),
                 "--repo-root", str(ROOT), "--out-dir", str(tmp_path / "o3")])
    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps({**art, "endpoint_task_order": list(reversed(LABELS))}))
    with pytest.raises(SystemExit, match="different task order"):
        se.main(["--config", str(EVAL_CONFIG), "--config-sha256", _sha(EVAL_CONFIG),
                 "--predictions", str(reordered), "--predictions-sha256", _sha(reordered),
                 "--repo-root", str(ROOT), "--out-dir", str(tmp_path / "o4")])


def test_scoring_refuses_a_degenerate_cell_rather_than_emitting_nan(tmp_path):
    """All labels inside their own bands zeroes the RAE denominator; NaN must not reach O150."""
    pytest.importorskip("pandas")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "cyp" / "scripts"))
    from cyp002_validation import compute_validation_macro_st_rae
    _, art = _artifact(tmp_path, partitions=1)
    for cell in art["cells"]:                      # tight labels, wide bands
        cell["target_y"] = [5.0, 5.01, 5.02, 5.03, 5.04]
        cell["target_conf_low"] = [v - 5.0 for v in cell["target_y"]]
        cell["target_conf_high"] = [v + 5.0 for v in cell["target_y"]]
        cell["target_predictions"] = list(cell["target_y"])
    part = se.group_by_partition(art["cells"], LABELS)["rep_01_fold_01"]
    with pytest.raises(SystemExit, match="non-finite ST-RAE"):
        se.score_partition(part, LABELS, compute_validation_macro_st_rae)


def test_stage2_rejects_a_partition_count_other_than_fifty(tmp_path):
    pytest.importorskip("pandas")
    preds, _ = _artifact(tmp_path, partitions=49)
    with pytest.raises(SystemExit, match="49 partitions; expected 50"):
        _run_stage2(tmp_path, preds)


def test_evaluation_scripts_run_under_python_isolated_mode():
    import subprocess
    for script in ("run_evaluation.py", "score_evaluation.py"):
        r = subprocess.run([sys.executable, "-I", str(HERE / script), "--help"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"{script} failed under -I: {r.stderr[-400:]}"
        assert "--out-dir" in r.stdout


def test_stage1_writes_no_score_and_stage2_demands_that():
    src = (HERE / "run_evaluation.py").read_text()
    assert "st_rae" not in src and "macro" not in src.replace("macro_st_rae", "")
    assert '"contains_no_scores": True' in src
