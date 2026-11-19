"""Tests for the PXR-003 baseline package. LightGBM-dependent tests skip if it is absent;
the smoke script (pxr003_smoke.py) is the loud check for that dependency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr003_common as common  # noqa: E402
from pxr.scripts import pxr003_final as final  # noqa: E402
from pxr.scripts import pxr003_models as models  # noqa: E402


def test_metrics_known_values():
    pred, truth = [1.0, 2.0, 4.0], [1.0, 3.0, 2.0]
    assert common.mae(pred, truth) == pytest.approx(1.0)
    assert common.rmse(pred, truth) == pytest.approx(np.sqrt(5 / 3))
    assert common.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_constant_metrics_uses_given_mean():
    result = common.constant_metrics(3.0, [2.0, 4.0])
    assert result["mae"] == pytest.approx(1.0) and result["rmse"] == pytest.approx(1.0)


def test_config_matches_approved_design():
    cfg = common.load_config()
    assert cfg["lgbm"]["n_trials"] == 60 and cfg["rf"]["n_trials"] == 40
    assert cfg["lgbm"]["n_estimators_cap"] == 3000 and cfg["lgbm"]["early_stopping_rounds"] == 100
    assert cfg["rf"]["n_estimators"] == 500
    assert cfg["seeds"]["search"] == 20261201
    assert cfg["seeds"]["confirm"] == [20261211, 20261212, 20261213]
    assert cfg["seeds"]["final"] == [20261220, 20261221, 20261222, 20261223, 20261224]
    assert not set(cfg["seeds"]["confirm"]) & set(cfg["seeds"]["final"])
    assert cfg["confirm_top_k"] == 3
    assert cfg["lgbm"]["fixed"]["metric"] == "l1"


@pytest.mark.parametrize("model", ["lgbm", "rf"])
def test_sampling_in_range_and_deterministic(model):
    ranges = common.load_config()[model]["ranges"]
    first = [models.sample_params(np.random.default_rng(5), ranges) for _ in range(1)][0]
    again = models.sample_params(np.random.default_rng(5), ranges)
    assert first == again
    rng = np.random.default_rng(1)
    for _ in range(200):
        params = models.sample_params(rng, ranges)
        for name, (kind, low, high) in ranges.items():
            assert low <= params[name] <= high
            if kind == "int":
                assert isinstance(params[name], int)


def test_select_top_and_tiebreak_lowest_trial():
    trials = pd.DataFrame({"trial": [3, 1, 2, 4], "val_mae": [0.5, 0.5, 0.4, 0.6]})
    assert models.select_top(trials, 3)["trial"].tolist() == [2, 1, 3]


def test_choose_best_by_mean_with_tiebreak():
    confirm = pd.DataFrame({"trial": [7, 7, 2, 2, 9, 9], "val_mae": [0.5, 0.5, 0.25, 0.75, 0.625, 0.625]})
    best = models.choose_best(confirm)
    assert best["trial"] == 2 and best["confirm_mean_val_mae"] == pytest.approx(0.5)  # tie with 7 -> lower trial


def test_load_cohort_rejects_overlap_free_but_wrong_counts():
    data = pd.DataFrame({"Molecule Name": list("ABCD"), "SMILES": ["CCO", "CCN", "CCC", "c1ccccc1"],
                         "pEC50": [4.0, 4.5, 5.0, 7.0]})
    assignments = pd.DataFrame({"Molecule Name": list("ABCD"), "split": ["train", "train", "validation", "holdout"]})
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        d, a = Path(tmp, "d.csv"), Path(tmp, "a.csv")
        data.to_csv(d, index=False)
        assignments.to_csv(a, index=False)
        train, validation, holdout_ids, _ = common.load_cohort(d, a, verify=False,
                                                              expected_counts={"train": 2, "validation": 1, "holdout": 1})
        assert holdout_ids == {"D"} and set(train["Molecule Name"]) == {"A", "B"}
        assert validation["pEC50"].notna().all()
        with pytest.raises(AssertionError):
            common.load_cohort(d, a, verify=False, expected_counts={"train": 3, "validation": 1, "holdout": 1})


def test_holdout_structures_have_no_target_column(tmp_path):
    pd.DataFrame({"Molecule Name": list("ABC"), "SMILES": ["CCO", "CCN", "CCC"], "pEC50": [1.0, 2.0, 3.0]}).to_csv(
        tmp_path / "d.csv", index=False)
    pd.DataFrame({"Molecule Name": list("ABC"), "split": ["train", "validation", "holdout"]}).to_csv(
        tmp_path / "a.csv", index=False)
    frame = common.load_holdout_structures(tmp_path / "d.csv", tmp_path / "a.csv")
    assert list(frame.columns) == ["Molecule Name", "SMILES"] and frame["Molecule Name"].tolist() == ["C"]


def _selection(model, config_sha, hashes=None):
    return {"model": model, "trial": 1, "params": {}, "confirm_mean_val_mae": 0.5, "config_sha256": config_sha,
            "input_hashes": hashes or {"train_csv": common.EXPECTED_TRAIN_SHA256,
                                       "assignments_csv": common.EXPECTED_ASSIGNMENTS_SHA256}}


def test_build_frozen_rejects_mismatched_config_or_inputs():
    frozen = final.build_frozen(_selection("lgbm", "x"), _selection("rf", "x"), "x")
    assert frozen["seeds_final"] == common.load_config()["seeds"]["final"]
    with pytest.raises(ValueError):
        final.build_frozen(_selection("lgbm", "x"), _selection("rf", "y"), "x")
    with pytest.raises(ValueError):
        final.build_frozen(_selection("lgbm", "x"), _selection("rf", "x", {"train_csv": "0", "assignments_csv": "0"}), "x")
    with pytest.raises(ValueError):
        final.build_frozen(_selection("rf", "x"), _selection("rf", "x"), "x")


def test_verify_frozen_hash_mismatch_exits(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"a": 1}))
    assert final.verify_frozen(path, common.sha256_file(path)) == {"a": 1}
    with pytest.raises(SystemExit):
        final.verify_frozen(path, "0" * 64)


def test_rf_fit_predicts_and_is_seed_deterministic():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(60, 32)).astype(np.float32)
    y = x[:, 0] * 2.0 + rng.normal(0, 0.1, 60)
    cfg = {"n_estimators": 10, "fixed": {"criterion": "squared_error", "bootstrap": True, "max_depth": None}}
    params = {"max_features": 0.5, "min_samples_leaf": 1, "max_samples": 0.8}
    p1, _ = models.fit_model("rf", params, cfg, x, y, x, y, 3, 1)
    p2, _ = models.fit_model("rf", params, cfg, x, y, x, y, 3, 1)
    assert np.allclose(p1(x), p2(x)) and p1(x).shape == (60,)


def test_lgbm_early_stops_on_validation_mae():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(200, 32)).astype(np.float32)
    y = x[:, 0] * 2.0 + rng.normal(0, 0.1, 200)
    cfg = dict(common.load_config()["lgbm"], n_estimators_cap=200, early_stopping_rounds=10)
    params = {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5, "colsample_bytree": 0.5,
              "subsample": 0.8, "reg_lambda": 1.0}
    predict, info = models.fit_model("lgbm", params, cfg, x[:150], y[:150], x[150:], y[150:], 1, 1)
    assert 1 <= info["best_iteration"] <= 200 and predict(x[150:]).shape == (50,)
