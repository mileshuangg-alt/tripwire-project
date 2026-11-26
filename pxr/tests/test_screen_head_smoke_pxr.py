"""Tests for the PXR-004 stage-1 screen-head loading smoke (Tripwire-CYP-010).

Each test is tied to a concrete failure risk: opening the stage-2 potency checkpoint instead
of the stage-1 screen head, a stage-1 checkpoint that does not match the one that fed stage 2,
a scaler that disagrees between the loaders, the stage-1 manifest and the checkpoint buffers,
reading validation-hook (standardized) units as native log2fc, taking the wrong head as the
8.25 uM head, and reusing an output directory. Writing these tests is not a passing run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pxr/scripts"))
import screen_head_smoke_pxr as smoke  # noqa: E402

PINS = REPO_ROOT / "pxr/artifacts/screen_heads_smoke/pxr_stage1_pins.json"


def _chemprop() -> bool:
    try:
        import chemprop  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


needs_chemprop = pytest.mark.skipif(not _chemprop(), reason="torch/chemprop unavailable")


# ------------------------------------------------------------------------- run files
def test_only_stage1_checkpoint_is_ever_named():
    paths = smoke.stage1_paths("/x/seed1/sequential_seed1")
    assert paths["checkpoint"].parts[-2:] == ("stage1", "best.ckpt")
    assert paths["stage2_manifest"].name == "manifest.json"
    assert not any(p.suffix == ".ckpt" and "stage2" in p.parts for p in paths.values())


def _write_run(tmp_path, sha, *, encoder_source=None, head=None, weights="W"):
    run = tmp_path / "run"
    (run / "stage1").mkdir(parents=True)
    (run / "stage2").mkdir(parents=True)
    (run / "stage1" / "best.ckpt").write_bytes(b"checkpoint")
    actual = smoke.sha256_file(run / "stage1" / "best.ckpt")
    head = head or {"mae": 0.2, "spearman": 0.5, "n": 974}
    (run / "stage1" / "metrics.json").write_text(json.dumps({
        "best_checkpoint_sha256": actual, "best_epoch": 2, "epochs_run": 23,
        "best_heads": {"log2fc_8p25uM": head},
    }))
    counts = {"log2fc_8p25uM": 8763}
    (run / "stage1" / "manifest.json").write_text(json.dumps({
        "hashes": {"weights_sha256": weights}, "train_label_counts": counts, "monitor_label_counts": {"log2fc_8p25uM": 974},
        "scaler_mean": [0.3, 0.6, 1.2], "scaler_scale": [0.4, 0.5, 0.6],
        "n_train_molecules": 8862, "n_monitor_molecules": 985,
    }))
    (run / "stage2" / "manifest.json").write_text(json.dumps({
        "encoder_source": encoder_source or smoke.expected_encoder_source(actual),
    }))
    pin = {"best_checkpoint_sha256": sha or actual, "best_epoch": 2, "epochs_run": 23,
           "head_8p25": {"mae": 0.2, "spearman": 0.5, "n": 974}}
    expected = {"head_order": ["log2fc_8p25uM", "log2fc_33uM", "log2fc_99uM"], "screen_head_index": 0,
                "train_label_counts": {"log2fc_8p25uM": 8763}, "monitor_label_counts": {"log2fc_8p25uM": 974}}
    return run, pin, expected


def test_consistent_seed_records_pass(tmp_path):
    run, pin, expected = _write_run(tmp_path, None)
    out = smoke.verify_seed_records(smoke.stage1_paths(run), pin, expected, "W")
    assert out["checkpoint_sha256"] == smoke.sha256_file(run / "stage1" / "best.ckpt")


def test_wrong_pinned_checkpoint_hash_is_rejected(tmp_path):
    run, pin, expected = _write_run(tmp_path, "0" * 64)
    with pytest.raises(ValueError):
        smoke.verify_seed_records(smoke.stage1_paths(run), pin, expected, "W")


def test_stage2_encoder_source_naming_another_checkpoint_is_rejected(tmp_path):
    run, pin, expected = _write_run(tmp_path, None, encoder_source="stage1_best_ckpt_sha256=" + "1" * 64)
    with pytest.raises(ValueError):
        smoke.verify_seed_records(smoke.stage1_paths(run), pin, expected, "W")


def test_changed_monitor_metric_or_weights_hash_is_rejected(tmp_path):
    run, pin, expected = _write_run(tmp_path, None, head={"mae": 0.3, "spearman": 0.5, "n": 974})
    with pytest.raises(ValueError):
        smoke.verify_seed_records(smoke.stage1_paths(run), pin, expected, "W")
    run2 = tmp_path / "second"
    run2.mkdir()
    run2, pin2, expected2 = _write_run(run2, None, weights="OTHER")
    with pytest.raises(ValueError):
        smoke.verify_seed_records(smoke.stage1_paths(run2), pin2, expected2, "W")


def test_staged_file_hash_check(tmp_path):
    (tmp_path / "a").write_text("x")
    good = {"a": smoke.sha256_file(tmp_path / "a")}
    assert smoke.verify_stage_files(tmp_path, good) == 1
    with pytest.raises(ValueError):
        smoke.verify_stage_files(tmp_path, {"a": "0" * 64})
    with pytest.raises(ValueError):
        smoke.verify_stage_files(tmp_path, {"missing": "0" * 64})


# ----------------------------------------------------------------- head order and scaling
def test_head_order_and_screen_index():
    order = ["log2fc_8p25uM", "log2fc_33uM", "log2fc_99uM"]
    assert smoke.check_head_order(order, order, 0) == "log2fc_8p25uM"
    with pytest.raises(ValueError):
        smoke.check_head_order(["log2fc_33uM", "log2fc_8p25uM", "log2fc_99uM"], order, 0)
    with pytest.raises(ValueError):
        smoke.check_head_order(order, order, 1)  # index 1 is not the 8.25 uM head


def test_scaler_agreement_three_way():
    mean, scale = np.array([0.3, 0.6, 1.2]), np.array([0.4, 0.5, 0.6])
    out = smoke.check_scaler_agreement((mean, scale), (mean, scale), (mean.astype(np.float32), scale.astype(np.float32)), 1e-9, 1e-6)
    assert out["mean"]["recomputed_vs_manifest"] == 0.0
    with pytest.raises(ValueError):
        smoke.check_scaler_agreement((mean, scale), (mean + 1e-3, scale), (mean, scale), 1e-9, 1e-6)
    with pytest.raises(ValueError):
        smoke.check_scaler_agreement((mean, scale), (mean, scale), (mean, scale * 1.01), 1e-9, 1e-6)
    with pytest.raises(ValueError):
        smoke.check_scaler_agreement((mean, scale), (mean[:2], scale[:2]), (mean, scale), 1e-9, 1e-6)


def test_unscale_identity_and_non_applied_transform():
    standardized = np.array([[0.0, 1.0, -1.0], [2.0, 0.5, 0.0]])
    mean, scale = np.array([0.3, 0.6, 1.2]), np.array([0.4, 0.5, 0.6])
    native = standardized * scale + mean
    assert smoke.unscale_consistency(native, standardized, mean, scale, 1e-5)["max_abs_difference"] < 1e-12
    with pytest.raises(ValueError):
        smoke.unscale_consistency(native + 0.01, standardized, mean, scale, 1e-5)
    with pytest.raises(ValueError):  # a transform that was never applied returns standardized values
        smoke.unscale_consistency(standardized, standardized, mean, scale, 1e-5)


def test_output_sanity_check_rejects_wrong_shape_nan_and_large_values():
    assert smoke.check_native_outputs(np.zeros((16, 3)), 16, 3, 10.0)["shape"] == [16, 3]
    with pytest.raises(ValueError):
        smoke.check_native_outputs(np.zeros((16, 4)), 16, 3, 10.0)
    bad = np.zeros((16, 3))
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        smoke.check_native_outputs(bad, 16, 3, 10.0)
    big = np.zeros((16, 3))
    big[1, 1] = 11.0
    with pytest.raises(ValueError):
        smoke.check_native_outputs(big, 16, 3, 10.0)


def test_output_directory_never_reused_and_recorder_needs_a_pass(tmp_path):
    target = tmp_path / "out"
    smoke.new_output_dir(target)
    with pytest.raises(FileExistsError):
        smoke.new_output_dir(target)
    checks = smoke.Checks()
    assert not checks.passed
    checks.record("a", True)
    assert checks.passed
    checks.record("b", False)
    assert not checks.passed


# ------------------------------------------------------- a synthetic stage-1-like model
@needs_chemprop
def test_stage1_like_model_buffers_in_state_dict_and_eval_vs_train_outputs():
    import torch
    from chemprop import data, models, nn
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    scaler = StandardScaler().fit(rng.normal(loc=[0.3, 0.6, 1.2], scale=[0.4, 0.5, 0.6], size=(50, 3)))
    predictor = nn.RegressionFFN(
        n_tasks=3, input_dim=16, hidden_dim=8, n_layers=1,
        output_transform=nn.transforms.UnscaleTransform.from_standard_scaler(scaler),
    )
    model = models.MPNN(
        message_passing=nn.BondMessagePassing(d_h=16, depth=2), agg=nn.MeanAggregation(),
        predictor=predictor, batch_norm=False,
    )
    mean, scale = smoke.scaler_from_state_dict(model.state_dict())
    np.testing.assert_allclose(mean, scaler.mean_, atol=1e-6)
    np.testing.assert_allclose(scale, scaler.scale_, atol=1e-6)

    points = [data.MoleculeDatapoint.from_smi(s, np.zeros(3)) for s in ["CCO", "CCN", "CCC", "c1ccccc1"]]
    batch = next(iter(data.build_dataloader(data.MoleculeDataset(points), batch_size=4, num_workers=0, shuffle=False)))
    bmg, V_d, X_d = batch[0], batch[1], batch[2]

    model.eval()
    with torch.inference_mode():
        native = model(bmg, V_d, X_d).numpy()
        model.predictor.output_transform.train()  # what the validation hook forces
        standardized = model(bmg, V_d, X_d).numpy()
        model.predictor.output_transform.eval()
    out = smoke.unscale_consistency(native, standardized, mean, scale, 1e-5)
    assert out["max_abs_difference"] < 1e-5


@needs_chemprop
def test_scaler_from_state_dict_rejects_a_model_without_the_transform():
    from chemprop import models, nn

    predictor = nn.RegressionFFN(n_tasks=3, input_dim=16, hidden_dim=8, n_layers=1)
    model = models.MPNN(
        message_passing=nn.BondMessagePassing(d_h=16, depth=2), agg=nn.MeanAggregation(),
        predictor=predictor, batch_norm=False,
    )
    with pytest.raises(ValueError):
        smoke.scaler_from_state_dict(model.state_dict())


def test_pins_file_is_consistent():
    pins = json.loads(PINS.read_text())
    assert set(pins["seeds"]) == {"20261125", "20261126", "20261127"}
    assert pins["expected"]["head_order"][pins["expected"]["screen_head_index"]] == "log2fc_8p25uM"
    assert len(pins["stage_files"]) == 16
    for pin in pins["seeds"].values():
        assert len(pin["best_checkpoint_sha256"]) == 64
        assert pin["run_dir"].endswith(f"sequential_seed{Path(pin['run_dir']).name[-8:]}")
        assert "stage2" not in pin["run_dir"]
