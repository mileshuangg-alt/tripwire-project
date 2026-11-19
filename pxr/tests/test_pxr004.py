"""Tests for the PXR-004 CheMeleon pilot code (D020).

Each test is tied to a concrete failure risk: the stage-1 monitor leaking into stage-1
training or validation, a scaler fit on monitor labels, group 417 entering the monitor,
stage 2 losing monitor molecules' pEC50 labels, the selection metric being computed in the
wrong units, stage 2 inheriting stage-1 state or an arm-dependent head initialization, and an
unverified or wrongly shaped foundation checkpoint. Real-data tests skip when the artifacts
are absent. Writing these tests is not a passing run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr004_data as d  # noqa: E402
from pxr.scripts import pxr004_train as t  # noqa: E402

V1_DIR = REPO_ROOT / "pxr/artifacts/pxr002_aux"
ASSIGNMENTS = REPO_ROOT / "pxr/artifacts/pxr001_split/pxr001_split_assignments.csv"
WEIGHTS = REPO_ROOT / "pxr/artifacts/pxr004_weights/chemeleon_mp.pt"
WEIGHTS_SHA256 = "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"

TINY_HP = {
    "d_v": 72, "d_e": 14, "d_h": 16, "bias": False, "depth": 2, "dropout": 0.0,
    "activation": "relu", "undirected": False, "d_vd": None, "V_d_transform": None, "graph_transform": None,
}
SMILES = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O", "CCOC", "CNC"]


def _chemprop() -> bool:
    try:
        import chemprop  # noqa: F401
        return True
    except Exception:
        return False


needs_chemprop = pytest.mark.skipif(not _chemprop(), reason="Chemprop unavailable")
needs_real = pytest.mark.skipif(
    not (V1_DIR.exists() and ASSIGNMENTS.exists()), reason="v1 artifact or split assignments absent"
)


def synth_frame(seed: int = 0) -> pd.DataFrame:
    """Many small groups plus one large group 417; some molecules without auxiliary labels."""
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    sizes = [1] * 60 + [2] * 10 + [3] * 4
    sizes.append(30)  # becomes group 417 below
    for i, size in enumerate(sizes):
        group = 417 if size == 30 else gid
        if size != 30:
            gid += 1
            if gid == 417:
                gid += 1
        for _ in range(size):
            k = len(rows)
            primary = rng.random() < 0.4
            labels = rng.normal(size=3)
            labels[rng.random(3) < 0.3] = np.nan
            rows.append({
                "row_id": f"R{k}", "role": "primary_train" if primary else "aux_only",
                "SMILES": SMILES[k % len(SMILES)], "pEC50": float(rng.normal(5, 1)) if primary else np.nan,
                "log2fc_8p25uM": labels[0], "log2fc_33uM": labels[1], "log2fc_99uM": labels[2],
                "OCNT_ID": f"O{k}", "group_id": group,
            })
    frame = pd.DataFrame(rows)
    frame["n_aux"] = frame[list(d.AUX_TASKS)].notna().sum(axis=1)
    return frame


# ------------------------------------------------------------------------------ monitor
def test_draw_is_deterministic_whole_groups_and_excludes_group_417():
    frame = synth_frame()
    first = d.draw_monitor_groups(frame)
    assert first == d.draw_monitor_groups(frame)
    assert d.EXCLUDED_GROUP not in first
    assert len(set(first)) == len(first)
    assert first != d.draw_monitor_groups(frame, seed=d.MONITOR_SEED + 1)  # the seed matters
    labelled = frame[frame["n_aux"] > 0]
    held = labelled[labelled["group_id"].isin(first)]
    assert len(held) >= d.MONITOR_FRACTION * len(labelled)
    assert len(held) < d.MONITOR_FRACTION * len(labelled) + 4  # overshoots by < one group


def test_stage1_split_is_a_partition_of_labelled_molecules_by_whole_groups():
    frame = synth_frame()
    groups = set(d.draw_monitor_groups(frame))
    train, monitor = d.stage1_frames(frame, groups)
    assert set(train["row_id"]).isdisjoint(monitor["row_id"])
    assert set(train["group_id"]).isdisjoint(set(monitor["group_id"]))
    assert set(train["row_id"]) | set(monitor["row_id"]) == set(frame.loc[frame["n_aux"] > 0, "row_id"])
    assert (frame.loc[frame["n_aux"] == 0, "row_id"].isin(train["row_id"]) == False).all()  # noqa: E712


def test_monitor_table_must_be_whole_groups_and_exclude_417(tmp_path):
    frame = synth_frame()
    chosen = d.draw_monitor_groups(frame)
    table = d.monitor_table(frame, chosen)
    d_dir = tmp_path / "m"
    d_dir.mkdir()
    table.to_csv(d_dir / d.MONITOR_FILE, index=False)
    (d_dir / "SHA256SUMS").write_text(f"{d.sha256(d_dir / d.MONITOR_FILE)}  {d.MONITOR_FILE}\n")
    assert d.load_monitor(d_dir, frame) == set(chosen)
    # a group only partly in the monitor is rejected
    bad = table.copy()
    big = bad[bad["group_id"] == 417].index[0]
    bad.loc[big, "in_monitor"] = True
    bad.to_csv(d_dir / d.MONITOR_FILE, index=False)
    (d_dir / "SHA256SUMS").write_text(f"{d.sha256(d_dir / d.MONITOR_FILE)}  {d.MONITOR_FILE}\n")
    with pytest.raises(ValueError):
        d.load_monitor(d_dir, frame)
    # an edited file no longer matches its recorded hash
    (d_dir / "SHA256SUMS").write_text(f"{'0' * 64}  {d.MONITOR_FILE}\n")
    with pytest.raises(RuntimeError):
        d.load_monitor(d_dir, frame)


# ----------------------------------------------------------------------------- datasets
@needs_chemprop
def test_stage1_scaler_is_fit_on_stage1_training_labels_only():
    frame = synth_frame()
    groups = set(d.draw_monitor_groups(frame))
    train_f, mon_f = d.stage1_frames(frame, groups)
    mon_f = mon_f.copy()
    for c in d.AUX_TASKS:
        mon_f[c] = mon_f[c].where(mon_f[c].isna(), 1000.0)  # extreme monitor labels
    sets = d.build_stage1_datasets(train_f, mon_f)
    train_mean = np.nanmean(train_f[list(d.AUX_TASKS)].to_numpy(dtype=float), axis=0)
    assert np.allclose(sets["y_scaler"].mean_, train_mean)
    original = mon_f[list(d.AUX_TASKS)].to_numpy(dtype=float)
    expected = (original - sets["y_scaler"].mean_) / sets["y_scaler"].scale_
    got = np.asarray(sets["monitor"].Y, dtype=float)
    assert np.allclose(got, expected, equal_nan=True)  # frozen scaler applied to the monitor


@needs_chemprop
def test_stage2_trains_on_every_primary_train_molecule_including_monitor_molecules():
    frame = synth_frame()
    groups = set(d.draw_monitor_groups(frame))
    in_monitor_primary = frame[(frame["role"] == "primary_train") & frame["group_id"].isin(groups) & (frame["n_aux"] > 0)]
    assert len(in_monitor_primary) > 0
    validation = frame[frame["role"] == "primary_train"].head(5).copy()
    sets = d.build_stage2_datasets(frame, validation)
    assert sets["n_train"] == int((frame["role"] == "primary_train").sum())
    assert len(sets["train"].data) == sets["n_train"]
    assert sets["y_scaler"].mean_.shape == (1,)


@needs_chemprop
def test_datasets_are_descriptor_free():
    frame = synth_frame()
    sets = d.build_stage2_datasets(frame, frame[frame["role"] == "primary_train"].head(3))
    assert all(dp.x_d is None for dp in sets["train"].data)
    assert all(dp.x_d is None for dp in sets["validation"].data)


# ------------------------------------------------------------------------------ metrics
def test_stage1_metrics_pooled_mse_and_per_head_units():
    preds = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, np.nan], [2.0, np.nan, np.nan]])
    targets = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, np.nan], [2.0, np.nan, np.nan]])
    scale = np.array([2.0, 10.0, 1.0])
    m = t.stage1_metrics(preds, targets, scale)
    # finite pairs: (1,0) (0,0) (2,0) (0,0) (1,3) (2,2): squared errors 1 0 4 0 4 0 -> 9/6
    assert m["pooled_mse"] == pytest.approx(9 / 6)
    assert m["heads"]["log2fc_8p25uM"]["n"] == 3
    assert m["heads"]["log2fc_8p25uM"]["mae"] == pytest.approx((1 + 0 + 0) / 3 * 2.0)
    assert m["heads"]["log2fc_33uM"]["mae"] == pytest.approx((0 + 2) / 2 * 10.0)
    assert m["heads"]["log2fc_99uM"]["n"] == 1


def test_stage1_metrics_rejects_an_empty_monitor():
    with pytest.raises(ValueError):
        t.stage1_metrics(np.zeros((2, 3)), np.full((2, 3), np.nan), np.ones(3))


def test_stage2_metrics_in_original_units():
    m = t.stage2_metrics(np.array([5.0, 6.0, 7.0]), np.array([5.0, 7.0, 6.0]))
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert m["spearman"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        t.stage2_metrics(np.array([1.0]), np.array([np.nan]))


# ------------------------------------------------------------------------------- models
def _scaler(n):
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit(np.random.default_rng(1).normal(5, 2, size=(20, n)))


def _state(seed):
    import torch
    from chemprop import nn

    torch.manual_seed(seed)
    return nn.BondMessagePassing(**TINY_HP).state_dict()


@needs_chemprop
def test_validation_runs_with_the_output_transform_in_train_mode():
    """Predictions during validation are in standardized units (identity transform)."""
    model = t.build_model("stage2", t.make_encoder(TINY_HP, _state(0)), _scaler(1))
    model.eval()
    assert model.predictor.output_transform.training is False
    model.on_validation_model_eval()
    assert model.predictor.output_transform.training is True


@needs_chemprop
def test_head_initialization_is_paired_across_arms():
    """Direct and sequential arms start stage 2 with identical fresh heads for a given seed,
    even though their encoder weights differ."""
    import torch
    from lightning import pytorch as pl

    def head_params(encoder_seed: int):
        state, scaler = _state(encoder_seed), _scaler(1)  # built before seeding: _state reseeds torch
        pl.seed_everything(123, workers=True)
        model = t.build_model("stage2", t.make_encoder(TINY_HP, state), scaler)
        return [p.detach().clone() for p in model.predictor.parameters()]

    a, b = head_params(1), head_params(2)
    assert len(a) == len(b) > 0
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@needs_chemprop
def test_head_is_the_library_default_ffn_and_mean_aggregation():
    from chemprop import nn

    model = t.build_model("stage1", t.make_encoder(TINY_HP, _state(0)), _scaler(3))
    assert isinstance(model.agg, nn.MeanAggregation)
    first = next(m for m in model.predictor.modules() if isinstance(m, __import__("torch").nn.Linear))
    assert first.in_features == 16 and first.out_features == t.FFN_HIDDEN == 300
    assert t.FFN_LAYERS == 1 and t.FFN_DROPOUT == 0.0
    assert (t.MAX_EPOCHS, t.PATIENCE, t.MIN_DELTA, t.BATCH_SIZE) == (100, 20, 0.0, 64)


@needs_chemprop
def test_only_encoder_weights_cross_the_stage_boundary(tmp_path):
    import torch

    state = _state(3)
    ck = {
        "state_dict": {**{f"message_passing.{k}": v for k, v in state.items()},
                       "predictor.ffn.0.weight": torch.zeros(2, 2)},
        "optimizer_states": [{"state": {"x": 1}}],
        "lr_schedulers": [{"last_epoch": 7}],
    }
    path = tmp_path / "best.ckpt"
    torch.save(ck, path)
    got = t.encoder_state_from_checkpoint(path)
    assert set(got) == set(state)
    assert all(torch.equal(got[k], state[k]) for k in state)
    path2 = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {"predictor.x": torch.zeros(1)}}, path2)
    with pytest.raises(ValueError):
        t.encoder_state_from_checkpoint(path2)


# ---------------------------------------------------------------------------- foundation
@needs_chemprop
def test_load_foundation_rejects_hash_mismatch_and_wrong_dimensions(tmp_path):
    import torch

    ck = tmp_path / "w" / t.WEIGHTS_NAME
    ck.parent.mkdir()
    torch.save({"hyper_parameters": TINY_HP, "state_dict": _state(0)}, ck)
    (ck.parent / "SHA256SUMS").write_text(f"{'0' * 64}  {t.WEIGHTS_NAME}\n")
    with pytest.raises(RuntimeError):  # hash mismatch is checked first
        t.load_foundation(ck)
    (ck.parent / "SHA256SUMS").write_text(f"{d.sha256(ck)}  {t.WEIGHTS_NAME}\n")
    with pytest.raises(ValueError):  # d_h 16 is not the CheMeleon encoder
        t.load_foundation(ck)


@needs_chemprop
@pytest.mark.skipif(not WEIGHTS.exists(), reason="CheMeleon weights not downloaded")
def test_real_chemeleon_checkpoint_loads_and_matches_the_recorded_hash():
    f = t.load_foundation(WEIGHTS)
    assert f["sha256"] == WEIGHTS_SHA256 == d.sha256(WEIGHTS)
    encoder = t.make_encoder(f["hyper_parameters"], f["state_dict"])
    assert sum(p.numel() for p in encoder.parameters()) == 8_714_240
    assert encoder.output_dim == 2048
    model = t.build_model("stage2", encoder, _scaler(1))
    assert model.agg.__class__.__name__ == "MeanAggregation"


# ------------------------------------------------------------------------------ real data
@needs_real
def test_real_frames_counts_and_monitor_draw(tmp_path):
    frames = d.load_frames(V1_DIR)
    assert len(frames["train"]) == 10_819 and len(frames["validation"]) == 621
    assert int((frames["train"]["role"] == "primary_train").sum()) == 2_897
    assert d.sha256(ASSIGNMENTS) == d.FROZEN_SPLIT_SHA256
    frame = d.attach_groups(frames["train"], frames["eligibility"], pd.read_csv(ASSIGNMENTS))
    assert int((frame["n_aux"] > 0).sum()) == 9_847
    assert {t_: int(frame[t_].notna().sum()) for t_ in d.AUX_TASKS} == {
        "log2fc_8p25uM": 9_737, "log2fc_33uM": 8_540, "log2fc_99uM": 606}
    chosen = d.draw_monitor_groups(frame)
    assert chosen == d.draw_monitor_groups(frame)
    assert 417 not in chosen
    train_f, mon_f = d.stage1_frames(frame, set(chosen))
    assert len(train_f) + len(mon_f) == 9_847
    # monitor molecules are training-pool molecules, never validation molecules
    assert set(mon_f["row_id"]).isdisjoint(frames["validation"]["row_id"])
    # artifact round trip, with no overwrite
    out = tmp_path / "monitor"
    counts = d.build_monitor_artifact(V1_DIR, ASSIGNMENTS, out)
    assert counts["monitor_molecules_labelled"] == len(mon_f)
    assert d.load_monitor(out, frame) == set(chosen)
    with pytest.raises(RuntimeError):
        d.build_monitor_artifact(V1_DIR, ASSIGNMENTS, out)
