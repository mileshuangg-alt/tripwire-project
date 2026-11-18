import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cyp/scripts"))

import actfound_model_side
from actfound_production_train_v2 import run_task


class TinyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_dict = torch.nn.ModuleDict({
            "linear": torch.nn.ParameterDict({
                "weights": torch.nn.Parameter(torch.ones(3)),
                "bias": torch.nn.Parameter(torch.zeros(1)),
            })
        })


@pytest.fixture
def args():
    return SimpleNamespace(
        inverse_ylabel=False, second_order=False,
        first_order_to_second_order_epoch=-1, num_updates=2,
        test_num_updates=2, use_byhand_lr=False,
        use_multi_step_loss_optimization=False,
        multi_step_loss_num_epochs=15,
    )


@pytest.fixture
def seen(monkeypatch):
    calls = []

    def stub(regressor, x, y, split, *, weights, observation_weights=None, **kwargs):
        calls.append(observation_weights)
        loss = sum(value.sum() for value in weights.values()) ** 2
        return loss, torch.zeros(int((split == 0).sum()))

    monkeypatch.setattr(actfound_model_side, "actfound_forward_with_regressor", stub)
    return calls


def episode(weighted):
    x = np.zeros((12, 3), dtype=np.float32)
    y = np.arange(12, dtype=np.float32)
    split = np.array([1] * 8 + [0] * 4, dtype=np.int64)
    if not weighted:
        return (x, y, split)
    return (x, y, split, np.linspace(1.0, 2.0, 12).astype(np.float32),
            tuple(f"M{i}" for i in range(12)))


def call(args, arrays, training, flag=None):
    regressor = TinyRegressor()
    rates = torch.nn.Parameter(torch.full((3,), 0.01))
    kwargs = {} if flag is None else {"use_observation_weights": flag}
    run_task(regressor, rates, arrays, args, epoch=0,
             is_training_phase=training, **kwargs)


def test_default_outside_training_drops_weights(args, seen):
    call(args, episode(True), training=False)
    assert seen and all(w is None for w in seen)


def test_default_in_training_keeps_weights(args, seen):
    call(args, episode(True), training=True)
    assert seen and all(w is not None for w in seen)


def test_flag_applies_weights_outside_training(args, seen):
    call(args, episode(True), training=False, flag=True)
    assert seen and all(w is not None for w in seen)
    torch.testing.assert_close(
        seen[0], torch.as_tensor(np.linspace(1.0, 2.0, 12).astype(np.float32))
    )


def test_flag_with_legacy_arrays_stays_unweighted(args, seen):
    call(args, episode(False), training=False, flag=True)
    assert seen and all(w is None for w in seen)
