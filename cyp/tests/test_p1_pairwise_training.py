from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import p1_pairwise
from cyp002_config import CYP002_TASK_NAMES
from p1_pairwise import _ordered_distinct_pair_from_index
from p1_pairwise_model import P1AntisymmetricPairHead
from p1_pairwise_training import (
    P1_ACCEPTED_PAIR_BATCH_SIZE,
    P1_FINAL_LR,
    P1_INITIAL_LR,
    P1_MAX_LR,
    P1_MAX_UPDATES,
    P1_UPDATES_PER_PSEUDO_EPOCH,
    P1_VALIDATION_INTERVAL_UPDATES,
    P1_WARMUP_UPDATES,
    P1ValidationController,
    accounting_after_batch,
    build_p1_optimizer_and_scheduler,
    build_pairwise_contract_from_training_labels,
    draw_p1_training_pair_batch,
    endpoint_weights_tensor,
    p1_forward_loss_backward_step,
    provenance_from_contract,
    pseudo_epoch_for_update,
    reconstruct_and_score_p1_validation,
    should_validate_after_update,
    tensors_from_pair_batch,
)


def _training_fixture():
    molecule_ids = ["a", "b", "c"]
    labels = [
        [1.0, 10.0, 100.0, 1000.0],
        [2.0, 20.0, None, 2000.0],
        [3.0, None, 300.0, 3000.0],
    ]
    means = (2.0, 15.0, 200.0, 2000.0)
    scales = (1.0, 5.0, 100.0, 1000.0)
    contract = build_pairwise_contract_from_training_labels(
        molecule_ids=molecule_ids,
        labels=labels,
        scaler_means=means,
        scaler_scales=scales,
    )
    return molecule_ids, labels, means, scales, contract


def _patch_pair_draws(monkeypatch, values):
    iterator = iter(values)

    def draw(_rng, pair_count):
        value = next(iterator)
        assert 0 <= value < pair_count
        return value

    monkeypatch.setattr(p1_pairwise, "_draw_ordered_pair_index", draw)


def test_training_pair_boundaries_weights_and_protected_exclusion(monkeypatch):
    molecule_ids, labels, means, scales, contract = _training_fixture()
    protected_test_name = "protected_test"
    assert protected_test_name not in contract.molecule_ids

    _patch_pair_draws(monkeypatch, [0, 1, 3, 4])
    batch = draw_p1_training_pair_batch(
        contract=contract,
        molecule_ids=molecule_ids,
        labels=labels,
        scaler_means=means,
        scaler_scales=scales,
        rng=np.random.default_rng(1),
        accepted_pair_count=4,
    )

    assert set(batch.molecule_a_ids).issubset(contract.molecule_ids)
    assert set(batch.molecule_b_ids).issubset(contract.molecule_ids)
    assert protected_test_name not in batch.molecule_a_ids
    assert protected_test_name not in batch.molecule_b_ids
    assert batch.accepted_count == 4
    assert batch.eligibility_masks[0] == (True, True, False, True)
    assert batch.eligibility_masks[1] == (True, False, True, True)

    e1, e2, e3, e4 = contract.endpoints
    assert (e1.n_labeled, e2.n_labeled, e3.n_labeled, e4.n_labeled) == (3, 2, 2, 3)
    assert (e1.weight, e2.weight, e3.weight, e4.weight) == (1.0, 2.0, 2.0, 1.0)
    weights = endpoint_weights_tensor(contract)
    torch.testing.assert_close(weights, torch.tensor([1.0, 2.0, 2.0, 1.0]))


def test_one_real_forward_weighted_loss_backward_optimizer_step(monkeypatch):
    molecule_ids, labels, means, scales, contract = _training_fixture()
    draws = [0, 1, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4]
    _patch_pair_draws(monkeypatch, draws)
    batch = draw_p1_training_pair_batch(
        contract=contract,
        molecule_ids=molecule_ids,
        labels=labels,
        scaler_means=means,
        scaler_scales=scales,
        rng=np.random.default_rng(2),
    )
    assert batch.accepted_count == P1_ACCEPTED_PAIR_BATCH_SIZE

    torch.manual_seed(20261107)
    embedding_table = {
        molecule_id: torch.randn(600)
        for molecule_id in contract.molecule_ids
    }
    h_a, h_b, targets, mask = tensors_from_pair_batch(batch, embedding_table)

    pair_head = P1AntisymmetricPairHead()
    optimizer, scheduler = build_p1_optimizer_and_scheduler(pair_head.parameters())
    weights = endpoint_weights_tensor(contract, dtype=h_a.dtype)
    before = [parameter.detach().clone() for parameter in pair_head.parameters()]
    loss, predictions = p1_forward_loss_backward_step(
        pair_head=pair_head,
        optimizer=optimizer,
        scheduler=scheduler,
        h_a=h_a,
        h_b=h_b,
        targets=targets,
        mask=mask,
        weights=weights,
    )

    assert torch.isfinite(loss)
    assert predictions.shape == (P1_ACCEPTED_PAIR_BATCH_SIZE, 4)
    assert torch.isfinite(predictions).all()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, pair_head.parameters(), strict=True)
    )
    assert all(torch.isfinite(parameter).all() for parameter in pair_head.parameters())


def test_schedule_accounting_and_noam_updates_in_optimizer_units():
    schedule_param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer, scheduler = build_p1_optimizer_and_scheduler([schedule_param])

    assert optimizer.param_groups[0]["lr"] == pytest.approx(P1_INITIAL_LR)
    assert P1_UPDATES_PER_PSEUDO_EPOCH == 235
    assert P1_WARMUP_UPDATES == 2350
    assert P1_MAX_UPDATES == 11750
    assert P1_VALIDATION_INTERVAL_UPDATES == 235
    assert should_validate_after_update(234) is False
    assert should_validate_after_update(235) is True
    assert pseudo_epoch_for_update(1) == 1
    assert pseudo_epoch_for_update(235) == 1
    assert pseudo_epoch_for_update(236) == 2

    for _ in range(P1_WARMUP_UPDATES):
        optimizer.zero_grad()
        schedule_param.backward(torch.ones_like(schedule_param))
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(P1_MAX_LR)

    for _ in range(P1_MAX_UPDATES - P1_WARMUP_UPDATES):
        optimizer.zero_grad()
        schedule_param.backward(torch.ones_like(schedule_param))
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(P1_FINAL_LR)


def test_training_reference_only_reconstruction_before_st_rae():
    predicted_delta_z = torch.zeros((2, 3, 4), dtype=torch.float64)
    reference_y = torch.tensor(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [3.0, 30.0, 300.0, 3000.0],
            [5.0, 50.0, 500.0, 5000.0],
        ],
        dtype=torch.float64,
    )
    reference_mask = torch.ones((3, 4), dtype=torch.bool)
    scaler_scales = torch.ones(4, dtype=torch.float64)
    validation_y_true = {
        task: np.array([2.0, 4.0], dtype=np.float64)
        for task in CYP002_TASK_NAMES
    }
    validation_conf_low = {
        task: np.array([1.5, 3.5], dtype=np.float64)
        for task in CYP002_TASK_NAMES
    }
    validation_conf_high = {
        task: np.array([2.5, 4.5], dtype=np.float64)
        for task in CYP002_TASK_NAMES
    }
    result = reconstruct_and_score_p1_validation(
        predicted_delta_z=predicted_delta_z,
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
        validation_y_true=validation_y_true,
        validation_conf_low=validation_conf_low,
        validation_conf_high=validation_conf_high,
    )

    assert np.isfinite(result["macro_st_rae"])
    assert result["reference_dispersion_diagnostic_shape"] == (2, 4)


def test_validation_controller_best_checkpoint_and_patience20():
    controller = P1ValidationController(patience=20)
    dummy_state = {"weight": torch.tensor([1.0])}

    assert controller.observe(
        update=235,
        validation_result={"macro_st_rae": 1.0},
        state_dict=dummy_state,
    ) is True
    assert controller.observe(
        update=470,
        validation_result={"macro_st_rae": 0.9},
        state_dict={"weight": torch.tensor([2.0])},
    ) is True
    assert controller.observe(
        update=705,
        validation_result={"macro_st_rae": 0.9},
        state_dict={"weight": torch.tensor([3.0])},
    ) is False
    assert controller.best_score == pytest.approx(0.9)
    assert controller.best_update == 470
    torch.testing.assert_close(controller.best_state_dict["weight"], torch.tensor([2.0]))

    for index in range(19):
        controller.observe(
            update=940 + index * 235,
            validation_result={"macro_st_rae": 0.91 + index * 0.001},
        )
    assert controller.bad_checks == 20
    assert controller.stopped is True


def test_deterministic_eval_antisymmetry_remains_intact():
    torch.manual_seed(20261107)
    pair_head = P1AntisymmetricPairHead()
    pair_head.eval()
    h_a = torch.randn(3, 600)
    h_b = torch.randn(3, 600)

    with torch.no_grad():
        ab = pair_head(h_a, h_b)
        ba = pair_head(h_b, h_a)
        aa = pair_head(h_a, h_a)

    torch.testing.assert_close(ab, -ba, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(aa, torch.zeros_like(aa), rtol=0.0, atol=0.0)


def test_malformed_nonfinite_states_fail_closed(monkeypatch):
    molecule_ids, labels, means, scales, contract = _training_fixture()
    labels[0][0] = np.inf
    with pytest.raises(ValueError, match="numeric and finite"):
        build_pairwise_contract_from_training_labels(
            molecule_ids=molecule_ids,
            labels=labels,
            scaler_means=means,
            scaler_scales=scales,
        )

    controller = P1ValidationController()
    with pytest.raises(ValueError, match="finite"):
        controller.observe(
            update=235,
            validation_result={"macro_st_rae": float("nan")},
        )

    _molecule_ids, labels, means, scales, contract = _training_fixture()
    _patch_pair_draws(monkeypatch, [0])
    batch = draw_p1_training_pair_batch(
        contract=contract,
        molecule_ids=_molecule_ids,
        labels=labels,
        scaler_means=means,
        scaler_scales=scales,
        rng=np.random.default_rng(3),
        accepted_pair_count=1,
    )
    embedding_table = {name: torch.zeros(600) for name in contract.molecule_ids}
    embedding_table[batch.molecule_a_ids[0]][0] = float("nan")
    with pytest.raises(ValueError, match="must contain only finite"):
        h_a, h_b, targets, mask = tensors_from_pair_batch(batch, embedding_table)
        pair_head = P1AntisymmetricPairHead()
        optimizer, scheduler = build_p1_optimizer_and_scheduler(pair_head.parameters())
        p1_forward_loss_backward_step(
            pair_head=pair_head,
            optimizer=optimizer,
            scheduler=scheduler,
            h_a=h_a,
            h_b=h_b,
            targets=targets,
            mask=mask,
            weights=endpoint_weights_tensor(contract),
        )


def test_training_provenance_and_accounting_record_required_fields(tmp_path, monkeypatch):
    molecule_ids, labels, means, scales, contract = _training_fixture()
    _patch_pair_draws(monkeypatch, [0, 1])
    batch = draw_p1_training_pair_batch(
        contract=contract,
        molecule_ids=molecule_ids,
        labels=labels,
        scaler_means=means,
        scaler_scales=scales,
        rng=np.random.default_rng(4),
        accepted_pair_count=2,
    )
    dataset_path = tmp_path / "dataset.csv"
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "t121.json"
    dataset_path.write_text("dataset\n", encoding="utf-8")
    manifest_path.write_text(json.dumps({"manifest": True}), encoding="utf-8")
    config_path.write_text(json.dumps({"sampled": True}), encoding="utf-8")

    pair_head = P1AntisymmetricPairHead()
    provenance = provenance_from_contract(
        dataset_path=dataset_path,
        split_manifest_path=manifest_path,
        t121_config_path=config_path,
        training_seed=20261101,
        scaler_means=means,
        scaler_scales=scales,
        contract=contract,
        modules={"pair_head": pair_head},
    )
    assert provenance.dataset_sha256
    assert provenance.split_manifest_sha256
    assert provenance.t121_config_sha256
    assert provenance.n_t == (3, 2, 2, 3)
    assert provenance.q_t[0] == pytest.approx(1.0)
    assert provenance.w_t == (1.0, 2.0, 2.0, 1.0)
    assert provenance.schedule.max_updates == 11750
    assert provenance.parameter_counts["pair_head"]["total"] > 0

    controller = P1ValidationController()
    controller.observe(update=235, validation_result={"macro_st_rae": 0.5})
    accounting = accounting_after_batch(
        update_count=1,
        validation_checks=1,
        batch=batch,
        contract=contract,
        controller=controller,
    )
    assert accounting.accepted_pair_rows == 2
    assert accounting.pair_attempts == 2
    assert accounting.best_validation_macro_st_rae == pytest.approx(0.5)
    assert accounting.weighted_target_contribution[1] == pytest.approx(
        accounting.eligible_target_count[1] * 2.0
    )
