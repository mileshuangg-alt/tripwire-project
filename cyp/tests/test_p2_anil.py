from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cyp002_config import CYP002_TASK_NAMES
from cyp007_infrastructure import D007_TRAINING_SEEDS, config_from_trial, sha256_file
from p2_anil_cell_artifacts import (
    BEST_CHECKPOINT_NAME,
    MANIFEST_NAME,
    METRICS_NAME,
    SHA256SUMS_NAME,
    P2CellIdentity,
    build_checkpoint_state,
    build_p2_cell_manifest,
    create_p2_cell_artifact_dir,
    finalize_p2_cell_manifest,
    load_checkpoint_state,
    reload_p2_complete_model_from_checkpoint,
    reload_p2_meta_head_from_checkpoint,
    save_selected_checkpoint,
    verify_p2_cell_inventory,
    write_metrics,
    write_planned_cell_manifest,
)
from p2_anil_model import (
    P2MetaHead,
    P2_DROPOUT,
    P2_EMBEDDING_DIM,
    P2_HIDDEN_DIM,
    adapt_selected_output_exact,
    build_p2_aggregation,
    build_p2_message_passing,
    build_p2_meta_head,
    query_loss_after_exact_adaptation,
    selected_task_prediction,
    standardized_mse,
)
from p2_anil_training import (
    P2_FINAL_LR,
    P2_INITIAL_LR,
    P2_MAX_LR,
    P2_MAX_UPDATES,
    P2_OUTER_UPDATES_PER_PSEUDO_EPOCH,
    P2_QUERY_COUNT,
    P2_SUPPORT_COUNT,
    P2_TASK_EPISODES_PER_PSEUDO_EPOCH,
    P2_VALIDATION_INTERVAL_UPDATES,
    P2_WARMUP_UPDATES,
    P2ValidationController,
    build_p2_optimizer_and_scheduler,
    draw_p2_episodes,
    enumerate_p2_cell_keys,
    fit_training_only_scaler,
    full_support_task_adaptation_prediction,
    p2_exact_outer_update_step,
    p2_outer_loss_from_episode_embeddings,
    p2_strict_o150_advancement,
    should_validate_after_update,
)

from p1_pairwise_real_workload_benchmark import build_training_reference_dataset
from p2_anil_benchmark import (
    EXPECTED_COMPLETE_MODEL_PARAMETERS,
    _complete_model_parameter_counts,
    p2_complete_model_outer_update_step,
)
from p2_anil_production import (
    EXPECTED_CELL_COUNT,
    P2CellKey,
    aggregate_completed_p2_cells,
    classify_cell,
    enumerate_p2_cells as enumerate_production_p2_cells,
    p2_cell_dir,
    p2_cell_id,
    preflight_p2_production,
    run_p2_cell,
)


T121_SAMPLED = {
    "batch_size": 16,
    "dropout": 0.4,
    "ffn_hidden_dim": 2000,
    "ffn_num_layers": 1,
    "final_lr_ratio": 0.24427727346003839,
    "init_lr_ratio": 0.017323283256612185,
    "max_lr": 0.0008245400010426866,
    "message_hidden_dim": 600,
    "message_passing_depth": 4,
    "warmup_epochs": 10,
}


def _config(seed=20261101):
    return config_from_trial(name="D010-P2-test", sampled=T121_SAMPLED, seed=seed)


def _synthetic_partition(n=80):
    molecule_ids = [f"m{i:03d}" for i in range(n)]
    labels = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        labels[i] = [1.0 + i * 0.1, 2.0 + i * 0.2, 3.0 + i * 0.3, 4.0 + i * 0.4]
    mask = np.ones((n, 4), dtype=bool)
    return molecule_ids, labels, mask


def _embedding_table(molecule_ids, *, seed=7):
    generator = torch.Generator().manual_seed(seed)
    return {name: torch.randn(P2_EMBEDDING_DIM, generator=generator) for name in molecule_ids}


def _real_graph_training_dataset(n=20):
    rows = []
    for index in range(n):
        row = {"Molecule_Name": f"m{index:03d}", "SMILES": "C" * (index % 8 + 1)}
        for task_index, task in enumerate(CYP002_TASK_NAMES):
            row[task] = float(index + task_index + 1)
        rows.append(row)
    dataset, *_ = build_training_reference_dataset(pd.DataFrame(rows))
    labels = np.asarray([[row[task] for task in CYP002_TASK_NAMES] for row in rows], dtype=np.float64)
    mask = np.ones_like(labels, dtype=bool)
    names = [row["Molecule_Name"] for row in rows]
    return dataset, names, labels, mask


def _write_toy_production_inputs(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "cyp" / "artifacts" / "cyp_baseline" / "molecule_split_assignments.csv"
    source.parent.mkdir(parents=True)
    train_names = [f"tr{i:02d}" for i in range(20)]
    validation_names = [f"va{i:02d}" for i in range(4)]
    test_names = ["protected"]
    source.write_text(
        "Molecule_Name,cluster_id,split\n"
        + "".join(f"{name},{i},train\n" for i, name in enumerate(train_names))
        + "".join(f"{name},{100+i},validation\n" for i, name in enumerate(validation_names))
        + "protected,999,test\n",
        encoding="utf-8",
    )
    rows = []
    for index, name in enumerate([*train_names, *validation_names, *test_names]):
        row = {"Molecule_Name": name, "SMILES": "C" * (index % 8 + 1)}
        for task_index, task in enumerate(CYP002_TASK_NAMES):
            value = float(index + task_index + 1)
            row[task] = value
            row[f"{task}_conf_low"] = value - 0.5
            row[f"{task}_conf_high"] = value + 0.5
        rows.append(row)
    dataset_path = tmp_path / "toy_cyp.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for repetition in range(1, 6):
        for fold in range(1, 11):
            manifest = {
                "schema_version": "D007-split-manifest-v1",
                "repetition": repetition,
                "fold": fold,
                "training_seeds": list(D007_TRAINING_SEEDS),
                "source_split_assignments": "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv",
                "source_split_assignments_sha256": sha256_file(source),
                "training_names": train_names,
                "validation_names": validation_names,
                "training_count": len(train_names),
                "validation_count": len(validation_names),
                "protected_test_count_in_partitions": 0,
            }
            (split_dir / f"rep_{repetition:02d}_fold_{fold:02d}.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
    t121 = tmp_path / "t121.json"
    t121.write_text(json.dumps({"sampled_config": T121_SAMPLED}), encoding="utf-8")
    benchmark = Path("cyp/artifacts/cyp_010/p2_d010_complete_model_benchmark_rep01_fold01_seed20261101_warm10_measured100.json")
    return repo, dataset_path, split_dir, t121, benchmark


class SpyMessagePassing(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, batch_mol_graph):
        self.calls += 1
        return self.inner(batch_mol_graph)


def _episodes_and_embeddings(seed=11):
    molecule_ids, labels, mask = _synthetic_partition()
    means, scales = fit_training_only_scaler(labels, mask)
    episodes = draw_p2_episodes(
        molecule_ids=molecule_ids,
        labels=labels,
        mask=mask,
        scaler_means=means,
        scaler_scales=scales,
        rng=np.random.default_rng(seed),
    )
    return episodes, _embedding_table(molecule_ids), means, scales, labels, mask, molecule_ids


def test_frozen_task_order_architecture_and_schedule_mapping():
    assert CYP002_TASK_NAMES == (
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    )
    config = _config()
    message_passing = build_p2_message_passing(config)
    aggregation = build_p2_aggregation(config)
    meta_head = build_p2_meta_head(config)
    assert message_passing.depth == 4
    assert aggregation.norm == 100
    assert meta_head.shared_hidden.in_features == 600
    assert meta_head.shared_hidden.out_features == 2000
    assert meta_head.output.out_features == 4
    assert meta_head.dropout.p == pytest.approx(P2_DROPOUT)
    assert P2_OUTER_UPDATES_PER_PSEUDO_EPOCH == 235
    assert P2_TASK_EPISODES_PER_PSEUDO_EPOCH == 940
    assert P2_WARMUP_UPDATES == 2350
    assert P2_MAX_UPDATES == 11750
    assert P2_VALIDATION_INTERVAL_UPDATES == 235
    assert should_validate_after_update(234) is False
    assert should_validate_after_update(235) is True
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer, scheduler = build_p2_optimizer_and_scheduler([param])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(P2_INITIAL_LR)
    for _ in range(P2_WARMUP_UPDATES):
        optimizer.zero_grad()
        param.backward(torch.ones_like(param))
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(P2_MAX_LR)
    for _ in range(P2_MAX_UPDATES - P2_WARMUP_UPDATES):
        optimizer.zero_grad()
        param.backward(torch.ones_like(param))
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(P2_FINAL_LR)


def test_episode_construction_firewall_accounting_no_imputation_and_fail_closed():
    molecule_ids, labels, mask = _synthetic_partition()
    validation_names = {"validation_only"}
    protected_names = {"protected_test"}
    d008_names = {"d008_observed"}
    means, scales = fit_training_only_scaler(labels, mask)
    episodes = draw_p2_episodes(
        molecule_ids=molecule_ids,
        labels=labels,
        mask=mask,
        scaler_means=means,
        scaler_scales=scales,
        rng=np.random.default_rng(1),
    )
    assert tuple(ep.task_name for ep in episodes) == CYP002_TASK_NAMES
    assert len(episodes) == 4
    for ep in episodes:
        assert len(ep.support_molecule_ids) == P2_SUPPORT_COUNT == 13
        assert len(ep.query_molecule_ids) == P2_QUERY_COUNT == 3
        assert set(ep.support_molecule_ids).isdisjoint(ep.query_molecule_ids)
        all_ids = set(ep.support_molecule_ids) | set(ep.query_molecule_ids)
        assert all_ids <= set(molecule_ids)
        assert all_ids.isdisjoint(validation_names | protected_names | d008_names)
        assert len(ep.support_targets_z) == 13
        assert len(ep.query_targets_z) == 3
        assert np.isfinite(ep.support_targets_z).all()
        assert np.isfinite(ep.query_targets_z).all()
    labels_bad = labels.copy()
    labels_bad[0, 0] = np.nan
    mask_bad = mask.copy()
    mask_bad[:, 0] = False
    mask_bad[:15, 0] = True
    with pytest.raises(ValueError):
        fit_training_only_scaler(labels_bad, mask_bad)
    with pytest.raises(ValueError):
        draw_p2_episodes(
            molecule_ids=molecule_ids,
            labels=labels,
            mask=mask_bad,
            scaler_means=means,
            scaler_scales=scales,
            rng=np.random.default_rng(2),
        )
    with pytest.raises(ValueError):
        draw_p2_episodes(
            molecule_ids=[*molecule_ids, molecule_ids[0]],
            labels=np.vstack([labels, labels[0]]),
            mask=np.vstack([mask, mask[0]]),
            scaler_means=means,
            scaler_scales=scales,
            rng=np.random.default_rng(3),
        )


def test_exact_gradient_differs_from_deliberately_detached_first_order_witness():
    torch.manual_seed(10)
    head_exact = P2MetaHead().double()
    head_first_order = P2MetaHead().double()
    head_first_order.load_state_dict(head_exact.state_dict())
    head_exact.train()
    head_first_order.train()
    support_x = torch.randn(13, P2_EMBEDDING_DIM, dtype=torch.float64, requires_grad=True)
    query_x = torch.randn(3, P2_EMBEDDING_DIM, dtype=torch.float64, requires_grad=True)
    support_y = torch.linspace(-0.7, 0.9, 13, dtype=torch.float64)
    query_y = torch.tensor([-0.2, 0.4, 0.8], dtype=torch.float64)
    torch.manual_seed(123)
    exact_loss, _support_loss = query_loss_after_exact_adaptation(
        head_exact,
        support_x,
        support_y,
        query_x,
        query_y,
        task_index=0,
    )
    exact_grad = torch.autograd.grad(exact_loss, head_exact.shared_hidden.weight, retain_graph=True)[0]
    init_exact_grad = torch.autograd.grad(exact_loss, head_exact.output.weight, retain_graph=True)[0][0]
    torch.manual_seed(123)
    support_features = head_first_order.features(support_x)
    selected_weight = head_first_order.output.weight[0]
    selected_bias = head_first_order.output.bias[0]
    support_pred = selected_task_prediction(
        support_features,
        task_index=0,
        weight=selected_weight,
        bias=selected_bias,
    )
    support_loss = standardized_mse(support_pred, support_y)
    grad_w, grad_b = torch.autograd.grad(
        support_loss,
        (selected_weight, selected_bias),
        create_graph=False,
    )
    assert grad_w.grad_fn is None
    assert grad_b.grad_fn is None
    adapted_w = selected_weight - 0.01 * grad_w.detach()
    adapted_b = selected_bias - 0.01 * grad_b.detach()
    query_features = head_first_order.features(query_x)
    first_order_loss = standardized_mse(
        selected_task_prediction(query_features, task_index=0, weight=adapted_w, bias=adapted_b),
        query_y,
    )
    first_order_grad = torch.autograd.grad(first_order_loss, head_first_order.shared_hidden.weight, retain_graph=True)[0]
    assert torch.isfinite(exact_grad).all()
    assert torch.isfinite(init_exact_grad).all()
    assert not torch.allclose(exact_grad, first_order_grad, rtol=1e-7, atol=1e-9)
    assert exact_loss.grad_fn is not None


def test_inner_adaptation_boundary_outer_gradients_dropout_and_one_optimizer_step():
    episodes, embeddings, *_ = _episodes_and_embeddings()
    head = P2MetaHead()
    head.train()
    before = {name: value.detach().clone() for name, value in head.state_dict().items()}
    ep = episodes[0]
    support_x = torch.stack([embeddings[name] for name in ep.support_molecule_ids])
    support_y = torch.tensor(ep.support_targets_z, dtype=support_x.dtype)
    adapted_w, adapted_b, support_loss = adapt_selected_output_exact(head, support_x, support_y, task_index=0)
    assert support_loss.grad_fn is not None
    after_inner = head.state_dict()
    for name, old in before.items():
        torch.testing.assert_close(after_inner[name], old)
    assert not torch.equal(adapted_w.detach(), head.output.weight[0].detach())
    assert not torch.equal(adapted_b.detach().reshape(()), head.output.bias[0].detach().reshape(()))
    optimizer, scheduler = build_p2_optimizer_and_scheduler(head.parameters())
    calls = {"optimizer": 0, "scheduler": 0}
    old_step = optimizer.step
    old_sched = scheduler.step

    def opt_step(*args, **kwargs):
        calls["optimizer"] += 1
        return old_step(*args, **kwargs)

    def sched_step(*args, **kwargs):
        calls["scheduler"] += 1
        return old_sched(*args, **kwargs)

    optimizer.step = opt_step
    scheduler.step = sched_step
    result = p2_exact_outer_update_step(
        meta_head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        episodes=episodes,
        embedding_table=embeddings,
    )
    assert calls == {"optimizer": 1, "scheduler": 1}
    assert len(result.query_losses) == 4
    torch.testing.assert_close(result.outer_loss, sum(result.query_losses) / 4)
    assert head.shared_hidden.weight.grad is not None
    assert head.output.weight.grad is not None
    assert torch.isfinite(head.shared_hidden.weight.grad).all()
    assert torch.isfinite(head.output.weight.grad).all()
    assert head.training is True
    assert head.dropout.training is True
    x = torch.stack([embeddings[episodes[1].support_molecule_ids[0]]] * 4)
    head.train()
    torch.manual_seed(4)
    train_a = head.features(x)
    torch.manual_seed(5)
    train_b = head.features(x)
    assert not torch.equal(train_a, train_b)
    head.eval()
    eval_a = head.features(x)
    eval_b = head.features(x)
    torch.testing.assert_close(eval_a, eval_b)


def test_evaluation_full_support_adaptation_fresh_state_and_minibatch_mean_equivalence():
    torch.manual_seed(14)
    head = P2MetaHead().double()
    head.eval()
    train_x = torch.randn(25, P2_EMBEDDING_DIM, dtype=torch.float64, requires_grad=True)
    val_x = torch.randn(5, P2_EMBEDDING_DIM, dtype=torch.float64)
    labels = torch.randn(25, 4, dtype=torch.float64)
    mask = torch.ones(25, 4, dtype=torch.bool)
    means = labels.mean(dim=0)
    scales = labels.std(dim=0, unbiased=False).clamp_min(0.1)
    state_before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    pred_full = full_support_task_adaptation_prediction(
        meta_head=head,
        train_embeddings=train_x,
        train_labels=labels,
        train_mask=mask,
        validation_embeddings=val_x,
        scaler_means=means,
        scaler_scales=scales,
        task_index=2,
        support_batch_size=None,
    )
    pred_batched = full_support_task_adaptation_prediction(
        meta_head=head,
        train_embeddings=train_x,
        train_labels=labels,
        train_mask=mask,
        validation_embeddings=val_x,
        scaler_means=means,
        scaler_scales=scales,
        task_index=2,
        support_batch_size=6,
    )
    torch.testing.assert_close(pred_full, pred_batched)
    for name, old in state_before.items():
        torch.testing.assert_close(head.state_dict()[name], old)
    pred_task1 = full_support_task_adaptation_prediction(
        meta_head=head,
        train_embeddings=train_x,
        train_labels=labels,
        train_mask=mask,
        validation_embeddings=val_x,
        scaler_means=means,
        scaler_scales=scales,
        task_index=1,
    )
    assert pred_task1.shape == pred_full.shape == (5,)
    for name, old in state_before.items():
        torch.testing.assert_close(head.state_dict()[name], old)


def test_validation_authority_strict_best_and_o150_advancement():
    controller = P2ValidationController(patience=20)
    state = {"p": torch.tensor([1.0])}
    assert controller.observe(update=235, validation_result={"macro_st_rae": 0.7}, state_dict=state)
    assert not controller.observe(update=470, validation_result={"macro_st_rae": 0.7}, state_dict=state)
    assert controller.best_score == pytest.approx(0.7)
    assert controller.best_update == 235
    scores = [0.701212576796] * 150
    tie = p2_strict_o150_advancement(scores)
    assert tie["p2_advances"] is False
    scores[0] -= 0.150000000001
    win = p2_strict_o150_advancement(scores)
    assert win["p2_advances"] is True
    with pytest.raises(ValueError):
        p2_strict_o150_advancement([0.7] * 149)
    with pytest.raises(ValueError):
        p2_strict_o150_advancement([0.7] * 149 + [float("nan")])


def test_deterministic_150_cell_routing():
    cells = enumerate_p2_cell_keys()
    assert len(cells) == 150
    assert len(set(cells)) == 150
    assert cells[0].repetition == 1
    assert cells[0].fold == 1
    assert cells[0].training_seed == D007_TRAINING_SEEDS[0]
    assert cells[-1].repetition == 5
    assert cells[-1].fold == 10
    assert cells[-1].training_seed == D007_TRAINING_SEEDS[-1]


def _identity(tmp_path: Path) -> P2CellIdentity:
    dataset = tmp_path / "dataset.csv"
    split = tmp_path / "split.json"
    for path in (dataset, split):
        path.write_text(path.name, encoding="utf-8")
    return P2CellIdentity(
        repetition=1,
        fold=1,
        training_seed=20261101,
        split_manifest_path=str(split),
        split_manifest_sha256=sha256_file(split),
        dataset_path=str(dataset),
        dataset_sha256=sha256_file(dataset),
    )


def test_artifact_checkpoint_reload_inference_equivalence_and_corruption_fail_closed(tmp_path):
    identity = _identity(tmp_path)
    config = _config()
    p2_config = {"model": "D010-P2", "exact_anil": True, "inner_lr": 0.01}
    cell_dir = create_p2_cell_artifact_dir(tmp_path / "cell")
    manifest = build_p2_cell_manifest(
        cell_identity=identity,
        p2_config=p2_config,
        output_dir=cell_dir,
        environment={"synthetic": True},
        provenance={"d010": "synthetic qualification"},
    )
    write_planned_cell_manifest(manifest, cell_dir / MANIFEST_NAME)
    from p2_anil_model import build_p2_message_passing, build_p2_aggregation

    message_passing = build_p2_message_passing(config)
    aggregation = build_p2_aggregation(config)
    head = build_p2_meta_head(config)
    controller = P2ValidationController()
    controller.observe(update=235, validation_result={"macro_st_rae": 0.5}, state_dict=head.state_dict())
    checkpoint = build_checkpoint_state(
        message_passing=message_passing,
        aggregation=aggregation,
        meta_head=head,
        optimizer=None,
        scheduler=None,
        validation_controller=controller,
        cell_identity=identity,
        p2_config=p2_config,
        scaler_metadata={"means": [0, 0, 0, 0], "scales": [1, 1, 1, 1]},
        update_state={"updates": 235},
        episode_state={"episodes_per_update": 4},
    )
    save_selected_checkpoint(checkpoint, cell_dir / BEST_CHECKPOINT_NAME)
    write_metrics({"best_validation": {"macro_st_rae": 0.5}}, cell_dir / METRICS_NAME)
    finalize_p2_cell_manifest(cell_dir)
    assert {MANIFEST_NAME, BEST_CHECKPOINT_NAME, METRICS_NAME}.issubset(verify_p2_cell_inventory(cell_dir))
    reloaded = reload_p2_meta_head_from_checkpoint(
        cell_dir / BEST_CHECKPOINT_NAME,
        expected_cell_identity=identity,
        expected_p2_config=p2_config,
        config=config,
    )
    x = torch.randn(4, P2_EMBEDDING_DIM)
    head.eval()
    reloaded.eval()
    torch.testing.assert_close(head(x), reloaded(x))
    reloaded_encoder, reloaded_aggregation, reloaded_head = reload_p2_complete_model_from_checkpoint(
        cell_dir / BEST_CHECKPOINT_NAME,
        expected_cell_identity=identity,
        expected_p2_config=p2_config,
        config=config,
    )
    assert reloaded_encoder.state_dict().keys() == message_passing.state_dict().keys()
    assert reloaded_aggregation.state_dict().keys() == aggregation.state_dict().keys()
    reloaded_head.eval()
    torch.testing.assert_close(reloaded_head(x), head(x))
    bad_identity = P2CellIdentity(**{**identity.__dict__, "fold": 2})
    with pytest.raises(ValueError):
        load_checkpoint_state(
            cell_dir / BEST_CHECKPOINT_NAME,
            expected_cell_identity=bad_identity,
            expected_p2_config=p2_config,
        )
    (cell_dir / SHA256SUMS_NAME).write_text("bad  best.ckpt\n", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_p2_cell_inventory(cell_dir)


def test_complete_model_parameter_count_optimizer_and_real_graph_outer_update_binding():
    torch.manual_seed(20261109)
    dataset, molecule_ids, labels, mask = _real_graph_training_dataset()
    means, scales = fit_training_only_scaler(labels, mask)
    episodes = draw_p2_episodes(
        molecule_ids=molecule_ids,
        labels=labels,
        mask=mask,
        scaler_means=means,
        scaler_scales=scales,
        rng=np.random.default_rng(20261109),
    )
    config = _config()
    from p2_anil_model import build_p2_message_passing, build_p2_aggregation

    message_passing = SpyMessagePassing(build_p2_message_passing(config))
    aggregation = build_p2_aggregation(config)
    head = build_p2_meta_head(config)
    counts = _complete_model_parameter_counts(message_passing, aggregation, head)
    assert counts["encoder"] == {"total": 815400, "trainable": 815400}
    assert counts["aggregation"] == {"total": 0, "trainable": 0}
    assert counts["shared_hidden"] == {"total": 1202000, "trainable": 1202000}
    assert counts["output_layer"] == {"total": 8004, "trainable": 8004}
    assert counts["meta_head"] == {"total": 1210004, "trainable": 1210004}
    assert counts["complete_p2_model"] == {
        "total": EXPECTED_COMPLETE_MODEL_PARAMETERS,
        "trainable": EXPECTED_COMPLETE_MODEL_PARAMETERS,
    }
    optimizer, scheduler = build_p2_optimizer_and_scheduler(
        list(message_passing.parameters()) + list(head.parameters())
    )
    optimizer_param_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert all(id(parameter) in optimizer_param_ids for parameter in message_passing.parameters())
    assert all(id(parameter) in optimizer_param_ids for parameter in head.parameters())
    before_calls = message_passing.calls
    result = p2_complete_model_outer_update_step(
        message_passing=message_passing,
        aggregation=aggregation,
        meta_head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        episodes=episodes,
        train_dataset=dataset,
        name_to_index={name: index for index, name in enumerate(molecule_ids)},
        device=torch.device("cpu"),
        batch_size=16,
    )
    assert message_passing.calls > before_calls
    assert len(result["query_losses"]) == 4
    torch.testing.assert_close(result["outer_loss"], sum(result["query_losses"]) / 4)
    assert message_passing.inner.W_i.weight.grad is not None
    assert torch.isfinite(message_passing.inner.W_i.weight.grad).all()
    assert head.shared_hidden.weight.grad is not None
    assert torch.isfinite(head.shared_hidden.weight.grad).all()


def test_complete_model_real_graph_path_has_no_synthetic_embedding_table(monkeypatch):
    import p2_anil_benchmark
    dataset, molecule_ids, labels, mask = _real_graph_training_dataset()
    means, scales = fit_training_only_scaler(labels, mask)
    episodes = draw_p2_episodes(
        molecule_ids=molecule_ids,
        labels=labels,
        mask=mask,
        scaler_means=means,
        scaler_scales=scales,
        rng=np.random.default_rng(20261110),
    )
    config = _config()
    from p2_anil_model import build_p2_message_passing, build_p2_aggregation

    message_passing = build_p2_message_passing(config)
    aggregation = build_p2_aggregation(config)
    head = build_p2_meta_head(config)
    optimizer, scheduler = build_p2_optimizer_and_scheduler(
        list(message_passing.parameters()) + list(head.parameters())
    )

    def forbidden_randn(*args, **kwargs):
        raise AssertionError("synthetic embedding substitution is forbidden in complete-model path")

    monkeypatch.setattr(p2_anil_benchmark.torch, "randn", forbidden_randn)
    result = p2_anil_benchmark.p2_complete_model_outer_update_step(
        message_passing=message_passing,
        aggregation=aggregation,
        meta_head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        episodes=episodes,
        train_dataset=dataset,
        name_to_index={name: index for index, name in enumerate(molecule_ids)},
        device=torch.device("cpu"),
        batch_size=16,
    )
    assert torch.isfinite(result["outer_loss"])


def test_connected_toy_production_cell_runner_artifacts_reload_and_firewall(tmp_path):
    repo, dataset_path, split_dir, t121, benchmark = _write_toy_production_inputs(tmp_path)
    cell_dir = tmp_path / "p2_cell"
    result = run_p2_cell(
        output_dir=cell_dir,
        repository_root=repo,
        dataset_path=dataset_path,
        split_manifest_path=split_dir / "rep_01_fold_01.json",
        t121_config_path=t121,
        training_seed=20261101,
        device="cpu",
        max_updates=1,
        validation_interval_updates=1,
    )
    assert result["state"] == "complete"
    assert (cell_dir / MANIFEST_NAME).exists()
    assert (cell_dir / BEST_CHECKPOINT_NAME).exists()
    assert (cell_dir / METRICS_NAME).exists()
    assert (cell_dir / SHA256SUMS_NAME).exists()
    assert verify_p2_cell_inventory(cell_dir)
    manifest = json.loads((cell_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["execution"]["status"] == "completed"
    assert manifest["provenance"]["protected_internal_test_excluded"] is True
    assert manifest["provenance"]["d008_evidence_excluded"] is True
    metrics = json.loads((cell_dir / METRICS_NAME).read_text(encoding="utf-8"))
    assert np.isfinite(metrics["best_validation"]["macro_st_rae"])
    assert metrics["best_validation"]["update"] == 1
    config = _config()
    identity = P2CellIdentity(**manifest["cell_identity"])
    from p2_anil_cell_artifacts import reload_p2_complete_model_from_checkpoint

    encoder, aggregation, head = reload_p2_complete_model_from_checkpoint(
        cell_dir / BEST_CHECKPOINT_NAME,
        expected_cell_identity=identity,
        expected_p2_config=manifest["p2_config"],
        config=config,
    )
    assert encoder is not None
    assert aggregation is not None
    assert head is not None
    with pytest.raises(FileExistsError):
        run_p2_cell(
            output_dir=cell_dir,
            repository_root=repo,
            dataset_path=dataset_path,
            split_manifest_path=split_dir / "rep_01_fold_01.json",
            t121_config_path=t121,
            training_seed=20261101,
            device="cpu",
            max_updates=1,
            validation_interval_updates=1,
        )
    (cell_dir / SHA256SUMS_NAME).write_text("bad  best.ckpt\n", encoding="utf-8")
    state = classify_cell(tmp_path, P2CellKey(1, 1, 20261101))
    assert state.state == "absent"
    assert classify_cell(cell_dir.parent, P2CellKey(1, 1, 20261101)).state in {"absent", "incomplete/failed"}


def _write_synthetic_complete_p2_cell(root: Path, key: P2CellKey, tmp_path: Path, macro: float):
    from p2_anil_cell_artifacts import build_p2_cell_manifest, write_planned_cell_manifest, write_metrics, finalize_p2_cell_manifest

    cell_dir = p2_cell_dir(root, key)
    cell_dir.mkdir(parents=True)
    identity = P2CellIdentity(
        repetition=key.repetition,
        fold=key.fold,
        training_seed=key.training_seed,
        split_manifest_path=str(tmp_path / "split.json"),
        split_manifest_sha256="synthetic",
        dataset_path=str(tmp_path / "dataset.csv"),
        dataset_sha256="synthetic",
    )
    manifest = build_p2_cell_manifest(
        cell_identity=identity,
        p2_config={"model": "D010-P2", "synthetic": True},
        output_dir=cell_dir,
        environment={"synthetic": True},
        provenance={"synthetic_aggregation": True},
    )
    write_planned_cell_manifest(manifest, cell_dir / MANIFEST_NAME)
    (cell_dir / BEST_CHECKPOINT_NAME).write_text("synthetic checkpoint\n", encoding="utf-8")
    write_metrics(
        {"best_validation": {"macro_st_rae": macro, "per_task": {task: macro for task in CYP002_TASK_NAMES}}},
        cell_dir / METRICS_NAME,
    )
    finalize_p2_cell_manifest(cell_dir)


def test_p2_preflight_accounting_and_synthetic_o150_aggregation(tmp_path):
    repo, dataset_path, split_dir, t121, benchmark = _write_toy_production_inputs(tmp_path)
    production_root = tmp_path / "prod"
    report = preflight_p2_production(
        production_root=production_root,
        repository_root=repo,
        split_manifest_dir=split_dir,
        dataset_path=dataset_path,
        t121_config_path=t121,
        benchmark_artifact_path=benchmark,
    )
    assert report["split_manifest_count"] == 50
    assert report["cell_count"] == EXPECTED_CELL_COUNT == 150
    assert report["cell_state_counts"] == {"absent": 150, "complete": 0, "incomplete/failed": 0}
    assert len(report["execution_plan_absent_cells_only"]) == 150
    assert report["performance_driven_scheduling"] is False
    assert report["execution_topology"]["requested_walltime_minutes"] == 20
    cells = enumerate_production_p2_cells()
    assert len(cells) == 150
    assert p2_cell_id(P2CellKey(1, 2, 20261101)) == "rep_01/fold_02/seed_20261101"
    for cell in cells:
        _write_synthetic_complete_p2_cell(production_root, cell, tmp_path, 0.7)
    result = aggregate_completed_p2_cells(production_root)
    assert result["cell_count"] == 150
    assert result["o150"] == pytest.approx(0.7)
    assert result["p2_advances"] is True
    assert result["endpoint_veto"] is False
    first_metrics = json.loads((p2_cell_dir(production_root, cells[0]) / METRICS_NAME).read_text(encoding="utf-8"))
    first_metrics["best_validation"]["macro_st_rae"] = 0.701212576796
    (p2_cell_dir(production_root, cells[0]) / METRICS_NAME).write_text(json.dumps(first_metrics), encoding="utf-8")
    with pytest.raises(RuntimeError):
        aggregate_completed_p2_cells(production_root)


def test_malformed_nonfinite_incomplete_mismatched_inputs_fail_closed():
    head = P2MetaHead()
    with pytest.raises(ValueError):
        head(torch.randn(2, 599))
    with pytest.raises(ValueError):
        head(torch.tensor([[float("inf")] * P2_EMBEDDING_DIM]))
    with pytest.raises(ValueError):
        standardized_mse(torch.tensor([1.0]), torch.tensor([float("nan")]))
    episodes, embeddings, *_ = _episodes_and_embeddings()
    bad = list(episodes)
    bad[0] = type(episodes[0])(
        task_index=0,
        task_name=episodes[1].task_name,
        support_molecule_ids=episodes[0].support_molecule_ids,
        query_molecule_ids=episodes[0].query_molecule_ids,
        support_targets_z=episodes[0].support_targets_z,
        query_targets_z=episodes[0].query_targets_z,
    )
    with pytest.raises(ValueError):
        p2_outer_loss_from_episode_embeddings(meta_head=head, episodes=bad, embedding_table=embeddings)
    with pytest.raises(KeyError):
        missing = dict(embeddings)
        missing.pop(episodes[0].support_molecule_ids[0])
        p2_outer_loss_from_episode_embeddings(meta_head=head, episodes=episodes, embedding_table=missing)


def test_benchmark_cli_contract_and_no_validation_performance_language():
    from p2_anil_benchmark import NO_PERFORMANCE_STATEMENT, build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--repository-root",
            "/repo",
            "--dataset-path",
            "/repo/cyp.csv",
            "--split-manifest-path",
            "/repo/split.json",
            "--t121-config-artifact-path",
            "/repo/t121.json",
            "--training-seed",
            "20261101",
            "--warmup-update-count",
            "2",
            "--measured-update-count",
            "3",
            "--output-json",
            "/tmp/out.json",
            "--inner-lr",
            "0.01",
            "--inner-steps",
            "1",
        ]
    )
    assert args.repository_root == Path("/repo")
    assert args.training_seed == 20261101
    assert args.warmup_update_count == 2
    assert args.measured_update_count == 3
    assert args.inner_lr == pytest.approx(0.01)
    assert args.inner_steps == 1
    assert "does not access validation labels" in NO_PERFORMANCE_STATEMENT
    assert "calculate ST-RAE" in NO_PERFORMANCE_STATEMENT
