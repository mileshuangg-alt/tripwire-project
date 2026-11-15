from __future__ import annotations

import csv
import math
import random
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "cyp" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from actfound_model_side import (  # noqa: E402
    GRAPH_QUALIFIED_COLUMNS,
    PROVENANCE,
    ActFoundFrozenMetaDataset,
    actfound_loss_and_preds,
    actfound_forward_with_regressor,
    adapt_final_linear_params,
    adapt_final_linear_weight,
    construct_pair_matrices,
    continuous_inference,
    continuous_inference_from_outputs,
    iter_frozen_graph_qualified_assays,
    jaccard_count_sim_matrix,
    make_episode_split,
    robust_square_error,
    select_actfound_inner_loop_params,
)


def _write_graph_fixture(
    path: Path,
    y_values: list[float],
    *,
    assay_key: str = "11_nM_IC50",
    relation: str = "=",
    unit: str = "nM",
    smiles: str = "CCO",
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(GRAPH_QUALIFIED_COLUMNS)
        for index, y_value in enumerate(y_values):
            writer.writerow(
                [
                    assay_key,
                    str(1000 + index),
                    "11",
                    str(2000 + index),
                    str(3000 + index),
                    "BAO_0000190",
                    relation,
                    f"{10 ** (-y_value):.16g}",
                    unit,
                    "IC50",
                    "B",
                    "BAO_0000357",
                    "CHEMBLASSAY",
                    "20",
                    smiles,
                    "190",
                ]
            )


def _tiny_episode() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    y = torch.tensor([1.0, 2.0, 4.0, 7.0])
    split = torch.tensor([1, 1, 0, 0])
    weight = torch.tensor([[0.3, -0.2, 0.5]], requires_grad=True)
    return x, y, split, weight


def test_dataset_adapter_preserves_upstream_representation_and_filters(tmp_path: Path):
    y_values = [5.0 + index * 0.05 for index in range(20)]
    path = tmp_path / "graph.tsv"
    _write_graph_fixture(path, y_values)

    assays = list(iter_frozen_graph_qualified_assays(path))

    assert len(assays) == 1
    assay = assays[0]
    assert assay.assay_id == "CHEMBLASSAY_nM_IC50"
    assert assay.smiles == ("CCO",) * 20
    assert assay.graph_pair_count == 190
    np.testing.assert_allclose(assay.y, np.asarray(y_values, dtype=np.float32), rtol=1e-6)

    featured = assay.with_features(np.ones((20, 4), dtype=np.float32))
    dataset = ActFoundFrozenMetaDataset([featured], train_assays={featured.assay_id})
    assert dataset.assaes == ["CHEMBLASSAY_nM_IC50"]
    assert dataset.train_indices == [0]
    assert dataset.data_length["train"] == 1
    np.testing.assert_array_equal(dataset.Xs[0], np.ones((20, 4), dtype=np.float32))
    np.testing.assert_array_equal(dataset.ys[0], featured.y)


def test_dataset_filters_match_upstream_chained_loader_path(tmp_path: Path):
    accepted = tmp_path / "accepted.tsv"
    _write_graph_fixture(accepted, [5.0 + index * 0.05 for index in range(20)])
    assert len(list(iter_frozen_graph_qualified_assays(accepted))) == 1

    percent = tmp_path / "percent.tsv"
    _write_graph_fixture(percent, [5.0 + index * 0.05 for index in range(20)], unit="%")
    assert list(iter_frozen_graph_qualified_assays(percent)) == []

    relation = tmp_path / "relation.tsv"
    _write_graph_fixture(relation, [5.0 + index * 0.05 for index in range(20)], relation=">")
    assert list(iter_frozen_graph_qualified_assays(relation)) == []

    too_small = tmp_path / "small.tsv"
    _write_graph_fixture(too_small, [5.0 + index * 0.05 for index in range(19)])
    assert list(iter_frozen_graph_qualified_assays(too_small)) == []

    low_std = tmp_path / "std.tsv"
    _write_graph_fixture(low_std, [5.0 + index * 0.001 for index in range(20)])
    assert list(iter_frozen_graph_qualified_assays(low_std)) == []

    bad_smiles = tmp_path / "bad_smiles.tsv"
    _write_graph_fixture(bad_smiles, [5.0 + index * 0.05 for index in range(20)], smiles="not-a-smiles")
    assert list(iter_frozen_graph_qualified_assays(bad_smiles)) == []


def test_pair_construction_exact_counts_and_labels():
    x, y, split, _ = _tiny_episode()

    pairs = construct_pair_matrices(x, y, split)

    assert pairs.support_idx.tolist() == [0, 1]
    assert pairs.target_idx.tolist() == [2, 3]
    assert pairs.support_support_topk_idx.shape == (1, 2)
    assert pairs.support_target_topk_idx.shape == (1, 2)
    assert pairs.target_target_topk_idx.shape == (1, 2)
    assert torch.equal(pairs.support_support_real, torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
    assert torch.equal(pairs.support_target_real, torch.tensor([[-3.0, -6.0], [-2.0, -5.0]]))
    assert torch.equal(pairs.target_target_real, torch.tensor([[0.0, -3.0], [3.0, 0.0]]))


def test_pairwise_loss_behavior_matches_formula():
    a = torch.tensor([[0.0, 3.0], [5.0, 1.0]])
    b = torch.zeros_like(a)
    topk_idx = torch.tensor([[0, 1]])

    loss = robust_square_error(a, b, topk_idx)

    # |a - b| = [[0, 3], [5, 1]] -> loss = [[0, 5], [7, 1]] (square if <= 2, else (d - 2) + 4).
    # gather(dim=0, [[0, 1]]) selects loss[0][0] = 0 and loss[1][1] = 1.
    expected = torch.mean(torch.tensor([0.0, 1.0]))
    assert torch.isclose(loss, expected)

    x, y, split, weight = _tiny_episode()
    embeddings = x
    values = torch.nn.functional.linear(x, weight)[:, 0]
    net_loss, preds = actfound_loss_and_preds(
        x,
        y,
        split,
        embeddings,
        values,
        is_support=False,
        is_training_phase=True,
    )
    assert net_loss.ndim == 0
    assert preds.shape == (2,)
    assert torch.isfinite(net_loss)


def test_sampling_determinism_under_fixed_seeds():
    x = np.arange(80 * 3, dtype=np.float32).reshape(80, 3)
    y = np.arange(80, dtype=np.float32)

    random.seed(123)
    first = make_episode_split(x, y, test_sup_num=0.5, is_test=False, rand_seed=777)
    random.seed(123)
    second = make_episode_split(x, y, test_sup_num=0.5, is_test=False, rand_seed=777)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    assert len(first[0]) == 64
    assert int(first[2].sum()) >= 10


def test_episode_rng_matches_upstream_mixed_random_sources():
    x = np.arange(80 * 3, dtype=np.float32).reshape(80, 3)
    y = np.arange(80, dtype=np.float32)
    rand_seed = 777

    random.seed(123)
    observed_x, observed_y, observed_split = make_episode_split(
        x,
        y,
        test_sup_num=0.5,
        is_test=False,
        rand_seed=rand_seed,
    )

    rng = np.random.RandomState(seed=rand_seed)
    select_idx = [1] * 64 + [0] * 16
    rng.shuffle(select_idx)
    select_idx = np.nonzero(np.asarray(select_idx))[0]
    expected_x = x[select_idx]
    expected_y = y[select_idx]

    random.seed(123)
    min_num = math.log10(max(10, int(0.3 * len(expected_x))))
    max_num = math.log10(int(0.85 * len(expected_x)))
    sup_num = math.floor(10 ** random.uniform(min_num, max_num))
    expected_split = [1] * sup_num + [0] * (len(expected_x) - sup_num)
    rng.shuffle(expected_split)

    np.testing.assert_array_equal(observed_x, expected_x)
    np.testing.assert_array_equal(observed_y, expected_y)
    np.testing.assert_array_equal(observed_split, np.asarray(expected_split))


def test_target_adaptation_inputs_outputs():
    x, y, split, weight = _tiny_episode()

    adapted, support_losses = adapt_final_linear_weight(
        x,
        y,
        split,
        weight,
        num_steps=2,
        lr=0.01,
    )

    assert adapted.shape == weight.shape
    assert len(support_losses) == 2
    assert all(math.isfinite(value) for value in support_losses)
    assert not torch.equal(adapted, weight)


def test_target_adaptation_parameter_set_and_gradient_dependency():
    x, y, split, weight = _tiny_episode()
    weight = weight.clone().requires_grad_(True)
    hidden = torch.nn.Parameter(torch.ones(3))
    params = select_actfound_inner_loop_params(
        [
            ("layer_dict.linear.weights", weight),
            ("layer_dict.linear0.weights", hidden),
        ]
    )

    assert set(params) == {"layer_dict.linear.weights"}
    adapted, losses = adapt_final_linear_params(
        x,
        y,
        split,
        params,
        num_steps=1,
        lr=0.01,
        create_graph=False,
    )

    assert len(losses) == 1
    assert not torch.equal(adapted["layer_dict.linear.weights"], weight)
    assert torch.equal(hidden.detach(), torch.ones(3))

    query_loss, _ = continuous_inference_from_outputs(
        x,
        y,
        split,
        x,
        torch.nn.functional.linear(
            x,
            adapted["layer_dict.linear.weights"],
        )[:, 0],
    )
    (grad_weight,) = torch.autograd.grad(query_loss, (weight,), allow_unused=False)
    assert torch.isfinite(grad_weight).all()

    adapted_second, _ = adapt_final_linear_params(
        x,
        y,
        split,
        params,
        num_steps=1,
        lr=0.01,
        create_graph=True,
    )
    second_query_loss, _ = continuous_inference_from_outputs(
        x,
        y,
        split,
        x,
        torch.nn.functional.linear(
            x,
            adapted_second["layer_dict.linear.weights"],
        )[:, 0],
    )
    second_grad = torch.autograd.grad(second_query_loss, weight, allow_unused=False)[0]
    assert torch.isfinite(second_grad).all()


def test_inference_output_semantics():
    x, y, split, weight = _tiny_episode()

    loss, preds = continuous_inference(x, y, split, weight)

    assert loss.ndim == 0
    assert preds.shape == (2,)
    assert torch.isfinite(loss)
    assert torch.isfinite(preds).all()

    embeddings = x
    values = torch.nn.functional.linear(x, weight)[:, 0]
    output_loss, output_preds = continuous_inference_from_outputs(x, y, split, embeddings, values)
    assert torch.equal(preds, output_preds)
    assert torch.equal(loss, output_loss)


def test_tiny_end_to_end_qualification():
    x, y, split, weight = _tiny_episode()
    pairs = construct_pair_matrices(x, y, split)
    adapted, support_losses = adapt_final_linear_weight(
        x,
        y,
        split,
        weight,
        num_steps=1,
        lr=0.01,
    )
    loss, preds = continuous_inference(x, y, split, adapted)

    assert pairs.support_target_real.shape == (2, 2)
    assert len(support_losses) == 1
    assert loss.ndim == 0
    assert preds.shape == (2,)


def test_similarity_binarizes_count_fingerprints():
    a = torch.tensor([[2.0, 0.0, 1.0]])
    b = torch.tensor([[3.0, 4.0, 0.0]])
    sim = jaccard_count_sim_matrix(a, b)
    assert torch.isclose(sim[0, 0], torch.tensor(1.0 / 3.0))


def test_forward_topology_requires_all_upstream_terms():
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    y = torch.tensor([1.0, 2.0, 3.5, 5.0, 8.0, 13.0])
    split = torch.tensor([1, 1, 1, 0, 0, 0])
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [2.0, 1.0],
            [-1.0, 2.0],
        ]
    )
    values = torch.tensor([0.2, -0.1, 0.4, 1.0, -0.7, 0.3])

    loss, preds = actfound_loss_and_preds(
        x,
        y,
        split,
        embeddings,
        values,
        is_support=False,
        is_training_phase=False,
    )

    sup_idx = torch.tensor([0, 1, 2])
    tgt_idx = torch.tensor([3, 4, 5])
    sup_y = y[sup_idx]
    tgt_y = y[tgt_idx]
    support_value = values[sup_idx]
    tgt_value = values[tgt_idx]
    rescale = max(0.2, sup_y.std())
    cross_sim = jaccard_count_sim_matrix(x[sup_idx], x[tgt_idx])
    _, topk_idx = torch.topk(cross_sim, dim=0, k=1)
    embed_sim = torch.mm(
        embeddings[sup_idx] / embeddings[sup_idx].norm(dim=1)[:, None],
        (embeddings[tgt_idx] / embeddings[tgt_idx].norm(dim=1)[:, None]).T,
    ) / 0.2
    ddg_pred_1 = support_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
    ddg_real_1 = sup_y.unsqueeze(-1) - tgt_y.unsqueeze(0)
    preds_all = sup_y.unsqueeze(-1).repeat(1, 3) - ddg_pred_1 * rescale
    expected_preds = torch.sum(
        torch.gather(preds_all, 0, topk_idx)
        * torch.nn.Softmax(dim=0)(torch.gather(embed_sim, 0, topk_idx)),
        dim=0,
    )
    tgt_sim = jaccard_count_sim_matrix(x[tgt_idx], x[tgt_idx]) - torch.eye(3)
    _, tgt_topk_idx = torch.topk(tgt_sim, dim=0, k=1)
    ddg_pred_2 = tgt_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
    ddg_real_2 = tgt_y.unsqueeze(-1) - tgt_y.unsqueeze(0)
    expected_loss = torch.sqrt(
        robust_square_error(ddg_pred_2, ddg_real_2, tgt_topk_idx) * 0.25
        + robust_square_error(ddg_pred_1, ddg_real_1, topk_idx) * 0.5
        + torch.mean((expected_preds - tgt_y) ** 2)
    )

    assert torch.allclose(preds, expected_preds)
    assert torch.allclose(loss, expected_loss)


def test_real_upstream_network_outputs_are_compatible_with_reconstructed_forward():
    upstream = ROOT / "external" / "ActFound"
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    try:
        from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR
    except Exception as exc:  # pragma: no cover - environment qualification guard
        pytest.skip(f"upstream network import unavailable: {exc}")

    args = SimpleNamespace(
        hid_dim=4,
        num_stages=1,
        rng=np.random.RandomState(seed=0),
        input_celline=False,
        num_updates=2,
        learnable_bn_gamma=True,
        learnable_bn_beta=True,
        enable_inner_loop_optimizable_bn_params=False,
        per_step_bn_statistics=False,
        norm_layer="batch_norm",
    )
    regressor = FCNReLUNormNetworkQSAR(input_shape=(6, 8), args=args, meta=True)
    x = torch.rand(6, 8)
    y = torch.tensor([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    split = torch.tensor([1, 1, 1, 0, 0, 0])

    weights = select_actfound_inner_loop_params(list(regressor.named_parameters()))
    assert set(weights) == {"layer_dict.linear.weights"}
    loss, preds = actfound_forward_with_regressor(
        regressor,
        x,
        y,
        split,
        weights=weights,
        is_support=False,
        is_training_phase=False,
        training=True,
        num_step=0,
    )
    assert loss.ndim == 0
    assert preds.shape == (3,)
    assert torch.isfinite(loss)
    assert torch.isfinite(preds).all()


def test_provenance_map_covers_required_scientific_components():
    assert set(PROVENANCE) == {
        "dataset_adapter",
        "pair_construction",
        "loss",
        "sampler",
        "target_adaptation",
        "inference",
    }
    for record in PROVENANCE.values():
        assert record["upstream_commit"]
        assert record["upstream_file"]
        assert record["upstream_lines"]
        assert record["project_implementation"]
        assert record["qualification_test"]
