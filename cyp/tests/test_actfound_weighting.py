import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cyp" / "scripts"))

from actfound_model_side import (
    ActFoundAssay,
    ActFoundFrozenMetaDataset,
    actfound_loss_and_preds,
    construct_pair_matrices,
    curve_fit_observation_weights,
    harmonic_pair_weights,
    robust_square_error,
    screen_observation_weights,
    _weighted_direct_mse,
)


class TestActFoundHarmonicWeighting(unittest.TestCase):
    def test_raw_curve_fit_and_screen_weights(self):
        ci = curve_fit_observation_weights(
            [2.0, 3.0, np.nan], [3.0, 5.0, np.nan],
            [True, True, False], ["M1", "M2", "M3"],
        )
        np.testing.assert_allclose(ci[:2], [1.0 / 1.25, 1.0 / 2.25])
        self.assertTrue(np.isnan(ci[2]))
        se = screen_observation_weights(
            [0.5, 1.0, np.nan], [True, True, False], ["M1", "M2", "M3"]
        )
        np.testing.assert_allclose(se[:2], [4.0, 1.0])
        self.assertTrue(np.isnan(se[2]))

    def test_invalid_observed_uncertainty_names_rows(self):
        with self.assertRaisesRegex(ValueError, "BAD_CI"):
            curve_fit_observation_weights([2.0], [2.0], [True], ["BAD_CI"])
        with self.assertRaisesRegex(ValueError, "BAD_SE"):
            screen_observation_weights([0.0], [True], ["BAD_SE"])

    def test_harmonic_cross_weights_and_selected_orientation(self):
        wa = torch.tensor([1.0, 2.0, 4.0])
        wb = torch.tensor([3.0, 6.0])
        raw = harmonic_pair_weights(wa, wb)
        expected_matrix = torch.tensor([
            [3.0 / 4.0, 6.0 / 7.0],
            [6.0 / 5.0, 12.0 / 8.0],
            [12.0 / 7.0, 24.0 / 10.0],
        ])
        torch.testing.assert_close(raw, expected_matrix)

        errors = torch.tensor([[1.0, 3.0], [4.0, 0.0], [2.0, 1.0]])
        zeros = torch.zeros_like(errors)
        idx = torch.tensor([[2, 0], [1, 2]])
        selected = torch.gather(raw, 0, idx)
        manual = torch.gather(torch.where(errors.abs() <= 2, errors.square(), errors.abs() + 2), 0, idx)
        normalized = selected / selected.mean()
        expected = torch.mean(manual * normalized)
        actual = robust_square_error(
            errors, zeros, idx,
            observation_weights_a=wa,
            observation_weights_b=wb,
        )
        torch.testing.assert_close(actual, expected)

    def test_query_query_harmonic_weights_and_selected_orientation(self):
        w = torch.tensor([2.0, 4.0, 8.0])
        raw = harmonic_pair_weights(w, w)
        expected = torch.tensor([
            [1.0, 4.0 / 3.0, 8.0 / 5.0],
            [4.0 / 3.0, 2.0, 8.0 / 3.0],
            [8.0 / 5.0, 8.0 / 3.0, 4.0],
        ])
        torch.testing.assert_close(raw, expected)
        errors = torch.tensor([[0.0, 1.0, 4.0], [2.0, 3.0, 0.0], [1.0, 5.0, 2.0]])
        idx = torch.tensor([[2, 0, 1], [1, 2, 0]])
        selected_w = torch.gather(raw, 0, idx)
        error_matrix = torch.where(errors.abs() <= 2, errors.square(), errors.abs() + 2)
        selected_e = torch.gather(error_matrix, 0, idx)
        expected_loss = torch.mean(selected_e * (selected_w / selected_w.mean()))
        actual = robust_square_error(
            errors, torch.zeros_like(errors), idx,
            observation_weights_a=w,
            observation_weights_b=w,
        )
        torch.testing.assert_close(actual, expected_loss)

    def test_unequal_partition_scales_are_not_normalized_before_pairing(self):
        support = torch.tensor([1.0, 100.0])
        query = torch.tensor([10.0, 10.0])
        combined = harmonic_pair_weights(support, query)
        separately_normalized = harmonic_pair_weights(
            support / support.mean(), query / query.mean()
        )
        self.assertFalse(torch.allclose(combined, separately_normalized))

    def test_uniform_weights_recover_original_support_and_query_losses(self):
        torch.manual_seed(412)
        x = torch.randint(0, 2, (8, 32)).float()
        y = torch.linspace(0.5, 4.0, 8)
        split = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
        embeddings = torch.randn(8, 12)
        values = torch.randn(8, requires_grad=True)
        for is_support in (True, False):
            base, base_preds = actfound_loss_and_preds(
                x, y, split, embeddings, values,
                is_support=is_support, is_training_phase=True,
            )
            weighted, weighted_preds = actfound_loss_and_preds(
                x, y, split, embeddings, values,
                is_support=is_support, is_training_phase=True,
                observation_weights=torch.ones_like(y),
                molecule_ids=tuple(f"M{i}" for i in range(len(y))),
            )
            torch.testing.assert_close(weighted, base, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(weighted_preds, base_preds)

    def test_uniform_weights_recover_each_direct_and_pair_reduction(self):
        predictions = torch.tensor([1.0, 3.0, 2.0])
        targets = torch.tensor([0.0, 1.0, 4.0])
        original_direct = torch.mean((predictions - targets) ** 2)
        weighted_direct = _weighted_direct_mse(
            predictions, targets, torch.ones_like(targets)
        )
        torch.testing.assert_close(weighted_direct, original_direct)

        cross_pred = torch.tensor([[0.0, 3.0], [2.0, 1.0], [4.0, 0.0]])
        cross_true = torch.tensor([[1.0, 0.0], [0.0, 3.0], [2.0, 1.0]])
        cross_idx = torch.tensor([[2, 0], [1, 2]])
        cross_base = robust_square_error(cross_pred, cross_true, cross_idx)
        cross_weighted = robust_square_error(
            cross_pred, cross_true, cross_idx,
            observation_weights_a=torch.ones(3),
            observation_weights_b=torch.ones(2),
        )
        torch.testing.assert_close(cross_weighted, cross_base)

        query_pred = torch.tensor([[0.0, 3.0, 2.0], [2.0, 1.0, 4.0], [4.0, 0.0, 1.0]])
        query_true = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0], [2.0, 1.0, 4.0]])
        query_idx = torch.tensor([[2, 0, 1], [1, 2, 0]])
        query_base = robust_square_error(query_pred, query_true, query_idx)
        query_weighted = robust_square_error(
            query_pred, query_true, query_idx,
            observation_weights_a=torch.ones(3),
            observation_weights_b=torch.ones(3),
        )
        torch.testing.assert_close(query_weighted, query_base)

    def test_metadata_alignment_and_sampling_are_unchanged(self):
        ids = tuple(f"M{i:02d}" for i in range(12))
        x = np.arange(12 * 32, dtype=np.float32).reshape(12, 32) % 2
        y = np.arange(12, dtype=np.float32)
        raw = np.arange(1, 13, dtype=np.float64)
        assay = ActFoundAssay(
            assay_id="A", smiles=tuple("C" for _ in ids), y=y, x=x,
            molecule_ids=ids, observation_weights=raw,
        )
        ds = ActFoundFrozenMetaDataset([assay], val_assays={"A"})
        first = ds.get_task_split(
            0, test_sup_num=0.5, is_test=True, rand_seed=919, include_metadata=True
        )
        second = ds.get_task_split(
            0, test_sup_num=0.5, is_test=True, rand_seed=919, include_metadata=True
        )
        self.assertEqual(len(first), 5)
        for left, right in zip(first[:4], second[:4]):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(first[4], second[4])
        xx, yy, split, ww, mm = first
        for row_x, row_y, row_w, molecule_id in zip(xx, yy, ww, mm):
            i = ids.index(molecule_id)
            np.testing.assert_array_equal(row_x, x[i])
            self.assertEqual(row_y, y[i])
            self.assertEqual(row_w, raw[i])

        p1 = construct_pair_matrices(torch.tensor(xx), torch.tensor(yy), torch.tensor(split))
        p2 = construct_pair_matrices(torch.tensor(xx), torch.tensor(yy), torch.tensor(split))
        torch.testing.assert_close(p1.support_target_topk_idx, p2.support_target_topk_idx)
        torch.testing.assert_close(p1.target_target_topk_idx, p2.target_target_topk_idx)

    def test_missing_labels_remain_masked_and_invalid_observed_weights_fail(self):
        assay = ActFoundAssay(
            assay_id="A", smiles=("C", "CC"),
            y=np.asarray([1.0, np.nan]),
            molecule_ids=("OBS", "MISSING"),
            observation_weights=np.asarray([2.0, np.nan]),
        )
        self.assertTrue(np.isnan(assay.y[1]))
        self.assertTrue(np.isnan(assay.observation_weights[1]))
        with self.assertRaisesRegex(ValueError, "BAD"):
            ActFoundAssay(
                assay_id="B", smiles=("C",), y=np.asarray([1.0]),
                molecule_ids=("BAD",), observation_weights=np.asarray([np.nan]),
            )

    def test_weighting_keeps_finite_model_gradients(self):
        torch.manual_seed(83)
        x = torch.randint(0, 2, (8, 32)).float()
        y = torch.linspace(0.5, 4.0, 8)
        split = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
        embeddings = torch.randn(8, 12, requires_grad=True)
        values = torch.randn(8, requires_grad=True)
        raw = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
        loss, _ = actfound_loss_and_preds(
            x, y, split, embeddings, values,
            is_support=False, is_training_phase=True,
            observation_weights=raw,
            molecule_ids=tuple(f"M{i}" for i in range(8)),
        )
        loss.backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertFalse(raw.requires_grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
