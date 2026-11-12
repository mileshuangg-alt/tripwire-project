from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_comparator_guard_rejects_p1_o150_artifact_as_t121() -> None:
    from d012_corrected_final_aggregation import assert_t121_source_argument

    p1_artifact = ROOT / "artifacts/d009_closure_workspace/p1_d009_o150_result_20261108.json"
    with pytest.raises(ValueError, match="P1 aggregate artifact"):
        assert_t121_source_argument(p1_artifact)


def test_comparator_guard_accepts_saved_t121_cell_directory() -> None:
    from d012_corrected_final_aggregation import load_corrected_t121

    cells = load_corrected_t121(ROOT / "artifacts/d009_closure_workspace/d009_t121_comparator_20261109/cells")
    assert len(cells) == 150
    assert abs(sum(row["macro_st_rae"] for row in cells.values()) / 150 - 0.701212576796) < 1e-12


def test_corrected_runner_preserves_b_batching_scaler_scoring_checkpoint_sources() -> None:
    original = (SCRIPTS / "d012_first_cell_runner.py").read_text(encoding="utf-8")
    corrected = (SCRIPTS / "d012_corrected_first_cell_runner.py").read_text(encoding="utf-8")

    unchanged_fragments = [
        "batches = list(range(0, train_values.shape[0], TARGET_BATCH_SIZE))",
        "batch_dataset = data.MoleculeDataset([train_dataset.data[i] for i in range(start, stop)])",
        "means, scales = fit_target_scaler(train_values, train_mask)",
        "loss = masked_mse(pred, batch_y, batch_mask)",
        "score = score_validation(",
        "if macro < best_score:",
        "bad += 1",
        "if bad >= TARGET_PATIENCE:",
        "writer.checkpoint(\"arm_b_best_target_checkpoint.pt\"",
    ]
    for fragment in unchanged_fragments:
        assert fragment in original
        assert fragment in corrected
    assert "target_scheduler = LinearAnchorScheduler(target_optimizer" in original
    assert "target_scheduler = chemprop_noam_target_scheduler(" in corrected


def test_corrected_runner_preserves_c_target_stage_except_adaptation_selection() -> None:
    corrected = (SCRIPTS / "d012_corrected_first_cell_runner.py").read_text(encoding="utf-8")

    assert "head.load_state_dict(model.scalar_head_initialization.state_dict())" in corrected
    assert "support_dataset = data.MoleculeDataset([train_dataset.data[i] for i, flag in enumerate(keep.tolist()) if flag])" in corrected
    assert "model.encoder.forward_dataset(support_dataset, device=device, batch_size=TARGET_BATCH_SIZE).detach()" in corrected
    assert "adapt_selected_output_exact_d012(head, support_embeddings, support_y)" in corrected
    assert "target_update\": \"selected_output_weight_bias_only_lr_0.01_k1_encoder_frozen\"" in corrected


def test_b_scheduler_differential_against_chemprop_noam_for_distinct_update_counts() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from chemprop import schedulers
    from d012_corrective_patch import chemprop_noam_target_scheduler

    target_rows = list(__import__("csv").DictReader((ROOT / "cyp/data/cyp-challenge-TRAIN_inhibition.csv").open(newline="", encoding="utf-8")))
    by_name = {row["Molecule_Name"]: row for row in target_rows}
    update_counts = set()
    for path in sorted((ROOT / "cyp/artifacts/cyp_007/split_manifests").glob("rep_*_fold_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        # D012 B target batching is fixed manifest-order batching over all training rows, not observed-label rows.
        update_counts.add(math.ceil(len(manifest["training_names"]) / 16))
        assert all(name in by_name for name in manifest["training_names"])
    assert update_counts

    max_abs = 0.0
    max_rel = 0.0
    for updates_per_epoch in sorted(update_counts):
        p_a = torch.nn.Parameter(torch.tensor([1.0]))
        p_b = torch.nn.Parameter(torch.tensor([1.0]))
        opt_a = torch.optim.Adam([p_a], lr=1.4283739994469765e-05)
        opt_b = torch.optim.Adam([p_b], lr=1.4283739994469765e-05)
        sched_a = chemprop_noam_target_scheduler(
            opt_a,
            updates_per_epoch=updates_per_epoch,
            warmup_epochs=10,
            max_epochs=50,
            init_lr=1.4283739994469765e-05,
            max_lr=0.0008245400010426866,
            final_lr=0.00020141638331344468,
        )
        sched_b = schedulers.build_NoamLike_LRSched(
            opt_b,
            warmup_steps=10 * updates_per_epoch,
            cooldown_steps=50 * updates_per_epoch - 10 * updates_per_epoch,
            init_lr=1.4283739994469765e-05,
            max_lr=0.0008245400010426866,
            final_lr=0.00020141638331344468,
        )
        seq_a = []
        seq_b = []
        for _ in range(50 * updates_per_epoch):
            opt_a.step(); sched_a.step(); seq_a.append(opt_a.param_groups[0]["lr"])
            opt_b.step(); sched_b.step(); seq_b.append(opt_b.param_groups[0]["lr"])
        assert len(seq_a) == len(seq_b) == 50 * updates_per_epoch
        for a, b in zip(seq_a, seq_b, strict=True):
            diff = abs(a - b)
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, diff / max(abs(b), 1e-300))
    assert max_abs <= 0.0
    assert max_rel <= 0.0


def test_c_selected_output_adaptation_matches_authoritative_primitive_and_preserves_hidden() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from d012_corrective_patch import adapt_selected_output_exact_d012, functional_scalar_head_prediction
    from p2_anil_model import selected_task_prediction, standardized_mse

    class TinyScalarHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = torch.nn.Linear(600, 2000)
            self.activation = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.0)
            self.output = torch.nn.Linear(2000, 1)

        def forward(self, embeddings):
            return self.output(self.dropout(self.activation(self.hidden(embeddings)))).squeeze(-1)

    torch.manual_seed(12012)
    head = TinyScalarHead().double()
    support = torch.randn(17, 600, dtype=torch.float64, requires_grad=True)
    query = torch.randn(5, 600, dtype=torch.float64, requires_grad=True)
    y = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    before = {name: value.detach().clone() for name, value in head.state_dict().items()}
    adapted, loss = adapt_selected_output_exact_d012(head, support, y)

    features = head.dropout(head.activation(head.hidden(support)))
    weight = head.output.weight.squeeze(0)
    bias = head.output.bias.squeeze(0)
    pred = selected_task_prediction(features, task_index=0, weight=weight, bias=bias)
    manual_loss = standardized_mse(pred, y)
    grad_w, grad_b = torch.autograd.grad(manual_loss, (weight, bias), create_graph=True, retain_graph=True)
    torch.testing.assert_close(adapted["output.weight"].squeeze(0), weight - 0.01 * grad_w)
    torch.testing.assert_close(adapted["output.bias"].squeeze(0), bias - 0.01 * grad_b)
    torch.testing.assert_close(loss, manual_loss)
    torch.testing.assert_close(adapted["hidden.weight"], head.hidden.weight)
    torch.testing.assert_close(adapted["hidden.bias"], head.hidden.bias)
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, before[name])

    query_loss = (functional_scalar_head_prediction(head, adapted, query) ** 2).mean()
    encoder_like_grad = torch.autograd.grad(query_loss, query, retain_graph=True)[0]
    assert torch.isfinite(encoder_like_grad).all()
    assert encoder_like_grad.abs().sum() > 0
