from __future__ import annotations

import json

import pytest
import torch

from cyp007_infrastructure import D007_TRAINING_SEEDS, config_from_trial, sha256_file
from p1_pairwise_cell_artifacts import (
    BEST_CHECKPOINT_NAME,
    MANIFEST_NAME,
    METRICS_NAME,
    SHA256SUMS_NAME,
    P1CellIdentity,
    build_checkpoint_state,
    build_p1_cell_manifest,
    create_p1_cell_artifact_dir,
    finalize_p1_cell_manifest,
    load_checkpoint_state,
    p1_checkpoint_inference,
    reload_p1_model_from_checkpoint,
    save_selected_checkpoint,
    verify_p1_cell_inventory,
    verify_resume_state,
    write_metrics,
    write_planned_cell_manifest,
)
from p1_pairwise_chemprop import build_p1_pair_head, build_p1_shared_message_passing
from p1_pairwise_training import (
    P1ValidationController,
    build_p1_optimizer_and_scheduler,
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


def _config():
    return config_from_trial(
        name="D009-P1-cell-artifact-test",
        sampled=T121_SAMPLED,
        seed=D007_TRAINING_SEEDS[0],
    )


def _identity(tmp_path):
    dataset = tmp_path / "dataset.csv"
    manifest = tmp_path / "split_manifest.json"
    t121 = tmp_path / "t121.json"
    dataset.write_text("dataset\n", encoding="utf-8")
    manifest.write_text(json.dumps({"split": True}), encoding="utf-8")
    t121.write_text(json.dumps({"sampled": T121_SAMPLED}), encoding="utf-8")
    return P1CellIdentity(
        repetition=1,
        fold=2,
        training_seed=D007_TRAINING_SEEDS[0],
        split_manifest_path=str(manifest),
        split_manifest_sha256=sha256_file(manifest),
        dataset_path=str(dataset),
        dataset_sha256=sha256_file(dataset),
        t121_config_path=str(t121),
        t121_config_sha256=sha256_file(t121),
    )


def _p1_config():
    return {
        "model": "D009-P1",
        "encoder": "shared Chemprop BondMessagePassing",
        "pair_head": "P1AntisymmetricPairHead",
        "reference_reconstruction": "all eligible training references",
        "selection": "strictly lowest validation macro ST-RAE",
    }


def _checkpoint_state(tmp_path):
    config = _config()
    shared_encoder = build_p1_shared_message_passing(config)
    pair_head = build_p1_pair_head(config)
    optimizer, scheduler = build_p1_optimizer_and_scheduler(
        list(shared_encoder.parameters()) + list(pair_head.parameters())
    )
    controller = P1ValidationController()
    controller.observe(
        update=235,
        validation_result={"macro_st_rae": 1.0},
        state_dict=pair_head.state_dict(),
    )
    controller.observe(
        update=470,
        validation_result={"macro_st_rae": 0.9},
        state_dict=pair_head.state_dict(),
    )
    controller.observe(
        update=705,
        validation_result={"macro_st_rae": 0.95},
        state_dict=pair_head.state_dict(),
    )
    identity = _identity(tmp_path)
    p1_config = _p1_config()
    reference_metadata = {
        "scaler_means": [2.0, 15.0, 200.0, 2000.0],
        "scaler_scales": [1.0, 5.0, 100.0, 1000.0],
        "n_t": [3, 2, 2, 3],
        "q_t": [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0],
        "w_t": [1.0, 2.0, 2.0, 1.0],
    }
    update_state = {
        "update_count": 705,
        "validation_checks": 3,
        "scheduler_last_epoch": scheduler.last_epoch,
    }
    sampler_state = {"numpy_bit_generator_state": {"state": "synthetic"}}
    state = build_checkpoint_state(
        shared_encoder=shared_encoder,
        pair_head=pair_head,
        optimizer=optimizer,
        scheduler=scheduler,
        validation_controller=controller,
        cell_identity=identity,
        p1_config=p1_config,
        reference_metadata=reference_metadata,
        update_state=update_state,
        sampler_state=sampler_state,
    )
    return config, shared_encoder, pair_head, identity, p1_config, state


def test_create_save_reload_and_inference_consistency(tmp_path):
    config, _shared_encoder, pair_head, identity, p1_config, state = _checkpoint_state(tmp_path)
    cell_dir = create_p1_cell_artifact_dir(tmp_path / "cell")
    manifest = build_p1_cell_manifest(
        cell_identity=identity,
        p1_config=p1_config,
        output_dir=cell_dir,
        environment={"host": "synthetic"},
        provenance={"qualification": "synthetic"},
    )
    write_planned_cell_manifest(manifest, cell_dir / MANIFEST_NAME)
    save_selected_checkpoint(state, cell_dir / BEST_CHECKPOINT_NAME)
    write_metrics(
        {
            "best_validation": state["validation_state"],
            "test_evaluation_performed": False,
        },
        cell_dir / METRICS_NAME,
    )
    finalized = finalize_p1_cell_manifest(cell_dir)
    assert finalized["outputs"]["best_checkpoint_sha256"] == sha256_file(
        cell_dir / BEST_CHECKPOINT_NAME
    )

    inventory = verify_p1_cell_inventory(cell_dir)
    assert set(inventory) == {MANIFEST_NAME, BEST_CHECKPOINT_NAME, METRICS_NAME}

    _reloaded_encoder, reloaded_pair_head, loaded_state = reload_p1_model_from_checkpoint(
        checkpoint_path=cell_dir / BEST_CHECKPOINT_NAME,
        config=config,
        expected_cell_identity=identity,
        expected_p1_config=p1_config,
    )
    assert loaded_state["validation_state"]["best_score"] == pytest.approx(0.9)
    assert loaded_state["validation_state"]["best_update"] == 470

    torch.manual_seed(20261107)
    query_embeddings = torch.randn(2, 600)
    reference_embeddings = torch.randn(3, 600)
    reference_y = torch.tensor(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
            [3.0, 30.0, 300.0, 3000.0],
        ],
        dtype=torch.float32,
    )
    reference_mask = torch.ones((3, 4), dtype=torch.bool)
    scaler_scales = torch.tensor([1.0, 5.0, 100.0, 1000.0])

    saved_mean, saved_dispersion = p1_checkpoint_inference(
        pair_head=pair_head,
        query_embeddings=query_embeddings,
        reference_embeddings=reference_embeddings,
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
    )
    reloaded_mean, reloaded_dispersion = p1_checkpoint_inference(
        pair_head=reloaded_pair_head,
        query_embeddings=query_embeddings,
        reference_embeddings=reference_embeddings,
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
    )
    torch.testing.assert_close(saved_mean, reloaded_mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(saved_dispersion, reloaded_dispersion, rtol=0.0, atol=0.0)


def test_collision_incomplete_corrupt_and_mismatch_fail_closed(tmp_path):
    config, _shared_encoder, _pair_head, identity, p1_config, state = _checkpoint_state(tmp_path)
    cell_dir = create_p1_cell_artifact_dir(tmp_path / "cell")
    (cell_dir / "extra.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_p1_cell_artifact_dir(cell_dir)

    empty_dir = create_p1_cell_artifact_dir(tmp_path / "empty_cell")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        verify_p1_cell_inventory(empty_dir)

    corrupt = tmp_path / "corrupt.ckpt"
    corrupt.write_text("not a torch checkpoint", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not load"):
        load_checkpoint_state(
            corrupt,
            expected_cell_identity=identity,
            expected_p1_config=p1_config,
        )

    checkpoint = tmp_path / "best.ckpt"
    save_selected_checkpoint(state, checkpoint)
    bad_identity = P1CellIdentity(
        repetition=9,
        fold=identity.fold,
        training_seed=identity.training_seed,
        split_manifest_path=identity.split_manifest_path,
        split_manifest_sha256=identity.split_manifest_sha256,
        dataset_path=identity.dataset_path,
        dataset_sha256=identity.dataset_sha256,
        t121_config_path=identity.t121_config_path,
        t121_config_sha256=identity.t121_config_sha256,
    )
    with pytest.raises(ValueError, match="cell identity mismatch"):
        load_checkpoint_state(
            checkpoint,
            expected_cell_identity=bad_identity,
            expected_p1_config=p1_config,
        )
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        load_checkpoint_state(
            checkpoint,
            expected_cell_identity=identity,
            expected_p1_config={**p1_config, "changed": True},
        )


def test_resume_verifies_required_training_state_and_identity(tmp_path):
    _config, _shared_encoder, _pair_head, identity, p1_config, state = _checkpoint_state(tmp_path)
    verify_resume_state(
        state,
        expected_cell_identity=identity,
        expected_p1_config=p1_config,
        expected_update=705,
    )

    with pytest.raises(ValueError, match="update count mismatch"):
        verify_resume_state(
            state,
            expected_cell_identity=identity,
            expected_p1_config=p1_config,
            expected_update=704,
        )

    incomplete = dict(state)
    incomplete["optimizer_state_dict"] = None
    with pytest.raises(ValueError, match="optimizer_state_dict"):
        verify_resume_state(
            incomplete,
            expected_cell_identity=identity,
            expected_p1_config=p1_config,
            expected_update=705,
        )


def test_checkpoint_requires_strictly_selected_best_validation(tmp_path):
    config = _config()
    shared_encoder = build_p1_shared_message_passing(config)
    pair_head = build_p1_pair_head(config)
    controller = P1ValidationController()
    controller.observe(update=235, validation_result={"macro_st_rae": 1.0})
    controller.observe(update=470, validation_result={"macro_st_rae": 1.0})
    assert controller.best_update == 235
    assert controller.bad_checks == 1

    state = build_checkpoint_state(
        shared_encoder=shared_encoder,
        pair_head=pair_head,
        optimizer=None,
        scheduler=None,
        validation_controller=controller,
        cell_identity=_identity(tmp_path),
        p1_config=_p1_config(),
        reference_metadata={"scaler_scales": [1.0, 1.0, 1.0, 1.0]},
        update_state={"update_count": 470},
        sampler_state={"state": "synthetic"},
    )
    assert state["validation_state"]["best_update"] == 235
