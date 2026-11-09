from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cyp002_config import CYP002_TASK_NAMES
from cyp007_infrastructure import D007_TRAINING_SEEDS, sha256_file
from p1_pairwise_cell_artifacts import (
    BEST_CHECKPOINT_NAME,
    MANIFEST_NAME,
    METRICS_NAME,
    SHA256SUMS_NAME,
    P1CellIdentity,
    build_p1_cell_manifest,
    finalize_p1_cell_manifest,
    save_selected_checkpoint,
    write_metrics,
    write_planned_cell_manifest,
    reload_p1_model_from_checkpoint,
    p1_checkpoint_inference,
)
from p1_pairwise_model import P1AntisymmetricPairHead
from p1_pairwise_training import P1ValidationController
from p1_pairwise_cell_artifacts import build_checkpoint_state
from p1_pairwise_production import (
    EXPECTED_CELL_COUNT,
    MEASURED_COMPONENT_SUBTOTAL_GPU_HOURS,
    PROSPECTIVE_GPU,
    PROSPECTIVE_WALLTIME_MINUTES,
    P1CellKey,
    aggregate_completed_p1_cells,
    classify_cell,
    enumerate_p1_cells,
    p1_cell_dir,
    p1_cell_id,
    preflight_p1_production,
    run_one_cell_wrapper,
    split_manifest_for_cell,
)
import p1_pairwise_production as production
from p1_pairwise_chemprop import build_p1_pair_head, build_p1_shared_message_passing
from cyp007_infrastructure import config_from_trial


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


def _write_inputs(tmp_path):
    dataset = tmp_path / "dataset.csv"
    t121 = tmp_path / "t121.json"
    dataset.write_text("dataset\n", encoding="utf-8")
    t121.write_text(json.dumps({"sampled": T121_SAMPLED}), encoding="utf-8")
    source_split = tmp_path / "cyp" / "artifacts" / "cyp_baseline" / "molecule_split_assignments.csv"
    source_split.parent.mkdir(parents=True)
    source_split.write_text(
        "Molecule_Name,cluster_id,split\n"
        "a,1,train\n"
        "b,2,train\n"
        "c,3,validation\n"
        "v,4,validation\n"
        "d,5,validation\n"
        "t,6,test\n",
        encoding="utf-8",
    )
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
                "source_split_assignments_sha256": sha256_file(source_split),
                "training_names": ["a", "b", "c"],
                "validation_names": ["v", "d"],
                "training_count": 3,
                "validation_count": 2,
                "protected_test_count_in_partitions": 0,
            }
            (split_dir / f"rep_{repetition:02d}_fold_{fold:02d}.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
    return dataset, t121, split_dir


def _write_toy_training_inputs(tmp_path):
    dataset, t121, split_dir = _write_inputs(tmp_path)
    rows = []
    values = {
        "a": [1.0, 10.0, 100.0, 1000.0],
        "b": [2.0, 20.0, 200.0, 2000.0],
        "c": [3.0, 30.0, 300.0, 3000.0],
        "v": [2.5, 25.0, 250.0, 2500.0],
        "d": [4.5, 45.0, 450.0, 4500.0],
        "t": [9.0, 90.0, 900.0, 9000.0],
    }
    smiles = {"a": "CCO", "b": "CCN", "c": "CCC", "v": "CCCl", "d": "CCF", "t": "CCBr"}
    for name in ["a", "b", "c", "v", "d", "t"]:
        row = {"Molecule_Name": name, "SMILES": smiles[name]}
        for task, value in zip(CYP002_TASK_NAMES, values[name], strict=True):
            row[task] = value
            row[f"{task}_conf_low"] = value - 0.5
            row[f"{task}_conf_high"] = value + 0.5
        rows.append(row)
    import pandas as pd

    pd.DataFrame(rows).to_csv(dataset, index=False)
    return dataset, t121, split_dir


def _cell_identity(key: P1CellKey, tmp_path: Path) -> P1CellIdentity:
    dataset = tmp_path / "dataset.csv"
    split = tmp_path / f"split_{key.repetition}_{key.fold}.json"
    t121 = tmp_path / "t121.json"
    for path in (dataset, split, t121):
        if not path.exists():
            path.write_text(path.name, encoding="utf-8")
    return P1CellIdentity(
        repetition=key.repetition,
        fold=key.fold,
        training_seed=key.training_seed,
        split_manifest_path=str(split),
        split_manifest_sha256=sha256_file(split),
        dataset_path=str(dataset),
        dataset_sha256=sha256_file(dataset),
        t121_config_path=str(t121),
        t121_config_sha256=sha256_file(t121),
    )


def _write_complete_cell(root: Path, key: P1CellKey, tmp_path: Path, macro: float) -> Path:
    cell_dir = p1_cell_dir(root, key)
    cell_dir.mkdir(parents=True, exist_ok=False)
    identity = _cell_identity(key, tmp_path)
    p1_config = {"model": "D009-P1", "selection": "strict macro ST-RAE"}
    manifest = build_p1_cell_manifest(
        cell_identity=identity,
        p1_config=p1_config,
        output_dir=cell_dir,
        environment={"synthetic": True},
        provenance={"unit": True},
    )
    write_planned_cell_manifest(manifest, cell_dir / MANIFEST_NAME)
    config = config_from_trial(
        name="D009-P1-production-test",
        sampled=T121_SAMPLED,
        seed=key.training_seed,
    )
    shared_encoder = build_p1_shared_message_passing(config)
    pair_head = build_p1_pair_head(config)
    controller = P1ValidationController()
    controller.observe(update=235, validation_result={"macro_st_rae": macro})
    state = build_checkpoint_state(
        shared_encoder=shared_encoder,
        pair_head=pair_head,
        optimizer=None,
        scheduler=None,
        validation_controller=controller,
        cell_identity=identity,
        p1_config=p1_config,
        reference_metadata={"scaler_scales": [1, 1, 1, 1]},
        update_state={"update_count": 235},
        sampler_state={"state": "synthetic"},
    )
    save_selected_checkpoint(state, cell_dir / BEST_CHECKPOINT_NAME)
    write_metrics(
        {
            "best_validation": {
                "macro_st_rae": macro,
                "per_task": {task: macro for task in CYP002_TASK_NAMES},
            }
        },
        cell_dir / METRICS_NAME,
    )
    finalize_p1_cell_manifest(cell_dir)
    return cell_dir


def test_exactly_150_deterministic_unique_cells_and_mapping():
    cells = enumerate_p1_cells()
    assert len(cells) == EXPECTED_CELL_COUNT == 150
    assert len(set(cells)) == 150
    assert p1_cell_id(P1CellKey(1, 2, 20261101)) == "rep_01/fold_02/seed_20261101"
    assert p1_cell_id(P1CellKey(5, 10, 20261103)) == "rep_05/fold_10/seed_20261103"
    with pytest.raises(ValueError):
        p1_cell_id(P1CellKey(6, 1, 20261101))


def test_preflight_manifest_seed_binding_and_absent_plan(tmp_path):
    dataset, t121, split_dir = _write_inputs(tmp_path)
    production_root = tmp_path / "p1_cells"
    report = preflight_p1_production(
        production_root=production_root,
        repository_root=tmp_path,
        split_manifest_dir=split_dir,
        dataset_path=dataset,
        t121_config_path=t121,
    )
    assert report["split_manifest_count"] == 50
    assert report["repository_root"] == str(tmp_path.resolve())
    assert report["training_seeds"] == list(D007_TRAINING_SEEDS)
    assert report["cell_count"] == 150
    assert report["cell_state_counts"] == {"absent": 150, "complete": 0, "incomplete/failed": 0}
    assert len(report["execution_plan_absent_cells_only"]) == 150
    assert report["protected_internal_test_excluded"] is True
    assert report["d008_evidence_excluded"] is True
    assert report["performance_driven_scheduling"] is False
    assert report["execution_topology"]["gpu_model"] == PROSPECTIVE_GPU
    assert report["execution_topology"]["walltime_minutes"] == PROSPECTIVE_WALLTIME_MINUTES
    assert report["execution_topology"]["measured_component_subtotal_gpu_hours"] == (
        MEASURED_COMPONENT_SUBTOTAL_GPU_HOURS
    )


def test_one_cell_wrapper_rejects_invalid_complete_and_incomplete(tmp_path):
    _dataset, _t121, split_dir = _write_inputs(tmp_path)
    root = tmp_path / "cells"
    root.mkdir()

    with pytest.raises(ValueError):
        run_one_cell_wrapper(
            production_root=root,
            split_manifest_dir=split_dir,
            key=P1CellKey(1, 1, 999),
            cell_runner=lambda *_args: None,
        )

    key = P1CellKey(1, 1, D007_TRAINING_SEEDS[0])
    _write_complete_cell(root, key, tmp_path, macro=0.6)
    with pytest.raises(FileExistsError):
        run_one_cell_wrapper(
            production_root=root,
            split_manifest_dir=split_dir,
            key=key,
            cell_runner=lambda *_args: None,
        )

    incomplete_key = P1CellKey(1, 1, D007_TRAINING_SEEDS[1])
    incomplete_dir = p1_cell_dir(root, incomplete_key)
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mid-cell resume is not authorized"):
        run_one_cell_wrapper(
            production_root=root,
            split_manifest_dir=split_dir,
            key=incomplete_key,
            cell_runner=lambda *_args: None,
        )


def test_cell_classification_inventory_hash_and_corrupt_failure(tmp_path):
    root = tmp_path / "cells"
    key = P1CellKey(1, 1, D007_TRAINING_SEEDS[0])
    assert classify_cell(root, key).state == "absent"
    cell_dir = _write_complete_cell(root, key, tmp_path, macro=0.6)
    assert classify_cell(root, key).state == "complete"
    (cell_dir / METRICS_NAME).write_text("corrupt", encoding="utf-8")
    state = classify_cell(root, key)
    assert state.state == "incomplete/failed"
    assert "mismatch" in state.reason


def test_completion_requires_exact_150_and_rejects_missing_or_mismatched(tmp_path):
    root = tmp_path / "cells"
    with pytest.raises(ValueError, match="not complete"):
        aggregate_completed_p1_cells(root)

    for key in enumerate_p1_cells():
        _write_complete_cell(root, key, tmp_path, macro=0.6)
    result = aggregate_completed_p1_cells(root)
    assert result["cell_count"] == 150
    assert result["o150"] == pytest.approx(0.6)
    assert result["p1_advances"] is True
    assert result["endpoint_veto"] is False

    bad_cell = p1_cell_dir(root, P1CellKey(1, 1, D007_TRAINING_SEEDS[0]))
    manifest = json.loads((bad_cell / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["cell_identity"]["fold"] = 9
    (bad_cell / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (bad_cell / SHA256SUMS_NAME).write_text(
        "\n".join(
            [
                f"{sha256_file(bad_cell / MANIFEST_NAME)}  {MANIFEST_NAME}",
                f"{sha256_file(bad_cell / BEST_CHECKPOINT_NAME)}  {BEST_CHECKPOINT_NAME}",
                f"{sha256_file(bad_cell / METRICS_NAME)}  {METRICS_NAME}",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity"):
        aggregate_completed_p1_cells(root)


def test_synthetic_o150_strict_t121_comparison_no_endpoint_veto(tmp_path):
    root = tmp_path / "cells"
    macros = []
    for index, key in enumerate(enumerate_p1_cells()):
        macro = 0.5 + index / 1000.0
        macros.append(macro)
        _write_complete_cell(root, key, tmp_path, macro=macro)
    result = aggregate_completed_p1_cells(root)
    assert result["o150"] == pytest.approx(sum(macros) / 150)
    assert result["p1_advances"] == (result["o150"] < 0.701212576796)
    assert result["strict_advancement_rule"] == (
        "P1 advances iff O150(P1) < 0.701212576796"
    )


def test_cli_preflight_writes_artifact_and_trains_nothing(tmp_path, monkeypatch):
    dataset, t121, split_dir = _write_inputs(tmp_path)
    output = tmp_path / "preflight.json"
    called = {"run_cell": False}

    def forbidden(*_args, **_kwargs):
        called["run_cell"] = True
        raise AssertionError("preflight must not run training")

    monkeypatch.setattr(production, "execute_qualified_p1_cell", forbidden)
    production.main([
        "preflight",
        "--production-root", str(tmp_path / "cells"),
        "--repository-root", str(tmp_path),
        "--dataset-path", str(dataset),
        "--split-manifest-dir", str(split_dir),
        "--t121-config-artifact", str(t121),
        "--output-json", str(output),
    ])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cell_count"] == 150
    assert report["cell_state_counts"]["absent"] == 150
    assert called["run_cell"] is False


def test_cli_run_cell_invokes_qualified_binding_and_correct_manifest(tmp_path, monkeypatch):
    dataset, t121, split_dir = _write_inputs(tmp_path)
    output = tmp_path / "run_cell.json"
    root = tmp_path / "cells"
    observed = {}

    def synthetic_qualified_runner(*, key, split_manifest_path, cell_path, repository_root, dataset_path, t121_config_path):
        observed["key"] = key
        observed["split_manifest_path"] = split_manifest_path
        observed["repository_root"] = repository_root
        observed["dataset_path"] = dataset_path
        observed["t121_config_path"] = t121_config_path
        _write_complete_cell(root, key, tmp_path, macro=0.6)

    monkeypatch.setattr(production, "execute_qualified_p1_cell", synthetic_qualified_runner)
    production.main([
        "run-cell",
        "--repetition", "1",
        "--fold", "2",
        "--training-seed", str(D007_TRAINING_SEEDS[0]),
        "--production-root", str(root),
        "--repository-root", str(tmp_path),
        "--dataset-path", str(dataset),
        "--split-manifest-dir", str(split_dir),
        "--t121-config-artifact", str(t121),
        "--output-json", str(output),
    ])
    key = P1CellKey(1, 2, D007_TRAINING_SEEDS[0])
    assert observed["key"] == key
    assert observed["split_manifest_path"] == split_manifest_for_cell(split_dir, key)
    assert classify_cell(root, key).state == "complete"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["state"]["state"] == "complete"


def test_same_run_cell_binding_completes_toy_cell_end_to_end(tmp_path):
    dataset, _t121, split_dir = _write_toy_training_inputs(tmp_path)
    t121 = tmp_path / "toy_t121.json"
    t121.write_text(json.dumps({"sampled": T121_SAMPLED}), encoding="utf-8")
    root = tmp_path / "toy_cells"
    key = P1CellKey(1, 1, D007_TRAINING_SEEDS[0])

    def runner(cell_key, split_manifest_path, cell_path):
        production.execute_qualified_p1_cell(
            key=cell_key,
            split_manifest_path=split_manifest_path,
            cell_path=cell_path,
            repository_root=tmp_path,
            dataset_path=dataset,
            t121_config_path=t121,
            max_updates=2,
            validation_interval_updates=1,
            patience=2,
        )

    cell_path = run_one_cell_wrapper(
        production_root=root,
        split_manifest_dir=split_dir,
        key=key,
        cell_runner=runner,
    )
    assert classify_cell(root, key).state == "complete"
    manifest = json.loads((cell_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    expected_identity = P1CellIdentity(**manifest["cell_identity"])
    config = config_from_trial(
        name="D009-P1-production-reload-test",
        sampled=T121_SAMPLED,
        seed=key.training_seed,
    )
    _encoder, pair_head, _state = reload_p1_model_from_checkpoint(
        checkpoint_path=cell_path / BEST_CHECKPOINT_NAME,
        config=config,
        expected_cell_identity=expected_identity,
        expected_p1_config=manifest["p1_config"],
    )
    query_embeddings = torch.zeros((1, 600), dtype=torch.float32)
    reference_embeddings = torch.zeros((3, 600), dtype=torch.float32)
    reference_y = torch.tensor(
        [[1.0, 10.0, 100.0, 1000.0], [2.0, 20.0, 200.0, 2000.0], [3.0, 30.0, 300.0, 3000.0]],
        dtype=torch.float32,
    )
    reference_mask = torch.ones((3, 4), dtype=torch.bool)
    scaler_scales = torch.ones(4, dtype=torch.float32)
    mean, dispersion = p1_checkpoint_inference(
        pair_head=pair_head,
        query_embeddings=query_embeddings,
        reference_embeddings=reference_embeddings,
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
    )
    assert torch.isfinite(mean).all()
    assert torch.isfinite(dispersion).all()


def test_cli_aggregate_requires_150_complete_cells(tmp_path):
    output = tmp_path / "aggregate.json"
    with pytest.raises(ValueError, match="not complete"):
        production.main([
            "aggregate",
            "--production-root", str(tmp_path / "cells"),
            "--output-json", str(output),
        ])


def test_cli_exposes_no_protected_d008_or_retry_controls():
    parser = production.build_parser()
    option_dests = {action.dest for action in parser._actions}
    for action in parser._actions:
        choices = getattr(action, "choices", None) or {}
        for subaction in choices.values():
            option_dests.update(sub.dest for sub in subaction._actions)
    forbidden = {
        "protected_test",
        "protected_test_labels",
        "d008",
        "d008_labels",
        "retry",
        "rerun_failed",
        "resume",
        "performance_threshold",
        "st_rae_cutoff",
    }
    assert option_dests.isdisjoint(forbidden)
    assert "max_updates" not in option_dests
    assert "validation_interval_updates" not in option_dests
    assert "patience" not in option_dests
