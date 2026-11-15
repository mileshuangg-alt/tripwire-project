from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "cyp" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from actfound_production_train import (  # noqa: E402
    PINNED_INPUT,
    PINNED_INPUT_SHA256,
    PINNED_UPSTREAM_COMMIT,
    UPSTREAM_DEFAULTS,
    file_sha256,
    main,
    validate_and_resolve_config,
)


def _write_reconciliation(path: Path) -> Path:
    payload = {
        "artifact_type": "ActFoundSplitReconciliationV1",
        "status": "complete",
        "frozen_dataset_sha256": PINNED_INPUT_SHA256,
        "unresolved_total": 0,
        "ambiguous_total": 0,
        "train_validation_overlap_after_mapping": 0,
        "mapped_total": 2,
        "train_split_ids": 1,
        "validation_split_ids": 1,
        "train_mapped": 1,
        "validation_mapped": 1,
    }
    recon_path = path.parent / "split_reconciliation.json"
    recon_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return recon_path


def _write_reconciliation_per_id(path: Path, train_assays, val_assays) -> Path:
    header = "upstream_split_id\tpartition\tclassification\tmapped_assay_key\n"
    rows = [
        f"{split_id}\t{partition}\tdeterministically_mapped_to_one_frozen_assay_key\tfrozen_{split_id}\n"
        for partition, split_ids in (("train", train_assays), ("validation", val_assays))
        for split_id in split_ids
    ]
    per_id_path = path.parent / "split_reconciliation_per_id.tsv"
    per_id_path.write_text(header + "".join(rows))
    return per_id_path


def _write_config(path: Path, *, hyperparameters=None, train_assays=None, val_assays=None, reconciliation=True) -> None:
    train_assays = ["assay_train"] if train_assays is None else train_assays
    val_assays = ["assay_val"] if val_assays is None else val_assays
    recon_path = _write_reconciliation(path) if reconciliation else None
    per_id_path = _write_reconciliation_per_id(path, train_assays, val_assays) if reconciliation else None
    payload = {
        "artifact_type": "ActFoundProductionConfigV1",
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
        "frozen_input_tsv": str(PINNED_INPUT),
        "frozen_input_sha256": PINNED_INPUT_SHA256,
        "output_dir": str(path.parent / "out"),
        "train_assays": train_assays,
        "val_assays": val_assays,
        "hyperparameters": hyperparameters if hyperparameters is not None else dict(UPSTREAM_DEFAULTS),
    }
    if recon_path is not None:
        payload["split_reconciliation_summary_path"] = str(recon_path)
        payload["split_reconciliation_summary_sha256"] = file_sha256(recon_path)
    if per_id_path is not None:
        payload["split_reconciliation_per_id_path"] = str(per_id_path)
        payload["split_reconciliation_per_id_sha256"] = file_sha256(per_id_path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_production_config_validate_only_contract(tmp_path: Path, capsys):
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    resolved = validate_and_resolve_config(config_path)
    assert resolved["train_assays"] == ["assay_train"]
    assert resolved["val_assays"] == ["assay_val"]
    assert resolved["split_reconciliation"]["summary"]["status"] == "complete"
    assert resolved["mapped_train_assays"] == ["frozen_assay_train"]
    assert resolved["mapped_val_assays"] == ["frozen_assay_val"]

    exit_code = main(["--config", str(config_path), "--validate-only"])
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated"


def test_production_config_fails_closed_without_completed_reconciliation(tmp_path: Path):
    config_path = tmp_path / "config.json"
    _write_config(config_path, reconciliation=False)

    with pytest.raises(ValueError, match="split_reconciliation"):
        validate_and_resolve_config(config_path)


def test_production_config_fails_closed_on_incomplete_reconciliation(tmp_path: Path):
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    payload = json.loads(config_path.read_text())
    recon_path = Path(payload["split_reconciliation_summary_path"])
    recon = json.loads(recon_path.read_text())
    recon["unresolved_total"] = 1
    recon["status"] = "incomplete_unresolved_or_ambiguous_present"
    recon_path.write_text(json.dumps(recon, indent=2, sort_keys=True) + "\n")
    payload["split_reconciliation_summary_sha256"] = file_sha256(recon_path)
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="not complete"):
        validate_and_resolve_config(config_path)


def test_production_config_fails_closed_without_required_split(tmp_path: Path):
    config_path = tmp_path / "config.json"
    _write_config(config_path, train_assays=[])

    with pytest.raises(ValueError, match="train_assays"):
        validate_and_resolve_config(config_path)


def test_production_config_fails_closed_on_hyperparameter_drift(tmp_path: Path):
    config_path = tmp_path / "config.json"
    hyper = dict(UPSTREAM_DEFAULTS)
    hyper["meta_lr"] = 0.123
    _write_config(config_path, hyperparameters=hyper)

    with pytest.raises(ValueError, match="meta_lr"):
        validate_and_resolve_config(config_path)


def test_production_config_fails_closed_on_missing_hyperparameter(tmp_path: Path):
    config_path = tmp_path / "config.json"
    hyper = dict(UPSTREAM_DEFAULTS)
    del hyper["num_updates"]
    _write_config(config_path, hyperparameters=hyper)

    with pytest.raises(ValueError, match="missing"):
        validate_and_resolve_config(config_path)


def test_slurm_wrapper_requires_external_reviewed_config():
    slurm = ROOT / "cyp" / "scripts" / "actfound_production_train.slurm"
    text = slurm.read_text()
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --gres=gpu:nvidia_l40s:1" in text
    assert "ACTFOUND_CONFIG must point" in text
    assert "/scratch/user/mhuang8/actfound/envs/actfound_c96f055" in text
