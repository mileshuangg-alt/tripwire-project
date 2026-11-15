from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "cyp" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from actfound_production_train_v2 import (  # noqa: E402
    CONFIG_ARTIFACT_TYPE,
    PINNED_UPSTREAM_COMMIT,
    UPSTREAM_DEFAULTS,
    file_sha256,
    main,
    validate_and_resolve_config,
    run_training,
)

FROZEN_TSV = "/scratch/user/mhuang8/actfound/chembl32/d013_firewalled/actfound_chembl32_d013_firewalled.tsv"
FROZEN_SHA = "ab" * 32


def _write_reconciliation(path: Path) -> Path:
    payload = {
        "artifact_type": "ActFoundSplitReconciliationV1",
        "status": "complete",
        "frozen_dataset_sha256": FROZEN_SHA,
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


def _write_manifest(path: Path, *, firewall=True, output_sha=FROZEN_SHA, output_path=FROZEN_TSV) -> Path:
    m = path.parent / "corpus_manifest.json"
    m.write_text(json.dumps({"artifact_type": "D013_firewalled_corpus_manifest_v1", "openadmet_firewall_applied": firewall,
                             "output_sha256": output_sha, "output_path": output_path}, indent=2, sort_keys=True) + "\n")
    return m


def _write_config(path: Path, *, hyperparameters=None, train_assays=None, val_assays=None, reconciliation=True, manifest=None) -> None:
    manifest = _write_manifest(path) if manifest is None else manifest
    train_assays = ["assay_train"] if train_assays is None else train_assays
    val_assays = ["assay_val"] if val_assays is None else val_assays
    recon_path = _write_reconciliation(path) if reconciliation else None
    per_id_path = _write_reconciliation_per_id(path, train_assays, val_assays) if reconciliation else None
    payload = {
        "artifact_type": CONFIG_ARTIFACT_TYPE,
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
        "frozen_input_tsv": FROZEN_TSV,
        "frozen_input_sha256": FROZEN_SHA,
        "output_dir": str(path.parent / "out"),
        "train_assays": train_assays,
        "val_assays": val_assays,
        "hyperparameters": hyperparameters if hyperparameters is not None else dict(UPSTREAM_DEFAULTS),
        "corpus_manifest_path": str(manifest),
        "corpus_manifest_sha256": file_sha256(manifest),
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
    slurm = ROOT / "cyp" / "scripts" / "actfound_production_train_v2.slurm"
    text = slurm.read_text()
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --gres=gpu:nvidia_l40s:1" in text
    assert "ACTFOUND_CONFIG must point" in text
    assert "PROJECT_ROOT must be set explicitly" in text and "ACTFOUND_ENV must be set explicitly" in text
    assert "#SBATCH --no-requeue" in text and "--mail-type" not in text and "--requeue\n" not in text
    assert "actfound_production_train_v2.py" in text
    for var in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
        assert f'export {var}="${{SCRATCH_RUNTIME}}' in text
    assert "wrapper_sha256=" in text and "runner_sha256=" in text and "config_sha256=" in text


def test_v2_rejects_v1_config(tmp_path: Path):
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    payload = json.loads(config_path.read_text())
    payload["artifact_type"] = "ActFoundProductionConfigV1"
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ActFoundProductionConfigV2"):
        validate_and_resolve_config(config_path)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"firewall": False}, "applied OpenADMET firewall"),
        ({"output_sha": "cd" * 32}, "output_sha256"),
        ({"output_path": "/scratch/user/mhuang8/actfound/chembl32/graph_production_20261113/actfound_chembl32_graph_qualified.tsv"}, "output_path"),
    ],
)
def test_v2_fails_closed_on_unfirewalled_or_mismatched_corpus(tmp_path: Path, kwargs, match):
    config_path = tmp_path / "config.json"
    manifest = _write_manifest(config_path, **kwargs)
    _write_config(config_path, manifest=manifest)
    with pytest.raises(RuntimeError, match=match):
        validate_and_resolve_config(config_path)


def test_v2_fails_closed_on_tampered_manifest(tmp_path: Path):
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    manifest = Path(json.loads(config_path.read_text())["corpus_manifest_path"])
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(RuntimeError, match="manifest SHA256"):
        validate_and_resolve_config(config_path)


def test_v2_refuses_existing_output_dir_before_any_work(tmp_path: Path):
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    resolved = validate_and_resolve_config(config_path)
    Path(resolved["output_dir"]).mkdir()
    with pytest.raises(FileExistsError, match="existing output directory"):
        run_training(resolved, tmp_path, tmp_path)
