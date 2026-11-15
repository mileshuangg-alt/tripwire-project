"""Tests for the D013 split/config v2 builder (stand-in loader; real loader runs on CoreHPC)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "cyp" / "scripts"))
import build_split_and_config_v2 as bs  # noqa: E402
import actfound_production_train_v2 as rv2  # noqa: E402


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _setup(tmp: Path, emitted_ok=True):
    corpus = tmp / "corpus.tsv"
    corpus.write_text("x\n")
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"openadmet_firewall_applied": emitted_ok, "output_sha256": _sha(corpus), "output_path": str(corpus)}))
    split = tmp / "split.json"
    split.write_text(json.dumps({"train": ["T1", "T2", "T3"], "valid": ["V1", "V2"], "test": [], "test_external": []}))
    v1 = json.loads((ROOT / "cyp/artifacts/actfound/actfound_chembl32_production_config_v1.json").read_text())
    v1.update({"train_assays": ["T1", "T2", "T3"], "val_assays": ["V1", "V2"], "validation_assays": ["V1", "V2"], "upstream_split_sha256": _sha(split)})
    v1p = tmp / "v1.json"
    v1p.write_text(json.dumps(v1))
    return SimpleNamespace(corpus_tsv=corpus, corpus_manifest=manifest, upstream_split=split, v1_config=v1p,
                           artifact_dir=tmp / "artifacts", training_output_dir=tmp / "train_out"), v1


def test_reconcile_drops_and_logs_in_order():
    kt, kv, d = bs.reconcile(["T1", "T2", "T3"], ["V1", "V2"], ["T3", "T1", "V2", "EXTRA"])
    assert kt == ["T1", "T3"] and kv == ["V2"] and d == [("T2", "train"), ("V1", "validation")]


def test_build_writes_config_that_runner_v2_validates(tmp_path: Path):
    a, v1 = _setup(tmp_path)
    out = bs.build(a, lambda p: ["T1", "T3", "V1", "V2"], rv2.validate_and_resolve_config)
    assert out["train_split_ids"] == 2 and out["dropped_train"] == 1 and out["dropped_validation"] == 0
    cfg = json.loads((a.artifact_dir / "actfound_chembl32_d013_config_v2.json").read_text())
    assert cfg["artifact_type"] == "ActFoundProductionConfigV2" and cfg["train_assays"] == ["T1", "T3"]
    for k in set(v1) - bs.REPLACED_KEYS:
        assert cfg[k] == v1[k]
    resolved = rv2.validate_and_resolve_config(a.artifact_dir / "actfound_chembl32_d013_config_v2.json")
    assert resolved["mapped_train_assays"] == ["T1", "T3"]


def test_build_refuses_unfirewalled_corpus(tmp_path: Path):
    a, _ = _setup(tmp_path, emitted_ok=False)
    with pytest.raises(SystemExit, match="firewall"):
        bs.build(a, lambda p: ["T1", "V1"], None)


def test_build_refuses_empty_split_and_existing_dirs(tmp_path: Path):
    a, _ = _setup(tmp_path)
    with pytest.raises(SystemExit, match="empty"):
        bs.build(a, lambda p: ["T1"], None)
    a.training_output_dir.mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        bs.build(a, lambda p: ["T1", "V1"], None)
