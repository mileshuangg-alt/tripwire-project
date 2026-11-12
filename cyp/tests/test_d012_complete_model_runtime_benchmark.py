from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from d012_complete_model_runtime_benchmark import (  # noqa: E402
    D012ScalarHead,
    MEASURED_COMPLETE_UPDATES,
    WARMUP_COMPLETE_UPDATES,
    d012_config,
    molecule_dataset,
    run_benchmark,
)
from p2_anil_model import build_p2_aggregation, build_p2_message_passing  # noqa: E402


def test_complete_model_protocol_constants() -> None:
    assert WARMUP_COMPLETE_UPDATES == 10
    assert MEASURED_COMPLETE_UPDATES == 100


def test_d012_architecture_is_chemprop_dmpnn_not_ecfp4() -> None:
    source = (SCRIPTS / "d012_complete_model_runtime_benchmark.py").read_text(encoding="utf-8")

    assert "from ecfp4" not in source.lower()
    assert "import ecfp4" not in source.lower()
    assert "MoleculeDatapoint.from_smi" in source
    assert "build_p2_message_passing" in source

    config = d012_config(20261101)
    message_passing = build_p2_message_passing(config)
    aggregation = build_p2_aggregation(config)
    head = D012ScalarHead()

    assert message_passing.depth == 4
    assert message_passing.W_i.out_features == 600
    assert aggregation.norm == 100
    assert head.hidden.in_features == 600
    assert head.hidden.out_features == 2000
    assert head.dropout.p == 0.4
    assert head.output.out_features == 1


def test_molecular_graph_dataset_constructs_from_smiles() -> None:
    dataset = molecule_dataset(["CCO", "c1ccccc1"], names=["a", "b"])

    assert len(dataset) == 2
    assert dataset.data[0].name == "a"
    assert dataset.data[1].name == "b"


def test_projection_formula_without_running_benchmark() -> None:
    # This checks the constants and architecture-bearing code import without
    # executing the CUDA benchmark locally.
    config = d012_config(20261101)
    assert config.batch_size == 16
    assert config.max_epochs == 50
    assert config.warmup_epochs == 10
    assert config.message_passing_depth == 4
    assert config.message_hidden_dim == 600
    assert config.ffn_hidden_dim == 2000
