from __future__ import annotations

import torch

from cyp002_model import build_cyp002_model
from cyp007_infrastructure import config_from_trial
from p1_pairwise_chemprop import (
    build_p1_pair_head,
    build_p1_shared_message_passing,
    count_parameters,
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


def _t121_config():
    return config_from_trial(
        name="D007-T121-P1-qualification",
        sampled=T121_SAMPLED,
        seed=20261101,
    )


def test_p1_uses_exact_t121_encoder_configuration():
    config = _t121_config()

    assert config.message_hidden_dim == 600
    assert config.message_passing_depth == 4
    assert config.dropout == 0.4
    assert config.activation == "relu"

    message_passing = build_p1_shared_message_passing(
        config
    )

    assert message_passing.n_components == 2
    assert message_passing.shared is True
    assert len(message_passing.blocks) == 2
    assert message_passing.output_dim == 1200


def test_p1_two_components_share_exact_same_encoder_object():
    config = _t121_config()

    message_passing = build_p1_shared_message_passing(
        config
    )

    assert (
        message_passing.blocks[0]
        is message_passing.blocks[1]
    )

    first_parameters = list(
        message_passing.blocks[0].parameters()
    )
    second_parameters = list(
        message_passing.blocks[1].parameters()
    )

    assert len(first_parameters) == len(second_parameters)

    for first, second in zip(
        first_parameters,
        second_parameters,
        strict=True,
    ):
        assert first is second


def test_p1_shared_encoder_does_not_double_parameter_count():
    config = _t121_config()

    message_passing = build_p1_shared_message_passing(
        config
    )

    shared_counts = count_parameters(
        message_passing
    )

    block_counts = count_parameters(
        message_passing.blocks[0]
    )

    assert shared_counts == block_counts


def test_p1_pair_head_accepts_native_two_component_output_dimension():
    config = _t121_config()

    message_passing = build_p1_shared_message_passing(
        config
    )
    pair_head = build_p1_pair_head(config)

    assert message_passing.output_dim == 1200

    first_linear = pair_head.pair_function[0]

    assert first_linear.in_features == 1200
    assert first_linear.out_features == 2000


def test_t121_and_p1_parameter_counts_are_finite_and_recordable():
    config = _t121_config()

    t121 = build_cyp002_model(
        config,
        variant="stock",
    )

    shared_encoder = build_p1_shared_message_passing(
        config
    )
    pair_head = build_p1_pair_head(config)

    t121_counts = count_parameters(t121)
    encoder_counts = count_parameters(
        shared_encoder
    )
    head_counts = count_parameters(
        pair_head
    )

    p1_total = (
        encoder_counts.total
        + head_counts.total
    )
    p1_trainable = (
        encoder_counts.trainable
        + head_counts.trainable
    )

    assert t121_counts.total > 0
    assert t121_counts.trainable > 0
    assert p1_total > 0
    assert p1_trainable > 0

    assert (
        t121_counts.total
        == t121_counts.trainable
    )
    assert p1_total == p1_trainable

    print()
    print(
        "T121 total/trainable:",
        t121_counts.total,
        t121_counts.trainable,
    )
    print(
        "P1 encoder total/trainable:",
        encoder_counts.total,
        encoder_counts.trainable,
    )
    print(
        "P1 head total/trainable:",
        head_counts.total,
        head_counts.trainable,
    )
    print(
        "P1 total/trainable:",
        p1_total,
        p1_trainable,
    )


def test_real_molecular_graphs_flow_through_shared_encoder_and_pair_head():
    from rdkit import Chem
    from chemprop import data, featurizers, nn

    config = _t121_config()

    message_passing = build_p1_shared_message_passing(
        config
    )

    aggregation = nn.NormAggregation(
        norm=config.aggregation_norm,
    )

    pair_head = build_p1_pair_head(config)
    pair_head.eval()

    # Synthetic qualification molecules only.
    smiles_a = [
        "CCO",
        "CCN",
        "c1ccccc1",
    ]

    smiles_b = [
        "CCN",
        "c1ccccc1",
        "CCO",
    ]

    mols_a = [
        Chem.MolFromSmiles(smiles)
        for smiles in smiles_a
    ]
    mols_b = [
        Chem.MolFromSmiles(smiles)
        for smiles in smiles_b
    ]

    assert all(mol is not None for mol in mols_a)
    assert all(mol is not None for mol in mols_b)

    featurizer = (
        featurizers.SimpleMoleculeMolGraphFeaturizer()
    )

    dataset_a = data.MoleculeDataset(
        [
            data.MoleculeDatapoint(mol)
            for mol in mols_a
        ],
        featurizer=featurizer,
    )

    dataset_b = data.MoleculeDataset(
        [
            data.MoleculeDatapoint(mol)
            for mol in mols_b
        ],
        featurizer=featurizer,
    )

    loader_a = data.build_dataloader(
        dataset_a,
        batch_size=3,
        num_workers=0,
        shuffle=False,
    )

    loader_b = data.build_dataloader(
        dataset_b,
        batch_size=3,
        num_workers=0,
        shuffle=False,
    )

    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))

    # Chemprop TrainingBatch position 0 is the BatchMolGraph.
    bmg_a = batch_a[0]
    bmg_b = batch_b[0]

    message_passing.eval()

    with torch.no_grad():
        atom_embeddings = message_passing(
            [bmg_a, bmg_b]
        )

        assert len(atom_embeddings) == 2

        h_a = aggregation(
            atom_embeddings[0],
            bmg_a.batch,
        )

        h_b = aggregation(
            atom_embeddings[1],
            bmg_b.batch,
        )

        paired_fingerprint = torch.cat(
            (h_a, h_b),
            dim=1,
        )

        predictions = pair_head(
            h_a,
            h_b,
        )

    assert h_a.shape == (3, 600)
    assert h_b.shape == (3, 600)

    assert paired_fingerprint.shape == (
        3,
        1200,
    )

    assert predictions.shape == (3, 4)

    assert torch.isfinite(h_a).all()
    assert torch.isfinite(h_b).all()
    assert torch.isfinite(
        paired_fingerprint
    ).all()
    assert torch.isfinite(predictions).all()
