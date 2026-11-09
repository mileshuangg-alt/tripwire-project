from __future__ import annotations

from dataclasses import dataclass

from chemprop import nn

from cyp002_config import CYP002ModelConfig
from p1_pairwise_model import P1AntisymmetricPairHead


P1_N_COMPONENTS = 2


@dataclass(frozen=True)
class P1ParameterCounts:
    total: int
    trainable: int


def build_p1_shared_message_passing(
    config: CYP002ModelConfig,
) -> nn.MulticomponentMessagePassing:
    """
    Build the D009 P1 shared two-component Chemprop encoder.

    Both molecular components use the exact same BondMessagePassing module
    object. This implements h_A = E_theta(A), h_B = E_theta(B).
    """
    if config.message_hidden_dim != 600:
        raise ValueError(
            "D009 P1 requires message_hidden_dim=600."
        )

    if config.message_passing_depth != 4:
        raise ValueError(
            "D009 P1 requires message_passing_depth=4."
        )

    if config.dropout != 0.4:
        raise ValueError(
            "D009 P1 requires dropout=0.4."
        )

    if config.activation != "relu":
        raise ValueError(
            "D009 P1 requires the inherited T121 activation='relu'."
        )

    shared_block = nn.BondMessagePassing(
        d_h=config.message_hidden_dim,
        depth=config.message_passing_depth,
        dropout=config.dropout,
        activation=config.activation,
    )

    message_passing = nn.MulticomponentMessagePassing(
        blocks=[shared_block],
        n_components=P1_N_COMPONENTS,
        shared=True,
    )

    result = message_passing
    return result


def build_p1_pair_head(
    config: CYP002ModelConfig,
) -> P1AntisymmetricPairHead:
    """Build the already-qualified D009 pair head after config validation."""
    if config.message_hidden_dim != 600:
        raise ValueError(
            "D009 P1 requires 600-dimensional molecular embeddings."
        )

    if config.ffn_hidden_dim != 2000:
        raise ValueError(
            "D009 P1 requires ffn_hidden_dim=2000."
        )

    if config.ffn_num_layers != 1:
        raise ValueError(
            "D009 P1 requires ffn_num_layers=1."
        )

    if config.dropout != 0.4:
        raise ValueError(
            "D009 P1 requires dropout=0.4."
        )

    if config.activation != "relu":
        raise ValueError(
            "D009 P1 requires the inherited T121 activation='relu'."
        )

    result = P1AntisymmetricPairHead()
    return result


def count_parameters(module) -> P1ParameterCounts:
    """Count total and trainable parameters without changing module state."""
    total = sum(
        parameter.numel()
        for parameter in module.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

    result = P1ParameterCounts(
        total=total,
        trainable=trainable,
    )
    return result
