from __future__ import annotations

from dataclasses import dataclass

from chemprop import nn as chemprop_nn
import torch
from torch import Tensor, nn

from cyp002_config import CYP002ModelConfig, CYP002_TASK_NAMES


P2_EMBEDDING_DIM = 600
P2_HIDDEN_DIM = 2000
P2_N_TASKS = 4
P2_DROPOUT = 0.4
P2_INNER_LR = 0.01


@dataclass(frozen=True)
class P2ParameterCounts:
    total: int
    trainable: int


class P2MetaHead(nn.Module):
    """D010 P2 shared hidden layer plus four-output regression head."""

    def __init__(self) -> None:
        super().__init__()
        self.shared_hidden = nn.Linear(P2_EMBEDDING_DIM, P2_HIDDEN_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(P2_DROPOUT)
        self.output = nn.Linear(P2_HIDDEN_DIM, P2_N_TASKS)

    def features(self, embeddings: Tensor) -> Tensor:
        _validate_embeddings(embeddings)
        features = self.dropout(self.activation(self.shared_hidden(embeddings)))
        if not torch.isfinite(features).all():
            raise FloatingPointError("P2 shared features are non-finite.")
        return features

    def forward(self, embeddings: Tensor) -> Tensor:
        predictions = self.output(self.features(embeddings))
        if not torch.isfinite(predictions).all():
            raise FloatingPointError("P2 predictions are non-finite.")
        return predictions


def _validate_embeddings(embeddings: Tensor) -> None:
    if embeddings.ndim != 2 or embeddings.shape[1] != P2_EMBEDDING_DIM:
        raise ValueError("P2 embeddings must have shape (batch, 600).")
    if not torch.isfinite(embeddings).all():
        raise ValueError("P2 embeddings must be finite.")


def selected_task_prediction(
    features: Tensor,
    *,
    task_index: int,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    if task_index < 0 or task_index >= P2_N_TASKS:
        raise ValueError("task_index is outside the frozen four-task range.")
    if features.ndim != 2 or features.shape[1] != P2_HIDDEN_DIM:
        raise ValueError("P2 features must have shape (batch, 2000).")
    if weight.shape != (P2_HIDDEN_DIM,):
        raise ValueError("selected task weight must have shape (2000,).")
    if bias.shape not in ((), (1,)):
        raise ValueError("selected task bias must be scalar.")
    result = features @ weight + bias.reshape(())
    if not torch.isfinite(result).all():
        raise FloatingPointError("selected task prediction is non-finite.")
    return result


def standardized_mse(predictions: Tensor, targets: Tensor) -> Tensor:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes.")
    if predictions.ndim != 1:
        raise ValueError("D010 support/query losses expect one-dimensional task vectors.")
    if predictions.numel() == 0:
        raise ValueError("D010 task loss cannot be empty.")
    if not torch.isfinite(targets).all():
        raise ValueError("D010 task targets must be finite; missing labels are not imputed.")
    loss = (predictions - targets).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("D010 standardized MSE is non-finite.")
    return loss


def adapt_selected_output_exact(
    meta_head: P2MetaHead,
    support_embeddings: Tensor,
    support_targets_z: Tensor,
    *,
    task_index: int,
    inner_lr: float = P2_INNER_LR,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return exact differentiable one-step ANIL-adapted w_t/b_t."""
    if support_targets_z.shape != (13,):
        raise ValueError("D010 meta-training support target shape must be exactly (13,).")
    if inner_lr != P2_INNER_LR:
        raise ValueError("D010 freezes the inner learning rate at 0.01.")
    features = meta_head.features(support_embeddings)
    weight = meta_head.output.weight[task_index]
    bias = meta_head.output.bias[task_index]
    support_pred = selected_task_prediction(
        features,
        task_index=task_index,
        weight=weight,
        bias=bias,
    )
    support_loss = standardized_mse(support_pred, support_targets_z)
    grad_w, grad_b = torch.autograd.grad(
        support_loss,
        (weight, bias),
        create_graph=True,
        retain_graph=True,
    )
    return weight - inner_lr * grad_w, bias - inner_lr * grad_b, support_loss


def query_loss_after_exact_adaptation(
    meta_head: P2MetaHead,
    support_embeddings: Tensor,
    support_targets_z: Tensor,
    query_embeddings: Tensor,
    query_targets_z: Tensor,
    *,
    task_index: int,
) -> tuple[Tensor, Tensor]:
    if query_targets_z.shape != (3,):
        raise ValueError("D010 meta-training query target shape must be exactly (3,).")
    adapted_w, adapted_b, support_loss = adapt_selected_output_exact(
        meta_head,
        support_embeddings,
        support_targets_z,
        task_index=task_index,
    )
    query_features = meta_head.features(query_embeddings)
    query_pred = selected_task_prediction(
        query_features,
        task_index=task_index,
        weight=adapted_w,
        bias=adapted_b,
    )
    return standardized_mse(query_pred, query_targets_z), support_loss


def build_p2_message_passing(config: CYP002ModelConfig) -> chemprop_nn.BondMessagePassing:
    if config.message_passing_depth != 4:
        raise ValueError("D010 P2 requires message-passing depth 4.")
    if config.message_hidden_dim != P2_EMBEDDING_DIM:
        raise ValueError("D010 P2 requires representation dimension 600.")
    if config.dropout != P2_DROPOUT:
        raise ValueError("D010 P2 requires dropout 0.4.")
    if config.activation != "relu":
        raise ValueError("D010 P2 requires ReLU activation.")
    if config.batch_norm:
        raise ValueError("D010 P2 forbids batch normalization.")
    if config.aggregation != "norm" or config.aggregation_norm != 100:
        raise ValueError("D010 P2 requires normalized aggregation with norm 100.")
    return chemprop_nn.BondMessagePassing(
        d_h=config.message_hidden_dim,
        depth=config.message_passing_depth,
        dropout=config.dropout,
        activation=config.activation,
    )


def build_p2_aggregation(config: CYP002ModelConfig) -> chemprop_nn.NormAggregation:
    if config.aggregation != "norm" or config.aggregation_norm != 100:
        raise ValueError("D010 P2 requires normalized aggregation with norm 100.")
    return chemprop_nn.NormAggregation(norm=config.aggregation_norm)


def build_p2_meta_head(config: CYP002ModelConfig) -> P2MetaHead:
    if config.ffn_hidden_dim != P2_HIDDEN_DIM:
        raise ValueError("D010 P2 requires shared hidden dimension 2000.")
    if config.ffn_num_layers != 1:
        raise ValueError("D010 P2 requires exactly one shared hidden layer.")
    if config.dropout != P2_DROPOUT:
        raise ValueError("D010 P2 requires dropout 0.4.")
    if config.num_tasks != P2_N_TASKS or tuple(CYP002_TASK_NAMES) != CYP002_TASK_NAMES:
        raise ValueError("D010 P2 requires the frozen four CYP tasks.")
    return P2MetaHead()


def count_parameters(module: nn.Module) -> P2ParameterCounts:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return P2ParameterCounts(total=total, trainable=trainable)
