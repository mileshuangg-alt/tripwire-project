"""D022: pretrained CheMeleon encoder in the updated-T121 two-stage pipeline.

Opt-in only. Nothing here runs unless a loaded ChemeleonArm is passed to the
T121 builders; with none, every existing path is unchanged (golden-hash
guarded in the descriptor-arm tests).

Frozen settings live in cyp/artifacts/d022_chemeleon/chemeleon_config.json and
are pinned by its SHA-256; the weights are pinned by the SHA-256 inside it.
The encoder is injected into an already-built updated-T121 model (the frozen
D007 search space cannot express d_h 2048). The head keeps updated T121's
width, depth and dropout (checked against the frozen file), re-initialised for
the 2048-wide input and seeded for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_KEYS = {
    "decision", "weights", "encoder", "aggregation", "head", "batch_size",
    "head_lr", "encoder_lr_ratio", "descriptors",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChemeleonArm:
    """Verified CheMeleon settings plus checkpoint tensors."""

    def __init__(self, config, config_sha256, weights_path, weights_sha256,
                 hyper_parameters, state_dict):
        self.config = config
        self.config_sha256 = config_sha256
        self.weights_path = str(weights_path)
        self.weights_sha256 = weights_sha256
        self.hyper_parameters = hyper_parameters
        self.state_dict = state_dict

    @property
    def encoder_lr_ratio(self):
        return float(self.config["encoder_lr_ratio"])

    @classmethod
    def load(cls, config_path, weights_path, *, expected_config_sha256):
        import torch

        config_path = Path(config_path)
        config_sha = _sha256(config_path)
        if config_sha != expected_config_sha256:
            raise ValueError(f"CheMeleon config hash mismatch: {config_sha}")
        config = json.loads(config_path.read_text())
        if set(config) != EXPECTED_KEYS:
            raise ValueError(f"Unexpected CheMeleon config keys: {sorted(config)}")
        if config["aggregation"] != "mean" or config["descriptors"] is not False:
            raise ValueError("D022 fixes mean aggregation and no descriptors.")
        weights_path = Path(weights_path)
        weights_sha = _sha256(weights_path)
        if weights_sha != config["weights"]["sha256"]:
            raise ValueError(f"CheMeleon weights hash mismatch: {weights_sha}")
        checkpoint = torch.load(weights_path, weights_only=True, map_location="cpu")
        if set(checkpoint) != {"hyper_parameters", "state_dict"}:
            raise ValueError(f"Unexpected checkpoint keys: {sorted(checkpoint)}")
        hp = dict(checkpoint["hyper_parameters"])
        for key, value in config["encoder"].items():
            if hp.get(key) != value:
                raise ValueError(f"checkpoint {key}={hp.get(key)!r}; expected {value!r}")
        return cls(config, config_sha, weights_path, weights_sha, hp,
                   checkpoint["state_dict"])

    def provenance(self):
        return {
            "decision": self.config["decision"],
            "config_sha256": self.config_sha256,
            "weights_sha256": self.weights_sha256,
            "encoder_lr_ratio": self.encoder_lr_ratio,
            "aggregation": self.config["aggregation"],
            "head": self.config["head"],
        }


class EncoderRatioOptimizerMixin:
    """Chemprop's optimizer and Noam-like schedule, with an optional separate
    encoder group at `encoder_lr_ratio` times the head rate throughout.

    encoder_lr_ratio None (the default) delegates to Chemprop unchanged.
    The single LambdaLR scales each group's own starting rate, so the encoder
    group (starting at ratio * init_lr) follows ratio * the head schedule at
    every step.
    """

    encoder_lr_ratio = None

    def configure_optimizers(self):
        if self.encoder_lr_ratio is None:
            return super().configure_optimizers()
        from torch import optim
        from chemprop.schedulers import build_NoamLike_LRSched

        encoder_params = list(self.message_passing.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        head_params = [p for p in self.parameters() if id(p) not in encoder_ids]
        if not encoder_params or not head_params:
            raise ValueError("Encoder and head parameter groups must both be non-empty.")
        ratio = float(self.encoder_lr_ratio)
        opt = optim.Adam([
            {"params": head_params, "lr": self.init_lr},
            {"params": encoder_params, "lr": ratio * self.init_lr},
        ])
        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            raise ValueError("D022 requires a finite max_epochs.")
        cooldown_steps = (self.trainer.max_epochs - self.warmup_epochs) * steps_per_epoch
        scheduler = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def check_recipe(arm, config):
    """The carried-over updated-T121 recipe must equal the frozen D022 file."""
    head = arm.config["head"]
    lr = arm.config["head_lr"]
    actual = {
        "ffn_hidden_dim": config.ffn_hidden_dim,
        "ffn_num_layers": config.ffn_num_layers,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "initial_lr": config.initial_lr,
        "max_lr": config.max_lr,
        "final_lr": config.final_lr,
        "warmup_epochs": config.warmup_epochs,
    }
    expected = {
        "ffn_hidden_dim": head["ffn_hidden_dim"],
        "ffn_num_layers": head["ffn_num_layers"],
        "dropout": head["dropout"],
        "batch_size": arm.config["batch_size"],
        "initial_lr": lr["init"],
        "max_lr": lr["peak"],
        "final_lr": lr["final"],
        "warmup_epochs": lr["warmup_epochs"],
    }
    if actual != expected:
        raise ValueError(f"Updated-T121 recipe differs from frozen D022: {actual} vs {expected}")


def apply_chemeleon(model, arm, config, seed):
    """Replace encoder, aggregation and head of an unfitted model in place.

    The criterion and output transform of the existing predictor are kept, so
    screening stays native-unit and pIC50 keeps its fold scaler. Returns a
    report for provenance.
    """
    import torch
    from chemprop import nn

    check_recipe(arm, config)
    if getattr(model, "_trainer", None) is not None:
        raise ValueError("CheMeleon can only be applied to an unfitted model.")
    old = model.predictor
    encoder = nn.BondMessagePassing(**arm.hyper_parameters)
    encoder.load_state_dict(arm.state_dict, strict=True)
    torch.manual_seed(seed)
    head = nn.RegressionFFN(
        n_tasks=config.num_tasks,
        input_dim=encoder.output_dim,
        hidden_dim=config.ffn_hidden_dim,
        n_layers=config.ffn_num_layers,
        dropout=config.dropout,
        activation=config.activation,
        criterion=old.criterion,
        output_transform=old.output_transform,
    )
    if encoder.output_dim != arm.config["head"]["input_dim"]:
        raise ValueError("Encoder output width differs from the frozen head input.")
    model.message_passing = encoder
    model.agg = nn.MeanAggregation()
    model.predictor = head
    model.encoder_lr_ratio = arm.encoder_lr_ratio
    return arm.provenance()
