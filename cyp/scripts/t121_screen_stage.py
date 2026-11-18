
"""Approved screening-stage schedule and selection for updated T121."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import torch
from lightning import pytorch as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from torch.utils.data import DataLoader, Subset

from chemprop import data
from t121_weighted_training import (
    T121ScreenTrainingMPNN,
    build_t121_screen_prefit_state,
    collate_uncertainty_batch,
    prepare_t121_pic50_transition,
)

SCREEN_EPOCHS = 50
SPLIT_SEED = 20261001


def screening_split(molecule_ids, smiles):
    """Deterministic whole-Butina-cluster split, nearest shuffled prefix to 10%."""
    ids = tuple(str(value) for value in molecule_ids)
    if len(ids) != len(smiles) or len(set(ids)) != len(ids):
        raise ValueError("Screen identities/structures are not uniquely aligned.")
    if len(ids) < 2:
        raise ValueError("At least two compounds are required.")

    # Canonical input ordering makes assignments independent of loader ordering.
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    ordered_ids = [ids[i] for i in order]
    fingerprints = []
    for i in order:
        mol = Chem.MolFromSmiles(smiles[i])
        if mol is None:
            raise ValueError(f"Fingerprint parse failure: {ids[i]}")
        fingerprints.append(
            AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, nBits=2048, useChirality=False
            )
        )

    distances = []
    for i in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[i], fingerprints[:i]
        )
        distances.extend(1.0 - value for value in similarities)

    clusters = Butina.ClusterData(
        distances, len(ids), 0.30, isDistData=True, reordering=False
    )
    # Stable cluster identifiers independent of Butina's returned cluster order.
    clusters = sorted(
        (tuple(sorted(cluster)) for cluster in clusters),
        key=lambda cluster: tuple(ordered_ids[i] for i in cluster),
    )
    if len(clusters) < 2:
        raise ValueError("Only one structural cluster: whole-cluster split impossible.")

    shuffled = np.random.RandomState(SPLIT_SEED).permutation(len(clusters))
    cumulative = np.cumsum([len(clusters[i]) for i in shuffled])
    target = 0.10 * len(ids)
    # Exclude empty validation and empty training. Exact distance tie:
    # choose the earlier prefix.
    prefix = min(
        range(1, len(clusters)),
        key=lambda k: (abs(float(cumulative[k - 1]) - target), k),
    )
    validation_clusters = set(int(i) for i in shuffled[:prefix])

    result = {}
    for cluster_id, members in enumerate(clusters):
        partition = "validation" if cluster_id in validation_clusters else "train"
        for member in members:
            result[ordered_ids[member]] = {
                "cluster_id": cluster_id,
                "partition": partition,
            }
    return result


def native_screen_score(predictions, targets, raw_weights):
    """Equal mean of four native-log2fc, inverse-SE² weighted MSEs."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(raw_weights, dtype=np.float64)
    if (
        predictions.shape != targets.shape
        or weights.shape != targets.shape
        or targets.ndim != 2
        or targets.shape[1] != 4
    ):
        raise ValueError("Expected aligned [compound, four enzymes] matrices.")

    scores = []
    for task in range(4):
        observed = np.isfinite(targets[:, task])
        if not observed.any():
            raise ValueError(f"No validation observations for enzyme index {task}.")
        p = predictions[observed, task]
        y = targets[observed, task]
        w = weights[observed, task]
        if not np.isfinite(p).all():
            raise ValueError(f"Non-finite validation predictions: enzyme {task}.")
        if not np.isfinite(w).all() or not (w > 0).all():
            raise ValueError(f"Invalid observed validation weights: enzyme {task}.")
        denominator = w.sum()
        numerator = np.sum(w * np.square(p - y))
        if not np.isfinite(denominator) or not np.isfinite(numerator):
            raise ValueError("Non-finite screening metric accumulation.")
        scores.append(float(numerator / denominator))
    result = {"macro": float(np.mean(scores)), "per_enzyme": scores}
    return result


def select_screen_epoch(records):
    """Minimum score; exact tie chooses earlier zero-based epoch."""
    if not records:
        raise ValueError("No screening checkpoint records.")
    epochs = [int(record["epoch"]) for record in records]
    if len(set(epochs)) != len(epochs):
        raise ValueError("Duplicate screening epoch records.")
    for record in records:
        if not np.isfinite(record["score"]):
            raise ValueError("Non-finite screening selection score.")
    result = min(records, key=lambda record: (record["score"], record["epoch"]))
    return result


class ValidatedScreenMPNN(T121ScreenTrainingMPNN):
    """Native-unit screening validation; no pIC50 validation hooks."""

    def on_validation_epoch_start(self):
        self._screen_validation = []

    def validation_step(self, batch, batch_idx):
        molecular = batch["chemprop_batch"]
        bmg, V_d, X_d, targets, _, _, _ = molecular
        # No screening scaling is fitted: model outputs and labels are native.
        predictions = self(bmg, V_d, X_d)
        self._screen_validation.append((
            predictions.detach().cpu(),
            targets.detach().cpu(),
            batch["raw_observation_weights"].detach().cpu(),
        ))

    def on_validation_epoch_end(self):
        if not self._screen_validation:
            raise ValueError("Screening validation produced no batches.")
        predictions, targets, weights = zip(*self._screen_validation)
        score = native_screen_score(
            torch.cat(predictions).numpy(),
            torch.cat(targets).numpy(),
            torch.cat(weights).numpy(),
        )
        self.screen_validation_result = score
        self.log("screen_validation_macro", score["macro"], logger=True)
        self._screen_validation = []


class ScreenEpochCheckpoints(pl.Callback):
    """Save every epoch; record selection without early stopping."""

    def __init__(self, directory):
        super().__init__()
        self.directory = Path(directory)
        self.records = []

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        if any(record["epoch"] == epoch for record in self.records):
            raise ValueError(f"Duplicate validation/checkpoint event: epoch {epoch}.")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"epoch_{epoch:03d}.ckpt"
        if path.exists():
            raise ValueError(f"Refusing to overwrite screening checkpoint: {path}")
        trainer.save_checkpoint(path)
        score = pl_module.screen_validation_result
        self.records.append({
            "epoch": epoch,
            "score": score["macro"],
            "per_enzyme": score["per_enzyme"],
            "checkpoint": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        payload = {
            "epoch_numbering": "zero_based",
            "records": self.records,
            "selected": select_screen_epoch(self.records),
        }
        (self.directory / "selection.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


def prepare_screen_stage(
    *,
    screen_path,
    identity_path,
    similarity_path,
    contract_path,
    proposal_path,
    output_dir,
    training_seed,
):
    """Prepare split, model and loaders; does not construct a Trainer or fit."""
    proposal_path = Path(proposal_path)
    proposal = json.loads(proposal_path.read_text())
    if proposal.get("trial_index") != 121:
        raise ValueError("Expected the actual trial_121 proposal.")
    sampled = proposal["sampled_config"]
    required = {
        "batch_size", "dropout", "ffn_hidden_dim", "ffn_num_layers",
        "message_hidden_dim", "message_passing_depth", "warmup_epochs",
        "max_lr", "init_lr_ratio", "final_lr_ratio",
    }
    missing = required - set(sampled)
    if missing:
        raise ValueError(f"Missing trial-specific settings: {sorted(missing)}")
    if not 0 < sampled["warmup_epochs"] < SCREEN_EPOCHS:
        raise ValueError("Trial warmup contradicts the 50-epoch screening schedule.")

    out = Path(output_dir)
    if out.exists():
        raise ValueError(f"Refusing to overwrite screening preparation: {out}")

    base = build_t121_screen_prefit_state(
        screen_path=screen_path,
        identity_path=identity_path,
        similarity_path=similarity_path,
        contract_path=contract_path,
        sampled_config=sampled,
        config_name="D007-T121",
        training_seed=training_seed,
    )
    dataset = base["training_dataset"]

    # Recover exact source SMILES by identity from the hash-verified source CSV.
    with Path(screen_path).open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    smiles = {}
    for row in source_rows:
        name = row["Molecule_Name"]
        if name in smiles and smiles[name] != row["SMILES"]:
            raise ValueError(f"Inconsistent screening SMILES: {name}")
        smiles[name] = row["SMILES"]

    assignments = screening_split(
        dataset.molecule_ids,
        [smiles[name] for name in dataset.molecule_ids],
    )
    indices = {
        partition: [
            i for i, name in enumerate(dataset.molecule_ids)
            if assignments[name]["partition"] == partition
        ]
        for partition in ("train", "validation")
    }

    loaders = {}
    for partition in ("train", "validation"):
        molecular_subset = Subset(
            dataset.molecular_dataset, indices[partition]
        )
        # Reuse Chemprop's sampler; attach the same-index metadata subset.
        ordinary = data.build_dataloader(
            molecular_subset,
            batch_size=base["config"].batch_size,
            num_workers=0,
            seed=training_seed,
            shuffle=partition == "train",
        )
        loaders[partition] = DataLoader(
            Subset(dataset, indices[partition]),
            batch_sampler=ordinary.batch_sampler,
            num_workers=0,
            collate_fn=collate_uncertainty_batch,
        )

    old = base["model"]
    config = base["config"]
    model = ValidatedScreenMPNN(
        message_passing=old.message_passing,
        agg=old.agg,
        predictor=old.predictor,
        batch_norm=config.batch_norm,
        warmup_epochs=config.warmup_epochs,
        init_lr=config.initial_lr,
        max_lr=config.max_lr,
        final_lr=config.final_lr,
    )
    model.load_state_dict(old.state_dict(), strict=True)

    out.mkdir(parents=True)
    with (out / "split_assignments.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule_id", "cluster_id", "partition"])
        for name in sorted(assignments):
            writer.writerow([
                name,
                assignments[name]["cluster_id"],
                assignments[name]["partition"],
            ])

    metadata = {
        "counts": {key: len(value) for key, value in indices.items()},
        "split_seed": SPLIT_SEED,
        "fingerprint": {
            "radius": 2, "n_bits": 2048, "useChirality": False,
        },
        "clustering": {
            "method": "Butina", "distance_cutoff": 0.30,
            "reordering": False,
        },
        "validation_allocation": "nearest shuffled whole-cluster prefix to 10%",
        "epochs": SCREEN_EPOCHS,
        "target_normalization": "none; native log2fc retained",
        "proposal_path": str(proposal_path),
        "proposal_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        "sampled_config": sampled,
        "full_screen_refit": False,
    }
    (out / "preparation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    result = {
        "model": model,
        "train_loader": loaders["train"],
        "validation_loader": loaders["validation"],
        "metadata": metadata,
        "output_dir": out,
        "sampled_config": sampled,
        "training_seed": training_seed,
    }
    return result


def fit_screen_stage(prepared, *, authorize_training=False):
    if authorize_training is not True:
        raise RuntimeError("Screening training is not authorized.")
    callback = ScreenEpochCheckpoints(
        prepared["output_dir"] / "epoch_checkpoints"
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=SCREEN_EPOCHS,
        callbacks=[callback],
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        deterministic=True,
        default_root_dir=prepared["output_dir"],
    )
    trainer.fit(
        prepared["model"],
        train_dataloaders=prepared["train_loader"],
        val_dataloaders=prepared["validation_loader"],
    )
    if len(callback.records) != SCREEN_EPOCHS:
        raise ValueError("Screening did not save all 50 epoch checkpoints.")
    result = select_screen_epoch(callback.records)
    return result


def prepare_selected_screen_handoff(
    *,
    prepared_screen,
    selected_record,
    training_dataframe,
    validation_dataframe,
    config_name,
    training_seed,
):
    """Load the selected screen state only; no optimizer/scheduler resume."""
    path = Path(selected_record["checkpoint"])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != selected_record["sha256"]:
        raise ValueError("Selected screening checkpoint checksum mismatch.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    prepared_screen["model"].load_state_dict(
        checkpoint["state_dict"], strict=True
    )
    result = prepare_t121_pic50_transition(
        screen_model=prepared_screen["model"],
        training_dataframe=training_dataframe,
        validation_dataframe=validation_dataframe,
        sampled_config=prepared_screen["sampled_config"],
        config_name=config_name,
        training_seed=training_seed,
    )
    return result
