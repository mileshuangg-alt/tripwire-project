"""Opt-in updated-T121 training components; historical CYP002 stays unchanged."""
from __future__ import annotations

import numpy as np
import torch
from chemprop import data

from cyp002_validation import CYP002ValidationMPNN
from t121_chemeleon import EncoderRatioOptimizerMixin, apply_chemeleon


def weighted_training_mse(
    criterion, predictions, targets, raw_weights, row_weights,
    lt_mask=None, gt_mask=None,
):
    """Normalize observed-cell weights; preserve stock MSE reduction."""
    if criterion.__class__.__name__ != "MSE":
        raise ValueError("Updated T121 requires the verified stock MSE criterion.")
    if predictions.shape != targets.shape:
        raise ValueError("Prediction/target shape mismatch.")

    mask = targets.isfinite()
    if not bool(mask.any()):
        raise ValueError("Training batch has no observed targets.")

    raw = torch.as_tensor(
        raw_weights, device=predictions.device, dtype=predictions.dtype
    ).detach()
    if raw.shape != targets.shape:
        raise ValueError("Uncertainty/target shape mismatch.")

    selected = raw[mask]
    if not bool(torch.isfinite(selected).all()) or not bool((selected > 0).all()):
        raise ValueError("Observed uncertainty weights must be finite and positive.")

    denominator = selected.mean()
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError("Invalid uncertainty normalization denominator.")

    normalized = torch.zeros_like(raw)
    normalized[mask] = selected / denominator

    rows = torch.as_tensor(
        row_weights, device=predictions.device, dtype=predictions.dtype
    ).detach()
    if rows.shape != (targets.shape[0], 1):
        raise ValueError("Chemprop row weights must have shape [batch, 1].")
    if not bool(torch.isfinite(rows).all()) or not bool((rows > 0).all()):
        raise ValueError("Invalid Chemprop row weights.")
    if not bool(torch.isfinite(predictions[mask]).all()):
        raise ValueError("Non-finite observed predictions.")

    # Chemprop accepts row weights only. Encode endpoint weights in
    # squared residuals while leaving its denominator/task weights intact.
    root = normalized.sqrt()
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    result = criterion(
        predictions * root,
        safe_targets * root,
        mask,
        rows,
        lt_mask,
        gt_mask,
    )
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Non-finite weighted training loss.")
    return result


class UncertaintyDataset:
    """Wrap an existing Chemprop dataset without changing its molecular records."""

    def __init__(self, molecular_dataset, raw_weights, molecule_ids, task_names):
        self.molecular_dataset = molecular_dataset
        self.raw_weights = np.asarray(raw_weights, dtype=np.float64)
        self.molecule_ids = tuple(str(value) for value in molecule_ids)
        self.task_names = tuple(task_names)
        expected = (len(molecular_dataset), len(self.task_names))
        if self.raw_weights.shape != expected:
            raise ValueError(f"Uncertainty shape must be {expected}.")
        if len(self.molecule_ids) != len(molecular_dataset):
            raise ValueError("Molecule IDs do not align with dataset.")
        if len(set(self.molecule_ids)) != len(self.molecule_ids):
            raise ValueError("Duplicate molecule IDs.")

    def __len__(self):
        return len(self.molecular_dataset)

    def __getitem__(self, index):
        result = (
            self.molecular_dataset[index],
            self.raw_weights[index].copy(),
            self.molecule_ids[index],
        )
        return result


def collate_uncertainty_batch(records):
    from chemprop.data.collate import collate_batch

    molecular_records, weights, ids = zip(*records)
    result = {
        "chemprop_batch": collate_batch(molecular_records),
        "raw_observation_weights": torch.as_tensor(
            np.stack(weights), dtype=torch.float64
        ),
        "molecule_ids": tuple(ids),
    }
    return result


def build_screen_dataset(
    screen_path, identity_path, similarity_path, contract_path,
    descriptor_cache=None,
):
    """Reuse frozen adjudications; retain four native-log2fc enzyme outputs.

    descriptor_cache (D021, opt-in): a loaded descriptor cache. Each datapoint
    then carries the cached descriptor vector for the SMILES it is built from;
    a SMILES missing from the cache raises. None leaves the graph-only
    behavior exactly as before.
    """
    from actfound_model_side import load_adjudicated_screen_assays

    assays = load_adjudicated_screen_assays(
        screen_path, identity_path, similarity_path, contract_path,
        featurize=False,
    )
    task_names = tuple(a.assay_id.removeprefix("screen:") for a in assays)
    if len(task_names) != 4 or len(set(task_names)) != 4:
        raise ValueError("Expected four distinct screening enzymes.")

    ids = tuple(dict.fromkeys(
        name for assay in assays for name in assay.molecule_ids
    ))
    positions = {name: index for index, name in enumerate(ids)}
    labels = np.full((len(ids), 4), np.nan, dtype=np.float32)
    weights = np.zeros((len(ids), 4), dtype=np.float64)
    smiles_by_id = {}

    for column, assay in enumerate(assays):
        for name, smiles, label, weight in zip(
            assay.molecule_ids, assay.smiles, assay.y,
            assay.observation_weights, strict=True,
        ):
            if name in smiles_by_id and smiles_by_id[name] != smiles:
                raise ValueError(f"Inconsistent source SMILES: {name}")
            smiles_by_id[name] = smiles
            row = positions[name]
            labels[row, column] = label
            weights[row, column] = weight

    observed = np.isfinite(labels)
    if not observed.any(axis=1).all():
        raise ValueError("Screen compound has no observed labels.")
    if not np.isfinite(weights[observed]).all() or not (weights[observed] > 0).all():
        raise ValueError("Invalid observed screen uncertainty weights.")

    datapoints = []
    for row, name in enumerate(ids):
        try:
            extra = (
                {} if descriptor_cache is None
                else {"x_d": descriptor_cache.lookup(smiles_by_id[name])}
            )
            point = data.MoleculeDatapoint.from_smi(
                smiles_by_id[name], labels[row].copy(), **extra
            )
            if point.mol is None:
                raise ValueError("No molecular structure returned.")
        except Exception as exc:
            raise ValueError(f"Chemprop preparation failed: {name}") from exc
        datapoints.append(point)

    # Scaling is deliberately not fitted here; the stage builder owns it.
    result = UncertaintyDataset(
        data.MoleculeDataset(datapoints), weights, ids, task_names
    )
    return result


class T121WeightedTrainingMPNN(EncoderRatioOptimizerMixin, CYP002ValidationMPNN):
    """Override training only; retain the existing validation implementation.

    screening_penalty (D024, opt-in): a t121_screening_penalty.ScreeningPenalty
    attached after construction. It is not a module or buffer, so model state
    and checkpoints are unchanged. None leaves the loss exactly as before.
    """

    screening_penalty = None

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.screening_penalty is not None:
            self.screening_penalty.reset_epoch()

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self.screening_penalty is not None:
            self.screening_penalty.flush_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        molecular = batch["chemprop_batch"]
        bmg, V_d, X_d, targets, row_weights, lt_mask, gt_mask = molecular
        ids = batch["molecule_ids"]
        if targets is None or len(ids) != targets.shape[0]:
            raise ValueError("Missing targets or misaligned molecule IDs.")

        raw = torch.as_tensor(
            batch["raw_observation_weights"],
            device=targets.device,
            dtype=targets.dtype,
        )
        if raw.shape != targets.shape:
            raise ValueError("Batch uncertainty/target shape mismatch.")
        invalid = targets.isfinite() & (~torch.isfinite(raw) | (raw <= 0))
        if bool(invalid.any()):
            rows = [
                {"molecule_id": ids[i], "output_index": j}
                for i, j in torch.nonzero(invalid).tolist()
            ]
            raise ValueError(f"Invalid observed uncertainty: {rows}")

        features = self.fingerprint(bmg, V_d, X_d)
        predictions = self.predictor.train_step(features)
        result = weighted_training_mse(
            self.criterion, predictions, targets, raw, row_weights,
            lt_mask, gt_mask,
        )
        # getattr: the screening model borrows this method and never has one.
        penalty = getattr(self, "screening_penalty", None)
        if penalty is not None:
            result = penalty.apply(
                result, predictions, targets, raw, row_weights, ids
            )
        self.log(
            "train_loss", result,
            batch_size=targets.shape[0],
            on_step=True, on_epoch=True, prog_bar=True,
        )
        return result


def build_weighted_pic50_train_loader(prefit_state, training_dataframe):
    """Attach native-unit CI weights to the existing normalized D007 dataset."""
    from torch.utils.data import DataLoader
    from cyp002_config import CYP002_TASK_NAMES

    original_loader = prefit_state["train_loader"]
    if original_loader.num_workers != 0:
        raise ValueError("Expected the verified D007 num_workers=0 loader.")

    dataset = original_loader.dataset
    tasks = tuple(CYP002_TASK_NAMES)
    ids = tuple(training_dataframe["Molecule_Name"].astype(str))
    if len(dataset) != len(ids):
        raise ValueError("D007 training dataset/dataframe row counts differ.")

    # The verified D007 prefit builder constructs this dataset directly from
    # training_dataframe, retaining row order, before fitting target scaling.
    native_targets = training_dataframe.loc[:, list(tasks)].to_numpy(
        dtype=np.float64, copy=True
    )
    low = np.column_stack([
        training_dataframe[f"{task}_conf_low"].to_numpy(
            dtype=np.float64, copy=True
        )
        for task in tasks
    ])
    high = np.column_stack([
        training_dataframe[f"{task}_conf_high"].to_numpy(
            dtype=np.float64, copy=True
        )
        for task in tasks
    ])
    observed = np.isfinite(native_targets)
    with np.errstate(over="ignore", invalid="ignore"):
        widths = high - low

    invalid = observed & (
        ~np.isfinite(low)
        | ~np.isfinite(high)
        | ~np.isfinite(widths)
        | (widths <= 0)
    )
    if np.any(invalid):
        rows = [
            {
                "molecule_id": ids[i],
                "task": tasks[j],
                "conf_low": float(low[i, j]),
                "conf_high": float(high[i, j]),
            }
            for i, j in np.argwhere(invalid)
        ]
        raise ValueError(f"Invalid observed pIC50 CI uncertainty: {rows}")

    raw = np.zeros(native_targets.shape, dtype=np.float64)
    raw[observed] = 1.0 / (widths[observed] + 0.25)
    if (
        not np.isfinite(raw[observed]).all()
        or not (raw[observed] > 0).all()
    ):
        raise ValueError("Invalid raw pIC50 uncertainty weights.")

    # Check normalized dataset masks against the native source labels.
    # Do not refit, undo, or otherwise change the existing target scaler.
    for index, name in enumerate(ids):
        normalized_targets = np.asarray(dataset[index][3])
        if normalized_targets.shape != (len(tasks),):
            raise ValueError(f"Unexpected normalized target shape: {name}")
        if not np.array_equal(
            np.isfinite(normalized_targets), observed[index]
        ):
            raise ValueError(f"Normalized/native target-mask mismatch: {name}")

    wrapped = UncertaintyDataset(dataset, raw, ids, tasks)
    loader = DataLoader(
        wrapped,
        batch_sampler=original_loader.batch_sampler,
        num_workers=0,
        collate_fn=collate_uncertainty_batch,
        pin_memory=original_loader.pin_memory,
    )

    result = {
        "train_loader": loader,
        "training_dataset": wrapped,
        "task_names": tasks,
        "observed_cells": int(observed.sum()),
        "source_units": "native pIC50 CI widths",
        "target_scaler": prefit_state["scaler"],
    }
    return result


def build_updated_t121_pic50_prefit_state(
    *,
    training_dataframe,
    validation_dataframe,
    sampled_config,
    config_name,
    training_seed,
    descriptor_dim=0,
    chemeleon=None,
):
    """Build the opt-in CI-weighted arm through the verified D007 prefit path.

    This constructs an untrained state. Screening checkpoint transfer is a
    separate required transition and is not performed by this function.

    descriptor_dim (D021, opt-in): number of descriptor columns (named with
    CYP002_DESCRIPTOR_COLUMN_PREFIX) carried by both frames. 0 is the
    graph-only arm and must find no descriptor columns.
    """
    from cyp007_cell_fit import build_d007_prefit_state

    if chemeleon is not None and descriptor_dim:
        raise ValueError("D022 CheMeleon arm carries no descriptors.")
    original = build_d007_prefit_state(
        training_dataframe=training_dataframe,
        validation_dataframe=validation_dataframe,
        sampled_config=sampled_config,
        config_name=config_name,
        training_seed=training_seed,
    )
    for partition, loader in (
        ("training", original["train_loader"]),
        ("validation", original["validation_loader"]),
    ):
        actual = int(loader.dataset.d_xd)
        if actual != descriptor_dim:
            raise ValueError(
                f"{partition} descriptor width {actual} != descriptor_dim "
                f"{descriptor_dim}."
            )
    if descriptor_dim > 0:
        from cyp002_model import build_cyp002_model
        from cyp002_train import _configure_determinism

        # The verified builder made a graph-only model. Rebuild it with the
        # widened first FFN layer, re-seeded so initialization is reproducible.
        _configure_determinism(training_seed)
        original["model"] = build_cyp002_model(
            original["config"],
            original["variant"],
            output_scaler=original["scaler"],
            validation_conf_low=original["validation_conf_low"],
            validation_conf_high=original["validation_conf_high"],
            extra_input_dim=descriptor_dim,
        )
    chemeleon_report = None
    if chemeleon is not None:
        # D022: swap in the pretrained encoder, mean aggregation and a head
        # sized to it. Stage transfer overwrites these with the screened
        # weights; the swap makes the architectures match.
        chemeleon_report = apply_chemeleon(
            original["model"], chemeleon, original["config"], training_seed
        )
    weighted_data = build_weighted_pic50_train_loader(
        original, training_dataframe
    )

    config = original["config"]
    baseline_model = original["model"]
    scaler = original["scaler"]

    # Reuse the exact modules initialized by the verified stock builder.
    # Construct the subclass normally; do not mutate __class__ or monkeypatch
    # the historical model's training_step.
    model = T121WeightedTrainingMPNN(
        message_passing=baseline_model.message_passing,
        agg=baseline_model.agg,
        predictor=baseline_model.predictor,
        batch_norm=config.batch_norm,
        warmup_epochs=config.warmup_epochs,
        init_lr=config.initial_lr,
        max_lr=config.max_lr,
        final_lr=config.final_lr,
        validation_conf_low=original["validation_conf_low"],
        validation_conf_high=original["validation_conf_high"],
        target_mean=getattr(scaler, "mean_", None),
        target_scale=getattr(scaler, "scale_", None),
    )
    model.encoder_lr_ratio = getattr(baseline_model, "encoder_lr_ratio", None)

    original_state = baseline_model.state_dict()
    updated_state = model.state_dict()
    if set(original_state) != set(updated_state):
        raise ValueError("Weighted subclass changed model state keys.")
    for name, value in original_state.items():
        if not torch.equal(value, updated_state[name]):
            raise ValueError(
                f"Weighted subclass changed initialized model state: {name}"
            )

    result = dict(original)
    result["model"] = model
    result["train_loader"] = weighted_data["train_loader"]
    result["updated_t121_weighting"] = {
        "measurement": "curve_fit",
        "raw_formula": "1 / (conf_high - conf_low + 0.25)",
        "normalization": "mean one over observed cells in each training batch",
        "observed_training_cells": weighted_data["observed_cells"],
        "evaluation_weighted": False,
    }
    result["descriptor_dim"] = descriptor_dim
    if chemeleon_report is not None:
        result["chemeleon"] = chemeleon_report
    return result


def transfer_screen_initialization(screen_model, pic50_model):
    """Transfer compatible shared state; retain fresh pIC50 head/normalization.

    Both arguments must be constructed model instances. This function does
    not load arbitrary checkpoints, create optimizers, or resume training.
    """
    source = screen_model.state_dict()
    destination = pic50_model.state_dict()

    # Identify the actual final prediction projection from the architecture,
    # rather than assuming an FFN index or hardcoding parameter names.
    def final_projection(model):
        candidates = [
            (name, module)
            for name, module in model.predictor.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        if not candidates:
            raise ValueError("Predictor contains no identifiable Linear projection.")
        name, module = candidates[-1]
        if module.out_features != 4:
            raise ValueError(
                f"Expected four-output final projection; found {module.out_features}."
            )
        result = ("predictor." + name, module)
        return result

    source_head, _ = final_projection(screen_model)
    destination_head, _ = final_projection(pic50_model)
    if source_head != destination_head:
        raise ValueError(
            f"Source/destination projection paths differ: "
            f"{source_head} != {destination_head}"
        )

    # Discover the normalization transform through its actual module path.
    # Its destination state belongs to the pIC50 training-fold scaler.
    normalization_prefixes = [
        "predictor." + name
        for name, _ in pic50_model.predictor.named_modules()
        if name and name.split(".")[-1] == "output_transform"
    ]
    source_normalization_prefixes = [
        "predictor." + name
        for name, _ in screen_model.predictor.named_modules()
        if name and name.split(".")[-1] == "output_transform"
    ]

    def under(key, prefix):
        return key == prefix or key.startswith(prefix + ".")

    def excluded(key):
        return (
            under(key, destination_head)
            or any(
                under(key, prefix)
                for prefix in (
                    normalization_prefixes + source_normalization_prefixes
                )
            )
        )

    source_shared = {key for key in source if not excluded(key)}
    destination_shared = {key for key in destination if not excluded(key)}
    if source_shared != destination_shared:
        raise ValueError(
            "Shared-state keys differ; refusing partial transfer. "
            f"source_only={sorted(source_shared - destination_shared)}, "
            f"destination_only={sorted(destination_shared - source_shared)}"
        )

    mismatches = [
        (key, tuple(source[key].shape), tuple(destination[key].shape))
        for key in sorted(destination_shared)
        if source[key].shape != destination[key].shape
        or source[key].dtype != destination[key].dtype
    ]
    if mismatches:
        raise ValueError(f"Incompatible shared-state shapes/dtypes: {mismatches}")

    retained = {
        key: value.detach().clone()
        for key, value in destination.items()
        if excluded(key)
    }
    if not any(under(key, destination_head) for key in retained):
        raise ValueError("No destination final-projection state identified.")

    merged = {
        key: (
            source[key].detach().clone()
            if key in destination_shared
            else value.detach().clone()
        )
        for key, value in destination.items()
    }
    pic50_model.load_state_dict(merged, strict=True)

    after = pic50_model.state_dict()
    for key in destination_shared:
        if not torch.equal(after[key], source[key]):
            raise ValueError(f"Transferred state verification failed: {key}")
    for key, value in retained.items():
        if not torch.equal(after[key], value):
            raise ValueError(f"Destination head/normalization changed: {key}")

    result = {
        "transferred_keys": sorted(destination_shared),
        "retained_destination_keys": sorted(retained),
        "final_projection": destination_head,
        "normalization_modules": normalization_prefixes,
        "optimizer_state_transferred": False,
        "scheduler_state_transferred": False,
        "target_normalization": "retained from fresh pIC50 training-fold prefit",
    }
    return result


from chemprop.models import MPNN as _ScreenMPNNBase


class T121ScreenTrainingMPNN(EncoderRatioOptimizerMixin, _ScreenMPNNBase):
    """Screening-only model: weighted training without pIC50 validation hooks."""

    training_step = T121WeightedTrainingMPNN.training_step


def build_t121_screen_prefit_state(
    *,
    screen_path,
    identity_path,
    similarity_path,
    contract_path,
    sampled_config,
    config_name,
    training_seed,
    descriptor_cache=None,
    chemeleon=None,
):
    """Construct the screening stage without fitting or selecting a checkpoint.

    descriptor_cache (D021, opt-in): carries cached descriptors on every
    screening datapoint and widens the first FFN layer to match. None keeps
    the graph-only construction exactly as before.
    """
    from torch.utils.data import DataLoader
    if chemeleon is not None and descriptor_cache is not None:
        raise ValueError("D022 CheMeleon arm carries no descriptors.")
    from cyp002_config import CYP002_VARIANT_STOCK
    from cyp002_model import build_cyp002_model
    from cyp002_train import _configure_determinism
    from cyp007_infrastructure import config_from_trial

    config = config_from_trial(
        name=config_name,
        sampled=sampled_config,
        seed=training_seed,
    )
    _configure_determinism(training_seed)

    dataset = build_screen_dataset(
        screen_path,
        identity_path,
        similarity_path,
        contract_path,
        descriptor_cache=descriptor_cache,
    )

    # Reuse Chemprop's existing sampler policy rather than replacing it
    # with a different shuffle/drop-last implementation.
    ordinary_loader = data.build_dataloader(
        dataset.molecular_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        seed=training_seed,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=ordinary_loader.batch_sampler,
        num_workers=0,
        collate_fn=collate_uncertainty_batch,
        pin_memory=ordinary_loader.pin_memory,
    )

    # No pIC50 scaler or validation metadata enters the screening stage.
    stock = build_cyp002_model(
        config,
        CYP002_VARIANT_STOCK,
        output_scaler=None,
        extra_input_dim=0 if descriptor_cache is None else descriptor_cache.dim,
    )
    chemeleon_report = None
    if chemeleon is not None:
        chemeleon_report = apply_chemeleon(stock, chemeleon, config, training_seed)
    model = T121ScreenTrainingMPNN(
        message_passing=stock.message_passing,
        agg=stock.agg,
        predictor=stock.predictor,
        batch_norm=config.batch_norm,
        warmup_epochs=config.warmup_epochs,
        init_lr=config.initial_lr,
        max_lr=config.max_lr,
        final_lr=config.final_lr,
    )
    model.encoder_lr_ratio = getattr(stock, "encoder_lr_ratio", None)

    original_state = stock.state_dict()
    screening_state = model.state_dict()
    if set(original_state) != set(screening_state):
        raise ValueError("Screening subclass changed model state keys.")
    for name, value in original_state.items():
        if not torch.equal(value, screening_state[name]):
            raise ValueError(f"Screening construction changed model state: {name}")

    result = {
        "config": config,
        "model": model,
        "train_loader": loader,
        "training_dataset": dataset,
        "training_seed": training_seed,
        "task_names": dataset.task_names,
        "target_units": "native log2fc",
        "target_scaler": None,
        "validation_loader": None,
        "checkpoint_selection": None,
    }
    if chemeleon_report is not None:
        result["chemeleon"] = chemeleon_report
    return result


def prepare_t121_pic50_transition(
    *,
    screen_model,
    training_dataframe,
    validation_dataframe,
    sampled_config,
    config_name,
    training_seed,
    descriptor_dim=0,
    chemeleon=None,
):
    """Prepare a fresh pIC50 stage and transfer screening shared weights.

    This is a stage transition, not checkpoint resume. No screening
    optimizer, scheduler, epoch counter or target scaler is accepted.
    """
    state = build_updated_t121_pic50_prefit_state(
        training_dataframe=training_dataframe,
        validation_dataframe=validation_dataframe,
        sampled_config=sampled_config,
        config_name=config_name,
        training_seed=training_seed,
        descriptor_dim=descriptor_dim,
        chemeleon=chemeleon,
    )
    destination = state["model"]
    if getattr(destination, "_trainer", None) is not None:
        raise ValueError("Expected a fresh, unattached pIC50 model.")

    report = transfer_screen_initialization(
        screen_model,
        destination,
    )
    state["screen_to_pic50_transition"] = report
    state["stage_initialization"] = "screen_shared_weights_fresh_pic50_state"

    result = state
    return result


def fit_prepared_t121_pic50_stage(
    *,
    prefit_state,
    checkpoint_dir,
    authorize_training=False,
):
    """Execute the existing D007 fit path only after explicit authorization."""
    if authorize_training is not True:
        raise RuntimeError(
            "Training is not authorized. Prepare and qualify the stage first."
        )
    if (
        prefit_state.get("stage_initialization")
        != "screen_shared_weights_fresh_pic50_state"
    ):
        raise ValueError("Missing prepared screening-to-pIC50 transition.")

    model = prefit_state["model"]
    if getattr(model, "_trainer", None) is not None:
        raise ValueError(
            "Prepared model already has a Trainer; refusing implicit resume."
        )

    from cyp007_cell_fit import fit_d007_cell

    # The verified executor constructs a new Trainer and calls fit without
    # ckpt_path. Its existing horizon, callbacks and validation stay intact.
    result = fit_d007_cell(
        prefit_state=prefit_state,
        checkpoint_dir=checkpoint_dir,
    )
    return result
