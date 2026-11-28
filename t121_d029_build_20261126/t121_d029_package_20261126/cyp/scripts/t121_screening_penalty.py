"""D024 screening-informed over-prediction penalty (opt-in penalty arms).

    total = base loss + lambda * P
    P = sum over observed training cells of
            w * relu(standardized prediction - standardized upper bound)^2
        divided by the same observed-cell count the base loss uses.

Weights w are computed per training fold and enzyme from labelled training-fold
molecules with verified screening values only (rules in the frozen arm config).
The CI-width weights of the base loss are not applied to P.

A config may name one of two further forms with the key `form` (a config without
the key is the D024 form above); weights and denominator are the same:
    two_sided_squared  adds w * relu(standardized lower bound - prediction)^2
    upper_huber        replaces relu(.)^2 by H(e): e^2 for e <= delta, else
                       2*delta*e - delta^2, with delta per enzyme and training
                       fold the median standardized confidence half-width over
                       observed training cells
A missing or incompatible bound (two-sided), or a missing, non-finite or zero
delta (Huber), stops the arm after its records are written.

The model holds one table (weight, standardized upper bound) keyed by molecule
ID; dataset, collate function and model parameters are unchanged. Every rule
constant comes from the frozen config (cyp/artifacts/d024_penalty/*.json).

`preview` builds the weights of every fold on CPU, with no model, so coverage
and identity checks happen before any training.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import math
import sys
import tarfile

import numpy as np

RULES = ("q3_cutoff", "response_rank")
UPPER_SQUARED, TWO_SIDED, UPPER_HUBER = "upper_squared", "two_sided_squared", "upper_huber"
FORMS = (UPPER_SQUARED, TWO_SIDED, UPPER_HUBER)
MISSING_TEXT = ("", "nan", "na", "null")
STATUSES = ("ok", "no_screening_row", "missing_value", "unverified_identity", "conflicting_records")
WEIGHT_COLUMNS = ("Molecule_Name", "enzyme", "status", "value", "weight")
POPULATION_FILE = "penalty_population.json"
WEIGHTS_FILE = "penalty_weights.csv"
HISTORY_FILE = "penalty_epochs.jsonl"
# Members of the screening transfer tarball (preview on the Mac).
SCREEN_MEMBER = "inputs/screen_full.csv"
IDENTITY_MEMBER = "inputs/adjudication.tsv"
CONTRACT_MEMBER = "qualification/actfound/screen_loader_20261117/contract.json"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


# --------------------------------------------------------------------------- config
class PenaltyArm:
    """One frozen penalty-arm config. Nothing has a default, except that a
    config without `form` is the D024 form (the D024 configs are frozen)."""

    def __init__(self, config, config_sha256):
        required = {
            "decision", "arm", "arm_label", "rule", "lambda", "minimum_population",
            "tasks", "screen", "identity",
        }
        missing = required - set(config)
        if missing:
            raise ValueError(f"Penalty config lacks: {sorted(missing)}")
        rule = config["rule"]
        if rule not in RULES:
            raise ValueError(f"Unknown penalty rule: {rule!r}")
        form = config.get("form", UPPER_SQUARED)
        if form not in FORMS:
            raise ValueError(f"Unknown penalty form: {form!r}")
        strength = config["lambda"]
        if (
            isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or strength < 0
        ):
            raise ValueError("Penalty lambda must be a finite non-negative number.")
        if rule == "q3_cutoff":
            absent = {"quantile", "quantile_method", "flag"} - set(config)
            if absent:
                raise ValueError(f"Cutoff config lacks: {sorted(absent)}")
            if not 0 < config["quantile"] < 1:
                raise ValueError("Cutoff quantile must lie between 0 and 1.")
            if config["quantile_method"] != "linear" or config["flag"] != "strictly_above":
                raise ValueError("Only linear quantiles with a strictly-above flag are defined.")
        if not isinstance(config["minimum_population"], int) or config["minimum_population"] < 2:
            raise ValueError("minimum_population must be an integer of at least 2.")
        self.config = config
        self.config_sha256 = config_sha256
        self.arm = config["arm"]
        self.arm_label = config["arm_label"]
        self.rule = rule
        self.form = form
        self.strength = float(strength)

    @classmethod
    def load(cls, path, *, expected_sha256):
        data = Path(path).read_bytes()
        actual = sha256_bytes(data)
        if actual != expected_sha256:
            raise ValueError(f"Penalty config hash mismatch: {path} {actual}")
        return cls(json.loads(data), actual)

    def provenance(self):
        return {"config_sha256": self.config_sha256, "config": self.config}


# --------------------------------------------------------------------------- inputs
def _rows(data, delimiter):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""), delimiter=delimiter))


def parse_screen(screen_bytes, identity_bytes, contract_bytes, arm):
    """Verify both files against the contract's hashes and index them."""
    contract = json.loads(contract_bytes)
    hashes = {"screen": sha256_bytes(screen_bytes), "identity": sha256_bytes(identity_bytes)}
    for kind, actual in hashes.items():
        if actual != contract["sha256"][kind]:
            raise ValueError(f"{kind} input does not match the contract's SHA-256: {actual}")
    columns = arm.config["screen"]
    identity_columns = arm.config["identity"]
    screen_rows = _rows(screen_bytes, ",")
    if not screen_rows:
        raise ValueError("Screening file has no rows.")
    needed = [columns[k] for k in ("molecule_column", "smiles_column", "enzyme_column", "value_column")]
    needed += list(columns["condition_guard_columns"])
    absent = [c for c in needed if c not in screen_rows[0]]
    if absent:
        raise ValueError(f"Screening file lacks columns: {absent}")
    # The condition key is (molecule, enzyme). It holds only while the file has
    # one assay condition; otherwise stop rather than average records.
    for column in columns["condition_guard_columns"]:
        distinct = {row[column].strip() for row in screen_rows}
        if len(distinct) != 1:
            raise ValueError(
                f"Screening column {column} has {len(distinct)} distinct values; "
                "the condition key must be updated before any aggregation."
            )
    records = {}
    for row in screen_rows:
        key = (row[columns["molecule_column"]], row[columns["enzyme_column"]])
        records.setdefault(key, []).append(
            (row[columns["smiles_column"]], row[columns["value_column"]].strip())
        )
    identity = {}
    for row in _rows(identity_bytes, "\t"):
        name = row[identity_columns["molecule_column"]]
        if name in identity:
            raise ValueError(f"Duplicate adjudication identity: {name}")
        identity[name] = (
            row[identity_columns["smiles_column"]],
            row[identity_columns["class_column"]],
            (row[identity_columns["tier_column"]] or "").strip(),
        )
    hashes["contract"] = sha256_bytes(contract_bytes)
    return {"records": records, "identity": identity, "sha256": hashes}


def load_screen_files(screen_path, identity_path, contract_path, arm):
    return parse_screen(
        Path(screen_path).read_bytes(), Path(identity_path).read_bytes(),
        Path(contract_path).read_bytes(), arm,
    )


def load_screen_tar(tar_path, arm):
    """Read the three members straight from the transfer tarball (nothing is extracted)."""
    data = {}
    with tarfile.open(tar_path, "r:*") as archive:
        for member in (SCREEN_MEMBER, IDENTITY_MEMBER, CONTRACT_MEMBER):
            handle = archive.extractfile(member)
            if handle is None:
                raise FileNotFoundError(f"{member} is not a file in {tar_path}")
            data[member] = handle.read()
    return parse_screen(data[SCREEN_MEMBER], data[IDENTITY_MEMBER], data[CONTRACT_MEMBER], arm)


def _value(text, label):
    if text.lower() in MISSING_TEXT:
        return None
    result = float(text)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite screening value: {label}")
    return result


def lookup(screen, arm, name, enzyme):
    """Screening status of one (molecule, enzyme).

    Returns (status, value or None, screening SMILES or None, collapsed count).
    A molecule adjudicated into a protected tier raises.
    """
    rules = arm.config["identity"]
    adjudication = screen["identity"].get(name)
    if adjudication is not None:
        tiers = {part.strip() for part in adjudication[2].replace(";", ",").split(",")}
        if tiers & set(rules["protected_tiers"]):
            raise ValueError(f"Protected-tier molecule reached the training population: {name}")
    rows = screen["records"].get((name, enzyme))
    if not rows:
        return "no_screening_row", None, None, 0
    distinct = {(smiles, _value(text, f"{name}:{enzyme}")) for smiles, text in rows}
    if len(distinct) > 1:
        return "conflicting_records", None, None, 0
    (smiles, value), = distinct
    collapsed = len(rows) - 1
    if (
        adjudication is None
        or adjudication[1] not in rules["accepted_classes"]
        or smiles != adjudication[0]
    ):
        return "unverified_identity", None, smiles, collapsed
    if value is None:
        return "missing_value", None, smiles, collapsed
    return "ok", value, smiles, collapsed


# --------------------------------------------------------------------------- weights
def linear_quantile(values, q):
    """Linear interpolation between neighbouring order statistics."""
    ordered = sorted(float(v) for v in values)
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    if lower + 1 >= len(ordered):
        return ordered[-1]
    return ordered[lower] + (position - lower) * (ordered[lower + 1] - ordered[lower])


def average_ranks(values):
    """Ascending 1-based ranks; tied values take their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        for position in range(start, stop + 1):
            ranks[order[position]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def population_weights(values, arm):
    """Mean-one weights over one enzyme's population (float64), plus a summary."""
    values = [float(v) for v in values]
    n = len(values)
    weights = [0.0] * n
    summary = {
        "population": n, "active": False, "reason": None, "q3": None,
        "flagged": 0, "flagged_fraction": None, "mean_weight": None,
    }
    if n < arm.config["minimum_population"]:
        summary["reason"] = "fewer_than_two_eligible"
        return weights, summary
    if min(values) == max(values):
        summary["reason"] = "all_responses_equal"
        return weights, summary
    if arm.rule == "q3_cutoff":
        cutoff = linear_quantile(values, arm.config["quantile"])
        summary["q3"] = cutoff
        flagged = [v > cutoff for v in values]
        count = sum(flagged)
        if count == 0:
            summary.update(reason="nothing_strictly_above_q3", flagged_fraction=0.0)
            return weights, summary
        weights = [n / count if flag else 0.0 for flag in flagged]
    else:
        raw = [(rank - 1.0) / (n - 1.0) for rank in average_ranks(values)]
        mean_raw = math.fsum(raw) / n
        weights = [value / mean_raw for value in raw]
    flagged_count = sum(1 for w in weights if w > 0)
    summary.update(
        active=True, flagged=flagged_count, flagged_fraction=flagged_count / n,
        mean_weight=math.fsum(weights) / n,
    )
    return weights, summary


def build_population(training_dataframe, screen, arm):
    """Weights for every labelled (training molecule, enzyme) cell of one fold.

    Only the training dataframe is consulted: a molecule outside it cannot
    change a cutoff, a rank or a normalization.
    """
    names = [str(v) for v in training_dataframe["Molecule_Name"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate training molecule names.")
    smiles = dict(zip(names, (str(v) for v in training_dataframe["SMILES"])))
    rows, enzymes, differences = [], {}, []
    for task, enzyme in arm.config["tasks"].items():
        labels = np.asarray(training_dataframe[task], dtype=np.float64)
        labelled = sorted(name for name, label in zip(names, labels) if np.isfinite(label))
        counts = {status: 0 for status in STATUSES}
        collapsed = 0
        looked_up = []
        for name in labelled:
            status, value, screen_smiles, extra = lookup(screen, arm, name, enzyme)
            counts[status] += 1
            collapsed += extra
            if status == "ok" and screen_smiles != smiles[name]:
                differences.append({
                    "Molecule_Name": name, "enzyme": enzyme,
                    "screen_smiles": screen_smiles, "training_smiles": smiles[name],
                })
            looked_up.append((name, status, value))
        eligible = [value for _, status, value in looked_up if status == "ok"]
        weights, summary = population_weights(eligible, arm)
        if not labelled:
            summary["reason"] = "no_labelled_training_molecules"
        by_position = iter(weights)
        for name, status, value in looked_up:
            rows.append({
                "Molecule_Name": name, "enzyme": enzyme, "status": status,
                "value": value, "weight": next(by_position) if status == "ok" else 0.0,
            })
        enzymes[enzyme] = {
            **summary, "labelled_training_molecules": len(labelled),
            "status_counts": counts, "collapsed_duplicate_records": collapsed,
        }
    return rows, {"enzymes": enzymes, "training_smiles_differences": differences}


def weights_text(rows):
    """Canonical CSV text (sorted, repr floats): equal inputs give equal bytes."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(WEIGHT_COLUMNS)
    for row in sorted(rows, key=lambda r: (r["enzyme"], r["Molecule_Name"])):
        writer.writerow([
            row["Molecule_Name"], row["enzyme"], row["status"],
            "" if row["value"] is None else repr(row["value"]), repr(row["weight"]),
        ])
    return buffer.getvalue()


def require_no_smiles_differences(summary):
    differences = summary["training_smiles_differences"]
    if differences:
        raise ValueError(
            f"{len(differences)} verified screening records differ from the training SMILES; "
            f"review before any training. First: {differences[:3]}"
        )


# --------------------------------------------------------------------------- loss
def penalty_cells(predictions, targets, weights, upper, lower=None, delta=None):
    """Weighted penalty per cell; zero where the target is unobserved.

    Neither `lower` nor `delta`: w * relu(prediction - upper)^2.
    `lower` (same shape): adds w * relu(lower - prediction)^2.
    `delta` (one value per output): the upper excess e is penalized as e^2 up
    to delta and as 2*delta*e - delta^2 above it.
    """
    import torch

    if lower is not None and delta is not None:
        raise ValueError("A penalty takes a lower bound or a Huber delta, not both.")
    if not (predictions.shape == targets.shape == weights.shape == upper.shape) or (
        lower is not None and lower.shape != predictions.shape
    ):
        raise ValueError("Penalty prediction/target/weight/bound shape mismatch.")
    mask = targets.isfinite()
    zero = torch.zeros_like(predictions)
    safe_upper = torch.where(mask, upper, zero)
    safe_weights = torch.where(mask, weights, zero)
    if not bool(torch.isfinite(safe_upper).all()) or not bool(torch.isfinite(safe_weights).all()):
        raise ValueError("Non-finite penalty weight or upper bound on an observed cell.")
    excess = torch.relu(predictions - safe_upper)
    size = excess.square()
    if lower is not None:
        safe_lower = torch.where(mask, lower, zero)
        if not bool(torch.isfinite(safe_lower).all()):
            raise ValueError("Non-finite lower bound on an observed cell.")
        size = size + torch.relu(safe_lower - predictions).square()
    if delta is not None:
        delta = delta.to(device=predictions.device, dtype=predictions.dtype)
        if delta.shape != (predictions.shape[1],) or not bool(
            (torch.isfinite(delta) & (delta > 0)).all()
        ):
            raise ValueError("Penalty delta must be one finite positive value per output.")
        size = torch.where(excess <= delta, size, 2.0 * delta * excess - delta.square())
    return safe_weights * size


class ScreeningPenalty:
    """Per-molecule penalty table plus the per-epoch loss-component record."""

    def __init__(self, *, strength, molecule_ids, weights, upper, enzymes, history_path=None,
                 lower=None, delta=None):
        import torch

        if lower is not None and delta is not None:
            raise ValueError("A penalty table takes lower or delta, not both.")

        self.strength = float(strength)
        self.enzymes = tuple(enzymes)
        ids = [str(v) for v in molecule_ids]
        self.index = {name: i for i, name in enumerate(ids)}
        if len(self.index) != len(ids):
            raise ValueError("Duplicate molecule IDs in the penalty table.")
        self.weights = torch.as_tensor(np.asarray(weights, dtype=np.float64))
        self.upper = torch.as_tensor(np.asarray(upper, dtype=np.float64))
        expected = (len(ids), len(self.enzymes))
        if tuple(self.weights.shape) != expected or tuple(self.upper.shape) != expected:
            raise ValueError(f"Penalty table shape must be {expected}.")
        self.lower = self.delta = None
        if lower is not None:
            self.lower = torch.as_tensor(np.asarray(lower, dtype=np.float64))
            if tuple(self.lower.shape) != expected:
                raise ValueError(f"Penalty table shape must be {expected}.")
        if delta is not None:
            values = np.asarray(delta, dtype=np.float64)
            if values.shape != (len(self.enzymes),) or not (
                np.isfinite(values).all() and (values > 0).all()
            ):
                raise ValueError(
                    f"Penalty delta must be {len(self.enzymes)} finite positive values."
                )
            self.delta = torch.as_tensor(values)
        self.history_path = None if history_path is None else Path(history_path)
        self.reset_epoch()

    def apply(self, base, predictions, targets, raw_weights, row_weights, molecule_ids):
        """base + strength * P for one batch; records the components."""
        import torch

        try:
            rows = [self.index[str(name)] for name in molecule_ids]
        except KeyError as exc:
            raise ValueError(f"Molecule absent from the penalty table: {exc}") from exc
        weights = self.weights[rows].to(device=predictions.device, dtype=predictions.dtype)
        upper = self.upper[rows].to(device=predictions.device, dtype=predictions.dtype)
        lower = None
        if self.lower is not None:
            lower = self.lower[rows].to(device=predictions.device, dtype=predictions.dtype)
        cells = penalty_cells(
            predictions, targets, weights, upper, lower=lower, delta=self.delta
        )
        count = targets.isfinite().sum()
        penalty = cells.sum() / count
        result = base + self.strength * penalty
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Non-finite penalized training loss.")
        self._accumulate(base, cells, predictions, targets, raw_weights, row_weights, weights)
        return result

    def reset_epoch(self):
        self._sums = None
        self._batches = 0

    def _accumulate(self, base, cells, predictions, targets, raw_weights, row_weights, weights):
        import torch

        with torch.no_grad():
            mask = targets.isfinite()
            raw = raw_weights.detach().double()
            normalized = torch.where(mask, raw / raw[mask].mean(), torch.zeros_like(raw))
            safe_targets = torch.where(mask, targets, torch.zeros_like(targets)).double()
            residual = predictions.detach().double() - safe_targets
            base_cells = torch.where(
                mask, row_weights.detach().double() * normalized * residual.square(),
                torch.zeros_like(raw),
            )
            batch = {
                "observed_cells": mask.sum(0).double(),
                "flagged_cells": (mask & (weights > 0)).sum(0).double(),
                "active_cells": (cells.detach() > 0).sum(0).double(),
                "base_sum": base_cells.sum(0),
                "penalty_sum": cells.detach().double().sum(0),
                "base_loss_times_cells": base.detach().double() * mask.sum(),
            }
            if self._sums is None:
                self._sums = batch
            else:
                self._sums = {key: self._sums[key] + batch[key] for key in batch}
            self._batches += 1

    def epoch_record(self, epoch):
        if self._sums is None:
            raise RuntimeError("No training batches were recorded for this epoch.")
        sums = {key: value.cpu().tolist() for key, value in self._sums.items()}
        total = math.fsum(sums["observed_cells"])
        unscaled = math.fsum(sums["penalty_sum"]) / total
        return {
            "epoch": int(epoch),
            "batches": self._batches,
            "lambda": self.strength,
            "observed_cells": int(total),
            "base_loss": sums["base_loss_times_cells"] / total,
            "penalty_unscaled": unscaled,
            "penalty_scaled": self.strength * unscaled,
            "per_enzyme": {
                enzyme: {
                    "observed_cells": int(sums["observed_cells"][i]),
                    "flagged_cells": int(sums["flagged_cells"][i]),
                    "active_cells": int(sums["active_cells"][i]),
                    "base_sum": sums["base_sum"][i],
                    "penalty_sum": sums["penalty_sum"][i],
                    "penalty_scaled_sum": self.strength * sums["penalty_sum"][i],
                }
                for i, enzyme in enumerate(self.enzymes)
            },
        }

    def flush_epoch(self, epoch):
        """Append this epoch's line immediately, so a failed run keeps its history."""
        record = self.epoch_record(epoch)
        if self.history_path is not None:
            with self.history_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.reset_epoch()
        return record


def standardized_bounds(bounds, mean, scale):
    """Native confidence bounds in the training fold's target-scaler units."""
    return (np.asarray(bounds, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / np.asarray(
        scale, dtype=np.float64
    )


standardized_upper_bounds = standardized_bounds


def bound_problems(low, high, observed, enzymes, molecule_ids):
    """Two-sided form: observed cells need finite bounds with lower < upper."""
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    bad = observed & ~(np.isfinite(low) & np.isfinite(high) & (low < high))
    reasons = []
    for j, enzyme in enumerate(enzymes):
        rows = np.flatnonzero(bad[:, j])
        if rows.size:
            reasons.append(
                f"{enzyme}: {rows.size} observed training cells with a missing, non-finite "
                f"or lower >= upper bound (first: {[molecule_ids[i] for i in rows[:3]]})"
            )
    return reasons


def delta_summary(low, high, observed, scale, enzymes):
    """Per-enzyme median half-width over observed cells, in scaler units.

    Returns (entries, reasons). `scale` None leaves the values in native units.
    A non-finite half-width on any observed cell is not skipped over. Invalid
    values are recorded as None.
    """
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    scales = None if scale is None else np.asarray(scale, dtype=np.float64)
    entries, reasons = {}, []
    for j, enzyme in enumerate(enzymes):
        cells = observed[:, j]
        target_scale = 1.0 if scales is None else float(scales[j])
        native = standardized = problem = None
        if not cells.any():
            problem = "missing (no observed training cells)"
        else:
            with np.errstate(invalid="ignore"):
                half = (high[cells, j] - low[cells, j]) / 2.0
            if not np.isfinite(half).all():
                problem = "non-finite (a bound is missing or non-finite on an observed cell)"
            else:
                native = float(np.median(half))
                if not (math.isfinite(target_scale) and target_scale > 0):
                    problem = f"non-finite or invalid target scale {target_scale}"
                elif native == 0:
                    problem = "zero"
                elif native < 0:
                    problem = "negative"
                else:
                    standardized = native / target_scale
                    if not (math.isfinite(standardized) and standardized > 0):
                        problem, standardized = "non-finite or zero after scaling", None
        if problem is not None:
            reasons.append(f"{enzyme}: delta {problem}")
        entries[enzyme] = {
            "observed_cells": int(cells.sum()),
            "median_native_half_width": native,
            "target_scale": (
                target_scale if scales is not None and math.isfinite(target_scale) else None
            ),
            "standardized_delta": standardized if scales is not None else None,
        }
    return entries, reasons


def standardized_deltas(low, high, observed, scale, enzymes):
    """One delta per enzyme in scaler units; raises if any is unusable."""
    entries, reasons = delta_summary(low, high, observed, scale, enzymes)
    if reasons:
        raise ValueError("; ".join(reasons))
    return np.array([entries[e]["standardized_delta"] for e in enzymes], dtype=np.float64)


def check_bounds(form, training_dataframe, tasks, enzymes, scale=None):
    """(stop reasons, delta entries or None) for one training fold. The D024
    form has no check here: it keeps its own upper-bound check."""
    if form == UPPER_SQUARED:
        return [], None
    observed = np.isfinite(_columns(training_dataframe, tasks, ""))
    low = _columns(training_dataframe, tasks, "_conf_low")
    high = _columns(training_dataframe, tasks, "_conf_high")
    if form == TWO_SIDED:
        ids = [str(v) for v in training_dataframe["Molecule_Name"]]
        return bound_problems(low, high, observed, enzymes, ids), None
    entries, reasons = delta_summary(low, high, observed, scale, enzymes)
    return reasons, entries


def _columns(dataframe, tasks, suffix):
    return np.column_stack([
        np.asarray(dataframe[f"{task}{suffix}"], dtype=np.float64) for task in tasks
    ])


def attach_penalty(*, prefit, training_dataframe, screen_inputs, arm, cell_dir):
    """Build one fold's penalty table, write its records and hang it on the model."""
    from cyp002_config import CYP002_TASK_NAMES

    tasks = tuple(CYP002_TASK_NAMES)
    if tuple(arm.config["tasks"]) != tasks:
        raise ValueError("Penalty config tasks differ from the model's task order.")
    model = prefit["model"]
    if getattr(model, "_trainer", None) is not None:
        raise ValueError("The penalty must be attached before the model has a Trainer.")
    cell_dir = Path(cell_dir)
    history_path = cell_dir / HISTORY_FILE
    if history_path.exists():
        raise FileExistsError(f"Refusing to append to an existing epoch log: {history_path}")
    screen = load_screen_files(
        screen_inputs["screen_path"], screen_inputs["identity_path"],
        screen_inputs["contract_path"], arm,
    )
    rows, summary = build_population(training_dataframe, screen, arm)
    text = weights_text(rows)
    report = {
        **arm.provenance(),
        "screen_input_sha256": screen["sha256"],
        "weights_sha256": sha256_bytes(text.encode()),
        **summary,
    }
    enzymes = [arm.config["tasks"][task] for task in tasks]
    scaler = prefit["scaler"]
    reasons, deltas = check_bounds(
        arm.form, training_dataframe, tasks, enzymes, scale=scaler.scale_
    )
    if arm.form != UPPER_SQUARED:
        report["bound_stop_reasons"] = reasons
    if deltas is not None:
        report["delta"] = deltas
    # Records are written before any check that can stop the cell.
    with (cell_dir / WEIGHTS_FILE).open("x", newline="") as handle:
        handle.write(text)
    with (cell_dir / POPULATION_FILE).open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    require_no_smiles_differences(summary)
    if reasons:
        raise ValueError(f"Penalty bounds stop arm {arm.arm}: " + "; ".join(reasons))

    ids = [str(v) for v in training_dataframe["Molecule_Name"]]
    position = {name: i for i, name in enumerate(ids)}
    weights = np.zeros((len(ids), len(tasks)), dtype=np.float64)
    for row in rows:
        weights[position[row["Molecule_Name"]], enzymes.index(row["enzyme"])] = row["weight"]
    high = _columns(training_dataframe, tasks, "_conf_high")
    upper = standardized_bounds(high, scaler.mean_, scaler.scale_)
    if not np.isfinite(upper[weights > 0]).all():
        raise ValueError("A penalized cell has no finite upper confidence bound.")
    lower = delta = None
    if arm.form == TWO_SIDED:
        low = _columns(training_dataframe, tasks, "_conf_low")
        lower = standardized_bounds(low, scaler.mean_, scaler.scale_)
    elif arm.form == UPPER_HUBER:
        delta = [deltas[enzyme]["standardized_delta"] for enzyme in enzymes]
    model.screening_penalty = ScreeningPenalty(
        strength=arm.strength, molecule_ids=ids, weights=weights, upper=upper,
        enzymes=enzymes, history_path=history_path, lower=lower, delta=delta,
    )
    return report


# --------------------------------------------------------------------------- preview
def preview(*, arms, screens, dataframe, split_dataframe, manifests, route_fold, output_dir):
    """Weights of every fold for every arm. Returns the report (with stop reasons)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    folds, disabled, differences, bound_stops = [], [], {}, []
    for manifest_path in manifests:
        manifest = json.loads(Path(manifest_path).read_text())
        routed = route_fold(
            dataframe=dataframe, original_split_dataframe=split_dataframe,
            split_manifest=manifest, training_seed=manifest["training_seeds"][0],
        )
        name = f"rep_{manifest['repetition']:02d}_fold_{manifest['fold']:02d}"
        for arm in arms:
            rows, summary = build_population(routed["training_dataframe"], screens[arm.arm], arm)
            text = weights_text(rows)
            (output_dir / arm.arm).mkdir(exist_ok=True)
            with (output_dir / arm.arm / f"{name}_{WEIGHTS_FILE}").open("x", newline="") as handle:
                handle.write(text)
            for item in summary["training_smiles_differences"]:
                differences[(item["Molecule_Name"], item["enzyme"])] = item
            for enzyme, entry in summary["enzymes"].items():
                if not entry["active"]:
                    disabled.append({"arm": arm.arm, "fold": name, "enzyme": enzyme,
                                     "reason": entry["reason"]})
            entry = {
                "arm": arm.arm, "repetition": manifest["repetition"], "fold": manifest["fold"],
                "weights_sha256": sha256_bytes(text.encode()), "enzymes": summary["enzymes"],
            }
            # Native units: the fold's target scaler is fitted only in the cell,
            # which repeats this check in scaler units before training.
            tasks = tuple(arm.config["tasks"])
            reasons, deltas = check_bounds(
                arm.form, routed["training_dataframe"], tasks,
                [arm.config["tasks"][task] for task in tasks],
            )
            if deltas is not None:
                entry["delta"] = deltas
            bound_stops.extend(
                {"arm": arm.arm, "fold": name, "reason": reason} for reason in reasons
            )
            folds.append(entry)
    report = {
        "arms": {arm.arm: arm.config_sha256 for arm in arms},
        "screen_input_sha256": screens[arms[0].arm]["sha256"],
        "folds": folds,
        "disabled": disabled,
        "training_smiles_differences": [differences[k] for k in sorted(differences)],
        "bound_stops": bound_stops,
        "stopped_arms": sorted({item["arm"] for item in bound_stops}),
        "stop": bool(disabled or differences or bound_stops),
    }
    with (output_dir / "preview.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def weight_mismatches(report, reference, reference_arm):
    """Folds whose weights differ from `reference_arm` of an earlier preview."""
    expected = {
        (f["repetition"], f["fold"]): f["weights_sha256"]
        for f in reference["folds"] if f["arm"] == reference_arm
    }
    if not expected:
        raise ValueError(f"Reference preview has no folds for arm {reference_arm!r}.")
    mismatches = []
    for arm in sorted({f["arm"] for f in report["folds"]}):
        mine = {
            (f["repetition"], f["fold"]): f["weights_sha256"]
            for f in report["folds"] if f["arm"] == arm
        }
        for key in sorted(set(mine) | set(expected)):
            if mine.get(key) != expected.get(key):
                mismatches.append({
                    "arm": arm, "repetition": key[0], "fold": key[1],
                    "weights_sha256": mine.get(key), "reference_sha256": expected.get(key),
                })
    return mismatches


def preview_table(report):
    """One line per (arm, enzyme): ranges over folds of the counts to review."""
    lines = ["arm       enzyme  folds  labelled     population   flagged_fraction   "
             "no_row  missing  unverified  conflicting"]
    keys = sorted({(f["arm"], e) for f in report["folds"] for e in f["enzymes"]})
    for arm, enzyme in keys:
        entries = [f["enzymes"][enzyme] for f in report["folds"] if f["arm"] == arm]

        def span(values):
            return f"{min(values)}-{max(values)}"

        fractions = [e["flagged_fraction"] for e in entries if e["flagged_fraction"] is not None]
        fraction = f"{min(fractions):.3f}-{max(fractions):.3f}" if fractions else "none"
        status = {s: span([e["status_counts"][s] for e in entries]) for s in STATUSES}
        lines.append(
            f"{arm:<9} {enzyme:<7} {len(entries):>5}  "
            f"{span([e['labelled_training_molecules'] for e in entries]):<12} "
            f"{span([e['population'] for e in entries]):<12} {fraction:<18} "
            f"{status['no_screening_row']:<7} {status['missing_value']:<8} "
            f"{status['unverified_identity']:<11} {status['conflicting_records']}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preview")
    pre.add_argument("--penalty-config", action="append", required=True)
    for name in ("screen-tar", "dataset", "original-split", "split-manifest-root",
                 "executor-source-dir", "executor-source-sha256", "output-dir"):
        pre.add_argument(f"--{name}", required=True)
    # Optional, all three or none: require the same weights as an earlier preview.
    for name in ("reference-preview", "reference-preview-sha256", "reference-arm"):
        pre.add_argument(f"--{name}")
    args = parser.parse_args(argv)
    reference_args = (args.reference_preview, args.reference_preview_sha256, args.reference_arm)
    if any(reference_args) and not all(reference_args):
        raise SystemExit("Give all of --reference-preview, --reference-preview-sha256 "
                         "and --reference-arm, or none.")
    reference = None
    if args.reference_preview:
        data = Path(args.reference_preview).read_bytes()
        if sha256_bytes(data) != args.reference_preview_sha256:
            raise SystemExit(f"Reference preview hash mismatch: {sha256_bytes(data)}")
        reference = json.loads(data)

    source = Path(args.executor_source_dir) / "cyp007_cell_fit.py"
    digest = sha256_file(source)
    if digest != args.executor_source_sha256:
        raise SystemExit(f"Executor source hash mismatch: {source} {digest}")
    for directory in (args.executor_source_dir, str(Path(__file__).resolve().parent)):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    import pandas as pd
    import cyp007_cell_fit as fit

    arms = [PenaltyArm(json.loads(Path(p).read_bytes()), sha256_file(p)) for p in args.penalty_config]
    if len({arm.arm for arm in arms}) != len(arms):
        raise SystemExit("Each --penalty-config must describe a different arm.")
    manifests = sorted(Path(args.split_manifest_root).glob("rep_*_fold_*.json"))
    if not manifests:
        raise SystemExit("No split manifests found.")
    report = preview(
        arms=arms,
        screens={arm.arm: load_screen_tar(args.screen_tar, arm) for arm in arms},
        dataframe=pd.read_csv(args.dataset),
        split_dataframe=pd.read_csv(args.original_split),
        manifests=manifests,
        route_fold=fit.route_d007_fold,
        output_dir=args.output_dir,
    )
    mismatches = []
    if reference is not None:
        mismatches = weight_mismatches(report, reference, args.reference_arm)
    print(preview_table(report))
    for arm, digest in report["arms"].items():
        print(f"config {arm}: {digest}")
    print(f"folds: {len(manifests)}  disabled (arm, fold, enzyme): {len(report['disabled'])}  "
          f"training-SMILES differences: {len(report['training_smiles_differences'])}")
    for item in report["disabled"][:20]:
        print("DISABLED", item)
    for item in report["training_smiles_differences"][:20]:
        print("SMILES-DIFFERENCE", item)
    widths = {}
    for entry in report["folds"]:
        for enzyme, values in entry.get("delta", {}).items():
            if values["median_native_half_width"] is not None:
                widths.setdefault((entry["arm"], enzyme), []).append(
                    values["median_native_half_width"])
    for (arm, enzyme), values in sorted(widths.items()):
        print(f"{arm} {enzyme} median native half-width over folds: "
              f"{min(values):.6f} to {max(values):.6f}")
    print(f"bound stops: {len(report['bound_stops'])}  stopped arms: {report['stopped_arms']}")
    for item in report["bound_stops"][:20]:
        print("BOUND-STOP", item)
    if reference is not None:
        print(f"weights differing from reference arm {args.reference_arm}: {len(mismatches)}")
        for item in mismatches[:20]:
            print("WEIGHT-MISMATCH", item)
    stop = report["stop"] or bool(mismatches)
    print("STOP: review before continuing." if stop else "No automatic stop condition.")
    return 2 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
