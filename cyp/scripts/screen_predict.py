"""Expose existing screen-head predictions for an explicit molecule file (Tripwire-CYP-010, D025).

Authorizes no run by itself. Needs an explicit input file; there is no default dataset and no
project dataset is read by this script except as disclosed below.

What it produces, in a NEW output directory (never an existing one):
  * four CYP screen outputs, native log2fc, from the single verified r1 screening checkpoint
    (epoch_016.ckpt), in the contract order CYP1A2, CYP2C9, CYP2D6, CYP3A4;
  * the PXR 8.25 uM screen output (output index 0), native log2fc, from each of the three
    verified PXR-004 stage-1 checkpoints (seeds 20261125, 20261126, 20261127), with each seed's
    prediction retained and their arithmetic mean reported;
  * a manifest with four checkpoint hashes, seeds, task order, units, input hash and code
    provenance.
No stage-2 potency checkpoint is ever used. Nothing is fitted, tuned or selected. No thresholds
or flags. The five potency outputs and every existing result file are neither read nor changed.

Disclosure: the CYP model is built through the pinned r1 builder
build_t121_screen_prefit_state. That builder reads the pinned screening input files (screen,
adjudication and contract files named in the CYP pins) to construct the model and dataset
object. No prediction is made on those molecules and nothing from them is written.

Input handling: the input file is read as text with the csv module. Identifiers are preserved
literally (no type inference, no stripping): "00042", "1E5", "NA" and "nan" stay as written.
Every input row yields exactly one output row in input order. Invalid rows are reported with a
status and reason and have empty prediction fields; none is dropped. Duplicate identifiers are
recorded per row and in the manifest. A malformed or blank CSV line stops the run. SMILES are
used as given: no standardization is applied, because none is recorded for the training data.

Model-mode rule: inference batches run with model.eval() and torch.inference_mode() only.
Nothing here toggles a model or its output transform into training mode on inference batches.
The PXR stage-1 transform is verified separately, and only when --verify-pxr-transform is given,
on a deep copy of the model; the model used for inference is not touched.

Order of work: the three PXR models first, then the CYP builder (which sets deterministic
algorithms globally), so that the PXR path runs in the same state as the loading smoke.

The join with the potency report is a separate, later step. CYP potency is keyed by
Molecule_Name and PXR potency by row_id; they are not assumed interchangeable, and a join needs
an explicit, reviewed mapping.

Usage (CoreHPC, run by the project owner; every argument is required):
    python -I screen_predict.py --cyp-pins P --cyp-pins-sha256 H --pxr-pins P --pxr-pins-sha256 H \
        --protected-hashes P --protected-hashes-sha256 H --input FILE --input-sha256 H \
        --id-column NAME --smiles-column NAME --out-dir NEW_DIR --batch-size N --threads N \
        [--verify-pxr-transform]
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CYP_ENZYMES = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
CYP_COLUMNS = tuple(f"{name}_screen_log2fc" for name in CYP_ENZYMES)
PXR_SEEDS = (20261125, 20261126, 20261127)
PXR_SEED_COLUMNS = tuple(f"PXR_screen_8p25uM_log2fc_seed{seed}" for seed in PXR_SEEDS)
PXR_MEAN_COLUMN = "PXR_screen_8p25uM_log2fc_mean3"
ROW_COLUMNS = ("input_row", "input_id", "input_smiles", "status", "invalid_reason", "warnings")
OUTPUT_COLUMNS = ROW_COLUMNS + CYP_COLUMNS + PXR_SEED_COLUMNS + (PXR_MEAN_COLUMN,)
OUTPUT_CSV = "screen_predictions.csv"
OUTPUT_MANIFEST = "screen_predictions_manifest.json"

STATUS_OK = "ok"
STATUS_MISSING_ID = "invalid_missing_id"
STATUS_EMPTY_SMILES = "invalid_empty_smiles"
STATUS_STRUCTURE = "invalid_structure"
STATUSES = (STATUS_OK, STATUS_MISSING_ID, STATUS_EMPTY_SMILES, STATUS_STRUCTURE)

JOIN_NOTE = (
    "CYP potency is keyed by Molecule_Name and PXR potency by row_id. They are not assumed "
    "interchangeable with input_id. A later join needs an explicit, reviewed mapping table. "
    "The CYP upload file has an exact six-column contract (D003) and must never be extended "
    "with these columns."
)


# ----------------------------------------------------------------------- pure helpers
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def new_output_dir(path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {path}")
    path.mkdir(parents=True)
    return path


def verify_hashed_files(entries: dict) -> dict:
    """Hash every {role: {path, sha256}} entry; raise on a missing file or any mismatch."""
    verified = {}
    for role, entry in sorted(entries.items()):
        path = Path(entry["path"])
        if not path.is_file():
            raise ValueError(f"Required file missing: {role}: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {role}: {path} (actual {actual}, expected {entry['sha256']})")
        verified[role] = {"path": str(path.resolve()), "sha256": actual}
    return verified


def verify_stage_files(stage_root, files: dict) -> int:
    stage_root = Path(stage_root)
    for relative, expected in sorted(files.items()):
        path = stage_root / relative
        if not path.is_file():
            raise ValueError(f"Staged file missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Hash mismatch: {path} (actual {actual}, expected {expected})")
    return len(files)


class ProtectedInputError(ValueError):
    pass


def load_protected_hashes(path, expected_sha256: str) -> dict:
    """sha256 -> {path, reason}. The list file itself must match its declared hash."""
    if sha256_file(path) != expected_sha256:
        raise ValueError("Protected-hash list differs from --protected-hashes-sha256.")
    entries = read_json(path)["entries"]
    table = {}
    for entry in entries:
        digest = entry["sha256"]
        if len(digest) != 64:
            raise ValueError(f"Malformed protected hash: {digest}")
        table[digest] = {"path": entry["path"], "reason": entry["reason"]}
    if not table:
        raise ValueError("Protected-hash list is empty.")
    return table


def check_input_not_protected(input_sha256: str, protected: dict) -> None:
    """Exact-file guard only; derived or excerpted protected data is not detected."""
    if input_sha256 in protected:
        hit = protected[input_sha256]
        raise ProtectedInputError(
            f"Input file is a protected file ({hit['path']}: {hit['reason']}); refusing to run."
        )


@dataclass(frozen=True)
class InputRow:
    index: int  # zero-based position among data rows
    line: int  # 1-based line number in the file (header is line 1)
    id: str  # literal text
    smiles: str  # literal text


def read_input_rows(path, id_column: str, smiles_column: str) -> list:
    """Read id and SMILES as literal strings. A malformed or blank line stops the run."""
    if id_column == smiles_column:
        raise ValueError("--id-column and --smiles-column must differ.")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("Input file is empty.") from None
        if len(set(header)) != len(header):
            raise ValueError("Input header has duplicate column names.")
        for name in (id_column, smiles_column):
            if name not in header:
                raise ValueError(f"Input file has no column named {name!r}; columns are {header}.")
        i_id, i_smiles = header.index(id_column), header.index(smiles_column)
        rows = []
        for record in reader:
            line = reader.line_num
            if len(record) != len(header):
                raise ValueError(
                    f"Line {line}: expected {len(header)} fields, found {len(record)} "
                    "(blank or malformed line); the run stops rather than dropping it."
                )
            rows.append(InputRow(len(rows), line, record[i_id], record[i_smiles]))
    if not rows:
        raise ValueError("Input file has no data rows.")
    return rows


def duplicate_ids(rows) -> dict:
    """Literal ids that occur more than once (blank ids excluded) -> list of row indices."""
    seen = {}
    for row in rows:
        if row.id.strip() == "":
            continue
        seen.setdefault(row.id, []).append(row.index)
    return {key: value for key, value in seen.items() if len(value) > 1}


def classify_rows(rows) -> list:
    """Pre-model checks: status, reason and warnings per row (structure is checked later)."""
    duplicates = duplicate_ids(rows)
    result = []
    for row in rows:
        status, reason, warnings = STATUS_OK, "", []
        if row.id.strip() == "":
            status, reason = STATUS_MISSING_ID, "input id is empty"
        elif row.smiles.strip() == "":
            status, reason = STATUS_EMPTY_SMILES, "input SMILES is empty"
        if row.id in duplicates:
            warnings.append("duplicate_id")
        if row.smiles != row.smiles.strip():
            warnings.append("smiles_has_surrounding_whitespace")
        if row.id != row.id.strip() and row.id.strip() != "":
            warnings.append("id_has_surrounding_whitespace")
        result.append({"status": status, "reason": reason, "warnings": warnings})
    return result


def fmt_float(value) -> str:
    """Full-precision text for a finite float; empty text for a missing value."""
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Refusing to write a non-finite prediction.")
    return repr(value)


def mean3(values) -> float | None:
    """Arithmetic mean in float64 of exactly three values; None unless all three are present."""
    values = tuple(values)
    if len(values) != 3 or any(v is None for v in values):
        return None
    return math.fsum(float(v) for v in values) / 3.0


def assemble_output_rows(rows, classification, cyp_values: dict, pxr_values: dict) -> list:
    """One dict per input row, in input order.

    cyp_values: row index -> sequence of four floats. pxr_values: seed -> {row index -> float}.
    A row that is not ok must have no values; an ok row must have all of them.
    """
    if len(rows) != len(classification):
        raise ValueError("Row and classification counts differ.")
    out = []
    for row, info in zip(rows, classification):
        record = {
            "input_row": row.index, "input_id": row.id, "input_smiles": row.smiles,
            "status": info["status"], "invalid_reason": info["reason"],
            "warnings": ";".join(info["warnings"]),
        }
        cyp = cyp_values.get(row.index)
        seeds = [pxr_values[seed].get(row.index) if seed in pxr_values else None for seed in PXR_SEEDS]
        if info["status"] == STATUS_OK:
            if cyp is None or len(cyp) != len(CYP_COLUMNS) or any(v is None for v in seeds):
                raise ValueError(f"Row {row.index} is ok but has missing predictions.")
        elif cyp is not None or any(v is not None for v in seeds):
            raise ValueError(f"Row {row.index} is invalid but has predictions.")
        for column, value in zip(CYP_COLUMNS, cyp if cyp is not None else [None] * len(CYP_COLUMNS)):
            record[column] = fmt_float(value)
        for column, value in zip(PXR_SEED_COLUMNS, seeds):
            record[column] = fmt_float(value)
        record[PXR_MEAN_COLUMN] = fmt_float(mean3(seeds))
        out.append(record)
    return out


def write_output_csv(path, records) -> None:
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        for record in records:
            writer.writerow([record[column] for column in OUTPUT_COLUMNS])


def status_counts(classification) -> dict:
    counts = {status: 0 for status in STATUSES}
    for info in classification:
        counts[info["status"]] += 1
    return counts


def collect_checkpoint_records(cyp_pins: dict, pxr_pins: dict) -> dict:
    """Hash-verify exactly four checkpoints: the r1 CYP screen checkpoint and three PXR stage-1 ones."""
    records = {}
    verified = verify_hashed_files({"checkpoint": cyp_pins["run_files"]["checkpoint"]})["checkpoint"]
    records["cyp_r1_epoch_016"] = {
        **verified, "zero_based_epoch": cyp_pins["expected"]["selected_epoch_zero_based"],
    }
    for seed in PXR_SEEDS:
        pin = pxr_pins["seeds"][str(seed)]
        path = Path(pin["run_dir"]) / "stage1" / "best.ckpt"
        if "stage2" in path.parts:
            raise ValueError("A stage-2 checkpoint path was constructed; refusing.")
        verified = verify_hashed_files({"checkpoint": {"path": str(path), "sha256": pin["best_checkpoint_sha256"]}})
        records[f"pxr_stage1_seed{seed}"] = {**verified["checkpoint"], "seed": seed}
    return records


def check_enzyme_order(task_names, contract_enzymes) -> None:
    if tuple(task_names) != CYP_ENZYMES or tuple(contract_enzymes) != CYP_ENZYMES:
        raise ValueError(
            f"CYP task order {tuple(task_names)} / contract {tuple(contract_enzymes)} "
            f"differs from {CYP_ENZYMES}."
        )


def build_manifest(**kwargs) -> dict:
    """Assemble the manifest; every provenance field is explicit and required."""
    required = {
        "script_sha256", "input", "checkpoints", "output_csv_sha256", "cyp", "pxr",
        "code_provenance", "pins", "environment", "run", "timing_seconds", "pxr_transform_check",
    }
    missing = required - set(kwargs)
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    checkpoints = kwargs["checkpoints"]
    expected = {"cyp_r1_epoch_016"} | {f"pxr_stage1_seed{seed}" for seed in PXR_SEEDS}
    if set(checkpoints) != expected:
        raise ValueError(f"Manifest must name exactly four checkpoints {sorted(expected)}.")
    return {
        "tool": "cyp/scripts/screen_predict.py",
        "decision": "D025",
        "status": "completed",
        "not_predictive_validation": True,
        "labels_used": False,
        "scored": False,
        "stage2_potency_checkpoints_used": False,
        "potency_outputs_read_or_changed": False,
        "outputs": {"csv": OUTPUT_CSV, "columns": list(OUTPUT_COLUMNS)},
        "join_note": JOIN_NOTE,
        **kwargs,
    }


# ------------------------------------------------------------------ model-dependent code
def scaler_from_state_dict(state_dict) -> tuple:
    """(mean, scale) vectors from the checkpoint's output-transform buffers; exactly one each."""
    import numpy as np

    means = [k for k in state_dict if k.endswith("output_transform.mean")]
    scales = [k for k in state_dict if k.endswith("output_transform.scale")]
    if len(means) != 1 or len(scales) != 1:
        raise ValueError(f"Expected one output-transform mean and scale; found {means}, {scales}.")
    mean = np.asarray(state_dict[means[0]].detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    scale = np.asarray(state_dict[scales[0]].detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    return mean, scale


def assert_eval_state(model) -> None:
    if any(module.training for module in model.modules()):
        raise ValueError("Model has modules in training mode; inference requires eval().")


def assert_native_identity_head(model, n_outputs: int) -> dict:
    import torch

    transform = model.predictor.output_transform
    if not isinstance(transform, torch.nn.Identity):
        raise ValueError(f"Output transform is {type(transform).__name__}, not Identity.")
    if [key for key in model.state_dict() if "output_transform" in key]:
        raise ValueError("Unexpected output-transform state in the CYP screening model.")
    linears = [
        (name, module) for name, module in model.predictor.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    name, last = linears[-1]
    if last.out_features != n_outputs:
        raise ValueError(f"Final projection {name} has {last.out_features} outputs, not {n_outputs}.")
    return {"output_transform": "Identity", "final_projection": name, "out_features": last.out_features}


def unscale_consistency(native, standardized, mean, scale, tol: float) -> dict:
    import numpy as np

    native = np.asarray(native, dtype=np.float64)
    standardized = np.asarray(standardized, dtype=np.float64)
    expected = standardized * np.asarray(scale, dtype=np.float64) + np.asarray(mean, dtype=np.float64)
    max_diff = float(np.max(np.abs(native - expected)))
    if max_diff > tol:
        raise ValueError(f"Eval output differs from standardized*scale+mean by {max_diff}.")
    if np.allclose(native, standardized, atol=tol):
        raise ValueError("Eval and train-mode outputs are identical: the transform is not applied.")
    return {"max_abs_difference": max_diff}


def featurization_error(smiles: str, featurizer):
    """None if the SMILES builds a molecule and featurizes; else a short error text."""
    from chemprop import data

    try:
        datapoint = data.MoleculeDatapoint.from_smi(smiles)
        if datapoint.mol is None:
            return "no molecule returned"
        featurizer(datapoint.mol, datapoint.V_f, datapoint.E_f)
    except Exception as exc:  # any failure is reported per row, never dropped
        text = f"{type(exc).__name__}: {exc}".replace("\n", " ").replace("\r", " ")
        return text[:200]
    return None


def predict_in_batches(model, smiles_list, featurizer, batch_size: int, n_outputs: int):
    """Native model outputs for smiles_list, in order. Eval mode and inference_mode only."""
    import numpy as np
    import torch
    from chemprop import data

    assert_eval_state(model)
    datapoints = [data.MoleculeDatapoint.from_smi(s) for s in smiles_list]
    kwargs = {} if featurizer is None else {"featurizer": featurizer}
    dataset = data.MoleculeDataset(datapoints, **kwargs)
    loader = data.build_dataloader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False
    )
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch[0], batch[1], batch[2])
            chunks.append(output.detach().cpu().numpy().astype(np.float64))
    result = np.concatenate(chunks, axis=0)
    if result.shape != (len(smiles_list), n_outputs):
        raise ValueError(f"Output shape {result.shape} differs from {(len(smiles_list), n_outputs)}.")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model outputs.")
    assert_eval_state(model)
    return result


def verify_pxr_transform_on_copy(model, smiles_list, featurizer, tol: float) -> dict:
    """Separate check on a deep copy: eval output equals train-mode output * scale + mean.

    The inference model is never toggled; its eval state and outputs are checked afterwards.
    """
    import numpy as np
    import torch
    from chemprop import data

    datapoints = [data.MoleculeDatapoint.from_smi(s) for s in smiles_list]
    kwargs = {} if featurizer is None else {"featurizer": featurizer}
    dataset = data.MoleculeDataset(datapoints, **kwargs)
    batch = next(iter(data.build_dataloader(
        dataset, batch_size=len(datapoints), num_workers=0, shuffle=False, drop_last=False)))
    bmg, V_d, X_d = batch[0], batch[1], batch[2]
    clone = copy.deepcopy(model)
    clone.eval()
    with torch.inference_mode():
        native = clone(bmg, V_d, X_d).detach().cpu().numpy().astype(np.float64)
        clone.predictor.output_transform.train()
        standardized = clone(bmg, V_d, X_d).detach().cpu().numpy().astype(np.float64)
    mean, scale = scaler_from_state_dict(clone.state_dict())
    result = unscale_consistency(native, standardized, mean, scale, tol)
    assert_eval_state(model)
    with torch.inference_mode():
        original = model(bmg, V_d, X_d).detach().cpu().numpy().astype(np.float64)
    if not np.array_equal(original, native):
        raise ValueError("Inference model output differs from the verified copy.")
    result.update({"n_molecules": len(datapoints), "inference_model_untouched": True})
    return result


def load_pxr_models(pins: dict, stage_root: Path, modules, tolerance32: float):
    """Three stage-1 screen heads. Scaler constants are pinned, then overwritten by the strict load."""
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler

    d, t = modules
    expected = pins["expected"]
    foundation = t.load_foundation(stage_root / pins["weights_relative_path"])
    if foundation["sha256"] != pins["weights_sha256"]:
        raise ValueError("CheMeleon weights hash differs from the pinned hash.")
    if d.AUX_TASKS[expected["screen_head_index"]] != "log2fc_8p25uM" or tuple(d.AUX_TASKS) != tuple(expected["head_order"]):
        raise ValueError("PXR head order differs from the pinned order.")
    models = {}
    for seed in PXR_SEEDS:
        pin = pins["seeds"][str(seed)]
        path = Path(pin["run_dir"]) / "stage1" / "best.ckpt"
        if "stage2" in path.parts:
            raise ValueError("A stage-2 checkpoint path was constructed; refusing.")
        actual = sha256_file(path)
        if actual != pin["best_checkpoint_sha256"]:
            raise ValueError(f"Stage-1 checkpoint hash mismatch for seed {seed}: {actual}")
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(expected["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(expected["scaler_scale"], dtype=np.float64)
        encoder = t.make_encoder(foundation["hyper_parameters"], foundation["state_dict"])
        model = t.build_model("stage1", encoder, scaler)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint for seed {seed} has no state_dict.")
        mean, scale = scaler_from_state_dict(checkpoint["state_dict"])
        if float(np.max(np.abs(mean - scaler.mean_))) > tolerance32 or float(np.max(np.abs(scale - scaler.scale_))) > tolerance32:
            raise ValueError(f"Checkpoint scaler buffers for seed {seed} differ from the pinned constants.")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        assert_eval_state(model)
        models[seed] = model
    return models


def load_cyp_model(pins: dict, weighted):
    """r1 screening model through the pinned builder; strict load of the selected checkpoint."""
    import torch

    inputs = pins["screen_inputs"]
    proposal = read_json(pins["pinned_files"]["proposal"]["path"])
    if proposal.get("trial_index") != 121:
        raise ValueError("Pinned proposal is not trial 121.")
    base = weighted.build_t121_screen_prefit_state(
        screen_path=inputs["screen_path"],
        identity_path=inputs["identity_path"],
        similarity_path=inputs["similarity_path"],
        contract_path=inputs["contract_path"],
        sampled_config=proposal["sampled_config"],
        config_name="D007-T121",
        training_seed=pins["training_seed"],
    )
    model, task_names = base["model"], tuple(base["training_dataset"].task_names)
    del base  # the screening dataset object is not used for any prediction
    contract = read_json(inputs["contract_path"])
    check_enzyme_order(task_names, contract["enzymes"])
    head = assert_native_identity_head(model, len(CYP_ENZYMES))
    checkpoint_entry = pins["run_files"]["checkpoint"]
    path = Path(checkpoint_entry["path"])
    actual = sha256_file(path)
    if actual != checkpoint_entry["sha256"]:
        raise ValueError(f"CYP checkpoint hash mismatch: {actual}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError("CYP checkpoint has no state_dict.")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    assert_eval_state(model)
    return model, task_names, head


# ------------------------------------------------------------------------------ the run
def run(args) -> dict:
    started = time.time()
    for path, expected in ((args.cyp_pins, args.cyp_pins_sha256), (args.pxr_pins, args.pxr_pins_sha256)):
        if sha256_file(path) != expected:
            raise ValueError(f"Pins file hash differs from the declared hash: {path}")
    cyp_pins, pxr_pins = read_json(args.cyp_pins), read_json(args.pxr_pins)

    # Input: hash, exact-file protected guard, literal parse, per-row classification.
    input_sha256 = sha256_file(args.input)
    if input_sha256 != args.input_sha256:
        raise ValueError("Input file hash differs from --input-sha256.")
    protected = load_protected_hashes(args.protected_hashes, args.protected_hashes_sha256)
    check_input_not_protected(input_sha256, protected)
    rows = read_input_rows(args.input, args.id_column, args.smiles_column)
    classification = classify_rows(rows)
    duplicates = duplicate_ids(rows)

    # Hash the pinned code and artifacts before any scientific import.
    verified_cyp = verify_hashed_files(cyp_pins["pinned_files"])
    stage_root = Path(pxr_pins["stage_root"])
    n_stage_files = verify_stage_files(stage_root, pxr_pins["stage_files"])
    checkpoints = collect_checkpoint_records(cyp_pins, pxr_pins)

    if str(stage_root) not in sys.path:
        sys.path.insert(0, str(stage_root))
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    import chemprop
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    from pxr.scripts import pxr004_data as d
    from pxr.scripts import pxr004_train as t

    for module in (d, t):
        if stage_root.resolve() not in Path(module.__file__).resolve().parents:
            raise ValueError(f"Imported {module.__name__} is not from the staged package.")
    pxr_featurizer = d.make_featurizer()
    cyp_featurizer = SimpleMoleculeMolGraphFeaturizer()

    # Per-row structure check with both featurizers; nothing is dropped.
    for row, info in zip(rows, classification):
        if info["status"] != STATUS_OK:
            continue
        errors = []
        for label, featurizer in (("cyp", cyp_featurizer), ("pxr", pxr_featurizer)):
            error = featurization_error(row.smiles, featurizer)
            if error is not None:
                errors.append(f"{label}: {error}")
        if errors:
            info["status"], info["reason"] = STATUS_STRUCTURE, "; ".join(errors)
    ok_rows = [row for row, info in zip(rows, classification) if info["status"] == STATUS_OK]
    ok_smiles = [row.smiles for row in ok_rows]

    pxr_values: dict = {}
    cyp_values: dict = {}
    transform_check = None
    cyp_info: dict = {"task_order": list(CYP_ENZYMES)}
    if ok_rows:
        # PXR first: same process state as the loading smoke.
        models = load_pxr_models(
            pxr_pins, stage_root, (d, t), pxr_pins["smoke"]["scaler_tolerance_float32"]
        )
        for seed in PXR_SEEDS:
            output = predict_in_batches(models[seed], ok_smiles, pxr_featurizer, args.batch_size, 3)
            pxr_values[seed] = {row.index: float(output[i, 0]) for i, row in enumerate(ok_rows)}
        if args.verify_pxr_transform:
            transform_check = verify_pxr_transform_on_copy(
                models[PXR_SEEDS[0]], ok_smiles[: args.batch_size], pxr_featurizer,
                pxr_pins["smoke"]["unscale_tolerance"],
            )
        del models

        # CYP last: the pinned builder sets deterministic algorithms globally.
        directory = str(Path(verified_cyp["screen_stage"]["path"]).parent)
        if directory != str(Path(verified_cyp["weighted_training"]["path"]).parent):
            raise ValueError("t121_screen_stage and t121_weighted_training must share a directory.")
        for entry in (directory,) + tuple(
            str(Path(verified_cyp[role]["path"]).parent) for role in verified_cyp if role.startswith("executor_")
        ):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        weighted = importlib.import_module(Path(verified_cyp["weighted_training"]["path"]).stem)
        if Path(weighted.__file__).resolve() != Path(verified_cyp["weighted_training"]["path"]).resolve():
            raise ValueError("Imported t121_weighted_training is not the verified file.")
        model, task_names, head = load_cyp_model(cyp_pins, weighted)
        output = predict_in_batches(model, ok_smiles, None, args.batch_size, len(CYP_ENZYMES))
        cyp_values = {row.index: [float(v) for v in output[i]] for i, row in enumerate(ok_rows)}
        cyp_info.update({"task_names_from_model": list(task_names), "head": head})

    records = assemble_output_rows(rows, classification, cyp_values, pxr_values)
    return {
        "records": records,
        "classification": classification,
        "manifest_parts": {
            "input": {
                "path": str(Path(args.input).resolve()), "sha256": input_sha256,
                "id_column": args.id_column, "smiles_column": args.smiles_column,
                "n_rows": len(rows), "status_counts": status_counts(classification),
                "duplicate_ids": {key: value for key, value in duplicates.items()},
                "ids_preserved_as_literal_text": True, "smiles_standardized": False,
                "protected_guard": {
                    "protected_list_sha256": args.protected_hashes_sha256,
                    "n_entries": len(protected),
                    "scope": "exact-file SHA-256 match only; derived or excerpted protected data is not detected",
                },
            },
            "checkpoints": checkpoints,
            "cyp": {
                **cyp_info,
                "units": "native log2fc (identity output transform; no target scaler)",
                "checkpoint_selection": "single r1 epoch_016.ckpt (zero-based epoch 16)",
                "builder_disclosure": (
                    "The pinned builder build_t121_screen_prefit_state reads the pinned screening "
                    "input files to construct the model and dataset object; no prediction is made "
                    "on those molecules and nothing from them is written."
                ),
                "screen_inputs": cyp_pins["screen_inputs"],
            },
            "pxr": {
                "head": "log2fc_8p25uM", "head_index": 0, "head_order": list(d.AUX_TASKS),
                "seeds": list(PXR_SEEDS),
                "units": "native log2fc (eval mode applies the stage-1 UnscaleTransform)",
                "mean": "arithmetic mean over the three seeds in float64, only when all three are present",
                "weights_sha256": pxr_pins["weights_sha256"],
                "scaler_source": "pinned constants, confirmed against the checkpoint buffers before the strict load",
                "stage": "stage-1 only; stage-2 potency checkpoints are never loaded",
            },
            "code_provenance": {
                "cyp_pinned_sources": verified_cyp,
                "pxr_stage_root": str(stage_root), "pxr_stage_files_verified": n_stage_files,
                "pxr_imported_modules": {
                    "pxr004_data": {"path": str(Path(d.__file__).resolve()), "sha256": sha256_file(d.__file__)},
                    "pxr004_train": {"path": str(Path(t.__file__).resolve()), "sha256": sha256_file(t.__file__)},
                },
            },
            "pins": {
                "cyp_pins_sha256": args.cyp_pins_sha256, "pxr_pins_sha256": args.pxr_pins_sha256,
            },
            "environment": {
                "python": platform.python_version(), "torch": torch.__version__,
                "chemprop": chemprop.__version__, "numpy": np.__version__, "host": platform.node(),
            },
            "run": {
                "device": "cpu", "threads": args.threads, "batch_size": args.batch_size,
                "model_mode": "eval and inference_mode on every inference batch; no training-mode toggling",
                "order": "PXR seeds first, then the CYP builder",
                "verify_pxr_transform_requested": bool(args.verify_pxr_transform),
            },
            "pxr_transform_check": transform_check,
        },
        "started": started,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--cyp-pins", "--pxr-pins", "--protected-hashes", "--input", "--out-dir"):
        parser.add_argument(flag, required=True, type=Path)
    for flag in ("--cyp-pins-sha256", "--pxr-pins-sha256", "--protected-hashes-sha256", "--input-sha256",
                 "--id-column", "--smiles-column"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--verify-pxr-transform", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("--batch-size and --threads must be at least 1.")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.out_dir}")

    result = run(args)
    out_dir = new_output_dir(args.out_dir)
    write_output_csv(out_dir / OUTPUT_CSV, result["records"])
    manifest = build_manifest(
        script_sha256=sha256_file(__file__),
        output_csv_sha256=sha256_file(out_dir / OUTPUT_CSV),
        timing_seconds={"total": time.time() - result["started"],
                        "note": "single run timing; not a benchmark and not a production estimate"},
        **result["manifest_parts"],
    )
    with open(out_dir / OUTPUT_MANIFEST, "x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": "completed",
        "rows": manifest["input"]["n_rows"],
        "status_counts": manifest["input"]["status_counts"],
        "csv": str(out_dir / OUTPUT_CSV),
        "manifest": str(out_dir / OUTPUT_MANIFEST),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
