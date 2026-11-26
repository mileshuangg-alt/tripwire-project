"""Updated-T121 potency predictions for an explicit molecule list (Tripwire-CYP-011, D028).

Reuses the 150 completed updated-T121 cell checkpoints (array job 3105880) without any
training. For each member the cell's own training/validation fold is rebuilt through the
same D007 path used for the original fit and for validation-only inference (route_d007_fold,
build_updated_t121_pic50_prefit_state), best.ckpt is loaded strictly, and the requested
molecules are predicted target-free (MoleculeDatapoint.from_smi, no labels).

Native units: the model is put in eval mode and called as model(bmg, V_d, X_d), as Chemprop's
own predict_step does. In eval mode the model's UnscaleTransform returns native pIC50 itself, so
no manual unscaling is applied; the transform's mean and scale must agree with the cell's
rebuilt training-only target scaler (checked before every prediction pass). Every member must first reproduce its saved validation-inference predictions
(--reference-dir, predictions_cell_NNN.csv) within --reference-tolerance on that cell's own
validation molecules; a mismatch stops the run and nothing is aggregated.

Ensemble rule (D028): equal-weight arithmetic mean over exactly the 150 members on the
original pIC50 scale. No substitution, exclusion or reweighting. Member SD (ddof=1) is
written to the ensemble file for description only and is not a report column.

These are deployment predictions of a pool whose members mostly trained on the 208-style
input molecules; they are not held-out predictions and the manifest says so.

Nothing has a default path, tolerance or accelerator. Inference must be authorized
explicitly. Writes only into a new, empty output directory. Chemprop-dependent steps import
lazily so the protocol and the aggregation code are testable without a GPU.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import screen_predict as sp  # noqa: E402
import t121_validation_inference as vi  # noqa: E402

TASKS = vi.TASKS
ISOFORMS = vi.ISOFORMS
N_MEMBERS = 150
REPETITIONS = tuple(range(1, 6))
FOLDS = tuple(range(1, 11))
SEEDS = (20261101, 20261102, 20261103)
EXPECTED_IDENTITIES = frozenset(itertools.product(REPETITIONS, FOLDS, SEEDS))

MEAN_COLUMNS = tuple(f"{task}_pred_mean150" for task in TASKS)
SD_COLUMNS = tuple(f"{task}_member_sd150" for task in TASKS)
ENSEMBLE_COLUMNS = sp.ROW_COLUMNS + MEAN_COLUMNS + SD_COLUMNS
MEMBER_COLUMNS = (
    "input_row", "input_id", "cell_index", "repetition", "fold", "seed",
) + TASKS
OUT_ENSEMBLE = "cyp_potency_ensemble.csv"
OUT_MEMBERS = "cyp_potency_members.csv"
OUT_MANIFEST = "cyp_potency_manifest.json"

DEPLOYMENT_NOTE = (
    "Deployment predictions of the 150-member updated-T121 pool. The members were fitted "
    "on cross-validation folds of the CYP development data, so a development-data molecule "
    "was in the training set of most members. These are not held-out predictions and are "
    "not cross-validation evaluation rows."
)


class PotencyStop(RuntimeError):
    """A condition that stops the run; nothing is aggregated or substituted."""


# ----------------------------------------------------------------------- pure helpers
def cell_paths(run_root, trial_index: int, cell_index: int):
    trial_dir = Path(run_root) / "trials" / f"trial_{trial_index:03d}"
    return trial_dir, trial_dir / "cells" / f"cell_{cell_index:03d}"


def expand_member_cells(text: str) -> list:
    cells = vi.parse_cells(text)
    if len(cells) != N_MEMBERS:
        raise PotencyStop(f"--cells must name exactly {N_MEMBERS} cells; got {len(cells)}.")
    return cells


def verify_cell(run_root, trial_index: int, cell_index: int) -> dict:
    """Identity and hash check of one member before its model is built. Stops on any failure."""
    _trial_dir, cell_dir = cell_paths(run_root, trial_index, cell_index)
    for required in ("COMPLETED.json", "result.json", "cell_plan.json"):
        if not (cell_dir / required).is_file():
            raise PotencyStop(f"Cell {cell_index} is not complete: missing {required}.")
    if (cell_dir / "INCOMPLETE.json").exists():
        raise PotencyStop(f"Cell {cell_index} has INCOMPLETE.json.")
    plan = vi.read_json(cell_dir / "cell_plan.json")
    result = vi.read_json(cell_dir / "result.json")
    if plan["trial_index"] != trial_index or plan["cell_index"] != cell_index:
        raise PotencyStop(f"Cell {cell_index}: identity does not match its frozen plan.")
    if result["trial_index"] != trial_index or result["cell_index"] != cell_index:
        raise PotencyStop(f"Cell {cell_index}: identity does not match its result.")
    checkpoint = cell_dir / "fit" / "best.ckpt"
    if not checkpoint.is_file():
        raise PotencyStop(f"Cell {cell_index}: best.ckpt is missing.")
    actual = vi.sha256_file(checkpoint)
    if actual != result["checkpoint_sha256"]:
        raise PotencyStop(
            f"Cell {cell_index}: best.ckpt hash {actual} differs from result.json "
            f"{result['checkpoint_sha256']}."
        )
    return {
        "cell_index": cell_index, "repetition": plan["repetition"],
        "fold": plan["fold"], "seed": plan["seed"],
        "checkpoint_sha256": actual,
        "split_manifest_sha256": result["split_manifest_sha256"],
    }


def verify_members(run_root, trial_index: int, cells) -> list:
    """Identity and hash check of all members before any model is built. Stops on any failure."""
    members, identities = [], set()
    for cell_index in cells:
        member = verify_cell(run_root, trial_index, cell_index)
        identity = (member["repetition"], member["fold"], member["seed"])
        if identity in identities:
            raise PotencyStop(f"Duplicate member identity {identity}.")
        identities.add(identity)
        members.append(member)
    if identities != EXPECTED_IDENTITIES:
        missing = sorted(EXPECTED_IDENTITIES - identities)
        extra = sorted(identities - EXPECTED_IDENTITIES)
        raise PotencyStop(
            f"Member identities are not exactly 5 repetitions x 10 folds x 3 seeds "
            f"(missing {missing[:3]}, unexpected {extra[:3]})."
        )
    return members


def ensemble_mean_sd(member_matrix):
    """member_matrix: (150, n_molecules, 4) native pIC50. Equal-weight mean; SD with ddof=1."""
    matrix = np.asarray(member_matrix, dtype=np.float64)
    if matrix.ndim != 3 or matrix.shape[0] != N_MEMBERS or matrix.shape[2] != len(TASKS):
        raise PotencyStop(f"Member matrix shape {matrix.shape} is not (150, n, 4).")
    if not np.isfinite(matrix).all():
        raise PotencyStop("Non-finite member predictions.")
    return matrix.mean(axis=0), matrix.std(axis=0, ddof=1)


def load_reference(reference_dir, cell_index: int) -> dict:
    """(Molecule_Name, isoform) -> saved y_pred for one cell. Duplicate keys stop the run."""
    path = Path(reference_dir) / f"predictions_cell_{cell_index:03d}.csv"
    if not path.is_file():
        raise PotencyStop(f"Reference predictions missing: {path}")
    values = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["cell_index"]) != cell_index:
                raise PotencyStop(f"{path}: row for a different cell.")
            key = (row["Molecule_Name"], row["isoform"])
            if key in values:
                raise PotencyStop(f"{path}: duplicate reference key {key}.")
            values[key] = float(row["y_pred"])
    if not values:
        raise PotencyStop(f"{path}: no reference rows.")
    return values


def compare_to_reference(molecule_names, native, reference: dict, tolerance: float) -> dict:
    """Max |prediction - saved y_pred| over every (molecule, isoform) the reference holds."""
    native = np.asarray(native, dtype=np.float64)
    index = {name: i for i, name in enumerate(molecule_names)}
    diffs = []
    for (name, isoform), saved in reference.items():
        if name not in index:
            raise PotencyStop(f"Reference molecule {name} is not in the gate molecules.")
        column = ISOFORMS.index(isoform)
        diffs.append(abs(float(native[index[name], column]) - saved))
    worst = max(diffs)
    return {
        "n_compared": len(diffs), "max_abs_diff": worst,
        "tolerance": tolerance, "within_tolerance": bool(worst <= tolerance),
    }


def ensemble_rows(rows, classification, structure_errors: dict, mean, sd) -> list:
    """One output record per input row; model values only for rows that were predicted."""
    records, k = [], 0
    for row, info in zip(rows, classification):
        status, reason = info["status"], info["reason"]
        if status == sp.STATUS_OK and row.index in structure_errors:
            status, reason = sp.STATUS_STRUCTURE, structure_errors[row.index]
        record = {
            "input_row": row.index, "input_id": row.id, "input_smiles": row.smiles,
            "status": status, "invalid_reason": reason,
            "warnings": ";".join(info["warnings"]),
        }
        if status == sp.STATUS_OK:
            for j in range(len(TASKS)):
                record[MEAN_COLUMNS[j]] = sp.fmt_float(mean[k, j])
                record[SD_COLUMNS[j]] = sp.fmt_float(sd[k, j])
            k += 1
        else:
            for name in MEAN_COLUMNS + SD_COLUMNS:
                record[name] = ""
        records.append(record)
    if k != mean.shape[0]:
        raise PotencyStop("Predicted molecule count does not match the ok rows.")
    return records


def write_csv_exclusive(path, columns, records) -> None:
    with open(path, "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


# ------------------------------------------------------------------ model-dependent code
def load_ops():
    import pandas as pd
    import torch
    import cyp007_cell_fit as fit
    import t121_weighted_training as weighted

    return {"pd": pd, "torch": torch, "fit": fit, "weighted": weighted}


def check_unscale_transform(transform, scaler, unscale_class, atol: float = 1e-5) -> None:
    """The model must carry an eval-mode UnscaleTransform whose mean and scale match the scaler."""
    if not isinstance(transform, unscale_class):
        raise PotencyStop(f"Model output transform is {type(transform).__name__}, not UnscaleTransform.")
    if transform.training:
        raise PotencyStop("Output transform is in training mode; it would not unscale.")
    for name, expected in (("mean", scaler.mean_), ("scale", scaler.scale_)):
        held = getattr(transform, name).detach().cpu().numpy().astype(np.float64).reshape(-1)
        wanted = np.asarray(expected, dtype=np.float64).reshape(-1)
        if held.shape != (len(TASKS),) or wanted.shape != (len(TASKS),):
            raise PotencyStop(f"Output transform {name} does not have four per-task values.")
        if not np.allclose(held, wanted, rtol=1e-6, atol=atol):
            raise PotencyStop(f"Output transform {name} {held} differs from the cell scaler {wanted}.")


def predict_native(model, scaler, smiles_list, batch_size: int):
    """Native pIC50 for smiles_list, in order. Eval mode and inference_mode only."""
    import torch
    from chemprop import data
    from chemprop.nn.transforms import UnscaleTransform

    if model.training:
        raise PotencyStop("Model must be in eval mode before inference.")
    check_unscale_transform(model.predictor.output_transform, scaler, UnscaleTransform)
    points = [data.MoleculeDatapoint.from_smi(s) for s in smiles_list]
    loader = data.build_dataloader(
        data.MoleculeDataset(points), batch_size=batch_size, num_workers=0,
        shuffle=False, drop_last=False,
    )
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            native = model(batch[0], batch[1], batch[2])  # eval mode: the transform unscales
            native = native.detach().cpu().numpy().astype(np.float64)
            if native.ndim != 2 or native.shape[1] != len(TASKS):
                raise PotencyStop(f"Unexpected prediction shape {native.shape}.")
            chunks.append(native)
    result = np.concatenate(chunks, axis=0)
    if result.shape != (len(smiles_list), len(TASKS)) or not np.isfinite(result).all():
        raise PotencyStop("Model output has the wrong shape or is non-finite.")
    if model.training:
        raise PotencyStop("Model left eval mode during inference.")
    return result


def build_member_model(ops, member, args, split_manifest_root):
    """Rebuild one cell exactly as validation-only inference does; load best.ckpt strictly."""
    pd, torch, fit, weighted = ops["pd"], ops["torch"], ops["fit"], ops["weighted"]
    cell_index = member["cell_index"]
    trial_dir, cell_dir = cell_paths(args.run_root, args.trial_index, cell_index)
    plan = vi.read_json(cell_dir / "cell_plan.json")
    proposal = vi.read_json(trial_dir / "proposal.json")
    manifest_path = Path(split_manifest_root) / (
        f"rep_{plan['repetition']:02d}_fold_{plan['fold']:02d}.json"
    )
    manifest = vi.read_json(manifest_path)
    if vi.sha256_file(manifest_path) != member["split_manifest_sha256"]:
        raise PotencyStop(f"Cell {cell_index}: split manifest hash mismatch.")
    if plan["seed"] not in tuple(manifest["training_seeds"]):
        raise PotencyStop(f"Cell {cell_index}: seed not in the frozen split manifest.")
    routed = fit.route_d007_fold(
        dataframe=pd.read_csv(args.dataset),
        original_split_dataframe=pd.read_csv(args.original_split),
        split_manifest=manifest, training_seed=plan["seed"],
    )
    prefit = weighted.build_updated_t121_pic50_prefit_state(
        training_dataframe=routed["training_dataframe"],
        validation_dataframe=routed["validation_dataframe"],
        sampled_config=proposal["sampled_config"],
        config_name=proposal.get("config_name", f"D007-T{args.trial_index:03d}"),
        training_seed=plan["seed"],
    )
    model = prefit["model"]
    checkpoint = torch.load(cell_dir / "fit" / "best.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, prefit["scaler"], routed["validation_dataframe"]


def run_timing_only(args, ops, builder, member, smiles, out_dir) -> dict:
    """One member end to end (build, reference gate, input prediction); writes only a timing record.

    No ensemble, member or report file is written, and the gate is enforced exactly as in a full run.
    The extrapolation is a straight multiple of this one member's total and is not a guarantee.
    """
    started = time.time()
    model, scaler, validation_df = builder(ops, member, args, args.split_manifest_root)
    built = time.time()
    names = validation_df["Molecule_Name"].tolist()
    gate_native = predict_native(model, scaler, validation_df["SMILES"].tolist(), args.batch_size)
    gated = time.time()
    gate = compare_to_reference(
        names, gate_native, load_reference(args.reference_dir, member["cell_index"]),
        args.reference_tolerance,
    )
    gate["cell_index"] = member["cell_index"]
    if not gate["within_tolerance"]:
        vi.write_json_exclusive(out_dir / "STOPPED_ON_REFERENCE_MISMATCH.json", {"gate": gate})
        raise PotencyStop(f"Reference mismatch in cell {member['cell_index']}: {gate}")
    predicted_native = predict_native(model, scaler, smiles, args.batch_size)
    finished = time.time()
    if predicted_native.shape != (len(smiles), len(TASKS)):
        raise PotencyStop("Timing prediction has an unexpected shape.")
    total = finished - started
    record = {
        "mode": "timing_only", "cell_index": member["cell_index"], "accelerator": args.accelerator,
        "hostname": socket.gethostname(), "batch_size": args.batch_size,
        "gate_molecules": len(names), "input_molecules_predicted": len(smiles),
        "seconds": {
            "build_and_load": built - started, "reference_gate_pass": gated - built,
            "input_prediction": finished - gated, "member_total": total,
        },
        "reference_gate": gate,
        "extrapolated_150_member_seconds": total * N_MEMBERS,
        "extrapolation_note": "one member times 150; one-member cold start included; not a guarantee",
        "ensemble_files_written": False, "script_sha256": vi.sha256_file(__file__),
    }
    vi.write_json_exclusive(out_dir / "cyp_potency_timing.json", record)
    return record


def run(args, ops=None, builder=build_member_model) -> dict:
    if args.authorize_inference is not True:
        raise PotencyStop("Updated-T121 potency inference is not authorized.")
    out_dir = sp.new_output_dir(args.out_dir)
    for role, path, expected in (
        ("dataset", args.dataset, args.dataset_sha256),
        ("original_split", args.original_split, args.original_split_sha256),
        ("input", args.input, args.input_sha256),
    ):
        actual = vi.sha256_file(path)
        if actual != expected:
            raise PotencyStop(f"{role} hash mismatch: {actual} != {expected}")
    protected = sp.load_protected_hashes(args.protected_hashes, args.protected_hashes_sha256)
    sp.check_input_not_protected(vi.sha256_file(args.input), protected)

    cells = expand_member_cells(args.cells)
    timing_cell = getattr(args, "timing_only_cell", None)
    if timing_cell is not None:
        if timing_cell not in cells:
            raise PotencyStop(f"--timing-only-cell {timing_cell} is not among --cells.")
        members = [verify_cell(args.run_root, args.trial_index, timing_cell)]
    else:
        members = verify_members(args.run_root, args.trial_index, cells)
    rows = sp.read_input_rows(args.input, args.id_column, args.smiles_column)
    classification = sp.classify_rows(rows)

    ops = load_ops() if ops is None else ops
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    featurizer = SimpleMoleculeMolGraphFeaturizer()
    structure_errors = {}
    for row, info in zip(rows, classification):
        if info["status"] == sp.STATUS_OK:
            error = sp.featurization_error(row.smiles, featurizer)
            if error is not None:
                structure_errors[row.index] = error
    ok_rows = [
        row for row, info in zip(rows, classification)
        if info["status"] == sp.STATUS_OK and row.index not in structure_errors
    ]
    smiles = [row.smiles for row in ok_rows]

    if timing_cell is not None:
        return run_timing_only(args, ops, builder, members[0], smiles, out_dir)

    matrix = np.empty((N_MEMBERS, len(ok_rows), len(TASKS)), dtype=np.float64)
    gates, timings = [], []
    for slot, member in enumerate(members):
        started = time.time()
        model, scaler, validation_df = builder(ops, member, args, args.split_manifest_root)
        names = validation_df["Molecule_Name"].tolist()
        gate_native = predict_native(model, scaler, validation_df["SMILES"].tolist(), args.batch_size)
        gate = compare_to_reference(
            names, gate_native, load_reference(args.reference_dir, member["cell_index"]),
            args.reference_tolerance,
        )
        gate["cell_index"] = member["cell_index"]
        gates.append(gate)
        if not gate["within_tolerance"]:
            vi.write_json_exclusive(out_dir / "STOPPED_ON_REFERENCE_MISMATCH.json", {
                "gate": gate, "note": "Stopped; nothing aggregated, no substitution.",
            })
            raise PotencyStop(f"Reference mismatch in cell {member['cell_index']}: {gate}")
        matrix[slot] = predict_native(model, scaler, smiles, args.batch_size)
        timings.append(time.time() - started)
        print(json.dumps({"cell": member["cell_index"], "gate_max_abs_diff": gate["max_abs_diff"],
                          "seconds": timings[-1]}), flush=True)

    mean, sd = ensemble_mean_sd(matrix)
    write_csv_exclusive(
        out_dir / OUT_ENSEMBLE, ENSEMBLE_COLUMNS,
        ensemble_rows(rows, classification, structure_errors, mean, sd),
    )
    member_records = []
    for slot, member in enumerate(members):
        for k, row in enumerate(ok_rows):
            record = {"input_row": row.index, "input_id": row.id,
                      "cell_index": member["cell_index"], "repetition": member["repetition"],
                      "fold": member["fold"], "seed": member["seed"]}
            for j, task in enumerate(TASKS):
                record[task] = sp.fmt_float(matrix[slot, k, j])
            member_records.append(record)
    write_csv_exclusive(out_dir / OUT_MEMBERS, MEMBER_COLUMNS, member_records)

    manifest = {
        "decision": "D028", "deployment_note": DEPLOYMENT_NOTE,
        "ensemble_rule": "equal-weight arithmetic mean over exactly 150 members, native pIC50",
        "n_members": N_MEMBERS, "members": members, "reference_gates": gates,
        "inputs": {
            "dataset_sha256": args.dataset_sha256, "original_split_sha256": args.original_split_sha256,
            "input_sha256": args.input_sha256,
            "protected_hashes_sha256": args.protected_hashes_sha256,
            "executor_source_sha256": args.executor_source_sha256,
        },
        "script_sha256": vi.sha256_file(__file__),
        "input_rows": len(rows), "predicted_rows": len(ok_rows),
        "status_counts": sp.status_counts(
            [{"status": (sp.STATUS_STRUCTURE if r.index in structure_errors else c["status"])}
             for r, c in zip(rows, classification)]),
        "accelerator": args.accelerator, "hostname": socket.gethostname(),
        "member_seconds_mean": float(np.mean(timings)), "member_seconds_total": float(np.sum(timings)),
        "outputs": {
            OUT_ENSEMBLE: vi.sha256_file(out_dir / OUT_ENSEMBLE),
            OUT_MEMBERS: vi.sha256_file(out_dir / OUT_MEMBERS),
        },
        "training_performed": False, "labels_read_for_input_molecules": False,
    }
    vi.write_json_exclusive(out_dir / OUT_MANIFEST, manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("run-root", "dataset", "dataset-sha256", "original-split", "original-split-sha256",
                 "split-manifest-root", "executor-source-dir", "executor-source-sha256",
                 "package-root", "reference-dir", "protected-hashes", "protected-hashes-sha256",
                 "input", "input-sha256", "id-column", "smiles-column", "out-dir", "cells"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--reference-tolerance", required=True, type=float)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--accelerator", required=True, choices=["cpu", "gpu"])
    parser.add_argument("--authorize-inference", action="store_true")
    parser.add_argument("--timing-only-cell", type=int, default=None,
                        help="run exactly this one member and write only a timing record")
    args = parser.parse_args(argv)

    source = Path(args.executor_source_dir) / "cyp007_cell_fit.py"
    digest = vi.sha256_file(source)
    if digest != args.executor_source_sha256:
        raise SystemExit(f"Executor source hash mismatch: {source} {digest}")
    package_root = Path(args.package_root)
    for directory in (args.executor_source_dir, str(package_root / "cyp/scripts"), str(package_root)):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    try:
        manifest = run(args)
    except PotencyStop as exc:
        raise SystemExit(f"STOPPED: {exc}")
    if manifest.get("mode") == "timing_only":
        print(json.dumps({k: manifest[k] for k in ("cell_index", "seconds", "extrapolated_150_member_seconds")}))
        return 0
    print(json.dumps({k: manifest[k] for k in ("n_members", "input_rows", "predicted_rows",
                                               "member_seconds_total")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
