"""CYP screening-head loading smoke (Tripwire-CYP-010). Authorizes no run by itself.

Verifies, on CPU and with no training, that the r1 updated-T121 screening checkpoint
(epoch 16, zero-based) loads strictly into the model the screening stage built, that its four
outputs are in the contract enzyme order (CYP3A4 = index 3), and that the outputs are native
log2fc (no target scaler, identity output transform). It runs one eval-mode forward pass on a
small deterministic set of screening molecules. It scores nothing: no loss, no MSE, no
comparison with labels.

Source of truth for every path and hash is the frozen pins file
(cyp/artifacts/screen_heads_smoke/cyp_r1_pins.json), whose own SHA-256 must be passed on the
command line. Every pinned file is hashed before any scientific import. Code is imported only
from the pinned r1 package, never from this repository checkout.

Scaling and eval mode (resolved from Chemprop 2.3.1 source):
  * the screening model is built with output_scaler=None, so its output transform is an
    identity and its outputs are native log2fc under both train and eval mode;
  * the PXR stage-1 model differs (it has an UnscaleTransform); see
    pxr/scripts/screen_head_smoke_pxr.py.

Usage (CoreHPC, run by the project owner):
    /scratch/user/mhuang8/envs/tripwire_cyp002/bin/python -I screen_head_smoke_cyp.py \
        --pins PINS.json --pins-sha256 HEX --out-dir NEW_DIR --threads 2
The out-dir must not exist. The script never writes anywhere else.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
import sys
import time
from pathlib import Path

CYP_ENZYMES = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


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


def verify_hashed_files(entries: dict) -> dict:
    """Hash every {role: {path, sha256}} entry; raise on a missing file or any mismatch."""
    verified = {}
    for role, entry in sorted(entries.items()):
        path = Path(entry["path"])
        if not path.is_file():
            raise ValueError(f"Required file missing: {role}: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"Hash mismatch: {role}: {path} (actual {actual}, expected {entry['sha256']})"
            )
        verified[role] = {"path": str(path.resolve()), "sha256": actual}
    return verified


def new_output_dir(path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {path}")
    path.mkdir(parents=True)
    return path


def check_enzyme_order(task_names, contract_enzymes, expected_order) -> int:
    """The model's task names must equal the contract order, which must equal the expected one.

    Returns the output index of CYP3A4.
    """
    task_names = tuple(task_names)
    contract_enzymes = tuple(contract_enzymes)
    expected_order = tuple(expected_order)
    if contract_enzymes != expected_order:
        raise ValueError(f"Contract enzymes {contract_enzymes} differ from expected {expected_order}.")
    if task_names != expected_order:
        raise ValueError(f"Model task names {task_names} differ from expected {expected_order}.")
    return task_names.index("CYP3A4")


def read_split_assignments(path) -> dict:
    """molecule_id -> partition. The file is written with CRLF line endings; csv handles them."""
    result = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or [n.strip() for n in reader.fieldnames] != [
            "molecule_id", "cluster_id", "partition",
        ]:
            raise ValueError(f"Unexpected split-assignment header: {reader.fieldnames}")
        for row in reader:
            name = row["molecule_id"].strip()
            if name in result:
                raise ValueError(f"Duplicate molecule in split assignments: {name}")
            result[name] = row["partition"].strip()
    return result


def partition_counts(assignments: dict) -> dict:
    counts = {}
    for partition in assignments.values():
        counts[partition] = counts.get(partition, 0) + 1
    return counts


def select_smoke_ids(assignments: dict, n_per_partition: int) -> dict:
    """First n molecule ids by sorted order in each partition (deterministic, label-free)."""
    selected = {}
    for partition in ("train", "validation"):
        ids = sorted(name for name, part in assignments.items() if part == partition)
        if len(ids) < n_per_partition:
            raise ValueError(f"Partition {partition} has fewer than {n_per_partition} molecules.")
        selected[partition] = ids[:n_per_partition]
    return selected


def verify_selection_records(selection, selected_record, expected, checkpoint_sha256) -> None:
    """Cross-check selection.json, selected_screen_record.json and the pinned expectations."""
    if selection.get("epoch_numbering") != "zero_based":
        raise ValueError("Selection record is not zero-based.")
    records = selection["records"]
    if len(records) != expected["n_epoch_records"]:
        raise ValueError(f"Expected {expected['n_epoch_records']} epoch records, found {len(records)}.")
    chosen = selection["selected"]
    if chosen != selected_record:
        raise ValueError("selected_screen_record.json differs from selection.json's selected record.")
    if chosen["epoch"] != expected["selected_epoch_zero_based"]:
        raise ValueError(f"Selected epoch {chosen['epoch']} differs from the expected epoch.")
    if chosen["sha256"] != checkpoint_sha256:
        raise ValueError("Selected record's checkpoint hash differs from the pinned checkpoint hash.")
    if chosen["score"] != expected["selected_score"]:
        raise ValueError("Selected score differs from the recorded value.")
    if list(chosen["per_enzyme"]) != list(expected["selected_per_enzyme"]):
        raise ValueError("Selected per-enzyme scores differ from the recorded values.")
    minimum = min(records, key=lambda record: (record["score"], record["epoch"]))
    if minimum["epoch"] != chosen["epoch"]:
        raise ValueError("Selected epoch is not the minimum-score epoch of the records.")


def check_native_outputs(values, n_rows: int, n_cols: int, abs_max: float) -> dict:
    """Shape, finiteness and a loose sanity band on a [rows, cols] array. Not a score."""
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n_rows, n_cols):
        raise ValueError(f"Output shape {array.shape} differs from {(n_rows, n_cols)}.")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite model outputs.")
    if float(np.abs(array).max()) > abs_max:
        raise ValueError(f"An output exceeds the sanity bound {abs_max}.")
    return {
        "shape": list(array.shape),
        "column_min": [float(v) for v in array.min(axis=0)],
        "column_max": [float(v) for v in array.max(axis=0)],
        "column_mean": [float(v) for v in array.mean(axis=0)],
    }


def assert_native_identity_head(model, n_outputs: int) -> dict:
    """Identity output transform, no scale buffers, final projection has n_outputs outputs."""
    import torch

    transform = model.predictor.output_transform
    if not isinstance(transform, torch.nn.Identity):
        raise ValueError(f"Output transform is {type(transform).__name__}, not Identity.")
    scale_keys = [key for key in model.state_dict() if "output_transform" in key]
    if scale_keys:
        raise ValueError(f"Unexpected output-transform state: {scale_keys}")
    linears = [
        (name, module) for name, module in model.predictor.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if not linears:
        raise ValueError("Predictor has no Linear projection.")
    name, last = linears[-1]
    if last.out_features != n_outputs:
        raise ValueError(f"Final projection {name} has {last.out_features} outputs, not {n_outputs}.")
    return {"output_transform": "Identity", "final_projection": name, "out_features": last.out_features}


class Checks:
    def __init__(self):
        self.items = []

    def record(self, name: str, ok: bool, detail=None) -> None:
        self.items.append({"check": name, "pass": bool(ok), "detail": detail})

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(item["pass"] for item in self.items)


# ------------------------------------------------------------------------- the smoke
def run(args) -> dict:
    started = time.time()
    if sha256_file(args.pins) != args.pins_sha256:
        raise ValueError("Pins file hash differs from --pins-sha256.")
    pins = read_json(args.pins)
    checks = Checks()
    checks.record("pins_file_hash", True, args.pins_sha256)

    # 1. Hash every pinned source/input and every run file before any scientific import.
    verified_sources = verify_hashed_files(pins["pinned_files"])
    verified_run = verify_hashed_files(pins["run_files"])
    checks.record("pinned_sources_and_inputs_hash", True, len(verified_sources))
    checks.record("run_files_hash", True, sorted(verified_run))

    expected = pins["expected"]
    run_files = pins["run_files"]
    selection = read_json(run_files["selection_json"]["path"])
    selected_record = read_json(run_files["selected_record"]["path"])
    verify_selection_records(selection, selected_record, expected, run_files["checkpoint"]["sha256"])
    checks.record("selection_records_consistent", True, {
        "epoch": selected_record["epoch"], "score": selected_record["score"],
    })

    assignments = read_split_assignments(run_files["split_assignments"]["path"])
    counts = partition_counts(assignments)
    checks.record("split_counts", counts == expected["split_counts"], counts)
    preparation = read_json(run_files["preparation_json"]["path"])
    checks.record(
        "preparation_matches_split",
        preparation["counts"] == expected["split_counts"]
        and preparation["split_seed"] == expected["split_seed"]
        and preparation["target_normalization"] == "none; native log2fc retained",
        {"counts": preparation["counts"], "target_normalization": preparation["target_normalization"]},
    )

    # 2. Import only the verified r1 package code.
    directory = str(Path(verified_sources["screen_stage"]["path"]).parent)
    if directory != str(Path(verified_sources["weighted_training"]["path"]).parent):
        raise ValueError("t121_screen_stage and t121_weighted_training must share a directory.")
    for entry in (directory,) + tuple(
        str(Path(verified_sources[role]["path"]).parent)
        for role in verified_sources if role.startswith("executor_")
    ):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    weighted = importlib.import_module(Path(verified_sources["weighted_training"]["path"]).stem)
    if Path(weighted.__file__).resolve() != Path(verified_sources["weighted_training"]["path"]).resolve():
        raise ValueError("Imported t121_weighted_training is not the verified file.")
    checks.record("imported_module_is_pinned_file", True, weighted.__file__)

    import chemprop
    from chemprop import data as chemprop_data
    from torch.utils.data import Subset

    # 3. Build the screening model and dataset exactly as the screening stage did; no fit.
    proposal = read_json(pins["pinned_files"]["proposal"]["path"])
    if proposal.get("trial_index") != 121:
        raise ValueError("Pinned proposal is not trial 121.")
    inputs = pins["screen_inputs"]
    t0 = time.time()
    base = weighted.build_t121_screen_prefit_state(
        screen_path=inputs["screen_path"],
        identity_path=inputs["identity_path"],
        similarity_path=inputs["similarity_path"],
        contract_path=inputs["contract_path"],
        sampled_config=proposal["sampled_config"],
        config_name="D007-T121",
        training_seed=pins["training_seed"],
    )
    build_seconds = time.time() - t0
    model, dataset = base["model"], base["training_dataset"]

    # 4. Output-column order, from the dataset's own task names and the contract.
    contract = read_json(inputs["contract_path"])
    cyp3a4_index = check_enzyme_order(dataset.task_names, contract["enzymes"], expected["enzyme_order"])
    checks.record("column_order_cyp3a4_index", cyp3a4_index == expected["cyp3a4_index"], {
        "task_names": list(dataset.task_names), "cyp3a4_index": cyp3a4_index,
    })
    population = set(dataset.molecule_ids)
    checks.record(
        "dataset_population_equals_split",
        population == set(assignments) and len(population) == sum(expected["split_counts"].values()),
        {"dataset": len(population), "split": len(assignments)},
    )

    # 5. Units: no target scaler, identity output transform, four outputs.
    head = assert_native_identity_head(model, expected["n_outputs"])
    checks.record("native_units_identity_head", True, head)

    # 6. Strict load of the selected checkpoint (state_dict only; no optimizer or scheduler).
    checkpoint_path = Path(run_files["checkpoint"]["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no state_dict.")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    checks.record("strict_state_dict_load", True, {
        "keys": len(checkpoint["state_dict"]),
        "checkpoint_epoch_field": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    })

    # 7. One eval-mode forward pass on a small deterministic set. Nothing is scored.
    model.eval()
    if any(module.training for module in model.modules()):
        raise ValueError("Model has modules still in training mode after eval().")
    chosen = select_smoke_ids(assignments, pins["smoke"]["n_per_partition"])
    ordered_ids = chosen["train"] + chosen["validation"]
    position = {name: index for index, name in enumerate(dataset.molecule_ids)}
    indices = [position[name] for name in ordered_ids]
    loader = chemprop_data.build_dataloader(
        Subset(dataset.molecular_dataset, indices),
        batch_size=len(indices), num_workers=0, shuffle=False,
    )
    batch = next(iter(loader))
    bmg, V_d, X_d = batch[0], batch[1], batch[2]
    t1 = time.time()
    with torch.inference_mode():
        first = model(bmg, V_d, X_d).detach().cpu().numpy()
        second = model(bmg, V_d, X_d).detach().cpu().numpy()
    forward_seconds = time.time() - t1
    stats = check_native_outputs(
        first, len(ordered_ids), len(expected["enzyme_order"]), pins["smoke"]["plausible_abs_max"]
    )
    checks.record("forward_shape_finite_sane", True, stats)
    checks.record("forward_repeatable_in_eval_mode", bool(np.array_equal(first, second)), None)

    result = {
        "status": "PASS" if checks.passed else "FAIL",
        "script": "cyp/scripts/screen_head_smoke_cyp.py",
        "script_sha256": sha256_file(__file__),
        "pins_sha256": args.pins_sha256,
        "checkpoint_sha256": run_files["checkpoint"]["sha256"],
        "selected_epoch_zero_based": selected_record["epoch"],
        "output_columns": list(dataset.task_names),
        "cyp3a4_index": cyp3a4_index,
        "units": "native log2fc (identity output transform; no target scaler)",
        "molecules": ordered_ids,
        "partitions": {name: "train" for name in chosen["train"]} | {name: "validation" for name in chosen["validation"]},
        "predictions": [[float(v) for v in row] for row in first],
        "checks": checks.items,
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "chemprop": chemprop.__version__,
        },
        "host": platform.node(),
        "device": "cpu",
        "threads": args.threads,
        "timing_seconds": {
            "build_dataset_and_model": build_seconds,
            "two_forward_passes": forward_seconds,
            "total": time.time() - started,
            "note": "single CPU smoke timing; not a benchmark and not a production estimate",
        },
        "labels_used": False,
        "scored": False,
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pins", required=True, type=Path)
    parser.add_argument("--pins-sha256", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=int)
    args = parser.parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.out_dir}")
    result = run(args)
    out_dir = new_output_dir(args.out_dir)
    with open(out_dir / "result.json", "x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "result": str(out_dir / "result.json")}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
