"""Targeted comparison of screen_predict.py against the saved PXR loading-smoke predictions.

Authorizes no run by itself. This is a check of CHANGED code only (the PXR loading path in
screen_predict.py differs from the loading smoke: the scaler constants are pinned instead of
recomputed from training frames, and the model is built without reading frames). It is not a
score reproduction and it uses no labels.

Two read-only subcommands; every output goes to a new path:

  build-reference-input
      Reads the three PXR loading-smoke result.json files (their 16 molecule ids and saved
      predictions) and looks up those 16 ids in the staged PXR training frame to write a small
      id,SMILES input file. Project-data reads, named here: (a) the three smoke result.json files;
      (b) pxr/artifacts/pxr002_aux/pxr002_train_frame.csv in the staged PXR-004 package (hash
      checked), read only to look up the SMILES of those 16 ids. No other row is written.

  compare
      Compares screen_predict.py's per-seed PXR 8.25 uM columns on that input against the saved
      smoke predictions (output index 0) by id, and checks that the reported mean equals the
      arithmetic mean of the three seeds. PASS requires every difference within --tolerance.

Usage (CoreHPC, run by the project owner): see the commands given with the plan.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SEEDS = (20261125, 20261126, 20261127)
SEED_COLUMNS = {seed: f"PXR_screen_8p25uM_log2fc_seed{seed}" for seed in SEEDS}
MEAN_COLUMN = "PXR_screen_8p25uM_log2fc_mean3"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_smoke_results(results_dir) -> dict:
    """seed -> {molecules, column0}; every run must be PASS and share the same molecule ids."""
    results_dir = Path(results_dir)
    loaded = {}
    for seed in SEEDS:
        path = results_dir / f"pxr_seed{seed}" / "result.json"
        result = json.loads(path.read_text())
        if result.get("status") != "PASS" or result.get("seed") != seed:
            raise ValueError(f"Smoke result is not a PASS for seed {seed}: {path}")
        molecules, predictions = result["molecules"], result["predictions"]
        if len(molecules) != len(predictions) or not molecules:
            raise ValueError(f"Smoke result has mismatched molecules and predictions: {path}")
        loaded[seed] = {"molecules": list(molecules), "column0": [float(row[0]) for row in predictions]}
    reference = loaded[SEEDS[0]]["molecules"]
    for seed in SEEDS[1:]:
        if loaded[seed]["molecules"] != reference:
            raise ValueError("The three smoke results do not share the same molecule ids.")
    if len(set(reference)) != len(reference):
        raise ValueError("Smoke molecule ids are not unique.")
    return loaded


def lookup_smiles(frame_path, ids) -> dict:
    """id -> SMILES for exactly the requested ids, from a csv with row_id and SMILES columns."""
    wanted = set(ids)
    found = {}
    with open(frame_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "row_id" not in reader.fieldnames or "SMILES" not in reader.fieldnames:
            raise ValueError("Frame must have row_id and SMILES columns.")
        for row in reader:
            if row["row_id"] in wanted:
                if row["row_id"] in found:
                    raise ValueError(f"Duplicate row_id in frame: {row['row_id']}")
                found[row["row_id"]] = row["SMILES"]
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{len(missing)} requested ids are missing from the frame: {sorted(missing)[:5]}")
    return found


def build_reference_input(args) -> dict:
    if sha256_file(args.frame) != args.frame_sha256:
        raise ValueError("Frame hash differs from --frame-sha256.")
    if Path(args.out).exists():
        raise FileExistsError(f"Refusing to overwrite: {args.out}")
    smoke = load_smoke_results(args.smoke_results_dir)
    ids = smoke[SEEDS[0]]["molecules"]
    smiles = lookup_smiles(args.frame, ids)
    with open(args.out, "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["row_id", "SMILES"])
        for name in ids:
            writer.writerow([name, smiles[name]])
    return {"rows": len(ids), "out": str(args.out), "out_sha256": sha256_file(args.out)}


def read_predictions(path) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {}
    for row in rows:
        if row["input_id"] in by_id:
            raise ValueError(f"Duplicate input_id in predictions: {row['input_id']}")
        by_id[row["input_id"]] = row
    return by_id


def compare(predictions_path, smoke_results_dir, tolerance: float) -> dict:
    if not (tolerance > 0 and math.isfinite(tolerance)):
        raise ValueError("--tolerance must be a positive finite number.")
    smoke = load_smoke_results(smoke_results_dir)
    ids = smoke[SEEDS[0]]["molecules"]
    predicted = read_predictions(predictions_path)
    if set(predicted) != set(ids):
        raise ValueError("Prediction ids differ from the smoke molecule ids.")
    per_seed, mean_diff = {}, 0.0
    for seed in SEEDS:
        diffs = []
        for name, saved in zip(ids, smoke[seed]["column0"]):
            row = predicted[name]
            if row["status"] != "ok" or row[SEED_COLUMNS[seed]] == "":
                raise ValueError(f"Row {name} is not ok or has no seed {seed} prediction.")
            diffs.append(abs(float(row[SEED_COLUMNS[seed]]) - saved))
        per_seed[str(seed)] = {"max_abs_difference": max(diffs), "n": len(diffs)}
    for name in ids:
        row = predicted[name]
        values = [float(row[SEED_COLUMNS[seed]]) for seed in SEEDS]
        mean_diff = max(mean_diff, abs(float(row[MEAN_COLUMN]) - math.fsum(values) / 3.0))
    passed = all(v["max_abs_difference"] <= tolerance for v in per_seed.values()) and mean_diff <= 1e-12
    return {
        "status": "PASS" if passed else "FAIL",
        "tolerance_absolute": tolerance,
        "per_seed": per_seed,
        "mean3_vs_arithmetic_mean_max_difference": mean_diff,
        "n_molecules": len(ids),
        "predictions_sha256": sha256_file(predictions_path),
        "labels_used": False,
        "scored": False,
        "scope": "targeted check of the changed PXR loading path against the saved 16-molecule smoke predictions",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-reference-input")
    b.add_argument("--smoke-results-dir", required=True, type=Path)
    b.add_argument("--frame", required=True, type=Path)
    b.add_argument("--frame-sha256", required=True)
    b.add_argument("--out", required=True, type=Path)
    c = sub.add_parser("compare")
    c.add_argument("--predictions", required=True, type=Path)
    c.add_argument("--smoke-results-dir", required=True, type=Path)
    c.add_argument("--tolerance", required=True, type=float)
    c.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "build-reference-input":
        print(json.dumps(build_reference_input(args), indent=2))
        return 0
    if args.out_json.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.out_json}")
    result = compare(args.predictions, args.smoke_results_dir, args.tolerance)
    with open(args.out_json, "x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
