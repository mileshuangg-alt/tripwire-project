"""D013 target-adaptation counting step: exact support/target counts and protected exclusions.

Counting only. No model, no GPU, no featurisation, no adaptation, no score.

For every frozen D007 partition and every CYP endpoint it reports the exact
support count (the cell's training molecules that carry an observed label for
that endpoint) and target count (its validation molecules likewise), and it
recomputes the protected-molecule exclusions rather than trusting the
manifests' own counts. Any protected molecule, any train/validation overlap or
any disagreement with a manifest's recorded totals aborts the run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

CONFIG_ARTIFACT = "D013TargetQualificationConfigV1"
MISSING = {"", "NA", "NaN", "nan", "None"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_config(path: Path, expected_sha: str) -> dict:
    if sha256(path) != expected_sha:
        raise SystemExit("qualification config SHA-256 mismatch")
    cfg = json.loads(path.read_text())
    if cfg.get("artifact_type") != CONFIG_ARTIFACT:
        raise SystemExit(f"not a {CONFIG_ARTIFACT}")
    return cfg


def read_table(path: Path, key: str = "Molecule_Name") -> dict:
    with path.open(newline="") as f:
        rows = {}
        for r in csv.DictReader(f):
            name = r[key]
            if name in rows:
                raise SystemExit(f"duplicate {key} in {path}: {name}")
            rows[name] = r
    if not rows:
        raise SystemExit(f"empty table: {path}")
    return rows


def labelled(labels: dict, column: str) -> set:
    """Molecules with an observed value in `column` (no imputation)."""
    out = set()
    for name, row in labels.items():
        if column not in row:
            raise SystemExit(f"label table has no column {column}")
        if row[column].strip() not in MISSING:
            out.add(name)
    return out


def verify_manifests(manifest_dir: Path, freeze_record: Path) -> dict:
    record = json.loads(freeze_record.read_text())["files"]
    manifests = {}
    for name, digest in sorted(record.items()):
        path = manifest_dir / name
        if not path.is_file():
            raise SystemExit(f"manifest missing: {path}")
        if sha256(path) != digest:
            raise SystemExit(f"manifest SHA-256 mismatch against FREEZE_RECORD: {name}")
        manifests[name[:-5] if name.endswith(".json") else name] = json.loads(path.read_text())
    if not manifests:
        raise SystemExit("FREEZE_RECORD lists no manifests")
    return manifests


def check_cell(cell: str, m: dict, protected: set, blinded: set) -> tuple:
    train = list(m["training_names"])
    val = list(m["validation_names"])
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise SystemExit(f"{cell}: duplicate molecule names")
    if len(train) != m["training_count"] or len(val) != m["validation_count"]:
        raise SystemExit(f"{cell}: name list disagrees with recorded count")
    overlap = set(train) & set(val)
    if overlap:
        raise SystemExit(f"{cell}: {len(overlap)} molecules in both training and validation")
    both = set(train) | set(val)
    hit_protected = sorted(both & protected)
    hit_blinded = sorted(both & blinded)
    if hit_protected or hit_blinded:
        raise SystemExit(f"{cell}: protected molecules present: "
                         f"internal_test={len(hit_protected)} blinded={len(hit_blinded)} "
                         f"examples={(hit_protected + hit_blinded)[:5]}")
    return train, val


def count(manifests: dict, labels: dict, cfg: dict, protected: set, blinded: set) -> dict:
    cells = {}
    totals = {e["name"]: {"support": 0, "target": 0} for e in cfg["endpoints"]}
    labelled_sets = {e["name"]: labelled(labels, e["label"]) for e in cfg["endpoints"]}
    for cell in sorted(manifests):
        m = manifests[cell]
        train, val = check_cell(cell, m, protected, blinded)
        unknown = sorted((set(train) | set(val)) - set(labels))
        if unknown:
            raise SystemExit(f"{cell}: {len(unknown)} molecules absent from the label table, "
                             f"examples={unknown[:5]}")
        per_endpoint = {}
        for e in cfg["endpoints"]:
            lab = labelled_sets[e["name"]]
            sup = sum(1 for n in train if n in lab)
            tgt = sum(1 for n in val if n in lab)
            if sup == 0 or tgt == 0:
                raise SystemExit(f"{cell}/{e['name']}: empty support or target set")
            per_endpoint[e["name"]] = {"support": sup, "target": tgt}
            totals[e["name"]]["support"] += sup
            totals[e["name"]]["target"] += tgt
        cells[cell] = {
            "training_count": m["training_count"],
            "validation_count": m["validation_count"],
            "training_seeds": m["training_seeds"],
            "endpoints": per_endpoint,
        }
    largest = max(totals, key=lambda n: totals[n]["support"])
    return {"cells": cells, "totals_by_endpoint": totals, "largest_endpoint_by_support": largest}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--freeze-record", type=Path, required=True)
    p.add_argument("--label-table", type=Path, required=True)
    p.add_argument("--label-table-sha256", required=True)
    p.add_argument("--split-assignments", type=Path, required=True)
    p.add_argument("--split-assignments-sha256", required=True)
    p.add_argument("--blinded", type=Path, required=True)
    p.add_argument("--blinded-sha256", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)

    cfg = read_config(a.config, a.config_sha256)
    for path, expected in ((a.label_table, a.label_table_sha256),
                           (a.split_assignments, a.split_assignments_sha256),
                           (a.blinded, a.blinded_sha256)):
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"input missing or SHA-256 mismatch: {path}")
    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")

    labels = read_table(a.label_table)
    assignments = read_table(a.split_assignments)
    blinded = set(read_table(a.blinded))
    protected = {n for n, r in assignments.items() if r["split"] in set(cfg["protected_splits"])}
    if not protected:
        raise SystemExit("no protected molecules found; refusing to run a vacuous exclusion check")
    manifests = verify_manifests(a.manifest_dir, a.freeze_record)

    result = count(manifests, labels, cfg, protected, blinded)
    summary = {
        "artifact_type": "D013_target_cell_counts_v1",
        "partitions": len(manifests),
        "endpoints": [e["name"] for e in cfg["endpoints"]],
        "protected_internal_test_molecules": len(protected),
        "blinded_molecules": len(blinded),
        "protected_molecules_in_any_cell": 0,
        "inputs": {
            "config_sha256": a.config_sha256,
            "label_table_sha256": a.label_table_sha256,
            "split_assignments_sha256": a.split_assignments_sha256,
            "blinded_sha256": a.blinded_sha256,
            "freeze_record_sha256": sha256(a.freeze_record),
            "script_sha256": sha256(Path(__file__)),
        },
        **result,
    }
    a.out_dir.mkdir(parents=True)
    out = a.out_dir / "cell_counts.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (a.out_dir / "SHA256SUMS.txt").write_text(f"{sha256(out)}  {out.name}\n")
    brief = {k: summary[k] for k in ("partitions", "endpoints", "totals_by_endpoint",
                                     "largest_endpoint_by_support",
                                     "protected_molecules_in_any_cell")}
    print(json.dumps(brief, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
