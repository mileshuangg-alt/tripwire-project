"""D013 firewalled ActFound ChEMBL32 foundation corpus: convert, compare, reconcile.

Runs on CoreHPC in the pinned ActFound environment (Python 3.9.18). Graph
qualification itself is NOT reimplemented here: it is the existing
cyp/scripts/actfound_graph_stage.py `process-shard` (reproducing upstream
chembl_preprocess.py lines 103-174), run on this script's firewalled output.

Subcommands
  convert    authors' chembl_processed_chembl32.csv  -> pregraph-format TSVs
             (unfirewalled and firewalled), sorted by the frozen external-sort key
  compare    Q9: value-by-value comparison of the unfirewalled conversion with
             the rebuilt corpus (job 3066823 input) on shared activity IDs
  reconcile  Q1: loader-exact split after firewall attrition, reconciliation
             artifacts in the runner's format, and config v2

Every subcommand verifies its inputs' SHA-256, refuses to overwrite outputs,
and writes a summary JSON with input/output hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional

AUDIT_COLUMNS = (
    "activity_id", "assay_id", "record_id_unused", "molregno", "bao_endpoint",
    "standard_relation", "standard_value", "standard_units", "standard_type",
    "assay_type", "bao_format", "assay_chembl_id", "count_scratch", "canonical_smiles",
)
COMPARED_FIELDS = tuple(i for i in range(14) if i not in (0, 12))  # activity_id is the key; count_scratch is recomputed by the graph stage
CLASSIFICATION = "exact_descriptor_present_after_d013_firewall"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def fresh(path: Path) -> Path:
    if path.exists():
        raise SystemExit(f"refusing to overwrite: {path}")
    return path


def order_key(row: list) -> tuple:
    # Identical to actfound_graph_stage.iter_assays' frozen external-sort key.
    return (int(row[1]), row[7], row[8], int(row[3]), int(row[0]))


def descriptor(assay_chembl_id: str, units: str, stype: str) -> str:
    # Identical to actfound_model_side._upstream_assay_descriptor.
    return "{}_{}_{}".format(assay_chembl_id, units, stype).replace("/", "_")


def write_summary(path: Path, payload: dict) -> None:
    payload.setdefault("runtime", {})["python"] = platform.python_version()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------- convert
def read_authors_rows(zip_path: Path, member: str, member_sha256: str) -> list:
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(member)
    actual = hashlib.sha256(data).hexdigest()
    if actual != member_sha256:
        raise SystemExit(f"authors CSV SHA-256 mismatch: {actual} != {member_sha256}")
    rows = []
    for n, row in enumerate(csv.reader(io.StringIO(data.decode("utf-8"))), start=1):
        if len(row) != 15:
            raise SystemExit(f"authors CSV line {n}: expected 15 fields, got {len(row)}")
        rows.append(row[:14])  # 13 raw fields + canonical SMILES; drop the old pair count
    return rows


def load_prohibited(path: Path) -> set:
    with path.open(newline="") as f:
        return {int(r["molregno"]) for r in csv.DictReader(f, delimiter="\t")}


def write_pregraph(path: Path, rows: Iterable[list]) -> int:
    n = 0
    with fresh(path).open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(AUDIT_COLUMNS)
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def convert(a: argparse.Namespace) -> int:
    started = time.time()
    require_sha(a.authors_zip, a.authors_zip_sha256)
    require_sha(a.prohibited, a.prohibited_sha256)
    fresh(a.out_dir).mkdir(parents=True)
    rows = read_authors_rows(a.authors_zip, a.member, a.member_sha256)
    rows.sort(key=order_key)
    keys = [order_key(r) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate sort keys in authors' corpus")
    prohibited = load_prohibited(a.prohibited)

    removed_by_assay: dict = {}
    kept = []
    for r in rows:
        if int(r[3]) in prohibited:
            k = "_".join((r[1], r[7], r[8]))
            removed_by_assay[k] = removed_by_assay.get(k, 0) + 1
        else:
            kept.append(r)
    unf = a.out_dir / "authors_pregraph_unfirewalled.tsv"
    fw = a.out_dir / "authors_pregraph_firewalled.tsv"
    n_unf = write_pregraph(unf, rows)
    n_fw = write_pregraph(fw, kept)
    with fresh(a.out_dir / "firewall_removed_rows_by_assay.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["assay_key", "removed_rows"])
        for k in sorted(removed_by_assay):
            w.writerow([k, removed_by_assay[k]])
    removed_mols = {int(r[3]) for r in rows} & prohibited
    write_summary(a.out_dir / "convert_summary.json", {
        "artifact_type": "D013_authors_corpus_conversion_v1",
        "inputs": {"authors_zip": str(a.authors_zip), "authors_zip_sha256": a.authors_zip_sha256,
                   "member": a.member, "member_sha256": a.member_sha256,
                   "prohibited": str(a.prohibited), "prohibited_sha256": a.prohibited_sha256,
                   "script_sha256": file_sha256(Path(__file__))},
        "authors_rows": n_unf, "firewalled_rows": n_fw, "removed_rows": n_unf - n_fw,
        "prohibited_molregnos_listed": len(prohibited), "prohibited_molregnos_present": len(removed_mols),
        "assays_with_removed_rows": len(removed_by_assay),
        "outputs": {p.name: file_sha256(p) for p in (unf, fw, a.out_dir / "firewall_removed_rows_by_assay.tsv")},
        "runtime": {"seconds": round(time.time() - started, 1)},
    })
    return 0


# ---------------------------------------------------------------- compare
def compare(a: argparse.Namespace) -> int:
    started = time.time()
    require_sha(a.converted, a.converted_sha256)
    require_sha(a.rebuilt_tsv, a.rebuilt_sha256)
    fresh(a.out_dir).mkdir(parents=True)
    rebuilt = {}
    with a.rebuilt_tsv.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if len(header) != 16 or header[0] != "assay_key":
            raise SystemExit("unexpected rebuilt TSV header")
        for row in reader:
            raw = row[1:15]  # drop assay_key and graph_pair_count
            rebuilt[int(raw[0])] = "\t".join(raw[i] for i in COMPARED_FIELDS)
    mismatch = {AUDIT_COLUMNS[i]: 0 for i in COMPARED_FIELDS}
    missing = 0
    shared = 0
    examples = []
    with a.converted.open(newline="") as f, fresh(a.out_dir / "differences.tsv").open("w", newline="") as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        w.writerow(["activity_id", "field", "authors_value", "rebuilt_value"])
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            other = rebuilt.get(int(row[0]))
            if other is None:
                missing += 1
                w.writerow([row[0], "<missing_in_rebuilt>", "", ""])
                continue
            shared += 1
            vals = other.split("\t")
            for j, i in enumerate(COMPARED_FIELDS):
                if row[i] != vals[j]:
                    mismatch[AUDIT_COLUMNS[i]] += 1
                    w.writerow([row[0], AUDIT_COLUMNS[i], row[i], vals[j]])
                    if len(examples) < 20:
                        examples.append([row[0], AUDIT_COLUMNS[i], row[i], vals[j]])
    write_summary(a.out_dir / "compare_summary.json", {
        "artifact_type": "D013_authors_vs_rebuilt_value_comparison_v1",
        "authority": "report_only",
        "inputs": {"converted": str(a.converted), "converted_sha256": a.converted_sha256,
                   "rebuilt_tsv": str(a.rebuilt_tsv), "rebuilt_sha256": a.rebuilt_sha256,
                   "script_sha256": file_sha256(Path(__file__))},
        "shared_activity_ids": shared, "authors_activity_ids_missing_in_rebuilt": missing,
        "mismatch_counts_by_field": mismatch, "examples": examples,
        "differences_sha256": file_sha256(a.out_dir / "differences.tsv"),
        "runtime": {"seconds": round(time.time() - started, 1)},
    })
    return 0


# ---------------------------------------------------------------- reconcile
def reconcile(a: argparse.Namespace) -> int:
    started = time.time()
    require_sha(a.graph_tsv, a.graph_sha256)
    require_sha(a.split_json, a.split_sha256)
    require_sha(a.base_config, a.base_config_sha256)
    fresh(a.out_dir).mkdir(parents=True)
    fresh(a.config_out)

    sys.path.insert(0, str(a.repo_root / "cyp" / "scripts"))
    import actfound_model_side as ms  # pinned env: loader gates exactly as training applies them

    emitted = [x.assay_id for x in ms.iter_frozen_graph_qualified_assays(a.graph_tsv)]
    if len(set(emitted)) != len(emitted):
        raise SystemExit("loader emitted a descriptor more than once")
    emitted_set = set(emitted)
    split = json.loads(a.split_json.read_text())
    parts = {"train": split["train"], "validation": split["valid"]}
    kept = {p: [i for i in ids if i in emitted_set] for p, ids in parts.items()}
    dropped = {p: [i for i in ids if i not in emitted_set] for p, ids in parts.items()}
    if set(kept["train"]) & set(kept["validation"]):
        raise SystemExit("train/validation overlap after firewall")

    per_id = a.out_dir / "actfound_split_reconciliation_per_id_d013.tsv"
    with fresh(per_id).open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["upstream_split_id", "partition", "classification", "mapped_assay_key"])
        for p in ("train", "validation"):
            for i in kept[p]:
                w.writerow([i, p, CLASSIFICATION, i])
    drop_path = a.out_dir / "firewall_dropped_split_ids_d013.tsv"
    with fresh(drop_path).open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["upstream_split_id", "partition", "reason"])
        for p in ("train", "validation"):
            for i in dropped[p]:
                w.writerow([i, p, "not_emitted_by_loader_after_d013_firewall"])
    graph_sha = a.graph_sha256
    summary_path = a.out_dir / "actfound_split_reconciliation_summary_d013.json"
    summary = {
        "artifact_type": "ActFoundSplitReconciliationV1",
        "status": "complete",
        "method": "D013: pinned upstream split restricted to descriptors the project loader emits from the firewalled corpus; identity mapping, no remapping",
        "frozen_dataset_path": str(a.graph_tsv), "frozen_dataset_sha256": graph_sha,
        "upstream_split_path": str(a.split_json), "upstream_split_sha256": a.split_sha256,
        "per_id_path": str(per_id),
        "train_split_ids": len(kept["train"]), "validation_split_ids": len(kept["validation"]),
        "train_mapped": len(kept["train"]), "validation_mapped": len(kept["validation"]),
        "mapped_total": len(kept["train"]) + len(kept["validation"]),
        "unresolved_total": 0, "ambiguous_total": 0, "absent_total": 0,
        "train_validation_overlap_after_mapping": 0,
        "firewall_dropped": {p: len(v) for p, v in dropped.items()},
        "upstream_counts": {p: len(v) for p, v in parts.items()},
        "loader_emitted_assays": len(emitted),
        "created_at_unix": time.time(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    config = json.loads(a.base_config.read_text())
    config.update({
        "frozen_input_tsv": str(a.graph_tsv), "frozen_input_sha256": graph_sha,
        "frozen_dataset_path": str(a.graph_tsv), "frozen_dataset_sha256": graph_sha,
        "train_assays": kept["train"], "val_assays": kept["validation"], "validation_assays": kept["validation"],
        "split_reconciliation_per_id_path": str(per_id), "split_reconciliation_per_id_sha256": file_sha256(per_id),
        "split_reconciliation_summary_path": str(summary_path), "split_reconciliation_summary_sha256": file_sha256(summary_path),
        "output_dir": str(a.run_output_dir),
        "config_version": "d013_v2",
        "derived_from_config_sha256": a.base_config_sha256,
    })
    a.config_out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_summary(a.out_dir / "reconcile_run_summary.json", {
        "artifact_type": "D013_split_reconciliation_run_v1",
        "inputs": {"graph_tsv": str(a.graph_tsv), "graph_sha256": graph_sha, "split_json": str(a.split_json),
                   "split_sha256": a.split_sha256, "base_config": str(a.base_config),
                   "base_config_sha256": a.base_config_sha256,
                   "loader_module_sha256": file_sha256(a.repo_root / "cyp" / "scripts" / "actfound_model_side.py"),
                   "script_sha256": file_sha256(Path(__file__))},
        "kept": {p: len(v) for p, v in kept.items()}, "dropped": {p: len(v) for p, v in dropped.items()},
        "outputs": {p.name: file_sha256(p) for p in (per_id, drop_path, summary_path, a.config_out)},
        "config_out": str(a.config_out),
        "runtime": {"seconds": round(time.time() - started, 1)},
    })
    return 0


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--authors-zip", type=Path, required=True)
    c.add_argument("--authors-zip-sha256", required=True)
    c.add_argument("--member", default="chembl_processed_chembl32.csv")
    c.add_argument("--member-sha256", required=True)
    c.add_argument("--prohibited", type=Path, required=True)
    c.add_argument("--prohibited-sha256", required=True)
    c.add_argument("--out-dir", type=Path, required=True)
    m = sub.add_parser("compare")
    m.add_argument("--converted", type=Path, required=True)
    m.add_argument("--converted-sha256", required=True)
    m.add_argument("--rebuilt-tsv", type=Path, required=True)
    m.add_argument("--rebuilt-sha256", required=True)
    m.add_argument("--out-dir", type=Path, required=True)
    r = sub.add_parser("reconcile")
    r.add_argument("--graph-tsv", type=Path, required=True)
    r.add_argument("--graph-sha256", required=True)
    r.add_argument("--split-json", type=Path, required=True)
    r.add_argument("--split-sha256", required=True)
    r.add_argument("--base-config", type=Path, required=True)
    r.add_argument("--base-config-sha256", required=True)
    r.add_argument("--repo-root", type=Path, required=True)
    r.add_argument("--run-output-dir", type=Path, required=True)
    r.add_argument("--config-out", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)
    return {"convert": convert, "compare": compare, "reconcile": reconcile}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
