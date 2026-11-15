"""D013 firewalled ActFound ChEMBL32 corpus, graph route C.

Runs on CoreHPC in the pinned ActFound environment (Python 3.9, RDKit
2023.03.1), with the exact graph-stage code that built the rebuilt corpus.

Route C:
  1. Assay universe = the authors' released chembl_processed_chembl32.csv
     (assay key = assay_id_units_type, as upstream chembl_preprocess.py).
  2. Input rows = the rebuilt PRE-graph population (chembl32_actfound_pregraph.tsv)
     restricted to those assays. Upstream's graph gate needs the pre-graph assay
     size N, which the authors' post-graph file no longer contains.
  3. Remove every row whose molregno is on the frozen D013 firewall list.
  4. Rerun the unchanged graph stage (actfound_graph_stage.process_shard, which
     reproduces chembl_preprocess.py lines 103-174) on what is left.
  The project loader then applies its own gates (>=20 usable rows etc.) at
  split-reconciliation and training time.

Reproduction mode (no firewall list): the same pipeline must reproduce the
authors' file. Every output row is compared field by field with the authors'
row for the same activity_id; all differences are counted and listed, never
resolved silently. The comparison is exact: there is no tolerance and no
normalization, so a formatting-only difference is a difference.

Stop rule (D013 Amendment 1, proposed): reproduction mode requires the
hash-pinned upstream split. A difference of any kind (field, missing or extra
row, assay-key mismatch) in an assay whose upstream split ID is in train or
valid makes the gate FAIL: outputs are kept for review and the exit status is
nonzero. Differences outside those splits are reported and do not block.
Firewall mode refuses to run without a PASS reproduction manifest produced by
this same script on the same inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

GRAPH_CODE = {
    "actfound_graph_stage.py": "1783c5fe8f237261e7f5a246910a8f2bad1cf990a8a6ac11a07af04b645a9995",
    "actfound_chembl32_projection.py": "22fca4084849fd58e791f80c94d24e9b282e1288900a37ddbf31c0872dcf8f3b",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def row_digest(fields) -> bytes:
    return hashlib.blake2b("\t".join(fields).encode(), digest_size=16).digest()


REPRODUCTION_ARTIFACT = "D013_route_c_reproduction_manifest_v1"
EXIT_GATE_FAIL = 3


def split_descriptor(fields) -> str:
    """Upstream split ID of a 15-field row: assay_chembl_id_units_type, '/' -> '_'
    (as actfound_model_side._upstream_assay_descriptor)."""
    return f"{fields[11]}_{fields[7]}_{fields[8]}".replace("/", "_")


def load_split(path: Path) -> set:
    s = json.loads(path.read_text())
    train, valid = s.get("train"), s.get("valid")
    if not isinstance(train, list) or not isinstance(valid, list) or not train or not valid:
        raise SystemExit("upstream split must have non-empty train and valid lists")
    if set(train) & set(valid):
        raise SystemExit("upstream split train/valid overlap")
    return set(train) | set(valid)


def authors_index(path: Path):
    """assay_key set and activity_id -> (assay_key, split descriptor, digest of the 15 fields)."""
    keys, rows = set(), {}
    with path.open(newline="") as f:
        for r in csv.reader(f):
            if len(r) != 15:
                raise SystemExit(f"authors row with {len(r)} fields")
            k = f"{r[1]}_{r[7]}_{r[8]}"
            keys.add(k)
            if r[0] in rows:
                raise SystemExit(f"duplicate authors activity_id {r[0]}")
            rows[r[0]] = (k, split_descriptor(r), row_digest(r))
    return keys, rows


EXCLUSION_BASES = ("full_standard_inchikey", "largest_fragment_full_inchikey")


def read_firewall(path: Path) -> dict:
    """molregno -> exclusion_basis from the D013 pretraining exclusion list v2 (fail closed on anything else)."""
    out = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if "exclusion_basis" not in (r.fieldnames or []):
            raise SystemExit("exclusion list has no exclusion_basis column (not the D013 v2 list)")
        for row in r:
            if row["exclusion_basis"] not in EXCLUSION_BASES:
                raise SystemExit(f"unknown exclusion_basis: {row['exclusion_basis']}")
            if row["molregno"] in out:
                raise SystemExit(f"duplicate molregno in exclusion list: {row['molregno']}")
            out[row["molregno"]] = row["exclusion_basis"]
    return out


def filter_pregraph(pregraph: Path, out: Path, keys: set, firewall: set, audit_columns) -> dict:
    kept = removed = outside = 0
    removed_by_assay = Counter()
    removed_by_basis = Counter()
    with pregraph.open(newline="") as f, out.open("w", newline="") as o:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        if header != list(audit_columns):
            raise SystemExit("unexpected pregraph header")
        w = csv.writer(o, delimiter="\t")  # same dialect as the graph stage's shard writer
        w.writerow(header)
        for row in r:
            k = f"{row[1]}_{row[7]}_{row[8]}"
            if k not in keys:
                outside += 1
                continue
            if row[3] in firewall:
                removed += 1
                removed_by_assay[k] += 1
                removed_by_basis[firewall[row[3]]] += 1
                continue
            w.writerow(row)
            kept += 1
    return {"rows_kept": kept, "rows_removed_by_firewall": removed, "rows_outside_authors_assays": outside,
            "assays_with_firewall_removals": len(removed_by_assay),
            "rows_removed_by_exclusion_basis": dict(sorted(removed_by_basis.items()))}


def compare_to_authors(corpus: Path, authors_rows: dict, split_ids: set, diff_path: Path) -> dict:
    """Exact row comparison, every difference scoped in_split / outside_split.

    A difference is in_split if the rebuilt row's or the authors' row's split
    descriptor is an upstream train/valid ID.
    """
    seen = set()
    c = {"in_split": Counter(), "outside_split": Counter()}
    present = set()
    with corpus.open(newline="") as f, diff_path.open("w", newline="") as d:
        r = csv.reader(f, delimiter="\t")
        next(r)
        w = csv.writer(d, delimiter="\t", lineterminator="\n")
        w.writerow(["activity_id", "assay_key", "split_descriptor", "scope", "kind"])

        def diff(aid, key, descs, kind):
            scope = "in_split" if any(x in split_ids for x in descs) else "outside_split"
            c[scope][kind] += 1
            w.writerow([aid, key, ";".join(sorted(set(descs))), scope, kind])

        for row in r:
            fields, key = row[1:], row[0]
            aid, desc = fields[0], split_descriptor(fields)
            present.add(desc)
            seen.add(aid)
            ref = authors_rows.get(aid)
            if ref is None:
                diff(aid, key, [desc], "extra_in_rebuilt")
            elif ref[0] != key or ref[1] != desc:
                diff(aid, key, [desc, ref[1]], "assay_key_mismatch")
            elif ref[2] != row_digest(fields):
                diff(aid, key, [desc], "field_mismatch")
            else:
                scope = "in_split" if desc in split_ids else "outside_split"
                c[scope]["identical_rows"] += 1
        for aid, (key, desc, _) in authors_rows.items():
            if aid not in seen:
                diff(aid, key, [desc], "missing_in_rebuilt")
    authors_descs = {v[1] for v in authors_rows.values()}
    blocking = sum(n for k, n in c["in_split"].items() if k != "identical_rows")
    return {
        "in_split": dict(c["in_split"]),
        "outside_split": dict(c["outside_split"]),
        "split_ids_absent_from_authors_file": len(split_ids - authors_descs),
        "split_ids_absent_from_rebuilt_corpus": len(split_ids - present),
        "blocking_differences": blocking,
        "gate": "PASS" if blocking == 0 else "FAIL",
        "comparison": "exact; no tolerance, no normalization",
    }


def check_reproduction_manifest(path: Path, expected_sha: str, current_inputs: dict) -> dict:
    if sha256(path) != expected_sha:
        raise SystemExit("reproduction manifest SHA-256 mismatch")
    m = json.loads(path.read_text())
    if m.get("artifact_type") != REPRODUCTION_ARTIFACT:
        raise SystemExit("not a route-C reproduction manifest")
    if m.get("reproduction_vs_authors", {}).get("gate") != "PASS":
        raise SystemExit("reproduction gate did not PASS; firewalled build blocked")
    for k, v in current_inputs.items():
        if m.get("inputs", {}).get(k) != v:
            raise SystemExit(f"reproduction manifest was made with a different {k}")
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pregraph", type=Path, required=True)
    p.add_argument("--pregraph-sha256", required=True)
    p.add_argument("--authors-csv", type=Path, required=True)
    p.add_argument("--authors-csv-sha256", required=True)
    p.add_argument("--graph-code-dir", type=Path, required=True)
    p.add_argument("--firewall-list", type=Path, help="omit for reproduction mode")
    p.add_argument("--firewall-list-sha256")
    p.add_argument("--upstream-split", type=Path, help="required in reproduction mode")
    p.add_argument("--upstream-split-sha256")
    p.add_argument("--reproduction-manifest", type=Path, help="required in firewall mode: a PASS reproduction manifest")
    p.add_argument("--reproduction-manifest-sha256")
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)

    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")
    for name, expected in GRAPH_CODE.items():
        if sha256(a.graph_code_dir / name) != expected:
            raise SystemExit(f"graph code differs from the code that built the rebuilt corpus: {name}")
    for path, expected in ((a.pregraph, a.pregraph_sha256), (a.authors_csv, a.authors_csv_sha256)):
        if sha256(path) != expected:
            raise SystemExit(f"input SHA-256 mismatch: {path}")
    reproduction = a.firewall_list is None
    inputs = {"pregraph_sha256": a.pregraph_sha256, "authors_csv_sha256": a.authors_csv_sha256,
              "graph_code_sha256": GRAPH_CODE, "script_sha256": sha256(Path(__file__))}
    split_ids = None
    if reproduction:
        if a.upstream_split is None or not a.upstream_split_sha256:
            raise SystemExit("reproduction mode requires --upstream-split and --upstream-split-sha256")
        if not a.upstream_split.is_file() or sha256(a.upstream_split) != a.upstream_split_sha256:
            raise SystemExit("upstream split missing or SHA-256 mismatch")
        split_ids = load_split(a.upstream_split)
    else:
        if sha256(a.firewall_list) != a.firewall_list_sha256:
            raise SystemExit("firewall list SHA-256 mismatch")
        if a.reproduction_manifest is None or not a.reproduction_manifest_sha256:
            raise SystemExit("firewall mode requires a PASS reproduction manifest and its SHA-256")
        rep_m = check_reproduction_manifest(a.reproduction_manifest, a.reproduction_manifest_sha256, inputs)

    sys.path.insert(0, str(a.graph_code_dir))
    import actfound_graph_stage as gs
    from actfound_chembl32_projection import AUDIT_COLUMNS

    started = time.time()
    a.out_dir.mkdir(parents=True)
    keys, authors_rows = authors_index(a.authors_csv)
    firewall = {} if reproduction else read_firewall(a.firewall_list)
    filtered = a.out_dir / "pregraph_authors_assays_filtered.tsv"
    filt = filter_pregraph(a.pregraph, filtered, keys, firewall, AUDIT_COLUMNS)
    corpus = a.out_dir / ("reproduction_corpus.tsv" if reproduction else "actfound_chembl32_d013_firewalled.tsv")
    graph = gs.process_shard(filtered, corpus, a.out_dir / "graph_stage_summary.json", 5000)

    manifest = {
        "artifact_type": "D013_firewalled_corpus_manifest_v1" if not reproduction else REPRODUCTION_ARTIFACT,
        "route": "C: authors' assay universe; rebuilt pre-graph rows; firewall; unchanged graph stage",
        "openadmet_firewall_applied": (not reproduction) and len(firewall) > 0,
        "firewall_list_path": None if reproduction else str(a.firewall_list),
        "firewall_list_sha256": None if reproduction else a.firewall_list_sha256,
        "firewall_molregnos": len(firewall),
        "firewall_molregnos_by_exclusion_basis": dict(sorted(Counter(firewall.values()).items())),
        "inputs": dict(inputs, pregraph=str(a.pregraph), authors_csv=str(a.authors_csv),
                       upstream_split=None if not reproduction else str(a.upstream_split),
                       upstream_split_sha256=None if not reproduction else a.upstream_split_sha256),
        "reproduction_manifest": None if reproduction else {"path": str(a.reproduction_manifest),
                                                            "sha256": a.reproduction_manifest_sha256,
                                                            "upstream_split_sha256": rep_m["inputs"]["upstream_split_sha256"]},
        "authors_assays": len(keys),
        "filter": filt,
        "graph_stage": {k: graph[k] for k in ("processed_assays", "processed_rows", "graph_size_rejected_assays",
                                              "graph_valid_count_rejected_assays", "graph_qualified_assays", "final_rows")},
        "output_path": str(corpus),
        "output_sha256": sha256(corpus),
        "runtime": {"python": platform.python_version(), "seconds": round(time.time() - started, 1)},
    }
    if reproduction:
        manifest["reproduction_vs_authors"] = compare_to_authors(corpus, authors_rows, split_ids,
                                                                 a.out_dir / "reproduction_differences.tsv")
    (a.out_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    names = [corpus.name, "corpus_manifest.json", "graph_stage_summary.json"] + (["reproduction_differences.tsv"] if reproduction else [])
    (a.out_dir / "SHA256SUMS.txt").write_text("".join(f"{sha256(a.out_dir / n)}  {n}\n" for n in names))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if reproduction and manifest["reproduction_vs_authors"]["gate"] != "PASS":
        print("REPRODUCTION GATE FAIL: differences inside the upstream train/valid splits; "
              "see reproduction_differences.tsv. Do not continue.", file=sys.stderr)
        return EXIT_GATE_FAIL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
