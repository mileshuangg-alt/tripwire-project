"""D013 Amendment 1 acceptance check of the built firewalled corpus.

"The built corpus must contain no row, in train or validation, with a full-key
or largest-fragment full-key match to a protected structure, verified by
recomputing identities from the built file." (D013 Amendment 1, Change 2)

Runs on the Mac in the frozen D012 identity environment (tripwire_cyp002:
Python 3.12.14, RDKit 2026.03.6, InChI 1.07.3) and refuses any other.
Identities are recomputed from each row's canonical_smiles with the same code
that built the exclusion list (identity_overlap.identity for the full Standard
InChIKey; build_firewall_list_v2.largest_fragment_key for the largest
fragment). Nothing is looked up by molregno except the report-only cross-check
against the exclusion list.

Gate (fail closed):
  PASS  no train or valid row has a full-key or largest-fragment full-key match,
        and every train or valid row's identity could be computed.
  Adjudicated identity failures: a train or valid row whose identity cannot be
        computed does not block only if it is listed, with the exact activity_id,
        molregno, canonical_smiles and status, in the hash-pinned adjudication
        file Miles decided (--adjudications). Adjudication never applies to a
        match. A listed adjudication that matches no row fails the gate, so a
        stale list cannot pass silently.
  Reported, never blocking: matches and identity failures in test,
        test_external and rows in no upstream split; first-block and
        largest-fragment first-block near matches (report-only under D013);
        exclusion-list molregnos present in the corpus.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "overlap_audit_20261115"))
import build_firewall_list_v2 as bf  # noqa: E402
import build_firewalled_corpus as bc  # noqa: E402
import identity_overlap as io  # noqa: E402

FROZEN_ENV = bf.FROZEN_ENV
SPLIT_PARTS = ("train", "valid", "test", "test_external")
BLOCKING_PARTS = ("train", "valid")
EXIT_GATE_FAIL = 3
SMILES_FIELD = 13   # within the 15 corpus fields (after assay_key)
MOLREGNO_FIELD = 3


def molecule_identity(smiles: str) -> tuple[str, str, str]:
    """(full key, largest-fragment full key, status), as build_firewall_list_v2 derives them."""
    _, key, status = io.identity(smiles)
    if status != "ok":
        return "", "", status
    if "." not in smiles:
        return key, key, "ok"
    fk = bf.largest_fragment_key(smiles)
    return (key, fk, "ok") if fk else (key, "", "largest_fragment_failure")


def protected_keys(audit: Path):
    """full, first-block, largest-fragment, largest-fragment first-block -> OpenADMET names,
    built exactly as build_firewall_list_v2.main builds them."""
    full, block, frag_full, frag_block = (defaultdict(set) for _ in range(4))
    tiers_of = {}
    with audit.open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["source_family"] != "openadmet":
                continue
            if r["identity_state"] != "represented":
                raise SystemExit(f"unrepresented protected structure: {r['source_id']}")
            name, key = r["source_id"], r["full_standard_inchikey"]
            tiers_of[name] = bf.TIER[r["source_partition"]]
            full[key].add(name)
            block[key[:14]].add(name)
            fk = bf.largest_fragment_key(r["source_smiles"]) if "." in r["source_smiles"] else key
            frag_full[fk].add(name)
            frag_block[fk[:14]].add(name)
    return full, block, frag_full, frag_block, tiers_of


def split_parts(path: Path) -> dict:
    s = json.loads(path.read_text())
    out = defaultdict(list)
    for part in SPLIT_PARTS:
        for x in s.get(part) or []:
            out[x].append(part)
    return out


def _identities(smiles_list, workers: int) -> dict:
    if workers <= 1:
        return {s: molecule_identity(s) for s in smiles_list}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return dict(zip(smiles_list, ex.map(molecule_identity, smiles_list, chunksize=2000)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--corpus-sha256", required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-sha256", required=True)
    p.add_argument("--upstream-split", type=Path, required=True)
    p.add_argument("--upstream-split-sha256", required=True)
    p.add_argument("--firewall-list", type=Path, required=True)
    p.add_argument("--firewall-list-sha256", required=True)
    p.add_argument("--adjudications", type=Path, help="hash-pinned identity-failure adjudications")
    p.add_argument("--adjudications-sha256")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = p.parse_args(argv)

    e = bf.env()
    if e != FROZEN_ENV:
        raise SystemExit(f"not the frozen identity environment: {e}")
    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")
    if (a.adjudications is None) != (a.adjudications_sha256 is None):
        raise SystemExit("--adjudications and --adjudications-sha256 go together")
    pinned = [(a.corpus, a.corpus_sha256), (a.audit, a.audit_sha256),
              (a.upstream_split, a.upstream_split_sha256), (a.firewall_list, a.firewall_list_sha256)]
    if a.adjudications is not None:
        pinned.append((a.adjudications, a.adjudications_sha256))
    for path, expected in pinned:
        if not path.is_file() or bf.sha256(path) != expected:
            raise SystemExit(f"input missing or SHA-256 mismatch: {path}")

    started = time.time()
    full, block, frag_full, frag_block, tiers_of = protected_keys(a.audit)
    parts_of = split_parts(a.upstream_split)
    excluded = set()
    with a.firewall_list.open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            excluded.add(r["molregno"])

    adjudicated = {}
    if a.adjudications is not None:
        with a.adjudications.open(newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                k = (r["activity_id"], r["molregno"], r["canonical_smiles"], r["status"])
                if not r.get("decision") or not r.get("evidence"):
                    raise SystemExit(f"adjudication without decision or evidence: {k}")
                adjudicated[k] = r["decision"]

    rows = []   # (assay_key, activity_id, molregno, smiles, scope)
    with a.corpus.open(newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            fields = row[1:]
            if len(fields) != 15:
                raise SystemExit(f"corpus row with {len(fields)} fields")
            parts = parts_of.get(bc.split_descriptor(fields), ["none"])
            rows.append((row[0], fields[0], fields[MOLREGNO_FIELD], fields[SMILES_FIELD], ",".join(parts)))
    distinct = sorted({x[3] for x in rows})
    ids = _identities(distinct, a.workers)

    def blocking(scope):
        return any(p in BLOCKING_PARTS for p in scope.split(","))

    a.out_dir.mkdir(parents=True)
    matches, near, failures = Counter(), Counter(), Counter()
    match_mols, blocking_n = defaultdict(set), 0
    applied, used = Counter(), set()
    on_list = Counter()
    with (a.out_dir / "matches.tsv").open("w", newline="") as fm, \
         (a.out_dir / "near_matches.tsv").open("w", newline="") as fn, \
         (a.out_dir / "identity_failures.tsv").open("w", newline="") as ff:
        wm, wn, wf = (csv.writer(x, delimiter="\t", lineterminator="\n") for x in (fm, fn, ff))
        wm.writerow(["activity_id", "assay_key", "molregno", "scope", "basis", "key", "tiers", "openadmet_names"])
        wn.writerow(["activity_id", "assay_key", "molregno", "scope", "kind", "key", "tiers", "openadmet_names"])
        wf.writerow(["activity_id", "assay_key", "molregno", "scope", "status", "canonical_smiles", "adjudicated"])
        for assay_key, aid, mr, smi, scope in rows:
            if mr in excluded:
                on_list[scope] += 1
            key, fk, status = ids[smi]
            row_blocks = False
            if status != "ok":
                failures[(scope, status)] += 1
                adj = (aid, mr, smi, status)
                if adj in adjudicated:
                    applied[(scope, status)] += 1
                    used.add(adj)
                else:
                    row_blocks = blocking(scope)
                wf.writerow([aid, assay_key, mr, scope, status, smi, "yes" if adj in adjudicated else ""])
            hit = None
            for basis, ck, table in (("full_standard_inchikey", key, full),
                                     ("largest_fragment_full_inchikey", fk, frag_full)):
                if ck and ck in table:
                    hit = (basis, ck, table[ck])
                    break
            if hit:
                basis, ck, names = hit
                matches[(scope, basis)] += 1
                match_mols[scope].add(mr)
                row_blocks = row_blocks or blocking(scope)
                wm.writerow([aid, assay_key, mr, scope, basis, ck,
                             ";".join(sorted({tiers_of[n] for n in names})), ";".join(sorted(names))])
            else:
                for kind, ck, table in (("first_block", key[:14], block),
                                        ("largest_fragment_first_block", fk[:14], frag_block)):
                    if ck and ck in table:
                        names = table[ck]
                        near[(scope, kind)] += 1
                        wn.writerow([aid, assay_key, mr, scope, kind, ck,
                                     ";".join(sorted({tiers_of[n] for n in names})), ";".join(sorted(names))])
                        break
            blocking_n += row_blocks

    unused = sorted(set(adjudicated) - used)
    blocking_n += len(unused)

    def by(counter):
        return {"%s|%s" % k: v for k, v in sorted(counter.items())}

    summary = {
        "artifact_type": "D013_firewalled_corpus_acceptance_v1",
        "rule": ("D013 Amendment 1 Change 2 acceptance: no train or valid row with a full-key or "
                 "largest-fragment full-key match to a protected structure, identities recomputed "
                 "from the built file; identity failures in train or valid fail closed unless listed in the "
                 "hash-pinned adjudication file; adjudication never clears a match; an unused adjudication fails"),
        "gate": "PASS" if blocking_n == 0 else "FAIL",
        "blocking_rows": blocking_n,
        "rows": len(rows),
        "rows_by_scope": dict(sorted(Counter(x[4] for x in rows).items())),
        "distinct_smiles": len(distinct),
        "protected_openadmet_structures": len(tiers_of),
        "matches_rows_by_scope_and_basis": by(matches),
        "matched_molregnos_by_scope": {k: len(v) for k, v in sorted(match_mols.items())},
        "identity_failures_by_scope_and_status": by(failures),
        "near_matches_rows_by_scope_and_kind_report_only": by(near),
        "exclusion_list_molregno_rows_by_scope_report_only": dict(sorted(on_list.items())),
        "adjudications": {
            "file_sha256": a.adjudications_sha256,
            "listed": len(adjudicated),
            "applied_rows_by_scope_and_status": by(applied),
            "unused_blocking": [list(k) for k in unused],
        },
        "identity_environment": e,
        "inputs": {"corpus": str(a.corpus), "corpus_sha256": a.corpus_sha256,
                   "audit_sha256": a.audit_sha256, "upstream_split_sha256": a.upstream_split_sha256,
                   "firewall_list_sha256": a.firewall_list_sha256,
                   "script_sha256": bf.sha256(Path(__file__)),
                   "identity_code_sha256": {"identity_overlap.py": bf.sha256(Path(io.__file__)),
                                            "build_firewall_list_v2.py": bf.sha256(Path(bf.__file__))}},
        "runtime_seconds": round(time.time() - started, 1),
        "workers": a.workers,
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    names = ["summary.json", "matches.tsv", "near_matches.tsv", "identity_failures.tsv"]
    (a.out_dir / "SHA256SUMS.txt").write_text("".join(f"{bf.sha256(a.out_dir / n)}  {n}\n" for n in names))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["gate"] != "PASS":
        print("ACCEPTANCE FAIL: protected structure or unverified identity in train/valid rows; "
              "see matches.tsv and identity_failures.tsv.", file=sys.stderr)
        return EXIT_GATE_FAIL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
