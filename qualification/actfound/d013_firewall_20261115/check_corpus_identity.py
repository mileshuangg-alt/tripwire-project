"""D013 acceptance check: no protected structure in the built pretraining corpus.

Runs on the Mac in the frozen identity environment (tripwire_cyp002: Python
3.12.14, RDKit 2026.03.6, InChI 1.07.3), on the built firewalled corpus TSV
copied from CoreHPC and hash-verified.

Identities are RECOMPUTED from the SMILES in the built file. The exclusion list
is not trusted for this; it is only cross-checked. The rule (D013 + Amendment 1,
approved 2026-11-15):
  blocking: a row in an upstream train or valid assay whose full Standard
    InChIKey, or whose largest-fragment full Standard InChIKey, equals that of
    a protected OpenADMET structure (full key / largest-fragment key).
    Also blocking, fail-closed: such a row whose identity cannot be computed,
    or whose molregno is on the exclusion list.
  report-only: the same findings outside train/valid, and first-block or
    largest-fragment first-block matches anywhere (reported, not deemed safe).
Largest fragment: the same function as the exclusion list builder.
Gate FAIL keeps all outputs and exits 3.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_firewall_list_v2 as bf  # noqa: E402  (same largest-fragment rule and tier map)

EXIT_GATE_FAIL = 3
CORPUS_HEADER_PREFIX = ["assay_key", "activity_id", "assay_id", "record_id_unused", "molregno"]
SMILES_COL = 14  # canonical_smiles in the graph-stage output (assay_key first)


def sha256(path: Path) -> str:
    return bf.sha256(path)


def identities(smiles: str):
    """(full key, largest-fragment full key) or ('', '') if not computable."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return smiles, "", ""
    std = inchi.MolToInchi(mol)
    key = inchi.InchiToInchiKey(std) if std else ""
    if not key:
        return smiles, "", ""
    fk = bf.largest_fragment_key(smiles) if "." in smiles else key
    return smiles, key, fk


def protected_tables(audit: Path):
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    for name in ("corpus", "exclusion-list", "audit", "upstream-split"):
        p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument(f"--{name}-sha256", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)

    e = bf.env()
    if e != bf.FROZEN_ENV:
        raise SystemExit(f"not the frozen identity environment: {e}")
    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")
    for name in ("corpus", "exclusion_list", "audit", "upstream_split"):
        path, expected = getattr(a, name), getattr(a, f"{name}_sha256")
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"{name} missing or SHA-256 mismatch: {path}")
    split = json.loads(a.upstream_split.read_text())
    train, valid = set(split.get("train") or []), set(split.get("valid") or [])
    if not train or not valid or train & valid:
        raise SystemExit("upstream split must have non-empty, disjoint train and valid lists")
    excluded = {}
    with a.exclusion_list.open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            excluded[r["molregno"]] = r["exclusion_basis"]

    started = time.time()
    full, block, frag_full, frag_block, tiers_of = protected_tables(a.audit)

    rows = []  # (scope, assay_key, activity_id, molregno, smiles)
    with a.corpus.open(newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        if header[:5] != CORPUS_HEADER_PREFIX or header[SMILES_COL] != "canonical_smiles":
            raise SystemExit("unexpected corpus header")
        for row in r:
            desc = f"{row[12]}_{row[8]}_{row[9]}".replace("/", "_")
            scope = "train" if desc in train else "validation" if desc in valid else "outside_split"
            rows.append((scope, row[0], row[1], row[4], row[SMILES_COL]))
    uniq = sorted({x[4] for x in rows})
    with Pool(a.workers) as pool:
        ident = {s: (k, fk) for s, k, fk in pool.imap_unordered(identities, uniq, chunksize=2000)}

    a.out_dir.mkdir(parents=True)
    counts = Counter()
    mols = defaultdict(set)
    with (a.out_dir / "identity_findings.tsv").open("w", newline="") as o:
        w = csv.writer(o, delimiter="\t", lineterminator="\n")
        w.writerow(["scope", "finding", "blocking", "assay_key", "activity_id", "molregno", "matched_key", "tiers", "openadmet_names"])
        for scope, akey, aid, mr, smi in rows:
            key, fk = ident[smi]
            findings = []
            if not key:
                findings.append(("identity_not_computable", "", set()))
            else:
                if key in full:
                    findings.append(("full_key_match", key, full[key]))
                if fk in frag_full:
                    findings.append(("largest_fragment_full_key_match", fk, frag_full[fk]))
                if not findings:
                    if key[:14] in block:
                        findings.append(("first_block_match", key[:14], block[key[:14]]))
                    if fk[:14] in frag_block:
                        findings.append(("largest_fragment_first_block_match", fk[:14], frag_block[fk[:14]]))
            if mr in excluded:
                findings.append(("molregno_on_exclusion_list", excluded[mr], set()))
            for finding, mk, names in findings:
                blocking = scope != "outside_split" and not finding.endswith("first_block_match")
                counts[f"{scope}|{finding}"] += 1
                mols[f"{scope}|{finding}"].add(mr)
                w.writerow([scope, finding, int(blocking), akey, aid, mr, mk,
                            ";".join(sorted({tiers_of[n] for n in names})), ";".join(sorted(names))])
    blocking_rows = sum(n for k, n in counts.items()
                        if not k.startswith("outside_split|") and not k.endswith("first_block_match"))
    summary = {
        "artifact_type": "D013_corpus_identity_acceptance_v1",
        "rule": "blocking: full-key or largest-fragment full-key match, uncomputable identity, or excluded molregno "
                "in a train/valid row; report-only: outside-split findings and first-block matches",
        "gate": "PASS" if blocking_rows == 0 else "FAIL",
        "blocking_rows": blocking_rows,
        "rows_by_scope": dict(Counter(x[0] for x in rows)),
        "unique_smiles": len(uniq),
        "finding_rows": dict(sorted(counts.items())),
        "finding_molregnos": {k: len(v) for k, v in sorted(mols.items())},
        "inputs": {n: {"path": str(getattr(a, n)), "sha256": getattr(a, f"{n}_sha256")}
                   for n in ("corpus", "exclusion_list", "audit", "upstream_split")},
        "script_sha256": sha256(Path(__file__)),
        "identity_environment": e,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    names = ["identity_findings.tsv", "summary.json"]
    (a.out_dir / "SHA256SUMS.txt").write_text("".join(f"{sha256(a.out_dir / n)}  {n}\n" for n in names))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["gate"] != "PASS":
        print("IDENTITY ACCEPTANCE FAIL: protected structures in train/valid rows; see identity_findings.tsv",
              file=sys.stderr)
        return EXIT_GATE_FAIL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
