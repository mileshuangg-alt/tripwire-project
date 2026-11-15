"""Read-only export of ActFound job 3066823 training-corpus membership.

Runs on CoreHPC in the pinned ActFound environment (Python 3.9). It decides
only WHICH rows the training loader used; it never adjudicates molecular
identity. Identity is computed later, on the Mac, in the frozen D012 identity
environment by identity_overlap.py.

Membership is loader-exact: the project loader
actfound_model_side.iter_frozen_graph_qualified_assays (the code job 3066823
loaded) is run on the frozen TSV, and every TSV row is aligned to the rows that
loader kept. Each row is labelled with its split (train / validation / none)
via the per-ID reconciliation, so both the loader-kept count and the
all-rows upper bound can be reported.

Outputs (new directory, nothing else is written):
  membership.tsv.gz   one line per TSV row
  molecules.tsv.gz    one line per molregno: TSV SMILES + ChEMBL32 source SMILES
  export_summary.json input hashes, counts, runtime
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

PCT = "pct_units"
REL = "relation_not_eq"
PARSE = "rdkit_parse_failure"
GT10K = "assay_gt_10000_rows"
NOT_EMITTED = "assay_not_emitted_lt20_or_std"
KEPT = "kept"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def descriptor(row: dict) -> str:
    # Mirrors actfound_model_side._upstream_assay_descriptor.
    return "{}_{}_{}".format(row["assay_chembl_id"], row["standard_units"], row["standard_type"]).replace("/", "_")


def iter_groups(rows: Iterable[dict]) -> Iterator[list[dict]]:
    """Group consecutive rows by assay_key, as the loader does."""
    current: Optional[str] = None
    group: list[dict] = []
    for row in rows:
        if current is not None and row["assay_key"] != current:
            yield group
            group = []
        current = row["assay_key"]
        group.append(row)
    if group:
        yield group


def label_group(group: list[dict], emitted: Optional[object]) -> Optional[list[str]]:
    """Return a per-row reason list if `emitted` is this group's loader output, else None.

    `emitted` is an ActFoundAssay-like object with .assay_id and .smiles.
    Rows that pass the unit/relation gates are matched in order against the
    loader's kept SMILES; a candidate that is skipped must have failed RDKit
    parsing (the only remaining per-row gate).
    """
    reasons: list[str] = []
    candidates: list[int] = []
    for i, row in enumerate(group):
        if row["standard_units"] == "%":
            reasons.append(PCT)
        elif row["standard_relation"] != "=":
            reasons.append(REL)
        else:
            reasons.append("")
            candidates.append(i)
    if emitted is None or not candidates:
        return None
    if descriptor(group[candidates[0]]) != emitted.assay_id:
        return None
    expected = list(emitted.smiles)
    j = 0
    for i in candidates:
        if j < len(expected) and group[i]["canonical_smiles"] == expected[j]:
            reasons[i] = KEPT
            j += 1
        else:
            reasons[i] = PARSE
    if j != len(expected):
        return None
    return reasons


def unemitted_reasons(group: list[dict]) -> list[str]:
    out = []
    for row in group:
        if row["standard_units"] == "%":
            out.append(PCT)
        elif row["standard_relation"] != "=":
            out.append(REL)
        else:
            out.append(GT10K if len(group) > 10000 else NOT_EMITTED)
    return out


def align(groups: Iterable[list[dict]], emitted_assays: Iterator[object]) -> Iterator[tuple[list[dict], list[str]]]:
    """Pair each TSV group with loader output; fail if any loader assay is left unmatched."""
    pending = next(emitted_assays, None)
    for group in groups:
        reasons = label_group(group, pending)
        if reasons is not None:
            pending = next(emitted_assays, None)
        else:
            reasons = unemitted_reasons(group)
        yield group, reasons
    if pending is not None:
        raise RuntimeError(f"loader assay not aligned to any TSV group: {pending.assay_id}")


def read_split(per_id_path: Path) -> dict[str, str]:
    split: dict[str, str] = {}
    with per_id_path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            part = "validation" if row["partition"] in ("val", "validation") else row["partition"]
            key = row["mapped_assay_key"]
            if key in split and split[key] != part:
                raise RuntimeError(f"mapped assay key in two partitions: {key}")
            split[key] = part
    return split


def chembl_structures(db_path: Path, molregnos: set[int]) -> dict[int, tuple[str, str]]:
    """Stream compound_structures read-only; return molregno -> (canonical_smiles, standard_inchi_key)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = {}
        for molregno, smi, key in con.execute(
            "SELECT molregno, canonical_smiles, standard_inchi_key FROM compound_structures"
        ):
            if molregno in molregnos:
                out[molregno] = (smi or "", key or "")
        return out
    finally:
        con.close()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", type=Path, required=True)
    p.add_argument("--tsv-sha256", required=True)
    p.add_argument("--per-id", type=Path, required=True)
    p.add_argument("--per-id-sha256", required=True)
    p.add_argument("--chembl-db", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)

    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")
    started = time.time()
    for path, expected in ((a.tsv, a.tsv_sha256), (a.per_id, a.per_id_sha256)):
        actual = file_sha256(path)
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    split = read_split(a.per_id)

    sys.path.insert(0, str(a.repo_root / "cyp" / "scripts"))
    import actfound_model_side as ms  # noqa: E402  (pinned env: torch, numpy, rdkit)

    a.out_dir.mkdir(parents=True)
    counts: dict[str, int] = {}
    mol_smiles: dict[int, str] = {}
    mol_flags: dict[int, set[str]] = {}

    with a.tsv.open(newline="") as f, gzip.open(a.out_dir / "membership.tsv.gz", "wt", newline="") as out:
        reader = csv.DictReader(f, delimiter="\t")
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        w.writerow(["assay_key", "descriptor", "split", "molregno", "loader_status", "tsv_smiles"])
        emitted = ms.iter_frozen_graph_qualified_assays(a.tsv)
        for group, reasons in align(iter_groups(reader), emitted):
            for row, reason in zip(group, reasons):
                desc = descriptor(row)
                part = split.get(desc, "none")
                mr = int(row["molregno"])
                w.writerow([row["assay_key"], desc, part, mr, reason, row["canonical_smiles"]])
                k = f"{part}|{reason}"
                counts[k] = counts.get(k, 0) + 1
                prev = mol_smiles.setdefault(mr, row["canonical_smiles"])
                if prev != row["canonical_smiles"]:
                    counts["molregno_multiple_tsv_smiles"] = counts.get("molregno_multiple_tsv_smiles", 0) + 1
                mol_flags.setdefault(mr, set()).add(f"{part}_{'kept' if reason == KEPT else 'dropped'}")

    structures = chembl_structures(a.chembl_db, set(mol_smiles))
    with gzip.open(a.out_dir / "molecules.tsv.gz", "wt", newline="") as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        w.writerow(["molregno", "tsv_smiles", "chembl32_canonical_smiles", "chembl32_standard_inchi_key_witness", "membership_flags"])
        for mr in sorted(mol_smiles):
            smi, key = structures.get(mr, ("", ""))
            w.writerow([mr, mol_smiles[mr], smi, key, ";".join(sorted(mol_flags[mr]))])

    summary = {
        "artifact_type": "ActFound_3066823_training_membership_export_v1",
        "authority": "membership_only_no_identity_authority",
        "inputs": {
            "tsv": str(a.tsv), "tsv_sha256": a.tsv_sha256,
            "per_id": str(a.per_id), "per_id_sha256": a.per_id_sha256,
            "chembl_db": str(a.chembl_db),
            "loader_module_sha256": file_sha256(a.repo_root / "cyp" / "scripts" / "actfound_model_side.py"),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "row_counts_by_split_and_loader_status": dict(sorted(counts.items())),
        "unique_molregnos": len(mol_smiles),
        "molregnos_missing_chembl32_structure": len(set(mol_smiles) - set(structures)),
        "runtime": {"python": platform.python_version(), "seconds": round(time.time() - started, 1)},
    }
    (a.out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
