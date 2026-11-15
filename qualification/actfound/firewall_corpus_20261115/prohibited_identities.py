"""D013 firewall: prohibited ChEMBL32 molregnos, plus a report-only near-match census.

Runs on the Mac in the frozen D012 identity environment (tripwire_cyp002:
Python 3.12.14, RDKit 2026.03.6, InChI 1.07.3). Refuses any other environment.

Firewall (D013): a molregno is prohibited when the full Standard InChIKey of its
ChEMBL32 source structure equals the full key of any of the 5,655 OpenADMET CYP
structures (development universe + internal test + blinded challenge). The keys
come from the completed overlap audit (identity_overlap.py full run), so no
identity is recomputed for the firewall itself.

Near-match census (Q4, report only, removes nothing):
  first_block   same InChIKey connectivity block (first 14 characters)
  largest_frag  same full key after RDKit LargestFragmentChooser on both sides
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

FROZEN_ENV = {"python": "3.12.14", "rdkit": "2026.03.6", "inchi": "1.07.3"}
TIER_OF_PARTITION = {"train": "development", "validation": "development",
                     "internal_test": "internal_test", "challenge_blinded_test": "blinded"}
KNOWN_PROHIBITED = {1370, 88100}  # Session 005 §30


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def runtime_versions() -> dict[str, str]:
    import rdkit
    from rdkit.Chem import inchi
    return {"python": sys.version.split()[0], "rdkit": rdkit.__version__, "inchi": inchi.GetInchiVersion()}


def verify_sums(directory: Path) -> dict[str, str]:
    sums = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        if file_sha256(directory / name) != digest:
            raise SystemExit(f"SHA-256 mismatch: {directory / name}")
        sums[name] = digest
    return sums


def largest_fragment_key(smiles: str) -> str:
    if not smiles:
        return ""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    frag = rdMolStandardize.LargestFragmentChooser().choose(mol)
    std = inchi.MolToInchi(frag)
    return inchi.InchiToInchiKey(std) if std else ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--overlap-run", type=Path, required=True, help="identity_overlap.py full_run directory")
    p.add_argument("--export-dir", type=Path, required=True, help="CoreHPC membership export (molecules.tsv.gz)")
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-sha256", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)

    env = runtime_versions()
    if env != FROZEN_ENV:
        raise SystemExit(f"not the frozen identity environment: {env}")
    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")
    if file_sha256(a.audit) != a.audit_sha256:
        raise SystemExit("audit SHA-256 mismatch")
    run_sums = verify_sums(a.overlap_run)
    export_sums = verify_sums(a.export_dir)
    summary_in = json.loads((a.overlap_run / "summary.json").read_text())
    if summary_in.get("scope") != "full":
        raise SystemExit("overlap run is not a full run")
    started = time.time()

    # OpenADMET prohibited keys, first blocks and largest-fragment keys.
    oa_full: dict[str, set] = defaultdict(set)
    oa_smiles: list[tuple[str, str]] = []
    with a.audit.open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["source_family"] != "openadmet" or r["identity_state"] != "represented":
                continue
            tier = TIER_OF_PARTITION[r["source_partition"]]
            oa_full[r["full_standard_inchikey"]].add(tier)
            oa_smiles.append((r["source_smiles"], tier))
    oa_block: dict[str, set] = defaultdict(set)
    for key, tiers in oa_full.items():
        oa_block[key[:14]] |= tiers
    oa_frag: dict[str, set] = defaultdict(set)
    for smi, tier in oa_smiles:
        k = largest_fragment_key(smi)
        if k:
            oa_frag[k].add(tier)

    rows = list(csv.DictReader((a.overlap_run / "molecule_identities.tsv").open(newline=""), delimiter="\t"))
    smiles_by_mr = {}
    with gzip.open(a.export_dir / "molecules.tsv.gz", "rt", newline="") as f:
        for m in csv.DictReader(f, delimiter="\t"):
            smiles_by_mr[m["molregno"]] = m["chembl32_canonical_smiles"]
    if len(smiles_by_mr) != len(rows):
        raise SystemExit("export and overlap run cover different molecule sets")

    a.out_dir.mkdir(parents=True)
    prohibited = 0
    tiers_count = Counter()
    with (a.out_dir / "prohibited_molregnos.tsv").open("w", newline="") as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        w.writerow(["molregno", "full_standard_inchikey", "tiers", "openadmet_names"])
        for r in rows:
            if r["tiers"]:
                if oa_full.get(r["source_key"]) is None:
                    raise SystemExit(f"overlap run tier without matching audit key: {r['molregno']}")
                w.writerow([r["molregno"], r["source_key"], r["tiers"], r["openadmet_names"]])
                prohibited += 1
                for t in r["tiers"].split(";"):
                    tiers_count[t] += 1
    listed = {int(r["molregno"]) for r in rows if r["tiers"]}
    if not KNOWN_PROHIBITED <= listed:
        raise SystemExit("known §30 molecules missing from the prohibited list")

    # Near-match census for molecules NOT already prohibited.
    candidates = [r for r in rows if not r["tiers"]]
    smiles = [smiles_by_mr[r["molregno"]] for r in candidates]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            frag_keys = list(ex.map(largest_fragment_key, smiles, chunksize=2000))
    else:
        frag_keys = [largest_fragment_key(s) for s in smiles]
    census = Counter()
    with (a.out_dir / "near_matches_report_only.tsv").open("w", newline="") as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        w.writerow(["molregno", "kind", "source_key", "largest_fragment_key", "matched_tiers", "membership_flags"])
        for r, fk in zip(candidates, frag_keys):
            key = r["source_key"]
            hits = []
            if key and key[:14] in oa_block:
                hits.append(("first_block", oa_block[key[:14]]))
            if fk and (fk in oa_full or fk in oa_frag):
                hits.append(("largest_frag", oa_full.get(fk, set()) | oa_frag.get(fk, set())))
            for kind, tiers in hits:
                w.writerow([r["molregno"], kind, key, fk, ";".join(sorted(tiers)), r["membership_flags"]])
                train = "train_kept" in r["membership_flags"].split(";")
                for t in tiers:
                    census[f"{kind}|{t}|{'train_kept' if train else 'other'}"] += 1

    summary = {
        "artifact_type": "D013_firewall_prohibited_molregnos_v1",
        "authority": "D013 firewall list (full Standard InChIKey); near-match census is report-only",
        "inputs": {"overlap_run_sha256sums": run_sums, "export_sha256sums": export_sums,
                   "audit": str(a.audit), "audit_sha256": a.audit_sha256,
                   "script_sha256": file_sha256(Path(__file__))},
        "identity_environment": env,
        "openadmet_full_keys": len(oa_full),
        "prohibited_molregnos": prohibited,
        "prohibited_molregnos_by_tier": dict(tiers_count),
        "near_match_census_report_only": dict(sorted(census.items())),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    names = ["prohibited_molregnos.tsv", "near_matches_report_only.tsv", "summary.json"]
    (a.out_dir / "SHA256SUMS.txt").write_text("".join(f"{file_sha256(a.out_dir / n)}  {n}\n" for n in names))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
