"""D013 pretraining exclusion list v2: full-key firewall plus conservative largest-fragment exclusion.

Runs in the frozen D012 identity environment (tripwire_cyp002: Python 3.12.14,
RDKit 2026.03.6, InChI 1.07.3). Identities come from the full overlap run
(qualification/actfound/overlap_audit_20261115/full_run), which already holds a
full Standard InChIKey for every molregno in the rebuilt corpus, a superset of
the authors' processed ChEMBL32 molecules.

Excluded (D013 + Amendment 1 pretraining exclusion, approved 2026-11-15):
  exclusion_basis = full_standard_inchikey: the molregno's full Standard
    InChIKey equals that of any of the 5,655 OpenADMET structures
    (development + internal_test + blinded). This is the identity rule.
  exclusion_basis = largest_fragment_full_inchikey: not matched above, but the
    full Standard InChIKey of its largest fragment equals that of a protected
    structure's largest fragment. A conservative pretraining exclusion only;
    these are largest-fragment matches, not proven identical active compounds
    or activities, and the identity rule is unchanged.
Largest fragment = the fragment with the most heavy atoms (ties broken by
canonical SMILES); a molecule with one fragment is its own largest fragment.

Report-only (never removed here, not deemed safe): molregnos not excluded whose
InChIKey first block, or largest-fragment first block, matches a protected
structure.

Same inputs, environment and matching code as v1 (build_firewall_list.py),
which is kept unchanged with its output in firewall_list/.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

FROZEN_ENV = {"python": "3.12.14", "rdkit": "2026.03.6", "inchi": "1.07.3"}
TIER = {"train": "development", "validation": "development", "internal_test": "internal_test", "challenge_blinded_test": "blinded"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def env() -> dict[str, str]:
    import rdkit
    from rdkit.Chem import inchi
    return {"python": sys.version.split()[0], "rdkit": rdkit.__version__, "inchi": inchi.GetInchiVersion()}


def largest_fragment_key(smiles: str) -> str:
    """Full Standard InChIKey of the largest fragment; '' if unparsable."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    best = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))
    std = inchi.MolToInchi(best)
    return inchi.InchiToInchiKey(std) if std else ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full-run", type=Path, required=True)
    p.add_argument("--export-molecules", type=Path, required=True)
    p.add_argument("--export-molecules-sha256", required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-sha256", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)

    e = env()
    if e != FROZEN_ENV:
        raise SystemExit(f"not the frozen identity environment: {e}")
    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")
    for line in (a.full_run / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(None, 1)
        if sha256(a.full_run / name.strip()) != digest:
            raise SystemExit(f"full-run file SHA-256 mismatch: {name}")
    if sha256(a.export_molecules) != a.export_molecules_sha256:
        raise SystemExit("export molecules SHA-256 mismatch")
    if sha256(a.audit) != a.audit_sha256:
        raise SystemExit("audit SHA-256 mismatch")

    # Protected structures: full key, first block, largest-fragment key -> OpenADMET names/tiers.
    full, block, frag_full, frag_block = (defaultdict(set) for _ in range(4))
    tiers_of: dict[str, str] = {}
    n_protected = 0
    with a.audit.open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["source_family"] != "openadmet":
                continue
            if r["identity_state"] != "represented":
                raise SystemExit(f"unrepresented protected structure: {r['source_id']}")
            n_protected += 1
            name, key = r["source_id"], r["full_standard_inchikey"]
            tiers_of[name] = TIER[r["source_partition"]]
            full[key].add(name)
            block[key[:14]].add(name)
            fk = largest_fragment_key(r["source_smiles"]) if "." in r["source_smiles"] else key
            frag_full[fk].add(name)
            frag_block[fk[:14]].add(name)

    smiles = {}
    with gzip.open(a.export_molecules, "rt", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            smiles[int(r["molregno"])] = r["chembl32_canonical_smiles"]

    a.out_dir.mkdir(parents=True)
    excluded_by_basis_tier = Counter()
    near_kinds = Counter()
    n_rows = 0
    with (a.full_run / "molecule_identities.tsv").open(newline="") as f, \
         (a.out_dir / "exclusion_molregnos.tsv").open("w", newline="") as fw, \
         (a.out_dir / "near_matches_not_removed.tsv").open("w", newline="") as nm:
        wf = csv.writer(fw, delimiter="\t", lineterminator="\n")
        wn = csv.writer(nm, delimiter="\t", lineterminator="\n")
        wf.writerow(["molregno", "exclusion_basis", "full_standard_inchikey", "matched_key", "tiers", "openadmet_names"])
        wn.writerow(["molregno", "kind", "chembl_key", "compared_key", "tiers", "openadmet_names"])
        for r in csv.DictReader(f, delimiter="\t"):
            n_rows += 1
            mr, key = int(r["molregno"]), r["source_key"]
            if not key:
                continue
            fk = largest_fragment_key(smiles.get(mr, "")) if "." in smiles.get(mr, "") else key
            for basis, ck, table in (("full_standard_inchikey", key, full),
                                     ("largest_fragment_full_inchikey", fk, frag_full)):
                if ck and ck in table:
                    names = sorted(table[ck])
                    tiers = sorted({tiers_of[n] for n in names})
                    wf.writerow([mr, basis, key, ck, ";".join(tiers), ";".join(names)])
                    for t in tiers:
                        excluded_by_basis_tier[f"{basis}|{t}"] += 1
                    break
            else:
                for kind, ck, table in (("first_block", key[:14], block),
                                        ("largest_fragment_first_block", fk[:14], frag_block)):
                    if ck and ck in table:
                        names = sorted(table[ck])
                        wn.writerow([mr, kind, key, ck, ";".join(sorted({tiers_of[n] for n in names})), ";".join(names)])
                        near_kinds[kind] += 1

    if n_rows != len(smiles):
        raise SystemExit(f"identity rows ({n_rows}) != export molecules ({len(smiles)})")
    summary = {
        "artifact_type": "D013_ActFound_pretraining_exclusion_list_v2",
        "authority": ("D013 firewall (full Standard InChIKey identity rule) plus D013 Amendment 1 conservative "
                      "pretraining exclusion (largest-fragment full Standard InChIKey); applies to all pretraining "
                      "rows, train and validation"),
        "inputs": {
            "full_run_sha256sums": (a.full_run / "SHA256SUMS.txt").read_text(),
            "export_molecules_sha256": a.export_molecules_sha256,
            "audit_sha256": a.audit_sha256,
            "script_sha256": sha256(Path(__file__)),
        },
        "identity_environment": e,
        "protected_openadmet_structures": n_protected,
        "molregnos_considered": n_rows,
        "excluded_molregnos": sum(1 for _ in (a.out_dir / "exclusion_molregnos.tsv").open()) - 1,
        "excluded_molregnos_by_basis_and_tier": dict(sorted(excluded_by_basis_tier.items())),
        "near_matches_not_removed_by_kind": dict(near_kinds),
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    names = ["exclusion_molregnos.tsv", "near_matches_not_removed.tsv", "summary.json"]
    (a.out_dir / "SHA256SUMS.txt").write_text("".join(f"{sha256(a.out_dir / n)}  {n}\n" for n in names))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
