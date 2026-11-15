"""Tests for the D013 firewall list builder. Run in tripwire_cyp002."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_firewall_list as bf  # noqa: E402

PROP = "CC(C)NCC(O)COc1cccc2ccccc12"          # propranolol free base
PROP_HCL = "CC(C)NCC(O)COc1cccc2ccccc12.Cl"   # its hydrochloride
OTHER = "CCO"


def _key(smi):
    from rdkit import Chem
    from rdkit.Chem import inchi
    i = inchi.MolToInchi(Chem.MolFromSmiles(smi))
    return i, inchi.InchiToInchiKey(i)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_firewall_and_salt_near_match(tmp_path: Path):
    full_run = tmp_path / "full_run"
    full_run.mkdir()
    rows = [(1, PROP), (2, PROP_HCL), (3, OTHER)]
    with (full_run / "molecule_identities.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["molregno", "source_key", "witness_key", "source_vs_witness", "tiers", "openadmet_names", "openadmet_partitions", "membership_flags", "chembl32_key_witness"])
        for mr, smi in rows:
            k = _key(smi)[1]
            w.writerow([mr, k, k, "agree", "", "", "", "train_kept", ""])
    (full_run / "SHA256SUMS.txt").write_text(f"{_sha(full_run / 'molecule_identities.tsv')}  molecule_identities.tsv\n")
    mols = tmp_path / "molecules.tsv.gz"
    with gzip.open(mols, "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["molregno", "tsv_smiles", "chembl32_canonical_smiles", "chembl32_standard_inchi_key_witness", "membership_flags"])
        for mr, smi in rows:
            w.writerow([mr, smi, smi, "", "train_kept"])
    audit = tmp_path / "audit.tsv"
    inc, key = _key(PROP)
    with audit.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source_family", "source_partition", "source_id", "identity_state", "source_smiles", "standard_inchi", "full_standard_inchikey"])
        w.writerow(["openadmet", "validation", "OCNT-1", "represented", PROP, inc, key])
    out = tmp_path / "out"
    assert bf.main(["--full-run", str(full_run), "--export-molecules", str(mols), "--export-molecules-sha256", _sha(mols),
                    "--audit", str(audit), "--audit-sha256", _sha(audit), "--out-dir", str(out)]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["firewall_molregnos"] == 1 and s["firewall_molregnos_by_tier"] == {"development": 1}
    near = list(csv.DictReader((out / "near_matches_not_removed.tsv").open(), delimiter="\t"))
    assert {(r["molregno"], r["kind"]) for r in near} >= {("2", "largest_fragment_full")}
    assert all(r["molregno"] != "1" and r["molregno"] != "3" for r in near)
    with pytest.raises(SystemExit, match="overwrite"):
        bf.main(["--full-run", str(full_run), "--export-molecules", str(mols), "--export-molecules-sha256", _sha(mols),
                 "--audit", str(audit), "--audit-sha256", _sha(audit), "--out-dir", str(out)])
