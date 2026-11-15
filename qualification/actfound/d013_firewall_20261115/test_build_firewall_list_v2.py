"""Tests for the D013 pretraining exclusion list v2. Run in tripwire_cyp002."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_firewall_list_v2 as bf  # noqa: E402

PROP = "CC(C)NCC(O)COc1cccc2ccccc12"            # propranolol free base (protected)
PROP_HCL = "CC(C)NCC(O)COc1cccc2ccccc12.Cl"     # its hydrochloride: largest-fragment match
S_PROP = "CC(C)NC[C@H](O)COc1cccc2ccccc12"      # (S)-propranolol: first-block only, stays report-only
ATEN_HCL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1.Cl"  # protected as a salt
ATEN = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"         # ChEMBL free base: largest-fragment match
OTHER = "CCO"


def _key(smi):
    from rdkit import Chem
    from rdkit.Chem import inchi
    i = inchi.MolToInchi(Chem.MolFromSmiles(smi))
    return i, inchi.InchiToInchiKey(i)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _inputs(tmp: Path, chembl, protected):
    full_run = tmp / "full_run"
    full_run.mkdir()
    with (full_run / "molecule_identities.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["molregno", "source_key", "witness_key", "source_vs_witness", "tiers", "openadmet_names",
                    "openadmet_partitions", "membership_flags", "chembl32_key_witness"])
        for mr, smi in chembl:
            k = _key(smi)[1]
            w.writerow([mr, k, k, "agree", "", "", "", "train_kept", ""])
    (full_run / "SHA256SUMS.txt").write_text(f"{_sha(full_run / 'molecule_identities.tsv')}  molecule_identities.tsv\n")
    mols = tmp / "molecules.tsv.gz"
    with gzip.open(mols, "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["molregno", "tsv_smiles", "chembl32_canonical_smiles", "chembl32_standard_inchi_key_witness", "membership_flags"])
        for mr, smi in chembl:
            w.writerow([mr, smi, smi, "", "train_kept"])
    audit = tmp / "audit.tsv"
    with audit.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source_family", "source_partition", "source_id", "identity_state", "source_smiles", "standard_inchi", "full_standard_inchikey"])
        for name, part, smi in protected:
            inc, key = _key(smi)
            w.writerow(["openadmet", part, name, "represented", smi, inc, key])
    return ["--full-run", str(full_run), "--export-molecules", str(mols), "--export-molecules-sha256", _sha(mols),
            "--audit", str(audit), "--audit-sha256", _sha(audit)]


def test_full_key_and_largest_fragment_excluded_first_block_reported(tmp_path: Path):
    chembl = [(1, PROP), (2, PROP_HCL), (3, S_PROP), (4, ATEN), (5, OTHER)]
    protected = [("OCNT-1", "validation", PROP), ("OCNT-2", "internal_test", ATEN_HCL)]
    args = _inputs(tmp_path, chembl, protected)
    out = tmp_path / "out"
    assert bf.main(args + ["--out-dir", str(out)]) == 0
    excl = {r["molregno"]: r for r in csv.DictReader((out / "exclusion_molregnos.tsv").open(), delimiter="\t")}
    assert {m: r["exclusion_basis"] for m, r in excl.items()} == {
        "1": "full_standard_inchikey", "2": "largest_fragment_full_inchikey", "4": "largest_fragment_full_inchikey"}
    assert excl["4"]["tiers"] == "internal_test"
    s = json.loads((out / "summary.json").read_text())
    assert s["excluded_molregnos"] == 3
    assert s["excluded_molregnos_by_basis_and_tier"] == {
        "full_standard_inchikey|development": 1, "largest_fragment_full_inchikey|development": 1,
        "largest_fragment_full_inchikey|internal_test": 1}
    near = list(csv.DictReader((out / "near_matches_not_removed.tsv").open(), delimiter="\t"))
    assert {r["molregno"] for r in near} == {"3"}  # (S)-propranolol: report-only, not excluded
    assert {r["kind"] for r in near} >= {"first_block"}


def test_refuses_existing_output(tmp_path: Path):
    args = _inputs(tmp_path, [(1, PROP)], [("OCNT-1", "train", PROP)])
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit, match="overwrite"):
        bf.main(args + ["--out-dir", str(out)])
