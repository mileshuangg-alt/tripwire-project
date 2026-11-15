"""Tests for the D013 corpus identity acceptance check. Run in tripwire_cyp002."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_corpus_identity as cc  # noqa: E402

PROP = "CC(C)NCC(O)COc1cccc2ccccc12"            # protected
PROP_HCL = "CC(C)NCC(O)COc1cccc2ccccc12.Cl"     # largest-fragment match
S_PROP = "CC(C)NC[C@H](O)COc1cccc2ccccc12"      # first-block only
CLEAN = ["CCO", "CCCO", "CCCCO"]
HEADER = ["assay_key", "activity_id", "assay_id", "record_id_unused", "molregno", "bao_endpoint", "standard_relation",
          "standard_value", "standard_units", "standard_type", "assay_type", "bao_format", "assay_chembl_id",
          "count_scratch", "canonical_smiles", "graph_pair_count"]


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _setup(tmp: Path, corpus_rows, excluded=()):
    """corpus_rows: (chembl_assay, molregno, smiles). CHEMBL1 = train, CHEMBL2 = valid, CHEMBL9 = outside."""
    from rdkit import Chem
    from rdkit.Chem import inchi
    corpus = tmp / "corpus.tsv"
    with corpus.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(HEADER)
        for i, (assay, mr, smi) in enumerate(corpus_rows):
            w.writerow([f"{assay[6:]}_nM_IC50", str(i + 1), assay[6:], "0", mr, "B", "=", "1", "nM", "IC50", "B", "B",
                        assay, "0", smi, "1"])
    excl = tmp / "excl.tsv"
    excl.write_text("molregno\texclusion_basis\tfull_standard_inchikey\tmatched_key\ttiers\topenadmet_names\n"
                    + "".join(f"{m}\tlargest_fragment_full_inchikey\tK\tK\tdevelopment\tOCNT-1\n" for m in excluded))
    audit = tmp / "audit.tsv"
    i = inchi.MolToInchi(Chem.MolFromSmiles(PROP))
    audit.write_text("source_family\tsource_partition\tsource_id\tidentity_state\tsource_smiles\tstandard_inchi\tfull_standard_inchikey\n"
                     f"openadmet\tvalidation\tOCNT-1\trepresented\t{PROP}\t{i}\t{inchi.InchiToInchiKey(i)}\n")
    split = tmp / "split.json"
    split.write_text(json.dumps({"train": ["CHEMBL1_nM_IC50"], "valid": ["CHEMBL2_nM_IC50"]}))
    args = []
    for n, pth in (("corpus", corpus), ("exclusion-list", excl), ("audit", audit), ("upstream-split", split)):
        args += [f"--{n}", str(pth), f"--{n}-sha256", _sha(pth)]
    return args + ["--workers", "2", "--out-dir", str(tmp / "out")]


def _run(tmp, rows, excluded=()):
    rc = cc.main(_setup(tmp, rows, excluded))
    return rc, json.loads((tmp / "out" / "summary.json").read_text())


def _clean_rows():
    return [("CHEMBL1", str(10 + i), s) for i, s in enumerate(CLEAN)] + [("CHEMBL2", "20", "CCN")]


def test_clean_corpus_passes(tmp_path: Path):
    rc, s = _run(tmp_path, _clean_rows())
    assert rc == 0 and s["gate"] == "PASS" and s["finding_rows"] == {}
    assert s["rows_by_scope"] == {"train": 3, "validation": 1}


def test_largest_fragment_match_in_validation_row_fails(tmp_path: Path):
    rc, s = _run(tmp_path, _clean_rows() + [("CHEMBL2", "30", PROP_HCL)])
    assert rc == cc.EXIT_GATE_FAIL and s["gate"] == "FAIL"
    assert s["finding_rows"] == {"validation|largest_fragment_full_key_match": 1}


def test_full_key_match_in_train_row_fails(tmp_path: Path):
    rc, s = _run(tmp_path, _clean_rows() + [("CHEMBL1", "31", PROP)])
    assert rc == cc.EXIT_GATE_FAIL
    assert s["finding_rows"]["train|full_key_match"] == 1


def test_outside_split_and_first_block_are_report_only(tmp_path: Path):
    rc, s = _run(tmp_path, _clean_rows() + [("CHEMBL9", "32", PROP_HCL), ("CHEMBL1", "33", S_PROP)])
    assert rc == 0 and s["gate"] == "PASS"
    # A one-fragment molecule is its own largest fragment, so it also matches on largest-fragment first block.
    assert s["finding_rows"] == {"outside_split|largest_fragment_full_key_match": 1, "train|first_block_match": 1,
                                 "train|largest_fragment_first_block_match": 1}


def test_uncomputable_identity_or_excluded_molregno_fail_closed(tmp_path: Path):
    rc, s = _run(tmp_path, _clean_rows() + [("CHEMBL1", "34", "not_a_smiles")], excluded=["10"])
    assert rc == cc.EXIT_GATE_FAIL
    assert s["finding_rows"]["train|identity_not_computable"] == 1
    assert s["finding_rows"]["train|molregno_on_exclusion_list"] == 1


def test_refuses_hash_mismatch(tmp_path: Path):
    args = _setup(tmp_path, _clean_rows())
    args[args.index("--upstream-split-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit, match="upstream_split missing or SHA-256 mismatch"):
        cc.main(args)
    assert not (tmp_path / "out").exists()
