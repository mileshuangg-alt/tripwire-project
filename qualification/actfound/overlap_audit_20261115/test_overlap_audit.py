"""Tests for the job 3066823 overlap audit. Run in the frozen identity env (tripwire_cyp002)."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import export_training_membership as ex  # noqa: E402
import identity_overlap as io  # noqa: E402

PROPRANOLOL = "CC(C)NCC(O)COc1cccc2ccccc12"
RITONAVIR = "CC(C)c1nc(CN(C)C(=O)N[C@H](C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1"


def _row(key, smi, units="nM", rel="=", chembl="CHEMBL1", stype="IC50", molregno="1"):
    return {"assay_key": key, "canonical_smiles": smi, "standard_units": units, "standard_relation": rel,
            "assay_chembl_id": chembl, "standard_type": stype, "molregno": molregno}


def test_alignment_labels_each_loader_gate():
    group = [
        _row("a", "CCO"),
        _row("a", "CCN", units="%"),
        _row("a", "CCC", rel=">"),
        _row("a", "not-a-smiles"),
        _row("a", "CCCl"),
    ]
    emitted = SimpleNamespace(assay_id="CHEMBL1_nM_IC50", smiles=("CCO", "CCCl"))
    (g, reasons), = ex.align([group], iter([emitted]))
    assert reasons == [ex.KEPT, ex.PCT, ex.REL, ex.PARSE, ex.KEPT]


def test_alignment_marks_unemitted_group_and_fails_on_leftover_loader_assay():
    group = [_row("b", "CCO", chembl="CHEMBL2")]
    leftover = SimpleNamespace(assay_id="CHEMBL9_nM_IC50", smiles=("CCO",))
    with pytest.raises(RuntimeError, match="not aligned"):
        list(ex.align([group], iter([leftover])))
    (g, reasons), = ex.align([group], iter([]))
    assert reasons == [ex.NOT_EMITTED]


def test_identity_reproduces_section_30_keys():
    assert io.identity(PROPRANOLOL)[1] == io.KNOWN_MATCHES[1370]
    assert io.identity(RITONAVIR)[1] == io.KNOWN_MATCHES[88100]
    assert io.identity("not-a-smiles")[2] == "parse_failure"
    assert io.identity("")[2] == "missing_structure"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_gz(path, header, rows):
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def test_end_to_end_counts_and_known_matches(tmp_path: Path):
    export = tmp_path / "export"
    export.mkdir()
    _write_gz(export / "molecules.tsv.gz",
              ["molregno", "tsv_smiles", "chembl32_canonical_smiles", "chembl32_standard_inchi_key_witness", "membership_flags"],
              [[1370, PROPRANOLOL, PROPRANOLOL, "", "train_kept"],
               [88100, RITONAVIR, RITONAVIR, "", "train_kept;validation_kept"],
               [5, "CCO", "CCO", "", "train_kept"],
               [6, "c1ccccc1", "", "", "none_dropped"]])
    _write_gz(export / "membership.tsv.gz",
              ["assay_key", "descriptor", "split", "molregno", "loader_status", "tsv_smiles"],
              [["305983_nM_IC50", "CHEMBL832148_nM_IC50", "train", 1370, "kept", PROPRANOLOL],
               ["621465_nM_IC50", "CHEMBL1100026_nM_IC50", "train", 88100, "kept", RITONAVIR],
               ["x_nM_IC50", "CHEMBLX_nM_IC50", "validation", 88100, "kept", RITONAVIR],
               ["y_nM_IC50", "CHEMBLY_nM_IC50", "train", 5, "kept", "CCO"]])
    (export / "SHA256SUMS.txt").write_text("".join(f"{_sha(export / n)}  {n}\n" for n in ("molecules.tsv.gz", "membership.tsv.gz")))

    audit = tmp_path / "audit.tsv"
    p_inchi, p_key, _ = io.identity(PROPRANOLOL)
    r_inchi, r_key, _ = io.identity(RITONAVIR)
    b_inchi, b_key, _ = io.identity("c1ccccc1")
    with audit.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source_family", "source_partition", "source_id", "identity_state", "standard_inchi", "full_standard_inchikey"])
        w.writerow(["openadmet", "train", "OCNT-0018684", "represented", p_inchi, p_key])
        w.writerow(["openadmet", "validation", "OCNT-0495340", "represented", r_inchi, r_key])
        w.writerow(["openadmet", "internal_test", "OCNT-T", "represented", b_inchi, b_key])
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "rep_01_fold_01.json").write_text(json.dumps({"training_names": ["OCNT-0018684"], "validation_names": ["OCNT-0495340"]}))

    out = tmp_path / "out"
    assert io.main(["--export-dir", str(export), "--audit", str(audit), "--audit-sha256", _sha(audit),
                    "--d007-manifests", str(manifests), "--out-dir", str(out)]) == 0
    s = json.loads((out / "summary.json").read_text())
    dev = s["matched_molecules_by_tier_and_membership"]["development"]
    assert dev["train_kept"] == 2 and dev["validation_kept"] == 1
    assert s["matched_molecules_by_tier_and_membership"]["internal_test"]["train_kept"] == 0
    assert s["matched_rows_by_tier_split_loader"]["development|train|kept"] == 2
    assert s["d007_development_names_seen_in_loader_kept_training_rows"] == 2
    assert s["identity_status_counts"]["source_missing_structure"] == 1
    assert s["source_vs_witness_key_disagreements"] == 0
    with pytest.raises(SystemExit, match="overwrite"):
        io.main(["--export-dir", str(export), "--audit", str(audit), "--audit-sha256", _sha(audit),
                 "--d007-manifests", str(manifests), "--out-dir", str(out)])
