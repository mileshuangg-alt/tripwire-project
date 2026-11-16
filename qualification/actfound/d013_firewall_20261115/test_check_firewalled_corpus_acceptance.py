"""Tests for the D013 Amendment 1 acceptance check. Run in tripwire_cyp002."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_firewalled_corpus_acceptance as ca  # noqa: E402

PROP = "CC(C)NCC(O)COc1cccc2ccccc12"            # protected (development)
PROP_HCL = "CC(C)NCC(O)COc1cccc2ccccc12.Cl"     # largest-fragment match only
S_PROP = "CC(C)NC[C@H](O)COc1cccc2ccccc12"      # first-block only: report-only
ATEN_HCL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1.Cl"  # protected as a salt (internal test)
ATEN = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"         # free base: largest-fragment match
OTHER = "CCO"
BAD = "C1CC"                                    # unparsable (unclosed ring)

AUDIT_COLS = ["source_family", "source_partition", "source_id", "isoform", "assay_chembl_id",
              "molecule_chembl_id", "source_smiles", "identity_state", "failure_reason",
              "canonical_isomeric_smiles_witness", "standard_inchi", "full_standard_inchikey"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _key(smi):
    from rdkit import Chem
    from rdkit.Chem import inchi
    return inchi.InchiToInchiKey(inchi.MolToInchi(Chem.MolFromSmiles(smi)))


def _inputs(tmp: Path, corpus_rows, split, listed=()):
    """corpus_rows: (smiles, assay_chembl_id, molregno)."""
    audit = tmp / "audit.tsv"
    with audit.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(AUDIT_COLS)
        for part, name, smi in (("train", "OCNT-1", PROP), ("internal_test", "OCNT-2", ATEN_HCL)):
            w.writerow(["openadmet", part, name, "", "", "", smi, "represented", "", "", "", _key(smi)])
        w.writerow(["chembl", "x", "C-1", "", "", "", OTHER, "represented", "", "", "", _key(OTHER)])
    corpus = tmp / "corpus.tsv"
    with corpus.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["assay_key"] + [f"c{i}" for i in range(15)])
        for i, (smi, chembl, mr) in enumerate(corpus_rows):
            w.writerow([f"{chembl}_key", str(i + 1), "1", "0", str(mr), "B", "=", "1", "nM", "IC50", "B", "B",
                        chembl, "20", smi, "0"])
    sp = tmp / "split.json"
    sp.write_text(json.dumps(split))
    fw = tmp / "firewall.tsv"
    fw.write_text("molregno\texclusion_basis\n" + "".join(f"{m}\tfull_standard_inchikey\n" for m in listed))
    return ["--corpus", str(corpus), "--corpus-sha256", _sha(corpus), "--audit", str(audit),
            "--audit-sha256", _sha(audit), "--upstream-split", str(sp), "--upstream-split-sha256", _sha(sp),
            "--firewall-list", str(fw), "--firewall-list-sha256", _sha(fw)]


SPLIT = {"train": ["T_nM_IC50"], "valid": ["V_nM_IC50"], "test": ["X_nM_IC50"], "test_external": ["E_nM_IC50"]}


def _run(tmp, rows, listed=(), workers="1"):
    out = tmp / "out"
    rc = ca.main(_inputs(tmp, rows, SPLIT, listed) + ["--out-dir", str(out), "--workers", workers])
    return rc, json.loads((out / "summary.json").read_text()), out


def _tsv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_clean_corpus_passes(tmp_path):
    rc, s, out = _run(tmp_path, [(OTHER, "T", 1), (OTHER, "V", 2)])
    assert rc == 0 and s["gate"] == "PASS" and s["blocking_rows"] == 0
    assert s["rows_by_scope"] == {"train": 1, "valid": 1}
    assert _tsv(out / "matches.tsv") == [] and _tsv(out / "identity_failures.tsv") == []


@pytest.mark.parametrize("smi,basis,scope_chembl", [
    (PROP, "full_standard_inchikey", "T"),
    (PROP_HCL, "largest_fragment_full_inchikey", "T"),
    (ATEN, "largest_fragment_full_inchikey", "V"),
    (ATEN_HCL, "full_standard_inchikey", "V"),
])
def test_protected_match_in_train_or_valid_fails(tmp_path, smi, basis, scope_chembl):
    rc, s, out = _run(tmp_path, [(smi, scope_chembl, 7), (OTHER, "T", 1)])
    scope = "train" if scope_chembl == "T" else "valid"
    assert rc == ca.EXIT_GATE_FAIL and s["gate"] == "FAIL" and s["blocking_rows"] == 1
    assert s["matches_rows_by_scope_and_basis"] == {f"{scope}|{basis}": 1}
    assert [(m["molregno"], m["basis"]) for m in _tsv(out / "matches.tsv")] == [("7", basis)]


@pytest.mark.parametrize("chembl,scope", [("X", "test"), ("E", "test_external"), ("Z", "none")])
def test_protected_match_outside_train_valid_is_reported_not_blocking(tmp_path, chembl, scope):
    rc, s, _ = _run(tmp_path, [(PROP, chembl, 7)])
    assert rc == 0 and s["gate"] == "PASS"
    assert s["matches_rows_by_scope_and_basis"] == {f"{scope}|full_standard_inchikey": 1}


def test_first_block_near_match_is_report_only(tmp_path):
    rc, s, out = _run(tmp_path, [(S_PROP, "T", 3)])
    assert rc == 0 and s["gate"] == "PASS"
    assert s["near_matches_rows_by_scope_and_kind_report_only"] == {"train|first_block": 1}
    assert _tsv(out / "matches.tsv") == []


def test_identity_failure_in_train_fails_closed(tmp_path):
    rc, s, out = _run(tmp_path, [(BAD, "T", 4)])
    assert rc == ca.EXIT_GATE_FAIL and s["blocking_rows"] == 1
    assert s["identity_failures_by_scope_and_status"] == {"train|parse_failure": 1}
    assert [f["molregno"] for f in _tsv(out / "identity_failures.tsv")] == ["4"]


def test_identity_failure_outside_train_valid_is_reported_not_blocking(tmp_path):
    rc, s, _ = _run(tmp_path, [(BAD, "X", 4), ("", "Z", 5)])
    assert rc == 0 and s["gate"] == "PASS"
    assert s["identity_failures_by_scope_and_status"] == {"none|missing_structure": 1, "test|parse_failure": 1}


def test_exclusion_list_molregno_is_reported_not_blocking(tmp_path):
    rc, s, _ = _run(tmp_path, [(OTHER, "T", 9)], listed=["9"])
    assert rc == 0 and s["exclusion_list_molregno_rows_by_scope_report_only"] == {"train": 1}


def test_identity_is_recomputed_from_smiles_not_molregno(tmp_path):
    """A protected structure under a molregno that is not on the list still fails."""
    rc, s, _ = _run(tmp_path, [(PROP, "T", 123456)], listed=["97"])
    assert rc == ca.EXIT_GATE_FAIL and s["exclusion_list_molregno_rows_by_scope_report_only"] == {}


def test_parallel_workers_give_the_same_result(tmp_path):
    rows = [(PROP, "T", 1), (S_PROP, "V", 2), (ATEN, "X", 3), (OTHER, "Z", 4)]
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    _, s1, o1 = _run(a, rows, workers="1")
    _, s2, o2 = _run(b, rows, workers="2")
    for k in ("gate", "blocking_rows", "matches_rows_by_scope_and_basis",
              "near_matches_rows_by_scope_and_kind_report_only"):
        assert s1[k] == s2[k]
    assert (o1 / "matches.tsv").read_bytes() == (o2 / "matches.tsv").read_bytes()


def test_refuses_hash_mismatch_wrong_env_and_existing_out_dir(tmp_path, monkeypatch):
    args = _inputs(tmp_path, [(OTHER, "T", 1)], SPLIT)
    bad = list(args)
    bad[bad.index("--corpus-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        ca.main(bad + ["--out-dir", str(tmp_path / "o1")])
    (tmp_path / "o2").mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ca.main(args + ["--out-dir", str(tmp_path / "o2")])
    monkeypatch.setattr(ca.bf, "env", lambda: {"python": "3.9", "rdkit": "2023.03.1", "inchi": "1.06"})
    with pytest.raises(SystemExit, match="frozen identity environment"):
        ca.main(args + ["--out-dir", str(tmp_path / "o3")])
    assert not (tmp_path / "o1").exists() and not (tmp_path / "o3").exists()


ADJ_COLS = ["activity_id", "molregno", "canonical_smiles", "status", "decision", "evidence"]


def _adj(tmp, entries, decision="Not protected (test)", evidence="keys checked (test)"):
    p = tmp / "adj.tsv"
    with p.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(ADJ_COLS)
        for aid, mr, smi, status in entries:
            w.writerow([aid, mr, smi, status, decision, evidence])
    return ["--adjudications", str(p), "--adjudications-sha256", _sha(p)]


def _run_adj(tmp, rows, entries, **kw):
    out = tmp / "out"
    rc = ca.main(_inputs(tmp, rows, SPLIT) + _adj(tmp, entries, **kw) + ["--out-dir", str(out), "--workers", "1"])
    return rc, json.loads((out / "summary.json").read_text()), out


def test_adjudicated_identity_failure_in_train_passes_and_is_shown(tmp_path):
    rc, s, out = _run_adj(tmp_path, [(BAD, "T", 4), (OTHER, "V", 2)], [("1", "4", BAD, "parse_failure")])
    assert rc == 0 and s["gate"] == "PASS" and s["blocking_rows"] == 0
    assert s["identity_failures_by_scope_and_status"] == {"train|parse_failure": 1}
    assert s["adjudications"]["applied_rows_by_scope_and_status"] == {"train|parse_failure": 1}
    assert s["adjudications"]["listed"] == 1 and s["adjudications"]["unused_blocking"] == []
    assert [f["adjudicated"] for f in _tsv(out / "identity_failures.tsv")] == ["yes"]


@pytest.mark.parametrize("entry", [
    ("2", "4", BAD, "parse_failure"),      # other activity
    ("1", "5", BAD, "parse_failure"),      # other molregno
    ("1", "4", "C1CCC", "parse_failure"),  # other SMILES
    ("1", "4", BAD, "inchi_failure"),      # other status
])
def test_adjudication_must_match_the_row_exactly(tmp_path, entry):
    rc, s, _ = _run_adj(tmp_path, [(BAD, "T", 4)], [entry])
    assert rc == ca.EXIT_GATE_FAIL
    assert s["blocking_rows"] == 2          # the unadjudicated row and the unused adjudication
    assert s["adjudications"]["unused_blocking"] == [list(entry)]


def test_unused_adjudication_fails_the_gate(tmp_path):
    rc, s, _ = _run_adj(tmp_path, [(OTHER, "T", 1)], [("9", "9", BAD, "parse_failure")])
    assert rc == ca.EXIT_GATE_FAIL and s["blocking_rows"] == 1


def test_adjudication_never_clears_a_match(tmp_path):
    rc, s, _ = _run_adj(tmp_path, [(PROP, "T", 7)], [("1", "7", PROP, "ok")])
    assert rc == ca.EXIT_GATE_FAIL
    assert s["matches_rows_by_scope_and_basis"] == {"train|full_standard_inchikey": 1}
    assert s["blocking_rows"] == 2          # the match and the unused adjudication


def test_adjudication_inputs_are_checked(tmp_path):
    args = _inputs(tmp_path, [(BAD, "T", 4)], SPLIT)
    adj = _adj(tmp_path, [("1", "4", BAD, "parse_failure")])
    with pytest.raises(SystemExit, match="go together"):
        ca.main(args + adj[:2] + ["--out-dir", str(tmp_path / "o1")])
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        ca.main(args + adj[:3] + ["0" * 64, "--out-dir", str(tmp_path / "o2")])
    blank = _adj(tmp_path, [("1", "4", BAD, "parse_failure")], evidence="")
    with pytest.raises(SystemExit, match="without decision or evidence"):
        ca.main(args + blank + ["--out-dir", str(tmp_path / "o3")])
