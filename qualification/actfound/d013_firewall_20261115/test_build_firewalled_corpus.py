"""Tests for the route-C corpus builder (local RDKit; production runs in the pinned env on CoreHPC)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
# On CoreHPC the graph-stage code lives outside the project tree; point at it with this variable.
GRAPH_SRC = Path(os.environ.get("ACTFOUND_GRAPH_CODE_DIR", str(ROOT / "cyp" / "scripts")))
sys.path.insert(0, str(HERE))
import build_firewalled_corpus as bc  # noqa: E402

pytest.importorskip("rdkit")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _code_dir(tmp: Path) -> Path:
    d = tmp / "code"
    d.mkdir()
    for n in bc.GRAPH_CODE:
        shutil.copy(GRAPH_SRC / n, d / n)
    return d


def _pregraph(tmp: Path) -> Path:
    sys.path.insert(0, str(GRAPH_SRC))
    from actfound_chembl32_projection import AUDIT_COLUMNS
    p = tmp / "pregraph.tsv"
    with p.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(AUDIT_COLUMNS)
        aid = 0
        for assay in ("101", "102"):
            for i in range(30):  # a homologous series: neighbours are similar
                aid += 1
                smi = "c1ccccc1" + "C" * (i % 15 + 1) + ("O" if i >= 15 else "N")
                w.writerow([str(aid), assay, "0", str(1000 + aid), "BAO_0000190", "=", f"{10 ** (i / 10):.4g}",
                            "nM", "IC50", "B", "BAO_0000357", f"CHEMBL{assay}", "0", smi])
    return p


def _run(tmp: Path, *extra):
    pre = _pregraph(tmp)
    code = _code_dir(tmp)
    return pre, code, ["--pregraph", str(pre), "--pregraph-sha256", _sha(pre), "--graph-code-dir", str(code), *extra]


def _split(tmp: Path, train, valid) -> list:
    sp = tmp / "split.json"
    sp.write_text(json.dumps({"train": train, "valid": valid, "test": [], "test_external": []}))
    return ["--upstream-split", str(sp), "--upstream-split-sha256", _sha(sp)]


def _authors_from_reference(tmp: Path, code: Path, pre: Path, edit=None) -> Path:
    """An 'authors' file built by the same graph stage; `edit(rows)` perturbs it."""
    sys.path.insert(0, str(code))
    import actfound_graph_stage as gs
    ref = tmp / "ref.tsv"
    gs.process_shard(pre, ref, tmp / "ref_summary.json", 1000)
    with ref.open(newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        rows = [row[1:] for row in r]
    assert {x[11] for x in rows} == {"CHEMBL101", "CHEMBL102"}, "fixture must qualify both assays"
    if edit:
        edit(rows)
    authors = tmp / "authors.csv"
    with authors.open("w", newline="") as o:
        csv.writer(o).writerows(rows)
    return authors


def _repro(tmp: Path, edit=None, split=(["CHEMBL101_nM_IC50"], ["CHEMBL102_nM_IC50"])):
    pre, code, base = _run(tmp)
    authors = _authors_from_reference(tmp, code, pre, edit)
    out = tmp / "out"
    rc = bc.main(base + ["--authors-csv", str(authors), "--authors-csv-sha256", _sha(authors), "--out-dir", str(out)]
                 + _split(tmp, *split))
    m = json.loads((out / "corpus_manifest.json").read_text())
    return rc, m, out, base, authors


def _diffs(out: Path):
    with (out / "reproduction_differences.tsv").open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_reproduction_mode_matches_authors_built_by_same_pipeline(tmp_path: Path):
    rc, m, out, _, authors = _repro(tmp_path)
    n = sum(1 for _ in authors.open())
    assert rc == 0 and m["openadmet_firewall_applied"] is False
    g = m["reproduction_vs_authors"]
    assert g["gate"] == "PASS" and g["blocking_differences"] == 0
    assert g["in_split"] == {"identical_rows": n} and g["outside_split"] == {}
    assert _diffs(out) == []


def _reformat_first(chembl):
    def edit(rows):  # formatting-only change to one value: "1" -> "1.0" style
        r = next(x for x in rows if x[11] == chembl)
        r[6] = r[6] + ".0" if "." not in r[6] else r[6] + "0"
    return edit


def test_formatting_only_difference_inside_split_blocks(tmp_path: Path):
    rc, m, out, _, _ = _repro(tmp_path, edit=_reformat_first("CHEMBL101"))
    g = m["reproduction_vs_authors"]
    assert rc == bc.EXIT_GATE_FAIL and g["gate"] == "FAIL" and g["in_split"]["field_mismatch"] == 1
    d = _diffs(out)
    assert [(x["scope"], x["kind"]) for x in d] == [("in_split", "field_mismatch")]
    assert (out / "SHA256SUMS.txt").exists()  # outputs kept for review


@pytest.mark.parametrize("edit,kind", [
    (lambda rows: rows.remove(next(x for x in rows if x[11] == "CHEMBL102")), "extra_in_rebuilt"),
    (lambda rows: rows.append(["999999"] + next(x for x in rows if x[11] == "CHEMBL102")[1:]), "missing_in_rebuilt"),
])
def test_missing_or_extra_row_inside_split_blocks(tmp_path: Path, edit, kind):
    rc, m, out, _, _ = _repro(tmp_path, edit=edit)
    assert rc == bc.EXIT_GATE_FAIL and m["reproduction_vs_authors"]["in_split"][kind] == 1


def test_assay_missing_from_rebuilt_inside_split_blocks(tmp_path: Path):
    def edit(rows):  # authors have an extra in-split assay the rebuild never produces
        rows.append(["888888", "103"] + rows[0][2:7] + ["nM", "IC50"] + rows[0][9:11] + ["CHEMBL103"] + rows[0][12:])
    rc, m, _, _, _ = _repro(tmp_path, edit=edit,
                            split=(["CHEMBL101_nM_IC50", "CHEMBL103_nM_IC50"], ["CHEMBL102_nM_IC50"]))
    assert rc == bc.EXIT_GATE_FAIL and m["reproduction_vs_authors"]["in_split"]["missing_in_rebuilt"] == 1


def test_difference_outside_split_is_reported_not_blocking(tmp_path: Path):
    rc, m, out, _, _ = _repro(tmp_path, edit=_reformat_first("CHEMBL102"),
                              split=(["CHEMBL101_nM_IC50"], ["CHEMBL_OTHER_nM_IC50"]))
    g = m["reproduction_vs_authors"]
    assert rc == 0 and g["gate"] == "PASS" and g["outside_split"]["field_mismatch"] == 1
    assert g["split_ids_absent_from_authors_file"] == 1
    assert [(x["scope"], x["kind"]) for x in _diffs(out)] == [("outside_split", "field_mismatch")]


def test_reproduction_fails_closed_on_missing_or_mismatched_split(tmp_path: Path):
    pre, code, base = _run(tmp_path)
    args = base + ["--authors-csv", str(pre), "--authors-csv-sha256", _sha(pre)]
    with pytest.raises(SystemExit, match="requires --upstream-split"):
        bc.main(args + ["--out-dir", str(tmp_path / "o1")])
    sp = _split(tmp_path, ["A"], ["B"])
    with pytest.raises(SystemExit, match="missing or SHA-256 mismatch"):
        bc.main(args + ["--out-dir", str(tmp_path / "o2"), sp[0], sp[1], sp[2], "0" * 64])
    with pytest.raises(SystemExit, match="missing or SHA-256 mismatch"):
        bc.main(args + ["--out-dir", str(tmp_path / "o3"), sp[0], str(tmp_path / "nope.json"), sp[2], sp[3]])
    assert not any((tmp_path / d).exists() for d in ("o1", "o2", "o3"))


def _firewall_inputs(tmp: Path):
    authors = tmp / "authors.csv"
    with authors.open("w", newline="") as o:  # assay universe: only assay 101
        csv.writer(o).writerow(["1", "101", "0", "1001", "B", "=", "1", "nM", "IC50", "B", "B", "CHEMBL101", "0", "C", "0"])
    fw = tmp / "firewall.tsv"
    fw.write_text("molregno\texclusion_basis\tfull_standard_inchikey\tmatched_key\ttiers\topenadmet_names\n"
                  "1001\tlargest_fragment_full_inchikey\tK\tK2\tdevelopment\tOCNT-X\n")
    return authors, fw


def _fake_repro_manifest(tmp: Path, base, authors, gate="PASS") -> list:
    m = {"artifact_type": bc.REPRODUCTION_ARTIFACT, "reproduction_vs_authors": {"gate": gate},
         "inputs": {"pregraph_sha256": base[3], "authors_csv_sha256": _sha(authors), "graph_code_sha256": bc.GRAPH_CODE,
                    "script_sha256": _sha(Path(bc.__file__)), "upstream_split_sha256": "s" * 64}}
    p = tmp / f"repro_{gate}.json"
    p.write_text(json.dumps(m))
    return ["--reproduction-manifest", str(p), "--reproduction-manifest-sha256", _sha(p)]


def test_firewall_removes_rows_before_graph_and_marks_manifest(tmp_path: Path):
    pre, code, base = _run(tmp_path)
    authors, fw = _firewall_inputs(tmp_path)
    out = tmp_path / "out"
    assert bc.main(base + ["--authors-csv", str(authors), "--authors-csv-sha256", _sha(authors),
                           "--firewall-list", str(fw), "--firewall-list-sha256", _sha(fw), "--out-dir", str(out)]
                   + _fake_repro_manifest(tmp_path, base, authors)) == 0
    m = json.loads((out / "corpus_manifest.json").read_text())
    assert m["openadmet_firewall_applied"] is True
    assert m["filter"]["rows_removed_by_firewall"] == 1 and m["filter"]["rows_outside_authors_assays"] == 30
    assert m["filter"]["rows_removed_by_exclusion_basis"] == {"largest_fragment_full_inchikey": 1}
    assert m["firewall_molregnos_by_exclusion_basis"] == {"largest_fragment_full_inchikey": 1}
    with (out / "actfound_chembl32_d013_firewalled.tsv").open(newline="") as f:
        assert all(r[4] != "1001" for r in csv.reader(f, delimiter="\t"))
    assert m["output_sha256"] == _sha(out / "actfound_chembl32_d013_firewalled.tsv")


def test_firewall_mode_blocked_without_passing_reproduction(tmp_path: Path):
    pre, code, base = _run(tmp_path)
    authors, fw = _firewall_inputs(tmp_path)
    args = base + ["--authors-csv", str(authors), "--authors-csv-sha256", _sha(authors),
                   "--firewall-list", str(fw), "--firewall-list-sha256", _sha(fw)]
    with pytest.raises(SystemExit, match="requires a PASS reproduction manifest"):
        bc.main(args + ["--out-dir", str(tmp_path / "o1")])
    with pytest.raises(SystemExit, match="did not PASS"):
        bc.main(args + ["--out-dir", str(tmp_path / "o2")] + _fake_repro_manifest(tmp_path, base, authors, "FAIL"))
    other = tmp_path / "other_authors.csv"
    other.write_text(authors.read_text() + "\n")
    with pytest.raises(SystemExit, match="different authors_csv_sha256"):
        bc.main(args + ["--out-dir", str(tmp_path / "o3")] + _fake_repro_manifest(tmp_path, base, other))
    assert not any((tmp_path / d).exists() for d in ("o1", "o2", "o3"))


def test_refuses_changed_graph_code(tmp_path: Path):
    pre, code, base = _run(tmp_path)
    (code / "actfound_graph_stage.py").write_text("# changed\n")
    with pytest.raises(SystemExit, match="graph code differs"):
        bc.main(base + ["--authors-csv", str(pre), "--authors-csv-sha256", _sha(pre), "--out-dir", str(tmp_path / "o")])


def test_firewall_mode_refuses_list_without_exclusion_basis(tmp_path: Path):
    pre, code, base = _run(tmp_path)
    authors, _ = _firewall_inputs(tmp_path)
    v1 = tmp_path / "v1_list.tsv"
    v1.write_text("molregno\tfull_standard_inchikey\ttiers\topenadmet_names\n1001\tK\tdevelopment\tOCNT-X\n")
    with pytest.raises(SystemExit, match="exclusion_basis"):
        bc.main(base + ["--authors-csv", str(authors), "--authors-csv-sha256", _sha(authors),
                        "--firewall-list", str(v1), "--firewall-list-sha256", _sha(v1), "--out-dir", str(tmp_path / "o")]
                + _fake_repro_manifest(tmp_path, base, authors))
