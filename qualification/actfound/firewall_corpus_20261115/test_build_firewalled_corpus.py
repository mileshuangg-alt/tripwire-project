"""Tests for build_firewalled_corpus.py convert/compare (reconcile needs the pinned env)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_firewalled_corpus as b  # noqa: E402


def _authors_row(act, assay, molregno, smi, units="nM", stype="IC50", chembl="CHEMBL9"):
    return [str(act), str(assay), "0", str(molregno), "BAO_1", "=", "10", units, stype, "B", "BAO_2", chembl, "20", smi, "190"]


def _zip(tmp: Path, rows) -> tuple[Path, str, str]:
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    data = buf.getvalue().encode()
    z = tmp / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("chembl_processed_chembl32.csv", data)
    return z, hashlib.sha256(z.read_bytes()).hexdigest(), hashlib.sha256(data).hexdigest()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_convert_sorts_firewalls_and_logs(tmp_path: Path):
    rows = [
        _authors_row(30, 7, 5, "CCO"),
        _authors_row(10, 2, 1370, "CC(C)NCC(O)COc1cccc2ccccc12"),
        _authors_row(20, 2, 4, "CCN"),
    ]
    z, zsha, msha = _zip(tmp_path, rows)
    proh = tmp_path / "prohibited.tsv"
    proh.write_text("molregno\tfull_standard_inchikey\ttiers\topenadmet_names\n1370\tK\tdevelopment\tX\n")
    out = tmp_path / "out"
    assert b.main(["convert", "--authors-zip", str(z), "--authors-zip-sha256", zsha, "--member-sha256", msha,
                   "--prohibited", str(proh), "--prohibited-sha256", _sha(proh), "--out-dir", str(out)]) == 0
    unf = list(csv.reader((out / "authors_pregraph_unfirewalled.tsv").open(), delimiter="\t"))
    fw = list(csv.reader((out / "authors_pregraph_firewalled.tsv").open(), delimiter="\t"))
    assert tuple(unf[0]) == b.AUDIT_COLUMNS and len(unf[1]) == 14
    # sorted by (assay_id, units, type, molregno, activity_id): molregno 4 precedes 1370 within assay 2
    assert [r[0] for r in unf[1:]] == ["20", "10", "30"]
    assert [r[3] for r in fw[1:]] == ["4", "5"]
    s = json.loads((out / "convert_summary.json").read_text())
    assert s["removed_rows"] == 1 and s["prohibited_molregnos_present"] == 1
    with pytest.raises(SystemExit, match="overwrite|exists"):
        b.main(["convert", "--authors-zip", str(z), "--authors-zip-sha256", zsha, "--member-sha256", msha,
                "--prohibited", str(proh), "--prohibited-sha256", _sha(proh), "--out-dir", str(out)])


def test_convert_rejects_wrong_member_hash(tmp_path: Path):
    z, zsha, _ = _zip(tmp_path, [_authors_row(1, 1, 1, "C")])
    proh = tmp_path / "p.tsv"
    proh.write_text("molregno\n")
    with pytest.raises(SystemExit, match="authors CSV SHA-256 mismatch"):
        b.main(["convert", "--authors-zip", str(z), "--authors-zip-sha256", zsha, "--member-sha256", "0" * 64,
                "--prohibited", str(proh), "--prohibited-sha256", _sha(proh), "--out-dir", str(tmp_path / "o")])


def test_compare_reports_field_differences_and_missing(tmp_path: Path):
    conv = tmp_path / "conv.tsv"
    with conv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(b.AUDIT_COLUMNS)
        w.writerow(_authors_row(1, 2, 3, "CCO")[:14])
        w.writerow(_authors_row(2, 2, 4, "CCN")[:14])
    rebuilt = tmp_path / "rebuilt.tsv"
    with rebuilt.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["assay_key", *b.AUDIT_COLUMNS, "graph_pair_count"])
        r = _authors_row(1, 2, 3, "CCO")[:14]
        r[6] = "11"  # standard_value differs
        r[12] = "99"  # count_scratch differences are ignored
        w.writerow(["2_nM_IC50", *r, "5"])
    out = tmp_path / "cmp"
    assert b.main(["compare", "--converted", str(conv), "--converted-sha256", _sha(conv),
                   "--rebuilt-tsv", str(rebuilt), "--rebuilt-sha256", _sha(rebuilt), "--out-dir", str(out)]) == 0
    s = json.loads((out / "compare_summary.json").read_text())
    assert s["shared_activity_ids"] == 1 and s["authors_activity_ids_missing_in_rebuilt"] == 1
    assert s["mismatch_counts_by_field"]["standard_value"] == 1
    assert sum(s["mismatch_counts_by_field"].values()) == 1
