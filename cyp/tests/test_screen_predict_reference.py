"""Tests for screen_predict_reference.py (Tripwire-CYP-010, D025).

Each test is tied to a concrete failure risk: comparing against smoke results that were not a
PASS or that cover different molecules, looking up the wrong SMILES or silently skipping a
missing id, reusing an output path, passing a comparison whose difference exceeds the declared
tolerance, and a mean that is not the arithmetic mean of the three retained seeds.
Writing these tests is not a passing run.
"""
import csv
import json
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cyp/scripts"))
import screen_predict_reference as ref  # noqa: E402

IDS = ["OADMET-0000001", "OADMET-0000002", "OCNT-0000003"]
SMILES = {"OADMET-0000001": "CCO", "OADMET-0000002": "CCN", "OCNT-0000003": "CCC"}
SAVED = {20261125: [0.5, 0.6, 0.7], 20261126: [0.55, 0.65, 0.75], 20261127: [0.45, 0.55, 0.65]}


def write_smoke(root, status="PASS", ids=None, overrides=None):
    for seed in ref.SEEDS:
        directory = Path(root) / f"pxr_seed{seed}"
        directory.mkdir(parents=True)
        molecules = list((overrides or {}).get(seed, ids or IDS))
        payload = {
            "status": status if seed != 20261126 or status == "PASS" else "PASS",
            "seed": seed, "molecules": molecules,
            "predictions": [[value, 9.0, 9.0] for value in SAVED[seed][: len(molecules)]],
        }
        (directory / "result.json").write_text(json.dumps(payload))


def write_predictions(path, shift=0.0, status="ok", mean_shift=0.0):
    columns = ["input_id", "status", *ref.SEED_COLUMNS.values(), ref.MEAN_COLUMN]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for i, name in enumerate(IDS):
            values = [SAVED[seed][i] + (shift if seed == 20261125 else 0.0) for seed in ref.SEEDS]
            writer.writerow([name, status, *values, sum(values) / 3 + mean_shift])


class ReferenceTests(unittest.TestCase):
    def test_smoke_results_must_be_pass_with_identical_molecules(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_smoke(Path(tmp) / "good")
            loaded = ref.load_smoke_results(Path(tmp) / "good")
            self.assertEqual(loaded[20261125]["molecules"], IDS)
            write_smoke(Path(tmp) / "bad", status="FAIL")
            (Path(tmp) / "bad" / "pxr_seed20261125" / "result.json").write_text(
                json.dumps({"status": "FAIL", "seed": 20261125, "molecules": IDS, "predictions": [[0, 0, 0]] * 3}))
            with self.assertRaises(ValueError):
                ref.load_smoke_results(Path(tmp) / "bad")
            write_smoke(Path(tmp) / "diff", overrides={20261127: ["A", "B", "C"]})
            with self.assertRaises(ValueError):
                ref.load_smoke_results(Path(tmp) / "diff")

    def test_lookup_returns_exactly_the_requested_ids_and_rejects_missing_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "frame.csv"
            frame.write_text("row_id,role,SMILES\nOADMET-0000001,primary_train,CCO\nOADMET-0000002,primary_train,CCN\n"
                             "OCNT-0000003,aux_only,CCC\nOCNT-9999999,aux_only,CCCC\n")
            found = ref.lookup_smiles(frame, IDS)
            self.assertEqual(found, SMILES)
            with self.assertRaises(ValueError):
                ref.lookup_smiles(frame, IDS + ["OCNT-0000404"])

    def test_build_reference_input_writes_only_the_smoke_molecules_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_smoke(Path(tmp) / "results")
            frame = Path(tmp) / "frame.csv"
            frame.write_text("row_id,SMILES\nOADMET-0000001,CCO\nOADMET-0000002,CCN\nOCNT-0000003,CCC\nOCNT-9,CCCC\n")
            args = type("A", (), {})()
            args.smoke_results_dir, args.frame = Path(tmp) / "results", frame
            args.frame_sha256, args.out = ref.sha256_file(frame), Path(tmp) / "ref.csv"
            summary = ref.build_reference_input(args)
            self.assertEqual(summary["rows"], 3)
            with open(args.out, newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["row_id", "SMILES"])
            self.assertEqual([r[0] for r in rows[1:]], IDS)
            with self.assertRaises(FileExistsError):
                ref.build_reference_input(args)
            args.frame_sha256 = "0" * 64
            args.out = Path(tmp) / "ref2.csv"
            with self.assertRaises(ValueError):
                ref.build_reference_input(args)

    def test_compare_passes_within_tolerance_and_fails_beyond_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_smoke(Path(tmp) / "results")
            predictions = Path(tmp) / "p.csv"
            write_predictions(predictions, shift=1e-7)
            self.assertEqual(ref.compare(predictions, Path(tmp) / "results", 1e-5)["status"], "PASS")
            write_predictions(predictions, shift=1e-3)
            self.assertEqual(ref.compare(predictions, Path(tmp) / "results", 1e-5)["status"], "FAIL")

    def test_compare_flags_a_mean_that_is_not_the_arithmetic_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_smoke(Path(tmp) / "results")
            predictions = Path(tmp) / "p.csv"
            write_predictions(predictions, mean_shift=0.01)
            self.assertEqual(ref.compare(predictions, Path(tmp) / "results", 1e-5)["status"], "FAIL")

    def test_compare_rejects_invalid_rows_wrong_ids_and_bad_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_smoke(Path(tmp) / "results")
            predictions = Path(tmp) / "p.csv"
            write_predictions(predictions, status="invalid_structure")
            with self.assertRaises(ValueError):
                ref.compare(predictions, Path(tmp) / "results", 1e-5)
            write_predictions(predictions)
            with self.assertRaises(ValueError):
                ref.compare(predictions, Path(tmp) / "results", 0.0)
            text = predictions.read_text().replace("OCNT-0000003", "OCNT-0000099")
            predictions.write_text(text)
            with self.assertRaises(ValueError):
                ref.compare(predictions, Path(tmp) / "results", 1e-5)


if __name__ == "__main__":
    unittest.main()
