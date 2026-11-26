"""Tests for build_screen_report.py (Tripwire-CYP-011, D027/D028).

Each test is tied to a concrete failure risk: input_id, Molecule_Name and row_id being treated
as interchangeable (the mapping must be the only bridge); an unmatched or unpredicted row
being dropped or filled with another molecule's values; a duplicate key being resolved by
guessing; potency text being altered on its way through; potency and screen columns being
swapped, blended or reordered; a screen/CYP-potency SMILES disagreement passing silently;
hash-unpinned inputs; and output overwrite. Fixtures are synthetic. Run in tripwire_cyp002.
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cyp/scripts"))

import build_screen_report as br  # noqa: E402

MAP_HEADER = ["OCNT_ID", "cyp_molecule_name", "pxr_row_id", "pxr_role", "pxr_eligibility_reason"]


def write(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen_row(i, smiles, status="ok"):
    values = [f"{i}.5{j}" for j in range(len(br.SCREEN_COLUMNS))]
    return [i, f"M{i}", smiles, status, "", ""] + values


SCREEN_HEADER = ["input_row", "input_id", "input_smiles", "status", "invalid_reason", "warnings"] + list(br.SCREEN_COLUMNS)
CYP_HEADER = ["input_row", "input_id", "input_smiles", "status", "invalid_reason", "warnings"] + list(
    br.CYP_POTENCY_COLUMNS) + [c.replace("pred_mean150", "member_sd150") for c in br.CYP_POTENCY_COLUMNS]


def cyp_row(i, smiles, status="ok"):
    mean = [f"{i}.{j}1234567890123" for j in range(4)] if status == "ok" else [""] * 4
    sd = ["0.1"] * 4 if status == "ok" else [""] * 4
    return [i, f"M{i}", smiles, status, "", ""] + mean + sd


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.mapping_rows = [
            ["M0", "M0", "R_A", "primary_validation", "ok"],
            ["M1", "M1", "M1", "aux_only", "aux_only_group_contains_test_structure"],
            ["M2", "M2", "R_C", "primary_validation", "ok"],
        ]
        self.screen_rows = [screen_row(0, "CCO"), screen_row(1, "CCC"), screen_row(2, "CCCC")]
        self.cyp_rows = [cyp_row(0, "CCO"), cyp_row(1, "CCC"), cyp_row(2, "CCCC")]
        self.pxr_rows = [["R_A", "4.1000000000001", "5.2"], ["M1", "4.3", "5.4"], ["R_C", "4.5", "5.6"]]

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, out="out"):
        t = self.t
        shas = {
            "screen": write(t / "s.csv", SCREEN_HEADER, self.screen_rows),
            "map": write(t / "m.csv", MAP_HEADER, self.mapping_rows),
            "cyp": write(t / "c.csv", CYP_HEADER, self.cyp_rows),
            "pxr": write(t / "p.csv", ["row_id", br.PXR_PRIMARY, br.PXR_SECONDARY], self.pxr_rows),
        }
        return SimpleNamespace(
            screen_predictions=str(t / "s.csv"), screen_predictions_sha256=shas["screen"],
            mapping=str(t / "m.csv"), mapping_sha256=shas["map"],
            cyp_potency=str(t / "c.csv"), cyp_potency_sha256=shas["cyp"],
            pxr_potency=str(t / "p.csv"), pxr_potency_sha256=shas["pxr"], out_dir=str(t / out))


class JoinBehaviour(Fixture):
    def test_full_join_keeps_values_as_text_and_orders_columns(self):
        args = self.args()
        counts = br.run(args)
        rows = list(csv.DictReader(open(Path(args.out_dir) / br.OUT_REPORT)))
        self.assertEqual(len(rows), 3)
        self.assertEqual(list(rows[0].keys()), list(br.REPORT_COLUMNS))
        self.assertEqual(rows[0][br.CYP_POTENCY_COLUMNS[0]], "0.01234567890123")
        self.assertEqual(rows[0][br.PXR_PRIMARY], "4.1000000000001")
        self.assertEqual(rows[0][br.SCREEN_COLUMNS[0]], "0.50")
        self.assertEqual({r["join_status_cyp"] for r in rows}, {br.MATCHED})
        self.assertEqual(counts["report_rows"], 3)
        potency_end = br.REPORT_COLUMNS.index(br.PXR_SECONDARY)
        self.assertLess(potency_end, br.REPORT_COLUMNS.index(br.SCREEN_COLUMNS[0]))

    def test_pxr_potency_is_reached_through_pxr_row_id_not_input_id(self):
        # M0's PXR row id is R_A; a potency row keyed by the CYP/input id must not be used.
        self.pxr_rows[0][0] = "M0"
        counts = br.run(self.args())
        rows = list(csv.DictReader(open(self.t / "out" / br.OUT_REPORT)))
        self.assertEqual(rows[0]["join_status_pxr"], br.NO_ROW)
        self.assertEqual(rows[0][br.PXR_PRIMARY], "")
        self.assertEqual(counts["pxr_potency_ids_not_in_mapping"], ["M0"])

    def test_unmatched_and_unpredicted_rows_are_kept_with_status(self):
        self.screen_rows.append(screen_row(9, "CCCCC"))          # no mapping row
        self.cyp_rows[1] = cyp_row(1, "CCC", status="invalid_structure")
        self.pxr_rows.pop(2)                                     # no PXR potency for R_C
        counts = br.run(self.args())
        rows = list(csv.DictReader(open(self.t / "out" / br.OUT_REPORT)))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[3]["join_status_cyp"], "no_mapping_row")
        self.assertEqual(rows[3][br.SCREEN_COLUMNS[0]], "9.50")
        self.assertEqual(rows[1]["join_status_cyp"], "potency_not_predicted:invalid_structure")
        self.assertEqual(rows[1][br.CYP_POTENCY_COLUMNS[0]], "")
        self.assertEqual(rows[2]["join_status_pxr"], br.NO_ROW)
        self.assertEqual(counts["screen_ids_not_in_mapping"], ["M9"])
        self.assertEqual(counts["join_status_pxr_counts"][br.NO_ROW], 1)  # R_C only
        self.assertEqual(counts["join_status_pxr_counts"]["no_mapping_row"], 1)  # the unmapped screen row

    def test_mapping_rows_without_screen_row_and_orphan_potency_rows_are_counted(self):
        self.screen_rows.pop(1)
        self.cyp_rows.append(cyp_row(7, "CC"))
        counts = br.run(self.args())
        self.assertEqual(counts["mapping_rows_without_screen_row"], ["M1"])
        self.assertEqual(counts["cyp_potency_ids_not_in_mapping"], ["M7"])

    def test_cyp_name_differing_from_ocnt_is_reported_and_used_for_the_cyp_lookup(self):
        self.mapping_rows[0][1] = "ALT0"
        self.cyp_rows[0][1] = "ALT0"
        counts = br.run(self.args())
        self.assertEqual(counts["mapping_cyp_name_differs_from_ocnt"], ["M0"])
        rows = list(csv.DictReader(open(self.t / "out" / br.OUT_REPORT)))
        self.assertEqual(rows[0]["join_status_cyp"], br.MATCHED)


class StopsAndGuards(Fixture):
    def test_duplicate_keys_stop_with_counts_and_no_report(self):
        self.mapping_rows.append(["M1", "M1", "R_Z", "aux_only", "ok"])
        args = self.args()
        with self.assertRaises(br.JoinStop):
            br.run(args)
        out = Path(args.out_dir)
        payload = json.loads((out / br.OUT_JOIN).read_text())
        self.assertEqual(payload["counts"]["duplicate_key_counts"]["mapping.OCNT_ID"], 1)
        self.assertFalse((out / br.OUT_REPORT).exists())

    def test_duplicate_in_each_potency_table_stops(self):
        self.pxr_rows.append(["R_A", "9", "9"])
        with self.assertRaises(br.JoinStop):
            br.run(self.args())

    def test_screen_and_cyp_smiles_disagreement_stops(self):
        self.cyp_rows[2][2] = "CCCCC"
        args = self.args()
        with self.assertRaises(br.JoinStop):
            br.run(args)
        payload = json.loads((Path(args.out_dir) / br.OUT_JOIN).read_text())
        self.assertEqual(payload["counts"]["cyp_smiles_disagree_with_screen"], ["M2"])

    def test_hash_mismatch_missing_column_and_existing_output_stop(self):
        args = self.args()
        args.mapping_sha256 = "0" * 64
        with self.assertRaises(br.JoinStop):
            br.run(args)
        args = self.args("out2")
        write(self.t / "p.csv", ["row_id", br.PXR_PRIMARY], [["R_A", "1"]])
        args.pxr_potency_sha256 = hashlib.sha256((self.t / "p.csv").read_bytes()).hexdigest()
        with self.assertRaises(br.JoinStop):
            br.run(args)
        args = self.args("out3")
        Path(args.out_dir).mkdir()
        with self.assertRaises(FileExistsError):
            br.run(args)

    def test_no_combined_score_or_flag_columns(self):
        for name in br.REPORT_COLUMNS:
            for banned in ("score", "flag", "queue", "cutoff", "spread", "trigger", "combined"):
                self.assertNotIn(banned, name.lower())


if __name__ == "__main__":
    unittest.main()
