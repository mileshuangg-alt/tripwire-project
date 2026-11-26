"""Tests for evaluate_screen_predictions.py (Tripwire-CYP-010, D026).

Each test is tied to a concrete failure risk: the 11 set-aside molecules being scored or having
their labels looked up, a prediction file that does not cover exactly the mapping ids, the
wrong label being joined (ids, enzymes, concentration), a baseline constant that uses evaluation
labels or a non-training molecule, a Spearman or MAE computed wrongly (ties, constants, missing
values), the recorded counts not being enforced, a changed input being read, and an output
directory being reused. Writing these tests is not a passing run.
"""
import csv
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cyp/scripts"))
import evaluate_screen_predictions as ev  # noqa: E402

MAP_HEADER = ["OCNT_ID", "cyp_molecule_name", "pxr_row_id", "pxr_role", "pxr_eligibility_reason", "cyp_tiers",
              "cyp_smiles", "pxr_source_smiles", "pxr_smiles_source", "smiles_string_equal"]
OFFSETS = {"CYP1A2": 0.0, "CYP2C9": 0.2, "CYP2D6": -0.3, "CYP3A4": 0.5}
SYN_EXPECTED = {
    "rows": 6, "set_aside": 1, "evaluated": 5, "roles": {"primary_validation": 3, "aux_only": 2},
    "tier_role": {("train", "primary_validation"): 2, ("validation", "primary_validation"): 1, ("train", "aux_only"): 2},
    "cyp_train_molecules": 4,
}
SPEC = [  # id, role, tier, reason, smiles
    ("OCNT-1", "primary_validation", "train", "primary_validation_aux_excluded", "CCO"),
    ("OCNT-2", "primary_validation", "train", "primary_validation_aux_excluded", "CCN"),
    ("OCNT-3", "primary_validation", "validation", "primary_validation_aux_excluded", "CCC"),
    ("OCNT-4", "aux_only", "train", "aux_only_group_contains_validation_or_holdout", "CCCC"),
    ("OCNT-5", "aux_only", "train", "aux_only_group_contains_validation_or_holdout", "CCCCC"),
    ("OCNT-6", "aux_only", "train", "aux_only_group_contains_test_structure", "CO"),
]


def label(i, enzyme):
    return 0.1 * i + OFFSETS[enzyme]


def mapping_rows(spec=SPEC):
    rows = []
    for i, (ocnt, role, tier, reason, smiles) in enumerate(spec, start=1):
        row_id = ocnt if role == "aux_only" else f"OADMET-{i}"
        rows.append(dict(zip(MAP_HEADER, [ocnt, ocnt, row_id, role, reason, tier, smiles, smiles,
                                          "TRAIN.csv" if role != "aux_only" else "single_concentration", "True"])))
    return rows


def prediction_rows(rows=None):
    out = []
    for i, m in enumerate(rows or mapping_rows(), start=1):
        row = {"input_id": m["OCNT_ID"], "input_smiles": m["cyp_smiles"], "status": "ok"}
        for e in ev.ENZYMES:
            row[ev.CYP_OUTPUTS[e]] = repr(label(i, e) + (0.1 if e == "CYP1A2" else -0.2))
        row[ev.PXR_OUTPUT] = repr(0.3 * i + 0.05)
        out.append(row)
    return out


def write_csv(path, header, rows, newline="\n"):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator=newline)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r[h] for h in header] if isinstance(r, dict) else r)


class Workspace:
    """A synthetic, self-consistent set of input files."""

    def __init__(self, tmp, split_extra=None):
        t = Path(tmp)
        self.maprows = mapping_rows()
        self.preds = prediction_rows(self.maprows)
        write_csv(t / "mapping.csv", MAP_HEADER, self.maprows)
        pred_header = ["input_id", "input_smiles", "status"] + [ev.CYP_OUTPUTS[e] for e in ev.ENZYMES] + [ev.PXR_OUTPUT]
        write_csv(t / "pred.csv", pred_header, self.preds)
        screen = [["Molecule_Name", "SMILES", "OCNT_Batch", "enzyme", "plate_id", "concentration_M", "log2fc_estimate"]]
        for i, (ocnt, *_rest) in enumerate(SPEC, start=1):
            for e in ev.ENZYMES:
                screen.append([ocnt, "x", "b", e, "p", "4.9e-05", repr(label(i, e))])
        for k in range(1, 5):  # screening-training molecules T1..T4; labels 0,1,2,3 -> mean 1.5 for every enzyme
            for e in ev.ENZYMES:
                screen.append([f"T{k}", "x", "b", e, "p", "4.9e-05", repr(float(k - 1))])
        for e in ev.ENZYMES:  # a validation-partition molecule must not enter the constants
            screen.append(["V1", "x", "b", e, "p", "4.9e-05", "100.0"])
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerows(screen)
        self.screen_bytes = buffer.getvalue().encode()
        with tarfile.open(t / "screen.tar.gz", "w:gz") as archive:
            info = tarfile.TarInfo(ev.SCREEN_MEMBER)
            info.size = len(self.screen_bytes)
            archive.addfile(info, io.BytesIO(self.screen_bytes))
        single = [["Molecule Name", "OCNT Batch", "SMILES", "plate_id", "compound_class", "concentration_M",
                   "log2_fc_estimate", "OCNT_ID"]]
        for i, (ocnt, *_rest) in enumerate(SPEC, start=1):
            single.append(["", "b", "x", "p", "Library", "8.251e-06", repr(0.3 * i), ocnt])
        single.append(["", "b", "x", "p", "Library", "3.3e-05", "9.9", "OCNT-1"])
        single.append(["", "b", "x", "p", "Positive Control", "9.803e-07", "7.7", "OCNT-2"])
        write_csv(t / "single.csv", single[0], single[1:])
        (t / "pins.json").write_text(json.dumps({"expected": {
            "head_order": ["log2fc_8p25uM", "log2fc_33uM", "log2fc_99uM"], "scaler_mean": [0.25, 0.5, 1.0]}}))
        split = [["molecule_id", "cluster_id", "partition"]] + [[f"T{k}", str(k), "train"] for k in range(1, 5)] \
            + [["V1", "9", "validation"]] + (split_extra or [])
        write_csv(t / "split.csv", split[0], split[1:], newline="\r\n")
        self.t = t

    def args(self, out="out", **overrides):
        t = self.t
        values = dict(
            predictions=t / "pred.csv", predictions_sha256=ev.sha256_file(t / "pred.csv"),
            mapping=t / "mapping.csv", mapping_sha256=ev.sha256_file(t / "mapping.csv"),
            screen_tarball=t / "screen.tar.gz", screen_sha256=ev.sha256_bytes(self.screen_bytes),
            pxr_single=t / "single.csv", pxr_single_sha256=ev.sha256_file(t / "single.csv"),
            pxr_pins=t / "pins.json", pxr_pins_sha256=ev.sha256_file(t / "pins.json"),
            cyp_train_split=t / "split.csv", cyp_train_split_sha256=ev.sha256_file(t / "split.csv"),
            out_dir=t / out)
        values.update(overrides)
        return SimpleNamespace(**values)


class MetricTests(unittest.TestCase):
    def test_average_ranks_and_spearman_against_scipy(self):
        from scipy.stats import spearmanr
        for x, y in (([1, 2, 3, 4], [10, 20, 30, 40]), ([1, 2, 3, 4], [4, 3, 2, 1]),
                     ([1, 2, 2, 3, 5], [2, 1, 4, 3, 9]), ([0.5, 0.1, 0.9, 0.3], [3, 1, 2, 2])):
            self.assertAlmostEqual(ev.spearman(x, y), float(spearmanr(x, y).statistic), places=12)
        self.assertEqual(ev.average_ranks([5, 1, 5, 2]), [3.5, 1.0, 3.5, 2.0])

    def test_spearman_undefined_cases_return_none(self):
        self.assertIsNone(ev.spearman([1, 2], [1, 2]))
        self.assertIsNone(ev.spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(ev.spearman([1, 2, 3], [4, 4, 4]))

    def test_mae_baseline_and_ratio(self):
        m = ev.metrics_for([1.0, 2.0, 4.0], [1.5, 2.0, 3.0], 2.0)
        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["mae"], (0.5 + 0.0 + 1.0) / 3)
        self.assertAlmostEqual(m["baseline_mae"], (0.5 + 0.0 + 1.0) / 3)
        self.assertAlmostEqual(m["mae_over_baseline"], 1.0)
        self.assertEqual(m["baseline_constant"], 2.0)

    def test_missing_values_are_dropped_from_n_and_empty_input_is_handled(self):
        m = ev.metrics_for([1.0, None, 3.0], [1.0, 2.0, None], 0.0)
        self.assertEqual(m["n"], 1)
        e = ev.metrics_for([None], [1.0], 0.0)
        self.assertEqual((e["n"], e["mae"], e["spearman"]), (0, None, None))
        zero = ev.metrics_for([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 2.0)
        self.assertIsNone(zero["mae_over_baseline"])


class CrosswalkTests(unittest.TestCase):
    def test_consistent_inputs_pass(self):
        info = ev.crosswalk_checks(prediction_rows(), mapping_rows())
        self.assertEqual(info["rows"], 6)

    def test_each_failure_stops(self):
        cases = {}
        p = prediction_rows(); p[0]["input_id"] = "OCNT-X"; cases["id set differs"] = (p, mapping_rows())
        p = prediction_rows(); p.append(dict(p[0])); cases["duplicate prediction id"] = (p, mapping_rows())
        p = prediction_rows(); p[1]["status"] = "invalid_structure"; cases["status not ok"] = (p, mapping_rows())
        p = prediction_rows(); p[2]["input_smiles"] = "CCCCCC"; cases["smiles differ"] = (p, mapping_rows())
        m = mapping_rows(); m[1]["pxr_row_id"] = m[0]["pxr_row_id"]; cases["duplicate row_id"] = (prediction_rows(), m)
        m = mapping_rows(); m[0]["cyp_molecule_name"] = "OTHER"; cases["cyp name differs"] = (prediction_rows(), m)
        m = mapping_rows(); m[3]["pxr_row_id"] = "OADMET-9"; cases["aux row_id differs"] = (prediction_rows(), m)
        m = mapping_rows(); m[0]["pxr_row_id"] = m[0]["OCNT_ID"]; cases["primary row_id equals ocnt"] = (prediction_rows(), m)
        m = mapping_rows(); m[0]["smiles_string_equal"] = "False"; cases["flag disagrees"] = (prediction_rows(), m)
        m = mapping_rows(); m[0]["pxr_source_smiles"] = "CCCCCCC"; m[0]["smiles_string_equal"] = "False"; cases["structure differs"] = (prediction_rows(), m)
        for name, (pred, mapping) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                ev.crosswalk_checks(pred, mapping)


class SelectionTests(unittest.TestCase):
    def test_set_aside_and_recorded_counts(self):
        result = ev.split_evaluation_set(mapping_rows(), SYN_EXPECTED)
        self.assertEqual(result["set_aside_count"], 1)
        self.assertNotIn("OCNT-6", [r["OCNT_ID"] for r in result["evaluated"]])
        self.assertEqual(result["tier_role"], SYN_EXPECTED["tier_role"])

    def test_count_mismatches_stop(self):
        for key, value in (("evaluated", 4), ("set_aside", 2), ("rows", 7),
                           ("roles", {"primary_validation": 2, "aux_only": 3}),
                           ("tier_role", {("train", "primary_validation"): 3})):
            with self.subTest(key), self.assertRaises(ValueError):
                ev.split_evaluation_set(mapping_rows(), {**SYN_EXPECTED, key: value})

    def test_recorded_real_counts_are_the_agreed_ones(self):
        self.assertEqual((ev.EXPECTED["rows"], ev.EXPECTED["set_aside"], ev.EXPECTED["evaluated"]), (208, 11, 197))
        self.assertEqual(ev.EXPECTED["roles"], {"primary_validation": 160, "aux_only": 37})
        self.assertEqual(sum(ev.EXPECTED["tier_role"].values()), 197)
        self.assertEqual(ev.EXPECTED["cyp_train_molecules"], 6252)


class LabelAndBaselineTests(unittest.TestCase):
    def test_labels_are_joined_by_name_and_enzyme_only_for_requested_molecules(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            text = ws.screen_bytes.decode()
            labels = ev.load_cyp_labels(text, {"OCNT-1": "OCNT-1", "OCNT-2": "OCNT-2"})
            self.assertEqual(set(labels), {"OCNT-1", "OCNT-2"})
            self.assertAlmostEqual(labels["OCNT-2"]["CYP2D6"], label(2, "CYP2D6"))
            with self.assertRaises(ValueError):
                ev.load_cyp_labels(text, {"OCNT-1": "OCNT-NOPE"})
            with self.assertRaises(ValueError):
                ev.load_cyp_labels(text + "OCNT-1,x,b,CYP1A2,p,4.9e-05,0.0\n", {"OCNT-1": "OCNT-1"})
            blank = text.replace(repr(label(1, "CYP3A4")), "")
            with self.assertRaises(ValueError):
                ev.load_cyp_labels(blank, {"OCNT-1": "OCNT-1"})

    def test_pxr_labels_use_the_8p25_library_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            text = (ws.t / "single.csv").read_text()
            labels = ev.load_pxr_labels(text, ["OCNT-1", "OCNT-2"])
            self.assertEqual(labels, {"OCNT-1": 0.3, "OCNT-2": 0.6})  # 33 uM and positive-control rows ignored
            with self.assertRaises(ValueError):
                ev.load_pxr_labels(text, ["OCNT-1", "OCNT-404"])
            with self.assertRaises(ValueError):
                ev.load_pxr_labels(text + ",b,x,p,Library,8.251e-06,1.0,OCNT-1\n", ["OCNT-1"])

    def test_cyp_constants_are_unweighted_training_means_and_exclude_other_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            train, everything = ev.read_train_split((ws.t / "split.csv").read_text(), 4)
            self.assertEqual(train, {"T1", "T2", "T3", "T4"})
            self.assertEqual(everything, {"T1", "T2", "T3", "T4", "V1"})
            constants = ev.cyp_training_constants(ws.screen_bytes.decode(), train, everything, ["OCNT-1", "OCNT-2"])
            for e in ev.ENZYMES:
                self.assertAlmostEqual(constants[e]["constant"], 1.5)  # V1's 100.0 is not used
                self.assertEqual(constants[e]["n_train_labels"], 4)

    def test_evaluation_molecule_in_the_split_stops_the_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            train, everything = ev.read_train_split((ws.t / "split.csv").read_text(), 4)
            with self.assertRaises(ValueError):
                ev.cyp_training_constants(ws.screen_bytes.decode(), train, everything | {"OCNT-1"}, ["OCNT-1"])

    def test_split_reading_handles_crlf_and_rejects_wrong_counts_header_and_duplicates(self):
        crlf = "molecule_id,cluster_id,partition\r\nA,1,train\r\nB,2,validation\r\n"
        train, everything = ev.read_train_split(crlf, 1)
        self.assertEqual((train, everything), ({"A"}, {"A", "B"}))
        with self.assertRaises(ValueError):
            ev.read_train_split(crlf, 2)
        with self.assertRaises(ValueError):
            ev.read_train_split("a,b,c\r\nA,1,train\r\n", 1)
        with self.assertRaises(ValueError):
            ev.read_train_split("molecule_id,cluster_id,partition\r\nA,1,train\r\nA,2,train\r\n", 2)

    def test_pxr_constant_comes_from_the_pins_and_the_head_order_is_checked(self):
        pins = {"expected": {"head_order": ["log2fc_8p25uM", "x", "y"], "scaler_mean": [0.3019, 1, 2]}}
        self.assertEqual(ev.pxr_training_constant(pins)["constant"], 0.3019)
        pins["expected"]["head_order"] = ["log2fc_33uM", "log2fc_8p25uM", "y"]
        with self.assertRaises(ValueError):
            ev.pxr_training_constant(pins)


class EndToEndTests(unittest.TestCase):
    def run_synthetic(self, tmp, **kwargs):
        ws = Workspace(tmp, **{k: v for k, v in kwargs.items() if k == "split_extra"})
        args = ws.args(**{k: v for k, v in kwargs.items() if k != "split_extra"})
        with mock.patch.object(ev, "EXPECTED", SYN_EXPECTED):
            return ws, args, ev.run(args)

    def test_metrics_match_hand_calculations_and_the_set_aside_molecule_is_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, args, out = self.run_synthetic(tmp)
            self.assertEqual(len(out["evaluated"]), 5)
            self.assertNotIn("OCNT-6", out["cyp_labels"])
            self.assertNotIn("OCNT-6", out["pxr_labels"])
            r = out["results"]
            a = r["all"]["CYP1A2"]
            self.assertEqual(a["n"], 5)
            self.assertAlmostEqual(a["mae"], 0.1)
            self.assertAlmostEqual(a["baseline_mae"], 1.2)  # c = 1.5 from training only; labels 0.1 .. 0.5
            self.assertAlmostEqual(a["mae_over_baseline"], 0.1 / 1.2)
            self.assertAlmostEqual(a["spearman"], 1.0)
            self.assertAlmostEqual(r["all"]["CYP2C9"]["mae"], 0.2)
            p = r["all"]["PXR_8p25uM"]
            self.assertAlmostEqual(p["mae"], 0.05)
            self.assertAlmostEqual(p["baseline_constant"], 0.25)
            self.assertAlmostEqual(p["baseline_mae"], 0.65)
            self.assertEqual(r["primary_validation"]["CYP1A2"]["n"], 3)
            self.assertEqual(r["aux_only"]["CYP1A2"]["n"], 2)
            self.assertIsNone(r["aux_only"]["CYP1A2"]["spearman"])  # n = 2
            self.assertEqual(out["report"]["set"]["set_aside_not_scored"], 1)
            self.assertFalse(out["report"]["labels"]["set_aside_molecules_scored"])

    def test_outputs_are_written_once_without_nan_and_exclude_the_set_aside_molecule(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, args, out = self.run_synthetic(tmp)
            files = ev.write_outputs(args.out_dir, out["evaluated"], out["pred_by_id"], out["cyp_labels"],
                                     out["pxr_labels"], out["results"], out["report"])
            self.assertEqual(set(files), {"evaluation_metrics.csv", "evaluation_rows.csv", "crosswalk_table.csv",
                                          "evaluation_report.json"})
            rows = list(csv.DictReader(open(args.out_dir / "evaluation_rows.csv", newline="")))
            self.assertEqual([r["OCNT_ID"] for r in rows], ["OCNT-1", "OCNT-2", "OCNT-3", "OCNT-4", "OCNT-5"])
            metrics = list(csv.DictReader(open(args.out_dir / "evaluation_metrics.csv", newline="")))
            self.assertEqual(len(metrics), 3 * 5)
            report = json.loads((args.out_dir / "evaluation_report.json").read_text())
            self.assertIn("train|primary_validation", report["tier_role_counts"])
            with self.assertRaises(FileExistsError):
                ev.write_outputs(args.out_dir, out["evaluated"], out["pred_by_id"], out["cyp_labels"],
                                 out["pxr_labels"], out["results"], out["report"])

    def test_changed_input_overlap_and_incomplete_predictions_stop_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            with mock.patch.object(ev, "EXPECTED", SYN_EXPECTED):
                with self.assertRaises(ValueError):
                    ev.run(ws.args(predictions_sha256="0" * 64))
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp, split_extra=[["OCNT-1", "7", "validation"]])
            with mock.patch.object(ev, "EXPECTED", SYN_EXPECTED):
                with self.assertRaises(ValueError):
                    ev.run(ws.args())
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            lines = (ws.t / "pred.csv").read_text().splitlines()[:-1]
            (ws.t / "pred.csv").write_text("\n".join(lines) + "\n")
            with mock.patch.object(ev, "EXPECTED", SYN_EXPECTED):
                with self.assertRaises(ValueError):
                    ev.run(ws.args(predictions_sha256=ev.sha256_file(ws.t / "pred.csv")))


class SourceInvariantTests(unittest.TestCase):
    def test_script_loads_no_model_and_reads_set_aside_labels_nowhere(self):
        source = (Path(__file__).resolve().parents[2] / "cyp/scripts/evaluate_screen_predictions.py").read_text()
        body = source.split('"""', 2)[2]
        for forbidden in ("torch", "chemprop", "load_state_dict"):
            self.assertNotIn(forbidden, body)
        # labels are requested only for the evaluated molecules, after the set-aside step (calls in run())
        self.assertLess(body.rindex("split_evaluation_set(mapping_rows, EXPECTED)"),
                        body.rindex("load_cyp_labels(screen_text"))
        self.assertLess(body.rindex("split_evaluation_set(mapping_rows, EXPECTED)"),
                        body.rindex("load_pxr_labels(single_text"))


if __name__ == "__main__":
    unittest.main()
