"""Tests for evaluate_screen_cyp_tier_split.py (Tripwire-CYP-010).

Each test is tied to a failure risk: a tier assigned to the wrong molecules, the signed residual
having the wrong sign, a baseline constant that differs between strata or tiers, a changed input
being read, pooled numbers not reproducing the recorded ones, and an output directory being
reused. Writing these tests is not a passing run.
"""
import csv
import hashlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cyp/scripts"))
import evaluate_screen_cyp_tier_split as ts  # noqa: E402
import evaluate_screen_predictions as ev  # noqa: E402

CONST = {"CYP1A2": -0.2, "CYP2C9": -0.4, "CYP2D6": 0.3, "CYP3A4": -1.0}


def make_rows():
    rows = []
    for i in range(8):
        tier = "validation" if i >= 5 else "train"
        row = {"OCNT_ID": f"OCNT-{i}", "pxr_role": "primary_validation", "cyp_tiers": tier}
        for k, e in enumerate(ev.ENZYMES):
            label = 0.1 * i - 0.3 + 0.05 * k
            row[f"label_{e}"] = repr(label)
            row[ev.CYP_OUTPUTS[e]] = repr(label + 0.25 + 0.01 * (i % 3))  # prediction above label
        rows.append(row)
    return rows


def metrics_rows(rows, strata=("all", "primary_validation")):
    out = []
    for s in strata:
        for e in ev.ENZYMES:
            m = ev.metrics_for([float(r[ev.CYP_OUTPUTS[e]]) for r in rows],
                               [float(r[f"label_{e}"]) for r in rows], CONST[e])
            out.append({"stratum": s, "output": e, "n": str(m["n"]), "mae": repr(m["mae"]),
                        "spearman": repr(m["spearman"]), "baseline_constant": repr(CONST[e]),
                        "baseline_mae": repr(m["baseline_mae"]), "mae_over_baseline": repr(m["mae_over_baseline"])})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TierSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rows = make_rows()
        self.rows_path = self.dir / "rows.csv"
        self.metrics_path = self.dir / "metrics.csv"
        write_csv(self.rows_path, self.rows)
        write_csv(self.metrics_path, metrics_rows(self.rows))

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, out="out"):
        return SimpleNamespace(rows=self.rows_path, rows_sha256=sha(self.rows_path), metrics=self.metrics_path,
                               metrics_sha256=sha(self.metrics_path), output_dir=self.dir / out)

    def test_tiers_and_signed_residual(self):
        constants = ts.constants_from_metrics(list(csv.DictReader(open(self.metrics_path))))
        table = ts.tier_rows(self.rows, constants)
        by = {(r["output"], r["cyp_tier"]): r for r in table}
        self.assertEqual(by[("CYP1A2", "train")]["n"], 5)
        self.assertEqual(by[("CYP1A2", "validation")]["n"], 3)
        for r in table:  # predictions are above labels, so prediction minus label is positive
            self.assertGreater(r["signed_mean_residual"], 0.24)
            self.assertLess(r["signed_mean_residual"], 0.28)

    def test_validation_tier_uses_only_its_molecules(self):
        constants = {e: CONST[e] for e in ev.ENZYMES}
        val = [r for r in self.rows if r["cyp_tiers"] == "validation"]
        expected = sum(abs(float(r["label_CYP2D6"]) - CONST["CYP2D6"]) for r in val) / len(val)
        got = {(r["output"], r["cyp_tier"]): r for r in ts.tier_rows(self.rows, constants)}
        self.assertAlmostEqual(got[("CYP2D6", "validation")]["baseline_mae"], expected, places=12)

    def test_constant_must_not_differ_between_strata(self):
        mrows = metrics_rows(self.rows)
        mrows[4]["baseline_constant"] = repr(CONST["CYP1A2"] + 0.01)  # second stratum, CYP1A2
        with self.assertRaises(ValueError):
            ts.constants_from_metrics(mrows)

    def test_changed_input_is_refused(self):
        args = self.args()
        self.rows_path.write_text(self.rows_path.read_text() + "\n")
        with self.assertRaises(ValueError):
            ts.run(args)

    def test_pooled_numbers_must_match_recorded(self):
        mrows = metrics_rows(self.rows)
        mrows[0]["mae"] = repr(float(mrows[0]["mae"]) + 1e-6)
        with self.assertRaises(ValueError):
            ts.pooled_check(self.rows, CONST, mrows)

    def test_unknown_tier_is_refused(self):
        self.rows[0]["cyp_tiers"] = "test"
        write_csv(self.rows_path, self.rows)
        with self.assertRaises(ValueError):
            ts.run(self.args())

    def test_output_directory_is_not_reused(self):
        ts.run(self.args("first"))
        with self.assertRaises(FileExistsError):
            ts.run(self.args("first"))


if __name__ == "__main__":
    unittest.main()
