"""Penalty-arm prediction diagnostics: pairing refusals, aggregation, signs,
error totals, seed split, counterfactual and coverage. Synthetic data only.

Arms of the fixture (e_i = (i - 9.5) / 10 is molecule i's unchanged error):
    shift    every row 0.2 lower than unchanged
    seeded   +0.3 on the first seed's rows only
    same     context arm identical to unchanged
"""
from pathlib import Path
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_t121_d024_aggregate as d024_agg  # noqa: E402  (also sets the script path)
import t121_penalty_prediction_diagnostics as diag  # noqa: E402

sha = d024_agg.sha
MOLECULES, COVERED = d024_agg.MOLECULES, d024_agg.COVERED
ISOFORMS = ("CYP1A2", "CYP3A4")
SEEDS = (20261101, 20261102, 20261103)
CONFIG = {
    "isoforms": list(ISOFORMS), "cell_seeds": list(SEEDS),
    "cells_per_seed": 2, "predictions_per_molecule": 6,
}
CELLS = range(6)
OVER, UNDER = ["M18", "M19"], ["M00", "M01"]


def error(i):
    return (i - 9.5) / 10.0


def arm_rows(arm):
    cells = {}
    for cell in CELLS:
        seed = SEEDS[cell % 3]
        rows = []
        for isoform in ISOFORMS:
            for i, molecule in enumerate(MOLECULES):
                y = 5.0 + 0.1 * i
                prediction = y + error(i)
                if arm == "shift":
                    prediction -= 0.2
                elif arm == "seeded" and seed == SEEDS[0]:
                    prediction += 0.3
                rows.append(dict(zip(diag.COLUMNS, (
                    121, cell, 1 + cell // 3, 1, seed, isoform, molecule,
                    y, prediction, y - 0.2, y + 0.2,
                ))))
        cells[cell] = rows
    return cells


def tail_mae(cells, isoform, members):
    return float(np.mean([
        abs(r["y_pred"] - r["y_true"])
        for rows in cells.values() for r in rows
        if r["isoform"] == isoform and r["Molecule_Name"] in members
    ]))


class Fixture:
    def __init__(self, root, mutate=None):
        self.root = Path(root)
        data = {arm: arm_rows(arm) for arm in ("unchanged", "shift", "seeded", "same")}
        if mutate is not None:
            mutate(data)
        entries, metrics = {}, {}
        macro = {"unchanged": 0.70, "shift": 0.69, "seeded": 0.71, "same": 0.70}
        for arm, cells in data.items():
            directory = self.root / f"pred_{arm}"
            directory.mkdir(parents=True)
            hashes = {}
            for cell, rows in cells.items():
                path = directory / f"predictions_cell_{cell:03d}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=diag.COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows[::-1])  # row order must not matter
                hashes[str(cell)] = sha(path.read_bytes())
            entries[arm] = {"cells_dir": "run/trials/trial_121/cells", "prediction_sha256_by_cell": hashes}
            metrics[arm] = {
                "O150": macro[arm], "per_isoform": {i: macro[arm] for i in ISOFORMS},
                "tails": {i: {
                    "over": {"mae": tail_mae(cells, i, OVER)},
                    "under": {"mae": tail_mae(cells, i, UNDER)},
                } for i in ISOFORMS},
            }
        self.config = self.write("config.json", CONFIG)
        self.cohorts = self.write("cohorts.json", {"cohorts": {
            i: {"n": 20, "size": 2, "over": OVER, "under": UNDER} for i in ISOFORMS}})
        self.screen = self.root / "screen.tar.gz"
        d024_agg.screen_tar(self.screen, COVERED)
        penalty_arm = diag.penalty.PenaltyArm.load(
            d024_agg.PENALTY_CONFIG_PATH,
            expected_sha256=sha(d024_agg.PENALTY_CONFIG_PATH.read_bytes()))
        self.frozen_coverage = {
            "isoform": "CYP3A4",
            "statuses": {m: "ok" if m in COVERED else "no_screening_row" for m in MOLECULES},
            "groups": {m: "covered" if m in COVERED else "not_covered" for m in MOLECULES},
            "coverage_config_sha256": sha(d024_agg.COVERAGE_CONFIG_PATH.read_bytes()),
            "penalty_config_sha256": penalty_arm.config_sha256,
            "screen_input_sha256": diag.penalty.load_screen_tar(self.screen, penalty_arm)["sha256"],
        }
        self.coverage = self.write("coverage.json", self.frozen_coverage)
        shared = {
            "cohorts_sha256": sha(self.cohorts.read_bytes()),
            "coverage_sha256": sha(self.coverage.read_bytes()),
        }
        self.report_data = {
            **shared, "decision_record": "synthetic",
            "inputs": {a: entries[a] for a in ("unchanged", "shift", "seeded")},
            "metrics": {a: metrics[a] for a in ("unchanged", "shift", "seeded")},
        }
        self.report = self.write("report.json", self.report_data)
        self.context = self.write("context.json", {
            **shared,
            "inputs": {a: entries[a] for a in ("unchanged", "same")},
            "metrics": {a: metrics[a] for a in ("unchanged", "same")},
        })

    def write(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload))
        return path

    def run(self, output="out", context_dir=None):
        digest = lambda p: sha(p.read_bytes())  # noqa: E731
        arguments = [
            "--config", self.config, "--report", self.report, "--report-sha256", digest(self.report),
            "--context-report", self.context, "--context-report-sha256", digest(self.context),
            "--cohorts", self.cohorts, "--cohorts-sha256", digest(self.cohorts),
            "--coverage", self.coverage, "--coverage-sha256", digest(self.coverage),
            "--coverage-config", d024_agg.COVERAGE_CONFIG_PATH,
            "--coverage-penalty-config", d024_agg.PENALTY_CONFIG_PATH,
            "--screen-tar", self.screen, "--output-dir", self.root / output,
            "--predictions", f"unchanged={self.root / 'pred_unchanged'}",
            "--predictions", f"shift={self.root / 'pred_shift'}",
            "--predictions", f"seeded={self.root / 'pred_seeded'}",
            "--context-predictions", f"same={context_dir or self.root / 'pred_same'}",
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            status = diag.main([str(a) for a in arguments])
        out = self.root / output
        return status, {p.name: pd.read_csv(p) for p in out.glob("*.csv")}, json.loads(
            (out / "run_metadata.json").read_text())


def row(table, **where):
    chosen = table
    for key, value in where.items():
        chosen = chosen[chosen[key] == value]
    assert len(chosen) == 1, f"{where}: {len(chosen)} rows"
    return chosen.iloc[0]


class Base(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)


class CleanRunTests(Base):
    def setUp(self):
        super().setUp()
        self.fixture = Fixture(self.root / "fx")
        self.status, self.tables, self.metadata = self.fixture.run()

    def test_completes_with_roles_and_reconciled_tails(self):
        self.assertEqual((self.status, self.metadata["stopped"]), (0, {}))
        self.assertEqual(self.metadata["arms"],
                         {"shift": "candidate", "seeded": "candidate", "same": "context"})
        self.assertLessEqual(max(self.metadata["checks"]["tail_mae_max_difference"].values()), 1e-12)

    def test_group_totals_counts_and_signs(self):
        summary = self.tables["group_summary.csv"]
        for isoform in ISOFORMS:
            whole = row(summary, isoform=isoform, arm="shift", group="all")
            self.assertEqual((whole.n_lower, whole.n_equal, whole.n_higher), (9, 0, 11))
            self.assertAlmostEqual(whole.total_error_reduction, 1.7)
            self.assertAlmostEqual(whole.total_error_increase, 2.1)
            self.assertAlmostEqual(whole.net_error_change, 0.4)
            self.assertAlmostEqual(whole.delta_mae, 0.02)
            self.assertAlmostEqual(whole.mae_arm - whole.mae_unchanged, 0.02)
            self.assertAlmostEqual(whole.mean_prediction_change, -0.2)
            self.assertAlmostEqual(whole.median_prediction_change, -0.2)
            self.assertAlmostEqual(whole.mean_signed_residual_unchanged, 0.0)
            self.assertAlmostEqual(whole.mean_signed_residual_arm, -0.2)
            for group, n, delta in (("over_tail", 2, -0.2), ("under_tail", 2, 0.2),
                                    ("remaining", 16, 0.025)):
                entry = row(summary, isoform=isoform, arm="shift", group=group)
                self.assertEqual(entry.n_molecules, n)
                self.assertAlmostEqual(entry.delta_mae, delta)
        for entry in summary.itertuples():
            self.assertEqual(entry.n_lower + entry.n_equal + entry.n_higher, entry.n_molecules)
            self.assertAlmostEqual(
                entry.total_error_increase - entry.total_error_reduction, entry.net_error_change)
            self.assertAlmostEqual(entry.net_error_change / entry.n_molecules, entry.delta_mae)
        same = row(summary, isoform="CYP1A2", arm="same", group="all")
        self.assertEqual((same.n_equal, same.total_error_reduction, same.total_error_increase), (20, 0, 0))

    def test_absolute_error_is_the_mean_of_row_errors_not_the_error_of_the_mean(self):
        molecule = row(self.tables["molecule_level.csv"], isoform="CYP1A2", Molecule_Name="M08")
        # unchanged error -0.15; rows are +0.15 (first seed) and -0.15 (the other two)
        self.assertAlmostEqual(molecule.pred_seeded - molecule.y_true, -0.05)
        self.assertAlmostEqual(molecule.signed_seeded, -0.05)
        self.assertAlmostEqual(molecule.abs_error_seeded, 0.15)
        self.assertAlmostEqual(molecule.change_seeded, 0.1)
        self.assertAlmostEqual(molecule.abs_error_change_shift, 0.2)
        self.assertEqual((molecule.n_predictions, molecule.group), (6, "remaining"))

    def test_seed_breakdown_keeps_each_seed_apart(self):
        seeds = self.tables["seed_breakdown.csv"]
        expected = float(np.mean([abs(error(i) + 0.3) - abs(error(i)) for i in range(20)]))
        values = {
            s: row(seeds, isoform="CYP1A2", arm="seeded", group="all", seed=s).delta_mae for s in SEEDS
        }
        self.assertAlmostEqual(values[SEEDS[0]], expected)
        self.assertAlmostEqual(values[SEEDS[1]], 0.0)
        self.assertAlmostEqual(values[SEEDS[2]], 0.0)
        pooled = row(self.tables["group_summary.csv"], isoform="CYP1A2", arm="seeded", group="all")
        self.assertAlmostEqual(pooled.delta_mae, sum(values.values()) / 3)

    def test_concentration_orders_each_direction_and_ends_at_its_own_total(self):
        curves = self.tables["concentration.csv"]
        chosen = curves[(curves.arm == "shift") & (curves.isoform == "CYP1A2")]
        for direction, n, total in (("improvement", 9, 1.7), ("deterioration", 11, 2.1)):
            part = chosen[chosen.direction == direction].sort_values("rank")
            self.assertEqual(len(part), n)
            self.assertTrue((np.diff(part.contribution) <= 0).all())
            self.assertAlmostEqual(part.contribution.sum(), total)
            self.assertAlmostEqual(part.cumulative_share.iloc[-1], 1.0)
            self.assertAlmostEqual(part.share_of_all_molecules.iloc[-1], n / 20)
        self.assertFalse((curves.arm == "same").any())

    def test_counterfactual_removes_the_mean_change_and_is_labelled(self):
        table = self.tables["counterfactual.csv"]
        self.assertEqual(set(table.status), {diag.COUNTERFACTUAL_STATUS})
        for group in diag.COUNTERFACTUAL_GROUPS:
            entry = row(table, isoform="CYP1A2", arm="shift", group=group)
            self.assertAlmostEqual(entry.mean_prediction_change_removed, -0.2)
            self.assertAlmostEqual(entry.mae_counterfactual, entry.mae_unchanged)
            self.assertAlmostEqual(entry.delta_counterfactual, 0.0)
        seeded = row(table, isoform="CYP1A2", arm="seeded", group="all")
        expected = float(np.mean([
            (abs(error(i) + 0.2) + 2 * abs(error(i) - 0.1)) / 3 for i in range(20)
        ]))
        self.assertAlmostEqual(seeded.mae_counterfactual, expected)

    def test_coverage_counts_and_changes(self):
        table = self.tables["coverage_summary.csv"]
        for coverage, delta in (("covered", 0.03), ("not_covered", 0.01)):
            entry = row(table, isoform="CYP3A4", arm="shift", coverage=coverage)
            self.assertEqual(entry.n_molecules, 10)
            self.assertAlmostEqual(entry.mean_prediction_change, -0.2)
            self.assertAlmostEqual(entry.delta_mae, delta)
        empty = row(table, isoform="CYP1A2", arm="shift", coverage="not_covered")
        self.assertEqual(empty.n_molecules, 0)
        self.assertTrue(np.isnan(empty.delta_mae))

    def test_macro_values_are_copied_and_outputs_are_never_reused(self):
        macro = self.tables["macro_st_rae.csv"]
        self.assertAlmostEqual(
            row(macro, arm="shift", scope="macro").difference_from_unchanged, -0.01)
        with self.assertRaises(FileExistsError):
            self.fixture.run()


class RefusalTests(Base):
    def stopped(self, mutate=None, adjust=None, arm="shift", **options):
        fixture = Fixture(self.root / "fx", mutate=mutate)
        if adjust is not None:
            adjust(fixture)
        status, tables, metadata = fixture.run(**options)
        self.assertEqual(status, 2)
        self.assertIn(arm, metadata["stopped"])
        self.assertNotIn(arm, metadata["arms"])
        self.assertNotIn(arm, set(tables["group_summary.csv"].arm))
        self.assertIn("seeded", metadata["arms"])
        return metadata["stopped"][arm]

    def test_a_different_label_is_refused(self):
        def mutate(data):
            for rows in data["shift"].values():
                for r in rows:
                    if r["Molecule_Name"] == "M05":
                        r["y_true"] += 0.001

        self.assertRegex(self.stopped(mutate), "label or bound column y_true")

    def test_a_different_molecule_is_refused(self):
        def mutate(data):
            for rows in data["shift"].values():
                for r in rows:
                    if r["Molecule_Name"] == "M05":
                        r["Molecule_Name"] = "M99"

        self.assertRegex(self.stopped(mutate), "keys differ")

    def test_swapped_cell_identity_is_refused(self):
        def mutate(data):
            for cell, other in ((0, SEEDS[1]), (1, SEEDS[0])):
                for r in data["shift"][cell]:
                    r["seed"] = other

        self.assertRegex(self.stopped(mutate), "cell identity column seed")

    def test_a_changed_file_is_refused(self):
        def adjust(fixture):
            path = fixture.root / "pred_shift" / "predictions_cell_002.csv"
            path.write_text(path.read_text() + "\n")

        self.assertRegex(self.stopped(adjust=adjust), "SHA-256")

    def test_a_tail_value_that_does_not_reconcile_is_refused(self):
        def adjust(fixture):
            fixture.report_data["metrics"]["shift"]["tails"]["CYP3A4"]["under"]["mae"] += 1e-6
            fixture.report = fixture.write("report.json", fixture.report_data)

        self.assertRegex(self.stopped(adjust=adjust), "tail MAE differs")

    def test_missing_context_predictions_stop_only_that_comparison(self):
        reason = self.stopped(arm="same", context_dir=self.root / "absent")
        self.assertRegex(reason, "missing prediction file")

    def test_an_unusable_unchanged_arm_stops_the_run(self):
        def mutate(data):
            data["unchanged"][0].pop()

        fixture = Fixture(self.root / "fx", mutate=mutate)
        with self.assertRaisesRegex(diag.Stop, "exactly once"):
            fixture.run()
        self.assertFalse((fixture.root / "out").exists())

    def test_coverage_that_differs_from_the_frozen_grouping_is_not_reported(self):
        fixture = Fixture(self.root / "fx")
        fixture.frozen_coverage["groups"]["M00"] = "not_covered"
        fixture.coverage = fixture.write("coverage.json", fixture.frozen_coverage)
        digest = sha(fixture.coverage.read_bytes())
        fixture.report_data["coverage_sha256"] = digest
        fixture.report = fixture.write("report.json", fixture.report_data)
        status, tables, metadata = fixture.run()
        self.assertEqual(status, 2)
        self.assertRegex(metadata["stopped"]["screening_coverage"], "differs from the frozen")
        self.assertNotIn("coverage_summary.csv", tables)
        self.assertEqual(set(metadata["arms"]), {"shift", "seeded", "same"})


if __name__ == "__main__":
    unittest.main()
