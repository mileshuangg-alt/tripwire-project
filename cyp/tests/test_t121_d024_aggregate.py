"""D024 aggregation: tail cohorts, every adoption clause, integrity checks, CLI."""
from pathlib import Path
import contextlib
import copy
import csv
import hashlib
import io
import json
import numpy as np
import sys
import tarfile
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cyp/scripts"))
import t121_d024_aggregate as agg

CONFIG_PATH = REPO / "cyp/artifacts/d024_penalty/evaluation.json"
COVERAGE_CONFIG_PATH = REPO / "cyp/artifacts/d024_penalty/coverage.json"
PENALTY_CONFIG_PATH = REPO / "cyp/artifacts/d024_penalty/penalty_q3.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
COVERAGE_CONFIG = json.loads(COVERAGE_CONFIG_PATH.read_text())
ISOFORMS = CONFIG["isoforms"]
SEEDS = CONFIG["cell_seeds"]
MOLECULES = [f"M{i:02d}" for i in range(20)]
COVERED = {m for i, m in enumerate(MOLECULES) if i % 2 == 0}  # synthetic CYP3A4 coverage


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prediction_rows(shift=0.0):
    """Each molecule appears in 15 cells; its signed error rises with its index."""
    rows = {}
    for isoform in ISOFORMS:
        for i, molecule in enumerate(MOLECULES):
            for k in range(15):
                cell = (i + 7 * k) % 150
                error = (i - 9.5) / 10.0
                shrunk = error - shift if error > 0 else error
                truth = 5.0 + 0.3 * (k % 5)
                rows[(cell, isoform, molecule)] = (truth, truth + shrunk, truth - 0.2, truth + 0.2)
    return rows


def screen_tar(path, covered=COVERED):
    """Synthetic screening inputs: CYP3A4 rows exist only for covered molecules."""
    import t121_screening_penalty as sp

    smiles = {m: "C" * (i + 1) for i, m in enumerate(MOLECULES)}
    screen = ["Molecule_Name,SMILES,enzyme,concentration_M,log2fc_estimate\r\n"]
    for m in MOLECULES:
        for isoform in ISOFORMS:
            if isoform != "CYP3A4" or m in covered:
                screen.append(f"{m},{smiles[m]},{isoform},5e-05,-1.0\r\n")
    identity = ["molecule_name\tsmiles\tclass\ttiers\r\n"] + [
        f"{m}\t{smiles[m]}\tfull_key_match\ttrain\r\n" for m in MOLECULES
    ]
    members = {
        sp.SCREEN_MEMBER: "".join(screen).encode(),
        sp.IDENTITY_MEMBER: "".join(identity).encode(),
    }
    members[sp.CONTRACT_MEMBER] = json.dumps({"sha256": {
        "screen": sha(members[sp.SCREEN_MEMBER]), "identity": sha(members[sp.IDENTITY_MEMBER]),
    }}).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def metrics(value=0.7, over=0.5, under=0.5):
    return {
        "O150": value,
        "by_isoform_seed": {i: {str(s): value for s in SEEDS} for i in ISOFORMS},
        "tails": {i: {"over": {"mae": over}, "under": {"mae": under}} for i in ISOFORMS},
    }


class CohortTests(unittest.TestCase):
    def test_cohorts_take_ten_percent_of_each_tail_from_mean_signed_error(self):
        cohorts = agg.build_cohorts(prediction_rows(), CONFIG)
        for isoform in ISOFORMS:
            self.assertEqual(cohorts[isoform]["size"], 2)
            self.assertEqual(cohorts[isoform]["over"], ["M18", "M19"])
            self.assertEqual(cohorts[isoform]["under"], ["M00", "M01"])

    def test_size_rounding_and_name_ties(self):
        for n, size in ((1201, 120), (1093, 109), (1269, 127), (1986, 199), (4, 1)):
            self.assertEqual(max(1, round(CONFIG["tail_fraction"] * n)), size)
        # three molecules tie at each end: each tail takes the first two names
        errors = {m: (-1.0 if i < 3 else 1.0 if i >= 17 else 0.0) for i, m in enumerate(MOLECULES)}
        rows = {
            (k, isoform, m): (5.0, 5.0 + errors[m], 4.8, 5.2)
            for isoform in ISOFORMS for m in MOLECULES for k in range(15)
        }
        cohort = agg.build_cohorts(rows, CONFIG)[ISOFORMS[0]]
        self.assertEqual(cohort["over"], ["M17", "M18"])
        self.assertEqual(cohort["under"], ["M00", "M01"])

    def test_wrong_prediction_count_raises(self):
        rows = prediction_rows()
        del rows[next(iter(rows))]
        with self.assertRaisesRegex(ValueError, "predictions per molecule"):
            agg.build_cohorts(rows, CONFIG)


class RuleTests(unittest.TestCase):
    def clauses(self, arm):
        return agg.arm_clauses(metrics(), arm, CONFIG)

    def test_all_clauses_pass_and_equality_rules(self):
        passing = metrics(value=0.69, over=0.49, under=0.5)  # equal under tail passes
        judged = self.clauses(passing)
        self.assertTrue(judged["passes"])
        self.assertEqual(judged["comparisons"], 12)
        one_equal = copy.deepcopy(passing)
        one_equal["by_isoform_seed"]["CYP2D6"][str(SEEDS[1])] = 0.7  # equal: "no worse"
        self.assertTrue(self.clauses(one_equal)["passes"])

    def test_each_clause_blocks_alone(self):
        good = metrics(value=0.69, over=0.49, under=0.49)
        cases = {}
        cases["a_lower_mean_macro_st_rae"] = {**copy.deepcopy(good), "O150": 0.7}
        worse = copy.deepcopy(good)
        worse["by_isoform_seed"]["CYP3A4"][str(SEEDS[2])] = 0.7000001
        cases["b_no_worse_enzyme_seed_mean"] = worse
        over = copy.deepcopy(good)
        over["tails"]["CYP1A2"]["over"]["mae"] = 0.5  # equal over tail is not lower
        cases["c_lower_over_tail_mae"] = over
        under = copy.deepcopy(good)
        under["tails"]["CYP2C9"]["under"]["mae"] = 0.5000001
        cases["d_no_higher_under_tail_mae"] = under
        for clause, arm in cases.items():
            judged = self.clauses(arm)
            self.assertFalse(judged["passes"], clause)
            self.assertEqual([k for k, v in judged["clauses"].items() if not v], [clause])

    def test_selection(self):
        base = metrics()
        better, best = metrics(0.69, 0.49, 0.49), metrics(0.68, 0.49, 0.49)

        def selected(q3, rank):
            return agg.decide({"unchanged": base, "q3": q3, "rank": rank}, CONFIG)["selected"]

        self.assertEqual(selected(better, best), "rank")
        self.assertEqual(selected(best, better), "q3")
        self.assertEqual(selected(better, copy.deepcopy(better)), "q3")  # exact tie
        self.assertEqual(selected(base, better), "rank")
        self.assertEqual(selected(base, copy.deepcopy(base)), "unchanged")


class CoverageGroupTests(unittest.TestCase):
    ROWS = {
        (0, "CYP3A4", "A"): (5.0, 5.5, 4.9, 5.1),
        (0, "CYP3A4", "B"): (6.0, 6.0, 5.9, 6.1),
        (0, "CYP3A4", "C"): (7.0, 7.5, 6.9, 7.1),
        (1, "CYP3A4", "A"): (5.0, 5.0, 4.9, 5.1),
    }
    COHORTS = {"CYP3A4": {"over": ["A"], "under": ["C"]}}

    def test_cell_st_rae_follows_the_scorer_definition(self):
        y, p = np.array([5.0, 6.0, 7.0]), np.array([5.5, 6.0, 7.5])
        # model soft error 0.4 + 0 + 0.4; constant-at-mean soft error 0.9 + 0 + 0.9
        self.assertAlmostEqual(agg._cell_st_rae(y, p, y - 0.1, y + 0.1), 0.8 / 1.8)
        self.assertIsNone(agg._cell_st_rae(y[:1], p[:1], y[:1] - 0.1, y[:1] + 0.1))

    def test_group_metrics_pool_rows_and_score_cells_separately(self):
        entry = agg.group_metrics(self.ROWS, "CYP3A4", {"A", "B", "C"}, {0: 20261101, 1: 20261102}, self.COHORTS)
        self.assertEqual((entry["molecules"], entry["rows"]), (3, 4))
        self.assertAlmostEqual(entry["mae"], 1.0 / 4)
        self.assertAlmostEqual(entry["signed_error"], 1.0 / 4)
        self.assertEqual((entry["st_rae_cells"], entry["cells_without_score"]), (1, 1))
        self.assertAlmostEqual(entry["st_rae_mean"], 0.8 / 1.8)
        self.assertEqual(list(entry["st_rae_by_seed"]), ["20261101"])
        self.assertEqual(entry["tails"]["over"], {"members": 1, "rows": 2, "mae": 0.25})
        self.assertEqual(entry["tails"]["under"]["members"], 1)
        only_a = agg.group_metrics(self.ROWS, "CYP3A4", {"A"}, {0: 1, 1: 2}, self.COHORTS)
        self.assertEqual(only_a["tails"]["under"], {"members": 0, "rows": 0, "mae": None})

    def test_build_coverage_groups_by_status(self):
        config = {"isoform": "CYP3A4", "covered_statuses": ["ok"], "groups": ["covered", "not_covered"]}
        built = agg.build_coverage(
            self.ROWS, config, lambda m: {"A": "ok", "B": "no_screening_row", "C": "ok"}[m]
        )
        self.assertEqual(built["groups"], {"A": "covered", "B": "not_covered", "C": "covered"})
        self.assertEqual(built["counts"], {"covered": 2, "not_covered": 1})
        with self.assertRaises(ValueError):
            agg.build_coverage(self.ROWS, {**config, "isoform": "CYP1A2"}, lambda m: "ok")


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.score = {"unchanged": 0.70, "q3": 0.69, "rank": 0.71}
        self.config_path = self.root / "evaluation.json"
        self.config_path.write_text(json.dumps({**CONFIG, "baseline_o150": 0.70}))
        for arm, value in self.score.items():
            self.write_arm(arm, value, shift={"unchanged": 0.0, "q3": 0.1, "rank": 0.05}[arm])
        self.sums = self.root / "sums.txt"
        directory = self.root / "pred_unchanged"
        self.sums.write_text("".join(
            f"{sha(p.read_bytes())}  {p.name}\n" for p in sorted(directory.iterdir())
        ))

    def write_arm(self, arm, value, shift, screen="s" * 64):
        cells = self.root / f"cells_{arm}"
        predictions = self.root / f"pred_{arm}"
        predictions.mkdir()
        rows = prediction_rows(shift)
        for index in range(150):
            cell = cells / f"cell_{index:03d}"
            cell.mkdir(parents=True)
            checkpoint = sha(f"{arm}{index}".encode())
            result = {
                "cell_index": index, "repetition": 1 + index // 30, "fold": (index // 3) % 10,
                "seed": SEEDS[index % 3], "status": "completed", "split_manifest_sha256": "m",
                "screen_checkpoint_sha256": screen, "macro_st_rae": value,
                "arm": CONFIG["arm_labels"][arm], "checkpoint_sha256": checkpoint,
                "best_validation": {
                    "macro_st_rae": value,
                    "per_task": {i + CONFIG["task_suffix"]: value for i in ISOFORMS},
                },
            }
            text = json.dumps(result).encode()
            (cell / "result.json").write_bytes(text)
            (cell / "COMPLETED.json").write_text(
                json.dumps({"status": "completed", "result_sha256": sha(text)})
            )
            mine = [(k, v) for k, v in rows.items() if k[0] == index]
            with (predictions / f"predictions_cell_{index:03d}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["cell_index", "isoform", "Molecule_Name",
                                 "y_true", "y_pred", "conf_low", "conf_high"])
                for (_, isoform, molecule), values in mine:
                    writer.writerow([index, isoform, molecule, *map(repr, values)])
            (predictions / f"reproduction_cell_{index:03d}.json").write_text(json.dumps({
                "cell_index": index, "status": "reproduced",
                "checkpoint_sha256": checkpoint, "prediction_rows": len(mine),
            }))

    def run_main(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            agg.main([str(a) for a in args])
        return output.getvalue()

    def cohorts(self):
        path = self.root / "cohorts.json"
        self.run_main("cohorts", "--baseline-predictions", self.root / "pred_unchanged",
                      "--sha256sums", self.sums, "--config", self.config_path, "--output", path)
        return path

    def coverage(self):
        path = self.root / "coverage.json"
        if not path.exists():
            screen_tar(self.root / "screen.tar.gz")
            self.run_main("coverage", "--baseline-predictions", self.root / "pred_unchanged",
                          "--sha256sums", self.sums, "--config", self.config_path,
                          "--coverage-config", COVERAGE_CONFIG_PATH,
                          "--penalty-config", PENALTY_CONFIG_PATH,
                          "--screen-tar", self.root / "screen.tar.gz", "--output", path)
        return path

    def evaluate(self, cohorts, output="report.json", digest=None, cells=None, coverage=None):
        coverage = coverage or self.coverage()
        arguments = ["evaluate", "--config", self.config_path, "--cohorts", cohorts,
                     "--cohorts-sha256", digest or sha(cohorts.read_bytes()),
                     "--coverage", coverage, "--coverage-sha256", sha(coverage.read_bytes()),
                     "--output", self.root / output]
        for arm in agg.ARMS:
            arguments += ["--cells", f"{arm}={(cells or {}).get(arm, self.root / f'cells_{arm}')}",
                          "--predictions", f"{arm}={self.root / f'pred_{arm}'}"]
        self.run_main(*arguments)
        return json.loads((self.root / output).read_text())

    def test_cohorts_then_evaluate_selects_the_passing_arm(self):
        report = self.evaluate(self.cohorts())
        self.assertEqual(report["decision"]["selected"], "q3")
        self.assertTrue(report["decision"]["arms"]["q3"]["passes"])
        self.assertFalse(report["decision"]["arms"]["rank"]["clauses"]["a_lower_mean_macro_st_rae"])
        tails = report["metrics"]["q3"]["tails"]["CYP1A2"]
        self.assertLess(tails["over"]["mae"], report["metrics"]["unchanged"]["tails"]["CYP1A2"]["over"]["mae"])
        self.assertEqual(tails["over"]["rows"], 30)

    def test_coverage_groups_are_built_from_the_screening_inputs_and_reported(self):
        built = json.loads(self.coverage().read_text())
        self.assertEqual({m for m, g in built["groups"].items() if g == "covered"}, COVERED)
        self.assertEqual(built["counts"], {"covered": 10, "not_covered": 10})
        self.assertEqual(built["status_counts"], {"no_screening_row": 10, "ok": 10})
        report = self.evaluate(self.cohorts())
        groups = report["coverage_groups"]
        self.assertEqual(groups["isoform"], "CYP3A4")
        for arm in agg.ARMS:
            self.assertEqual(groups["groups"][arm]["covered"]["molecules"], 10)
            self.assertEqual(groups["groups"][arm]["not_covered"]["rows"], 150)
        for name in ("covered", "not_covered"):
            difference = groups["differences_from_unchanged"]["q3"][name]
            self.assertLess(difference["mae"], 0)  # q3 shrinks the positive errors in both groups
            self.assertLess(difference["signed_error"], 0)
            self.assertEqual(groups["differences_from_unchanged"]["rank"][name]["over_tail_mae"] < 0, True)
        # grouping never enters the adoption rule
        self.assertEqual(report["decision"]["selected"], "q3")

    def test_coverage_file_is_pinned_and_must_match_the_baseline(self):
        path = self.cohorts()
        coverage = self.coverage()
        with self.assertRaisesRegex(ValueError, "Coverage file does not match its pinned"):
            self.run_main("evaluate", "--config", self.config_path, "--cohorts", path,
                          "--cohorts-sha256", sha(path.read_bytes()), "--coverage", coverage,
                          "--coverage-sha256", "0" * 64, "--output", self.root / "x.json",
                          *sum([["--cells", f"{a}={self.root / f'cells_{a}'}",
                                 "--predictions", f"{a}={self.root / f'pred_{a}'}"] for a in agg.ARMS], []))
        changed = json.loads(coverage.read_text())
        del changed["groups"]["M05"]
        other = self.root / "coverage_other.json"
        other.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "exactly the baseline's molecules"):
            self.evaluate(path, coverage=other)

    def test_pinned_cohorts_and_checksums_are_enforced(self):
        path = self.cohorts()
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            self.evaluate(path, digest="0" * 64)
        target = self.root / "pred_unchanged" / "predictions_cell_000.csv"
        target.write_text(target.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum file"):
            self.run_main("cohorts", "--baseline-predictions", self.root / "pred_unchanged",
                          "--sha256sums", self.sums, "--config", self.config_path,
                          "--output", self.root / "again.json")

    def test_mismatched_inputs_are_refused(self):
        path = self.cohorts()
        with self.assertRaisesRegex(ValueError, "arm label"):
            self.evaluate(path, cells={"q3": self.root / "cells_rank"})
        record = self.root / "pred_rank" / "reproduction_cell_005.json"
        original = record.read_text()
        record.write_text(original.replace(sha(b"rank5"), "f" * 64))
        with self.assertRaisesRegex(ValueError, "best checkpoints"):
            self.evaluate(path)
        record.write_text(original.replace("reproduced", "MISMATCH"))
        with self.assertRaisesRegex(ValueError, "not a reproduced record"):
            self.evaluate(path)

    def test_a_different_screening_checkpoint_or_baseline_score_is_refused(self):
        path = self.cohorts()
        import shutil
        shutil.rmtree(self.root / "cells_q3")
        shutil.rmtree(self.root / "pred_q3")
        self.write_arm("q3", 0.69, 0.1, screen="t" * 64)
        with self.assertRaisesRegex(ValueError, "same screening checkpoint"):
            self.evaluate(path)
        self.config_path.write_text(json.dumps({**CONFIG, "baseline_o150": 0.5}))
        shutil.rmtree(self.root / "cells_q3")
        shutil.rmtree(self.root / "pred_q3")
        self.write_arm("q3", 0.69, 0.1)
        with self.assertRaisesRegex(ValueError, "does not reproduce its recorded O150"):
            self.evaluate(path)


if __name__ == "__main__":
    unittest.main()
