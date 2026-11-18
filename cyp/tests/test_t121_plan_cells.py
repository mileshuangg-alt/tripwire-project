from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qualification" / "t121_executor_source_20261117"))
import t121_plan_cells as planner
from cyp007_production_helpers import cell_keys


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.proposal = self.root / "proposal.json"
        self.proposal.write_text('{"trial_index": 121}')
        self.sha = hashlib.sha256(self.proposal.read_bytes()).hexdigest()

    def plan(self, **overrides):
        arguments = dict(
            run_root=self.root / "run", proposal_path=self.proposal,
            proposal_sha256=self.sha, cell_keys=cell_keys(),
        )
        arguments.update(overrides)
        return planner.plan_cells(**arguments)

    def test_150_cells_are_planned_with_the_production_layout(self):
        plans = self.plan()
        self.assertEqual([p["cell_index"] for p in plans], list(range(150)))
        trial = self.root / "run/trials/trial_121"
        self.assertEqual((trial / "proposal.json").read_bytes(), self.proposal.read_bytes())
        first = json.loads((trial / "cells/cell_000/cell_plan.json").read_text())
        self.assertEqual(
            (first["repetition"], first["fold"], first["seed"]), (1, 1, 20261101)
        )
        last = json.loads((trial / "cells/cell_149/cell_plan.json").read_text())
        self.assertEqual(
            (last["repetition"], last["fold"], last["seed"]), (5, 10, 20261103)
        )
        for key in (
            "cell_index", "trial_index", "repetition", "fold", "seed",
            "status", "proposal_sha256_source", "fidelity",
        ):
            self.assertIn(key, first)
        ready = json.loads((trial / "cell_array_ready.json").read_text())
        self.assertEqual((ready["cell_count"], ready["fidelity"]), (150, "full"))

    def test_wrong_proposal_hash_blocks_before_anything_is_written(self):
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            self.plan(proposal_sha256="0" * 64)
        self.assertFalse((self.root / "run").exists())

    def test_existing_plan_directory_is_not_reused(self):
        self.plan()
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_partial_or_duplicate_grid_is_rejected(self):
        for keys in (cell_keys()[:30], cell_keys()[:149] + [cell_keys()[0]]):
            with self.assertRaisesRegex(ValueError, "150 distinct"):
                self.plan(cell_keys=keys)
        self.assertFalse((self.root / "run").exists())


if __name__ == "__main__":
    unittest.main()
