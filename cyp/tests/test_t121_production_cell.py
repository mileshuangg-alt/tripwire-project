from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import t121_production_cell as cell_module

SAMPLED = {"batch_size": 64, "warmup_epochs": 10}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload))


class CellTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.root = root
        self.run_root = root / "run"
        self.trial_dir = self.run_root / "trials" / "trial_121"
        self.cell_dir = self.trial_dir / "cells" / "cell_007"
        write(self.cell_dir / "cell_plan.json", {
            "trial_index": 121, "cell_index": 7,
            "repetition": 1, "fold": 2, "seed": 20261101,
        })
        write(self.trial_dir / "proposal.json", {
            "sampled_config": SAMPLED, "config_name": "D007-T121",
        })
        self.manifests = root / "manifests"
        write(self.manifests / "rep_01_fold_02.json", {
            "repetition": 1, "fold": 2, "training_seeds": [20261101, 20261102],
        })
        self.dataset = root / "data.csv"
        self.dataset.write_text("a\n1\n")
        self.split = root / "split.csv"
        self.split.write_text("b\n1\n")
        self.checkpoint = root / "screen.ckpt"
        self.checkpoint.write_bytes(b"screen")
        self.selection = root / "selected.json"
        write(self.selection, {
            "epoch": 31, "score": 0.5,
            "checkpoint": str(self.checkpoint), "sha256": sha(self.checkpoint),
        })
        self.calls = []

    def ops(self, screen_config=None, fail_in=None):
        calls = self.calls
        fit_dir = self.cell_dir / "fit"

        def route_fold(**kwargs):
            calls.append("route")
            self.routed_input = kwargs["dataframe"]
            return {"training_dataframe": "T", "validation_dataframe": "V"}

        def prepare_screen_stage(**kwargs):
            calls.append("screen_shell")
            self.screen_kwargs = kwargs
            return {"sampled_config": screen_config or SAMPLED}

        def handoff(**kwargs):
            calls.append("handoff")
            self.handoff_kwargs = kwargs
            return {
                "screen_to_pic50_transition": {"copied": 12},
                "updated_t121_weighting": {"evaluation_weighted": False},
            }

        def fit_stage(**kwargs):
            calls.append("fit")
            self.fit_kwargs = kwargs
            if fail_in == "fit":
                raise RuntimeError("fit failed")
            fit_dir.mkdir(parents=True)
            best = fit_dir / "best.ckpt"
            best.write_bytes(b"best")
            history = [{"epoch": 0, "macro_st_rae": 0.9}]
            return {
                "checkpoint_path": best,
                "best_validation": {"macro_st_rae": 0.9},
                "validation_history": history,
                "epochs_completed": 1,
            }

        import json as _json
        return SimpleNamespace(
            read_csv=lambda path: "frame",
            read_json=lambda p: _json.loads(Path(p).read_text()),
            write_json_atomic=lambda p, v: Path(p).write_text(_json.dumps(v)),
            write_json_exclusive=self.exclusive,
            sha256_file=sha,
            cell_paths=lambda run_root, t, c: (self.trial_dir, self.cell_dir),
            route_fold=route_fold,
            prepare_screen_stage=prepare_screen_stage,
            prepare_selected_screen_handoff=handoff,
            fit_stage=fit_stage,
        )

    @staticmethod
    def exclusive(path, payload):
        with Path(path).open("x") as handle:
            json.dump(payload, handle)

    def run_cell(self, **overrides):
        arguments = dict(
            run_root=self.run_root, trial_index=121, cell_index=7,
            dataset_path=self.dataset, original_split_path=self.split,
            split_manifest_root=self.manifests,
            screen_selection_path=self.selection,
            screen_selection_sha256=sha(self.selection),
            screen_proposal_path=self.root / "proposal.json",
            screen_inputs={"screen_path": "s"},
            authorize_training=True, ops=self.ops(),
        )
        arguments.update(overrides)
        return cell_module.run_updated_t121_cell(**arguments)

    def test_completed_cell_matches_the_production_cell_schema(self):
        result = self.run_cell()
        self.assertEqual(self.calls, ["route", "screen_shell", "handoff", "fit"])
        for key in (
            "trial_index", "cell_index", "repetition", "fold", "seed", "status",
            "macro_st_rae", "best_validation", "checkpoint_path",
            "checkpoint_sha256", "split_manifest_path",
            "split_manifest_sha256", "proposal_path", "config_name",
        ):
            self.assertIn(key, result)
        self.assertEqual(result["screen_selected_epoch"], 31)
        self.assertTrue(self.fit_kwargs["authorize_training"])
        self.assertEqual(self.handoff_kwargs["selected_record"]["epoch"], 31)
        completion = json.loads((self.cell_dir / "COMPLETED.json").read_text())
        self.assertEqual(
            completion["result_sha256"], sha(self.cell_dir / "result.json")
        )
        provenance = json.loads((self.cell_dir / "provenance.json").read_text())
        self.assertEqual(provenance["screen_to_pic50_transition"], {"copied": 12})
        self.assertFalse(provenance["updated_t121_weighting"]["evaluation_weighted"])

    def test_authorization_is_required(self):
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            self.run_cell(authorize_training=False)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.cell_dir / "STARTED.json").exists())

    def test_changed_selection_record_blocks_before_start(self):
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            self.run_cell(screen_selection_sha256="0" * 64)
        self.assertFalse((self.cell_dir / "STARTED.json").exists())

    def test_screen_and_cell_proposals_must_agree(self):
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            self.run_cell(ops=self.ops(screen_config={"batch_size": 32}))
        self.assertTrue((self.cell_dir / "INCOMPLETE.json").exists())
        self.assertNotIn("fit", self.calls)

    def test_failure_writes_incomplete_and_blocks_a_blind_retry(self):
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            self.run_cell(ops=self.ops(fail_in="fit"))
        incomplete = json.loads((self.cell_dir / "INCOMPLETE.json").read_text())
        self.assertTrue(incomplete["requires_human_adjudication"])
        self.assertFalse((self.cell_dir / "COMPLETED.json").exists())
        with self.assertRaisesRegex(RuntimeError, "STARTED without COMPLETED"):
            self.run_cell()

    def test_completed_cell_is_returned_without_refitting(self):
        first = self.run_cell()
        self.calls.clear()
        again = self.run_cell()
        self.assertEqual(again, first)
        self.assertEqual(self.calls, [])

    def test_arm_r_passes_no_descriptor_arguments_and_keeps_its_arm_label(self):
        result = self.run_cell()
        self.assertNotIn("descriptor_cache", self.screen_kwargs)
        self.assertNotIn("descriptor_dim", self.handoff_kwargs)
        self.assertEqual(result["arm"], "updated_t121_from_screen_checkpoint")
        provenance = json.loads((self.cell_dir / "provenance.json").read_text())
        self.assertNotIn("descriptors", provenance)

    def test_arm_d_routes_the_cache_to_both_stages_and_labels_the_arm(self):
        class Cache:
            dim = 196

            def provenance(self):
                return {"manifest_sha256": "abc", "dim": 196}

        attached = []
        ops = self.ops()
        ops.attach_descriptors = lambda frame, cache: attached.append((frame, cache)) or "frame+xd"
        cache = Cache()
        result = self.run_cell(ops=ops, descriptor_cache=cache)
        # Routing sees the raw dataset (protected rows included, not in the
        # cache); descriptors attach only to the routed training/validation frames.
        self.assertEqual(self.routed_input, "frame")
        self.assertEqual(attached, [("T", cache), ("V", cache)])
        self.assertIs(self.screen_kwargs["descriptor_cache"], cache)
        self.assertEqual(self.handoff_kwargs["descriptor_dim"], 196)
        self.assertEqual(result["arm"], "updated_t121_descriptors_from_screen_checkpoint")
        started = json.loads((self.cell_dir / "STARTED.json").read_text())
        self.assertEqual(started["arm"], "updated_t121_descriptors_from_screen_checkpoint")
        provenance = json.loads((self.cell_dir / "provenance.json").read_text())
        self.assertEqual(provenance["descriptors"]["manifest_sha256"], "abc")


if __name__ == "__main__":
    unittest.main()
