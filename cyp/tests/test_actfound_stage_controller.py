
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from actfound_stage_controller import run_actfound_stage_chain


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = self.root / "qualification.json"
        self.evidence.write_text(json.dumps({
            "status": "PASS",
            "scope": "GPU fixed-episode validation and checkpoint correctness",
            "optimizer_steps": 0,
            "runner_sha256": "a" * 64,
            "model_side_sha256": "b" * 64,
            "fixture_only": True,
        }))
        self.calls = []

    def pins(self, **fields):
        bound = {"runner_sha256": "a" * 64, "model_side_sha256": "b" * 64}
        bound.update(fields)
        return {
            "evidence_sha256": hashlib.sha256(
                self.evidence.read_bytes()
            ).hexdigest(),
            "bound_fields": bound,
        }

    def executor(self, name):
        def execute(*, output_dir, initialization):
            self.calls.append((name, initialization))
            if name == "pic50":
                artifact = output_dir / "fixture_evaluation_result.json"
                artifact.write_text(json.dumps({"fixture_only": True}))
                manifest = output_dir / "evaluation_manifest.json"
                manifest.write_text(json.dumps({
                    "status": "completed",
                    "fixture_only": True,
                    "artifacts": [artifact.name],
                }))
                return {"evaluation_manifest": str(manifest)}
            checkpoint = output_dir / "selected.pt"
            checkpoint.write_bytes(("fixture:" + name).encode())
            return {"selected_checkpoint": str(checkpoint)}
        return execute

    def executors(self):
        return {
            name: self.executor(name)
            for name in ("foundation", "screening", "pic50")
        }

    def run_chain(self, executors=None, **overrides):
        arguments = dict(
            output_dir=self.root / "run",
            executors=self.executors() if executors is None else executors,
            qualification_result_path=self.evidence,
            evidence_pins=self.pins(),
            authorize_training=True,
        )
        arguments.update(overrides)
        return run_actfound_stage_chain(**arguments)

    def assert_blocked_before_run(self):
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "run").exists())

    def test_evidence_for_other_runner_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "runner_sha256"):
            self.run_chain(evidence_pins=self.pins(runner_sha256="c" * 64))
        self.assert_blocked_before_run()

    def test_evidence_file_changed_after_pinning_is_rejected(self):
        pins = self.pins()
        evidence = json.loads(self.evidence.read_text())
        evidence["fixture_only"] = False
        self.evidence.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            self.run_chain(evidence_pins=pins)
        self.assert_blocked_before_run()

    def test_evidence_missing_bound_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lacks bound field"):
            self.run_chain(evidence_pins=self.pins(unrecorded_sha256="d" * 64))
        self.assert_blocked_before_run()

    def test_empty_or_malformed_pins_are_rejected(self):
        for pins in (
            {"evidence_sha256": self.pins()["evidence_sha256"], "bound_fields": {}},
            {"bound_fields": {"runner_sha256": "a" * 64}},
            None,
        ):
            with self.assertRaises(ValueError):
                self.run_chain(evidence_pins=pins)
        self.assert_blocked_before_run()

    def test_bound_fields_are_recorded_in_the_run(self):
        self.run_chain()
        reference = json.loads(
            (self.root / "run" / "qualification_reference.json").read_text()
        )
        self.assertEqual(reference["bound_fields"], self.pins()["bound_fields"])

    def test_stage_order_and_checkpoint_lineage(self):
        result = self.run_chain()
        self.assertEqual(
            [name for name, _ in self.calls],
            ["foundation", "screening", "pic50"],
        )
        self.assertIsNone(self.calls[0][1]["source_checkpoint"])
        for index, previous in ((1, "foundation"), (2, "screening")):
            initialization = self.calls[index][1]
            self.assertEqual(
                initialization["source_checkpoint"],
                result["stages"][previous]["selected_checkpoint"],
            )
            self.assertFalse(initialization["import_optimizer_state"])
            self.assertFalse(initialization["import_scheduler_state"])
            self.assertFalse(initialization["import_epoch_state"])

    def test_authorization_blocks_before_execution(self):
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            self.run_chain(authorize_training=False)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "run").exists())

    def test_pending_qualification_blocks_execution(self):
        self.evidence.write_text(json.dumps({"status": "PENDING"}))
        with self.assertRaises(ValueError):
            self.run_chain()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "run").exists())

    def test_failure_stops_downstream_and_records_error(self):
        executors = self.executors()
        def fail(**kwargs):
            raise RuntimeError("fixture screening failure")
        executors["screening"] = fail
        with self.assertRaisesRegex(RuntimeError, "fixture screening failure"):
            self.run_chain(executors)
        self.assertFalse((self.root / "run/pic50").exists())
        self.assertTrue((self.root / "run/screening/failure.json").is_file())
        self.assertFalse((self.root / "run/completion.json").exists())

    def test_missing_checkpoint_stops_chain(self):
        executors = self.executors()
        executors["foundation"] = lambda **kwargs: {
            "selected_checkpoint": str(self.root / "absent.pt")
        }
        with self.assertRaises(ValueError):
            self.run_chain(executors)
        self.assertFalse((self.root / "run/screening").exists())

    def test_existing_run_is_not_implicitly_resumed(self):
        self.run_chain()
        count = len(self.calls)
        with self.assertRaises(FileExistsError):
            self.run_chain()
        self.assertEqual(len(self.calls), count)

    def test_mutated_input_checkpoint_is_detected(self):
        executors = self.executors()
        ordinary = executors["screening"]
        def mutate(*, output_dir, initialization):
            Path(initialization["source_checkpoint"]["path"]).write_bytes(
                b"unexpected mutation"
            )
            return ordinary(
                output_dir=output_dir, initialization=initialization
            )
        executors["screening"] = mutate
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_chain(executors)
        self.assertFalse((self.root / "run/pic50").exists())



    def test_pic50_returns_manifest_not_selected_checkpoint(self):
        result = self.run_chain()
        endpoint = result["stages"]["pic50"]
        self.assertNotIn("selected_checkpoint", endpoint)
        self.assertEqual(len(endpoint["evaluation_artifacts"]), 1)
        self.assertTrue(Path(endpoint["evaluation_manifest"]["path"]).is_file())

    def test_pic50_checkpoint_only_is_rejected(self):
        executors = self.executors()
        def checkpoint_only(*, output_dir, initialization):
            checkpoint = output_dir / "incorrect_endpoint.pt"
            checkpoint.write_bytes(b"fixture")
            return {"selected_checkpoint": str(checkpoint)}
        executors["pic50"] = checkpoint_only
        with self.assertRaisesRegex(ValueError, "omitted evaluation_manifest"):
            self.run_chain(executors)
        self.assertFalse((self.root / "run/completion.json").exists())

    def test_incomplete_evaluation_is_rejected(self):
        executors = self.executors()
        def incomplete(*, output_dir, initialization):
            manifest = output_dir / "evaluation_manifest.json"
            manifest.write_text(json.dumps({
                "status": "incomplete",
                "artifacts": [],
            }))
            return {"evaluation_manifest": str(manifest)}
        executors["pic50"] = incomplete
        with self.assertRaisesRegex(ValueError, "not completed"):
            self.run_chain(executors)

    def test_missing_evaluation_artifact_is_rejected(self):
        executors = self.executors()
        def missing(*, output_dir, initialization):
            manifest = output_dir / "evaluation_manifest.json"
            manifest.write_text(json.dumps({
                "status": "completed",
                "artifacts": ["absent.json"],
            }))
            return {"evaluation_manifest": str(manifest)}
        executors["pic50"] = missing
        with self.assertRaises(ValueError):
            self.run_chain(executors)
        self.assertFalse((self.root / "run/completion.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
