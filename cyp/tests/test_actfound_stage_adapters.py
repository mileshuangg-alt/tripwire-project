
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import actfound_stage_adapters as a


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "stage"
        self.output.mkdir()
        self.source = self.root / "selected.pt"
        self.source.write_bytes(b"fixture checkpoint")
        self.initialization = {
            "mode": "stage_transfer_not_resume",
            "source_checkpoint": {
                "path": str(self.source),
                "sha256": a.sha256(self.source),
            },
            "import_optimizer_state": False,
            "import_scheduler_state": False,
            "import_epoch_state": False,
        }

    def pic50(self):
        paths = {}
        for name in (
            "evaluation_config", "qualification_config", "counts",
            "frozen_train_config", "freeze_record", "label_table",
        ):
            path = self.root / (name + ".json")
            path.write_text("{}")
            paths[name] = path
        return a.pic50_adapter(
            actfound_python="/fixture/actfound/python",
            scoring_python="/fixture/scoring/python",
            evaluation_dir=self.root,
            manifest_dir=self.root / "manifests",
            repo_root=self.root,
            upstream_dir=self.root / "upstream",
            **paths,
        )

    def fake_execute(self, command, log_path):
        command = [str(value) for value in command]
        Path(log_path).write_text("fixture subprocess output\n")
        destination = Path(command[command.index("--out-dir") + 1])
        destination.mkdir()
        if command[2].endswith("run_evaluation.py"):
            a.write_new_json(
                destination / "cyp_adaptation_predictions.json",
                {"unique_adaptations": 200, "seed_slots_expanded_at_scoring": 3},
            )
        else:
            a.write_new_json(destination / "cyp_o150_result.json", {
                "cell_count": 150,
                "seed_slots_are_replicates": "fixture: replicas, not independent fits",
                "comparison_is_final": False,
                "o150": 0.75,
            })

    def test_foundation_calls_existing_runner_without_hash_bypass(self):
        config_path = self.root / "frozen.json"
        config_path.write_text("{}")
        calls = []
        def train(config, repo, upstream, skip_hash):
            calls.append((config, skip_hash))
            out = Path(config["output_dir"])
            out.mkdir()
            checkpoint = out / "model_best.pt"
            checkpoint.write_bytes(b"fixture")
            a.write_new_json(out / "completion.json", {
                "status": "completed", "best_checkpoint": str(checkpoint)
            })
        runner = SimpleNamespace(
            validate_and_resolve_config=lambda path: {
                "hyperparameters": {"fixture": True}
            },
            run_training=train,
        )
        execute = a.foundation_adapter(
            runner=runner, frozen_config_path=config_path,
            repo_root=self.root, upstream_dir=self.root,
        )
        initialization = dict(self.initialization)
        initialization.update(
            mode="original_runner_defined_initialization",
            source_checkpoint=None,
        )
        result = execute(output_dir=self.output, initialization=initialization)
        self.assertTrue(Path(result["selected_checkpoint"]).is_file())
        self.assertFalse(calls[0][1])

    def test_foundation_rejects_historical_checkpoint(self):
        execute = a.foundation_adapter(
            runner=None, frozen_config_path="unused",
            repo_root="unused", upstream_dir="unused",
        )
        with self.assertRaises(ValueError):
            execute(output_dir=self.output, initialization=self.initialization)

    def test_pic50_uses_two_interpreters_and_preserves_replica_semantics(self):
        execute = self.pic50()
        with patch.object(a, "run_logged", side_effect=self.fake_execute) as mocked:
            result = execute(
                output_dir=self.output, initialization=self.initialization
            )
        self.assertEqual(mocked.call_count, 2)
        first = mocked.call_args_list[0].args[0]
        second = mocked.call_args_list[1].args[0]
        self.assertEqual(first[0], "/fixture/actfound/python")
        self.assertEqual(second[0], "/fixture/scoring/python")
        self.assertEqual(
            str(first[first.index("--checkpoint") + 1]), str(self.source)
        )
        manifest = a.read_json(result["evaluation_manifest"])
        self.assertEqual(manifest["evaluation_slots"], 150)
        self.assertEqual(manifest["unique_endpoint_adaptations"], 200)
        self.assertFalse(manifest["comparison_is_final"])
        for name, digest in manifest["artifact_sha256"].items():
            self.assertEqual(a.sha256(self.output / name), digest)

    def test_changed_checkpoint_blocks_subprocesses(self):
        execute = self.pic50()
        self.source.write_bytes(b"changed")
        with patch.object(a, "run_logged") as mocked:
            with self.assertRaises(ValueError):
                execute(output_dir=self.output, initialization=self.initialization)
        mocked.assert_not_called()

    def test_prediction_failure_prevents_scoring_and_completion(self):
        execute = self.pic50()
        with patch.object(a, "run_logged", side_effect=RuntimeError("fixture")) as mocked:
            with self.assertRaises(RuntimeError):
                execute(output_dir=self.output, initialization=self.initialization)
        self.assertEqual(mocked.call_count, 1)
        self.assertFalse((self.output / "evaluation_manifest.json").exists())

    def test_wrong_adaptation_count_blocks_scoring(self):
        execute = self.pic50()
        def wrong(command, log_path):
            self.fake_execute(command, log_path)
            path = self.output / "predictions/cyp_adaptation_predictions.json"
            path.write_text(json.dumps({
                "unique_adaptations": 199,
                "seed_slots_expanded_at_scoring": 3,
            }))
        with patch.object(a, "run_logged", side_effect=wrong) as mocked:
            with self.assertRaises(ValueError):
                execute(output_dir=self.output, initialization=self.initialization)
        self.assertEqual(mocked.call_count, 1)

    def test_transition_rejects_optimizer_resume(self):
        execute = self.pic50()
        self.initialization["import_optimizer_state"] = True
        with self.assertRaises(ValueError):
            execute(output_dir=self.output, initialization=self.initialization)

    def test_logged_subprocess_failure_is_propagated(self):
        with patch.object(
            a.subprocess, "run",
            side_effect=subprocess.CalledProcessError(1, ["fixture"]),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                a.run_logged(["fixture"], self.output / "failure.log")
        self.assertTrue((self.output / "failure.log").is_file())


    def test_screening_config_serializes_the_real_make_args_namespace(self):
        # Regression: vars(make_args(...)) holds a live numpy RandomState, which
        # JSON cannot encode; job 3105596 failed on it after the foundation.
        import actfound_production_train_v2 as runner
        config = json.loads((
            Path(__file__).resolve().parents[1]
            / "artifacts/actfound/actfound_chembl32_production_config_v1.json"
        ).read_text())
        args = runner.make_args(dict(config["hyperparameters"]))
        self.assertTrue(hasattr(args, "rng"))
        effective = a.effective_screening_config(
            args, self.initialization, {"copied": 1}, {"screen": "fixture"}
        )
        self.assertNotIn("rng", effective["hyperparameters"])
        self.assertIn("torch_seed_source_seed", effective["hyperparameters"])
        a.write_new_json(self.output / "effective_screening_config.json", effective)
        self.assertEqual(
            json.loads((self.output / "effective_screening_config.json").read_text())[
                "source_checkpoint"
            ],
            self.initialization["source_checkpoint"],
        )

    def test_unserializable_value_leaves_no_partial_file(self):
        target = self.output / "config.json"
        with self.assertRaises(TypeError):
            a.write_new_json(target, {"ok": 1, "bad": object()})
        self.assertFalse(target.exists())
        a.write_new_json(target, {"ok": 1})
        self.assertTrue(target.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
