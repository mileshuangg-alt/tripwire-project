from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import t121_screen_chain_run as launcher


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScreenLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        pinned = {}
        for role, name in (
            ("screen_stage", "t121_screen_stage.py"),
            ("weighted_training", "t121_weighted_training.py"),
            ("proposal", "proposal.json"),
        ):
            path = self.root / name
            path.write_text(f"# {role}\n")
            pinned[role] = {"path": str(path), "sha256": sha(path)}
        check = self.root / "check.json"
        check.write_text(json.dumps({"status": "PASS", "module_sha256": "m" * 8}))
        self.check = check
        self.manifest = {
            "evidence": [{
                "path": str(check), "sha256": sha(check),
                "bound_fields": {"module_sha256": "m" * 8},
            }],
            "output_dir": str(self.root / "screen_run"),
            "training_seed": 20261101,
            "proposal_path": pinned["proposal"]["path"],
            "pinned_files": pinned,
            "screen_inputs": {k: "x" for k in launcher.SCREEN_INPUT_KEYS},
        }
        self.calls = []

    def stage(self, bad_hash=False):
        calls = self.calls

        def prepare_screen_stage(**kwargs):
            calls.append(("prepare", kwargs))
            return {"output_dir": Path(kwargs["output_dir"])}

        def fit_screen_stage(prepared, *, authorize_training=False):
            calls.append(("fit", authorize_training))
            directory = prepared["output_dir"] / "epoch_checkpoints"
            directory.mkdir(parents=True)
            checkpoint = directory / "epoch_007.ckpt"
            checkpoint.write_bytes(b"ckpt")
            record = {
                "epoch": 7, "score": 0.4, "per_enzyme": {},
                "checkpoint": str(checkpoint),
                "sha256": "0" * 64 if bad_hash else sha(checkpoint),
            }
            (directory / "selection.json").write_text(json.dumps({
                "records": [record], "selected": record,
            }))
            return record

        return SimpleNamespace(
            prepare_screen_stage=prepare_screen_stage,
            fit_screen_stage=fit_screen_stage,
        )

    def test_evidence_passes_and_records_hashes(self):
        verified = launcher.verify_evidence(self.manifest["evidence"])
        self.assertEqual(verified[0]["evidence_sha256"], sha(self.check))

    def test_evidence_for_other_module_is_rejected(self):
        entry = copy.deepcopy(self.manifest["evidence"])
        entry[0]["bound_fields"]["module_sha256"] = "x" * 8
        with self.assertRaisesRegex(ValueError, "module_sha256"):
            launcher.verify_evidence(entry)

    def test_changed_or_failing_evidence_is_rejected(self):
        entry = copy.deepcopy(self.manifest["evidence"])
        self.check.write_text(json.dumps({"status": "FAIL"}))
        with self.assertRaisesRegex(ValueError, "did not pass"):
            launcher.verify_evidence(entry)
        self.check.write_text(json.dumps({"status": "PASS", "module_sha256": "m" * 8, "x": 1}))
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            launcher.verify_evidence(entry)

    def test_empty_evidence_list_is_rejected(self):
        broken = copy.deepcopy(self.manifest)
        broken["evidence"] = []
        with self.assertRaises(ValueError):
            launcher.validate_manifest(broken)

    def test_manifest_validates(self):
        launcher.validate_manifest(self.manifest)

    def test_seed_must_be_explicit_integer(self):
        for seed in (None, "20261101", True, 1.5):
            broken = copy.deepcopy(self.manifest)
            broken["training_seed"] = seed
            with self.assertRaises(ValueError):
                launcher.validate_manifest(broken)

    def test_missing_extra_key_or_role_is_rejected(self):
        broken = copy.deepcopy(self.manifest)
        del broken["screen_inputs"]["contract_path"]
        with self.assertRaises(ValueError):
            launcher.validate_manifest(broken)
        broken = copy.deepcopy(self.manifest)
        broken["default_seed"] = 1
        with self.assertRaises(ValueError):
            launcher.validate_manifest(broken)
        broken = copy.deepcopy(self.manifest)
        del broken["pinned_files"]["weighted_training"]
        with self.assertRaises(ValueError):
            launcher.validate_manifest(broken)

    def test_proposal_path_must_be_the_pinned_file(self):
        broken = copy.deepcopy(self.manifest)
        broken["proposal_path"] = str(self.root / "other.json")
        with self.assertRaisesRegex(ValueError, "pinned proposal"):
            launcher.validate_manifest(broken)

    def test_selected_record_is_written_and_completion_summarizes_it(self):
        completion = launcher.run_screen_chain(self.manifest, self.stage())
        self.assertEqual([c[0] for c in self.calls], ["prepare", "fit"])
        self.assertTrue(self.calls[1][1])
        self.assertEqual(self.calls[0][1]["training_seed"], 20261101)
        record = Path(self.manifest["output_dir"]) / "selected_screen_record.json"
        self.assertEqual(json.loads(record.read_text())["epoch"], 7)
        self.assertEqual(completion["selected_record_sha256"], sha(record))
        self.assertFalse(completion["full_screen_refit"])

    def test_existing_output_is_not_reused(self):
        Path(self.manifest["output_dir"]).mkdir()
        with self.assertRaises(FileExistsError):
            launcher.run_screen_chain(self.manifest, self.stage())
        self.assertEqual(self.calls, [])

    def test_selected_checkpoint_hash_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            launcher.run_screen_chain(self.manifest, self.stage(bad_hash=True))
        self.assertFalse(
            (Path(self.manifest["output_dir"]) / "selected_screen_record.json").exists()
        )

    def test_without_authorize_flag_nothing_is_imported_or_run(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        before = set(sys.modules)
        self.assertEqual(launcher.main(["--manifest", str(path)]), 0)
        self.assertEqual(set(sys.modules) - before, set())
        self.assertFalse(Path(self.manifest["output_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
