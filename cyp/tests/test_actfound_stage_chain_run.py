from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import actfound_stage_chain_run as launcher


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        pinned = {}
        for role in ("runner", "model_side", "stage_adapters", "stage_controller"):
            path = self.root / f"{role}.py"
            path.write_text(f"# {role}\n")
            pinned[role] = {"path": str(path), "sha256": digest(path)}
        evidence = self.root / "evidence.json"
        evidence.write_text("{}")
        self.manifest = {
            "output_dir": str(self.root / "run"),
            "pinned_files": pinned,
            "evidence": {
                "path": str(evidence),
                "sha256": digest(evidence),
                "bound_fields": {"runner_sha256": pinned["runner"]["sha256"]},
            },
            "foundation": {k: "x" for k in launcher.FOUNDATION_KEYS},
            "screening": {
                "frozen_config_path": "x",
                "upstream_dir": "x",
                "screen_inputs": {k: "x" for k in launcher.SCREEN_INPUT_KEYS},
            },
            "pic50": {k: "x" for k in launcher.PIC50_KEYS},
        }

    def write_manifest(self, manifest):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_complete_manifest_validates_and_files_verify(self):
        launcher.validate_manifest(self.manifest)
        verified = launcher.verify_pinned_files(self.manifest["pinned_files"])
        self.assertEqual(set(verified), set(self.manifest["pinned_files"]))

    def test_missing_or_extra_key_is_rejected(self):
        for section, key in (("pic50", "label_table"), ("foundation", "repo_root")):
            broken = copy.deepcopy(self.manifest)
            del broken[section][key]
            with self.assertRaises(ValueError):
                launcher.validate_manifest(broken)
        broken = copy.deepcopy(self.manifest)
        broken["unexpected_default"] = 1
        with self.assertRaises(ValueError):
            launcher.validate_manifest(broken)

    def test_required_pinned_role_is_enforced(self):
        broken = copy.deepcopy(self.manifest)
        del broken["pinned_files"]["model_side"]
        with self.assertRaises(ValueError):
            launcher.validate_manifest(broken)

    def test_changed_pinned_file_is_rejected(self):
        Path(self.manifest["pinned_files"]["runner"]["path"]).write_text("# edited\n")
        with self.assertRaisesRegex(ValueError, "Pinned file changed: runner"):
            launcher.verify_pinned_files(self.manifest["pinned_files"])

    def test_missing_pinned_file_is_rejected(self):
        Path(self.manifest["pinned_files"]["model_side"]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Pinned file missing"):
            launcher.verify_pinned_files(self.manifest["pinned_files"])

    def test_without_authorize_flag_nothing_is_imported_or_run(self):
        path = self.write_manifest(self.manifest)
        before = set(sys.modules)
        self.assertEqual(launcher.main(["--manifest", str(path)]), 0)
        self.assertEqual(set(sys.modules) - before, set())
        self.assertFalse((self.root / "run").exists())

    def test_evidence_pins_pass_through_unchanged(self):
        pins = launcher.evidence_pins(self.manifest)
        self.assertEqual(pins["evidence_sha256"], self.manifest["evidence"]["sha256"])
        self.assertEqual(pins["bound_fields"], self.manifest["evidence"]["bound_fields"])


class ContinuationManifestTests(unittest.TestCase):
    def setUp(self):
        base = LauncherTests()
        base.setUp()
        self.addCleanup(base.temporary.cleanup)
        self.manifest = copy.deepcopy(base.manifest)
        del self.manifest["foundation"]
        self.manifest["completed_foundation"] = {
            "run_dir": "x", "completion_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
        }

    def test_continuation_manifest_validates_without_foundation_section(self):
        self.assertTrue(launcher.is_continuation(self.manifest))
        launcher.validate_manifest(self.manifest)

    def test_continuation_rejects_foundation_section_and_missing_pins(self):
        broken = copy.deepcopy(self.manifest)
        broken["foundation"] = {k: "x" for k in launcher.FOUNDATION_KEYS}
        with self.assertRaises(ValueError):
            launcher.validate_manifest(broken)
        broken = copy.deepcopy(self.manifest)
        del broken["completed_foundation"]["checkpoint_sha256"]
        with self.assertRaises(ValueError):
            launcher.validate_manifest(broken)

    def test_continuation_builds_no_foundation_executor(self):
        made = []
        adapters = type("A", (), {
            "foundation_adapter": staticmethod(lambda **k: made.append("f")),
            "screening_adapter": staticmethod(lambda **k: "s"),
            "pic50_adapter": staticmethod(lambda **k: "p"),
        })
        executors = launcher.build_executors(self.manifest, object(), adapters)
        self.assertEqual(set(executors), {"screening", "pic50"})
        self.assertEqual(made, [])
        full = launcher.build_executors(
            {k: v for k, v in self.manifest.items() if k != "completed_foundation"}
            | {"foundation": {k: "x" for k in launcher.FOUNDATION_KEYS}},
            object(), adapters,
        )
        self.assertEqual(set(full), {"foundation", "screening", "pic50"})


if __name__ == "__main__":
    unittest.main()
