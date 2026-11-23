from pathlib import Path
import json
import re
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cyp/scripts"))
import t121_d022_layout as layout
import t121_screen_chain_run as launcher

REFERENCE = REPO / "screen_manifest_v2.json"


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.reference = layout.reference_manifest_checked(REFERENCE)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package = Path(self.temporary.name)
        wanted = {
            entry["path"][len(layout.R_PACKAGE) + 1:]
            for entry in self.reference["pinned_files"].values()
            if entry["path"].startswith(layout.R_PACKAGE + "/")
        }
        wanted |= set(layout.C_EDITED_ROLES.values())
        for relative in wanted:
            path = self.package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"synthetic {relative}\n")
        weights = self.package / layout.C_EDITED_ROLES["chemeleon_weights"]
        config = self.package / layout.C_EDITED_ROLES["chemeleon_config"]
        config.write_text(json.dumps({"weights": {"sha256": layout.sha256_file(weights)}}))
        check = self.package / layout.SMOKE_CHECK_REL
        check.parent.mkdir(parents=True, exist_ok=True)
        check.write_text("synthetic check\n")
        self.pins = layout.build_pins(self.package, self.reference)
        self.evidence = layout.smoke_evidence(self.pins, "c" * 64)

    def test_arm_c_manifest_validates_and_pins_the_chemeleon_files(self):
        for replicate in (1, 2, 3):
            manifest = layout.arm_c_manifest(self.pins, self.evidence, replicate)
            launcher.validate_manifest(manifest)
            self.assertEqual(manifest["training_seed"], (20261101, 20261102, 20261103)[replicate - 1])
            self.assertTrue(manifest["output_dir"].endswith(f"t121_d022_c_s{replicate}_screen_run"))
            self.assertNotIn("descriptor_cache", manifest)
            spec = manifest["chemeleon"]
            pinned = manifest["pinned_files"]
            self.assertEqual(spec["config_sha256"], pinned["chemeleon_config"]["sha256"])
            self.assertTrue(pinned["chemeleon_weights"]["path"].startswith(layout.C_PACKAGE))
            for role in ("screen_stage", "weighted_training", "chemeleon_module"):
                self.assertTrue(pinned[role]["path"].startswith(layout.C_PACKAGE))
        with self.assertRaises(ValueError):
            layout.arm_c_manifest(self.pins, [], 1)
        with self.assertRaises(ValueError):
            layout.arm_c_manifest(self.pins, self.evidence, 4)

    def test_manifest_validation_refuses_unpinned_or_mismatched_chemeleon_files(self):
        manifest = layout.arm_c_manifest(self.pins, self.evidence, 1)
        broken = json.loads(json.dumps(manifest))
        del broken["pinned_files"]["chemeleon_weights"]
        with self.assertRaisesRegex(ValueError, "pinned roles"):
            launcher.validate_manifest(broken)
        broken = json.loads(json.dumps(manifest))
        broken["chemeleon"]["config_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "config_sha256"):
            launcher.validate_manifest(broken)
        broken = json.loads(json.dumps(manifest))
        broken["descriptor_cache"] = {"directory": "x", "manifest_sha256": "y"}
        with self.assertRaisesRegex(ValueError, "no descriptors"):
            launcher.validate_manifest(broken)

    def test_pins_refuse_weights_that_do_not_match_the_frozen_config(self):
        (self.package / layout.C_EDITED_ROLES["chemeleon_weights"]).write_text("other bytes\n")
        with self.assertRaisesRegex(ValueError, "weights hash"):
            layout.build_pins(self.package, self.reference)

    def test_smoke_evidence_binds_the_check_config_weights_and_modules(self):
        bound = self.evidence[0]["bound_fields"]
        files = self.pins["pinned_files"]
        self.assertEqual(bound["check_py_sha256"], layout.sha256_file(self.package / layout.SMOKE_CHECK_REL))
        self.assertEqual(bound["chemeleon_config_sha256"], files["chemeleon_config"]["sha256"])
        self.assertEqual(bound["chemeleon_weights_sha256"], files["chemeleon_weights"]["sha256"])
        self.assertEqual(bound["chemeleon_module_sha256"], files["chemeleon_module"]["sha256"])
        self.assertEqual(bound["screen_stage_sha256"], files["screen_stage"]["sha256"])
        with self.assertRaises(ValueError):
            layout.smoke_evidence(self.pins, "short")

    def test_unedited_roles_keep_reference_paths_and_external_inputs_are_unchanged(self):
        reference = self.reference["pinned_files"]
        for role in ("launcher_base", "cyp002_validation", "cyp007_infrastructure",
                     "executor_cell_fit", "stage_controller", "proposal"):
            relative = reference[role]["path"][len(layout.R_PACKAGE) + 1:]
            self.assertEqual(self.pins["pinned_files"][role]["path"], f"{layout.C_PACKAGE}/{relative}")
        for role in ("screen", "screen_identity", "screen_similarity", "screen_contract"):
            self.assertEqual(self.pins["pinned_files"][role], reference[role])

    def test_only_c_runs_and_unique_roots_that_cannot_touch_existing_runs(self):
        with self.assertRaises(ValueError):
            layout.run_name("r", 1)
        with self.assertRaises(ValueError):
            layout.run_name("d", 1)
        roots = [layout.screen_run_dir(a, n) for a, n in layout.NEW_RUNS] + \
                [layout.cells_run_dir(a, n) for a, n in layout.NEW_RUNS]
        self.assertEqual(len(set(roots)), 6)
        self.assertTrue(all("d022_c_s" in r for r in roots))

    def test_cell_environment_covers_every_required_slurm_variable_and_the_optional_trio(self):
        text = (REPO / "cyp/scripts/t121_cells.slurm").read_text()
        required = set()
        for line in text.splitlines():
            if line.startswith(': "${'):
                required |= set(re.findall(r'\$\{([A-Z0-9_]+):\?', line))
        for arm, replicate in layout.NEW_RUNS:
            env = layout.cell_environment(replicate, self.pins)
            self.assertEqual(required - set(env) - {"SCREEN_SELECTION_SHA256"}, set())
            for name in ("CHEMELEON_CONFIG", "CHEMELEON_CONFIG_SHA256", "CHEMELEON_WEIGHTS"):
                self.assertIn(name, env)
                self.assertIn(name, text)
            self.assertNotIn("DESCRIPTOR_CACHE_DIR", env)
            self.assertEqual(env["PACKAGE_ROOT"], layout.C_PACKAGE)
            self.assertTrue(env["CELL_RUN_ROOT"].endswith(f"t121_d022_c_s{replicate}_cells_run"))


if __name__ == "__main__":
    unittest.main()
