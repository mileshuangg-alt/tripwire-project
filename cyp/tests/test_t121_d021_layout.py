from pathlib import Path
import json
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cyp/scripts"))
import t121_d021_layout as layout
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
        wanted |= set(layout.D_EDITED_ROLES.values())
        wanted |= {f"{layout.DESCRIPTOR_CACHE_REL}/descriptor_manifest.json",
                   f"{layout.DESCRIPTOR_CACHE_REL}/descriptors.npz"}
        for relative in wanted:
            path = self.package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"synthetic {relative}\n")
        check = self.package / layout.SMOKE_CHECK_REL
        check.parent.mkdir(parents=True, exist_ok=True)
        check.write_text("synthetic check\n")
        self.pins = layout.build_pins(self.package, self.reference)
        self.evidence = [{"path": "/x/result.json", "sha256": "a" * 64, "bound_fields": {"check_py_sha256": "b" * 64}}]

    def test_reference_manifest_is_hash_checked(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            handle.write("{}")
            handle.flush()
            with self.assertRaises(ValueError):
                layout.reference_manifest_checked(handle.name)

    def test_arm_r_manifest_differs_only_in_output_and_seed(self):
        for replicate in (2, 3):
            manifest = layout.arm_r_manifest(self.reference, replicate)
            changed = {k for k in manifest if manifest[k] != self.reference[k]}
            self.assertEqual(changed, {"output_dir", "training_seed"})
            self.assertEqual(manifest["training_seed"], layout.SEEDS[replicate - 1])
            self.assertTrue(manifest["output_dir"].endswith(f"t121_d021_r_s{replicate}_screen_run"))
            launcher.validate_manifest(manifest)
        with self.assertRaises(ValueError):
            layout.arm_r_manifest(self.reference, 1)

    def test_arm_d_manifest_validates_and_pins_the_descriptor_files(self):
        for replicate in (1, 2, 3):
            manifest = layout.arm_d_manifest(self.pins, self.evidence, replicate)
            launcher.validate_manifest(manifest)
            self.assertEqual(manifest["training_seed"], layout.SEEDS[replicate - 1])
            self.assertEqual(manifest["descriptor_cache"]["manifest_sha256"],
                             manifest["pinned_files"]["descriptor_manifest"]["sha256"])
            for role in ("screen_stage", "weighted_training", "descriptor_cache_module"):
                self.assertTrue(manifest["pinned_files"][role]["path"].startswith(layout.D_PACKAGE))
        with self.assertRaises(ValueError):
            layout.arm_d_manifest(self.pins, [], 1)

    def test_smoke_evidence_binds_the_check_modules_and_cache(self):
        evidence = layout.smoke_evidence(self.pins, "c" * 64)
        self.assertEqual(len(evidence), 1)
        bound = evidence[0]["bound_fields"]
        self.assertEqual(bound["screen_stage_sha256"], self.pins["pinned_files"]["screen_stage"]["sha256"])
        self.assertEqual(bound["check_py_sha256"], layout.sha256_file(self.package / layout.SMOKE_CHECK_REL))
        launcher.validate_manifest(layout.arm_d_manifest(self.pins, evidence, 1))
        with self.assertRaises(ValueError):
            layout.smoke_evidence(self.pins, "short")

    def test_pins_keep_reference_files_for_unedited_roles(self):
        reference = self.reference["pinned_files"]
        for role in ("launcher_base", "cyp002_validation", "cyp007_infrastructure",
                     "executor_cell_fit", "stage_controller", "proposal"):
            relative = reference[role]["path"][len(layout.R_PACKAGE) + 1:]
            self.assertEqual(self.pins["pinned_files"][role]["path"], f"{layout.D_PACKAGE}/{relative}")
        for role in ("screen", "screen_identity", "screen_similarity", "screen_contract"):
            self.assertEqual(self.pins["pinned_files"][role], reference[role])

    def test_run_roots_are_unique_per_arm_and_seed(self):
        roots = [layout.screen_run_dir(a, n) for a, n in layout.NEW_RUNS] + \
                [layout.cells_run_dir(a, n) for a, n in layout.NEW_RUNS]
        self.assertEqual(len(roots), len(set(roots)))
        self.assertNotIn("production_chain_20261117", " ".join(roots))

    def test_cell_environment_covers_every_required_slurm_variable(self):
        text = (REPO / "cyp/scripts/t121_cells.slurm").read_text()
        import re
        required = set()
        for line in text.splitlines():
            if line.startswith(': "${'):
                required |= set(re.findall(r'\$\{([A-Z0-9_]+):\?', line))
        self.assertIn("PACKAGE_ROOT", required)
        self.assertIn("SCREEN_CONTRACT", required)
        for arm, replicate in layout.NEW_RUNS:
            env = layout.cell_environment(arm, replicate, self.pins if arm == "d" else None, self.reference)
            missing = required - set(env) - {"SCREEN_SELECTION_SHA256"}
            self.assertEqual(missing, set(), (arm, replicate))
            self.assertEqual(("DESCRIPTOR_CACHE_DIR" in env), arm == "d")
            self.assertTrue(env["CELL_RUN_ROOT"].endswith(f"{layout.run_name(arm, replicate)}_cells_run"))
        d_env = layout.cell_environment("d", 1, self.pins, self.reference)
        self.assertEqual(d_env["PACKAGE_ROOT"], layout.D_PACKAGE)
        r_env = layout.cell_environment("r", 2, None, self.reference)
        self.assertEqual(r_env["PACKAGE_ROOT"], layout.R_PACKAGE)


if __name__ == "__main__":
    unittest.main()
