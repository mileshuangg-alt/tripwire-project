"""D024 layout and package: pins, run environments and an offline package build."""
from pathlib import Path
import re
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cyp/scripts"))
import t121_d024_build_package as builder
import t121_d024_layout as layout

REFERENCE = REPO / "screen_manifest_v2.json"
REFERENCE_TAR = REPO / "t121_production_chain_20261117.tar.gz"
SLURM = REPO / "cyp/scripts/t121_cells.slurm"


@unittest.skipUnless(REFERENCE.is_file(), "reference screening manifest not present (deployed package)")
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
        } | set(layout.P_EDITED_ROLES.values())
        for relative in wanted:
            path = self.package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"synthetic {relative}\n")
        self.pins = layout.build_pins(self.package, self.reference)

    def test_environments_set_every_variable_the_cell_array_requires(self):
        required = set(re.findall(r"\$\{([A-Z0-9_]+):\?\}", SLURM.read_text()))
        self.assertIn("SCREEN_SELECTION_SHA256", required)
        environments = {arm: layout.cell_environment(arm, self.pins, "a" * 64) for arm in layout.ARMS}
        for arm, environment in environments.items():
            self.assertFalse(required - set(environment), arm)
            role = layout.CONFIG_ROLES[arm]
            self.assertEqual(environment["PENALTY_CONFIG"], self.pins["pinned_files"][role]["path"])
            self.assertEqual(environment["PENALTY_CONFIG_SHA256"],
                             self.pins["pinned_files"][role]["sha256"])
            self.assertEqual(environment["SCREEN_SELECTION"], layout.R1_SELECTION)
            self.assertNotIn("DESCRIPTOR_CACHE_DIR", environment)
            self.assertNotIn("CHEMELEON_CONFIG", environment)
            self.assertIn("export PENALTY_CONFIG=", layout.shell_exports(environment))
        q3, rank = environments["q3"], environments["rank"]
        different = {k for k in q3 if q3[k] != rank[k]}
        self.assertEqual(different, {"CELL_RUN_ROOT", "PENALTY_CONFIG", "PENALTY_CONFIG_SHA256"})

    def test_bad_hash_or_arm_is_refused(self):
        for bad in ("", "xyz", "A" * 64):
            with self.assertRaises(ValueError):
                layout.cell_environment("q3", self.pins, bad)
        with self.assertRaises(ValueError):
            layout.cells_run_dir("unchanged")

    def test_every_pinned_package_file_is_shipped_by_the_builder(self):
        shipped = set(builder.OVERLAY) | {"executor_source/cyp007_production_helpers.py"}
        self.assertFalse(set(layout.P_EDITED_ROLES.values()) - shipped)
        for relative in builder.OVERLAY:
            self.assertTrue((REPO / relative).is_file(), relative)


@unittest.skipUnless(REFERENCE_TAR.is_file(), "reference package archive not present")
class BuildTests(unittest.TestCase):
    def test_offline_build_pins_and_sums_match_the_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            package, archive, count = builder.build(REFERENCE_TAR, REPO, Path(tmp) / "out")
            self.assertTrue(archive.is_file())
            lines = (package / "SHA256SUMS.txt").read_text().splitlines()
            self.assertEqual(len(lines), count)
            for line in lines:
                digest, relative = line.split("  ", 1)
                self.assertEqual(layout.sha256_file(package / relative), digest, relative)
            for relative in builder.OVERLAY:
                self.assertEqual((package / relative).read_bytes(), (REPO / relative).read_bytes())


if __name__ == "__main__":
    unittest.main()
