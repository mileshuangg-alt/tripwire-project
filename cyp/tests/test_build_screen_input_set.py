"""Tests for build_screen_input_set.py (Tripwire-CYP-010, D025 Amendment 3).

Each test is tied to a concrete failure risk: a protected tier, the consumed PXR holdout or a
bridging molecule leaking into the input set, a molecule that a screen model used for training
being selected, a molecule without both measurements being selected, structurally different
SMILES being silently accepted, string-only SMILES differences being wrongly dropped, a
tampered selection passing the final guard, an output directory being reused, a changed input
file being read after a hash mismatch, and the mapping file failing to carry both identifiers.
Writing these tests is not a passing run.
"""
import csv
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cyp/scripts"))
import build_screen_input_set as b  # noqa: E402

SCREEN_HEADER = "Molecule_Name,SMILES,OCNT_Batch,enzyme,plate_id,concentration_M,log2fc_estimate\n"
ADJ_HEADER = "molecule_name\tsmiles\tclass\ttiers\n"
CLUSTER_HEADER = "molecule_name\tidentity_class\tmax_tanimoto\tnearest_protected\tnearest_tier\tcluster_verdict\tbridged_tiers\n"
ELIG_HEADER = "OCNT_ID,role,eligible,reason,group_id,has_aux_labels\n"
SINGLE_HEADER = "Molecule Name,OCNT Batch,SMILES,plate_id,compound_class,concentration_M,log2_fc_estimate,OCNT_ID\n"
TRAIN_HEADER = "Molecule Name,SMILES,OCNT_ID\n"

# name: (cyp smiles, class, tiers, verdict, cyp3a4 value, pxr role, eligible, reason, pxr smiles, 8.25 concentration)
MOLECULES = {
    "OCNT-A": ("CCO", "full_key_match", "train", "", "-0.5", "primary_validation", "False", "primary_validation_aux_excluded", "CCO", "8.251e-06"),
    "OCNT-B": ("CCN", "full_key_match", "internal_test", "", "-0.5", "primary_validation", "False", "primary_validation_aux_excluded", "CCN", "8.251e-06"),
    "OCNT-C": ("CCC", "full_key_match", "train", "", "-0.5", "primary_holdout", "False", "primary_holdout_aux_excluded", "CCC", "8.251e-06"),
    "OCNT-D": ("CCCC", "no_identity_link", "", "bridges_protected", "-0.5", "aux_only", "False", "aux_only_group_contains_validation_or_holdout", "CCCC", "8.251e-06"),
    "OCNT-E": ("CCO", "full_key_match", "train", "", "-0.5", "primary_validation", "False", "primary_validation_aux_excluded", "CCN", "8.251e-06"),
    "OCNT-F": ("OCC", "full_key_match", "validation", "", "-0.5", "primary_validation", "False", "primary_validation_aux_excluded", "CCO", "8.251e-06"),
    "OCNT-G": ("CCCCC", "no_identity_link", "", "cluster_clear", "-0.5", "aux_only", "False", "aux_only_eligible", "CCCCC", "8.251e-06"),
    "OCNT-H": ("CCCCCC", "full_key_match", "train", "", "-0.5", "aux_only", "True", "aux_only_eligible", "CCCCCC", "8.251e-06"),
    "OCNT-I": ("CCOC", "full_key_match", "train", "", "", "primary_validation", "False", "primary_validation_aux_excluded", "CCOC", "8.251e-06"),
    "OCNT-J": ("CCNC", "full_key_match", "train", "", "-0.5", "primary_validation", "False", "primary_validation_aux_excluded", "CCNC", "3.3e-05"),
    "OCNT-L": ("COC", "full_key_match", "validation;challenge_blinded_test", "", "-0.5", "primary_validation", "False", "primary_validation_aux_excluded", "COC", "8.251e-06"),
    "OCNT-M": ("bad(", "full_key_match", "train", "", "-0.5", "primary_validation", "False", "primary_validation_aux_excluded", "CCO", "8.251e-06"),
    "OCNT-N": ("CN", "full_key_match", "train", "", "-0.5", "aux_only", "False", "aux_only_group_contains_test_structure", "CN", "8.251e-06"),
}
EXPECTED_BASE = ["OCNT-A", "OCNT-B", "OCNT-C", "OCNT-D", "OCNT-E", "OCNT-F", "OCNT-L", "OCNT-M", "OCNT-N"]
EXPECTED_SELECTED = ["OCNT-A", "OCNT-F", "OCNT-N"]


def texts():
    screen, adj, cluster, elig, single, train = SCREEN_HEADER, ADJ_HEADER, CLUSTER_HEADER, ELIG_HEADER, SINGLE_HEADER, TRAIN_HEADER
    for k, (name, v) in enumerate(MOLECULES.items()):
        smiles, klass, tiers, verdict, value, role, eligible, reason, pxr_smiles, conc = v
        screen += f"{name},{smiles},b,CYP3A4,p,4.9e-05,{value}\n"
        screen += f"{name},{smiles},b,CYP1A2,p,4.9e-05,-0.1\n"
        adj += f"{name}\t{smiles}\t{klass}\t{tiers}\n"
        if verdict:
            cluster += f"{name}\t{klass}\t0.5\tX\ttrain\t{verdict}\t\n"
        elig += f"{name},{role},{eligible},{reason},{k},True\n"
        single += f"x,b,{pxr_smiles},p,Library,{conc},0.5,{name}\n"
        if role.startswith("primary_"):
            train += f"OADMET-{name[-1]},{pxr_smiles},{name}\n"
    return screen, adj, cluster, elig, single, train


def tables():
    screen, adj, cluster, elig, single, train = texts()
    return b.build_cyp_table(screen, adj, cluster), b.build_pxr_table(elig, single, train)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cyp, self.pxr = tables()
        self.result = b.select_input_set(self.cyp, self.pxr)

    def test_base_requires_outside_screening_stage_unused_by_pxr_and_both_measured(self):
        self.assertEqual(self.result["base"], EXPECTED_BASE)
        for absent in ("OCNT-G", "OCNT-H", "OCNT-I", "OCNT-J"):
            self.assertNotIn(absent, self.result["base"])

    def test_protected_tiers_holdout_bridges_and_smiles_mismatch_are_removed_with_reasons(self):
        reasons = self.result["reasons"]
        self.assertEqual(reasons["OCNT-B"], ["cyp_protected_tier"])
        self.assertEqual(reasons["OCNT-L"], ["cyp_protected_tier"])  # token split on ";"
        self.assertEqual(reasons["OCNT-C"], ["pxr_holdout_role"])
        self.assertEqual(reasons["OCNT-D"], ["cyp_bridges_protected"])
        self.assertEqual(reasons["OCNT-E"], ["smiles_structure_mismatch_or_unparseable"])
        self.assertEqual(reasons["OCNT-M"], ["smiles_structure_mismatch_or_unparseable"])

    def test_string_only_smiles_difference_is_kept(self):
        self.assertIn("OCNT-F", self.result["selected"])
        self.assertFalse(self.result["smiles_info"]["OCNT-F"]["string_equal"])
        self.assertTrue(self.result["smiles_info"]["OCNT-F"]["canonical_equal"])

    def test_selected_set_and_step_counts(self):
        self.assertEqual(self.result["selected"], EXPECTED_SELECTED)
        self.assertEqual(self.result["counts"], {
            "base": 9, "after_protected_and_holdout": 6, "after_bridges": 5, "after_smiles_rule": 3})
        self.assertFalse(self.result["counts_match_records"])  # synthetic counts are not the recorded ones

    def test_summary_reports_aux_only_touching_official_test_structures(self):
        summary = b.summarize(self.result, self.cyp, self.pxr)
        self.assertEqual(summary["selected_pxr_aux_only_touching_official_test_structures"], 1)
        self.assertEqual(summary["selected_by_pxr_role"], {"aux_only": 1, "primary_validation": 2})
        self.assertEqual(summary["smiles_canonical_equal_but_string_differs_among_selected"], 1)

    def test_final_guard_accepts_the_result_and_rejects_tampered_selections(self):
        b.assert_protected_excluded(self.result, self.cyp, self.pxr)
        for intruder in ("OCNT-B", "OCNT-C", "OCNT-D", "OCNT-E", "OCNT-G", "OCNT-H", "OCNT-I", "OCNT-J"):
            tampered = dict(self.result, selected=self.result["selected"] + [intruder])
            with self.assertRaises((ValueError, KeyError)):
                b.assert_protected_excluded(tampered, self.cyp, self.pxr)
        with self.assertRaises(ValueError):
            b.assert_protected_excluded(dict(self.result, selected=["OCNT-A", "OCNT-A"]), self.cyp, self.pxr)


class TableTests(unittest.TestCase):
    def test_pxr_row_ids_and_sources(self):
        _, pxr = tables()
        self.assertEqual(pxr["OCNT-A"]["row_id"], "OADMET-A")
        self.assertEqual(pxr["OCNT-A"]["smiles_source"], "TRAIN.csv")
        self.assertEqual(pxr["OCNT-N"]["row_id"], "OCNT-N")
        self.assertEqual(pxr["OCNT-N"]["smiles_source"], "single_concentration")
        self.assertFalse(pxr["OCNT-J"]["measured_8p25"])
        self.assertTrue(pxr["OCNT-A"]["measured_8p25"])

    def test_inconsistent_smiles_unknown_class_and_missing_identity_stop_the_build(self):
        screen, adj, cluster, elig, single, train = texts()
        with self.assertRaises(ValueError):
            b.build_cyp_table(screen + "OCNT-A,CCCN,b,CYP2D6,p,4.9e-05,-0.1\n", adj, cluster)
        with self.assertRaises(ValueError):
            b.build_cyp_table(screen, adj.replace("full_key_match", "mystery", 1), cluster)
        with self.assertRaises(ValueError):
            b.build_cyp_table(screen + "OCNT-Z,CC,b,CYP2D6,p,4.9e-05,-0.1\n", adj, cluster)
        with self.assertRaises(ValueError):
            b.build_pxr_table(elig + "OCNT-Z,primary_validation,False,x,1,True\n", single, train)

    def test_cyp3a4_measured_flag_uses_presence_not_value(self):
        cyp, _ = tables()
        self.assertTrue(cyp["OCNT-A"]["cyp3a4_measured"])
        self.assertFalse(cyp["OCNT-I"]["cyp3a4_measured"])


class IoTests(unittest.TestCase):
    def test_tar_member_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / "t.tar.gz"
            payload = b"a,b\n1,2\n"
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo("inputs/screen_full.csv")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            good = b.sha256_bytes(payload)
            self.assertEqual(b.read_tar_member(tar_path, "inputs/screen_full.csv", good), "a,b\n1,2\n")
            with self.assertRaises(ValueError):
                b.read_tar_member(tar_path, "inputs/screen_full.csv", "0" * 64)
            with self.assertRaises(KeyError):
                b.read_tar_member(tar_path, "inputs/missing.csv", good)

    def test_verified_text_rejects_a_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            path.write_text("x\n")
            self.assertEqual(b.read_verified_text(path, b.sha256_file(path)), "x\n")
            with self.assertRaises(ValueError):
                b.read_verified_text(path, "0" * 64)

    def test_outputs_have_the_input_mapping_and_exclusion_files_and_are_never_reused(self):
        cyp, pxr = tables()
        result = b.select_input_set(cyp, pxr)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            paths = b.write_outputs(out, result, cyp, pxr)
            with open(paths["input"], newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["OCNT_ID", "SMILES"])
            self.assertEqual([r[0] for r in rows[1:]], EXPECTED_SELECTED)
            self.assertEqual(dict(rows[1:])["OCNT-F"], "OCC")  # the CYP screening-file string is written
            with open(paths["mapping"], newline="") as handle:
                mapping = {r["OCNT_ID"]: r for r in csv.DictReader(handle)}
            self.assertEqual(mapping["OCNT-A"]["pxr_row_id"], "OADMET-A")
            self.assertEqual(mapping["OCNT-A"]["cyp_molecule_name"], "OCNT-A")
            self.assertEqual(mapping["OCNT-F"]["pxr_source_smiles"], "CCO")
            self.assertEqual(mapping["OCNT-F"]["smiles_string_equal"], "False")
            with open(paths["exclusions"], newline="") as handle:
                excluded = {r["OCNT_ID"]: r["reasons"] for r in csv.DictReader(handle)}
            self.assertEqual(set(excluded), {"OCNT-B", "OCNT-C", "OCNT-D", "OCNT-E", "OCNT-L", "OCNT-M"})
            with self.assertRaises(FileExistsError):
                b.write_outputs(out, result, cyp, pxr)

    def test_tier_tokens_and_canonical_smiles_helpers(self):
        self.assertEqual(b.tier_tokens("train,validation"), ["train", "validation"])
        self.assertEqual(b.tier_tokens("a;b|c d"), ["a", "b", "c", "d"])
        self.assertEqual(b.tier_tokens(""), [])
        self.assertEqual(b.canonical_isomeric("OCC"), b.canonical_isomeric("CCO"))
        self.assertIsNone(b.canonical_isomeric("bad("))
        self.assertIsNone(b.canonical_isomeric(""))


class SourceInvariantTests(unittest.TestCase):
    def test_script_never_reads_label_columns_into_computation_and_loads_no_model(self):
        source = (Path(__file__).resolve().parents[2] / "cyp/scripts/build_screen_input_set.py").read_text()
        for forbidden in ("torch", "chemprop", "load_state_dict", "log2_fc_estimate\"]", "float(row[\"log2fc"):
            self.assertNotIn(forbidden, source.split('"""', 2)[2])


if __name__ == "__main__":
    unittest.main()
