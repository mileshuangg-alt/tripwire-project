"""Build the first real input set for screen_predict.py (Tripwire-CYP-010, D025 Amendment 3).

Authorizes no run by itself; the project owner runs it. Read-only on project data, no model is
loaded and no prediction is made. It writes only into a NEW output directory.

Selected set: molecules with a measured CYP3A4 screening value AND a measured PXR 8.25 uM
single-concentration value that both screen models excluded from training and checkpoint
selection:
  * CYP: not in the 6,947-molecule screening stage, i.e. identity-linked (full_key_match or
    largest_fragment_match) or bridges_protected (the screening loader's own exclusion rules);
  * PXR: auxiliary labels not in the stage-1 training frame (pxr002_eligibility eligible False)
    and has auxiliary labels.
Removed by construction, in this order (each molecule is recorded with all of its reasons):
  1. CYP tiers internal_test or challenge_blinded_test (protected);
  2. PXR role primary_holdout (the consumed holdout);
  3. CYP adjudication bridges_protected (near protected clusters);
  4. SMILES rule: the CYP screening-file SMILES must give the same canonical isomeric structure
     as the PXR source SMILES (released TRAIN.csv for PXR primary roles, the single-concentration
     file for aux-only molecules); a molecule that fails to parse, or whose structures differ, is
     removed.
The input SMILES written is the CYP screening-file string. input_id is OCNT_ID, which equals the
CYP Molecule_Name. An explicit mapping file lists OCNT_ID, CYP Molecule_Name, PXR row_id and both
SMILES strings; a later join with the potency report still needs that mapping to be reviewed.

Only measurement presence, identities, tiers, roles and SMILES strings are used. No label value
is read into any computation.

The exact-file protected hash guard in screen_predict.py does not cover this derived file; the
protection here is by construction, enforced by final assertions and reported in the report.
The report counts selected PXR aux-only molecules whose group touches official-test structures
(eligibility reason aux_only_group_contains_test_structure); excluding them is not part of the
approved rule.

Usage (project owner, Mac, repository root): see the commands given with the plan.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import tarfile
from pathlib import Path

import numpy as np

SCREEN_MEMBER = "inputs/screen_full.csv"
ADJUDICATION_MEMBER = "inputs/adjudication.tsv"
CLUSTER_MEMBER = "inputs/cluster_adjudication.tsv"
LINKED_CLASSES = ("full_key_match", "largest_fragment_match")
KNOWN_CLASSES = LINKED_CLASSES + ("no_identity_link", "first_block_near_match_only")
PROTECTED_TIERS = frozenset({"internal_test", "challenge_blinded_test"})
HOLDOUT_ROLE = "primary_holdout"
PXR_8P25_CONCENTRATION = 8.251e-06
EXPECTED_COUNTS_FROM_RECORDS = {"base": 449, "after_protected_and_holdout": 213, "after_bridges": 209}


# ----------------------------------------------------------------------- helpers
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tier_tokens(text) -> list:
    return [token for token in re.split(r"[,;|\s]+", text or "") if token]


def canonical_isomeric(smiles):
    """RDKit canonical isomeric SMILES, or None if the string does not parse."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol, isomericSmiles=True)


def read_tar_member(tar_path, member: str, expected_sha256: str) -> str:
    """Text of one tar member after its SHA-256 has been checked."""
    with tarfile.open(tar_path, "r:gz") as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"Tar member missing: {member}")
        data = handle.read()
    actual = sha256_bytes(data)
    if actual != expected_sha256:
        raise ValueError(f"Tar member hash mismatch: {member} (actual {actual}, expected {expected_sha256})")
    return data.decode("utf-8")


def read_verified_text(path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Input hash mismatch: {path} (actual {actual}, expected {expected_sha256})")
    return Path(path).read_text(encoding="utf-8")


def dict_rows(text: str, delimiter: str = ","):
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


# -------------------------------------------------------------------- table building
def build_cyp_table(screen_text: str, adjudication_text: str, cluster_text: str) -> dict:
    """Molecule_Name -> identity class, tiers, cluster verdict, SMILES, CYP3A4 measured flag."""
    identity = {}
    for row in dict_rows(adjudication_text, "\t"):
        name = row["molecule_name"]
        if name in identity:
            raise ValueError(f"Duplicate adjudication identity: {name}")
        identity[name] = row
    cluster = {}
    for row in dict_rows(cluster_text, "\t"):
        name = row["molecule_name"]
        if name in cluster:
            raise ValueError(f"Duplicate cluster adjudication: {name}")
        cluster[name] = row
    table = {}
    for row in dict_rows(screen_text):
        name = row["Molecule_Name"]
        if name not in identity:
            raise ValueError(f"Missing identity adjudication: {name}")
        klass = identity[name]["class"]
        if klass not in KNOWN_CLASSES:
            raise ValueError(f"Unresolved identity class: {name}: {klass}")
        entry = table.setdefault(name, {
            "smiles": row["SMILES"], "klass": klass, "tiers": identity[name].get("tiers", ""),
            "verdict": cluster.get(name, {}).get("cluster_verdict", ""), "cyp3a4_measured": False,
        })
        if entry["smiles"] != row["SMILES"]:
            raise ValueError(f"Inconsistent CYP screening SMILES: {name}")
        if row["enzyme"] == "CYP3A4" and row["log2fc_estimate"].strip() != "":
            entry["cyp3a4_measured"] = True
    return table


def cyp_outside_screening_stage(entry: dict) -> bool:
    """The screening loader's own exclusions: identity-linked, or bridging a protected cluster."""
    return entry["klass"] in LINKED_CLASSES or entry["verdict"] == "bridges_protected"


def build_pxr_table(eligibility_text: str, single_text: str, train_text: str) -> dict:
    """OCNT_ID -> role, eligibility, reason, PXR source SMILES, row_id, 8.25 uM measured flag."""
    measured_8p25 = set()
    single_smiles = {}
    for row in dict_rows(single_text):
        if row["compound_class"] != "Library":
            continue
        ocnt = row["OCNT_ID"]
        if ocnt in single_smiles and single_smiles[ocnt] != row["SMILES"]:
            raise ValueError(f"Inconsistent single-concentration SMILES: {ocnt}")
        single_smiles[ocnt] = row["SMILES"]
        if abs(float(row["concentration_M"]) - PXR_8P25_CONCENTRATION) <= 1e-3 * PXR_8P25_CONCENTRATION:
            measured_8p25.add(ocnt)
    train = {}
    for row in dict_rows(train_text):
        if row["OCNT_ID"] in train:
            raise ValueError(f"Duplicate OCNT_ID in TRAIN.csv: {row['OCNT_ID']}")
        train[row["OCNT_ID"]] = (row["Molecule Name"], row["SMILES"])
    table = {}
    for row in dict_rows(eligibility_text):
        ocnt = row["OCNT_ID"]
        if ocnt in table:
            raise ValueError(f"Duplicate OCNT_ID in eligibility: {ocnt}")
        role = row["role"]
        if role.startswith("primary_"):
            if ocnt not in train:
                raise ValueError(f"Primary-role molecule missing from TRAIN.csv: {ocnt}")
            row_id, smiles, source = train[ocnt][0], train[ocnt][1], "TRAIN.csv"
        elif role == "aux_only":
            if ocnt not in single_smiles:
                raise ValueError(f"Aux-only molecule missing from the single-concentration file: {ocnt}")
            row_id, smiles, source = ocnt, single_smiles[ocnt], "single_concentration"
        else:
            raise ValueError(f"Unknown PXR role: {role}")
        table[ocnt] = {
            "role": role, "eligible": row["eligible"] == "True", "reason": row["reason"],
            "has_aux": row["has_aux_labels"] == "True", "row_id": row_id,
            "smiles": smiles, "smiles_source": source, "measured_8p25": ocnt in measured_8p25,
        }
    return table


# ------------------------------------------------------------------------- selection
def select_input_set(cyp: dict, pxr: dict) -> dict:
    """Apply the approved rule. Returns steps, selected rows and an exclusion record per molecule."""
    base = []
    for name in sorted(cyp):
        c = cyp[name]
        p = pxr.get(name)
        if p is None or not p["has_aux"] or p["eligible"]:
            continue
        if not cyp_outside_screening_stage(c) or not c["cyp3a4_measured"] or not p["measured_8p25"]:
            continue
        base.append(name)

    reasons = {name: [] for name in base}
    for name in base:
        c, p = cyp[name], pxr[name]
        if PROTECTED_TIERS & set(tier_tokens(c["tiers"])):
            reasons[name].append("cyp_protected_tier")
        if p["role"] == HOLDOUT_ROLE:
            reasons[name].append("pxr_holdout_role")
    after_first_two = [n for n in base if not reasons[n]]
    for name in base:
        if cyp[name]["verdict"] == "bridges_protected":
            reasons[name].append("cyp_bridges_protected")
    after_bridges = [n for n in base if not reasons[n]]
    smiles_info = {}
    for name in base:
        a, b = canonical_isomeric(cyp[name]["smiles"]), canonical_isomeric(pxr[name]["smiles"])
        smiles_info[name] = {
            "string_equal": cyp[name]["smiles"] == pxr[name]["smiles"],
            "cyp_parses": a is not None, "pxr_parses": b is not None,
            "canonical_equal": a is not None and a == b,
        }
        if not smiles_info[name]["canonical_equal"]:
            reasons[name].append("smiles_structure_mismatch_or_unparseable")
    selected = [n for n in base if not reasons[n]]
    counts = {
        "base": len(base),
        "after_protected_and_holdout": len(after_first_two),
        "after_bridges": len(after_bridges),
        "after_smiles_rule": len(selected),
    }
    return {
        "base": base, "selected": selected, "reasons": reasons, "smiles_info": smiles_info,
        "counts": counts,
        "counts_match_records": all(counts[k] == v for k, v in EXPECTED_COUNTS_FROM_RECORDS.items()),
    }


def assert_protected_excluded(result: dict, cyp: dict, pxr: dict) -> None:
    """Final guard: nothing protected, closed, bridging or structurally ambiguous is selected."""
    seen = set()
    for name in result["selected"]:
        if name in seen:
            raise ValueError(f"Duplicate selected molecule: {name}")
        seen.add(name)
        c, p = cyp[name], pxr[name]
        if PROTECTED_TIERS & set(tier_tokens(c["tiers"])):
            raise ValueError(f"Protected CYP tier selected: {name}")
        if p["role"] == HOLDOUT_ROLE:
            raise ValueError(f"PXR holdout-role molecule selected: {name}")
        if c["verdict"] == "bridges_protected":
            raise ValueError(f"Bridging molecule selected: {name}")
        if not (c["cyp3a4_measured"] and p["measured_8p25"]):
            raise ValueError(f"Molecule without both measurements selected: {name}")
        if not cyp_outside_screening_stage(c) or p["eligible"]:
            raise ValueError(f"Molecule used by a screen model selected: {name}")
        if not result["smiles_info"][name]["canonical_equal"]:
            raise ValueError(f"SMILES-ambiguous molecule selected: {name}")


def summarize(result: dict, cyp: dict, pxr: dict) -> dict:
    selected = result["selected"]
    count = lambda values: {key: values.count(key) for key in sorted(set(values))}  # noqa: E731
    return {
        "selected_by_pxr_role": count([pxr[n]["role"] for n in selected]),
        "selected_by_pxr_reason": count([pxr[n]["reason"] for n in selected]),
        "selected_by_cyp_tier": count([cyp[n]["tiers"] or "(not identity-linked)" for n in selected]),
        "selected_pxr_aux_only_touching_official_test_structures": sum(
            1 for n in selected if pxr[n]["reason"] == "aux_only_group_contains_test_structure"),
        "smiles_string_equal_among_selected": sum(1 for n in selected if result["smiles_info"][n]["string_equal"]),
        "smiles_canonical_equal_but_string_differs_among_selected": sum(
            1 for n in selected if not result["smiles_info"][n]["string_equal"]),
        "excluded_by_reason": count([r for n in result["base"] for r in result["reasons"][n]]),
    }


# -------------------------------------------------------------------------- writing
def write_outputs(out_dir, result: dict, cyp: dict, pxr: dict) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    paths = {
        "input": out_dir / "screen_input_set.csv",
        "mapping": out_dir / "screen_input_mapping.csv",
        "exclusions": out_dir / "screen_input_exclusions.csv",
    }
    with open(paths["input"], "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["OCNT_ID", "SMILES"])
        for name in result["selected"]:
            writer.writerow([name, cyp[name]["smiles"]])
    with open(paths["mapping"], "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "OCNT_ID", "cyp_molecule_name", "pxr_row_id", "pxr_role", "pxr_eligibility_reason",
            "cyp_tiers", "cyp_smiles", "pxr_source_smiles", "pxr_smiles_source", "smiles_string_equal",
        ])
        for name in result["selected"]:
            c, p = cyp[name], pxr[name]
            writer.writerow([
                name, name, p["row_id"], p["role"], p["reason"], c["tiers"], c["smiles"],
                p["smiles"], p["smiles_source"], result["smiles_info"][name]["string_equal"],
            ])
    with open(paths["exclusions"], "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["OCNT_ID", "reasons", "pxr_role", "cyp_tiers"])
        for name in result["base"]:
            if result["reasons"][name]:
                writer.writerow([name, ";".join(result["reasons"][name]), pxr[name]["role"], cyp[name]["tiers"]])
    return paths


def build(args) -> dict:
    screen = read_tar_member(args.screen_tarball, SCREEN_MEMBER, args.screen_sha256)
    adjudication = read_tar_member(args.screen_tarball, ADJUDICATION_MEMBER, args.adjudication_sha256)
    cluster = read_tar_member(args.screen_tarball, CLUSTER_MEMBER, args.cluster_sha256)
    eligibility = read_verified_text(args.eligibility, args.eligibility_sha256)
    single = read_verified_text(args.pxr_single, args.pxr_single_sha256)
    train = read_verified_text(args.pxr_train, args.pxr_train_sha256)
    cyp = build_cyp_table(screen, adjudication, cluster)
    pxr = build_pxr_table(eligibility, single, train)
    result = select_input_set(cyp, pxr)
    assert_protected_excluded(result, cyp, pxr)
    if not result["selected"]:
        raise ValueError("The selection is empty.")
    paths = write_outputs(args.out_dir, result, cyp, pxr)
    import rdkit

    report = {
        "tool": "cyp/scripts/build_screen_input_set.py",
        "decision": "D025 Amendment 3",
        "script_sha256": sha256_file(__file__),
        "inputs": {
            "screen_tarball": {"path": str(Path(args.screen_tarball).resolve()), "sha256": sha256_file(args.screen_tarball)},
            "members": {SCREEN_MEMBER: args.screen_sha256, ADJUDICATION_MEMBER: args.adjudication_sha256, CLUSTER_MEMBER: args.cluster_sha256},
            "eligibility": {"path": str(Path(args.eligibility).resolve()), "sha256": args.eligibility_sha256},
            "pxr_single_concentration": {"path": str(Path(args.pxr_single).resolve()), "sha256": args.pxr_single_sha256},
            "pxr_train": {"path": str(Path(args.pxr_train).resolve()), "sha256": args.pxr_train_sha256},
        },
        "rule": __doc__.split("Selected set:")[1].split("Only measurement presence")[0].strip(),
        "counts": result["counts"],
        "expected_counts_from_records": EXPECTED_COUNTS_FROM_RECORDS,
        "counts_match_records": result["counts_match_records"],
        "summary": summarize(result, cyp, pxr),
        "outputs": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()},
        "input_columns": {"id_column": "OCNT_ID", "smiles_column": "SMILES"},
        "labels_used": False,
        "label_values_read_into_any_computation": False,
        "protected_sets_excluded_by_construction": True,
        "exact_file_guard_note": "The screen_predict.py hash guard does not cover this derived file; protection is by construction and the final assertions.",
        "join_note": "A later join needs the mapping file reviewed; CYP Molecule_Name and PXR row_id are not assumed interchangeable.",
        "environment": {"python": platform.python_version(), "rdkit": rdkit.__version__, "numpy": np.__version__},
    }
    report_path = Path(args.out_dir) / "screen_input_set_report.json"
    with open(report_path, "x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--screen-tarball", "--eligibility", "--pxr-single", "--pxr-train", "--out-dir"):
        parser.add_argument(flag, required=True, type=Path)
    for flag in ("--screen-sha256", "--adjudication-sha256", "--cluster-sha256",
                 "--eligibility-sha256", "--pxr-single-sha256", "--pxr-train-sha256"):
        parser.add_argument(flag, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.out_dir}")
    report = build(args)
    print(json.dumps({
        "counts": report["counts"], "counts_match_records": report["counts_match_records"],
        "summary": report["summary"],
        "outputs": {key: value["path"] for key, value in report["outputs"].items()},
        "input_sha256": report["outputs"]["input"]["sha256"],
    }, indent=2))
    return 0 if report["counts_match_records"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
