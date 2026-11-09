from __future__ import annotations

import pandas as pd
import pytest

from post_d008_lane1_check1 import reconcile_member_inventory


def test_member_repetition_across_molecules_is_expected():
    table = pd.DataFrame(
        {
            "molecule": ["mol1", "mol1", "mol2", "mol2"],
            "member": ["m1", "m2", "m1", "m2"],
        }
    )

    result = reconcile_member_inventory(
        table,
        ["m1", "m2"],
        molecule_id_column="molecule",
        member_id_column="member",
    )

    assert result["membership_exact"] is True
    assert result["n_intended_unique_members"] == 2
    assert result["n_observed_unique_members"] == 2
    assert result["missing_members"] == []
    assert result["unintended_members"] == []


def test_same_member_count_with_wrong_identity_does_not_pass():
    table = pd.DataFrame(
        {
            "molecule": ["mol1", "mol1", "mol2", "mol2"],
            "member": ["m1", "wrong", "m1", "wrong"],
        }
    )

    result = reconcile_member_inventory(
        table,
        ["m1", "m2"],
        molecule_id_column="molecule",
        member_id_column="member",
    )

    assert result["membership_exact"] is False
    assert result["n_intended_unique_members"] == 2
    assert result["n_observed_unique_members"] == 2
    assert result["missing_members"] == ["m2"]
    assert result["unintended_members"] == ["wrong"]


def test_duplicate_molecule_member_record_fails_closed():
    table = pd.DataFrame(
        {
            "molecule": ["mol1", "mol1", "mol1"],
            "member": ["m1", "m1", "m2"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate molecule/member",
    ):
        reconcile_member_inventory(
            table,
            ["m1", "m2"],
            molecule_id_column="molecule",
            member_id_column="member",
        )


def test_missing_member_column_fails_closed():
    table = pd.DataFrame(
        {
            "molecule": ["mol1", "mol2"],
        }
    )

    with pytest.raises(
        KeyError,
        match="missing required columns",
    ):
        reconcile_member_inventory(
            table,
            ["m1", "m2"],
            molecule_id_column="molecule",
            member_id_column="member",
        )


def test_duplicate_frozen_inventory_identity_fails_closed():
    table = pd.DataFrame(
        {
            "molecule": ["mol1", "mol1"],
            "member": ["m1", "m2"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Frozen inventory contains duplicate",
    ):
        reconcile_member_inventory(
            table,
            ["m1", "m1"],
            molecule_id_column="molecule",
            member_id_column="member",
        )


from post_d008_lane1_check1 import assess_recorded_artifact, sha256_file


def test_available_artifact_with_matching_recorded_hash_is_verified(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_bytes(b"frozen artifact\n")
    expected = sha256_file(artifact)

    result = assess_recorded_artifact(
        artifact,
        historically_recorded_sha256=expected,
    )

    assert result["status"] == "available_verified"
    assert result["observed_sha256"] == expected


def test_missing_artifact_is_unresolved(tmp_path):
    artifact = tmp_path / "missing.txt"

    result = assess_recorded_artifact(
        artifact,
        historically_recorded_sha256="a" * 64,
    )

    assert result["status"] == "missing_unresolved"
    assert result["observed_sha256"] is None


def test_hash_mismatch_requires_adjudication(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_bytes(b"current bytes\n")

    result = assess_recorded_artifact(
        artifact,
        historically_recorded_sha256="0" * 64,
    )

    assert result["status"] == "hash_mismatch_adjudication_required"
    assert result["observed_sha256"] != "0" * 64


def test_available_artifact_without_recorded_hash_is_not_claimed_verified(
    tmp_path,
):
    artifact = tmp_path / "artifact.txt"
    artifact.write_bytes(b"available but historically unhashed\n")

    result = assess_recorded_artifact(
        artifact,
        historically_recorded_sha256=None,
    )

    assert result["status"] == "available_identity_unverified"
    assert result["observed_sha256"] == sha256_file(artifact)


import json

from post_d008_lane1_check1 import audit_membership_and_coverage


AUDIT_ENDPOINTS = ["CYP1", "CYP2", "CYP3", "CYP4"]


def _write_audit_inventory(path):
    records = [
        {
            "identity": {
                "repetition": 1,
                "fold": 1,
                "seed": 101,
            }
        },
        {
            "identity": {
                "repetition": 1,
                "fold": 2,
                "seed": 102,
            }
        },
    ]
    path.write_text(json.dumps(records))


def _write_audit_member_table(path):
    rows = []
    for molecule in ["mol1", "mol2", "mol3"]:
        for fold, seed in [(1, 101), (2, 102)]:
            row = {
                "Molecule_Name": molecule,
                "repetition": 1,
                "fold": fold,
                "seed": seed,
            }
            for endpoint in AUDIT_ENDPOINTS:
                row[endpoint] = 999.0
            rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def _write_audit_ensemble_table(path):
    frame = pd.DataFrame(
        {
            "Molecule_Name": ["mol1", "mol2", "mol3"],
            **{
                endpoint: [999.0, 999.0, 999.0]
                for endpoint in AUDIT_ENDPOINTS
            },
        }
    )
    frame.to_csv(path, index=False)


def test_audit_complete_membership_and_coverage_pass(tmp_path):
    member = tmp_path / "members.csv"
    ensemble = tmp_path / "ensemble.csv"
    inventory = tmp_path / "inventory.json"

    _write_audit_member_table(member)
    _write_audit_ensemble_table(ensemble)
    _write_audit_inventory(inventory)

    result = audit_membership_and_coverage(
        member,
        ensemble,
        inventory,
        endpoint_columns=AUDIT_ENDPOINTS,
    )

    assert result["membership_status"] == "PASS"
    assert result["coverage_status"] == "PASS"
    assert result["numerical_status"] == "UNRESOLVED"
    assert result["n_intended_members"] == 2
    assert result["n_observed_members"] == 2
    assert result["n_ensemble_molecules"] == 3


def test_audit_wrong_member_identity_requires_adjudication(tmp_path):
    member = tmp_path / "members.csv"
    ensemble = tmp_path / "ensemble.csv"
    inventory = tmp_path / "inventory.json"

    _write_audit_member_table(member)
    _write_audit_ensemble_table(ensemble)
    _write_audit_inventory(inventory)

    frame = pd.read_csv(member)
    frame.loc[frame["fold"] == 2, "seed"] = 999
    frame.to_csv(member, index=False)

    result = audit_membership_and_coverage(
        member,
        ensemble,
        inventory,
        endpoint_columns=AUDIT_ENDPOINTS,
    )

    assert (
        result["membership_status"]
        == "FAIL_ADJUDICATION_REQUIRED"
    )
    assert result["coverage_status"] == (
        "NOT_EVALUATED_DUE_TO_MEMBERSHIP_FAILURE"
    )
    assert result["numerical_status"] == "UNRESOLVED"


def test_audit_missing_member_molecule_fails_coverage(tmp_path):
    member = tmp_path / "members.csv"
    ensemble = tmp_path / "ensemble.csv"
    inventory = tmp_path / "inventory.json"

    _write_audit_member_table(member)
    _write_audit_ensemble_table(ensemble)
    _write_audit_inventory(inventory)

    frame = pd.read_csv(member)
    frame = frame[
        ~(
            (frame["Molecule_Name"] == "mol3")
            & (frame["fold"] == 2)
        )
    ]
    frame.to_csv(member, index=False)

    result = audit_membership_and_coverage(
        member,
        ensemble,
        inventory,
        endpoint_columns=AUDIT_ENDPOINTS,
    )

    assert result["membership_status"] == "PASS"
    assert (
        result["coverage_status"]
        == "FAIL_ADJUDICATION_REQUIRED"
    )


def test_audit_unexpected_member_molecule_fails_coverage(tmp_path):
    member = tmp_path / "members.csv"
    ensemble = tmp_path / "ensemble.csv"
    inventory = tmp_path / "inventory.json"

    _write_audit_member_table(member)
    _write_audit_ensemble_table(ensemble)
    _write_audit_inventory(inventory)

    frame = pd.read_csv(member)

    extra = frame.iloc[[0]].copy()
    extra["Molecule_Name"] = "unexpected"
    frame = pd.concat([frame, extra], ignore_index=True)
    frame.to_csv(member, index=False)

    result = audit_membership_and_coverage(
        member,
        ensemble,
        inventory,
        endpoint_columns=AUDIT_ENDPOINTS,
    )

    assert result["membership_status"] == "PASS"
    assert (
        result["coverage_status"]
        == "FAIL_ADJUDICATION_REQUIRED"
    )


def test_audit_duplicate_molecule_member_record_fails_closed(tmp_path):
    member = tmp_path / "members.csv"
    ensemble = tmp_path / "ensemble.csv"
    inventory = tmp_path / "inventory.json"

    _write_audit_member_table(member)
    _write_audit_ensemble_table(ensemble)
    _write_audit_inventory(inventory)

    frame = pd.read_csv(member)
    frame = pd.concat(
        [frame, frame.iloc[[0]]],
        ignore_index=True,
    )
    frame.to_csv(member, index=False)

    with pytest.raises(
        ValueError,
        match="Duplicate molecule/member",
    ):
        audit_membership_and_coverage(
            member,
            ensemble,
            inventory,
            endpoint_columns=AUDIT_ENDPOINTS,
        )


def test_audit_duplicate_ensemble_molecule_fails_closed(tmp_path):
    member = tmp_path / "members.csv"
    ensemble = tmp_path / "ensemble.csv"
    inventory = tmp_path / "inventory.json"

    _write_audit_member_table(member)
    _write_audit_ensemble_table(ensemble)
    _write_audit_inventory(inventory)

    frame = pd.read_csv(ensemble)
    frame.loc[2, "Molecule_Name"] = "mol1"
    frame.to_csv(ensemble, index=False)

    with pytest.raises(
        ValueError,
        match="duplicate molecule",
    ):
        audit_membership_and_coverage(
            member,
            ensemble,
            inventory,
            endpoint_columns=AUDIT_ENDPOINTS,
        )


def test_audit_missing_endpoint_column_fails_closed(tmp_path):
    member = tmp_path / "members.csv"
    ensemble = tmp_path / "ensemble.csv"
    inventory = tmp_path / "inventory.json"

    _write_audit_member_table(member)
    _write_audit_ensemble_table(ensemble)
    _write_audit_inventory(inventory)

    frame = pd.read_csv(member)
    frame = frame.drop(columns=["CYP4"])
    frame.to_csv(member, index=False)

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        audit_membership_and_coverage(
            member,
            ensemble,
            inventory,
            endpoint_columns=AUDIT_ENDPOINTS,
        )


def test_audit_never_claims_numerical_resolution(tmp_path):
    member = tmp_path / "members.csv"
    ensemble = tmp_path / "ensemble.csv"
    inventory = tmp_path / "inventory.json"

    _write_audit_member_table(member)
    _write_audit_ensemble_table(ensemble)
    _write_audit_inventory(inventory)

    result = audit_membership_and_coverage(
        member,
        ensemble,
        inventory,
        endpoint_columns=AUDIT_ENDPOINTS,
    )

    assert result["numerical_status"] == "UNRESOLVED"
    assert "No tolerance was selected" in result[
        "numerical_limitation"
    ]
