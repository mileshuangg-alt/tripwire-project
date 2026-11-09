from __future__ import annotations

from collections.abc import Hashable, Iterable

import pandas as pd


def reconcile_member_inventory(
    member_predictions: pd.DataFrame,
    frozen_member_ids: Iterable[Hashable],
    *,
    molecule_id_column: str,
    member_id_column: str,
) -> dict[str, object]:
    """
    Reconcile observed member identity against the frozen intended inventory.

    Repetition of a member across different molecules is expected.
    Duplicate (molecule, member) records are forbidden.

    This function performs identity/integrity measurement only. It does not
    reconstruct ensemble predictions or score model performance.
    """
    required_columns = {molecule_id_column, member_id_column}
    missing_columns = required_columns - set(member_predictions.columns)
    if missing_columns:
        raise KeyError(
            "Member-prediction table missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if member_predictions[molecule_id_column].isna().any():
        raise ValueError(
            "Member-prediction table contains missing molecule IDs."
        )

    if member_predictions[member_id_column].isna().any():
        raise ValueError(
            "Member-prediction table contains missing member IDs."
        )

    frozen_ids = list(frozen_member_ids)

    if any(
        member_id is None or pd.isna(member_id)
        for member_id in frozen_ids
    ):
        raise ValueError("Frozen inventory contains missing member IDs.")

    if len(frozen_ids) != len(set(frozen_ids)):
        raise ValueError(
            "Frozen inventory contains duplicate member identities."
        )

    duplicate_record_mask = member_predictions.duplicated(
        subset=[molecule_id_column, member_id_column],
        keep=False,
    )
    if duplicate_record_mask.any():
        duplicate_records = (
            member_predictions.loc[
                duplicate_record_mask,
                [molecule_id_column, member_id_column],
            ]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(
            "Duplicate molecule/member records in member-prediction table: "
            f"{duplicate_records}"
        )

    intended_members = set(frozen_ids)
    observed_members = set(
        member_predictions[member_id_column].tolist()
    )

    missing_members = sorted(
        intended_members - observed_members,
        key=str,
    )
    unintended_members = sorted(
        observed_members - intended_members,
        key=str,
    )

    result = {
        "n_intended_unique_members": len(intended_members),
        "n_observed_unique_members": len(observed_members),
        "missing_members": missing_members,
        "unintended_members": unintended_members,
        "membership_exact": (
            observed_members == intended_members
            and len(frozen_ids) == len(intended_members)
        ),
    }
    return result


from pathlib import Path
import hashlib


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 identity of an existing file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assess_recorded_artifact(
    path: str | Path,
    *,
    historically_recorded_sha256: str | None,
) -> dict[str, object]:
    """
    Assess availability and, when available, identity against a historically
    recorded SHA-256.

    Missing evidence is unresolved.
    A hash mismatch requires adjudication and does not establish that the
    historical artifact or original D008 calculation was wrong.
    """
    artifact_path = Path(path)

    if not artifact_path.is_file():
        result = {
            "path": str(artifact_path),
            "status": "missing_unresolved",
            "historically_recorded_sha256": historically_recorded_sha256,
            "observed_sha256": None,
        }
        return result

    if historically_recorded_sha256 is None:
        result = {
            "path": str(artifact_path),
            "status": "available_identity_unverified",
            "historically_recorded_sha256": None,
            "observed_sha256": sha256_file(artifact_path),
        }
        return result

    observed_sha256 = sha256_file(artifact_path)

    if observed_sha256 != historically_recorded_sha256:
        result = {
            "path": str(artifact_path),
            "status": "hash_mismatch_adjudication_required",
            "historically_recorded_sha256": historically_recorded_sha256,
            "observed_sha256": observed_sha256,
        }
        return result

    result = {
        "path": str(artifact_path),
        "status": "available_verified",
        "historically_recorded_sha256": historically_recorded_sha256,
        "observed_sha256": observed_sha256,
    }
    return result


import json


def audit_membership_and_coverage(
    member_path: str | Path,
    ensemble_path: str | Path,
    inventory_path: str | Path,
    *,
    endpoint_columns: list[str],
) -> dict[str, object]:
    """
    Audit evidence-supported Check-1 membership and coverage invariants.

    Prediction values are not loaded or compared. Numerical consistency is
    explicitly unresolved because no justified quantitative comparison
    criterion is available from the preserved provenance.
    """
    member_path = Path(member_path)
    ensemble_path = Path(ensemble_path)
    inventory_path = Path(inventory_path)

    member_metadata_columns = [
        "Molecule_Name",
        "repetition",
        "fold",
        "seed",
    ]

    # Headers only: establish required endpoint/schema presence without loading
    # any prediction values.
    member_header = pd.read_csv(member_path, nrows=0)
    ensemble_header = pd.read_csv(ensemble_path, nrows=0)

    required_member_columns = (
        set(member_metadata_columns) | set(endpoint_columns)
    )
    missing_member_columns = (
        required_member_columns - set(member_header.columns)
    )
    if missing_member_columns:
        raise ValueError(
            "Member artifact missing required columns: "
            f"{sorted(missing_member_columns)}"
        )

    required_ensemble_columns = (
        {"Molecule_Name"} | set(endpoint_columns)
    )
    missing_ensemble_columns = (
        required_ensemble_columns - set(ensemble_header.columns)
    )
    if missing_ensemble_columns:
        raise ValueError(
            "Ensemble artifact missing required columns: "
            f"{sorted(missing_ensemble_columns)}"
        )

    # Metadata only. Endpoint prediction values are deliberately not loaded.
    members = pd.read_csv(
        member_path,
        usecols=member_metadata_columns,
    )
    ensemble = pd.read_csv(
        ensemble_path,
        usecols=["Molecule_Name"],
    )

    inventory_raw = json.loads(
        inventory_path.read_text(encoding="utf-8")
    )
    if not isinstance(inventory_raw, list):
        raise ValueError(
            "Frozen checkpoint inventory is not a list."
        )

    intended_ids = []
    for record in inventory_raw:
        if not isinstance(record, dict):
            raise ValueError(
                "Frozen checkpoint inventory contains a non-dictionary record."
            )

        identity = record.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(
                "Inventory record missing identity dictionary."
            )

        try:
            member_id = (
                int(identity["repetition"]),
                int(identity["fold"]),
                int(identity["seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Inventory identity does not contain valid "
                "repetition/fold/seed."
            ) from exc

        intended_ids.append(member_id)

    observed = members.copy()

    try:
        observed["member_identity"] = list(
            zip(
                observed["repetition"].astype(int),
                observed["fold"].astype(int),
                observed["seed"].astype(int),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Member artifact contains invalid repetition/fold/seed."
        ) from exc

    inventory_result = reconcile_member_inventory(
        observed,
        intended_ids,
        molecule_id_column="Molecule_Name",
        member_id_column="member_identity",
    )

    if not inventory_result["membership_exact"]:
        result = {
            "membership_status": "FAIL_ADJUDICATION_REQUIRED",
            "coverage_status": (
                "NOT_EVALUATED_DUE_TO_MEMBERSHIP_FAILURE"
            ),
            "numerical_status": "UNRESOLVED",
            "inventory": inventory_result,
            "numerical_limitation": (
                "Preserved provenance does not justify a quantitative "
                "criterion relating separately serialized member and "
                "ensemble predictions. No tolerance was selected."
            ),
        }
        return result

    if ensemble["Molecule_Name"].isna().any():
        raise ValueError(
            "Ensemble artifact contains missing molecule identity."
        )

    if ensemble["Molecule_Name"].duplicated().any():
        raise ValueError(
            "Ensemble artifact contains duplicate molecule identity."
        )

    expected_molecules = set(
        ensemble["Molecule_Name"].tolist()
    )

    coverage_failures = []

    for member_id, group in observed.groupby(
        "member_identity",
        sort=False,
    ):
        observed_molecules = set(
            group["Molecule_Name"].tolist()
        )

        if observed_molecules != expected_molecules:
            coverage_failures.append(
                {
                    "member_identity": list(member_id),
                    "missing_molecules": sorted(
                        expected_molecules - observed_molecules,
                        key=str,
                    ),
                    "unexpected_molecules": sorted(
                        observed_molecules - expected_molecules,
                        key=str,
                    ),
                }
            )

    result = {
        "membership_status": "PASS",
        "coverage_status": (
            "PASS"
            if not coverage_failures
            else "FAIL_ADJUDICATION_REQUIRED"
        ),
        "numerical_status": "UNRESOLVED",
        "n_intended_members": len(set(intended_ids)),
        "n_observed_members": int(
            observed["member_identity"].nunique()
        ),
        "n_ensemble_molecules": len(expected_molecules),
        "expected_endpoint_columns": list(endpoint_columns),
        "inventory": inventory_result,
        "coverage_failures": coverage_failures,
        "numerical_limitation": (
            "Preserved provenance does not justify a quantitative "
            "criterion relating separately serialized member and "
            "ensemble predictions. No tolerance was selected."
        ),
    }
    return result
