from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


# Thirteen-column compatibility record consumed by the pinned
# ActFound chembl_preprocess.py, before that code appends SMILES.
#
# Column 2 is not consumed by pinned preprocessing/runtime code.
# We preserve ChEMBL record_id there as provenance.
#
# Column 12 is overwritten by upstream preprocessing with assay size
# after canonical SMILES has been appended.
RAW_COLUMNS = (
    "activity_id",          # 0
    "assay_id",             # 1
    "record_id_unused",     # 2
    "molregno",             # 3
    "bao_endpoint",         # 4
    "standard_relation",    # 5
    "standard_value",       # 6
    "standard_units",       # 7
    "standard_type",        # 8
    "assay_type",           # 9
    "bao_format",           # 10
    "assay_chembl_id",      # 11
    "count_scratch",        # 12
)

AUDIT_COLUMNS = RAW_COLUMNS + ("canonical_smiles",)


def projection_sql(
    limit: int | None = None,
    *,
    unit_eligible_only: bool = False,
) -> tuple[str, tuple[object, ...]]:
    """
    Return the ChEMBL32 -> ActFound compatibility projection.

    When unit_eligible_only=True, apply the exact pinned ActFound unit
    whitelist in SQLite. This is execution pushdown only; the scientific
    eligibility set is unchanged.
    """
    sql = """
    SELECT
        act.activity_id,
        act.assay_id,
        act.record_id,
        act.molregno,
        act.bao_endpoint,
        act.standard_relation,
        act.standard_value,
        act.standard_units,
        act.standard_type,
        ass.assay_type,
        ass.bao_format,
        ass.chembl_id,
        0 AS count_scratch,
        cs.canonical_smiles
    FROM activities AS act
    JOIN assays AS ass
      ON ass.assay_id = act.assay_id
    JOIN compound_structures AS cs
      ON cs.molregno = act.molregno
    WHERE act.standard_value IS NOT NULL
      AND act.standard_units IS NOT NULL
      AND act.standard_type IS NOT NULL
      AND cs.canonical_smiles IS NOT NULL
    """

    params: list[object] = []

    if unit_eligible_only:
        units = tuple(sorted(ACTFOUND_SELECTED_UNITS))
        placeholders = ",".join("?" for _ in units)
        sql += f"\n      AND act.standard_units IN ({placeholders})"
        params.extend(units)

    sql += "\n    ORDER BY act.activity_id"

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        sql += "\n    LIMIT ?"
        params.append(int(limit))

    result = (sql, tuple(params))
    return result


def compatibility_record(row: tuple[object, ...]) -> list[str]:
    """
    Convert one named SQL projection row into the 13 positional fields
    expected by pinned ActFound chembl_preprocess.py.

    Canonical SMILES is deliberately excluded here because upstream
    preprocessing obtains it separately from molsmi.csv and appends it.
    """
    if len(row) != len(AUDIT_COLUMNS):
        raise ValueError(
            f"expected {len(AUDIT_COLUMNS)} projected values, got {len(row)}"
        )

    raw = [
        "" if value is None else str(value)
        for value in row[:13]
    ]

    if len(raw) != 13:
        raise AssertionError("ActFound raw compatibility record must have 13 fields")

    result = raw
    return result


def write_projection(
    db_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """
    Write an audit TSV containing the 13 compatibility fields plus the
    canonical SMILES witness.

    This is not yet chembl_processed_chembl32.csv.
    """
    sql, params = projection_sql(limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    row_count = 0

    try:
        cursor = con.execute(sql, params)

        with output_path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(AUDIT_COLUMNS)

            for row in cursor:
                raw = compatibility_record(row)
                smiles = "" if row[13] is None else str(row[13])

                writer.writerow(raw + [smiles])
                row_count += 1
    finally:
        con.close()

    result = {
        "db_path": str(db_path),
        "output_path": str(output_path),
        "row_count": row_count,
        "limit": limit,
        "raw_field_count": len(RAW_COLUMNS),
        "audit_columns": list(AUDIT_COLUMNS),
    }
    return result




def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "projection",
            "materialize-eligible",
            "pregraph",
            "full-preprocess",
        ),
        default="projection",
    )
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
    )

    args = parser.parse_args()

    if args.mode == "projection":
        if args.output is None:
            raise ValueError(
                "--output is required for --mode projection"
            )

        result = write_projection(
            args.db,
            args.output,
            limit=args.limit,
        )

    elif args.mode == "materialize-eligible":
        if args.output is None:
            raise ValueError(
                "--output is required for --mode materialize-eligible"
            )

        if args.limit is not None:
            raise ValueError(
                "--limit is not permitted for "
                "--mode materialize-eligible"
            )

        result = materialize_unit_eligible_projection(
            args.db,
            args.output,
            progress_every=args.progress_every,
        )

    elif args.mode == "pregraph":
        if args.output is None:
            raise ValueError(
                "--output is required for --mode pregraph"
            )

        if args.limit is not None:
            raise ValueError(
                "--limit is not permitted for --mode pregraph"
            )

        result = process_materialized_pregraph_population(
            Path(args.db),
            args.output,
            progress_every=args.progress_every,
        )

    else:
        if args.output_dir is None:
            raise ValueError(
                "--output-dir is required for --mode full-preprocess"
            )

        if args.limit is not None:
            raise ValueError(
                "--limit is not permitted for --mode full-preprocess"
            )

        result = run_complete_preprocessing(
            args.db,
            args.output_dir,
        )

    import json
    print(json.dumps(result, indent=2, sort_keys=True))





ACTFOUND_SELECTED_UNITS_C = (
    "nM", "uM", "mm", "umol.kg-1", "mM", "uM kg-1", "10'-2umol",
    "M", "nM kg-1", "um", "umol/Kg", "nm", "10'-6M", "10'-7M",
    "10^-2mm", "nmol",
)

ACTFOUND_SELECTED_UNITS_M = (
    "ug.mL-1", "ug ml-1", "mg.kg-1", "mg kg-1",
    "mg/L", "ng/ml", "mg/ml", "ug kg-1", "mg/kg/day",
    "mg kg-1 day-1", "10'-4 ug/ml", "M kg-1", "10'-6 ug/ml",
    "ng/L", "pmg kg-1", "10'-8mg/ml", "ng ml-1",
    "10'-3 ug/ml", "10'-1 ug/ml",
)

ACTFOUND_SELECTED_UNITS = frozenset(
    ACTFOUND_SELECTED_UNITS_C + ACTFOUND_SELECTED_UNITS_M
)


def actfound_assay_key(record: list[str]) -> str:
    """Reproduce pinned chembl_preprocess.py lines 25-29."""
    if len(record) != 13:
        raise ValueError("ActFound raw record must have exactly 13 fields")

    result = f"{record[1]}_{record[7]}_{record[8]}"
    return result


def actfound_unit_eligible(record: list[str]) -> bool:
    """Reproduce the executed ChEMBL unit gate at upstream lines 25-27."""
    if len(record) != 13:
        raise ValueError("ActFound raw record must have exactly 13 fields")

    result = record[7] in ACTFOUND_SELECTED_UNITS
    return result


def remove_duplicate_ligands_exact_actfound(
    records: list[list[str]],
) -> list[list[str]]:
    """
    Reproduce pinned ActFound chembl_preprocess.py lines 38-71.

    The upstream second-largest-value comparison is intentionally preserved.
    """
    by_molregno: dict[str, list[list[str]]] = {}

    for record in records:
        if len(record) != 13:
            raise ValueError(
                "ActFound raw record must have exactly 13 fields"
            )

        molregno = record[3]
        by_molregno.setdefault(molregno, []).append(record)

    retained: list[list[str]] = []

    for _molregno, group in by_molregno.items():
        if len(group) == 1:
            representative = list(group[0])
        else:
            values = sorted(float(record[6].strip()) for record in group)

            if values[0] <= 0:
                continue

            # Exact upstream behavior: compare SECOND-LARGEST, not maximum.
            if values[-2] > 10 * values[0]:
                continue

            representative = list(group[0])
            representative[6] = str(sum(values) / len(values))

        if float(representative[6]) <= 0:
            continue

        retained.append(representative)

    result = retained
    return result

def group_unit_eligible_records_exact_actfound(
    records: list[list[str]],
) -> dict[str, list[list[str]]]:
    """
    Reproduce chembl_preprocess.py lines 23-35.

    Only units in the executed ActFound ChEMBL whitelist enter an assay task.
    Task identity is assay_id + standard_units + standard_type.
    """
    grouped: dict[str, list[list[str]]] = {}

    for record in records:
        if not actfound_unit_eligible(record):
            continue

        key = actfound_assay_key(record)
        grouped.setdefault(key, []).append(list(record))

    result = grouped
    return result


def pregraph_assay_filter_exact_actfound(
    grouped: dict[str, list[list[str]]],
) -> dict[str, list[list[str]]]:
    """
    Reproduce the executed behavior of chembl_preprocess.py lines 74-89.

    The upstream '%' branch is unreachable because '%' is not admitted by the
    earlier unit whitelist. We therefore do not introduce a new percent path.

    Raw assay task must contain >=20 records.
    Duplicate-molregno handling is then applied.
    Deduplicated assay task must still contain >=20 molecules.
    """
    retained: dict[str, list[list[str]]] = {}

    for assay_key, records in grouped.items():
        if len(records) < 20:
            continue

        deduplicated = remove_duplicate_ligands_exact_actfound(records)

        if len(deduplicated) < 20:
            continue

        # Upstream writes assay size into raw index 12 after SMILES has been
        # appended. Our compatibility record keeps that scratch position at 12.
        for record in deduplicated:
            record[12] = str(len(deduplicated))

        retained[assay_key] = deduplicated

    result = retained
    return result


def actfound_pair_allowed(
    *,
    prefix_a: str,
    prefix_b: str,
    affinity_a: float,
    affinity_b: float,
    similarity: float,
) -> bool:
    """
    Reproduce chembl_preprocess.py lines 137-150 after the >=0.3 similarity
    candidate gate has already been satisfied.
    """
    if prefix_a == prefix_b == "<":
        return False

    if prefix_a == prefix_b == ">":
        return False

    if affinity_a - affinity_b == 0:
        if similarity <= 0.7:
            return False

    result = True
    return result


def actfound_degree_threshold(n_molecules: int) -> float:
    """
    Exact threshold expression from upstream line 167:
        sqrt(sqrt(N)) * 2
    """
    if n_molecules <= 0:
        raise ValueError("n_molecules must be positive")

    import math

    result = math.sqrt(math.sqrt(n_molecules)) * 2
    return result


def construct_graph_exact_actfound(
    records: list[list[str]],
    smiles_by_molregno: dict[str, str],
) -> tuple[list[int] | None, int | None]:
    """
    Reproduce chembl_preprocess.py lines 103-151.

    Returns:
        degree vector, number of retained molecular pairs

    Upstream behavior deliberately preserved:
      - assay size >=10000 -> rejected before graph construction
      - Morgan radius 2
      - 1024 bits
      - chirality enabled
      - Tanimoto candidate threshold >=0.3
      - same '<'/'<' and '>'/'>' pairs excluded
      - equal transformed affinity with similarity <=0.7 excluded
    """
    if len(records) >= 10000:
        return None, None

    import math
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fps = []
    affinities = []

    for record in records:
        molregno = record[3]

        if molregno not in smiles_by_molregno:
            raise KeyError(f"missing canonical SMILES for molregno {molregno}")

        smiles = smiles_by_molregno[molregno]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(
                f"RDKit failed to parse canonical SMILES for molregno {molregno}"
            )

        mol = Chem.RemoveHs(mol)

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=1024,
            useChirality=True,
        )

        fps.append(fp)
        affinities.append(-math.log10(float(record[6])))

    degree = [0 for _ in fps]
    pair_count = 0

    for i in range(len(fps) - 1):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fps[i],
                fps[(i + 1):],
            )
        )

        candidate_indexes = np.where(similarities >= 0.3)[0]

        for relative_index in candidate_indexes:
            j = i + 1 + int(relative_index)
            similarity = float(similarities[relative_index])

            if not actfound_pair_allowed(
                prefix_a=records[i][5],
                prefix_b=records[j][5],
                affinity_a=affinities[i],
                affinity_b=affinities[j],
                similarity=similarity,
            ):
                continue

            degree[i] += 1
            degree[j] += 1
            pair_count += 1

    result = (degree, pair_count)
    return result


def postgraph_assay_filter_exact_actfound(
    records: list[list[str]],
    degree: list[int] | None,
    pair_count: int | None,
) -> list[list[str]] | None:
    """
    Reproduce chembl_preprocess.py lines 160-174.

    Eligibility is based on the number of molecules satisfying:
        degree >= sqrt(sqrt(N)) * 2

    At least 20 such molecules are required.

    The final emitted assay nevertheless contains ALL degree>0 molecules,
    matching upstream behavior.
    """
    if degree is None or pair_count is None:
        return None

    if len(records) != len(degree):
        raise ValueError("degree vector length does not match assay records")

    threshold = actfound_degree_threshold(len(records))

    valid_count = sum(
        1
        for value in degree
        if value >= threshold
    )

    if valid_count < 20:
        return None

    retained = [
        list(record)
        for record, value in zip(records, degree)
        if value > 0
    ]

    for record in retained:
        record[12] = str(len(retained))

    result = retained
    return result


def final_actfound_row(
    record: list[str],
    *,
    canonical_smiles: str,
    pair_count: int,
) -> list[str]:
    """
    Construct the exact final positional row consumed by read_chembl_assay().

    Raw fields 0..12
    + canonical SMILES at 13
    + graph pair count at 14.
    """
    if len(record) != 13:
        raise ValueError("ActFound raw record must have 13 fields")

    if canonical_smiles == "":
        raise ValueError("canonical SMILES must be nonempty")

    result = list(record) + [
        canonical_smiles,
        str(pair_count),
    ]
    return result


def iter_projection_rows(
    db_path: Path,
    *,
    unit_eligible_only: bool = False,
) -> object:
    """
    Stream named ChEMBL32 compatibility projection rows.

    unit_eligible_only=True pushes the exact pinned ActFound unit gate
    into SQLite without changing scientific eligibility.
    """
    sql, params = projection_sql(
        limit=None,
        unit_eligible_only=unit_eligible_only,
    )

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    try:
        cursor = con.execute(sql, params)

        for row in cursor:
            raw = compatibility_record(row)
            smiles = "" if row[13] is None else str(row[13])
            yield raw, smiles
    finally:
        con.close()


def build_pregraph_population(
    db_path: Path,
) -> tuple[
    dict[str, list[list[str]]],
    dict[str, str],
    dict[str, int],
]:
    """
    Materialize the population immediately before ActFound graph construction.

    This reproduces:
      projection
      -> unit gate
      -> assay grouping
      -> raw >=20
      -> duplicate handling
      -> deduplicated >=20

    No RDKit operation occurs here.
    """
    grouped: dict[str, list[list[str]]] = {}
    smiles_by_molregno: dict[str, str] = {}

    projected_records = 0
    unit_eligible_records = 0

    for raw, smiles in iter_projection_rows(
        db_path,
        unit_eligible_only=True,
    ):
        # SQL has already applied the exact upstream unit gate.
        # Retain the Python assertion as a scientific invariant.
        if not actfound_unit_eligible(raw):
            raise AssertionError(
                "SQL unit pushdown admitted an ActFound-ineligible record"
            )

        projected_records += 1
        unit_eligible_records += 1

        molregno = raw[3]

        previous = smiles_by_molregno.get(molregno)

        if previous is not None and previous != smiles:
            raise ValueError(
                "multiple canonical SMILES observed for "
                f"molregno {molregno}"
            )

        smiles_by_molregno[molregno] = smiles

        key = actfound_assay_key(raw)
        grouped.setdefault(key, []).append(raw)

    grouped_assays = len(grouped)
    grouped_records = sum(len(x) for x in grouped.values())

    raw_ge20 = {
        key: records
        for key, records in grouped.items()
        if len(records) >= 20
    }

    raw_ge20_assays = len(raw_ge20)
    raw_ge20_records = sum(len(x) for x in raw_ge20.values())

    deduplicated_all: dict[str, list[list[str]]] = {}

    for key, records in raw_ge20.items():
        deduplicated_all[key] = (
            remove_duplicate_ligands_exact_actfound(records)
        )

    deduplicated_records_before_ge20 = sum(
        len(x)
        for x in deduplicated_all.values()
    )

    pregraph: dict[str, list[list[str]]] = {}

    for key, records in deduplicated_all.items():
        if len(records) < 20:
            continue

        copied = [list(record) for record in records]

        for record in copied:
            record[12] = str(len(copied))

        pregraph[key] = copied

    summary = {
        "unit_eligible_projected_records": projected_records,
        "unit_eligible_records": unit_eligible_records,
        "grouped_assays": grouped_assays,
        "grouped_records": grouped_records,
        "raw_ge20_assays": raw_ge20_assays,
        "raw_ge20_records": raw_ge20_records,
        "deduplicated_records_before_ge20":
            deduplicated_records_before_ge20,
        "pregraph_assays": len(pregraph),
        "pregraph_records": sum(len(x) for x in pregraph.values()),
    }

    result = (pregraph, smiles_by_molregno, summary)
    return result


def run_graph_stage(
    pregraph: dict[str, list[list[str]]],
    smiles_by_molregno: dict[str, str],
) -> tuple[
    dict[str, tuple[list[list[str]], int]],
    dict[str, int],
]:
    """
    Execute the exact ActFound graph stage.

    Output mapping:
        assay_key -> (final degree>0 records, pair_count)
    """
    final: dict[str, tuple[list[list[str]], int]] = {}

    graph_attempted_assays = 0
    graph_size_rejected_assays = 0
    graph_valid_count_rejected_assays = 0
    graph_qualified_assays = 0

    graph_input_records = 0
    final_records = 0

    for assay_key, records in pregraph.items():
        graph_input_records += len(records)

        if len(records) >= 10000:
            graph_size_rejected_assays += 1
            continue

        graph_attempted_assays += 1

        degree, pair_count = construct_graph_exact_actfound(
            records,
            smiles_by_molregno,
        )

        retained = postgraph_assay_filter_exact_actfound(
            records,
            degree,
            pair_count,
        )

        if retained is None:
            graph_valid_count_rejected_assays += 1
            continue

        assert pair_count is not None

        final[assay_key] = (retained, pair_count)
        graph_qualified_assays += 1
        final_records += len(retained)

    summary = {
        "graph_input_assays": len(pregraph),
        "graph_input_records": graph_input_records,
        "graph_attempted_assays": graph_attempted_assays,
        "graph_size_rejected_assays": graph_size_rejected_assays,
        "graph_valid_count_rejected_assays":
            graph_valid_count_rejected_assays,
        "graph_qualified_assays": graph_qualified_assays,
        "final_records": final_records,
    }

    result = (final, summary)
    return result


def write_final_actfound_outputs(
    final_population: dict[str, tuple[list[list[str]], int]],
    smiles_by_molregno: dict[str, str],
    output_csv: Path,
    provenance_tsv: Path,
) -> dict[str, object]:
    """
    Write:
      1. headerless 15-field CSV consumed by pinned read_chembl_assay()
      2. named provenance TSV for project auditing/firewall construction
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    provenance_tsv.parent.mkdir(parents=True, exist_ok=True)

    provenance_header = [
        "assay_key",
        "activity_id",
        "assay_id",
        "record_id_unused",
        "molregno",
        "bao_endpoint",
        "standard_relation",
        "standard_value",
        "standard_units",
        "standard_type",
        "assay_type",
        "bao_format",
        "assay_chembl_id",
        "final_assay_size",
        "canonical_smiles",
        "graph_pair_count",
    ]

    row_count = 0

    with (
        output_csv.open("w", newline="") as csv_file,
        provenance_tsv.open("w", newline="") as provenance_file,
    ):
        csv_writer = csv.writer(csv_file)
        provenance_writer = csv.writer(
            provenance_file,
            delimiter="\t",
        )

        provenance_writer.writerow(provenance_header)

        for assay_key in sorted(final_population):
            records, pair_count = final_population[assay_key]

            for record in records:
                molregno = record[3]
                smiles = smiles_by_molregno[molregno]

                final_row = final_actfound_row(
                    record,
                    canonical_smiles=smiles,
                    pair_count=pair_count,
                )

                csv_writer.writerow(final_row)

                provenance_writer.writerow(
                    [assay_key]
                    + list(record)
                    + [smiles, str(pair_count)]
                )

                row_count += 1

    result = {
        "row_count": row_count,
        "assay_count": len(final_population),
        "output_csv": str(output_csv),
        "provenance_tsv": str(provenance_tsv),
    }
    return result


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    result = h.hexdigest()
    return result


def runtime_versions() -> dict[str, str]:
    import platform

    versions = {
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
    }

    try:
        import rdkit
        versions["rdkit"] = rdkit.__version__
    except Exception as exc:
        # RDKit is scientifically required for the graph stage, but a
        # local provenance-only run must not fail merely because the
        # developer environment cannot import the pinned RDKit build.
        versions["rdkit"] = "UNAVAILABLE"
        versions["rdkit_import_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    result = versions
    return result


def run_complete_preprocessing(
    db_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """
    Execute the complete upstream-compatible ChEMBL32 preprocessing.

    IMPORTANT:
    This reconstructs the ActFound source corpus only.
    The project's OpenADMET identity firewall is deliberately NOT applied here.
    """
    import json
    import time

    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    pregraph, smiles_by_molregno, pregraph_summary = (
        build_pregraph_population(db_path)
    )

    graph_started = time.time()

    final_population, graph_summary = run_graph_stage(
        pregraph,
        smiles_by_molregno,
    )

    graph_seconds = time.time() - graph_started

    output_csv = output_dir / "chembl_processed_chembl32.csv"
    provenance_tsv = (
        output_dir
        / "chembl_processed_chembl32_provenance.tsv"
    )

    output_summary = write_final_actfound_outputs(
        final_population,
        smiles_by_molregno,
        output_csv,
        provenance_tsv,
    )

    total_seconds = time.time() - started

    summary = {
        "artifact_type":
            "ActFound_ChEMBL32_upstream_compatible_reconstruction_v1",
        "source_database": str(db_path),
        "openadmet_firewall_applied": False,
        "pregraph": pregraph_summary,
        "graph": graph_summary,
        "outputs": output_summary,
        "runtime": {
            **runtime_versions(),
            "graph_seconds": graph_seconds,
            "total_seconds": total_seconds,
        },
    }

    summary_path = output_dir / "preprocessing_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    hashes = {
        output_csv.name: file_sha256(output_csv),
        provenance_tsv.name: file_sha256(provenance_tsv),
        summary_path.name: file_sha256(summary_path),
    }

    hash_path = output_dir / "SHA256SUMS.txt"

    with hash_path.open("w") as f:
        for name in sorted(hashes):
            f.write(f"{hashes[name]}  {name}\n")

    result = {
        **summary,
        "sha256": hashes,
        "sha256_manifest": str(hash_path),
    }
    return result


def materialize_unit_eligible_projection(
    db_path: Path,
    output_path: Path,
    *,
    progress_every: int = 250_000,
) -> dict[str, object]:
    """
    Materialize the exact ChEMBL32 rows surviving the pinned ActFound
    ChEMBL unit gate.

    This is a semantics-preserving execution boundary:
      ChEMBL32 SQLite
      -> exact upstream unit whitelist
      -> deterministic activity_id order
      -> 13-field compatibility record + canonical SMILES witness

    No duplicate handling, >=20 assay rule, graph filtering, or OpenADMET
    firewall is applied here.
    """
    import json
    import sys
    import time

    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sql, params = projection_sql(
        limit=None,
        unit_eligible_only=True,
    )

    started = time.perf_counter()
    row_count = 0

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    try:
        cursor = con.execute(sql, params)

        with output_path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(AUDIT_COLUMNS)

            for row in cursor:
                raw = compatibility_record(row)

                # Independent invariant: SQL pushdown may not broaden or
                # otherwise alter the upstream scientific unit gate.
                if not actfound_unit_eligible(raw):
                    raise AssertionError(
                        "SQL unit pushdown emitted an ActFound-ineligible row"
                    )

                smiles = "" if row[13] is None else str(row[13])

                if smiles == "":
                    raise AssertionError(
                        "materialized projection contains empty canonical SMILES"
                    )

                writer.writerow(raw + [smiles])
                row_count += 1

                if row_count % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    rate = row_count / elapsed if elapsed > 0 else 0.0
                    print(
                        f"progress_rows={row_count} "
                        f"elapsed_seconds={elapsed:.3f} "
                        f"rows_per_second={rate:.1f}",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        con.close()

    elapsed = time.perf_counter() - started

    output_sha256 = file_sha256(output_path)

    summary = {
        "artifact_type":
            "ActFound_ChEMBL32_unit_eligible_projection_v1",
        "source_database": str(db_path),
        "source_database_size_bytes": db_path.stat().st_size,
        "scientific_gate":
            "exact pinned ActFound ChEMBL unit whitelist",
        "ordering": "activity_id ascending",
        "row_count": row_count,
        "columns": list(AUDIT_COLUMNS),
        "elapsed_seconds": elapsed,
        "rows_per_second":
            row_count / elapsed if elapsed > 0 else None,
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "runtime": runtime_versions(),
        "script_sha256": file_sha256(Path(__file__)),
        "openadmet_firewall_applied": False,
        "duplicate_resolution_applied": False,
        "assay_size_filter_applied": False,
        "graph_filter_applied": False,
    }

    summary_path = output_path.with_suffix(
        output_path.suffix + ".summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    manifest_path = output_path.with_suffix(
        output_path.suffix + ".SHA256SUMS.txt"
    )

    manifest_path.write_text(
        f"{output_sha256}  {output_path.name}\n"
        f"{file_sha256(summary_path)}  {summary_path.name}\n"
    )

    result = {
        **summary,
        "summary_path": str(summary_path),
        "sha256_manifest": str(manifest_path),
    }
    return result


def iter_materialized_unit_eligible_projection(
    projection_path: Path,
) -> object:
    """
    Read the frozen unit-eligible projection without changing its semantics.

    Yields:
        (raw_13_field_record, canonical_smiles)
    """
    with projection_path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")

        header = next(reader, None)

        if header != list(AUDIT_COLUMNS):
            raise ValueError(
                "materialized projection header does not match "
                "the frozen ActFound compatibility contract"
            )

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(AUDIT_COLUMNS):
                raise ValueError(
                    f"line {line_number}: expected "
                    f"{len(AUDIT_COLUMNS)} fields, got {len(row)}"
                )

            raw = row[:13]
            smiles = row[13]

            if not actfound_unit_eligible(raw):
                raise AssertionError(
                    f"line {line_number}: unit-ineligible row "
                    "found in frozen eligible projection"
                )

            if smiles == "":
                raise AssertionError(
                    f"line {line_number}: empty canonical SMILES"
                )

            yield raw, smiles


def process_materialized_pregraph_population(
    projection_path: Path,
    output_path: Path,
    *,
    progress_every: int = 500_000,
) -> dict[str, object]:
    """
    Process the frozen ActFound unit-eligible projection through the exact
    upstream pregraph rules:

        unit-eligible frozen rows
        -> assay grouping
        -> raw >=20 assay gate
        -> exact duplicate-ligand handling
        -> deduplicated >=20 assay gate

    No RDKit graph construction occurs here.

    The output is a TSV containing the 13 raw compatibility fields plus
    canonical SMILES, one row per molecule entering the graph stage.
    """
    import json
    import sys
    import time

    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    # The upstream code groups an assay in memory before applying the
    # >=20 / duplicate filters. We preserve that behavior exactly.
    grouped: dict[str, list[list[str]]] = {}

    input_rows = 0
    progress_start = started

    for raw, smiles in iter_materialized_unit_eligible_projection(
        projection_path
    ):
        input_rows += 1

        record = list(raw)

        # Preserve SMILES outside the positional record while grouping.
        record.append(smiles)

        key = actfound_assay_key(record[:13])
        grouped.setdefault(key, []).append(record)

        if input_rows % progress_every == 0:
            elapsed = time.perf_counter() - progress_start
            rate = (
                input_rows / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"group_progress_rows={input_rows} "
                f"elapsed_seconds={elapsed:.3f} "
                f"rows_per_second={rate:.1f} "
                f"assays_so_far={len(grouped)}",
                file=sys.stderr,
                flush=True,
            )

    grouped_assays = len(grouped)

    raw_ge20 = {
        key: rows
        for key, rows in grouped.items()
        if len(rows) >= 20
    }

    raw_ge20_assays = len(raw_ge20)
    raw_ge20_rows = sum(len(rows) for rows in raw_ge20.values())

    deduplicated: dict[str, list[list[str]]] = {}

    duplicate_input_rows = 0
    duplicate_removed_molecules = 0

    for key, rows_with_smiles in raw_ge20.items():
        # Strip the separately stored SMILES before applying the exact
        # upstream duplicate function.
        raw_records = [row[:13] for row in rows_with_smiles]

        unique_molregnos = len({row[3] for row in raw_records})

        cleaned = remove_duplicate_ligands_exact_actfound(
            raw_records
        )

        duplicate_input_rows += len(raw_records)
        duplicate_removed_molecules += (
            unique_molregnos - len(cleaned)
        )

        # Restore canonical SMILES by molregno.
        smiles_for_key = {
            row[3]: row[13]
            for row in rows_with_smiles
        }

        if len(cleaned) < 20:
            continue

        restored = []

        for raw in cleaned:
            restored.append(
                raw + [smiles_for_key[raw[3]]]
            )

        for row in restored:
            row[12] = str(len(restored))

        deduplicated[key] = restored

    dedup_ge20_assays = len(deduplicated)
    dedup_ge20_rows = sum(
        len(rows)
        for rows in deduplicated.values()
    )

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow(AUDIT_COLUMNS)

        for key in sorted(deduplicated):
            for row in deduplicated[key]:
                writer.writerow(row)

    elapsed = time.perf_counter() - started

    output_sha256 = file_sha256(output_path)

    summary = {
        "artifact_type":
            "ActFound_ChEMBL32_pregraph_population_v1",
        "source_projection":
            str(projection_path),
        "source_projection_sha256":
            file_sha256(projection_path),
        "input_rows": input_rows,
        "grouped_assays": grouped_assays,
        "raw_ge20_assays": raw_ge20_assays,
        "raw_ge20_rows": raw_ge20_rows,
        "dedup_ge20_assays": dedup_ge20_assays,
        "dedup_ge20_rows": dedup_ge20_rows,
        "duplicate_removed_molecules":
            duplicate_removed_molecules,
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "elapsed_seconds": elapsed,
        "rows_per_second":
            input_rows / elapsed
            if elapsed > 0
            else None,
        "rdkit_graph_applied": False,
        "openadmet_firewall_applied": False,
        "runtime": runtime_versions(),
        "script_sha256": file_sha256(Path(__file__)),
    }

    summary_path = output_path.with_suffix(
        output_path.suffix + ".summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    manifest_path = output_path.with_suffix(
        output_path.suffix + ".SHA256SUMS.txt"
    )

    manifest_path.write_text(
        f"{output_sha256}  {output_path.name}\n"
        f"{file_sha256(summary_path)}  {summary_path.name}\n"
    )

    result = {
        **summary,
        "summary_path": str(summary_path),
        "sha256_manifest": str(manifest_path),
    }
    return result

if __name__ == "__main__":
    main()
