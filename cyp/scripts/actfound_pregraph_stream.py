from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sqlite3
import time
from pathlib import Path
from typing import Iterator

from actfound_chembl32_projection import (
    AUDIT_COLUMNS,
    actfound_assay_key,
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
    }

    try:
        import rdkit

        versions["rdkit"] = rdkit.__version__
    except Exception as exc:
        versions["rdkit"] = "UNAVAILABLE"
        versions["rdkit_import_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return versions


def iter_sorted_projection(
    path: Path,
) -> Iterator[list[str]]:
    """
    Read the externally sorted 14-field ActFound unit-eligible projection.

    Expected sort semantics:
        assay_id, standard_units, standard_type,
        molregno, activity_id

    No scientific filtering occurs here.
    """
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")

        header = next(reader, None)

        if header != list(AUDIT_COLUMNS):
            raise ValueError(
                "sorted projection header does not match "
                "AUDIT_COLUMNS"
            )

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(AUDIT_COLUMNS):
                raise ValueError(
                    f"line {line_number}: expected "
                    f"{len(AUDIT_COLUMNS)} fields, got {len(row)}"
                )

            yield row


def resolve_molecule_group_exact_actfound(
    rows: list[list[str]],
) -> list[str] | None:
    """
    Exact upstream remove_duplicate_ligands() behavior for one molregno.

    The input rows are activity_id-sorted, so rows[0] is the same
    representative that upstream selects as lines[0].

    Returns:
        retained representative row, or None if upstream discards it.
    """
    if not rows:
        raise ValueError("empty molecule group")

    if len(rows) == 1:
        representative = list(rows[0])
    else:
        values = sorted(
            float(row[6].strip())
            for row in rows
        )

        if values[0] <= 0:
            return None

        # Preserve the upstream second-largest comparison exactly.
        if values[-2] > 10 * values[0]:
            return None

        representative = list(rows[0])
        representative[6] = str(sum(values) / len(values))

    if float(representative[6]) <= 0:
        return None

    return representative


def process_one_assay_exact_actfound(
    rows: list[list[str]],
) -> tuple[list[list[str]] | None, dict[str, int]]:
    """
    Reproduce the upstream pregraph assay-stage semantics.

    The input is one complete assay task in the exact upstream task-key
    order. No RDKit graph logic is performed here.
    """
    raw_count = len(rows)

    stats = {
        "raw_rows": raw_count,
        "unique_molecules": 0,
        "duplicate_groups": 0,
        "duplicate_rows": 0,
        "discarded_molecules": 0,
        "deduplicated_rows": 0,
        "passed": 0,
        "raw_ge20_rejected": 0,
        "dedup_ge20_rejected": 0,
    }

    if raw_count < 20:
        stats["raw_ge20_rejected"] = 1
        return None, stats

    deduplicated: list[list[str]] = []

    current_mol = None
    current_rows: list[list[str]] = []

    def flush_group() -> None:
        if not current_rows:
            return

        stats["unique_molecules"] += 1

        if len(current_rows) > 1:
            stats["duplicate_groups"] += 1
            stats["duplicate_rows"] += len(current_rows) - 1

        representative = resolve_molecule_group_exact_actfound(
            current_rows
        )

        if representative is None:
            stats["discarded_molecules"] += 1
            return

        deduplicated.append(representative)

    for row in rows:
        molregno = row[3]

        if current_mol is None:
            current_mol = molregno

        if molregno != current_mol:
            flush_group()
            current_rows = []
            current_mol = molregno

        current_rows.append(row)

    flush_group()

    stats["deduplicated_rows"] = len(deduplicated)

    if len(deduplicated) < 20:
        stats["dedup_ge20_rejected"] = 1
        return None, stats

    for row in deduplicated:
        row[12] = str(len(deduplicated))

    stats["passed"] = 1
    return deduplicated, stats


def process_sorted_projection(
    input_path: Path,
    output_path: Path,
    *,
    progress_every: int = 500_000,
) -> dict[str, object]:
    """
    Stream the sorted unit-eligible projection one assay task at a time.

    Memory usage is proportional to the largest single assay task, rather
    than to the entire 12.4M-row corpus.
    """
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    input_rows = 0
    assay_tasks = 0
    raw_ge20_assays = 0
    raw_ge20_rows = 0
    dedup_ge20_assays = 0
    dedup_ge20_rows = 0
    duplicate_groups = 0
    duplicate_rows = 0
    discarded_molecules = 0
    raw_ge20_rejected_assays = 0
    dedup_ge20_rejected_assays = 0

    current_key = None
    current_rows: list[list[str]] = []

    with output_path.open("w", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(AUDIT_COLUMNS)

        def flush_assay() -> None:
            nonlocal assay_tasks
            nonlocal raw_ge20_assays
            nonlocal raw_ge20_rows
            nonlocal dedup_ge20_assays
            nonlocal dedup_ge20_rows
            nonlocal duplicate_groups
            nonlocal duplicate_rows
            nonlocal discarded_molecules
            nonlocal raw_ge20_rejected_assays
            nonlocal dedup_ge20_rejected_assays

            if not current_rows:
                return

            assay_tasks += 1

            retained, stats = process_one_assay_exact_actfound(
                current_rows
            )

            if stats["raw_ge20_rejected"]:
                raw_ge20_rejected_assays += 1

            if stats["dedup_ge20_rejected"]:
                dedup_ge20_rejected_assays += 1

            if stats["raw_rows"] >= 20:
                raw_ge20_assays += 1
                raw_ge20_rows += stats["raw_rows"]

            duplicate_groups += stats["duplicate_groups"]
            duplicate_rows += stats["duplicate_rows"]
            discarded_molecules += stats["discarded_molecules"]

            if retained is None:
                return

            dedup_ge20_assays += 1
            dedup_ge20_rows += len(retained)

            for row in retained:
                writer.writerow(row)

        for row in iter_sorted_projection(input_path):
            input_rows += 1

            key = actfound_assay_key(row[:13])

            if current_key is None:
                current_key = key

            if key != current_key:
                flush_assay()
                current_rows = []
                current_key = key

            current_rows.append(row)

            if input_rows % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"stream_progress_rows={input_rows} "
                    f"elapsed_seconds={elapsed:.3f} "
                    f"rows_per_second="
                    f"{input_rows / elapsed if elapsed > 0 else 0.0:.1f} "
                    f"assays_completed={assay_tasks}",
                    flush=True,
                )

        flush_assay()

    elapsed = time.perf_counter() - started

    output_sha256 = file_sha256(output_path)

    summary = {
        "artifact_type":
            "ActFound_ChEMBL32_streamed_pregraph_population_v1",
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "input_rows": input_rows,
        "assay_tasks": assay_tasks,
        "raw_ge20_assays": raw_ge20_assays,
        "raw_ge20_rows": raw_ge20_rows,
        "raw_ge20_rejected_assays":
            raw_ge20_rejected_assays,
        "dedup_ge20_assays": dedup_ge20_assays,
        "dedup_ge20_rows": dedup_ge20_rows,
        "dedup_ge20_rejected_assays":
            dedup_ge20_rejected_assays,
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": duplicate_rows,
        "discarded_molecules": discarded_molecules,
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "elapsed_seconds": elapsed,
        "rows_per_second":
            input_rows / elapsed if elapsed > 0 else None,
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

    return {
        **summary,
        "summary_path": str(summary_path),
        "sha256_manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
    )
    args = parser.parse_args()

    result = process_sorted_projection(
        args.input,
        args.output,
        progress_every=args.progress_every,
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
