from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Iterator

from actfound_chembl32_projection import (
    AUDIT_COLUMNS,
    construct_graph_exact_actfound,
    file_sha256,
    final_actfound_row,
    postgraph_assay_filter_exact_actfound,
)


def iter_assays(path: Path) -> Iterator[tuple[str, list[list[str]]]]:
    """
    Read the frozen pregraph TSV, which is already ordered by exact
    ActFound assay key. Yield one complete assay task at a time.
    """
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)

        if header != list(AUDIT_COLUMNS):
            raise ValueError("unexpected pregraph header")

        current_key = None
        current_rows: list[list[str]] = []
        previous_order = None

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(AUDIT_COLUMNS):
                raise ValueError(
                    f"line {line_number}: expected "
                    f"{len(AUDIT_COLUMNS)} fields, got {len(row)}"
                )

            # Must exactly match the external GNU sort key used to create
            # the frozen pregraph input:
            #   assay_id numeric, units lexical, standard_type lexical,
            #   molregno numeric, activity_id numeric.
            order_key = (
                int(row[1]),
                row[7],
                row[8],
                int(row[3]),
                int(row[0]),
            )

            assay_key = "_".join(
                (row[1], row[7], row[8])
            )

            if previous_order is not None and order_key < previous_order:
                raise ValueError(
                    "pregraph TSV is not ordered by the frozen external-sort key"
                )

            previous_order = order_key

            if current_key is None:
                current_key = assay_key

            if assay_key != current_key:
                yield current_key, current_rows
                current_rows = []
                current_key = assay_key

            current_rows.append(row)

        if current_key is not None:
            yield current_key, current_rows


def shard_pregraph(
    input_path: Path,
    output_dir: Path,
    n_shards: int,
) -> dict[str, object]:
    """
    Deterministically assign complete assay tasks to shards using a greedy
    minimum-row-load rule.

    Assays are never split across shards.
    """
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)

    handles = []
    writers = []
    row_counts = [0] * n_shards
    assay_counts = [0] * n_shards
    cost_estimates = [0] * n_shards

    try:
        for i in range(n_shards):
            path = output_dir / f"shard_{i:03d}.tsv"
            handle = path.open("w", newline="")
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(AUDIT_COLUMNS)
            handles.append(handle)
            writers.append(writer)

        total_rows = 0
        total_assays = 0

        for assay_key, rows in iter_assays(input_path):
            n = len(rows)

            # Exact upstream graph behavior:
            # assays with N >= 10000 are rejected before any fingerprint
            # or pairwise graph construction. They therefore contribute
            # no graph pair work.
            if n >= 10000:
                pair_work = 0
            else:
                pair_work = n * (n - 1) // 2

            # Graph cost scales approximately with the number of
            # molecular pairs for assays that actually reach graph
            # construction.
            shard = min(
                range(n_shards),
                key=lambda i: (cost_estimates[i], row_counts[i], i),
            )

            for row in rows:
                writers[shard].writerow(row)

            row_counts[shard] += n
            assay_counts[shard] += 1
            cost_estimates[shard] += pair_work

            total_rows += n
            total_assays += 1

    finally:
        for handle in handles:
            handle.close()

    shards = []

    for i in range(n_shards):
        path = output_dir / f"shard_{i:03d}.tsv"
        shards.append(
            {
                "shard_id": i,
                "path": str(path),
                "rows": row_counts[i],
                "assays": assay_counts[i],
                "pair_work_estimate": cost_estimates[i],
                "sha256": file_sha256(path),
            }
        )

    result = {
        "artifact_type": "ActFound_graph_stage_shards_v1",
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "input_rows": total_rows,
        "input_assays": total_assays,
        "n_shards": n_shards,
        "shards": shards,
    }
    return result


def process_shard(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    progress_every: int = 1000,
) -> dict[str, object]:
    """
    Run the exact ActFound RDKit graph construction and assay gate on one
    complete shard, one assay at a time.
    """
    import sys

    processed_assays = 0
    processed_rows = 0
    graph_size_rejected = 0
    graph_valid_count_rejected = 0
    graph_qualified = 0
    final_rows = 0
    pair_count_total = 0

    started = time.perf_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as out:
        writer = csv.writer(out, delimiter="\t")

        writer.writerow(
            [
                "assay_key",
                *[
                    x
                    for x in (
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
                        "count_scratch",
                        "canonical_smiles",
                        "graph_pair_count",
                    )
                ],
            ]
        )

        with input_path.open(newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)

            if header != list(AUDIT_COLUMNS):
                raise ValueError("unexpected shard header")

            current_key = None
            current_rows: list[list[str]] = []

            def flush_assay() -> None:
                nonlocal processed_assays
                nonlocal processed_rows
                nonlocal graph_size_rejected
                nonlocal graph_valid_count_rejected
                nonlocal graph_qualified
                nonlocal final_rows
                nonlocal pair_count_total

                if not current_rows:
                    return

                processed_assays += 1
                processed_rows += len(current_rows)

                records = [row[:13] for row in current_rows]

                smiles = {
                    row[3]: row[13]
                    for row in current_rows
                }

                degree, pair_count = construct_graph_exact_actfound(
                    records,
                    smiles,
                )

                if degree is None:
                    graph_size_rejected += 1
                    return

                retained = postgraph_assay_filter_exact_actfound(
                    records,
                    degree,
                    pair_count,
                )

                if retained is None:
                    graph_valid_count_rejected += 1
                    return

                graph_qualified += 1
                pair_count_total += int(pair_count)

                for record in retained:
                    molregno = record[3]

                    writer.writerow(
                        [
                            current_key,
                            *final_actfound_row(
                                record,
                                canonical_smiles=smiles[molregno],
                                pair_count=int(pair_count),
                            ),
                        ]
                    )

                final_rows += len(retained)

                if processed_assays % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"graph_assays={processed_assays} "
                        f"graph_rows={processed_rows} "
                        f"elapsed_seconds={elapsed:.3f}",
                        file=sys.stderr,
                        flush=True,
                    )

            for row in reader:
                key = "_".join((row[1], row[7], row[8]))

                if current_key is None:
                    current_key = key

                if key != current_key:
                    flush_assay()
                    current_rows = []
                    current_key = key

                current_rows.append(row)

            flush_assay()

    elapsed = time.perf_counter() - started

    summary = {
        "artifact_type": "ActFound_graph_stage_shard_result_v1",
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "processed_assays": processed_assays,
        "processed_rows": processed_rows,
        "graph_size_rejected_assays": graph_size_rejected,
        "graph_valid_count_rejected_assays":
            graph_valid_count_rejected,
        "graph_qualified_assays": graph_qualified,
        "final_rows": final_rows,
        "pair_count_total": pair_count_total,
        "elapsed_seconds": elapsed,
        "runtime": {
            "python": platform.python_version(),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    with summary_path.with_suffix(
        summary_path.suffix + ".SHA256SUMS.txt"
    ).open("w") as f:
        f.write(
            f"{file_sha256(output_path)}  {output_path.name}\n"
        )
        f.write(
            f"{file_sha256(summary_path)}  {summary_path.name}\n"
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="mode", required=True)

    shard = sub.add_parser("shard")
    shard.add_argument("--input", required=True, type=Path)
    shard.add_argument("--output-dir", required=True, type=Path)
    shard.add_argument("--n-shards", required=True, type=int)

    process = sub.add_parser("process-shard")
    process.add_argument("--input", required=True, type=Path)
    process.add_argument("--output", required=True, type=Path)
    process.add_argument("--summary", required=True, type=Path)
    process.add_argument("--progress-every", type=int, default=1000)

    args = parser.parse_args()

    if args.mode == "shard":
        result = shard_pregraph(
            args.input,
            args.output_dir,
            args.n_shards,
        )
    else:
        result = process_shard(
            args.input,
            args.output,
            args.summary,
            args.progress_every,
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
