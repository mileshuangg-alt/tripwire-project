from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from d012_corrective_patch import CORRECTED_AGGREGATION_ROOT, FROZEN_T121_CELLS, assert_not_p1_o150_artifact, load_frozen_t121_cells
from d012_final_aggregation import CellKey, collect_arm_metrics, join_and_summarize, write_outputs


def load_corrected_t121(cells_dir: Path = FROZEN_T121_CELLS) -> dict[CellKey, dict[str, Any]]:
    cells = load_frozen_t121_cells(cells_dir)
    result = {}
    for row in cells.values():
        key = CellKey(int(row["repetition"]), int(row["fold"]), int(row["seed"]))
        result[key] = {
            "macro_st_rae": float(row["macro_st_rae"]),
            "per_endpoint_st_rae": {task: float(value) for task, value in row["best_validation"]["per_task"].items()},
        }
    if len(result) != 150:
        raise ValueError("corrected T121 source must contain exactly 150 cells")
    return result


def assert_t121_source_argument(path: Path) -> None:
    if path.is_file():
        assert_not_p1_o150_artifact(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("artifact_type") != "D012_corrected_t121_cell_source_manifest_v1":
            raise ValueError(f"unsupported corrected T121 source manifest type: {path}")
    elif path.is_dir():
        load_frozen_t121_cells(path)
    else:
        raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate corrected D012 with guarded frozen T121 comparator evidence.")
    parser.add_argument("--d012-production-root", required=True, type=Path)
    parser.add_argument("--t121-cells-dir", default=FROZEN_T121_CELLS, type=Path)
    parser.add_argument("--output-dir", default=CORRECTED_AGGREGATION_ROOT, type=Path)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    assert_t121_source_argument(args.t121_cells_dir)
    b = collect_arm_metrics(args.d012_production_root, "B")
    c = collect_arm_metrics(args.d012_production_root, "C")
    t121 = load_corrected_t121(args.t121_cells_dir)
    rows, summary = join_and_summarize(b, c, t121)
    summary["t121_source"] = {
        "path": str(args.t121_cells_dir),
        "artifact_type": "D007_T121_saved_cell_results_directory",
        "cell_count": 150,
        "guard": "P1 aggregate artifacts rejected by explicit provenance/type assertion",
    }
    if args.write_outputs:
        write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
