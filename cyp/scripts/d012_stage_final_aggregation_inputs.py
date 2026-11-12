from __future__ import annotations

import argparse
import json
from pathlib import Path


LIGHTWEIGHT_PATTERNS = {
    "B": [
        "startup_config.json",
        "completion.json",
        "arm_b_target_metrics.json",
        "SHA256SUMS.txt",
    ],
    "C": [
        "startup_config.json",
        "completion.json",
        "arm_c_validation_metrics.json",
        "SHA256SUMS.txt",
    ],
}


def cell_specs() -> list[dict[str, object]]:
    rows = []
    for repetition in range(1, 6):
        for fold in range(1, 11):
            manifest_id = f"rep_{repetition:02d}_fold_{fold:02d}"
            for seed in (20261101, 20261102, 20261103):
                rows.append({
                    "repetition": repetition,
                    "fold": fold,
                    "manifest_id": manifest_id,
                    "training_seed": seed,
                })
    return rows


def manifest(source_root: str, staged_root: str) -> dict[str, object]:
    files = []
    for arm, names in LIGHTWEIGHT_PATTERNS.items():
        arm_lower = arm.lower()
        for cell in cell_specs():
            cell_dir = f"{cell['manifest_id']}_seed{cell['training_seed']}"
            for name in names:
                files.append({
                    "arm": arm,
                    "cell": cell,
                    "source": f"{source_root}/d012_remaining/arm_{arm_lower}/{cell_dir}/{name}",
                    "destination": f"{staged_root}/arm_{arm_lower}/{cell_dir}/{name}",
                    "checkpoint_payload": False,
                })
    return {
        "artifact_type": "D012_final_aggregation_lightweight_collection_manifest_v1",
        "source_root_parameter": source_root,
        "staged_root_parameter": staged_root,
        "file_count": len(files),
        "cells_per_arm": 150,
        "arms": ["B", "C"],
        "checkpoint_payloads_required": False,
        "files": files,
        "notes": [
            "Copy only lightweight metrics/config/provenance/hash artifacts needed for frozen D012 final aggregation.",
            "Do not copy checkpoint payloads for final aggregation.",
            "The completed first cell may reside outside d012_remaining; include equivalent first-cell lightweight files in staged arm_b/arm_c cell directories before aggregation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write D012 final aggregation lightweight collection manifest.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--staged-root", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    payload = manifest(args.source_root, args.staged_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
