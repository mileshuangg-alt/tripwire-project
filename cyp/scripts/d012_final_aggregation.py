from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
D007_MANIFESTS = ROOT / "cyp/artifacts/cyp_007/split_manifests"
T121_RESULT = ROOT / "artifacts/d009_closure_workspace/p1_d009_o150_result_20261108.json"
OUT = ROOT / "qualification/cyp_external_task_qualification_20261109/d012_final_aggregation"
TASKS = (
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
)
SEEDS = (20261101, 20261102, 20261103)


@dataclass(frozen=True, order=True)
class CellKey:
    repetition: int
    fold: int
    training_seed: int

    @property
    def manifest_id(self) -> str:
        return f"rep_{self.repetition:02d}_fold_{self.fold:02d}"

    @property
    def key(self) -> str:
        return f"{self.manifest_id}|seed_{self.training_seed}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_cells() -> list[CellKey]:
    cells = []
    for repetition in range(1, 6):
        for fold in range(1, 11):
            manifest = D007_MANIFESTS / f"rep_{repetition:02d}_fold_{fold:02d}.json"
            if not manifest.exists():
                raise FileNotFoundError(manifest)
            for seed in SEEDS:
                cells.append(CellKey(repetition, fold, seed))
    if len(cells) != 150 or len(set(cells)) != 150:
        raise RuntimeError("frozen D007 cell matrix is not exactly 150 unique cells")
    return cells


def verify_cell_hashes(cell_dir: Path, required_names: set[str]) -> None:
    sums = cell_dir / "SHA256SUMS.txt"
    if not sums.exists():
        raise FileNotFoundError(f"missing SHA256SUMS.txt in {cell_dir}")
    observed_names = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        observed_names.add(name)
        if name not in required_names:
            continue
        path = cell_dir / name
        if not path.exists():
            raise FileNotFoundError(f"hash-listed file is missing: {path}")
        observed = sha256_file(path)
        if observed != digest:
            raise RuntimeError(f"hash mismatch for {path}: {observed} != {digest}")
    missing = sorted(required_names - observed_names)
    if missing:
        raise FileNotFoundError(f"SHA256SUMS is missing required lightweight files in {cell_dir}: {missing}")


def key_from_startup(cell_dir: Path) -> CellKey:
    startup = read_json(cell_dir / "startup_config.json")
    manifest_id = startup["manifest_id"]
    _rep, rep_value, _fold, fold_value = manifest_id.split("_")
    return CellKey(int(rep_value), int(fold_value), int(startup["training_seed"]))


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite metric {label}: {value!r}")
    return result


def verify_macro(per_endpoint: dict[str, Any], macro: float, *, label: str) -> None:
    if set(per_endpoint) != set(TASKS):
        raise ValueError(f"{label} per-endpoint keys do not match frozen task order")
    values = [finite(per_endpoint[task], f"{label}:{task}") for task in TASKS]
    recomputed = sum(values) / len(values)
    if abs(recomputed - macro) > 1e-12:
        raise ValueError(f"{label} macro mismatch: {recomputed} != {macro}")


def parse_b_metric(cell_dir: Path) -> dict[str, Any]:
    verify_cell_hashes(cell_dir, {"startup_config.json", "completion.json", "arm_b_target_metrics.json"})
    completion = read_json(cell_dir / "completion.json")
    if completion.get("status") != "completed":
        raise ValueError(f"B cell is not complete: {cell_dir}")
    metrics = read_json(cell_dir / "arm_b_target_metrics.json")
    macro = finite(metrics["best_macro_st_rae"], "B best_macro_st_rae")
    per_endpoint = metrics.get("best_per_endpoint_st_rae")
    if per_endpoint is not None:
        verify_macro(per_endpoint, macro, label="B")
    metric_path = (cell_dir / "arm_b_target_metrics.json").resolve()
    return {"macro_st_rae": macro, "per_endpoint_st_rae": per_endpoint, "metric_path": str(metric_path.relative_to(ROOT))}


def parse_c_metric(cell_dir: Path) -> dict[str, Any]:
    verify_cell_hashes(cell_dir, {"startup_config.json", "completion.json", "arm_c_validation_metrics.json"})
    completion = read_json(cell_dir / "completion.json")
    if completion.get("status") != "completed":
        raise ValueError(f"C cell is not complete: {cell_dir}")
    metrics = read_json(cell_dir / "arm_c_validation_metrics.json")
    score = metrics["score"]
    macro = finite(score["macro_st_rae"], "C macro_st_rae")
    per_endpoint = score.get("per_task")
    if per_endpoint is not None:
        verify_macro(per_endpoint, macro, label="C")
    metric_path = (cell_dir / "arm_c_validation_metrics.json").resolve()
    return {"macro_st_rae": macro, "per_endpoint_st_rae": per_endpoint, "metric_path": str(metric_path.relative_to(ROOT))}


def collect_arm_metrics(root: Path, arm: str) -> dict[CellKey, dict[str, Any]]:
    pattern = "arm_b_target_metrics.json" if arm == "B" else "arm_c_validation_metrics.json"
    parser = parse_b_metric if arm == "B" else parse_c_metric
    found: dict[CellKey, list[Path]] = {}
    for metric_path in root.rglob(pattern):
        cell_dir = metric_path.parent
        key = key_from_startup(cell_dir)
        found.setdefault(key, []).append(cell_dir)
    duplicates = {key.key: [str(path) for path in paths] for key, paths in found.items() if len(paths) != 1}
    if duplicates:
        raise ValueError(f"duplicate {arm} cell metric artifacts: {duplicates}")
    return {key: parser(paths[0]) for key, paths in found.items()}


def load_t121(path: Path = T121_RESULT) -> dict[CellKey, dict[str, Any]]:
    payload = read_json(path)
    if payload.get("cell_count") != 150:
        raise ValueError("T121 evidence does not declare 150 cells")
    result = {}
    for row in payload["cells"]:
        cell = row["cell"]
        key = CellKey(int(cell["repetition"]), int(cell["fold"]), int(cell["training_seed"]))
        macro = finite(row["macro_st_rae"], f"T121:{key.key}")
        per_endpoint = row.get("per_endpoint_st_rae")
        if per_endpoint is not None:
            verify_macro(per_endpoint, macro, label=f"T121:{key.key}")
        if key in result:
            raise ValueError(f"duplicate T121 cell key: {key.key}")
        result[key] = {"macro_st_rae": macro, "per_endpoint_st_rae": per_endpoint}
    return result


def paired_stats(values: list[float]) -> dict[str, Any]:
    if len(values) != 150:
        raise ValueError("paired characterization requires exactly 150 values")
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "sample_sd": statistics.stdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "first_method_wins_lower_is_better": sum(value < 0 for value in values),
        "second_method_wins_lower_is_better": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
    }


def o150(metrics: dict[CellKey, dict[str, Any]]) -> float:
    if len(metrics) != 150:
        raise ValueError("O150 requires exactly 150 cells")
    return sum(row["macro_st_rae"] for row in metrics.values()) / 150


def join_and_summarize(b: dict[CellKey, dict[str, Any]], c: dict[CellKey, dict[str, Any]], t121: dict[CellKey, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = set(expected_cells())
    if set(b) != expected:
        raise ValueError(f"B cells differ from expected frozen matrix: missing={len(expected - set(b))} extra={len(set(b) - expected)}")
    if set(c) != expected:
        raise ValueError(f"C cells differ from expected frozen matrix: missing={len(expected - set(c))} extra={len(set(c) - expected)}")
    if set(t121) != expected:
        raise ValueError(f"T121 cells differ from expected frozen matrix: missing={len(expected - set(t121))} extra={len(set(t121) - expected)}")
    rows = []
    for key in sorted(expected):
        b_macro = b[key]["macro_st_rae"]
        c_macro = c[key]["macro_st_rae"]
        t_macro = t121[key]["macro_st_rae"]
        rows.append({
            "cell_key": key.key,
            "repetition": key.repetition,
            "fold": key.fold,
            "manifest_id": key.manifest_id,
            "training_seed": key.training_seed,
            "b_macro_st_rae": b_macro,
            "c_macro_st_rae": c_macro,
            "t121_macro_st_rae": t_macro,
            "delta_c_minus_b": c_macro - b_macro,
            "delta_b_minus_t121": b_macro - t_macro,
            "delta_c_minus_t121": c_macro - t_macro,
            "b_metric_path": b[key]["metric_path"],
            "c_metric_path": c[key]["metric_path"],
        })
    summary = {
        "artifact_type": "D012_final_development_aggregation_summary_v1",
        "cell_count": 150,
        "lower_st_rae_is_better": True,
        "difference_sign_convention": {
            "primary": "delta_c_minus_b = S_c(C) - S_c(B); negative means Arm C is better/lower for that cell",
            "secondary_b_vs_t121": "delta_b_minus_t121 = S_c(B) - S_c(T121); negative means Arm B is better/lower",
            "secondary_c_vs_t121": "delta_c_minus_t121 = S_c(C) - S_c(T121); negative means Arm C is better/lower",
        },
        "o150": {"B": o150(b), "C": o150(c), "T121": o150(t121)},
        "paired_differences": {
            "C_minus_B": paired_stats([row["delta_c_minus_b"] for row in rows]),
            "B_minus_T121": paired_stats([row["delta_b_minus_t121"] for row in rows]),
            "C_minus_T121": paired_stats([row["delta_c_minus_t121"] for row in rows]),
        },
        "t121_source": {
            "path": str(T121_RESULT.relative_to(ROOT)),
            "sha256": sha256_file(T121_RESULT),
            "cell_key_schema": "cell.repetition + cell.fold + cell.training_seed",
        },
    }
    return rows, summary


CELL_FIELDS = [
    "cell_key", "repetition", "fold", "manifest_id", "training_seed",
    "b_macro_st_rae", "c_macro_st_rae", "t121_macro_st_rae",
    "delta_c_minus_b", "delta_b_minus_t121", "delta_c_minus_t121",
    "b_metric_path", "c_metric_path",
]


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_path = output_dir / "d012_cell_level_joined_results.tsv"
    with cell_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CELL_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "d012_final_aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "artifact_type": "D012_final_aggregation_provenance_v1",
        "inputs": summary["t121_source"],
        "output_hashes": {
            "d012_cell_level_joined_results.tsv": sha256_file(cell_path),
            "d012_final_aggregate_summary.json": sha256_file(summary_path),
        },
    }
    provenance_path = output_dir / "d012_final_aggregation_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for path in [cell_path, summary_path, provenance_path]:
            handle.write(f"{sha256_file(path)}  {path.name}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate frozen D012 final development metrics.")
    parser.add_argument("--d012-production-root", required=True, type=Path)
    parser.add_argument("--t121-result", default=T121_RESULT, type=Path)
    parser.add_argument("--output-dir", default=OUT, type=Path)
    parser.add_argument("--write-outputs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    global T121_RESULT
    T121_RESULT = args.t121_result
    b = collect_arm_metrics(args.d012_production_root, "B")
    c = collect_arm_metrics(args.d012_production_root, "C")
    t121 = load_t121(args.t121_result)
    rows, summary = join_and_summarize(b, c, t121)
    if args.write_outputs:
        write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
