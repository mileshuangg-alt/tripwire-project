from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any

import torch

from p1_pairwise_benchmark import _all_reference_pair_head
from p1_pairwise_model import P1AntisymmetricPairHead


ARTIFACT_TYPE = "D009_P1_synthetic_gpu_capacity_probe_v1"
SYNTHETIC_STATEMENT = (
    "Synthetic operational capacity evidence only: uses random 600-dimensional "
    "query/reference embeddings and the qualified D009 P1 pair head; contains "
    "no CYP data, molecules, labels, checkpoints, validation data, ST-RAE, "
    "production pairs, or training. Results do not select or recommend a "
    "production chunk size."
)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        )

    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic D009 P1 GPU capacity probe. This is operational "
            "plumbing only and does not access scientific CYP inputs."
        )
    )
    parser.add_argument(
        "--query-count",
        required=True,
        type=_positive_int,
    )
    parser.add_argument(
        "--reference-count",
        required=True,
        type=_positive_int,
    )
    parser.add_argument(
        "--dtype",
        required=True,
        choices=tuple(DTYPES),
    )
    parser.add_argument(
        "--rng-seed",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--chunk-size-candidates",
        required=True,
        nargs="+",
        type=_positive_int,
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--warmup-iterations",
        default=1,
        type=_positive_int,
    )
    return parser


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _parameter_count(model: P1AntisymmetricPairHead) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _environment_record(
    *,
    cuda_backend: Any,
    device_index: int,
    gpu_model: str,
    p1_head_parameter_count: int,
    visible_gpu_count: int,
) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": bool(cuda_backend.is_available()),
        "gpu_model": gpu_model,
        "visible_gpu_count": visible_gpu_count,
        "device_index": device_index,
        "p1_head_parameter_count": p1_head_parameter_count,
    }


def _peak_allocated(cuda_backend: Any, device: torch.device) -> int | None:
    try:
        return int(cuda_backend.max_memory_allocated(device))
    except (AttributeError, RuntimeError):
        return None


def _peak_reserved(cuda_backend: Any, device: torch.device) -> int | None:
    try:
        return int(cuda_backend.max_memory_reserved(device))
    except (AttributeError, RuntimeError):
        return None


def _clear_oom_state(cuda_backend: Any) -> None:
    try:
        cuda_backend.empty_cache()
    except AttributeError:
        return


def run_synthetic_capacity_probe(
    *,
    query_count: int,
    reference_count: int,
    dtype_name: str,
    rng_seed: int,
    chunk_size_candidates: list[int],
    warmup_iterations: int = 1,
    cuda_backend: Any = torch.cuda,
    traversal=_all_reference_pair_head,
    require_cuda: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if query_count <= 0 or reference_count <= 0:
        raise ValueError("query_count and reference_count must be positive.")

    if dtype_name not in DTYPES:
        raise ValueError("dtype_name must be one of the supported dtype names.")

    if not chunk_size_candidates:
        raise ValueError("at least one chunk-size candidate is required.")

    if any(chunk_size <= 0 for chunk_size in chunk_size_candidates):
        raise ValueError("chunk-size candidates must all be positive.")

    if warmup_iterations <= 0:
        raise ValueError("warmup_iterations must be positive.")

    visible_gpu_count = int(cuda_backend.device_count())
    if require_cuda:
        if not cuda_backend.is_available():
            raise RuntimeError("CUDA is required for the synthetic capacity probe.")

        if visible_gpu_count != 1:
            raise RuntimeError(
                "Exactly one visible GPU is required for the synthetic capacity probe."
            )

    if device is None:
        device = torch.device("cuda:0" if require_cuda else "cpu")

    device_index = 0 if device.index is None else int(device.index)
    if require_cuda:
        cuda_backend.set_device(device)
        gpu_model = str(cuda_backend.get_device_name(device_index))
    else:
        gpu_model = str(
            cuda_backend.get_device_name(device_index)
            if visible_gpu_count > 0
            else "synthetic-test-device"
        )

    dtype = DTYPES[dtype_name]
    pair_head = P1AntisymmetricPairHead().to(device=device, dtype=dtype)
    pair_head.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(rng_seed)

    with torch.no_grad():
        query_embeddings = torch.randn(
            (query_count, 600),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        reference_embeddings = torch.randn(
            (reference_count, 600),
            device=device,
            dtype=dtype,
            generator=generator,
        )

    unique_pairs = query_count * reference_count
    directional_g_rows = 2 * unique_pairs

    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "synthetic_operational_statement": SYNTHETIC_STATEMENT,
        "environment": _environment_record(
            cuda_backend=cuda_backend,
            device_index=device_index,
            gpu_model=gpu_model,
            p1_head_parameter_count=_parameter_count(pair_head),
            visible_gpu_count=visible_gpu_count,
        ),
        "inputs": {
            "query_count": query_count,
            "reference_count": reference_count,
            "unique_query_reference_pair_count": unique_pairs,
            "directional_g_rows": directional_g_rows,
            "dtype": dtype_name,
            "rng_seed": rng_seed,
            "chunk_size_candidates": list(chunk_size_candidates),
            "warmup_iterations": warmup_iterations,
        },
        "candidate_results": [],
    }

    with torch.no_grad():
        for chunk_size in chunk_size_candidates:
            result = {
                "chunk_size": chunk_size,
                "success": False,
                "oom": False,
                "elapsed_seconds": None,
                "unique_query_reference_pair_count": unique_pairs,
                "directional_g_rows": directional_g_rows,
                "pairs_per_second": None,
                "peak_allocated_gpu_memory_bytes": None,
                "peak_reserved_gpu_memory_bytes": None,
                "error_type": None,
                "error_message": None,
            }

            try:
                for _ in range(warmup_iterations):
                    traversal(
                        pair_head,
                        query_embeddings,
                        reference_embeddings,
                        chunk_size,
                    )

                cuda_backend.reset_peak_memory_stats(device)
                cuda_backend.synchronize(device)
                start = time.perf_counter()
                traversal(
                    pair_head,
                    query_embeddings,
                    reference_embeddings,
                    chunk_size,
                )
                cuda_backend.synchronize(device)
                elapsed = time.perf_counter() - start

                if elapsed < 0.0:
                    raise FloatingPointError("elapsed time must be nonnegative.")

                result["success"] = True
                result["elapsed_seconds"] = float(elapsed)
                result["pairs_per_second"] = float(unique_pairs / elapsed) if elapsed > 0 else None
                result["peak_allocated_gpu_memory_bytes"] = _peak_allocated(
                    cuda_backend,
                    device,
                )
                result["peak_reserved_gpu_memory_bytes"] = _peak_reserved(
                    cuda_backend,
                    device,
                )
            except RuntimeError as exc:
                if not _is_cuda_oom(exc):
                    raise

                result["oom"] = True
                result["error_type"] = type(exc).__name__
                result["error_message"] = str(exc)
                result["peak_allocated_gpu_memory_bytes"] = _peak_allocated(
                    cuda_backend,
                    device,
                )
                result["peak_reserved_gpu_memory_bytes"] = _peak_reserved(
                    cuda_backend,
                    device,
                )
                _clear_oom_state(cuda_backend)

            artifact["candidate_results"].append(result)

    return artifact


def write_json_artifact(artifact: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifact = run_synthetic_capacity_probe(
        query_count=args.query_count,
        reference_count=args.reference_count,
        dtype_name=args.dtype,
        rng_seed=args.rng_seed,
        chunk_size_candidates=args.chunk_size_candidates,
        warmup_iterations=args.warmup_iterations,
    )
    write_json_artifact(artifact, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
