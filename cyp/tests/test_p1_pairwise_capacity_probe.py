from __future__ import annotations

import json

import pytest
import torch

from p1_pairwise_capacity_probe import (
    ARTIFACT_TYPE,
    build_parser,
    run_synthetic_capacity_probe,
    write_json_artifact,
)


class FakeCudaBackend:
    def __init__(self):
        self.reset_calls = []
        self.synchronize_calls = []
        self.empty_cache_calls = 0

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def set_device(self, device):
        self.device = device

    def get_device_name(self, device_index):
        return "Synthetic NVIDIA L40S"

    def reset_peak_memory_stats(self, device):
        self.reset_calls.append(device)

    def synchronize(self, device):
        self.synchronize_calls.append(device)

    def max_memory_allocated(self, device):
        return 123456

    def max_memory_reserved(self, device):
        return 234567

    def empty_cache(self):
        self.empty_cache_calls += 1


def _run_probe(**overrides):
    kwargs = {
        "query_count": 2,
        "reference_count": 3,
        "dtype_name": "float32",
        "rng_seed": 20261107,
        "chunk_size_candidates": [5, 2],
        "warmup_iterations": 1,
        "cuda_backend": FakeCudaBackend(),
        "require_cuda": False,
        "device": torch.device("cpu"),
    }
    kwargs.update(overrides)
    return run_synthetic_capacity_probe(**kwargs)


def test_cli_requires_explicit_inputs_and_validates_positive_chunks():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--query-count",
                "2",
                "--reference-count",
                "3",
                "--dtype",
                "float32",
                "--rng-seed",
                "7",
                "--chunk-size-candidates",
                "0",
                "--output-json",
                "probe.json",
            ]
        )

    args = parser.parse_args(
        [
            "--query-count",
            "2",
            "--reference-count",
            "3",
            "--dtype",
            "float16",
            "--rng-seed",
            "7",
            "--chunk-size-candidates",
            "9",
            "4",
            "9",
            "--output-json",
            "probe.json",
        ]
    )

    assert args.query_count == 2
    assert args.reference_count == 3
    assert args.dtype == "float16"
    assert args.rng_seed == 7
    assert args.chunk_size_candidates == [9, 4, 9]


def test_supplied_chunk_order_is_preserved_exactly():
    chunks = [8, 3, 8, 1]
    artifact = _run_probe(chunk_size_candidates=chunks)

    assert artifact["inputs"]["chunk_size_candidates"] == chunks
    assert [
        result["chunk_size"]
        for result in artifact["candidate_results"]
    ] == chunks


def test_accounting_and_json_schema_content(tmp_path):
    artifact = _run_probe(
        query_count=4,
        reference_count=5,
        chunk_size_candidates=[7],
    )

    assert artifact["artifact_type"] == ARTIFACT_TYPE
    assert "Synthetic operational capacity evidence only" in artifact[
        "synthetic_operational_statement"
    ]
    assert artifact["inputs"]["query_count"] == 4
    assert artifact["inputs"]["reference_count"] == 5
    assert artifact["inputs"]["unique_query_reference_pair_count"] == 20
    assert artifact["inputs"]["directional_g_rows"] == 40
    assert artifact["environment"]["gpu_model"] == "Synthetic NVIDIA L40S"
    assert artifact["environment"]["visible_gpu_count"] == 1
    assert artifact["environment"]["device_index"] == 0
    assert artifact["environment"]["p1_head_parameter_count"] > 0

    candidate = artifact["candidate_results"][0]
    assert candidate["success"] is True
    assert candidate["oom"] is False
    assert candidate["unique_query_reference_pair_count"] == 20
    assert candidate["directional_g_rows"] == 40
    assert candidate["peak_allocated_gpu_memory_bytes"] == 123456
    assert candidate["peak_reserved_gpu_memory_bytes"] == 234567

    output_json = tmp_path / "probe.json"
    write_json_artifact(artifact, output_json)
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["artifact_type"] == ARTIFACT_TYPE
    assert loaded["candidate_results"][0]["chunk_size"] == 7


def test_oom_is_recorded_for_candidate_without_automatic_fallback():
    fake_cuda = FakeCudaBackend()
    calls = []

    def traversal(pair_head, query_embeddings, reference_embeddings, chunk_size):
        calls.append(chunk_size)
        if chunk_size == 11 and calls.count(11) == 2:
            raise RuntimeError("CUDA out of memory in synthetic test")
        return None

    artifact = _run_probe(
        chunk_size_candidates=[11, 5],
        cuda_backend=fake_cuda,
        traversal=traversal,
    )

    assert calls == [11, 11, 5, 5]
    assert fake_cuda.empty_cache_calls == 1

    first, second = artifact["candidate_results"]
    assert first["chunk_size"] == 11
    assert first["success"] is False
    assert first["oom"] is True
    assert first["elapsed_seconds"] is None
    assert first["pairs_per_second"] is None

    assert second["chunk_size"] == 5
    assert second["success"] is True
    assert second["oom"] is False


def test_non_oom_exception_fails_closed():
    def traversal(pair_head, query_embeddings, reference_embeddings, chunk_size):
        raise RuntimeError("synthetic non-OOM failure")

    with pytest.raises(RuntimeError, match="synthetic non-OOM failure"):
        _run_probe(traversal=traversal)


def test_cuda_and_one_visible_gpu_are_required_for_actual_probe():
    class NoCuda(FakeCudaBackend):
        def is_available(self):
            return False

    with pytest.raises(RuntimeError, match="CUDA is required"):
        run_synthetic_capacity_probe(
            query_count=1,
            reference_count=1,
            dtype_name="float32",
            rng_seed=1,
            chunk_size_candidates=[1],
            cuda_backend=NoCuda(),
            require_cuda=True,
        )

    class TwoGpus(FakeCudaBackend):
        def device_count(self):
            return 2

    with pytest.raises(RuntimeError, match="Exactly one visible GPU"):
        run_synthetic_capacity_probe(
            query_count=1,
            reference_count=1,
            dtype_name="float32",
            rng_seed=1,
            chunk_size_candidates=[1],
            cuda_backend=TwoGpus(),
            require_cuda=True,
        )


def test_no_production_or_scientific_input_surface():
    parser = build_parser()
    option_dests = {
        action.dest
        for action in parser._actions
    }

    forbidden = {
        "smiles",
        "molecules",
        "labels",
        "validation_labels",
        "checkpoint",
        "cyp_data",
        "st_rae",
        "metric",
        "training_pairs",
        "production_pairs",
    }

    assert option_dests.isdisjoint(forbidden)
    assert {
        "query_count",
        "reference_count",
        "dtype",
        "rng_seed",
        "chunk_size_candidates",
        "output_json",
    }.issubset(option_dests)
