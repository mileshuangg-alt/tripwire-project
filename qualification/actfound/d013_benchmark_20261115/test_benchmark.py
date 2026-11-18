"""Tests for the D013 interactive-benchmark helpers (stdlib only; any Python >= 3.9)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_progress as bp  # noqa: E402
import make_benchmark_config as mc  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _v2(tmp: Path) -> Path:
    p = tmp / "config_v2.json"
    p.write_text(json.dumps({"artifact_type": "ActFoundProductionConfigV2", "output_dir": "/prod/out",
                             "hyperparameters": {"meta_batch_size": 16}, "train_assays": ["a", "b"]}))
    return p


def test_benchmark_config_changes_only_output_dir(tmp_path):
    v2 = _v2(tmp_path)
    out = tmp_path / "bench" / "config.json"
    rec = mc.make(v2, _sha(v2), tmp_path / "bench" / "run", out)
    a, b = json.loads(v2.read_text()), json.loads(out.read_text())
    assert rec["keys_changed"] == ["output_dir"] and b["output_dir"] == str(tmp_path / "bench" / "run")
    assert {k: v for k, v in a.items() if k != "output_dir"} == {k: v for k, v in b.items() if k != "output_dir"}
    assert rec["benchmark_config_sha256"] == _sha(out)
    assert not (tmp_path / "bench" / "run").exists()


def test_benchmark_config_refusals(tmp_path):
    v2 = _v2(tmp_path)
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        mc.make(v2, "0" * 64, tmp_path / "run", tmp_path / "c.json")
    with pytest.raises(SystemExit, match="must differ"):
        mc.make(v2, _sha(v2), Path("/prod/out"), tmp_path / "c.json")
    (tmp_path / "exists").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        mc.make(v2, _sha(v2), tmp_path / "exists", tmp_path / "c.json")
    (tmp_path / "c.json").write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        mc.make(v2, _sha(v2), tmp_path / "run", tmp_path / "c.json")


class FakeRun:
    """A runner that writes events as the fake clock advances; int(dt // 2) batches per sleep."""

    def __init__(self, progress: Path):
        self.progress, self.t, self.batches = progress, 1000.0, 0

    def clock(self):
        return self.t

    def write(self, row):
        with self.progress.open("a") as f:
            f.write(json.dumps(row) + "\n")

    pending = False

    def sleep(self, dt):
        self.t += dt
        if self.pending:          # finish the half-written line: it now counts as one batch
            with self.progress.open("a") as f:
                f.write('1}\n')
            self.batches += 1
            self.pending = False
        if self.t == 1030:
            self.write({"event": "dataset_loaded", "train_indices": 3210, "val_indices": 9})
        if self.t == 1060:
            self.write({"event": "validation", "epoch": 0, "phase": "initial"})
        if self.t > 1060:
            for _ in range(int(dt // 2)):
                self.batches += 1
                self.write({"event": "train_batch", "epoch": 0, "batch": self.batches - 1, "loss": 0.1})
            # leave a half-written train_batch line at poll time; it is completed on the next sleep
            with self.progress.open("a") as f:
                f.write('{"event": "train_batch", "epoch": 0, "batch": -1, "loss": 0.')
            self.pending = True


def _sampled(tmp: Path, seconds: float):
    prog = tmp / "progress.jsonl"
    prog.write_text("")
    run = FakeRun(prog)
    (tmp / "startup.json").write_text(json.dumps({"started_at_unix": 1000.0}))
    n = bp.sample(prog, tmp / "samples.tsv", 15, max_seconds=seconds, clock=run.clock, sleep=run.sleep, gpu=lambda: "4096")
    return n, prog, run


def test_sampler_counts_complete_lines_and_stops(tmp_path):
    n, prog, run = _sampled(tmp_path, 600)
    with (tmp_path / "samples.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert n == len(rows) == 41
    assert int(rows[-1]["train_batch"]) == run.batches
    assert rows[-1]["dataset_loaded"] == "1" and rows[-1]["validation"] == "1"
    assert rows[-1]["gpu_memory_used_mib"] == "4096"


def test_summary_rate_bpe_and_extrapolation(tmp_path):
    _sampled(tmp_path, 900)
    r = bp.summarize(tmp_path / "samples.tsv", tmp_path / "startup.json", tmp_path / "progress.jsonl",
                     epochs=60, meta_batch_size=16, ref_elapsed_s=44168, ref_batches_per_epoch=2410,
                     requested_wall_s=259200)
    m, e = r["measured"], r["extrapolation"]
    assert m["batches_per_epoch"] == 3210 // 16
    assert m["epoch0_batches_per_second"] == pytest.approx(8 / 15)   # fake runner: 7 + 1 completed partial per 15 s
    assert m["load_seconds"] == pytest.approx(30) and m["initial_validation_seconds"] == pytest.approx(30)
    assert m["epoch0_observed_seconds"] is None and e["per_epoch_basis"].startswith("rate")
    per_epoch = (3210 // 16) / m["epoch0_batches_per_second"] + 30
    assert e["first_order_lower_bound_seconds"] == pytest.approx(60 + 60 * per_epoch)
    assert e["reference_scaled_seconds"] == pytest.approx(44168 * (3210 // 16) / 2410)
    assert e["margin_vs_reference_scaled"] == pytest.approx(259200 / e["reference_scaled_seconds"])
    assert m["max_gpu_memory_used_mib"] == 4096


def test_summary_refuses_short_rate_window(tmp_path):
    _sampled(tmp_path, 200)
    with pytest.raises(SystemExit, match="rate window shorter"):
        bp.summarize(tmp_path / "samples.tsv", tmp_path / "startup.json", tmp_path / "progress.jsonl",
                     epochs=60, meta_batch_size=16, ref_elapsed_s=1, ref_batches_per_epoch=1, requested_wall_s=1)


def test_summary_uses_observed_epoch_when_epoch0_completes(tmp_path):
    _sampled(tmp_path, 900)
    rows = list(csv.DictReader((tmp_path / "samples.tsv").open(), delimiter="\t"))
    rows[-1]["epoch_complete"] = "1"           # epoch 0 ends at the last sample
    with (tmp_path / "samples.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bp.COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    r = bp.summarize(tmp_path / "samples.tsv", tmp_path / "startup.json", tmp_path / "progress.jsonl",
                     epochs=60, meta_batch_size=16, ref_elapsed_s=44168, ref_batches_per_epoch=2410,
                     requested_wall_s=259200)
    obs = float(rows[-1]["unix_time"]) - 1060
    assert r["measured"]["epoch0_observed_seconds"] == pytest.approx(obs)
    assert r["extrapolation"]["per_epoch_basis"] == "observed epoch 0"
    assert r["extrapolation"]["first_order_lower_bound_seconds"] == pytest.approx(60 + 60 * obs)


def test_sampler_refuses_existing_output(tmp_path):
    (tmp_path / "samples.tsv").write_text("x")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        bp.sample(tmp_path / "p.jsonl", tmp_path / "samples.tsv", 1, max_seconds=1)


def test_rate_excludes_samples_after_epoch0_ends(tmp_path):
    """After epoch 0 the count stalls (end-of-epoch validation); those samples must not dilute the rate."""
    _sampled(tmp_path, 900)
    path = tmp_path / "samples.tsv"
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    stall = rows[-1]["train_batch"]
    for i in range(1, 21):                     # 20 more samples, epoch 0 complete, no new batches
        rows.append(dict(rows[-1], unix_time=str(float(rows[-1]["unix_time"]) + 15), epoch_complete="1",
                         train_batch=stall))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bp.COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    r = bp.summarize(path, tmp_path / "startup.json", tmp_path / "progress.jsonl", epochs=60, meta_batch_size=16,
                     ref_elapsed_s=1, ref_batches_per_epoch=1, requested_wall_s=1)
    assert r["measured"]["epoch0_batches_per_second"] == pytest.approx(8 / 15)
