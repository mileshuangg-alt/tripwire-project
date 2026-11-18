"""D013 interactive benchmark: sample runner v2's progress.jsonl and summarize timings.

Runner v2's train_batch events carry no timestamps, so this reads the file from
outside on a fixed interval and records (wall time, event counts). It only
reads; it never writes into the runner's output directory.

  sample     poll progress.jsonl until the watched PID exits or --max-seconds
  summarize  turn the samples into measured timings and an extrapolation

Measured: load time (startup -> dataset_loaded), initial validation, the
epoch-0 train-batch rate, batches per epoch (len(train_indices) // meta_batch_size,
exactly as runner v2's train_batches), and, if epoch 0 finishes, its full
duration including end-of-epoch validation and checkpointing.

Epoch 0 is first-order training. Runner v2 switches to second order at
first_order_to_second_order_epoch and adds the lr loss at begin_lrloss_epoch,
both unreachable in a bounded interactive session, so 60 x epoch-0 time is a
lower bound, reported as such. The reference run (job 3066823: elapsed and
batches per epoch, passed in, not hardcoded) gives the whole-run cost, scaled
by the batch-count ratio.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

COLUMNS = ["unix_time", "train_batch", "dataset_loaded", "validation", "epoch_complete",
           "last_epoch", "last_batch", "gpu_memory_used_mib"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _gpu_mem() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


class ProgressReader:
    """Incrementally counts events in a growing JSONL file (only complete lines)."""

    def __init__(self, path: Path):
        self.path, self.offset, self.buf = path, 0, b""
        self.counts = {"train_batch": 0, "dataset_loaded": 0, "validation": 0, "epoch_complete": 0}
        self.last_epoch = self.last_batch = ""
        self.dataset_loaded = None

    def poll(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        self.offset += len(data)
        self.buf += data
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            ev = row.get("event")
            if ev in self.counts:
                self.counts[ev] += 1
            if ev == "dataset_loaded" and self.dataset_loaded is None:
                self.dataset_loaded = row
            if ev == "train_batch":
                self.last_epoch, self.last_batch = row.get("epoch", ""), row.get("batch", "")


def sample(progress: Path, out: Path, interval: float, pid: int = 0, max_seconds: float = 0.0,
           clock=time.time, sleep=time.sleep, gpu=_gpu_mem) -> int:
    if out.exists():
        raise SystemExit(f"refusing to overwrite: {out}")
    reader = ProgressReader(progress)
    start = clock()
    n = 0
    with out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        while True:
            reader.poll()
            c = reader.counts
            w.writerow([f"{clock():.3f}", c["train_batch"], c["dataset_loaded"], c["validation"],
                        c["epoch_complete"], reader.last_epoch, reader.last_batch, gpu()])
            f.flush()
            n += 1
            if pid and not _alive(pid):
                break
            if max_seconds and clock() - start >= max_seconds:
                break
            sleep(interval)
    return n


def _first(samples, col, threshold=1):
    for s in samples:
        if int(s[col]) >= threshold:
            return float(s["unix_time"])
    return None


def summarize(samples_path: Path, startup: Path, progress: Path, *, epochs: int,
              meta_batch_size: int, ref_elapsed_s: float, ref_batches_per_epoch: int,
              requested_wall_s: float, min_rate_window_s: float = 300.0) -> dict:
    with samples_path.open(newline="") as f:
        samples = list(csv.DictReader(f, delimiter="\t"))
    if not samples:
        raise SystemExit("no samples")
    started = float(json.loads(startup.read_text())["started_at_unix"])
    reader = ProgressReader(progress)
    reader.poll()
    if reader.dataset_loaded is None:
        raise SystemExit("dataset never loaded; no timings")
    train_indices = int(reader.dataset_loaded["train_indices"])
    bpe = train_indices // meta_batch_size

    t_loaded = _first(samples, "dataset_loaded")
    t_init_val = _first(samples, "validation")
    t_epoch0 = _first(samples, "epoch_complete")
    # rate over epoch-0 training only: samples after the first batch and before epoch 0 ends
    window = [s for s in samples if int(s["train_batch"]) > 0 and int(s["epoch_complete"]) == 0]
    rate = None
    if len(window) >= 2:
        t0, t1 = float(window[0]["unix_time"]), float(window[-1]["unix_time"])
        b0, b1 = int(window[0]["train_batch"]), int(window[-1]["train_batch"])
        if t1 - t0 >= min_rate_window_s and b1 > b0:
            rate = (b1 - b0) / (t1 - t0)
    if rate is None:
        raise SystemExit(f"rate window shorter than {min_rate_window_s:.0f} s or no batches; no extrapolation")

    load_s = t_loaded - started if t_loaded else None
    init_val_s = t_init_val - t_loaded if (t_init_val and t_loaded) else None
    epoch0_train_s = bpe / rate
    epoch0_observed_s = t_epoch0 - t_init_val if (t_epoch0 and t_init_val) else None
    end_of_epoch_s = (epoch0_observed_s - epoch0_train_s) if epoch0_observed_s else None
    per_epoch_s = epoch0_observed_s if epoch0_observed_s else epoch0_train_s + (init_val_s or 0.0)
    lower_bound_s = (load_s or 0.0) + (init_val_s or 0.0) + epochs * per_epoch_s
    ref_scaled_s = ref_elapsed_s * bpe / ref_batches_per_epoch
    gpu = [int(s["gpu_memory_used_mib"]) for s in samples if s["gpu_memory_used_mib"].strip().isdigit()]
    return {
        "artifact_type": "D013_interactive_benchmark_summary_v1",
        "measured": {
            "load_seconds": load_s, "initial_validation_seconds": init_val_s,
            "train_indices": train_indices, "batches_per_epoch": bpe,
            "epoch0_batches_per_second": rate, "epoch0_train_seconds_from_rate": epoch0_train_s,
            "epoch0_observed_seconds": epoch0_observed_s,
            "end_of_epoch_validation_checkpoint_seconds": end_of_epoch_s,
            "rate_window_seconds": float(window[-1]["unix_time"]) - float(window[0]["unix_time"]),
            "max_gpu_memory_used_mib": max(gpu) if gpu else None,
            "sample_interval_resolution_note": "first-seen times are accurate to one sampling interval",
        },
        "extrapolation": {
            "epochs": epochs,
            "first_order_lower_bound_seconds": lower_bound_s,
            "per_epoch_seconds_used": per_epoch_s,
            "per_epoch_basis": "observed epoch 0" if epoch0_observed_s else "rate x batches + initial-validation time",
            "reference_scaled_seconds": ref_scaled_s,
            "reference": {"elapsed_seconds": ref_elapsed_s, "batches_per_epoch": ref_batches_per_epoch},
            "requested_wall_seconds": requested_wall_s,
            "margin_vs_reference_scaled": requested_wall_s / ref_scaled_s,
            "margin_vs_lower_bound": requested_wall_s / lower_bound_s,
            "note": ("epochs >= first_order_to_second_order_epoch are second order and slower; the lower "
                     "bound uses first-order epoch 0 only. The reference-scaled figure covers the whole run."),
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--progress", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--interval", type=float, default=15.0)
    s.add_argument("--pid", type=int, default=0)
    s.add_argument("--max-seconds", type=float, default=0.0)
    m = sub.add_parser("summarize")
    m.add_argument("--samples", type=Path, required=True)
    m.add_argument("--startup", type=Path, required=True)
    m.add_argument("--progress", type=Path, required=True)
    m.add_argument("--epochs", type=int, required=True)
    m.add_argument("--meta-batch-size", type=int, required=True)
    m.add_argument("--ref-elapsed-seconds", type=float, required=True)
    m.add_argument("--ref-batches-per-epoch", type=int, required=True)
    m.add_argument("--requested-wall-seconds", type=float, required=True)
    m.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "sample":
        sample(a.progress, a.out, a.interval, a.pid, a.max_seconds)
        return 0
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite: {a.out}")
    r = summarize(a.samples, a.startup, a.progress, epochs=a.epochs, meta_batch_size=a.meta_batch_size,
                  ref_elapsed_s=a.ref_elapsed_seconds, ref_batches_per_epoch=a.ref_batches_per_epoch,
                  requested_wall_s=a.requested_wall_seconds)
    a.out.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    print(json.dumps(r, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
