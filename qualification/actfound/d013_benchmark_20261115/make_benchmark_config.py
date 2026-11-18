"""D013 interactive-benchmark config: an exact copy of config v2 with only output_dir changed.

Runs on CoreHPC in the pinned ActFound environment (Python 3.9). The production
output directory written into config v2 is never created or touched; the
benchmark writes only under its own output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ONLY_CHANGED_KEY = "output_dir"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def make(config_v2: Path, config_v2_sha256: str, output_dir: Path, out: Path) -> dict:
    if sha256(config_v2) != config_v2_sha256:
        raise SystemExit("config v2 SHA-256 mismatch")
    v2 = json.loads(config_v2.read_text())
    if v2.get("artifact_type") != "ActFoundProductionConfigV2":
        raise SystemExit("not an ActFoundProductionConfigV2")
    if Path(output_dir) == Path(v2[ONLY_CHANGED_KEY]):
        raise SystemExit("benchmark output_dir must differ from the production output_dir")
    if output_dir.exists():
        raise SystemExit(f"benchmark output_dir already exists: {output_dir}")
    if out.exists():
        raise SystemExit(f"refusing to overwrite: {out}")
    bench = dict(v2)
    bench[ONLY_CHANGED_KEY] = str(output_dir)
    changed = sorted(k for k in set(v2) | set(bench) if v2.get(k) != bench.get(k))
    if changed != [ONLY_CHANGED_KEY]:
        raise SystemExit(f"benchmark config differs from config v2 in {changed}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bench, indent=2, sort_keys=True) + "\n")
    record = {
        "config_v2": str(config_v2), "config_v2_sha256": config_v2_sha256,
        "benchmark_config": str(out), "benchmark_config_sha256": sha256(out),
        "production_output_dir_unchanged_in_v2": v2[ONLY_CHANGED_KEY],
        "benchmark_output_dir": str(output_dir), "keys_changed": changed,
        "script_sha256": sha256(Path(__file__)),
    }
    (out.parent / "benchmark_config_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config-v2", type=Path, required=True)
    p.add_argument("--config-v2-sha256", required=True)
    p.add_argument("--benchmark-output-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    print(json.dumps(make(a.config_v2, a.config_v2_sha256, a.benchmark_output_dir, a.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
