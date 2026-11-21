"""Build the D021 arm-D package locally (controller side; runs no model code).

Starts from the extracted reference package (t121_production_chain_20261117,
SHA-256 pinned below), which stays untouched on CoreHPC and on disk, and
overlays exactly the D021 files from this checkout. Files not overlaid keep the
reference package's bytes, so arm D differs from arm R only by the D021 edits.

Usage:
    python t121_d021_build_package.py --reference-tar T --repo R --output-dir OUT
OUT must not exist. Writes OUT/t121_d021_package_20261119/ and a .tar.gz beside it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t121_d021_layout as layout  # noqa: E402

REFERENCE_TAR_SHA256 = "9ecb6ccfe24ce1f5df8a744913379f389c9fdcfc93189df14745d1226b531b28"
PACKAGE_NAME = "t121_d021_package_20261119"
REFERENCE_NAME = "t121_production_chain_20261117"

# (repo-relative source, package-relative destination)
OVERLAY = [
    ("cyp/scripts/t121_screen_stage.py", "cyp/scripts/t121_screen_stage.py"),
    ("cyp/scripts/t121_weighted_training.py", "cyp/scripts/t121_weighted_training.py"),
    ("cyp/scripts/cyp002_config.py", "cyp/scripts/cyp002_config.py"),
    ("cyp/scripts/cyp002_model.py", "cyp/scripts/cyp002_model.py"),
    ("cyp/scripts/cyp002_train.py", "cyp/scripts/cyp002_train.py"),
    ("cyp/scripts/t121_production_cell.py", "cyp/scripts/t121_production_cell.py"),
    ("cyp/scripts/t121_screen_chain_run.py", "cyp/scripts/t121_screen_chain_run.py"),
    ("cyp/scripts/t121_descriptor_cache.py", "cyp/scripts/t121_descriptor_cache.py"),
    ("cyp/scripts/t121_plan_cells.py", "cyp/scripts/t121_plan_cells.py"),
    ("cyp/scripts/t121_cells.slurm", "cyp/scripts/t121_cells.slurm"),
    ("cyp/scripts/t121_d021_layout.py", "cyp/scripts/t121_d021_layout.py"),
    ("cyp/scripts/t121_d021_build_package.py", "cyp/scripts/t121_d021_build_package.py"),
    ("cyp/tests/test_t121_production_cell.py", "cyp/tests/test_t121_production_cell.py"),
    ("cyp/tests/test_t121_screen_chain_run.py", "cyp/tests/test_t121_screen_chain_run.py"),
    ("cyp/tests/test_t121_plan_cells.py", "cyp/tests/test_t121_plan_cells.py"),
    (f"{layout.DESCRIPTOR_CACHE_REL}/descriptor_manifest.json", f"{layout.DESCRIPTOR_CACHE_REL}/descriptor_manifest.json"),
    (f"{layout.DESCRIPTOR_CACHE_REL}/descriptors.npz", f"{layout.DESCRIPTOR_CACHE_REL}/descriptors.npz"),
    ("qualification/t121/d021_descriptor_smoke_20261119/check.py", "qualification/t121/d021_descriptor_smoke_20261119/check.py"),
    ("qualification/t121/d021_descriptor_smoke_20261119/check.sbatch", "qualification/t121/d021_descriptor_smoke_20261119/check.sbatch"),
    ("qualification/t121/d021_timing_20261119/timing.py", "qualification/t121_d021_timing/timing.py"),
]
# Reference-package files that belong to the reference arm only.
DROP = ("screen_manifest.json", "smoke", "SHA256SUMS.txt", "PACKAGE_SCOPE.txt")
SCOPE = (
    "D021 arm-D package (updated T121 plus 196 fixed RDKit descriptors). Built from the\n"
    "unchanged reference package t121_production_chain_20261117 plus the D021 files listed in\n"
    "SHA256SUMS.txt; files not overlaid are the reference package's bytes. Datasets, split\n"
    "manifests, screening data and environments are referenced by pinned path and SHA-256.\n"
    "No run or submission is authorized by this package; screening manifests are generated\n"
    "separately after the qualification check passes (t121_d021_layout.py manifests).\n"
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(reference_tar, repo, output_dir):
    reference_tar, repo, output_dir = Path(reference_tar), Path(repo), Path(output_dir)
    if sha256_file(reference_tar) != REFERENCE_TAR_SHA256:
        raise ValueError("Reference package archive does not match its pinned SHA-256.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse {output_dir}")
    output_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as scratch:
        with tarfile.open(reference_tar) as archive:
            archive.extractall(scratch, filter="data")
        package = output_dir / PACKAGE_NAME
        shutil.copytree(Path(scratch) / REFERENCE_NAME, package)
    for name in DROP:
        target = package / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for source, destination in OVERLAY:
        src = repo / source
        if not src.is_file():
            raise FileNotFoundError(f"Overlay source missing: {source}")
        dst = package / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    (package / "smoke").mkdir()
    (package / "smoke" / "README.txt").write_text("Slurm output of the D021 qualification check goes here.\n")
    (package / "PACKAGE_SCOPE.txt").write_text(SCOPE)
    reference = layout.reference_manifest_checked(repo / "screen_manifest_v2.json")
    pins = layout.build_pins(package, reference)
    layout.write_new_json(package / "d021_pins.json", pins)
    lines = []
    for path in sorted(p for p in package.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        lines.append(f"{sha256_file(path)}  {path.relative_to(package).as_posix()}")
    (package / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    archive_path = output_dir / f"{PACKAGE_NAME}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(package, arcname=PACKAGE_NAME)
    return package, archive_path, len(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-tar", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    package, archive, count = build(args.reference_tar, args.repo, args.output_dir)
    print(json.dumps({
        "package": str(package), "archive": str(archive),
        "archive_sha256": sha256_file(archive), "files_in_sums": count,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
