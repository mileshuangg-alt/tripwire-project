"""Build the D022 CheMeleon package locally (controller side; runs no model code).

Same method as the D021 builder: start from the unchanged reference package
(t121_production_chain_20261117, SHA-256 pinned in t121_d021_build_package),
overlay exactly the files listed here from this checkout, keep every other
file's reference bytes. The CheMeleon weights are copied from the PXR-004
artifact directory and checked against the frozen config's hash.

Usage:
    python t121_d022_build_package.py --reference-tar T --repo R --output-dir OUT
OUT must not exist. Writes OUT/t121_d022_package_20261121/ and a .tar.gz beside it.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t121_d021_build_package as d21  # noqa: E402
import t121_d022_layout as layout  # noqa: E402

PACKAGE_NAME = "t121_d022_package_20261121"
WEIGHTS_SOURCE = "pxr/artifacts/pxr004_weights/chemeleon_mp.pt"
sha256_file = d21.sha256_file

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
    ("cyp/scripts/t121_chemeleon.py", "cyp/scripts/t121_chemeleon.py"),
    ("cyp/scripts/t121_plan_cells.py", "cyp/scripts/t121_plan_cells.py"),
    ("cyp/scripts/t121_cells.slurm", "cyp/scripts/t121_cells.slurm"),
    ("cyp/scripts/t121_d021_aggregate.py", "cyp/scripts/t121_d021_aggregate.py"),
    ("cyp/scripts/t121_d022_aggregate.py", "cyp/scripts/t121_d022_aggregate.py"),
    ("cyp/scripts/t121_d021_layout.py", "cyp/scripts/t121_d021_layout.py"),
    ("cyp/scripts/t121_d021_build_package.py", "cyp/scripts/t121_d021_build_package.py"),
    ("cyp/scripts/t121_d022_layout.py", "cyp/scripts/t121_d022_layout.py"),
    ("cyp/scripts/t121_d022_build_package.py", "cyp/scripts/t121_d022_build_package.py"),
    ("cyp/artifacts/d022_chemeleon/chemeleon_config.json", "cyp/artifacts/d022_chemeleon/chemeleon_config.json"),
    (WEIGHTS_SOURCE, "cyp/artifacts/d022_chemeleon/chemeleon_mp.pt"),
    ("cyp/tests/test_t121_production_cell.py", "cyp/tests/test_t121_production_cell.py"),
    ("cyp/tests/test_t121_screen_chain_run.py", "cyp/tests/test_t121_screen_chain_run.py"),
    ("cyp/tests/test_t121_plan_cells.py", "cyp/tests/test_t121_plan_cells.py"),
    ("cyp/tests/test_t121_chemeleon.py", "cyp/tests/test_t121_chemeleon.py"),
    ("cyp/tests/test_t121_d022_aggregate.py", "cyp/tests/test_t121_d022_aggregate.py"),
    ("cyp/tests/test_t121_d021_aggregate.py", "cyp/tests/test_t121_d021_aggregate.py"),
    ("cyp/tests/test_t121_descriptor_arm.py", "cyp/tests/test_t121_descriptor_arm.py"),
    ("cyp/tests/test_t121_d022_layout.py", "cyp/tests/test_t121_d022_layout.py"),
    ("qualification/t121/d022_chemeleon_smoke_20261121/check.py", "qualification/t121/d022_chemeleon_smoke_20261121/check.py"),
    ("qualification/t121/d022_chemeleon_smoke_20261121/check.sbatch", "qualification/t121/d022_chemeleon_smoke_20261121/check.sbatch"),
    ("qualification/t121/d022_timing_20261121/timing.py", "qualification/t121_d022_timing/timing.py"),
    ("qualification/t121/d022_timing_20261121/timing.sbatch", "qualification/t121_d022_timing/timing.sbatch"),
]
SCOPE = (
    "D022 CheMeleon package (pretrained CheMeleon encoder, fully fine-tuned, in updated T121).\n"
    "Built from the unchanged reference package t121_production_chain_20261117 plus the files\n"
    "listed in SHA256SUMS.txt; files not overlaid are the reference package's bytes. Datasets,\n"
    "split manifests, screening data and environments are referenced by pinned path and SHA-256.\n"
    "No run or submission is authorized by this package; screening manifests are generated\n"
    "separately after the qualification check passes (t121_d022_layout.py manifests).\n"
)


def build(reference_tar, repo, output_dir):
    reference_tar, repo, output_dir = Path(reference_tar), Path(repo), Path(output_dir)
    if sha256_file(reference_tar) != d21.REFERENCE_TAR_SHA256:
        raise ValueError("Reference package archive does not match its pinned SHA-256.")
    config = json.loads((repo / "cyp/artifacts/d022_chemeleon/chemeleon_config.json").read_text())
    if sha256_file(repo / WEIGHTS_SOURCE) != config["weights"]["sha256"]:
        raise ValueError("CheMeleon weights do not match the frozen config's weights hash.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse {output_dir}")
    output_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as scratch:
        with tarfile.open(reference_tar) as archive:
            archive.extractall(scratch, filter="data")
        package = output_dir / PACKAGE_NAME
        shutil.copytree(Path(scratch) / d21.REFERENCE_NAME, package)
    for name in d21.DROP:
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
    (package / "smoke" / "README.txt").write_text("Slurm output of the D022 qualification check goes here.\n")
    (package / "PACKAGE_SCOPE.txt").write_text(SCOPE)
    reference = layout.reference_manifest_checked(repo / "screen_manifest_v2.json")
    pins = layout.build_pins(package, reference)
    layout.write_new_json(package / "d022_pins.json", pins)
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
