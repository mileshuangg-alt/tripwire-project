"""Build the D030 label-rank penalty package locally (controller side; runs no model code).

Same method as the D024 builder: start from the unchanged reference package
(t121_production_chain_20261117, SHA-256 pinned in t121_d021_build_package),
overlay exactly the files listed here from this checkout, and keep every other
file's reference bytes.

Usage:
    python t121_d030_build_package.py --reference-tar T --repo R --output-dir OUT
OUT must not exist. Writes OUT/t121_d030_package_20261126/ and a .tar.gz beside it.
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
import t121_d024_build_package as d24  # noqa: E402
import t121_d030_layout as layout  # noqa: E402

PACKAGE_NAME = "t121_d030_package_20261126"
sha256_file = d21.sha256_file

# Repo-relative paths; each keeps the same package-relative destination. The D024
# files stay in: the shared module, its tests and the coverage config are reused,
# and the D024 configs are what the new tests compare against.
OVERLAY = [
    *d24.OVERLAY,
    "cyp/artifacts/d029_penalty/penalty_rank_twosided.json",
    "cyp/artifacts/d029_penalty/penalty_rank_huber.json",
    "cyp/artifacts/d029_penalty/evaluation.json",
    "cyp/tests/test_t121_penalty_forms.py",
    "cyp/scripts/t121_d030_layout.py",
    "cyp/scripts/t121_d030_build_package.py",
    "cyp/artifacts/d030_penalty/penalty_label_mirrored.json",
    "cyp/artifacts/d030_penalty/evaluation.json",
    "cyp/tests/test_t121_penalty_mirrored.py",
]
SCOPE = (
    "D030 package (updated T121 with the label-rank mirrored band-penalty arm).\n"
    "Built from the unchanged reference package t121_production_chain_20261117 plus the files\n"
    "listed in the builder's overlay; files not overlaid are the reference package's bytes.\n"
    "Datasets, split manifests, screening data, the r1 selected screening record and\n"
    "environments are referenced by path and pinned SHA-256. This package starts nothing.\n"
)


def build(reference_tar, repo, output_dir):
    reference_tar, repo, output_dir = Path(reference_tar), Path(repo), Path(output_dir)
    if sha256_file(reference_tar) != d21.REFERENCE_TAR_SHA256:
        raise ValueError("Reference package archive does not match its pinned SHA-256.")
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
    for relative in OVERLAY:
        source = repo / relative
        if not source.is_file():
            raise FileNotFoundError(f"Overlay source missing: {relative}")
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (package / "PACKAGE_SCOPE.txt").write_text(SCOPE)
    reference = layout.reference_manifest_checked(repo / "screen_manifest_v2.json")
    pins = layout.build_pins(package, reference)
    layout.write_new_json(package / "d030_pins.json", pins)
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
