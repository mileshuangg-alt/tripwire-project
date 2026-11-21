"""Frozen molecule-level descriptor cache for the D021 comparison.

196 fixed descriptors per molecule: Chemprop 2.3.1 `v1_rdkit_2d_normalized`
(200 columns, values bounded in [0, 1]) with positions 39, 41, 43 and 45 (the
four partial-charge descriptors) dropped for every molecule. Computed from the
SMILES a stage actually uses, once, label-free, and shipped as a hashed
artifact; never recomputed where the model trains. Nothing is imputed and no
molecule is excluded: a non-finite value after the drop raises.

Artifact: `descriptors.npz` (`smiles` unicode array, `values` float64 [n, 196])
plus `descriptor_manifest.json` (its SHA-256, versions, positions, counts, the
hashes of the source files and of the molecule-set rules). Loading verifies
both hashes and refuses non-finite values. A SMILES that is not in the cache
raises instead of being computed, which also keeps protected molecules out.

Usage (build; Miles runs it, output directory must not exist):
    python -I t121_descriptor_cache.py build --dataset D.csv --original-split S.csv \
      --screen screen_full.csv --identity adjudication.tsv \
      --similarity cluster_adjudication.tsv --contract contract.json --output-dir OUT
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyp002_config import CYP002_DESCRIPTOR_COLUMN_PREFIX  # noqa: E402

FEATURIZER_NAME = "v1_rdkit_2d_normalized"
SOURCE_COLUMNS = 200
DROPPED_POSITIONS = (39, 41, 43, 45)
KEPT_POSITIONS = tuple(i for i in range(SOURCE_COLUMNS) if i not in DROPPED_POSITIONS)
DIM = len(KEPT_POSITIONS)  # 196
ARTIFACT_TYPE = "D021_descriptor_cache_v1"
NPZ_NAME = "descriptors.npz"
MANIFEST_NAME = "descriptor_manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def column_names():
    width = len(str(DIM - 1))
    return [f"{CYP002_DESCRIPTOR_COLUMN_PREFIX}{i:0{width}d}" for i in range(DIM)]


def compute_matrix(smiles_list, featurizer=None):
    """Descriptor matrix [n, 196] for the given SMILES; raises on any problem."""
    from rdkit import Chem

    if featurizer is None:
        from chemprop.featurizers import MoleculeFeaturizerRegistry

        featurizer = MoleculeFeaturizerRegistry[FEATURIZER_NAME]()
    rows = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Unparsable SMILES: {smiles}")
        vector = np.asarray(featurizer(molecule), dtype=np.float64)
        if vector.shape != (SOURCE_COLUMNS,):
            raise ValueError(f"Expected {SOURCE_COLUMNS} source descriptors, got {vector.shape}.")
        rows.append(vector[list(KEPT_POSITIONS)])
    matrix = np.vstack(rows) if rows else np.zeros((0, DIM))
    bad = ~np.isfinite(matrix)
    if bad.any():
        molecules = [smiles_list[i] for i in np.where(bad.any(axis=1))[0][:10]]
        raise ValueError(
            f"{int(bad.sum())} non-finite descriptor values in "
            f"{int(bad.any(axis=1).sum())} molecules after the column drop "
            f"(no imputation, no exclusion); first: {molecules}"
        )
    return matrix


class DescriptorCache:
    """Read-only lookup of frozen descriptors by exact SMILES string."""

    def __init__(self, smiles, values, manifest=None, manifest_sha256=None):
        values = np.asarray(values, dtype=np.float64)
        smiles = [str(s) for s in smiles]
        if values.ndim != 2 or values.shape != (len(smiles), DIM):
            raise ValueError(f"Cache values must be [n, {DIM}].")
        if len(set(smiles)) != len(smiles):
            raise ValueError("Duplicate SMILES in descriptor cache.")
        if not np.isfinite(values).all():
            raise ValueError("Descriptor cache contains non-finite values.")
        self._index = {s: i for i, s in enumerate(smiles)}
        self._values = values
        self._manifest = manifest or {}
        self._manifest_sha256 = manifest_sha256

    @property
    def dim(self):
        return DIM

    def __len__(self):
        return len(self._index)

    def __contains__(self, smiles):
        return smiles in self._index

    def lookup(self, smiles):
        try:
            return self._values[self._index[smiles]].copy()
        except KeyError:
            raise KeyError(
                f"SMILES not in the frozen descriptor cache (not computed on demand): {smiles}"
            ) from None

    def provenance(self):
        return {
            "artifact_type": ARTIFACT_TYPE,
            "manifest_sha256": self._manifest_sha256,
            "npz_sha256": self._manifest.get("npz_sha256"),
            "featurizer": FEATURIZER_NAME,
            "dropped_positions": list(DROPPED_POSITIONS),
            "dim": DIM,
            "molecules": len(self),
        }

    @classmethod
    def load(cls, directory, *, expected_manifest_sha256):
        directory = Path(directory)
        manifest_path = directory / MANIFEST_NAME
        actual_manifest = sha256_file(manifest_path)
        if actual_manifest != expected_manifest_sha256:
            raise ValueError(
                f"Descriptor manifest hash mismatch: {actual_manifest} != {expected_manifest_sha256}"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("artifact_type") != ARTIFACT_TYPE:
            raise ValueError("Not a D021 descriptor cache manifest.")
        npz_path = directory / NPZ_NAME
        if sha256_file(npz_path) != manifest["npz_sha256"]:
            raise ValueError("Descriptor npz hash does not match its manifest.")
        if (
            manifest["dim"] != DIM
            or list(manifest["dropped_positions"]) != list(DROPPED_POSITIONS)
            or manifest["featurizer"] != FEATURIZER_NAME
        ):
            raise ValueError("Descriptor manifest does not describe the approved 196-column set.")
        with np.load(npz_path, allow_pickle=False) as archive:
            smiles = archive["smiles"].tolist()
            values = archive["values"]
        if len(smiles) != manifest["molecules"]:
            raise ValueError("Descriptor molecule count differs from its manifest.")
        return cls(smiles, values, manifest, actual_manifest)


def attach_descriptor_columns(frame, cache, smiles_column="SMILES"):
    """Copy of `frame` with the 196 descriptor columns (named by position)."""
    import pandas as pd

    matrix = np.vstack([cache.lookup(s) for s in frame[smiles_column].tolist()]) if len(frame) else np.zeros((0, DIM))
    columns = pd.DataFrame(matrix, columns=column_names(), index=frame.index)
    overlap = set(columns.columns) & set(frame.columns)
    if overlap:
        raise ValueError("Frame already has descriptor columns.")
    return pd.concat([frame, columns], axis=1)


def build_cache(smiles_sets, output_dir, *, sources, featurizer=None):
    """Compute and write the cache. `smiles_sets` maps a rule name to its SMILES list."""
    output_dir = Path(output_dir)
    union = []
    seen = set()
    for name in sorted(smiles_sets):
        for smiles in smiles_sets[name]:
            if smiles not in seen:
                seen.add(smiles)
                union.append(smiles)
    union.sort()
    matrix = compute_matrix(union, featurizer=featurizer)
    output_dir.mkdir(parents=True, exist_ok=False)
    npz_path = output_dir / NPZ_NAME
    with open(npz_path, "xb") as handle:
        np.savez(handle, smiles=np.array(union, dtype=str), values=matrix)
    try:
        import chemprop
        import rdkit
        versions = {"chemprop": chemprop.__version__, "rdkit": rdkit.__version__}
    except Exception:  # versions are provenance only
        versions = {}
    span = matrix.max(axis=0) - matrix.min(axis=0) if len(matrix) else np.zeros(DIM)
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "featurizer": FEATURIZER_NAME,
        "source_columns": SOURCE_COLUMNS,
        "dropped_positions": list(DROPPED_POSITIONS),
        "kept_positions": list(KEPT_POSITIONS),
        "dim": DIM,
        "molecules": len(union),
        "set_sizes": {name: len(set(values)) for name, values in smiles_sets.items()},
        "value_min": float(matrix.min()) if len(matrix) else None,
        "value_max": float(matrix.max()) if len(matrix) else None,
        "constant_columns_in_cache": int((span == 0).sum()),
        "npz_sha256": sha256_file(npz_path),
        "sources": sources,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, **versions},
        "note": "label-free; computed once; no imputation; no molecule excluded after the column drop",
    }
    with open(output_dir / MANIFEST_NAME, "x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def development_smiles(dataset_path, split_path):
    """SMILES of the development universe only; the protected internal test is never read."""
    import pandas as pd

    data = pd.read_csv(dataset_path)
    split = pd.read_csv(split_path)
    protected = set(split.loc[split["split"] == "test", "Molecule_Name"])
    development = data[~data["Molecule_Name"].isin(protected)]
    if set(development["Molecule_Name"]) & protected:
        raise RuntimeError("Protected molecule in the development set.")
    return development["SMILES"].tolist()


def screening_smiles(screen, identity, similarity, contract):
    """SMILES of exactly the molecules the screening stage uses (the frozen loader's output)."""
    from actfound_model_side import load_adjudicated_screen_assays

    assays = load_adjudicated_screen_assays(screen, identity, similarity, contract, featurize=False)
    return [smiles for assay in assays for smiles in assay.smiles]


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    for name in ("dataset", "original-split", "screen", "identity", "similarity", "contract", "output-dir"):
        build.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    paths = {
        "dataset": args.dataset, "original_split": args.original_split, "screen": args.screen,
        "identity": args.identity, "similarity": args.similarity, "contract": args.contract,
    }
    sources = {name: {"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for name, p in paths.items()}
    sets = {
        "development_universe": development_smiles(args.dataset, args.original_split),
        "screening_stage_molecules": screening_smiles(args.screen, args.identity, args.similarity, args.contract),
    }
    manifest = build_cache(sets, args.output_dir, sources=sources)
    print(json.dumps({k: manifest[k] for k in ("molecules", "set_sizes", "npz_sha256", "constant_columns_in_cache")}))
    print("manifest sha256:", sha256_file(Path(args.output_dir) / MANIFEST_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
