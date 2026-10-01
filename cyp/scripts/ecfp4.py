from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


RADIUS = 2
N_BITS = 2048


def smiles_to_ecfp4(smiles: str) -> np.ndarray:
    """Convert one SMILES string to the frozen 2048-bit Morgan radius-2 fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to parse SMILES: {smiles}")

    fp = AllChem.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
    ).GetFingerprint(mol)

    result = np.zeros(N_BITS, dtype=np.uint8)
    for bit in fp.GetOnBits():
        result[bit] = 1

    return result


def ecfp4_matrix(smiles_values) -> np.ndarray:
    """Convert an iterable of SMILES strings to a 2D ECFP4 matrix."""
    rows = [smiles_to_ecfp4(smiles) for smiles in smiles_values]

    if not rows:
        return np.empty((0, N_BITS), dtype=np.uint8)

    result = np.vstack(rows)
    return result


def check_ecfp4_invariants() -> None:
    smiles = "CCO"

    first = smiles_to_ecfp4(smiles)
    second = smiles_to_ecfp4(smiles)

    if first.shape != (N_BITS,):
        raise AssertionError(
            f"Expected one fingerprint of shape {(N_BITS,)}, got {first.shape}"
        )

    if not np.array_equal(first, second):
        raise AssertionError("Identical SMILES produced different fingerprints.")

    if not np.isin(first, [0, 1]).all():
        raise AssertionError("Fingerprint contains values other than 0 and 1.")

    batch = ecfp4_matrix(["CCO", "CCN", "c1ccccc1"])

    if batch.shape != (3, N_BITS):
        raise AssertionError(
            f"Expected batch shape {(3, N_BITS)}, got {batch.shape}"
        )

    if batch.dtype != np.uint8:
        raise AssertionError(
            f"Expected uint8 fingerprint matrix, got {batch.dtype}"
        )

    print("PASS: frozen Morgan radius-2 / 2048-bit ECFP4 invariants.")


if __name__ == "__main__":
    check_ecfp4_invariants()
