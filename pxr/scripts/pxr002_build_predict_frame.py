"""Structure-only PXR-002 prediction artifact for an explicit molecule list (Tripwire-CYP-011).

Builds the inputs the ten frozen PXR-002 final models (D018 Amendment 6) need to PREDICT a list
of molecules, and nothing else: row ids, SMILES and the fixed v1_rdkit_2d_normalized descriptors
computed before any learned scaling (the model's own X_d scaler lives in each checkpoint).
No label column is read or written.

The molecule list is the explicit CYP/PXR mapping table built by build_screen_input_set.py
(analysis/cyp_010_screen_input_set_20261125/screen_input_mapping.csv). Row ids are its
pxr_row_id column and SMILES its pxr_source_smiles column (the role-appropriate PXR string);
nothing is assumed interchangeable with OCNT_ID or cyp_molecule_name. Duplicate or empty ids or
SMILES stop the run. A non-finite descriptor stops the run and lists the row ids; imputation and
exclusion are not approved.

Output (new, empty directory): pxr002_predict_frame.csv (row_id, SMILES),
pxr002_predict_descriptors.npy (n x 200 float64), SHA256SUMS, PROVENANCE.txt.

Usage (repository root; the Chemprop environment that also has descriptastorus; every argument
is required):
    python -I pxr/scripts/pxr002_build_predict_frame.py MAPPING_CSV MAPPING_SHA256 OUT_DIR
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr002_dataset as ds  # noqa: E402

PREDICT_FRAME = "pxr002_predict_frame.csv"
PREDICT_DESCRIPTORS = "pxr002_predict_descriptors.npy"
ID_FIELD = "pxr_row_id"
SMILES_FIELD = "pxr_source_smiles"
REQUIRED_FIELDS = ("OCNT_ID", "cyp_molecule_name", ID_FIELD, "pxr_role", SMILES_FIELD)


class BuildStop(ValueError):
    """A condition that stops the build; nothing is dropped or imputed."""


def read_mapping_molecules(path, expected_sha256: str) -> list:
    """[(row_id, smiles)] in file order, literal text, from the hash-pinned mapping table."""
    actual = ds.sha256(path)
    if actual != expected_sha256:
        raise BuildStop(f"Mapping hash {actual} differs from the declared {expected_sha256}.")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_FIELDS if name not in (reader.fieldnames or [])]
        if missing:
            raise BuildStop(f"Mapping table lacks columns {missing}.")
        molecules, seen = [], set()
        for record in reader:
            row_id, smiles = record[ID_FIELD], record[SMILES_FIELD]
            if row_id.strip() == "" or smiles.strip() == "":
                raise BuildStop(f"Line {reader.line_num}: empty {ID_FIELD} or {SMILES_FIELD}.")
            if row_id in seen:
                raise BuildStop(f"Duplicate {ID_FIELD}: {row_id}")
            seen.add(row_id)
            molecules.append((row_id, smiles))
    if not molecules:
        raise BuildStop("Mapping table has no rows.")
    return molecules


def compute_checked_descriptors(smiles: list, row_ids: list, compute=ds.compute_descriptors):
    """Descriptors for every row; a non-finite value stops the build and names the rows."""
    matrix = np.asarray(compute(smiles, allow_nonfinite=True), dtype=np.float64)
    if matrix.shape != (len(smiles), ds.N_DESCRIPTORS):
        raise BuildStop(f"Descriptor matrix shape {matrix.shape} is not ({len(smiles)}, {ds.N_DESCRIPTORS}).")
    bad = np.flatnonzero(~np.isfinite(matrix).all(axis=1))
    if bad.size:
        listed = [row_ids[i] for i in bad]
        raise BuildStop(f"Non-finite descriptors in {bad.size} molecules (imputation not approved): {listed}")
    return matrix


def build(mapping_path, mapping_sha256: str, out_dir, compute=ds.compute_descriptors) -> dict:
    out = Path(out_dir)
    if out.exists():
        raise BuildStop(f"{out} already exists; refusing to overwrite an artifact.")
    molecules = read_mapping_molecules(mapping_path, mapping_sha256)
    row_ids = [m[0] for m in molecules]
    smiles = [m[1] for m in molecules]
    matrix = compute_checked_descriptors(smiles, row_ids, compute)
    out.mkdir(parents=True)
    with open(out / PREDICT_FRAME, "x", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["row_id", "SMILES"])
        writer.writerows(molecules)
    np.save(out / PREDICT_DESCRIPTORS, np.ascontiguousarray(matrix, dtype=np.float64), allow_pickle=False)
    names = [PREDICT_FRAME, PREDICT_DESCRIPTORS]
    (out / "SHA256SUMS").write_text("".join(f"{ds.sha256(out / n)}  {n}\n" for n in names))
    (out / "PROVENANCE.txt").write_text(
        "PXR-002 structure-only prediction artifact (Tripwire-CYP-011)\n"
        f"mapping table sha256: {mapping_sha256}\n"
        f"row id column: {ID_FIELD}; SMILES column: {SMILES_FIELD}\n"
        f"molecules: {len(molecules)}; descriptors: v1_rdkit_2d_normalized, cached before learned scaling\n"
        "no label column read or written; no descriptor imputed or molecule excluded\n"
    )
    return {"molecules": len(molecules), "descriptor_shape": list(matrix.shape),
            "sums_sha256": ds.sha256(out / "SHA256SUMS")}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        raise SystemExit(__doc__)
    try:
        report = build(*argv)
    except BuildStop as exc:
        raise SystemExit(f"STOPPED: {exc}")
    print(f"molecules {report['molecules']}; descriptors {tuple(report['descriptor_shape'])}; "
          f"SHA256SUMS sha256 {report['sums_sha256']}; no labels read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
