"""Screen-response report: potency outputs beside the approved screen columns (Tripwire-CYP-011).

Joins four hash-pinned tables through the explicit CYP/PXR mapping table and nothing else
(D027, D028). The values of every input table are passed through as text, unchanged.

    screen_predictions.csv  .input_id        -> mapping.OCNT_ID
    cyp_potency_ensemble.csv .input_id       -> mapping.cyp_molecule_name
    pxr_potency_means.csv   .row_id          -> mapping.pxr_row_id

input_id, Molecule_Name and row_id are not interchangeable and no key is substituted for
another. Every screen row is kept (left join); nothing is dropped. Unmatched keys, rows that
were not predicted and rows of a potency table that no mapping row reaches are counted and
listed in join_report.json. A duplicate key in any table or a smiles disagreement between the
screen and CYP potency rows stops the run with the counts written, because resolving it would
be a guess.

Report columns, in order: identifiers; the potency block (four CYP 150-member means and the
PXR five-seed multitask mean, with the aux-off mean as the secondary column); the screen block
(four CYP screen log2fc, three PXR per-seed 8.25 uM log2fc and their mean3); join statuses.
No combined score, spread column, flag, queue or cutoff is produced (D027).

Usage (Mac; every argument is required):
    python -I cyp/scripts/build_screen_report.py --screen-predictions F --screen-predictions-sha256 H \
        --mapping F --mapping-sha256 H --cyp-potency F --cyp-potency-sha256 H \
        --pxr-potency F --pxr-potency-sha256 H --out-dir NEW_DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

CYP_TASKS = tuple(f"{e}_pIC50_direct_inhibition" for e in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"))
CYP_POTENCY_COLUMNS = tuple(f"{t}_pred_mean150" for t in CYP_TASKS)
PXR_PRIMARY = "PXR_pEC50_multitask_mean5"
PXR_SECONDARY = "PXR_pEC50_aux_off_mean5"
SCREEN_COLUMNS = tuple(f"{e}_screen_log2fc" for e in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")) + (
    "PXR_screen_8p25uM_log2fc_seed20261125",
    "PXR_screen_8p25uM_log2fc_seed20261126",
    "PXR_screen_8p25uM_log2fc_seed20261127",
    "PXR_screen_8p25uM_log2fc_mean3",
)
IDENTIFIER_COLUMNS = ("OCNT_ID", "cyp_molecule_name", "pxr_row_id", "pxr_role", "pxr_eligibility_reason")
STATUS_COLUMNS = ("screen_status", "join_status_cyp", "join_status_pxr")
REPORT_COLUMNS = (
    IDENTIFIER_COLUMNS + CYP_POTENCY_COLUMNS + (PXR_PRIMARY, PXR_SECONDARY) + SCREEN_COLUMNS + STATUS_COLUMNS
)
OUT_REPORT = "screen_report.csv"
OUT_JOIN = "join_report.json"

MATCHED = "matched"
NO_ROW = "no_potency_row"

NOTES = (
    "CYP potency values are deployment predictions of the 150-member updated-T121 pool (D028); "
    "most members trained on these molecules, so they are not held-out predictions. Screen and "
    "potency values are reported side by side and are not blended into a combined score (D027). "
    "No trigger, flag, queue or cutoff is part of this report."
)


class JoinStop(ValueError):
    """A condition that stops the build; nothing is dropped or guessed."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path, expected_sha256: str, required: tuple) -> list:
    """Rows as dicts of literal text. Hash, required columns and duplicate headers are checked."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise JoinStop(f"{path}: hash {actual} differs from the declared {expected_sha256}.")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(set(header)) != len(header):
            raise JoinStop(f"{path}: duplicate column names.")
        missing = [name for name in required if name not in header]
        if missing:
            raise JoinStop(f"{path}: missing columns {missing}.")
        rows = list(reader)
    return rows


def duplicates(rows: list, key: str) -> dict:
    seen = {}
    for index, row in enumerate(rows):
        seen.setdefault(row[key], []).append(index)
    return {k: v for k, v in seen.items() if len(v) > 1}


def index_by(rows: list, key: str) -> dict:
    return {row[key]: row for row in rows}


def potency_status(row, columns: tuple) -> str:
    """matched only when the row was predicted and every value is present."""
    if row is None:
        return NO_ROW
    status = row.get("status", "ok")
    if status != "ok":
        return f"potency_not_predicted:{status}"
    if any(row.get(column, "") == "" for column in columns):
        return "potency_value_missing"
    return MATCHED


def build_join(screen: list, mapping: list, cyp: list, pxr: list) -> tuple:
    """Return (report_records, counts). Raises JoinStop, with counts attached, on any ambiguity."""
    counts = {
        "screen_rows": len(screen), "mapping_rows": len(mapping),
        "cyp_potency_rows": len(cyp), "pxr_potency_rows": len(pxr),
        "duplicates": {
            "screen.input_id": duplicates(screen, "input_id"),
            "mapping.OCNT_ID": duplicates(mapping, "OCNT_ID"),
            "mapping.cyp_molecule_name": duplicates(mapping, "cyp_molecule_name"),
            "mapping.pxr_row_id": duplicates(mapping, "pxr_row_id"),
            "cyp_potency.input_id": duplicates(cyp, "input_id"),
            "pxr_potency.row_id": duplicates(pxr, "row_id"),
        },
    }
    counts["duplicate_key_counts"] = {k: len(v) for k, v in counts["duplicates"].items()}
    if any(counts["duplicates"].values()):
        raise JoinStop(f"Duplicate keys; not resolved by guessing: {counts['duplicate_key_counts']}", counts)

    by_ocnt = index_by(mapping, "OCNT_ID")
    by_cyp_name = index_by(mapping, "cyp_molecule_name")
    by_pxr_id = index_by(mapping, "pxr_row_id")
    cyp_by_id, pxr_by_id = index_by(cyp, "input_id"), index_by(pxr, "row_id")

    screen_ids = {r["input_id"] for r in screen}
    counts["screen_ids_not_in_mapping"] = sorted(i for i in screen_ids if i not in by_ocnt)
    counts["mapping_rows_without_screen_row"] = sorted(i for i in by_ocnt if i not in screen_ids)
    counts["cyp_potency_ids_not_in_mapping"] = sorted(i for i in cyp_by_id if i not in by_cyp_name)
    counts["pxr_potency_ids_not_in_mapping"] = sorted(i for i in pxr_by_id if i not in by_pxr_id)
    counts["mapping_cyp_name_differs_from_ocnt"] = sorted(
        m["OCNT_ID"] for m in mapping if m["cyp_molecule_name"] != m["OCNT_ID"])

    smiles_disagreements = []
    records = []
    for row in screen:
        mapped = by_ocnt.get(row["input_id"])
        record = {name: "" for name in REPORT_COLUMNS}
        record["screen_status"] = row.get("status", "")
        for name in SCREEN_COLUMNS:
            record[name] = row.get(name, "")
        if mapped is None:
            record["OCNT_ID"] = row["input_id"]
            record["join_status_cyp"] = record["join_status_pxr"] = "no_mapping_row"
            records.append(record)
            continue
        for name in IDENTIFIER_COLUMNS:
            record[name] = mapped.get(name, "")
        cyp_row = cyp_by_id.get(mapped["cyp_molecule_name"])
        pxr_row = pxr_by_id.get(mapped["pxr_row_id"])
        record["join_status_cyp"] = potency_status(cyp_row, CYP_POTENCY_COLUMNS)
        record["join_status_pxr"] = potency_status(pxr_row, (PXR_PRIMARY, PXR_SECONDARY))
        if cyp_row is not None and cyp_row.get("input_smiles", "") != row.get("input_smiles", ""):
            smiles_disagreements.append(row["input_id"])
        if cyp_row is not None and record["join_status_cyp"] == MATCHED:
            for name in CYP_POTENCY_COLUMNS:
                record[name] = cyp_row[name]
        if pxr_row is not None and record["join_status_pxr"] == MATCHED:
            record[PXR_PRIMARY], record[PXR_SECONDARY] = pxr_row[PXR_PRIMARY], pxr_row[PXR_SECONDARY]
        records.append(record)
    counts["cyp_smiles_disagree_with_screen"] = smiles_disagreements
    if smiles_disagreements:
        raise JoinStop(
            f"{len(smiles_disagreements)} rows have a different SMILES in the screen and CYP potency "
            "tables; the two were not produced from the same input.", counts)

    for column in ("join_status_cyp", "join_status_pxr", "screen_status"):
        tally = {}
        for record in records:
            tally[record[column]] = tally.get(record[column], 0) + 1
        counts[f"{column}_counts"] = tally
    counts["report_rows"] = len(records)
    return records, counts


def write_outputs(out_dir, records, counts, hashes: dict, script_sha256: str) -> None:
    out = Path(out_dir)
    with open(out / OUT_REPORT, "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    payload = {"notes": NOTES, "counts": counts, "input_sha256": hashes,
               "report_sha256": sha256_file(out / OUT_REPORT), "script_sha256": script_sha256}
    with open(out / OUT_JOIN, "x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))


def run(args) -> dict:
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {out}")
    screen = read_table(args.screen_predictions, args.screen_predictions_sha256,
                        ("input_id", "input_smiles", "status") + SCREEN_COLUMNS)
    mapping = read_table(args.mapping, args.mapping_sha256,
                         ("OCNT_ID", "cyp_molecule_name", "pxr_row_id", "pxr_role", "pxr_eligibility_reason"))
    cyp = read_table(args.cyp_potency, args.cyp_potency_sha256,
                     ("input_id", "input_smiles", "status") + CYP_POTENCY_COLUMNS)
    pxr = read_table(args.pxr_potency, args.pxr_potency_sha256, ("row_id", PXR_PRIMARY, PXR_SECONDARY))
    out.mkdir(parents=True)
    hashes = {"screen_predictions": args.screen_predictions_sha256, "mapping": args.mapping_sha256,
              "cyp_potency": args.cyp_potency_sha256, "pxr_potency": args.pxr_potency_sha256}
    try:
        records, counts = build_join(screen, mapping, cyp, pxr)
    except JoinStop as exc:
        counts = exc.args[1] if len(exc.args) > 1 else {}
        with open(out / OUT_JOIN, "x") as handle:
            handle.write(json.dumps({"stopped": str(exc.args[0]), "counts": counts, "input_sha256": hashes},
                                    indent=2, sort_keys=True))
        raise
    write_outputs(out, records, counts, hashes, sha256_file(__file__))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("screen-predictions", "screen-predictions-sha256", "mapping", "mapping-sha256",
                 "cyp-potency", "cyp-potency-sha256", "pxr-potency", "pxr-potency-sha256", "out-dir"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        counts = run(args)
    except JoinStop as exc:
        raise SystemExit(f"STOPPED: {exc.args[0]} (counts written to join_report.json)")
    print(json.dumps({k: counts[k] for k in ("report_rows", "join_status_cyp_counts", "join_status_pxr_counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
