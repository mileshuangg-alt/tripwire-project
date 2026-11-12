from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path("qualification/cyp_external_task_qualification_20261109")
RAW = ROOT / "raw_bulk"
STATE = ROOT / "bulk_state"
MAN = ROOT / "manifests_bulk"

for p in (RAW, STATE, MAN):
    p.mkdir(parents=True, exist_ok=True)

CHEMBL_TARGETS = {
    "CYP1A2": "CHEMBL3356",
    "CYP2C9": "CHEMBL3397",
    "CYP2D6": "CHEMBL289",
    "CYP3A4": "CHEMBL340",
}

PUBCHEM_AIDS = {
    "CYP1A2": 410,
    "CYP2C9": 883,
    "CYP2D6": 891,
    "CYP3A4": 884,
}

UA = "Tripwire-CYP-source-qualification/2.0"

PUBCHEM_CSV_METADATA_TAGS = {
    "RESULT_TYPE",
    "RESULT_DESCR",
    "RESULT_UNIT",
    "RESULT_IS_ACTIVE_CONCENTRATION",
    "RESULT_IS_ACTIVE_CONCENTRATION_QUALIFIER",
    "RESULT_ATTR_CONC_MICROMOL",
}


def fetch(url: str, retries: int = 5) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as exc:
            last = exc
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise last


def post_form(url: str, params: dict, retries: int = 5) -> bytes:
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": UA,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as exc:
            last = exc
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise last


def fetch_pubchem_assay_csv_for_sids(aid: int, sids: list[str]) -> bytes:
    # POST keeps long SID lists out of the request URI while using the same
    # official PUG-REST assay CSV interface.
    return post_form(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV",
        {"sid": ",".join(sids)},
    )


def split_pubchem_csv_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metadata_rows = []
    result_rows = []
    for row in rows:
        tag = str(row.get("PUBCHEM_RESULT_TAG", "")).strip()
        sid = str(row.get("PUBCHEM_SID", "")).strip()
        if tag in PUBCHEM_CSV_METADATA_TAGS:
            metadata_rows.append(row)
            continue
        if not sid:
            raise RuntimeError(
                f"PubChem CSV nonmetadata row lacks SID: tag={tag!r}"
            )
        result_rows.append(row)
    return metadata_rows, result_rows


def get_json(endpoint: str, params: dict) -> dict:
    q = urllib.parse.urlencode(params)
    url = f"https://www.ebi.ac.uk/chembl/api/data/{endpoint}.json?{q}"
    return json.loads(fetch(url))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_target_activities(isoform: str, target: str) -> dict:
    out = RAW / f"chembl37_{isoform}_activities.jsonl"
    done = STATE / f"chembl37_{isoform}_activities.complete.json"

    if done.exists():
        d = json.loads(done.read_text())
        if out.exists() and digest(out) == d["sha256"]:
            print(f"{isoform}: activities already complete ({d['records']})")
            return d
        raise RuntimeError(f"{isoform}: completion record/hash mismatch")

    tmp = out.with_suffix(".jsonl.partial")
    tmp.unlink(missing_ok=True)

    offset = 0
    limit = 1000
    count = 0
    assay_ids = set()

    with tmp.open("w", encoding="utf-8") as handle:
        while True:
            doc = get_json("activity", {
                "target_chembl_id": target,
                "limit": limit,
                "offset": offset,
            })
            rows = doc.get("activities", [])

            for row in rows:
                row["_qualification_isoform"] = isoform
                row["_qualification_target"] = target
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                count += 1
                if row.get("assay_chembl_id"):
                    assay_ids.add(row["assay_chembl_id"])

            print(
                f"{isoform}: activities {count}",
                flush=True,
            )

            if not rows or not doc.get("page_meta", {}).get("next"):
                break

            offset += len(rows)
            time.sleep(0.20)

    tmp.replace(out)

    record = {
        "isoform": isoform,
        "target_chembl_id": target,
        "records": count,
        "distinct_assay_ids": len(assay_ids),
        "sha256": digest(out),
    }
    done.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def collect_required_assay_ids() -> set[str]:
    ids = set()
    for path in RAW.glob("chembl37_*_activities.jsonl"):
        with path.open() as f:
            for line in f:
                aid = json.loads(line).get("assay_chembl_id")
                if aid:
                    ids.add(aid)
    return ids


def extract_global_assays(required: set[str]) -> dict:
    out = RAW / "chembl37_required_assays.jsonl"
    done = STATE / "chembl37_required_assays.complete.json"
    tmp = out.with_suffix(".jsonl.partial")

    if done.exists():
        d = json.loads(done.read_text())
        if out.exists() and digest(out) == d["sha256"]:
            print(f"assays: already complete ({d['records']})")
            return d
        raise RuntimeError("assay completion record/hash mismatch")

    found = {}

    # Resume only from a structurally valid partial artifact.
    if tmp.exists():
        with tmp.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                aid = row.get("assay_chembl_id")

                if not aid:
                    raise RuntimeError(
                        f"partial assay row {line_number} lacks assay_chembl_id"
                    )
                if aid not in required:
                    raise RuntimeError(
                        f"partial assay row contains unrequested ID: {aid}"
                    )
                if aid in found:
                    raise RuntimeError(
                        f"duplicate assay ID in partial artifact: {aid}"
                    )

                found[aid] = row

        print(
            f"assays: resuming with {len(found)}/{len(required)} "
            f"already recovered",
            flush=True,
        )

    missing_ids = sorted(required - set(found))
    batch_size = 100
    request_count_this_run = 0

    with tmp.open("a", encoding="utf-8") as handle:
        for start in range(0, len(missing_ids), batch_size):
            batch = missing_ids[start:start + batch_size]

            doc = get_json(
                "assay",
                {
                    "assay_chembl_id__in": ",".join(batch),
                    "limit": batch_size + 10,
                },
            )

            rows = doc.get("assays", [])
            request_count_this_run += 1

            got = [
                row.get("assay_chembl_id")
                for row in rows
                if row.get("assay_chembl_id")
            ]

            requested_set = set(batch)
            got_set = set(got)

            missing = sorted(requested_set - got_set)
            extra = sorted(got_set - requested_set)

            if missing or extra:
                raise RuntimeError(
                    f"assay batch mismatch at remaining offset {start}: "
                    f"missing={missing[:10]} extra={extra[:10]}"
                )

            if len(got) != len(got_set):
                raise RuntimeError(
                    f"duplicate assay IDs returned in batch at offset {start}"
                )

            for row in rows:
                aid = row["assay_chembl_id"]

                if aid in found:
                    raise RuntimeError(
                        f"assay returned more than once: {aid}"
                    )

                found[aid] = row
                handle.write(json.dumps(row, sort_keys=True) + "\n")

            # Make every successful batch durable before the next request.
            handle.flush()

            print(
                f"assays: recovered={len(found)}/{len(required)} "
                f"requests_this_run={request_count_this_run}",
                flush=True,
            )

            time.sleep(0.25)

    found_set = set(found)
    missing = sorted(required - found_set)
    extra = sorted(found_set - required)

    if missing or extra:
        raise RuntimeError(
            f"final assay identity mismatch: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    if len(found) != len(required):
        raise RuntimeError(
            f"expected {len(required)} unique assays, got {len(found)}"
        )

    tmp.replace(out)

    record = {
        "records": len(found),
        "required_assay_ids": len(required),
        "batch_size": batch_size,
        "lookup_filter": "assay_chembl_id__in",
        "identity_set_equality": True,
        "resume_capable": True,
        "sha256": digest(out),
    }

    done.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )

    return record


def extract_pubchem() -> dict:
    summary = {}

    for isoform, aid in PUBCHEM_AIDS.items():
        desc = RAW / f"pubchem_AID{aid}_description.json"
        table = RAW / f"pubchem_AID{aid}_full.csv"
        sid_inventory = RAW / f"pubchem_AID{aid}_sids.txt"
        chunk_dir = RAW / f"pubchem_AID{aid}_chunks"
        done = STATE / f"pubchem_AID{aid}.complete.json"

        if done.exists():
            d = json.loads(done.read_text())

            # AID410 was completed from the authoritative full table before
            # SID-list chunking was needed. Its v2 contract is sufficient.
            if (
                d.get("artifact_contract")
                == "PubChem_full_table_with_schema_rows_v2"
                and desc.exists()
                and table.exists()
                and digest(desc) == d["description_sha256"]
                and digest(table) == d["table_sha256"]
            ):
                print(f"PubChem AID {aid}: already complete")
                summary[isoform] = d
                continue

            if (
                desc.exists()
                and table.exists()
                and sid_inventory.exists()
                and digest(desc) == d["description_sha256"]
                and digest(table) == d["table_sha256"]
                and digest(sid_inventory) == d["sid_inventory_sha256"]
            ):
                print(f"PubChem AID {aid}: already complete")
                summary[isoform] = d
                continue

            raise RuntimeError(
                f"PubChem AID {aid}: completion/hash mismatch"
            )

        if not desc.exists():
            desc.write_bytes(fetch(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
                f"assay/aid/{aid}/description/JSON"
            ))
            time.sleep(0.25)

        sid_inventory_existed = sid_inventory.exists()
        if sid_inventory_existed:
            sid_text = sid_inventory.read_text(encoding="utf-8")
        else:
            sid_bytes = fetch(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
                f"assay/aid/{aid}/sids/TXT"
            )
            sid_text = sid_bytes.decode("utf-8")

        sids = [
            token.strip()
            for token in sid_text.replace(",", "\n").split()
            if token.strip()
        ]

        if not sids:
            raise RuntimeError(f"PubChem AID {aid}: empty SID inventory")
        if not all(x.isdigit() for x in sids):
            raise RuntimeError(f"PubChem AID {aid}: nonnumeric SID")
        if len(sids) != len(set(sids)):
            raise RuntimeError(
                f"PubChem AID {aid}: duplicate SIDs in inventory"
            )

        if not sid_inventory_existed:
            sid_inventory.write_text(
                "".join(f"{sid}\n" for sid in sids),
                encoding="utf-8",
            )

        # SID-filtered assay CSV requests include PubChem RESULT_* metadata
        # rows with blank SIDs. They are column definitions/attributes, not
        # assay-result rows, and must not be counted as measurements.
        chunk_size = 100
        chunk_dir.mkdir(exist_ok=True)

        expected_header = None
        all_result_rows = []
        metadata_rows_for_output = None

        for chunk_start in range(0, len(sids), chunk_size):
            chunk = sids[chunk_start:chunk_start + chunk_size]
            chunk_end = chunk_start + len(chunk) - 1
            chunk_path = (
                chunk_dir /
                f"chunk_{chunk_start:06d}_{chunk_end:06d}.csv"
            )

            if not chunk_path.exists():
                data = fetch_pubchem_assay_csv_for_sids(aid, chunk)
                chunk_tmp = chunk_path.with_suffix(".csv.partial")
                chunk_tmp.write_bytes(data)
                chunk_tmp.replace(chunk_path)
                time.sleep(0.25)

            text = chunk_path.read_text(encoding="utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            fields = reader.fieldnames or []
            rows = list(reader)
            metadata_rows, result_rows = split_pubchem_csv_rows(rows)

            if expected_header is None:
                expected_header = fields
                metadata_rows_for_output = metadata_rows
            elif fields != expected_header:
                raise RuntimeError(
                    f"PubChem AID {aid}: header changed between chunks"
                )
            elif metadata_rows != metadata_rows_for_output:
                raise RuntimeError(
                    f"PubChem AID {aid}: metadata rows changed between "
                    f"chunks at {chunk_start}"
                )

            returned_sids = [
                str(r.get("PUBCHEM_SID", "")).strip()
                for r in result_rows
            ]

            returned_sid_set = set(returned_sids)
            requested_sid_set = set(chunk)

            missing = sorted(requested_sid_set - returned_sid_set)
            extra = sorted(returned_sid_set - requested_sid_set)

            if missing or extra:
                raise RuntimeError(
                    f"PubChem AID {aid}: nonblank SID mismatch at "
                    f"chunk {chunk_start}; "
                    f"missing={missing[:10]} extra={extra[:10]}"
                )

            all_result_rows.extend(result_rows)

            print(
                f"PubChem AID {aid}: "
                f"{len(all_result_rows)} result rows from "
                f"{chunk_start + len(chunk)}/{len(sids)} SIDs",
                flush=True,
            )

        tmp = table.with_suffix(".csv.partial")
        with tmp.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(
                out_handle,
                fieldnames=expected_header,
                lineterminator="\n",
            )
            writer.writeheader()
            for row in metadata_rows_for_output or []:
                writer.writerow(row)
            for row in all_result_rows:
                writer.writerow(row)

        tmp.replace(table)

        with table.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            final_rows = list(reader)

        metadata_rows, result_rows = split_pubchem_csv_rows(final_rows)

        final_sids = [
            str(r.get("PUBCHEM_SID", "")).strip()
            for r in result_rows
        ]

        if set(final_sids) != set(sids):
            missing = sorted(set(sids) - set(final_sids))
            extra = sorted(set(final_sids) - set(sids))
            raise RuntimeError(
                f"PubChem AID {aid}: final nonblank SID-set mismatch; "
                f"missing={missing[:10]} extra={extra[:10]}"
            )

        field_groups = {
            "curve_quality": [
                x for x in fields
                if any(t in x.lower() for t in (
                    "curve", "fit", "chi2", "excluded",
                    "number of points", "data analysis qc"
                ))
            ],
            "activity_outcome_score": [
                x for x in fields
                if x in (
                    "PUBCHEM_ACTIVITY_OUTCOME",
                    "PUBCHEM_ACTIVITY_SCORE",
                    "Activity Direction",
                    "Activity Qualifier",
                )
            ],
            "potency": [
                x for x in fields
                if any(t in x.lower() for t in (
                    "ac50", "ic50", "potency"
                ))
            ],
            "efficacy_literal": [
                x for x in fields
                if "effic" in x.lower()
                and "hill coefficient" not in x.lower()
            ],
            "hill_slope": [
                x for x in fields
                if "hill coefficient" in x.lower()
            ],
            "concentration_response": [
                x for x in fields
                if x.lower().startswith("activity at ")
            ],
        }

        rec = {
            "isoform": isoform,
            "aid": aid,

            "raw_csv_rows": len(final_rows) + 1,
            "raw_records": len(final_rows) + 1,
            "schema_metadata_rows": len(metadata_rows),
            "schema_metadata_tag_counts": dict(Counter(
                str(r.get("PUBCHEM_RESULT_TAG", "")).strip()
                for r in metadata_rows
            )),
            "result_rows": len(result_rows),
            "stored_result_rows": len(result_rows),
            "unique_result_sids": len(set(final_sids)),
            "duplicate_result_sids": len(final_sids) - len(set(final_sids)),
            "sid_inventory_count": len(sids),
            "result_sid_identity_complete": True,
            "sid_set_equality": True,
            "sid_set_equality_semantics": (
                "Only nonblank assay-result rows are compared to the SID "
                "inventory; PubChem RESULT_* rows with blank SID are "
                "CSV metadata/column-definition rows."
            ),

            "stored_table_contains_schema_metadata_rows": True,
            "schema_metadata_semantics": sorted(PUBCHEM_CSV_METADATA_TAGS),

            "result_rows_with_cid": sum(
                bool(str(r.get("PUBCHEM_CID", "")).strip())
                for r in result_rows
            ),
            "result_rows_without_cid": sum(
                not bool(str(r.get("PUBCHEM_CID", "")).strip())
                for r in result_rows
            ),

            "full_retrieval_chunk_size": chunk_size,
            "retrieval_method": (
                "PUG-REST POST /rest/pug/assay/aid/{aid}/CSV "
                "with form field sid=<comma-separated SIDs>; chunk CSVs "
                "are stored durably before final table assembly."
            ),
            "artifact_contract":
                "PubChem_SID_POST_filtered_complete_table_with_metadata_rows_v3",
            "columns": fields,
            "field_groups": field_groups,
            "description_sha256": digest(desc),
            "table_sha256": digest(table),
            "sid_inventory_sha256": digest(sid_inventory),
        }

        done.write_text(
            json.dumps(rec, indent=2, sort_keys=True) + "\n"
        )

        summary[isoform] = rec

        print(
            f"PubChem AID {aid}: COMPLETE "
            f"({len(result_rows)} result rows)",
            flush=True,
        )

    return summary


def main() -> None:
    activity_summary = {}
    for iso, target in CHEMBL_TARGETS.items():
        activity_summary[iso] = extract_target_activities(iso, target)

    required = collect_required_assay_ids()
    print(f"unique required ChEMBL assays: {len(required)}", flush=True)

    assay_summary = extract_global_assays(required)
    pubchem_summary = extract_pubchem()

    provenance = {
        "artifact_type": "CYP_external_source_bulk_extraction_v1",
        "purpose": "source/task qualification only",
        "chembl_release": 37,
        "chembl_targets": CHEMBL_TARGETS,
        "pubchem_aids": PUBCHEM_AIDS,
        "openadmet_public_test_labels_accessed": False,
        "chembl_activity_summary": activity_summary,
        "chembl_assay_summary": assay_summary,
        "pubchem_summary": pubchem_summary,
        "notes": [
            "Raw measurement volume is distinct from task diversity.",
            "Known PubChem AIDs are one assay per listed isoform.",
            "No task-size threshold was applied.",
            "No activity eligibility filtering was applied.",
        ],
    }

    prov = MAN / "source_provenance.json"
    prov.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    files = sorted(
        p for p in RAW.rglob("*") if p.is_file()
    ) + sorted(
        p for p in STATE.rglob("*") if p.is_file()
    ) + [prov]

    with (MAN / "SHA256SUMS.txt").open("w") as f:
        for p in files:
            f.write(f"{digest(p)}  {p.relative_to(ROOT)}\n")

    print("BULK EXTRACTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
