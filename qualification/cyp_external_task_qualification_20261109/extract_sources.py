from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path("qualification/cyp_external_task_qualification_20261109")
RAW = ROOT / "raw"
DERIVED = ROOT / "derived"
MANIFESTS = ROOT / "manifests"

for p in (RAW, DERIVED, MANIFESTS):
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

UA = "Tripwire-CYP-source-qualification/1.0"


def fetch_bytes(url: str, retries: int = 5) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            last = e
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise last


def write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def chembl_get(endpoint: str, params: dict) -> dict:
    q = urllib.parse.urlencode(params)
    url = f"https://www.ebi.ac.uk/chembl/api/data/{endpoint}.json?{q}"
    return json.loads(fetch_bytes(url))


def extract_chembl() -> dict:
    summary = {}

    for isoform, target in CHEMBL_TARGETS.items():
        out = RAW / f"chembl_{isoform}_activities.jsonl"
        count = 0
        assay_ids = set()

        with out.open("w", encoding="utf-8") as handle:
            offset = 0
            limit = 1000

            while True:
                doc = chembl_get(
                    "activity",
                    {
                        "target_chembl_id": target,
                        "limit": limit,
                        "offset": offset,
                    },
                )

                rows = doc.get("activities", [])
                for row in rows:
                    row["_qualification_isoform"] = isoform
                    row["_qualification_target"] = target
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    count += 1
                    if row.get("assay_chembl_id"):
                        assay_ids.add(row["assay_chembl_id"])

                meta = doc.get("page_meta", {})
                nxt = meta.get("next")
                if not nxt or not rows:
                    break

                offset += len(rows)
                time.sleep(0.25)

        # Retrieve assay metadata separately so task semantics are not inferred
        # from activity rows alone.
        assay_out = RAW / f"chembl_{isoform}_assays.jsonl"
        with assay_out.open("w", encoding="utf-8") as handle:
            for i, aid in enumerate(sorted(assay_ids), 1):
                doc = chembl_get("assay", {"assay_chembl_id": aid, "limit": 20})
                rows = doc.get("assays", [])
                if len(rows) != 1:
                    raise RuntimeError(
                        f"{isoform} {aid}: expected one assay record, got {len(rows)}"
                    )
                row = rows[0]
                row["_qualification_isoform"] = isoform
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                if i % 4 == 0:
                    time.sleep(0.25)

        summary[isoform] = {
            "target_chembl_id": target,
            "raw_activity_records": count,
            "distinct_assay_ids": len(assay_ids),
            "activities_sha256": sha256(out),
            "assays_sha256": sha256(assay_out),
        }

    return summary


def extract_pubchem() -> dict:
    summary = {}

    for isoform, aid in PUBCHEM_AIDS.items():
        desc_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"assay/aid/{aid}/description/JSON"
        )
        data_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"assay/aid/{aid}/CSV"
        )

        desc_path = RAW / f"pubchem_AID{aid}_description.json"
        csv_path = RAW / f"pubchem_AID{aid}_full.csv"

        write_bytes(desc_path, fetch_bytes(desc_url))
        time.sleep(0.3)
        write_bytes(csv_path, fetch_bytes(data_url))
        time.sleep(0.3)

        # Parse without assuming fixed readout column names.
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            rows = list(reader)

        lower = {x.lower(): x for x in fields}

        # Preserve/report presence of the requested quality/outcome fields.
        requested_terms = {
            "curve": [x for x in fields if "curve" in x.lower()],
            "activity": [
                x for x in fields
                if "activity" in x.lower() or "outcome" in x.lower()
            ],
            "efficacy": [x for x in fields if "effic" in x.lower()],
            "potency": [
                x for x in fields
                if any(t in x.lower() for t in ("ac50", "ic50", "potency"))
            ],
        }

        summary[isoform] = {
            "aid": aid,
            "raw_records": len(rows),
            "columns": fields,
            "quality_field_matches": requested_terms,
            "description_sha256": sha256(desc_path),
            "full_table_sha256": sha256(csv_path),
        }

    return summary


def main() -> None:
    provenance = {
        "purpose": "CYP external assay/task source qualification only",
        "openadmet_public_test_labels_accessed": False,
        "chembl_targets": CHEMBL_TARGETS,
        "pubchem_aids": PUBCHEM_AIDS,
        "pubchem_note": (
            "Known PubChem assays establish measurement volume only; "
            "they are not counted as multiple assays per isoform."
        ),
    }

    print("Extracting ChEMBL...")
    provenance["chembl"] = extract_chembl()

    print("Extracting PubChem...")
    provenance["pubchem"] = extract_pubchem()

    prov_path = MANIFESTS / "source_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    files = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.txt"
    )

    with (MANIFESTS / "SHA256SUMS.txt").open("w") as f:
        for p in files:
            f.write(f"{sha256(p)}  {p.relative_to(ROOT)}\n")

    print(json.dumps(provenance, indent=2))
    print("RAW EXTRACTION COMPLETE")


if __name__ == "__main__":
    main()
