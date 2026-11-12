from __future__ import annotations

import csv
import io
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from extract_sources_bulk import (
    PUBCHEM_CSV_METADATA_TAGS,
    fetch_pubchem_assay_csv_for_sids,
    split_pubchem_csv_rows,
)

ROOT = Path("qualification/cyp_external_task_qualification_20261109")
SID_INVENTORY = ROOT / "raw_bulk" / "pubchem_AID883_sids.txt"
LEGACY_METADATA_TAGS = {"RESULT_TYPE", "RESULT_DESCR", "RESULT_UNIT"}


def parse_csv(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def main() -> None:
    sids = [
        x.strip()
        for x in SID_INVENTORY.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ][:100]
    requested = set(sids)

    get_url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/883/CSV?"
        + urllib.parse.urlencode({"sid": ",".join(sids)})
    )
    get_req = urllib.request.Request(
        get_url,
        headers={"User-Agent": "Tripwire-CYP-source-qualification-smoke/1.0"},
    )
    with urllib.request.urlopen(get_req, timeout=120) as response:
        get_rows = parse_csv(response.read())

    legacy_result_rows = [
        row for row in get_rows
        if str(row.get("PUBCHEM_RESULT_TAG", "")).strip()
        not in LEGACY_METADATA_TAGS
    ]
    legacy_sids = [
        str(row.get("PUBCHEM_SID", "")).strip()
        for row in legacy_result_rows
    ]

    assert len(legacy_result_rows) == 102
    assert len(set(legacy_sids)) == 101
    assert sorted(requested - set(legacy_sids)) == []
    assert sorted(set(legacy_sids) - requested) == [""]
    assert Counter(legacy_sids)[""] == 2

    post_rows = parse_csv(fetch_pubchem_assay_csv_for_sids(883, sids))
    metadata_rows, result_rows = split_pubchem_csv_rows(post_rows)
    corrected_sids = [
        str(row.get("PUBCHEM_SID", "")).strip()
        for row in result_rows
    ]

    assert len(result_rows) == 100
    assert len(set(corrected_sids)) == 100
    assert sorted(requested - set(corrected_sids)) == []
    assert sorted(set(corrected_sids) - requested) == []
    assert all(corrected_sids)

    metadata_tag_counts = Counter(
        str(row.get("PUBCHEM_RESULT_TAG", "")).strip()
        for row in metadata_rows
    )
    assert metadata_tag_counts["RESULT_IS_ACTIVE_CONCENTRATION"] == 1
    assert metadata_tag_counts["RESULT_ATTR_CONC_MICROMOL"] == 1
    assert set(metadata_tag_counts).issubset(PUBCHEM_CSV_METADATA_TAGS)

    print(json.dumps({
        "aid": 883,
        "requested_sids": len(sids),
        "legacy_get_raw_rows": len(get_rows),
        "legacy_get_result_rows_under_old_assumption": len(
            legacy_result_rows
        ),
        "legacy_get_unique_sids_under_old_assumption": len(
            set(legacy_sids)
        ),
        "legacy_get_extra_sids_under_old_assumption": sorted(
            set(legacy_sids) - requested
        ),
        "legacy_get_duplicate_sids_under_old_assumption": {
            sid: count for sid, count in Counter(legacy_sids).items()
            if count > 1
        },
        "post_raw_rows": len(post_rows),
        "post_metadata_tag_counts": dict(metadata_tag_counts),
        "post_result_rows": len(result_rows),
        "post_unique_nonblank_sids": len(set(corrected_sids)),
        "post_missing_nonblank_sids": sorted(
            requested - set(corrected_sids)
        ),
        "post_extra_nonblank_sids": sorted(
            set(corrected_sids) - requested
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
