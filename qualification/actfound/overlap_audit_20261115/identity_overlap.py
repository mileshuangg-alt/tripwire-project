"""Identity overlap between job 3066823's training corpus and protected OpenADMET sets.

Runs on the Mac in the frozen D012 identity environment (conda env
tripwire_cyp002: Python 3.12.14, RDKit 2026.03.6, InChI 1.07.3; no
project-added standardization). It refuses to run in any other environment.

Primary identity: full Standard InChIKey computed from the ChEMBL32 source
structure (compound_structures.canonical_smiles) for each molregno. The
ActFound TSV SMILES is a witness; disagreements are reported, not resolved.

Prohibited tiers, from the frozen bounded representation audit:
  development   OpenADMET train + validation (the D007 development universe)
  internal_test reported separately
  blinded       challenge_blinded_test, reported separately

Report only: no retraining, eligibility or checkpoint conclusions.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

FROZEN_ENV = {"python": "3.12.14", "rdkit": "2026.03.6", "inchi": "1.07.3"}
TIER_OF_PARTITION = {
    "train": "development",
    "validation": "development",
    "internal_test": "internal_test",
    "challenge_blinded_test": "blinded",
}
KNOWN_MATCHES = {  # §30, adjudicated 2026-11-15
    1370: "AQHHHDLHHXJYJD-UHFFFAOYSA-N",
    88100: "NCDNCNXCDXHOMX-XGKFQTDJSA-N",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def runtime_versions() -> dict[str, str]:
    import rdkit
    from rdkit.Chem import inchi

    return {"python": sys.version.split()[0], "rdkit": rdkit.__version__, "inchi": inchi.GetInchiVersion()}


def identity(smiles: str) -> tuple[str, str, str]:
    """Return (standard_inchi, full_standard_inchikey, status) with no added standardization."""
    if not smiles:
        return "", "", "missing_structure"
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", "", "parse_failure"
    std_inchi = inchi.MolToInchi(mol)
    if not std_inchi:
        return "", "", "inchi_failure"
    return std_inchi, inchi.InchiToInchiKey(std_inchi), "ok"


def _identity_pair(args: tuple[str, str]) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    source, witness = args
    return identity(source), identity(witness)


def load_prohibited(audit_path: Path) -> tuple[dict[str, dict], Counter]:
    """full key -> {tier set, standard_inchi set, names, partitions}."""
    keys: dict[str, dict] = {}
    partitions: Counter = Counter()
    with audit_path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["source_family"] != "openadmet" or row["identity_state"] != "represented":
                continue
            tier = TIER_OF_PARTITION[row["source_partition"]]
            entry = keys.setdefault(row["full_standard_inchikey"], {"tiers": set(), "inchis": set(), "names": set(), "partitions": set()})
            entry["tiers"].add(tier)
            entry["inchis"].add(row["standard_inchi"])
            entry["names"].add(row["source_id"])
            entry["partitions"].add(row["source_partition"])
            partitions[row["source_partition"]] += 1
    return keys, partitions


def d007_universe(manifest_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(glob.glob(str(manifest_dir / "rep_*_fold_*.json"))):
        m = json.loads(Path(path).read_text())
        names |= set(m["training_names"]) | set(m["validation_names"])
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", type=Path, required=True, help="unpacked CoreHPC export with SHA256SUMS.txt")
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-sha256", required=True)
    p.add_argument("--d007-manifests", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sample", type=int, default=0, help="first N molregnos only (timing/determinism); full run if 0")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)

    env = runtime_versions()
    if env != FROZEN_ENV:
        raise SystemExit(f"not the frozen identity environment: {env} != {FROZEN_ENV}")
    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")
    if file_sha256(a.audit) != a.audit_sha256:
        raise SystemExit("audit SHA-256 mismatch")
    sums = {}
    for line in (a.export_dir / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(None, 1)
        sums[name.strip().lstrip("*")] = digest
        if file_sha256(a.export_dir / name.strip().lstrip("*")) != digest:
            raise SystemExit(f"export file SHA-256 mismatch: {name}")

    started = time.time()
    prohibited, partition_counts = load_prohibited(a.audit)
    dev_names = d007_universe(a.d007_manifests)
    audit_dev_names = {n for e in prohibited.values() if "development" in e["tiers"] for n in e["names"]}
    if dev_names != audit_dev_names:
        raise SystemExit(f"D007 universe ({len(dev_names)}) != audit train+validation ({len(audit_dev_names)})")

    with gzip.open(a.export_dir / "molecules.tsv.gz", "rt", newline="") as f:
        molecules = list(csv.DictReader(f, delimiter="\t"))
    if a.sample:
        molecules = molecules[: a.sample]
    work = [(m["chembl32_canonical_smiles"], m["tsv_smiles"]) for m in molecules]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_identity_pair, work, chunksize=2000))
    else:
        results = [_identity_pair(w) for w in work]
    identity_seconds = time.time() - started

    a.out_dir.mkdir(parents=True)
    matched: dict[int, dict] = {}
    status = Counter()
    disagreements = 0
    with (a.out_dir / "molecule_identities.tsv").open("w", newline="") as out, (a.out_dir / "failures.tsv").open("w", newline="") as fail:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        fw = csv.writer(fail, delimiter="\t", lineterminator="\n")
        w.writerow(["molregno", "source_key", "witness_key", "source_vs_witness", "tiers", "openadmet_names", "openadmet_partitions", "membership_flags", "chembl32_key_witness"])
        fw.writerow(["molregno", "which", "status", "smiles"])
        for m, ((s_inchi, s_key, s_stat), (t_inchi, t_key, t_stat)) in zip(molecules, results):
            mr = int(m["molregno"])
            status[f"source_{s_stat}"] += 1
            status[f"witness_{t_stat}"] += 1
            if s_stat != "ok":
                fw.writerow([mr, "chembl32_source", s_stat, m["chembl32_canonical_smiles"]])
            if t_stat != "ok":
                fw.writerow([mr, "tsv_witness", t_stat, m["tsv_smiles"]])
            agree = "agree" if s_key and s_key == t_key else ("disagree" if s_key and t_key else "incomplete")
            disagreements += agree == "disagree"
            hit = prohibited.get(s_key) if s_key else None
            witness_hit = prohibited.get(t_key) if t_key else None
            if hit is None and witness_hit is not None:
                status["witness_only_match"] += 1
            if hit is not None:
                if s_inchi not in hit["inchis"]:
                    status["same_key_different_inchi"] += 1
                matched[mr] = {"key": s_key, "tiers": hit["tiers"], "names": hit["names"], "flags": set(m["membership_flags"].split(";"))}
            w.writerow([
                mr, s_key, t_key, agree,
                ";".join(sorted(hit["tiers"])) if hit else "",
                ";".join(sorted(hit["names"])) if hit else "",
                ";".join(sorted(hit["partitions"])) if hit else "",
                m["membership_flags"], m["chembl32_standard_inchi_key_witness"],
            ])

    for mr, key in KNOWN_MATCHES.items():
        if not a.sample and matched.get(mr, {}).get("key") != key:
            raise SystemExit(f"known §30 match not reproduced for molregno {mr}")

    # Row and assay counts for matched molecules, from the membership export.
    rows = Counter()
    assays: dict[str, set] = defaultdict(set)
    if matched:
        with gzip.open(a.export_dir / "membership.tsv.gz", "rt", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                mr = int(r["molregno"])
                if mr not in matched:
                    continue
                kept = "kept" if r["loader_status"] == "kept" else "dropped"
                for tier in matched[mr]["tiers"]:
                    rows[f"{tier}|{r['split']}|{kept}"] += 1
                    assays[f"{tier}|{r['split']}|{kept}"].add(r["assay_key"])

    def molecules_with(tier: str, flag: str) -> int:
        return sum(1 for v in matched.values() if tier in v["tiers"] and flag in v["flags"])

    tiers = ("development", "internal_test", "blinded")
    seen_dev_names = {n for v in matched.values() if "train_kept" in v["flags"] for n in v["names"] if n in dev_names}
    summary = {
        "artifact_type": "ActFound_3066823_training_corpus_identity_overlap_v1",
        "authority": "report_only_no_eligibility_or_checkpoint_authority",
        "scope": "sample" if a.sample else "full",
        "sample_size": a.sample or None,
        "inputs": {
            "export_sha256sums": sums,
            "audit": str(a.audit), "audit_sha256": a.audit_sha256,
            "d007_manifest_dir": str(a.d007_manifests), "d007_universe_names": len(dev_names),
            "openadmet_records_by_partition": dict(partition_counts),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "identity_environment": env,
        "molecules_processed": len(molecules),
        "identity_status_counts": dict(sorted(status.items())),
        "source_vs_witness_key_disagreements": disagreements,
        "matched_molecules_by_tier_and_membership": {
            t: {f: molecules_with(t, f) for f in ("train_kept", "train_dropped", "validation_kept", "validation_dropped", "none_kept", "none_dropped")}
            for t in tiers
        },
        "matched_rows_by_tier_split_loader": dict(sorted(rows.items())),
        "matched_assays_by_tier_split_loader": {k: len(v) for k, v in sorted(assays.items())},
        "d007_development_names_seen_in_loader_kept_training_rows": len(seen_dev_names),
        "runtime": {"identity_seconds": round(identity_seconds, 1), "total_seconds": round(time.time() - started, 1), "workers": a.workers},
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    names = ["molecule_identities.tsv", "failures.tsv", "summary.json"]
    (a.out_dir / "SHA256SUMS.txt").write_text("".join(f"{file_sha256(a.out_dir / n)}  {n}\n" for n in names))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
