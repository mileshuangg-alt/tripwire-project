"""D013: split reconciliation v2 and production config v2 for the firewalled corpus.

Runs on CoreHPC in the pinned ActFound environment (Python 3.9), after the
firewalled corpus and its manifest exist. Implements accepted answer Q1:

  study split = pinned upstream train / valid split, minus every assay that the
  project loader does not emit from the firewalled corpus (e.g. fewer than 20
  usable molecules after the firewall), each drop-out logged with its reason.

Eligibility is loader-exact: actfound_model_side.iter_frozen_graph_qualified_assays
(the same code the runner uses) is run on the firewalled TSV. Config v2 is
config v1 with only these keys replaced: artifact_type, frozen input/dataset
path and SHA-256, train/val/validation assay lists, reconciliation paths and
SHA-256s, corpus manifest path and SHA-256, output_dir. Every other key
(hyperparameters, model/optimizer/scheduler configuration, seeds, runtime pins,
upstream commit and split source) is copied unchanged and asserted equal.

Finally the written config is validated with runner v2's own
validate_and_resolve_config.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

REPLACED_KEYS = {
    "artifact_type", "frozen_input_tsv", "frozen_input_sha256", "frozen_dataset_path", "frozen_dataset_sha256",
    "train_assays", "val_assays", "validation_assays",
    "split_reconciliation_summary_path", "split_reconciliation_summary_sha256",
    "split_reconciliation_per_id_path", "split_reconciliation_per_id_sha256",
    "corpus_manifest_path", "corpus_manifest_sha256", "output_dir",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reconcile(train: list, valid: list, emitted: Iterable[str]) -> tuple:
    """Return (kept_train, kept_valid, dropouts) preserving upstream order."""
    emitted = set(emitted)
    if set(train) & set(valid):
        raise ValueError("upstream train/valid overlap")
    kept_train = [a for a in train if a in emitted]
    kept_valid = [a for a in valid if a in emitted]
    dropouts = [(a, "train") for a in train if a not in emitted] + [(a, "validation") for a in valid if a not in emitted]
    return kept_train, kept_valid, dropouts


def build(a, emitted_fn: Callable[[Path], Iterable[str]], validate_fn: Optional[Callable[[Path], object]]) -> dict:
    manifest = json.loads(a.corpus_manifest.read_text())
    corpus_sha = sha256(a.corpus_tsv)
    if manifest.get("openadmet_firewall_applied") is not True:
        raise SystemExit("corpus manifest does not record an applied OpenADMET firewall")
    if manifest.get("output_sha256") != corpus_sha or Path(manifest.get("output_path", "")) != a.corpus_tsv:
        raise SystemExit("corpus manifest does not describe this corpus file")
    split_sha = sha256(a.upstream_split)
    v1 = json.loads(a.v1_config.read_text())
    if split_sha != v1["upstream_split_sha256"]:
        raise SystemExit("upstream split SHA-256 differs from config v1")
    split = json.loads(a.upstream_split.read_text())
    if v1["train_assays"] != split["train"] or v1["val_assays"] != split["valid"]:
        raise SystemExit("config v1 assay lists are not the pinned upstream train/valid split")
    if a.artifact_dir.exists():
        raise SystemExit(f"refusing to overwrite existing artifact directory: {a.artifact_dir}")

    kept_train, kept_valid, dropouts = reconcile(split["train"], split["valid"], emitted_fn(a.corpus_tsv))
    if not kept_train or not kept_valid:
        raise SystemExit("firewall left an empty train or validation split")
    a.artifact_dir.mkdir(parents=True)

    per_id = a.artifact_dir / "d013_split_reconciliation_per_id.tsv"
    with per_id.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["upstream_split_id", "partition", "classification", "mapped_assay_key"])
        for part, ids in (("train", kept_train), ("validation", kept_valid)):
            for i in ids:
                w.writerow([i, part, "emitted_by_loader_from_firewalled_corpus", i])
    drop = a.artifact_dir / "d013_split_dropouts.tsv"
    with drop.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["upstream_split_id", "partition", "reason"])
        for i, part in dropouts:
            w.writerow([i, part, "not_emitted_by_loader_from_firewalled_corpus"])
    summary = {
        "artifact_type": "ActFoundSplitReconciliationV1",
        "status": "complete",
        "method": "D013 Q1: pinned upstream train/valid minus assays not emitted by the project loader from the firewalled corpus",
        "frozen_dataset_path": str(a.corpus_tsv),
        "frozen_dataset_sha256": corpus_sha,
        "upstream_split_path": str(a.upstream_split),
        "upstream_split_sha256": split_sha,
        "upstream_train_ids": len(split["train"]),
        "upstream_validation_ids": len(split["valid"]),
        "train_split_ids": len(kept_train),
        "validation_split_ids": len(kept_valid),
        "train_mapped": len(kept_train),
        "validation_mapped": len(kept_valid),
        "mapped_total": len(kept_train) + len(kept_valid),
        "unresolved_total": 0,
        "ambiguous_total": 0,
        "train_validation_overlap_after_mapping": 0,
        "dropped_train": sum(1 for _, p in dropouts if p == "train"),
        "dropped_validation": sum(1 for _, p in dropouts if p == "validation"),
        "dropouts_path": str(drop),
        "dropouts_sha256": sha256(drop),
        "script_sha256": sha256(Path(__file__)),
    }
    summary_path = a.artifact_dir / "d013_split_reconciliation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    v2 = dict(v1)
    v2.update({
        "artifact_type": "ActFoundProductionConfigV2",
        "frozen_input_tsv": str(a.corpus_tsv), "frozen_input_sha256": corpus_sha,
        "frozen_dataset_path": str(a.corpus_tsv), "frozen_dataset_sha256": corpus_sha,
        "train_assays": kept_train, "val_assays": kept_valid, "validation_assays": kept_valid,
        "split_reconciliation_summary_path": str(summary_path), "split_reconciliation_summary_sha256": sha256(summary_path),
        "split_reconciliation_per_id_path": str(per_id), "split_reconciliation_per_id_sha256": sha256(per_id),
        "corpus_manifest_path": str(a.corpus_manifest), "corpus_manifest_sha256": sha256(a.corpus_manifest),
        "output_dir": str(a.training_output_dir),
    })
    for k in set(v1) | set(v2):
        if k not in REPLACED_KEYS and v1.get(k) != v2.get(k):
            raise SystemExit(f"config v2 changed a key it must copy unchanged: {k}")
    if Path(a.training_output_dir).exists():
        raise SystemExit(f"training output directory already exists: {a.training_output_dir}")
    config_path = a.artifact_dir / "actfound_chembl32_d013_config_v2.json"
    config_path.write_text(json.dumps(v2, indent=2, sort_keys=True) + "\n")
    if validate_fn is not None:
        validate_fn(config_path)
    names = [per_id.name, drop.name, summary_path.name, config_path.name]
    (a.artifact_dir / "SHA256SUMS.txt").write_text("".join(f"{sha256(a.artifact_dir / n)}  {n}\n" for n in names))
    out = {k: summary[k] for k in ("upstream_train_ids", "upstream_validation_ids", "train_split_ids", "validation_split_ids", "dropped_train", "dropped_validation", "frozen_dataset_sha256")}
    out["config_v2_sha256"] = sha256(config_path)
    print(json.dumps(out, indent=2, sort_keys=True))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-tsv", type=Path, required=True)
    p.add_argument("--corpus-manifest", type=Path, required=True)
    p.add_argument("--upstream-split", type=Path, required=True)
    p.add_argument("--v1-config", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--training-output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, str(a.repo_root / "cyp" / "scripts"))
    import actfound_model_side as ms  # pinned env
    import actfound_production_train_v2 as rv2

    def emitted(path: Path):
        return [x.assay_id for x in ms.iter_frozen_graph_qualified_assays(path)]

    build(a, emitted, rv2.validate_and_resolve_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
