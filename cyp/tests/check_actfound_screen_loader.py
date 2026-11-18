"""Real-data screening-loader qualification; no model training."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cyp/scripts"))

from actfound_model_side import (
    ActFoundFrozenMetaDataset,
    load_adjudicated_screen_assays,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"STOP: output already exists: {args.output}")

    assays = load_adjudicated_screen_assays(
        args.inputs / "screen_full.csv",
        args.inputs / "adjudication.tsv",
        args.inputs / "cluster_adjudication.tsv",
        ROOT / "qualification/actfound/screen_loader_20261117/contract.json",
        featurize=True,
    )

    expected = (
        "screen:CYP1A2",
        "screen:CYP2C9",
        "screen:CYP2D6",
        "screen:CYP3A4",
    )
    if tuple(a.assay_id for a in assays) != expected:
        raise AssertionError("Four enzyme task identities/order differ")

    population = set()
    records = []

    for assay in assays:
        population.update(assay.molecule_ids)
        if assay.x is None or len(assay.x) != len(assay.y):
            raise AssertionError(f"Feature/label alignment failed: {assay.assay_id}")
        if not np.isfinite(assay.x).all():
            raise AssertionError(f"Non-finite features: {assay.assay_id}")
        if not np.isfinite(assay.y).all():
            raise AssertionError(f"Non-finite observed labels: {assay.assay_id}")
        if not np.isfinite(assay.observation_weights).all():
            raise AssertionError(f"Non-finite weights: {assay.assay_id}")
        if not (assay.observation_weights > 0).all():
            raise AssertionError(f"Non-positive weights: {assay.assay_id}")

        dataset = ActFoundFrozenMetaDataset(
            [assay], val_assays={assay.assay_id}
        )
        legacy = dataset.get_task_split(
            0, test_sup_num=0.5, is_test=True, rand_seed=919
        )
        weighted = dataset.get_task_split(
            0, test_sup_num=0.5, is_test=True, rand_seed=919,
            include_metadata=True,
        )
        for original, with_metadata in zip(legacy, weighted[:3]):
            np.testing.assert_array_equal(original, with_metadata)

        x, y, split, weights, ids = weighted
        lookup = {
            name: index for index, name in enumerate(assay.molecule_ids)
        }
        if len(lookup) != len(assay.molecule_ids):
            raise AssertionError(f"Duplicate molecule IDs: {assay.assay_id}")

        for index, name in enumerate(ids):
            original = lookup[name]
            np.testing.assert_array_equal(x[index], assay.x[original])
            np.testing.assert_equal(y[index], assay.y[original])
            np.testing.assert_equal(
                weights[index], assay.observation_weights[original]
            )

        support = int(np.count_nonzero(split))
        query = int(len(split) - support)
        if support < 2 or query < 2:
            raise AssertionError(f"Empty/insufficient episode: {assay.assay_id}")

        record = {
            "assay_id": assay.assay_id,
            "observed_rows": len(assay.y),
            "feature_shape": list(assay.x.shape),
            "support_rows": support,
            "query_rows": query,
            "metadata_alignment": "PASS",
            "legacy_episode_equality": "PASS",
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    if len(population) != 6947:
        raise AssertionError(f"Population mismatch: {len(population)}")

    result = {
        "status": "PASS",
        "eligible_compounds": len(population),
        "tasks": records,
        "scope": "Real loader, input hashes, eligibility, features and episode metadata",
        "not_tested": [
            "model forward/backward through production runner",
            "staged checkpoint transfer",
            "normalization and optimizer resets",
            "training or evaluation performance",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"PASS: saved {args.output}", flush=True)

if __name__ == "__main__":
    main()
