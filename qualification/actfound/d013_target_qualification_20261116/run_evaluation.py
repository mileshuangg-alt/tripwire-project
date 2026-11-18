"""D013 stage 1: adapt the frozen checkpoint to every D007 cell and write predictions.

Runs on CoreHPC in the pinned ActFound environment (Python 3.9, RDKit 2023.03.1)
and refuses any other RDKit, because Morgan bit assignment is version-sensitive.

200 unique cell-endpoint adaptations: 50 frozen D007 partitions x 4 CYP
endpoints. Support is the cell's complete labelled training set; the upstream
adaptation path (run_task with is_training_phase=False) is called unchanged.
Validation labels are never passed into the inner loop's decisions: the source
path never reads them for predictions, and the qualification measured that
scrambling them leaves predictions bit-identical (Session 005 entry 47).

It writes predictions, labels and bands, and computes NO score of any kind.
Scoring is stage 2 (score_evaluation.py) in a Python >= 3.10 environment,
because the frozen ST-RAE scorer cannot be imported here (D013 Amendment 4
item 5). The three seed slots are expanded at scoring time, not here: this
stage computes each cell-endpoint exactly once.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))   # `python -I` implies -P: the script dir is not added
import count_cells as cc  # noqa: E402
import qualify_adaptation as qa  # noqa: E402

CONFIG_ARTIFACT = "D013EvaluationConfigV1"


def read_eval_config(path: Path, expected_sha: str) -> dict:
    if qa.sha256(path) != expected_sha:
        raise SystemExit("evaluation config SHA-256 mismatch")
    cfg = json.loads(path.read_text())
    if cfg.get("artifact_type") != CONFIG_ARTIFACT:
        raise SystemExit(f"not a {CONFIG_ARTIFACT}")
    return cfg


def endpoint_order(eval_cfg: dict, qual_cfg: dict) -> list:
    """Endpoint names in the frozen task order, cross-checked against the label columns."""
    by_label = {e["label"]: e for e in qual_cfg["endpoints"]}
    order = []
    for label in eval_cfg["endpoint_task_order"]:
        if label not in by_label:
            raise SystemExit(f"task order names a label absent from the qualification config: {label}")
        order.append(by_label[label])
    if len(order) != int(eval_cfg["expected_endpoints"]):
        raise SystemExit(f"task order has {len(order)} endpoints; expected "
                         f"{eval_cfg['expected_endpoints']}")
    return order


class FeatureCache:
    """One fingerprint row per distinct SMILES, reused across cells and endpoints.

    Molecules recur across the 50 partitions, so this removes most of the
    featurisation cost. It is a cache only: the same SMILES always yields the
    same row, and every row still comes from the pinned featuriser.
    """

    def __init__(self, featurizer):
        self.featurizer, self.rows, self.misses, self.hits = featurizer, {}, 0, 0

    def matrix(self, smiles):
        import numpy as np
        unseen = [s for s in dict.fromkeys(smiles) if s not in self.rows]
        if unseen:
            built = qa.featurize(unseen, self.featurizer)
            for s, row in zip(unseen, built):
                self.rows[s] = np.asarray(row, dtype=np.float32)
            self.misses += len(unseen)
        self.hits += len(smiles) - len(unseen)
        return np.stack([self.rows[s] for s in smiles], axis=0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--qualification-config", type=Path, required=True)
    p.add_argument("--counts", type=Path, required=True)
    p.add_argument("--counts-sha256", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--train-config", type=Path, required=True)
    p.add_argument("--train-config-sha256", required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--freeze-record", type=Path, required=True)
    p.add_argument("--label-table", type=Path, required=True)
    p.add_argument("--label-table-sha256", required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--upstream-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)

    eval_cfg = read_eval_config(a.config, a.config_sha256)
    qual_cfg = qa.read_config(a.qualification_config, eval_cfg["qualification_config_sha256"])
    env = qa.assert_pinned_env(qual_cfg["required_rdkit"])
    for path, expected in ((a.counts, a.counts_sha256), (a.checkpoint, a.checkpoint_sha256),
                           (a.train_config, a.train_config_sha256),
                           (a.label_table, a.label_table_sha256)):
        if not path.is_file() or qa.sha256(path) != expected:
            raise SystemExit(f"input missing or SHA-256 mismatch: {path}")
    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {a.out_dir}")

    sys.path.insert(0, str(a.repo_root / "cyp" / "scripts"))
    sys.path.insert(0, str(a.upstream_dir))
    import numpy as np
    import torch
    import actfound_production_train_v2 as rv2
    from actfound_model_side import upstream_count_morgan_features
    from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR

    counts = json.loads(a.counts.read_text())
    labels = cc.read_table(a.label_table)
    manifests = cc.verify_manifests(a.manifest_dir, a.freeze_record)
    if len(manifests) != int(eval_cfg["expected_partitions"]):
        raise SystemExit(f"{len(manifests)} partitions; expected {eval_cfg['expected_partitions']}")
    endpoints = endpoint_order(eval_cfg, qual_cfg)

    train_cfg = json.loads(a.train_config.read_text())
    args = rv2.make_args(train_cfg["hyperparameters"])
    rv2.seed_like_upstream(args)
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    if ckpt["upstream_commit"] != rv2.PINNED_UPSTREAM_COMMIT:
        raise SystemExit("checkpoint upstream_commit does not match the pinned commit")
    regressor = FCNReLUNormNetworkQSAR(input_shape=(2, args.dim_w), args=args, meta=True).cuda()
    regressor.load_state_dict(ckpt["regressor_state_dict"])
    inner_lrs = ckpt["inner_lrs"].cuda()
    epoch = int(ckpt["epoch"])
    cache = FeatureCache(upstream_count_morgan_features)

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    cells = []
    for name in sorted(manifests):
        m = manifests[name]
        for spec in endpoints:
            c = qa.build_cell(m, labels, spec)
            expected = counts["cells"][name]["endpoints"][spec["name"]]
            if (c["support_count"], c["target_count"]) != (expected["support"], expected["target"]):
                raise SystemExit(f"{name}/{spec['name']}: counts disagree with the counting step")
            ph = qa.Phases(sync=torch.cuda.synchronize)
            with ph("featurize"):
                xs = cache.matrix([labels[n]["SMILES"] for n in c["support_names"]])
                xt = cache.matrix([labels[n]["SMILES"] for n in c["target_names"]])
            with ph("adapt"):
                x, y, split = qa.assemble_arrays(c, xs, xt)
                _, _, preds, _, _, support_losses, _ = rv2.run_task(
                    regressor, inner_lrs, (x, y, split), args, epoch=epoch, is_training_phase=False)
                preds = preds.detach().cpu().numpy()
            if len(preds) != c["target_count"]:
                raise SystemExit(f"{name}/{spec['name']}: {len(preds)} predictions for "
                                 f"{c['target_count']} targets")
            cells.append({
                "partition": name, "repetition": int(m["repetition"]), "fold": int(m["fold"]),
                "endpoint": spec["name"], "endpoint_label": spec["label"],
                "training_seeds": list(m["training_seeds"]),
                "support_count": c["support_count"], "target_count": c["target_count"],
                "target_names": list(c["target_names"]),
                "target_y": [float(v) for v in c["target_y"]],
                "target_conf_low": [float(v) for v in c["target_conf_low"]],
                "target_conf_high": [float(v) for v in c["target_conf_high"]],
                "target_predictions": [float(v) for v in preds],
                "inner_support_loss_steps": len(support_losses),
                "seconds": dict(ph.times),
            })
    if len(cells) != int(eval_cfg["expected_partitions"]) * int(eval_cfg["expected_endpoints"]):
        raise SystemExit(f"{len(cells)} cell-endpoint results; expected "
                         f"{eval_cfg['expected_partitions']} x {eval_cfg['expected_endpoints']}")

    out = {
        "artifact_type": "D013_cyp_adaptation_predictions_v1",
        "contains_no_scores": True,
        "scored_separately_by": "score_evaluation.py in a Python >= 3.10 environment",
        "unique_adaptations": len(cells),
        "seed_slots_expanded_at_scoring": int(eval_cfg["expected_seed_slots"]),
        "environment": env, "checkpoint_epoch": epoch,
        "inner_steps": int(args.test_num_updates),
        "endpoint_task_order": list(eval_cfg["endpoint_task_order"]),
        "feature_cache": {"distinct_molecules_featurised": cache.misses, "reused_rows": cache.hits},
        "runtime": {"seconds": round(time.time() - started, 1),
                    "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                    "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
                    "peak_host_rss_mib": qa.host_peak_mib()},
        "inputs": {"config_sha256": a.config_sha256,
                   "qualification_config_sha256": eval_cfg["qualification_config_sha256"],
                   "counts_sha256": a.counts_sha256, "checkpoint_sha256": a.checkpoint_sha256,
                   "train_config_sha256": a.train_config_sha256,
                   "label_table_sha256": a.label_table_sha256,
                   "freeze_record_sha256": qa.sha256(a.freeze_record),
                   "script_sha256": qa.sha256(Path(__file__)),
                   "qualify_adaptation_sha256": qa.sha256(HERE / "qualify_adaptation.py"),
                   "model_side_sha256": qa.sha256(a.repo_root / "cyp" / "scripts" / "actfound_model_side.py"),
                   "runner_sha256": qa.sha256(a.repo_root / "cyp" / "scripts" / "actfound_production_train_v2.py")},
        "cells": cells,
    }
    a.out_dir.mkdir(parents=True)
    path = a.out_dir / "cyp_adaptation_predictions.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (a.out_dir / "SHA256SUMS.txt").write_text(f"{qa.sha256(path)}  {path.name}\n")
    print(json.dumps({k: out[k] for k in ("unique_adaptations", "environment", "checkpoint_epoch",
                                          "feature_cache", "runtime")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
