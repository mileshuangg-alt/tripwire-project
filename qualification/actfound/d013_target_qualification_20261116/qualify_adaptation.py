"""D013 target-adaptation qualification: resource measurement and three correctness checks.

Runs on CoreHPC in the pinned ActFound environment (Python 3.9, RDKit 2023.03.1,
torch 1.12.1) and refuses any other RDKit, because Morgan bit assignment is
version-sensitive and the foundation corpus was featurised under 2023.03.1.

Measurement only. It calls the unchanged upstream adaptation path
(actfound_production_train_v2.run_task with is_training_phase=False) and selects
nothing. Support is the cell's complete labelled training set; no cap, no change
to the anchor count k.

  smoke     one adaptation on a synthetic cell of fixture molecules, no CYP data.
            Runs first and must complete with non-zero peak GPU memory and
            finite positive phase times, or the session aborts.
  resource  wall time by phase and peak memory for the fixed partitions on one
            endpoint. Writes counts, timings and memory only: no predictions,
            no losses, no error metrics, so no evaluation result can inform a
            method choice.
  checks    the three correctness checks, which may write the diagnostics they
            require. It writes predictions, labels and bands but computes no
            ST-RAE: the frozen scorer needs Python >= 3.10 and cannot be
            imported here, so score_predictions.py scores this file separately.
              repeat   two seed values, identical cell inputs. Any difference is
                       GPU non-determinism, not a seed effect.
              offset   support/target labels on the pretraining scale
                       (-log10(nM)) versus pIC50. Post-adaptation predictions
                       must differ by exactly the 9-unit offset, within the
                       declared tolerance.
              scramble target labels replaced deterministically. Predictions read
                       no target label, so the bar is exact equality.

The ST-RAE consequence of a prediction difference is derived, not declared: the
scorer's per-molecule error clip(pred-upper,0)+clip(lower-pred,0) is 1-Lipschitz
in the prediction and its denominator is a prediction-independent constant-mean
baseline, so |dST-RAE| <= sum|dpred| / sum(baseline error).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))   # `python -I` implies -P: the script dir is not added
import count_cells as cc  # noqa: E402  sibling module, stdlib only

CONFIG_ARTIFACT = "D013TargetQualificationConfigV1"
SCRAMBLE_SHIFT = 1.0   # deterministic target-label replacement: reversed order plus this
SMOKE_SUPPORT = 40
SMOKE_TARGET = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rdkit_version() -> str:
    import rdkit
    return rdkit.__version__


def assert_pinned_env(required_rdkit: str, version_fn=rdkit_version) -> dict:
    v = version_fn()
    if v != required_rdkit:
        raise SystemExit(f"refusing to featurise under RDKit {v}; the corpus requires {required_rdkit}")
    return {"rdkit": v, "python": sys.version.split()[0]}


def read_config(path: Path, expected_sha: str) -> dict:
    if sha256(path) != expected_sha:
        raise SystemExit("qualification config SHA-256 mismatch")
    cfg = json.loads(path.read_text())
    if cfg.get("artifact_type") != CONFIG_ARTIFACT:
        raise SystemExit(f"not a {CONFIG_ARTIFACT}")
    return cfg


def endpoint_spec(cfg: dict, name: str) -> dict:
    for e in cfg["endpoints"]:
        if e["name"] == name:
            return e
    raise SystemExit(f"endpoint not in the qualification config: {name}")


MISSING = {"", "NA", "NaN", "nan", "None"}


def build_cell(manifest: dict, labels: dict, spec: dict) -> dict:
    """Support = labelled training molecules; target = labelled validation molecules.

    Order follows the manifest, so the assembled arrays are reproducible.
    """
    def pick(names):
        out = []
        for n in names:
            row = labels.get(n)
            if row is None:
                raise SystemExit(f"molecule absent from the label table: {n}")
            if row[spec["label"]].strip() in MISSING:
                continue
            out.append((n, float(row[spec["label"]]),
                        float(row[spec["conf_low"]]), float(row[spec["conf_high"]])))
        return out

    sup = pick(manifest["training_names"])
    tgt = pick(manifest["validation_names"])
    if not sup or not tgt:
        raise SystemExit("empty support or target set")
    if set(n for n, *_ in sup) & set(n for n, *_ in tgt):
        raise SystemExit("support and target overlap")
    return {
        "support_names": [n for n, *_ in sup],
        "target_names": [n for n, *_ in tgt],
        "support_y": [y for _, y, _, _ in sup],
        "target_y": [y for _, y, _, _ in tgt],
        "target_conf_low": [t[2] for t in tgt],
        "target_conf_high": [t[3] for t in tgt],
        "support_count": len(sup),
        "target_count": len(tgt),
    }


def synthetic_cell(n_support: int = SMOKE_SUPPORT, n_target: int = SMOKE_TARGET) -> dict:
    """A cell of fixture molecules for the smoke test. No CYP data of any kind."""
    n = n_support + n_target
    smiles = ["C" * (i % 8 + 1) + ("O" if i % 2 else "N") for i in range(n)]
    y = [5.0 + 0.05 * i for i in range(n)]
    return {
        "support_names": [f"SMOKE_S{i:03d}" for i in range(n_support)],
        "target_names": [f"SMOKE_T{i:03d}" for i in range(n_target)],
        "support_smiles": smiles[:n_support],
        "target_smiles": smiles[n_support:],
        "support_y": y[:n_support],
        "target_y": y[n_support:],
        "target_conf_low": [v - 0.5 for v in y[n_support:]],
        "target_conf_high": [v + 0.5 for v in y[n_support:]],
        "support_count": n_support,
        "target_count": n_target,
    }


def check_smoke(times: dict, peak_gpu_mib: float) -> None:
    """The smoke test must prove the path ran and the instrumentation recorded it."""
    import math
    if not times:
        raise SystemExit("smoke test recorded no phase times")
    for name, seconds in sorted(times.items()):
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise SystemExit(f"smoke test phase {name!r} recorded a non-positive time: {seconds}")
    if not math.isfinite(peak_gpu_mib) or peak_gpu_mib <= 0.0:
        raise SystemExit(f"smoke test recorded no GPU memory: {peak_gpu_mib}")


def featurize(smiles, featurizer) -> "object":
    """Feature rows must align with the molecules one-to-one.

    upstream_count_morgan_features silently drops a molecule RDKit cannot parse,
    which would misalign features and labels, so the count is asserted here.
    """
    x = featurizer(smiles)
    if len(x) != len(smiles):
        raise SystemExit(f"featuriser returned {len(x)} rows for {len(smiles)} molecules; "
                         "at least one SMILES failed to parse")
    return x


def assemble_arrays(cell: dict, x_support, x_target, offset: float = 0.0):
    import numpy as np
    x = np.concatenate([np.asarray(x_support, dtype=np.float32),
                        np.asarray(x_target, dtype=np.float32)], axis=0)
    y = np.asarray(list(cell["support_y"]) + list(cell["target_y"]), dtype=np.float32) + offset
    split = np.asarray([1] * cell["support_count"] + [0] * cell["target_count"], dtype=np.int64)
    return x, y, split


def scramble_target_labels(y, support_count: int):
    """Deterministic replacement of the target labels only."""
    import numpy as np
    out = np.array(y, dtype=np.float32, copy=True)
    tail = out[support_count:]
    out[support_count:] = tail[::-1] + SCRAMBLE_SHIFT
    return out


def strae_lipschitz_bound(preds_a, preds_b, y_true, conf_low, conf_high) -> dict:
    """Derived bound on the ST-RAE difference implied by a prediction difference."""
    import numpy as np
    a = np.asarray(preds_a, dtype=np.float64)
    b = np.asarray(preds_b, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(conf_low, dtype=np.float64)
    hi = np.asarray(conf_high, dtype=np.float64)
    mean = float(y.mean())
    baseline = float((np.clip(mean - hi, 0.0, None) + np.clip(lo - mean, 0.0, None)).sum())
    total = float(np.abs(a - b).sum())
    return {
        "sum_abs_pred_difference": total,
        "baseline_soft_threshold_error": baseline,
        "strae_difference_upper_bound": None if baseline == 0.0 else total / baseline,
    }


def verdict(name: str, observed_max: float, tolerance: float) -> dict:
    return {"check": name, "observed_max_abs_pred_difference": observed_max,
            "tolerance": tolerance, "passed": bool(observed_max <= tolerance)}


class Phases:
    """Wall time by phase, with the GPU synchronised at each boundary."""

    def __init__(self, sync=None, clock=time.perf_counter):
        self.sync, self.clock, self.times = sync, clock, {}

    def __call__(self, name):
        return _Phase(self, name)


class _Phase:
    def __init__(self, owner: Phases, name: str):
        self.owner, self.name = owner, name

    def __enter__(self):
        if self.owner.sync:
            self.owner.sync()
        self.t0 = self.owner.clock()
        return self

    def __exit__(self, *exc):
        if self.owner.sync:
            self.owner.sync()
        self.owner.times[self.name] = self.owner.clock() - self.t0
        return False


def host_peak_mib() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024.0 if sys.platform.startswith("linux") else ru / (1024.0 * 1024.0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("smoke", "resource", "checks"), required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--counts", type=Path, required=True, help="cell_counts.json from count_cells.py")
    p.add_argument("--counts-sha256", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--train-config", type=Path, required=True, help="ActFoundProductionConfigV2")
    p.add_argument("--train-config-sha256", required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--freeze-record", type=Path, required=True)
    p.add_argument("--label-table", type=Path, required=True)
    p.add_argument("--label-table-sha256", required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--upstream-dir", type=Path, required=True)
    p.add_argument("--endpoint", help="default: largest_endpoint_by_support from the counts file")
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)

    cfg = read_config(a.config, a.config_sha256)
    env = assert_pinned_env(cfg["required_rdkit"])
    for path, expected in ((a.counts, a.counts_sha256), (a.checkpoint, a.checkpoint_sha256),
                           (a.train_config, a.train_config_sha256),
                           (a.label_table, a.label_table_sha256)):
        if not path.is_file() or sha256(path) != expected:
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
    endpoint = a.endpoint or counts["largest_endpoint_by_support"]
    spec = endpoint_spec(cfg, endpoint)
    labels = cc.read_table(a.label_table)
    manifests = cc.verify_manifests(a.manifest_dir, a.freeze_record)
    train_cfg = json.loads(a.train_config.read_text())
    args = rv2.make_args(train_cfg["hyperparameters"])
    rv2.seed_like_upstream(args)   # as production does, including cudnn.benchmark

    ckpt = torch.load(a.checkpoint, map_location="cpu")
    if ckpt["upstream_commit"] != rv2.PINNED_UPSTREAM_COMMIT:
        raise SystemExit("checkpoint upstream_commit does not match the pinned commit")
    regressor = FCNReLUNormNetworkQSAR(input_shape=(2, args.dim_w), args=args, meta=True).cuda()
    regressor.load_state_dict(ckpt["regressor_state_dict"])
    inner_lrs = ckpt["inner_lrs"].cuda()
    epoch = int(ckpt["epoch"])

    def adapt(x, y, split):
        """One unchanged upstream adaptation; returns target predictions."""
        _, _, preds, split_t, _, support_losses, _ = rv2.run_task(
            regressor, inner_lrs, (x, y, split), args, epoch=epoch, is_training_phase=False)
        return preds.detach().cpu().numpy(), support_losses

    a.out_dir.mkdir(parents=True)
    partitions = list(cfg["qualification_partitions"])
    inputs = {"config_sha256": a.config_sha256, "counts_sha256": a.counts_sha256,
              "checkpoint_sha256": a.checkpoint_sha256, "train_config_sha256": a.train_config_sha256,
              "label_table_sha256": a.label_table_sha256,
              "script_sha256": sha256(Path(__file__)),
              "model_side_sha256": sha256(a.repo_root / "cyp" / "scripts" / "actfound_model_side.py"),
              "runner_sha256": sha256(a.repo_root / "cyp" / "scripts" / "actfound_production_train_v2.py")}

    cells, features = {}, {}
    for cell in (partitions if a.mode != "smoke" else []):
        if cell not in manifests:
            raise SystemExit(f"partition not in FREEZE_RECORD: {cell}")
        c = build_cell(manifests[cell], labels, spec)
        expected = counts["cells"][cell]["endpoints"][endpoint]
        if (c["support_count"], c["target_count"]) != (expected["support"], expected["target"]):
            raise SystemExit(f"{cell}: counts disagree with the counting step")
        cells[cell] = c
        ph = Phases(sync=torch.cuda.synchronize)
        with ph("featurize"):
            xs = featurize([labels[n]["SMILES"] for n in c["support_names"]], upstream_count_morgan_features)
            xt = featurize([labels[n]["SMILES"] for n in c["target_names"]], upstream_count_morgan_features)
        features[cell] = (xs, xt, ph.times["featurize"])

    if a.mode == "smoke":
        c = synthetic_cell()
        torch.cuda.reset_peak_memory_stats()
        ph = Phases(sync=torch.cuda.synchronize)
        with ph("featurize"):
            xs = featurize(c["support_smiles"], upstream_count_morgan_features)
            xt = featurize(c["target_smiles"], upstream_count_morgan_features)
        with ph("assemble"):
            x, y, split = assemble_arrays(c, xs, xt)
        with ph("adapt_total"):
            preds, support_losses = adapt(x, y, split)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        check_smoke(ph.times, peak)
        if len(preds) != c["target_count"]:
            raise SystemExit(f"smoke test returned {len(preds)} predictions for {c['target_count']} targets")
        out = {"artifact_type": "D013_target_smoke_v1", "synthetic_fixture_data_only": True,
               "environment": env, "checkpoint_epoch": epoch,
               "inner_steps": int(args.test_num_updates),
               "support_count": c["support_count"], "target_count": c["target_count"],
               "seconds": dict(ph.times), "peak_gpu_allocated_mib": peak,
               "peak_host_rss_mib": host_peak_mib(),
               "support_loss_steps": len(support_losses), "inputs": inputs}
        name = "smoke_summary.json"
    elif a.mode == "resource":
        rows = []
        for cell in partitions:
            xs, xt, feat_s = features[cell]
            torch.cuda.reset_peak_memory_stats()
            ph = Phases(sync=torch.cuda.synchronize)
            with ph("assemble"):
                x, y, split = assemble_arrays(cells[cell], xs, xt)
            with ph("adapt_total"):
                adapt(x, y, split)
            rows.append({
                "partition": cell, "endpoint": endpoint,
                "support_count": cells[cell]["support_count"],
                "target_count": cells[cell]["target_count"],
                "featurize_seconds": feat_s,
                "seconds": dict(ph.times),
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
                "peak_host_rss_mib": host_peak_mib(),
            })
        out = {"artifact_type": "D013_target_resource_qualification_v1",
               "contains_no_predictions_or_scores": True,
               "endpoint": endpoint, "environment": env, "checkpoint_epoch": epoch,
               "inner_steps": int(args.test_num_updates), "cells": rows, "inputs": inputs}
        name = "resource_summary.json"
    else:
        results = []
        for cell in partitions:
            xs, xt, _ = features[cell]
            c = cells[cell]
            x, y_pic50, split = assemble_arrays(c, xs, xt)
            offset = float(cfg["pretraining_scale_offset"])
            arms = {}
            for seed in cfg["seed_values"]:
                # Set every global seed the production runner sets, to the cell's seed
                # value. Nothing in the adaptation path draws from them; that is what
                # the repeat check measures.
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                arms[f"pic50_seed_{seed}"] = adapt(x, y_pic50, split)[0]
            reference = arms[f"pic50_seed_{cfg['seed_values'][0]}"]
            arms["pretraining_scale"] = adapt(x, y_pic50 + offset, split)[0]
            arms["target_labels_scrambled"] = adapt(
                x, scramble_target_labels(y_pic50, c["support_count"]), split)[0]

            tol = cfg["tolerances"]
            repeat_other = arms[f"pic50_seed_{cfg['seed_values'][1]}"]
            offset_corrected = arms["pretraining_scale"] - offset
            checks = [
                dict(verdict("repeat", float(np.max(np.abs(reference - repeat_other))),
                             tol["repeat_max_abs_pred_pic50"]),
                     seeds=cfg["seed_values"],
                     note="identical cell inputs; any difference is GPU non-determinism"),
                dict(verdict("offset", float(np.max(np.abs(reference - offset_corrected))),
                             tol["offset_max_abs_pred_pic50"]),
                     offset=offset,
                     note="declared screening threshold, not a proven error bound"),
                dict(verdict("scramble", float(np.max(np.abs(reference - arms["target_labels_scrambled"]))),
                             tol["scramble_max_abs_pred_pic50"]),
                     note="predictions read no target label; exact equality expected"),
            ]
            arms["pretraining_scale_corrected"] = offset_corrected
            results.append({
                "partition": cell, "endpoint": endpoint,
                "support_count": c["support_count"], "target_count": c["target_count"],
                "checks": checks,
                "target_names": list(c["target_names"]),
                "target_y": [float(v) for v in c["target_y"]],
                "target_conf_low": [float(v) for v in c["target_conf_low"]],
                "target_conf_high": [float(v) for v in c["target_conf_high"]],
                "strae_derived_bounds": {
                    "offset": strae_lipschitz_bound(reference, offset_corrected, c["target_y"],
                                                    c["target_conf_low"], c["target_conf_high"]),
                    "repeat": strae_lipschitz_bound(reference, repeat_other, c["target_y"],
                                                    c["target_conf_low"], c["target_conf_high"]),
                },
                "target_predictions": {k: [float(z) for z in v] for k, v in arms.items()},
            })
        out = {"artifact_type": "D013_target_adaptation_checks_v1",
               "strae_scored_separately": ("the frozen ST-RAE scorer requires Python >= 3.10 and cannot be "
                                           "imported in the pinned ActFound environment; score this file with "
                                           "score_predictions.py in a >= 3.10 environment"),
               "endpoint": endpoint, "environment": env, "checkpoint_epoch": epoch,
               "inner_steps": int(args.test_num_updates),
               "all_checks_passed": all(ch["passed"] for r in results for ch in r["checks"]),
               "cells": results, "inputs": inputs}
        name = "adaptation_checks.json"

    path = a.out_dir / name
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (a.out_dir / "SHA256SUMS.txt").write_text(f"{sha256(path)}  {name}\n")
    if a.mode == "smoke":
        print(json.dumps({k: out[k] for k in ("environment", "seconds", "peak_gpu_allocated_mib",
                                              "support_count", "target_count", "support_loss_steps")},
                         indent=2, sort_keys=True))
    elif a.mode == "resource":
        print(json.dumps({k: out[k] for k in ("endpoint", "environment", "cells")},
                         indent=2, sort_keys=True))
    else:
        print(json.dumps({"all_checks_passed": out["all_checks_passed"],
                          "cells": [{k: r[k] for k in ("partition", "support_count", "target_count",
                                                       "checks", "strae_derived_bounds")}
                                    for r in out["cells"]]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
