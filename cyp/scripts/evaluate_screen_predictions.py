"""Descriptive evaluation of existing screen-response predictions (Tripwire-CYP-010, D026).

Authorizes no run by itself; the project owner runs it, locally, on CPU. No model is loaded, no
inference is rerun, and nothing is fitted or tuned. It writes only into a NEW output directory.

Scope (D026 and Amendment 1): the existing predictions from screen_predict.py on the 208-molecule
input set, restricted to the 197 molecules left after setting aside the 11 PXR aux-only molecules
whose group touches official-test structures (eligibility reason
aux_only_group_contains_test_structure). The 11 stay identifiable in the mapping file, are removed
BEFORE any label is looked up, and are not scored. Protected CYP tiers, the PXR holdout role and
bridging molecules are not in the set and are not reintroduced.

Outputs evaluated: the four CYP screen outputs (native log2fc) against the screening-file
log2fc_estimate of the matching enzyme, and the PXR 8.25 uM three-seed mean against the
single-concentration log2_fc_estimate at 8.251e-06 M. Strata: all 197, PXR primary_validation (160)
and PXR aux_only (37), reported separately.

Metrics (native units): MAE = mean |prediction - label|; Spearman rank correlation (average ranks;
no p-value and no significance claim); n = molecules with a finite prediction and label. A constant
baseline MAE = mean |label - c| with c from existing TRAINING-ONLY data:
  * PXR c = the recorded stage-1 training-label mean at 8.25 uM (pins expected.scaler_mean[0]);
  * CYP c per enzyme = the unweighted mean of observed native log2fc over the screening-training
    partition (partition train of the existing split file); the script checks that none of the
    evaluation molecules is in that split and that no evaluation label enters the constant.
A constant has no meaningful Spearman. The CYP training constant is an unweighted floor, not the
MAE-optimal constant, and the screening model trained with inverse-standard-error weights.

Join keys are explicit and are not assumed interchangeable: predictions are matched to the mapping by
input_id = mapping OCNT_ID (checked on literal text, with the SMILES also checked); CYP labels by the
mapping's cyp_molecule_name; PXR labels by OCNT_ID. pxr_row_id is not used.

Limits: descriptive only. Molecules share scaffold groups and are not independent; no confidence
intervals. This is not challenge performance, not validation of the full potency-plus-screen
report, and not evidence of confirmed PXR-specific activity. No threshold is chosen or tuned.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import tarfile
from pathlib import Path

ENZYMES = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
CYP_OUTPUTS = {e: f"{e}_screen_log2fc" for e in ENZYMES}
PXR_OUTPUT = "PXR_screen_8p25uM_log2fc_mean3"
OUTPUTS = tuple(ENZYMES) + ("PXR_8p25uM",)
SET_ASIDE_REASON = "aux_only_group_contains_test_structure"
STRATA = ("all", "primary_validation", "aux_only")
PXR_8P25_CONCENTRATION = 8.251e-06
SCREEN_MEMBER = "inputs/screen_full.csv"
EXPECTED = {
    "rows": 208, "set_aside": 11, "evaluated": 197,
    "roles": {"primary_validation": 160, "aux_only": 37},
    "tier_role": {
        ("train", "primary_validation"): 137, ("train", "aux_only"): 32,
        ("validation", "primary_validation"): 23, ("validation", "aux_only"): 5,
    },
    "cyp_train_molecules": 6252,
}


# ----------------------------------------------------------------------- helpers
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_verified_text(path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Input hash mismatch: {path} (actual {actual}, expected {expected})")
    return Path(path).read_text(encoding="utf-8")


def read_tar_member(tar_path, member: str, expected: str) -> str:
    with tarfile.open(tar_path, "r:gz") as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"Tar member missing: {member}")
        data = handle.read()
    if sha256_bytes(data) != expected:
        raise ValueError(f"Tar member hash mismatch: {member}")
    return data.decode("utf-8")


def dict_rows(text: str, delimiter: str = ","):
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


def num(text):
    """Finite float from text, else None."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# --------------------------------------------------------------------- crosswalk
def crosswalk_checks(pred_rows, mapping_rows) -> dict:
    """Coverage, duplicates and recorded structure checks. Any failure stops the run."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    pred_ids = [r["input_id"] for r in pred_rows]
    map_ids = [r["OCNT_ID"] for r in mapping_rows]
    if len(set(pred_ids)) != len(pred_ids):
        raise ValueError("Duplicate input_id in the predictions.")
    if len(set(map_ids)) != len(map_ids):
        raise ValueError("Duplicate OCNT_ID in the mapping.")
    if set(pred_ids) != set(map_ids):
        raise ValueError("Prediction ids differ from the mapping ids.")
    row_ids = [r["pxr_row_id"] for r in mapping_rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Duplicate pxr_row_id in the mapping.")
    by_id = {r["OCNT_ID"]: r for r in mapping_rows}
    for r in pred_rows:
        if r["status"] != "ok":
            raise ValueError(f"Prediction row is not ok: {r['input_id']}")
        m = by_id[r["input_id"]]
        if r["input_smiles"] != m["cyp_smiles"]:
            raise ValueError(f"Prediction SMILES differs from the mapping: {r['input_id']}")
        if m["cyp_molecule_name"] != m["OCNT_ID"]:
            raise ValueError(f"cyp_molecule_name differs from OCNT_ID: {m['OCNT_ID']}")
        if m["pxr_role"] == "aux_only" and m["pxr_row_id"] != m["OCNT_ID"]:
            raise ValueError(f"Aux-only pxr_row_id differs from OCNT_ID: {m['OCNT_ID']}")
        if m["pxr_role"] != "aux_only" and m["pxr_row_id"] == m["OCNT_ID"]:
            raise ValueError(f"Primary-role pxr_row_id equals OCNT_ID: {m['OCNT_ID']}")
        if (m["smiles_string_equal"] == "True") != (m["cyp_smiles"] == m["pxr_source_smiles"]):
            raise ValueError(f"smiles_string_equal flag disagrees with the strings: {m['OCNT_ID']}")
        a, b = Chem.MolFromSmiles(m["cyp_smiles"]), Chem.MolFromSmiles(m["pxr_source_smiles"])
        if a is None or b is None or Chem.MolToSmiles(a) != Chem.MolToSmiles(b):
            raise ValueError(f"Recorded structure match fails: {m['OCNT_ID']}")
    return {
        "rows": len(pred_rows), "unique_ids": len(set(pred_ids)), "all_status_ok": True,
        "smiles_equal_to_mapping": True, "cyp_molecule_name_equals_ocnt_id": True,
        "unique_pxr_row_id": True, "recorded_structure_matches": True,
    }


def split_evaluation_set(mapping_rows, expected=EXPECTED) -> dict:
    """Set aside the 11 by eligibility reason, before any label lookup; assert the recorded counts."""
    if len(mapping_rows) != expected["rows"]:
        raise ValueError(f"Mapping has {len(mapping_rows)} rows, expected {expected['rows']}.")
    aside = [r for r in mapping_rows if r["pxr_eligibility_reason"] == SET_ASIDE_REASON]
    keep = [r for r in mapping_rows if r["pxr_eligibility_reason"] != SET_ASIDE_REASON]
    if len(aside) != expected["set_aside"] or len(keep) != expected["evaluated"]:
        raise ValueError(f"Set-aside {len(aside)} / evaluated {len(keep)} differ from the recorded counts.")
    roles = {}
    for r in keep:
        roles[r["pxr_role"]] = roles.get(r["pxr_role"], 0) + 1
    if roles != expected["roles"]:
        raise ValueError(f"PXR role counts {roles} differ from {expected['roles']}.")
    table = {}
    for r in keep:
        key = (r["cyp_tiers"], r["pxr_role"])
        table[key] = table.get(key, 0) + 1
    if table != expected["tier_role"]:
        raise ValueError(f"CYP tier x PXR role counts {table} differ from the recorded counts.")
    return {"evaluated": keep, "set_aside_count": len(aside), "roles": roles, "tier_role": table}


# ------------------------------------------------------------------------ labels
def load_cyp_labels(screen_text: str, name_by_id: dict) -> dict:
    """OCNT_ID -> {enzyme: label} for exactly the requested molecules; missing or duplicate stops."""
    wanted = {name: ocnt for ocnt, name in name_by_id.items()}
    labels = {ocnt: {} for ocnt in name_by_id}
    for row in dict_rows(screen_text):
        ocnt = wanted.get(row["Molecule_Name"])
        if ocnt is None or row["enzyme"] not in ENZYMES:
            continue
        if row["enzyme"] in labels[ocnt]:
            raise ValueError(f"Duplicate CYP label row: {ocnt} {row['enzyme']}")
        value = num(row["log2fc_estimate"])
        if value is None:
            raise ValueError(f"Missing CYP label: {ocnt} {row['enzyme']}")
        labels[ocnt][row["enzyme"]] = value
    for ocnt, values in labels.items():
        if set(values) != set(ENZYMES):
            raise ValueError(f"CYP labels incomplete for {ocnt}: {sorted(values)}")
    return labels


def load_pxr_labels(single_text: str, ids) -> dict:
    wanted = set(ids)
    labels = {}
    for row in dict_rows(single_text):
        if row["compound_class"] != "Library" or row["OCNT_ID"] not in wanted:
            continue
        if abs(float(row["concentration_M"]) - PXR_8P25_CONCENTRATION) > 1e-3 * PXR_8P25_CONCENTRATION:
            continue
        if row["OCNT_ID"] in labels:
            raise ValueError(f"Duplicate PXR 8.25 uM label row: {row['OCNT_ID']}")
        value = num(row["log2_fc_estimate"])
        if value is None:
            raise ValueError(f"Missing PXR label: {row['OCNT_ID']}")
        labels[row["OCNT_ID"]] = value
    if set(labels) != wanted:
        raise ValueError(f"PXR 8.25 uM labels missing for {len(wanted - set(labels))} molecules.")
    return labels


# ---------------------------------------------------------------------- baselines
def read_train_split(split_text: str, expected_train: int) -> tuple:
    """(train ids, all split ids). The file has CRLF line endings; csv handles them."""
    reader = csv.reader(io.StringIO(split_text, newline=""))
    header = [h.strip() for h in next(reader)]
    if header != ["molecule_id", "cluster_id", "partition"]:
        raise ValueError(f"Unexpected split header: {header}")
    train, seen = set(), set()
    for record in reader:
        name, partition = record[0].strip(), record[2].strip()
        if name in seen:
            raise ValueError(f"Duplicate molecule in the split: {name}")
        seen.add(name)
        if partition == "train":
            train.add(name)
    if len(train) != expected_train:
        raise ValueError(f"Train partition has {len(train)} molecules, expected {expected_train}.")
    return train, seen


def cyp_training_constants(screen_text: str, train_ids: set, split_ids: set, evaluation_names) -> dict:
    """Per-enzyme unweighted mean of observed native log2fc over the screening-training partition."""
    overlap = set(evaluation_names) & split_ids
    if overlap:
        raise ValueError(f"{len(overlap)} evaluation molecules are in the screening split; refusing.")
    sums = {e: 0.0 for e in ENZYMES}
    counts = {e: 0 for e in ENZYMES}
    for row in dict_rows(screen_text):
        if row["Molecule_Name"] not in train_ids or row["enzyme"] not in ENZYMES:
            continue
        value = num(row["log2fc_estimate"])
        if value is None:
            continue
        sums[row["enzyme"]] += value
        counts[row["enzyme"]] += 1
    if any(counts[e] == 0 for e in ENZYMES):
        raise ValueError("A CYP enzyme has no observed training label.")
    return {e: {"constant": sums[e] / counts[e], "n_train_labels": counts[e]} for e in ENZYMES}


def pxr_training_constant(pins: dict) -> dict:
    expected = pins["expected"]
    if expected["head_order"][0] != "log2fc_8p25uM":
        raise ValueError("Pinned PXR head 0 is not the 8.25 uM head.")
    return {"constant": float(expected["scaler_mean"][0]),
            "source": "pxr_stage1_pins.json expected.scaler_mean[0]: mean of the 8,763 stage-1 training labels at 8.25 uM"}


# ------------------------------------------------------------------------ metrics
def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation with average ranks; None if undefined (n < 3 or a constant input)."""
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = average_ranks(list(x)), average_ranks(list(y))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy)


def metrics_for(predictions, labels, constant: float) -> dict:
    pairs = [(p, y) for p, y in zip(predictions, labels) if p is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "mae": None, "spearman": None, "baseline_constant": constant,
                "baseline_mae": None, "mae_over_baseline": None}
    mae = sum(abs(p - y) for p, y in pairs) / n
    base = sum(abs(y - constant) for _, y in pairs) / n
    return {
        "n": n, "mae": mae, "spearman": spearman([p for p, _ in pairs], [y for _, y in pairs]),
        "baseline_constant": constant, "baseline_mae": base,
        "mae_over_baseline": None if base == 0 else mae / base,
    }


def evaluate(evaluated_rows, pred_by_id, cyp_labels, pxr_labels, cyp_constants, pxr_constant) -> dict:
    results = {}
    for stratum in STRATA:
        rows = [r for r in evaluated_rows if stratum == "all" or r["pxr_role"] == stratum]
        block = {}
        for e in ENZYMES:
            p = [num(pred_by_id[r["OCNT_ID"]][CYP_OUTPUTS[e]]) for r in rows]
            y = [cyp_labels[r["OCNT_ID"]][e] for r in rows]
            block[e] = metrics_for(p, y, cyp_constants[e]["constant"])
        p = [num(pred_by_id[r["OCNT_ID"]][PXR_OUTPUT]) for r in rows]
        y = [pxr_labels[r["OCNT_ID"]] for r in rows]
        block["PXR_8p25uM"] = metrics_for(p, y, pxr_constant["constant"])
        results[stratum] = block
    return results


# -------------------------------------------------------------------------- output
def write_outputs(out_dir, evaluated_rows, pred_by_id, cyp_labels, pxr_labels, results, report) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    paths = {name: out_dir / name for name in (
        "evaluation_metrics.csv", "evaluation_rows.csv", "crosswalk_table.csv", "evaluation_report.json")}
    with open(paths["evaluation_metrics.csv"], "x", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["stratum", "output", "n", "mae", "spearman", "baseline_constant", "baseline_mae", "mae_over_baseline"])
        for stratum in STRATA:
            for output in OUTPUTS:
                m = results[stratum][output]
                w.writerow([stratum, output, m["n"]] + ["" if m[k] is None else repr(m[k]) for k in (
                    "mae", "spearman", "baseline_constant", "baseline_mae", "mae_over_baseline")])
    with open(paths["evaluation_rows.csv"], "x", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, lineterminator="\n")
        header = ["OCNT_ID", "pxr_role", "cyp_tiers"] + [CYP_OUTPUTS[e] for e in ENZYMES] + [PXR_OUTPUT] \
            + [f"label_{e}" for e in ENZYMES] + ["label_PXR_8p25uM"]
        w.writerow(header)
        for r in evaluated_rows:
            i = r["OCNT_ID"]
            w.writerow([i, r["pxr_role"], r["cyp_tiers"]] + [pred_by_id[i][CYP_OUTPUTS[e]] for e in ENZYMES]
                       + [pred_by_id[i][PXR_OUTPUT]] + [repr(cyp_labels[i][e]) for e in ENZYMES] + [repr(pxr_labels[i])])
    with open(paths["crosswalk_table.csv"], "x", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["cyp_tier", "pxr_role", "n"])
        for (tier, role), n in sorted(report["tier_role_counts"].items()):
            w.writerow([tier, role, n])
    report = {**report, "tier_role_counts": {f"{t}|{r}": n for (t, r), n in sorted(report["tier_role_counts"].items())}}
    with open(paths["evaluation_report.json"], "x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {name: {"path": str(p), "sha256": sha256_file(p)} for name, p in paths.items()}


def run(args) -> dict:
    pred_text = read_verified_text(args.predictions, args.predictions_sha256)
    mapping_text = read_verified_text(args.mapping, args.mapping_sha256)
    screen_text = read_tar_member(args.screen_tarball, SCREEN_MEMBER, args.screen_sha256)
    single_text = read_verified_text(args.pxr_single, args.pxr_single_sha256)
    pins_text = read_verified_text(args.pxr_pins, args.pxr_pins_sha256)
    split_text = read_verified_text(args.cyp_train_split, args.cyp_train_split_sha256)

    pred_rows, mapping_rows = dict_rows(pred_text), dict_rows(mapping_text)
    crosswalk = crosswalk_checks(pred_rows, mapping_rows)
    selection = split_evaluation_set(mapping_rows, EXPECTED)
    evaluated = selection["evaluated"]
    pred_by_id = {r["input_id"]: r for r in pred_rows}
    name_by_id = {r["OCNT_ID"]: r["cyp_molecule_name"] for r in evaluated}

    train_ids, split_ids = read_train_split(split_text, EXPECTED["cyp_train_molecules"])
    cyp_constants = cyp_training_constants(screen_text, train_ids, split_ids, name_by_id.values())
    pxr_constant = pxr_training_constant(json.loads(pins_text))
    cyp_labels = load_cyp_labels(screen_text, name_by_id)
    pxr_labels = load_pxr_labels(single_text, [r["OCNT_ID"] for r in evaluated])
    results = evaluate(evaluated, pred_by_id, cyp_labels, pxr_labels, cyp_constants, pxr_constant)

    import rdkit

    report = {
        "tool": "cyp/scripts/evaluate_screen_predictions.py",
        "decision": "D026 and Amendment 1",
        "script_sha256": sha256_file(__file__),
        "inputs": {
            "predictions": {"path": str(Path(args.predictions).resolve()), "sha256": args.predictions_sha256},
            "mapping": {"path": str(Path(args.mapping).resolve()), "sha256": args.mapping_sha256},
            "screen_member": {"tarball": str(Path(args.screen_tarball).resolve()), "member": SCREEN_MEMBER, "sha256": args.screen_sha256},
            "pxr_single_concentration": {"path": str(Path(args.pxr_single).resolve()), "sha256": args.pxr_single_sha256},
            "pxr_pins": {"path": str(Path(args.pxr_pins).resolve()), "sha256": args.pxr_pins_sha256},
            "cyp_train_split": {"path": str(Path(args.cyp_train_split).resolve()), "sha256": args.cyp_train_split_sha256},
        },
        "crosswalk": crosswalk,
        "set": {"rows": len(mapping_rows), "set_aside_not_scored": selection["set_aside_count"],
                "evaluated": len(evaluated), "pxr_roles": selection["roles"]},
        "tier_role_counts": selection["tier_role"],
        "baselines": {
            "cyp": {e: {**cyp_constants[e], "definition": "unweighted mean of observed native log2fc over the screening-training partition (partition train, 6,252 molecules)"} for e in ENZYMES},
            "pxr_8p25uM": pxr_constant,
            "note": "Constants come from training-only data; no evaluation label enters them. An unweighted mean is a floor, not the MAE-optimal constant.",
        },
        "metrics": results,
        "metric_definitions": {
            "mae": "mean |prediction - label| in native log2fc units over molecules with finite prediction and label",
            "spearman": "average-rank Spearman; null if n < 3 or either input is constant; no p-value, no significance claim",
            "baseline_mae": "mean |label - constant| with the training-only constant",
            "mae_over_baseline": "mae / baseline_mae; a constant has no meaningful Spearman",
        },
        "limits": [
            "descriptive only; molecules share scaffold groups and are not independent; no confidence intervals",
            "not challenge performance, not validation of the full potency-plus-screen report, not evidence of confirmed PXR-specific activity",
            "no review threshold is chosen or tuned from these results",
        ],
        "labels": {"set_aside_molecules_scored": False, "labels_read_for_evaluated_molecules_only": True,
                   "training_labels_read_for_baseline_only": True},
        "environment": {"python": platform.python_version(), "rdkit": rdkit.__version__},
    }
    return {"report": report, "evaluated": evaluated, "pred_by_id": pred_by_id,
            "cyp_labels": cyp_labels, "pxr_labels": pxr_labels, "results": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--predictions", "--mapping", "--screen-tarball", "--pxr-single", "--pxr-pins", "--cyp-train-split", "--out-dir"):
        parser.add_argument(flag, required=True, type=Path)
    for flag in ("--predictions-sha256", "--mapping-sha256", "--screen-sha256", "--pxr-single-sha256",
                 "--pxr-pins-sha256", "--cyp-train-split-sha256"):
        parser.add_argument(flag, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.out_dir}")
    out = run(args)
    files = write_outputs(args.out_dir, out["evaluated"], out["pred_by_id"], out["cyp_labels"],
                          out["pxr_labels"], out["results"], out["report"])
    summary = {s: {o: {k: out["results"][s][o][k] for k in ("n", "mae", "spearman", "baseline_mae", "mae_over_baseline")}
                   for o in OUTPUTS} for s in STRATA}
    print(json.dumps({"status": "completed", "evaluated": len(out["evaluated"]), "metrics": summary,
                      "outputs": {k: v["path"] for k, v in files.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
