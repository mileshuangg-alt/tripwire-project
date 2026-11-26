"""CYP train/validation tier split of the D026 descriptive evaluation (Tripwire-CYP-010).

Authorizes no run by itself; the project owner runs it locally. It reads only two existing files
(evaluation_rows.csv and evaluation_metrics.csv from the D026 evaluation), both hash-checked, and
writes into a NEW output directory. No model, no inference, no training, no new baseline, no
bootstrap and no threshold. It reuses spearman and metrics_for from evaluate_screen_predictions.py.

For each of the four CYP outputs and each CYP tier (train, validation; cyp_tiers is one value per
molecule, so every output uses the same molecule-level tier) it reports: n, MAE, Spearman, the
existing baseline constant, the baseline MAE within the tier, the model/baseline MAE ratio, and
the signed mean residual (prediction minus label).

Interpretation limits: the constants are the existing training-only constants from the D026
evaluation, unchanged; they are not bounds on achievable error and have no Spearman. The signed
mean residual measures aggregate offset only; by itself it does not separate bias from noise.
The validation tier is small, molecules share scaffold groups, and there are no confidence
intervals. Descriptive only; not challenge performance; no threshold is chosen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_screen_predictions as ev  # noqa: E402

TIERS = ("train", "validation")
FIELDS = ["output", "cyp_tier", "n", "mae", "spearman", "baseline_constant", "baseline_mae",
          "mae_over_baseline", "signed_mean_residual"]


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_verified(path, expected: str) -> list:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Input hash mismatch: {path} (actual {actual}, expected {expected})")
    return list(csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8"), newline="")))


def constants_from_metrics(metrics_rows) -> dict:
    """One baseline constant per CYP enzyme; it must be identical in every stratum."""
    found = {}
    for r in metrics_rows:
        if r["output"] not in ev.ENZYMES:
            continue
        found.setdefault(r["output"], set()).add(r["baseline_constant"])
    for e in ev.ENZYMES:
        if e not in found:
            raise ValueError(f"No baseline constant for {e} in the metrics file.")
        if len(found[e]) != 1:
            raise ValueError(f"{e} baseline constant differs between strata: {sorted(found[e])}")
    return {e: float(next(iter(found[e]))) for e in ev.ENZYMES}


def tier_rows(rows, constants) -> list:
    out = []
    for e in ev.ENZYMES:
        for tier in TIERS:
            subset = [r for r in rows if r["cyp_tiers"] == tier]
            pred = [ev.num(r[ev.CYP_OUTPUTS[e]]) for r in subset]
            lab = [ev.num(r[f"label_{e}"]) for r in subset]
            m = ev.metrics_for(pred, lab, constants[e])
            pairs = [(p, y) for p, y in zip(pred, lab) if p is not None and y is not None]
            m["signed_mean_residual"] = sum(p - y for p, y in pairs) / len(pairs) if pairs else None
            out.append({"output": e, "cyp_tier": tier, **{k: m[k] for k in FIELDS[2:]}})
    return out


def pooled_check(rows, constants, metrics_rows, tol=1e-12):
    """The pooled (all-tier) numbers recomputed here must equal the recorded all-stratum rows."""
    recorded = {r["output"]: r for r in metrics_rows if r["stratum"] == "all"}
    for e in ev.ENZYMES:
        pred = [ev.num(r[ev.CYP_OUTPUTS[e]]) for r in rows]
        lab = [ev.num(r[f"label_{e}"]) for r in rows]
        m = ev.metrics_for(pred, lab, constants[e])
        rec = recorded[e]
        if m["n"] != int(rec["n"]):
            raise ValueError(f"{e}: pooled n {m['n']} differs from the recorded {rec['n']}.")
        for key in ("mae", "spearman", "baseline_mae", "mae_over_baseline"):
            if abs(m[key] - float(rec[key])) > tol:
                raise ValueError(f"{e}: pooled {key} {m[key]!r} differs from the recorded {rec[key]!r}.")


def run(args) -> dict:
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to reuse {out_dir}")
    rows = read_verified(args.rows, args.rows_sha256)
    metrics_rows = read_verified(args.metrics, args.metrics_sha256)
    unknown = {r["cyp_tiers"] for r in rows} - set(TIERS)
    if unknown:
        raise ValueError(f"Unexpected cyp_tiers values: {sorted(unknown)}")
    if len({r["OCNT_ID"] for r in rows}) != len(rows):
        raise ValueError("Duplicate OCNT_ID in the evaluation rows.")
    constants = constants_from_metrics(metrics_rows)
    pooled_check(rows, constants, metrics_rows)
    table = tier_rows(rows, constants)
    counts = {t: sum(1 for r in rows if r["cyp_tiers"] == t) for t in TIERS}
    out_dir.mkdir(parents=True)
    with open(out_dir / "cyp_tier_split.csv", "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)
    report = {
        "inputs": {"evaluation_rows": {"path": str(args.rows), "sha256": args.rows_sha256},
                   "evaluation_metrics": {"path": str(args.metrics), "sha256": args.metrics_sha256}},
        "molecules_by_cyp_tier": counts,
        "baseline_constants": constants,
        "pooled_consistency_check": "passed (n, MAE, Spearman, baseline MAE and ratio equal the recorded all-stratum rows)",
        "limits": __doc__.split("Interpretation limits:")[1].strip(),
    }
    (out_dir / "cyp_tier_split_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for r in table:
        print(f"{r['output']:7s} {r['cyp_tier']:10s} n {r['n']:4d}  MAE {r['mae']:.4f}  Spearman "
              f"{'NA' if r['spearman'] is None else format(r['spearman'], '.4f')}  baseline {r['baseline_constant']:+.4f}"
              f"  baseline MAE {r['baseline_mae']:.4f}  ratio {r['mae_over_baseline']:.4f}"
              f"  signed mean residual {r['signed_mean_residual']:+.4f}")
    print("tier counts", counts)
    for name in ("cyp_tier_split.csv", "cyp_tier_split_report.json"):
        print(sha256_file(out_dir / name), name)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--rows", "--metrics", "--output-dir"):
        parser.add_argument(flag, required=True, type=Path)
    for flag in ("--rows-sha256", "--metrics-sha256"):
        parser.add_argument(flag, required=True)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
