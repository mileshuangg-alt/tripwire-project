"""Create the 150-cell plan for the updated-T121 pIC50 stage (D015).

Writes <run-root>/trials/trial_121/proposal.json (a copy of the pinned
proposal), one cells/cell_NNN/cell_plan.json per canonical D007 cell 0-149, and
cell_array_ready.json, in the same layout and fields as the "full" branch of
prepare_trial_cells in cyp007_production_execution.py. That module cannot be
imported here: it needs aggregate_d007_reduced_objective and
D007_REDUCED_OBJECTIVE from cyp007_infrastructure, which the checked-in
cyp007_infrastructure.py does not define. The cell grid comes from the
pinned cyp007_production_helpers.cell_keys, the same source that module uses.
Controller side only: it runs no model code.

Usage:
    python -I t121_plan_cells.py --run-root R --proposal P --proposal-sha256 H \
        --executor-source-dir D --helpers-sha256 H2
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

TRIAL_INDEX = 121
TOTAL_CELLS = 150


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def plan_cells(*, run_root, proposal_path, proposal_sha256, cell_keys):
    if sha256_file(proposal_path) != proposal_sha256:
        raise ValueError("Trial-121 proposal does not match its pinned SHA-256.")
    keys = list(cell_keys)
    if len(keys) != TOTAL_CELLS or len(set(keys)) != TOTAL_CELLS:
        raise ValueError("Expected exactly 150 distinct D007 cells.")
    trial_dir = Path(run_root) / "trials" / f"trial_{TRIAL_INDEX:03d}"
    if trial_dir.exists():
        raise FileExistsError(f"Refusing to reuse cell plan directory: {trial_dir}")
    cells_root = trial_dir / "cells"
    cells_root.mkdir(parents=True)
    shutil.copyfile(proposal_path, trial_dir / "proposal.json")
    plans = []
    for index, (repetition, fold, seed) in enumerate(keys):
        cell_dir = cells_root / f"cell_{index:03d}"
        cell_dir.mkdir()
        plan = {
            "cell_index": index,
            "trial_index": TRIAL_INDEX,
            "repetition": repetition,
            "fold": fold,
            "seed": seed,
            "status": "planned",
            "proposal_sha256_source": "proposal.json",
            "fidelity": "full",
        }
        write_new_json(cell_dir / "cell_plan.json", plan)
        plans.append(plan)
    write_new_json(trial_dir / "cell_array_ready.json", {
        "status": "ready",
        "cell_count": len(plans),
        "cell_indices": list(range(TOTAL_CELLS)),
        "fidelity": "full",
        "proposal": json.loads((trial_dir / "proposal.json").read_text()),
    })
    return plans


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in (
        "run-root", "proposal", "proposal-sha256",
        "executor-source-dir", "helpers-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)

    helpers = Path(args.executor_source_dir) / "cyp007_production_helpers.py"
    if sha256_file(helpers) != args.helpers_sha256:
        raise SystemExit(f"Helpers hash mismatch: {helpers}")
    sys.path.insert(0, args.executor_source_dir)
    from cyp007_production_helpers import cell_keys

    plans = plan_cells(
        run_root=args.run_root,
        proposal_path=args.proposal,
        proposal_sha256=args.proposal_sha256,
        cell_keys=cell_keys(),
    )
    print(f"planned {len(plans)} cells under {args.run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
