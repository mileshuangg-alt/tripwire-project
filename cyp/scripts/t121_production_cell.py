"""One updated-T121 pIC50 production cell (D015), fitted from the selected
T121 screening checkpoint.

Mirrors cyp007_cell_fit.run_production_cell (same STARTED / COMPLETED /
INCOMPLETE protocol, same result.json and provenance.json fields) so the
existing O150 collection and aggregation read these cells unchanged. The
differences are only where the model comes from:

  screening shell + selected screening checkpoint -> fresh pIC50 state with
  shared weights transferred (prepare_selected_screen_handoff; no optimizer,
  scheduler, epoch counter or target scaler is carried over) -> the existing
  D007 fit (fit_prepared_t121_pic50_stage).

Nothing here has a default path or value. Training must be authorized
explicitly. The scientific helpers are injected through `ops` so the protocol
is testable without Chemprop; load_ops() binds the real ones.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import socket
import sys
import time


def load_ops():
    import pandas as pd
    import cyp007_cell_fit as fit
    import t121_screen_stage as screen
    import t121_weighted_training as weighted

    return SimpleNamespace(
        read_csv=pd.read_csv,
        read_json=fit._read_json,
        write_json_atomic=fit._write_json_atomic,
        write_json_exclusive=fit._write_json_exclusive,
        sha256_file=fit._sha256_file,
        cell_paths=fit._cell_paths,
        route_fold=fit.route_d007_fold,
        prepare_screen_stage=screen.prepare_screen_stage,
        prepare_selected_screen_handoff=screen.prepare_selected_screen_handoff,
        fit_stage=weighted.fit_prepared_t121_pic50_stage,
    )


def jsonable(value):
    return json.loads(json.dumps(value, default=str))


def run_updated_t121_cell(
    *,
    run_root,
    trial_index,
    cell_index,
    dataset_path,
    original_split_path,
    split_manifest_root,
    screen_selection_path,
    screen_selection_sha256,
    screen_proposal_path,
    screen_inputs,
    authorize_training=False,
    ops=None,
):
    if authorize_training is not True:
        raise RuntimeError("Updated-T121 cell training is not authorized.")
    ops = load_ops() if ops is None else ops

    run_root = Path(run_root)
    trial_dir, cell_dir = ops.cell_paths(run_root, trial_index, cell_index)
    plan_path = cell_dir / "cell_plan.json"
    proposal_path = trial_dir / "proposal.json"
    result_path = cell_dir / "result.json"
    complete_path = cell_dir / "COMPLETED.json"
    started_path = cell_dir / "STARTED.json"
    incomplete_path = cell_dir / "INCOMPLETE.json"

    if complete_path.exists() and result_path.exists():
        return ops.read_json(result_path)
    if started_path.exists() and not complete_path.exists():
        raise RuntimeError(
            "T121 cell has STARTED without COMPLETED; human adjudication required."
        )
    if result_path.exists() or complete_path.exists():
        raise RuntimeError("T121 cell has partial result evidence; refusing overwrite.")

    selection_path = Path(screen_selection_path)
    if ops.sha256_file(selection_path) != screen_selection_sha256:
        raise ValueError("Selected screening record does not match its pinned SHA-256.")
    selection = ops.read_json(selection_path)
    if not {"epoch", "score", "checkpoint", "sha256"} <= set(selection):
        raise ValueError("Selected screening record lacks epoch, score, checkpoint or sha256.")

    plan = ops.read_json(plan_path)
    proposal = ops.read_json(proposal_path)
    if plan["trial_index"] != trial_index or plan["cell_index"] != cell_index:
        raise RuntimeError("T121 cell identity does not match its frozen plan.")
    sampled_config = proposal.get("sampled_config")
    if not isinstance(sampled_config, dict):
        raise RuntimeError("T121 proposal is missing frozen sampled_config.")
    config_name = proposal.get("config_name", f"D007-T{trial_index:03d}")

    split_manifest_path = (
        Path(split_manifest_root)
        / f"rep_{plan['repetition']:02d}_fold_{plan['fold']:02d}.json"
    )
    split_manifest = ops.read_json(split_manifest_path)
    if plan["seed"] not in tuple(split_manifest["training_seeds"]):
        raise RuntimeError("T121 cell seed is not in the frozen split manifest.")
    if (
        split_manifest["repetition"] != plan["repetition"]
        or split_manifest["fold"] != plan["fold"]
    ):
        raise RuntimeError("T121 split manifest does not match cell plan.")

    ops.write_json_exclusive(started_path, {
        "status": "started",
        "trial_index": trial_index,
        "cell_index": cell_index,
        "repetition": plan["repetition"],
        "fold": plan["fold"],
        "seed": plan["seed"],
        "hostname": socket.gethostname(),
        "started_time": time.time(),
        "split_manifest_path": str(split_manifest_path),
        "proposal_path": str(proposal_path),
        "arm": "updated_t121_from_screen_checkpoint",
        "screen_selection_sha256": screen_selection_sha256,
    })

    try:
        routed = ops.route_fold(
            dataframe=ops.read_csv(dataset_path),
            original_split_dataframe=ops.read_csv(original_split_path),
            split_manifest=split_manifest,
            training_seed=plan["seed"],
        )
        prepared_screen = ops.prepare_screen_stage(
            **dict(screen_inputs),
            proposal_path=screen_proposal_path,
            output_dir=cell_dir / "screen_shell",
            training_seed=plan["seed"],
        )
        if prepared_screen["sampled_config"] != sampled_config:
            raise RuntimeError(
                "Screening proposal and cell proposal disagree on sampled_config."
            )
        prefit = ops.prepare_selected_screen_handoff(
            prepared_screen=prepared_screen,
            selected_record=selection,
            training_dataframe=routed["training_dataframe"],
            validation_dataframe=routed["validation_dataframe"],
            config_name=config_name,
            training_seed=plan["seed"],
        )
        fit = ops.fit_stage(
            prefit_state=prefit,
            checkpoint_dir=cell_dir / "fit",
            authorize_training=True,
        )
        checkpoint_path = Path(fit["checkpoint_path"])
        best_validation = fit["best_validation"]
        result = {
            "trial_index": trial_index,
            "cell_index": cell_index,
            "repetition": plan["repetition"],
            "fold": plan["fold"],
            "seed": plan["seed"],
            "status": "completed",
            "macro_st_rae": best_validation["macro_st_rae"],
            "best_validation": best_validation,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": ops.sha256_file(checkpoint_path),
            "split_manifest_path": str(split_manifest_path),
            "split_manifest_sha256": ops.sha256_file(split_manifest_path),
            "proposal_path": str(proposal_path),
            "config_name": config_name,
            "arm": "updated_t121_from_screen_checkpoint",
            "screen_selected_epoch": selection["epoch"],
            "screen_checkpoint_sha256": selection["sha256"],
        }
        provenance = {
            "result": result,
            "sampled_config": sampled_config,
            "epochs_completed": fit["epochs_completed"],
            "validation_history": fit["validation_history"],
            "dataset_path": str(dataset_path),
            "dataset_sha256": ops.sha256_file(dataset_path),
            "original_split_path": str(original_split_path),
            "original_split_sha256": ops.sha256_file(original_split_path),
            "screen_selection": selection,
            "screen_selection_sha256": screen_selection_sha256,
            "screen_to_pic50_transition": jsonable(
                prefit.get("screen_to_pic50_transition")
            ),
            "updated_t121_weighting": jsonable(
                prefit.get("updated_t121_weighting")
            ),
        }
        ops.write_json_atomic(cell_dir / "provenance.json", provenance)
        ops.write_json_exclusive(result_path, result)
        ops.write_json_exclusive(complete_path, {
            "status": "completed",
            "result_sha256": ops.sha256_file(result_path),
            "provenance_sha256": ops.sha256_file(cell_dir / "provenance.json"),
        })
        return result
    except BaseException as exc:
        if not incomplete_path.exists():
            ops.write_json_atomic(incomplete_path, {
                "status": "incomplete",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "requires_human_adjudication": True,
            })
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in (
        "run-root", "dataset", "original-split", "split-manifest-root",
        "screen-selection", "screen-selection-sha256", "screen-proposal",
        "screen", "identity", "similarity", "contract",
        "executor-source-dir", "executor-source-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--cell-index", required=True, type=int)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.executor_source_dir) / "cyp007_cell_fit.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != args.executor_source_sha256:
        raise SystemExit(f"Executor source hash mismatch: {source} {digest}")
    for directory in (args.executor_source_dir, str(Path(__file__).resolve().parent)):
        if directory not in sys.path:
            sys.path.insert(0, directory)

    result = run_updated_t121_cell(
        run_root=args.run_root,
        trial_index=args.trial_index,
        cell_index=args.cell_index,
        dataset_path=args.dataset,
        original_split_path=args.original_split,
        split_manifest_root=args.split_manifest_root,
        screen_selection_path=args.screen_selection,
        screen_selection_sha256=args.screen_selection_sha256,
        screen_proposal_path=args.screen_proposal,
        screen_inputs={
            "screen_path": args.screen,
            "identity_path": args.identity,
            "similarity_path": args.similarity,
            "contract_path": args.contract,
        },
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
