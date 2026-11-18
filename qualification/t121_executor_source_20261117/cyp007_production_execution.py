from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from cyp007_infrastructure import d007_search_space
from cyp007_infrastructure import aggregate_d007_reduced_objective
from cyp007_infrastructure import D007_REDUCED_OBJECTIVE
from cyp007_production_helpers import (
    D007_CONTROLLER_CONFIG,
    D007_CONTROLLER_RNG_SEED,
    D007_TOTAL_CELLS,
    aggregate_d007_objective,
    cell_keys,
    controller_config,
    reduced_cell_indices,
    D007_REDUCED_TOTAL_CELLS,
    reduced_cell_keys,
)

D007_METRIC = "mean_validation_macro_st_rae_50_folds_x_3_seeds"
D007_REDUCED_METRIC = D007_REDUCED_OBJECTIVE
D007_MODE = "min"

RESOURCE_TOPOLOGY = {
    "controller": {
        "partition": "cpu",
        "cpus": 1,
        "memory": "4G",
        "role": "sequential adaptive HyperOpt/TPE controller",
        "environment": "/scratch/user/mhuang8/envs/tripwire_d006_controller",
    },
    "cell_array": {
        "partition": "gpu",
        "gres": "gpu:nvidia_l40s:1",
        "cpus_per_task": 4,
        "memory": "32G",
        "time": "12:00:00",
        "array": "0-149",
        "evidence": "preserved D006 search wrapper d006_search.slurm",
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_json_exclusive(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def trial_id(trial_index: int) -> str:
    return f"trial_{trial_index:03d}"


def trial_dir(root: Path, trial_index: int) -> Path:
    return root / "trials" / trial_id(trial_index)


def cell_dir(trial_path: Path, cell_index: int) -> Path:
    return trial_path / "cells" / f"cell_{cell_index:03d}"


def hyperopt_space() -> dict[str, Any]:
    from hyperopt import hp

    frozen = d007_search_space()
    return {
        "message_passing_depth": hp.quniform("message_passing_depth", frozen["message_passing_depth"]["low"], frozen["message_passing_depth"]["high"], 1),
        "message_hidden_dim": hp.quniform("message_hidden_dim", frozen["message_hidden_dim"]["low"], frozen["message_hidden_dim"]["high"], frozen["message_hidden_dim"]["step"]),
        "ffn_hidden_dim": hp.quniform("ffn_hidden_dim", frozen["ffn_hidden_dim"]["low"], frozen["ffn_hidden_dim"]["high"], frozen["ffn_hidden_dim"]["step"]),
        "ffn_num_layers": hp.quniform("ffn_num_layers", frozen["ffn_num_layers"]["low"], frozen["ffn_num_layers"]["high"], 1),
        "dropout": hp.choice("dropout", frozen["dropout"]["values"]),
        "batch_size": hp.choice("batch_size", frozen["batch_size"]["values"]),
        "max_lr": hp.loguniform("max_lr", math.log(frozen["max_lr"]["low"]), math.log(frozen["max_lr"]["high"])),
        "init_lr_ratio": hp.loguniform("init_lr_ratio", math.log(frozen["init_lr_ratio"]["low"]), math.log(frozen["init_lr_ratio"]["high"])),
        "final_lr_ratio": hp.loguniform("final_lr_ratio", math.log(frozen["final_lr_ratio"]["low"]), math.log(frozen["final_lr_ratio"]["high"])),
        "warmup_epochs": hp.quniform("warmup_epochs", frozen["warmup_epochs"]["low"], frozen["warmup_epochs"]["high"], 1),
    }


def normalize_sampled_config(sampled: dict[str, Any]) -> dict[str, Any]:
    required = set(d007_search_space())
    if set(sampled) != required:
        raise ValueError(f"D007 sampled config keys mismatch: {sorted(sampled)}")
    result = {
        "message_passing_depth": int(sampled["message_passing_depth"]),
        "message_hidden_dim": int(sampled["message_hidden_dim"]),
        "ffn_hidden_dim": int(sampled["ffn_hidden_dim"]),
        "ffn_num_layers": int(sampled["ffn_num_layers"]),
        "dropout": float(sampled["dropout"]),
        "batch_size": int(sampled["batch_size"]),
        "max_lr": float(sampled["max_lr"]),
        "init_lr_ratio": float(sampled["init_lr_ratio"]),
        "final_lr_ratio": float(sampled["final_lr_ratio"]),
        "warmup_epochs": int(sampled["warmup_epochs"]),
    }
    frozen = d007_search_space()
    for key in ["message_passing_depth", "ffn_num_layers", "warmup_epochs"]:
        if not frozen[key]["low"] <= result[key] <= frozen[key]["high"]:
            raise ValueError(f"D007 sampled {key} outside frozen range: {result[key]}")
    for key in ["message_hidden_dim", "ffn_hidden_dim"]:
        spec = frozen[key]
        if not spec["low"] <= result[key] <= spec["high"] or (result[key] - spec["low"]) % spec["step"] != 0:
            raise ValueError(f"D007 sampled {key} outside frozen stepped range: {result[key]}")
    for key in ["max_lr", "init_lr_ratio", "final_lr_ratio"]:
        if not frozen[key]["low"] <= result[key] <= frozen[key]["high"]:
            raise ValueError(f"D007 sampled {key} outside frozen range: {result[key]}")
    if result["batch_size"] not in frozen["batch_size"]["values"]:
        raise ValueError("D007 sampled batch_size outside frozen values")
    if result["dropout"] not in set(frozen["dropout"]["values"]):
        raise ValueError("D007 sampled dropout outside frozen values")
    return result


def new_searcher():
    from ray.tune.search.hyperopt import HyperOptSearch

    return HyperOptSearch(
        hyperopt_space(),
        metric=D007_METRIC,
        mode=D007_MODE,
        n_initial_points=D007_CONTROLLER_CONFIG.random_startup_trials,
        random_state_seed=D007_CONTROLLER_RNG_SEED,
    )


def new_reduced_searcher():
    """
    Construct the amended D007 HyperOpt searcher.

    Search space, RNG, startup count, mode, and sequential HyperOpt
    behavior are unchanged. Only the metric identity changes from
    O150 to the frozen O30 search objective.
    """
    from ray.tune.search.hyperopt import HyperOptSearch

    result = HyperOptSearch(
        hyperopt_space(),
        metric=D007_REDUCED_METRIC,
        mode=D007_MODE,
        n_initial_points=D007_CONTROLLER_CONFIG.random_startup_trials,
        random_state_seed=D007_CONTROLLER_RNG_SEED,
    )

    return result


class D007ProductionController:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.state_path = self.root / "controller_state.json"
        self.searcher_path = self.root / "hyperopt_searcher.pkl"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            self.searcher = new_searcher()
            if not self.searcher_path.is_file():
                raise RuntimeError("Missing persisted HyperOpt searcher state.")
            self.searcher.restore(str(self.searcher_path))
        else:
            self.searcher = new_searcher()
            self.state = self._initial_state()
            self._persist()

    def _initial_state(self) -> dict:
        return {
            "schema_version": "D007-production-controller-state-v2",
            "controller_config": controller_config(),
            "metric": D007_METRIC,
            "mode": D007_MODE,
            "resource_topology": RESOURCE_TOPOLOGY,
            "observations": [],
            "pending_trial_index": None,
            "next_trial_index": 1,
            "status": "initialized",
        }

    def _persist(self) -> None:
        write_json_atomic(self.state_path, self.state)
        self.searcher.save(str(self.searcher_path))

    def propose_next(self) -> dict:
        pending = self.state.get("pending_trial_index")
        if pending is not None:
            raise RuntimeError(f"D007 trial {pending} is incomplete; refusing next proposal.")
        trial_index = self.state["next_trial_index"]
        if trial_index > D007_CONTROLLER_CONFIG.total_trials:
            raise RuntimeError("D007 128-trial limit reached.")
        tid = trial_id(trial_index)
        tdir = trial_dir(self.root, trial_index)
        if tdir.exists():
            if (tdir / "proposal.json").is_file():
                proposal = read_json(tdir / "proposal.json")
                self.state["pending_trial_index"] = trial_index
                self.state["status"] = "trial_pending"
                self._persist()
                return proposal
            raise RuntimeError(f"D007 trial directory exists without proposal: {tdir}")
        raw = self.searcher.suggest(tid)
        if raw is None:
            raise RuntimeError("HyperOptSearch returned no D007 suggestion.")
        config = normalize_sampled_config(raw)
        strategy = "random_startup" if trial_index <= D007_CONTROLLER_CONFIG.random_startup_trials else "tpe"
        proposal = {
            "trial_index": trial_index,
            "trial_id": tid,
            "config_name": f"D007-T{trial_index:03d}",
            "strategy": strategy,
            "sampled_config": config,
            "metric": D007_METRIC,
            "mode": D007_MODE,
            "controller_rng_seed": D007_CONTROLLER_RNG_SEED,
            "imported_observations": 0,
        }
        tdir.mkdir(parents=True, exist_ok=False)
        write_json_exclusive(tdir / "proposal.json", proposal)
        self.state["pending_trial_index"] = trial_index
        self.state["status"] = "trial_pending"
        self._persist()
        return proposal

    def complete_trial(self, trial_index: int) -> dict:
        if self.state.get("pending_trial_index") != trial_index:
            raise RuntimeError("D007 controller is not pending this trial.")
        tdir = trial_dir(self.root, trial_index)
        objective_path = tdir / "objective.json"
        if objective_path.exists():
            raise RuntimeError("D007 objective already exists; duplicate completion rejected.")
        results = collect_cell_results(self.root, trial_index)
        objective = aggregate_d007_objective(results)
        observation = {"trial_index": trial_index, "trial_id": trial_id(trial_index), "objective": objective}
        self.searcher.on_trial_complete(trial_id(trial_index), result={D007_METRIC: objective})
        write_json_exclusive(objective_path, observation)
        self.state["observations"].append(observation)
        self.state["pending_trial_index"] = None
        self.state["next_trial_index"] = trial_index + 1
        self.state["status"] = "ready_for_next_proposal" if self.state["next_trial_index"] <= D007_CONTROLLER_CONFIG.total_trials else "completed"
        self._persist()
        return observation


class D007ReducedProductionController:
    """
    Amended D007 search controller using the frozen O30 objective.

    This class is intentionally separate from D007ProductionController,
    which remains the historical O150 implementation used to create the
    pre-amendment trials and persisted state.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.state_path = self.root / "controller_state.json"
        self.searcher_path = self.root / "hyperopt_searcher.pkl"

        self.root.mkdir(parents=True, exist_ok=True)

        if self.state_path.exists():
            self.state = read_json(self.state_path)

            if self.state.get("metric") != D007_REDUCED_METRIC:
                raise RuntimeError(
                    "Refusing to load non-reduced D007 state with "
                    "the amended controller."
                )

            self.searcher = new_reduced_searcher()

            if not self.searcher_path.is_file():
                raise RuntimeError(
                    "Missing persisted amended HyperOpt searcher state."
                )

            self.searcher.restore(str(self.searcher_path))
        else:
            self.searcher = new_reduced_searcher()
            self.state = self._initial_state()
            self._persist()

    def _initial_state(self) -> dict:
        config = dict(controller_config())
        config["objective"] = D007_REDUCED_METRIC

        result = {
            "schema_version": "D007-reduced-production-controller-state-v1",
            "controller_config": config,
            "metric": D007_REDUCED_METRIC,
            "mode": D007_MODE,
            "resource_topology": RESOURCE_TOPOLOGY,
            "observations": [],
            "pending_trial_index": None,
            "next_trial_index": 1,
            "status": "initialized",
            "search_fidelity": "reduced",
            "search_cell_count": D007_REDUCED_TOTAL_CELLS,
            "final_selection_metric": D007_METRIC,
        }

        return result

    def _persist(self) -> None:
        write_json_atomic(
            self.state_path,
            self.state,
        )
        self.searcher.save(
            str(self.searcher_path)
        )

    def propose_next(self) -> dict:
        pending = self.state.get("pending_trial_index")

        if pending is not None:
            raise RuntimeError(
                f"D007 trial {pending} is incomplete; "
                "refusing next proposal."
            )

        trial_index = self.state["next_trial_index"]

        if trial_index > D007_CONTROLLER_CONFIG.total_trials:
            raise RuntimeError(
                "D007 128-trial limit reached."
            )

        tid = trial_id(trial_index)
        tdir = trial_dir(
            self.root,
            trial_index,
        )

        if tdir.exists():
            if (tdir / "proposal.json").is_file():
                proposal = read_json(
                    tdir / "proposal.json"
                )

                if proposal.get("metric") != D007_REDUCED_METRIC:
                    raise RuntimeError(
                        "Existing proposal does not carry the "
                        "amended reduced metric."
                    )

                self.state["pending_trial_index"] = trial_index
                self.state["status"] = "trial_pending"
                self._persist()

                return proposal

            raise RuntimeError(
                "D007 trial directory exists without proposal: "
                f"{tdir}"
            )

        raw = self.searcher.suggest(tid)

        if raw is None:
            raise RuntimeError(
                "HyperOptSearch returned no D007 suggestion."
            )

        config = normalize_sampled_config(raw)

        strategy = (
            "random_startup"
            if trial_index
            <= D007_CONTROLLER_CONFIG.random_startup_trials
            else "tpe"
        )

        proposal = {
            "trial_index": trial_index,
            "trial_id": tid,
            "config_name": f"D007-T{trial_index:03d}",
            "strategy": strategy,
            "sampled_config": config,
            "metric": D007_REDUCED_METRIC,
            "mode": D007_MODE,
            "controller_rng_seed": D007_CONTROLLER_RNG_SEED,
            "imported_observations": 0,
            "search_fidelity": "reduced",
            "search_cell_count": D007_REDUCED_TOTAL_CELLS,
        }

        tdir.mkdir(
            parents=True,
            exist_ok=False,
        )

        write_json_exclusive(
            tdir / "proposal.json",
            proposal,
        )

        self.state["pending_trial_index"] = trial_index
        self.state["status"] = "trial_pending"
        self._persist()

        return proposal

    def complete_trial(self, trial_index: int) -> dict:
        if self.state.get("pending_trial_index") != trial_index:
            raise RuntimeError(
                "D007 reduced controller is not pending this trial."
            )

        tdir = trial_dir(
            self.root,
            trial_index,
        )

        objective_path = (
            tdir / "reduced_objective.json"
        )

        if objective_path.exists():
            raise RuntimeError(
                "D007 reduced objective already exists; "
                "duplicate completion rejected."
            )

        objective = calculate_reduced_trial_objective(
            self.root,
            trial_index,
        )

        observation = {
            "trial_index": trial_index,
            "trial_id": trial_id(trial_index),
            "metric": D007_REDUCED_METRIC,
            "fidelity": "reduced",
            "cell_count": D007_REDUCED_TOTAL_CELLS,
            "objective": objective,
        }

        self.searcher.on_trial_complete(
            trial_id(trial_index),
            result={
                D007_REDUCED_METRIC: objective,
            },
            error=False,
        )

        write_json_exclusive(
            objective_path,
            observation,
        )

        self.state["observations"].append(
            observation
        )
        self.state["pending_trial_index"] = None
        self.state["next_trial_index"] = trial_index + 1

        if (
            self.state["next_trial_index"]
            <= D007_CONTROLLER_CONFIG.total_trials
        ):
            self.state["status"] = "ready_for_next_proposal"
        else:
            self.state["status"] = "search_completed"

        self._persist()

        result = observation

        return result


def initialize_run(root: str | Path) -> dict:
    return D007ProductionController(root).state


def load_state(root: str | Path) -> dict:
    return read_json(Path(root) / "controller_state.json")


def save_state(root: str | Path, state: dict) -> None:
    write_json_atomic(Path(root) / "controller_state.json", state)


def propose_configuration(root: str | Path) -> dict:
    return D007ProductionController(root).propose_next()


def prepare_trial_cells(
    root: str | Path,
    trial_index: int,
    *,
    fidelity: str,
) -> list[dict]:
    """
    Plan D007 cells at an explicitly declared evidence fidelity.

    reduced:
        the frozen 30-cell search panel, retaining canonical 0-149
        cell indices.

    full:
        the complete original 150-cell D007 evaluation matrix.
    """
    if fidelity == "reduced":
        planned_indices = reduced_cell_indices()
    elif fidelity == "full":
        planned_indices = list(range(D007_TOTAL_CELLS))
    else:
        raise ValueError(
            "D007 fidelity must be exactly 'reduced' or 'full'."
        )

    tdir = trial_dir(Path(root), trial_index)
    proposal = read_json(tdir / "proposal.json")
    cells_root = tdir / "cells"

    expected_indices = set(planned_indices)

    if cells_root.exists() and any(cells_root.iterdir()):
        plans = sorted(cells_root.glob("cell_*/cell_plan.json"))

        observed_indices = {
            int(read_json(path)["cell_index"])
            for path in plans
        }

        if observed_indices != expected_indices:
            raise RuntimeError(
                "D007 existing cell plans do not match the "
                f"requested {fidelity!r} fidelity."
            )

        return [read_json(path) for path in plans]

    cells_root.mkdir(exist_ok=True)

    full_keys = cell_keys()
    plans = []

    for idx in planned_indices:
        repetition, fold, seed = full_keys[idx]
        cdir = cell_dir(tdir, idx)
        cdir.mkdir(exist_ok=False)

        plan = {
            "cell_index": idx,
            "trial_index": trial_index,
            "repetition": repetition,
            "fold": fold,
            "seed": seed,
            "status": "planned",
            "proposal_sha256_source": "proposal.json",
            "fidelity": fidelity,
        }

        write_json_exclusive(
            cdir / "cell_plan.json",
            plan,
        )
        plans.append(plan)

    write_json_exclusive(
        tdir / "cell_array_ready.json",
        {
            "status": "ready",
            "cell_count": len(plans),
            "cell_indices": planned_indices,
            "fidelity": fidelity,
            "proposal": proposal,
        },
    )

    result = plans

    return result


def write_dummy_cell_result(root: str | Path, trial_index: int, cell_index: int, macro_st_rae: float, *, status: str = "completed", filename: str = "result.json") -> None:
    cdir = cell_dir(trial_dir(Path(root), trial_index), cell_index)
    plan = read_json(cdir / "cell_plan.json")
    payload = {"trial_index": trial_index, "cell_index": cell_index, "repetition": plan["repetition"], "fold": plan["fold"], "seed": plan["seed"], "status": status, "macro_st_rae": macro_st_rae}
    write_json_exclusive(cdir / filename, payload)


def collect_reduced_cell_results(
    root: str | Path,
    trial_index: int,
) -> list[dict]:
    """
    Collect exactly the frozen 30-cell D007 search evidence.

    Additional non-reduced cells may exist, as for historical trials
    already evaluated at full fidelity, but they have no authority in
    the reduced search objective.
    """
    tdir = trial_dir(Path(root), trial_index)
    expected = set(reduced_cell_keys())

    seen = {}
    failures = []

    for cdir in sorted((tdir / "cells").glob("cell_*")):
        result_files = sorted(cdir.glob("result*.json"))

        if not result_files:
            continue

        for result_path in result_files:
            result = read_json(result_path)

            key = (
                result.get("repetition"),
                result.get("fold"),
                result.get("seed"),
            )

            if key not in expected:
                continue

            if key in seen:
                failures.append(
                    f"duplicate reduced result for {key}: {result_path}"
                )
                continue

            seen[key] = result

            if result.get("status") != "completed":
                failures.append(
                    f"failed reduced cell {key}: "
                    f"{result.get('status')}"
                )

            value = result.get("macro_st_rae")

            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False

            if not finite:
                failures.append(
                    f"nonfinite reduced cell {key}: {value!r}"
                )

    missing = expected - set(seen)

    if missing:
        failures.append(
            "missing reduced cells: "
            f"{sorted(missing)[:5]}"
        )

    if failures:
        raise RuntimeError("; ".join(failures))

    result = [
        seen[key]
        for key in reduced_cell_keys()
    ]

    if len(result) != D007_REDUCED_TOTAL_CELLS:
        raise RuntimeError(
            "D007 reduced collection did not produce exactly "
            f"{D007_REDUCED_TOTAL_CELLS} cells."
        )

    return result


def collect_cell_results(root: str | Path, trial_index: int) -> list[dict]:
    tdir = trial_dir(Path(root), trial_index)
    seen = {}
    failures = []
    for cdir in sorted((tdir / "cells").glob("cell_*")):
        result_files = sorted(cdir.glob("result*.json"))
        if not result_files:
            continue
        for result_path in result_files:
            result = read_json(result_path)
            key = (result.get("repetition"), result.get("fold"), result.get("seed"))
            if key in seen:
                failures.append(f"duplicate result for {key}: {result_path}")
            seen[key] = result
            if result.get("status") != "completed":
                failures.append(f"failed cell {key}: {result.get('status')}")
            value = result.get("macro_st_rae")
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                failures.append(f"nonfinite cell {key}: {value!r}")
    if failures:
        write_json_atomic(tdir / "aggregation_blocked.json", {"status": "blocked", "reasons": failures})
        raise RuntimeError("; ".join(failures))
    results = list(seen.values())
    if len(results) != D007_TOTAL_CELLS:
        reason = f"missing cells: expected {D007_TOTAL_CELLS}, got {len(results)}"
        write_json_atomic(tdir / "aggregation_blocked.json", {"status": "blocked", "reasons": [reason]})
        raise RuntimeError(reason)
    return results


def calculate_reduced_trial_objective(
    root: str | Path,
    trial_index: int,
) -> float:
    """
    Calculate O30 from exactly the frozen reduced-search evidence.
    """
    results = collect_reduced_cell_results(
        root,
        trial_index,
    )

    result = aggregate_d007_reduced_objective(results)

    return result


def missing_full_fidelity_cells(
    root: str | Path,
    trial_index: int,
) -> list[int]:
    """
    Return canonical cells still required for a finalist's O150.

    Existing valid completed results are reused. A result counts as
    reusable only when its recorded scientific identity matches the
    canonical D007 identity for that cell index, its status is
    completed, and macro ST-RAE is finite.
    """
    tdir = trial_dir(
        Path(root),
        trial_index,
    )
    full_keys = cell_keys()
    reusable = set()

    for idx, expected_key in enumerate(full_keys):
        result_path = (
            cell_dir(tdir, idx)
            / "result.json"
        )

        if not result_path.is_file():
            continue

        result = read_json(result_path)

        observed_key = (
            result.get("repetition"),
            result.get("fold"),
            result.get("seed"),
        )

        if observed_key != expected_key:
            raise RuntimeError(
                f"D007 cell {idx} identity mismatch: "
                f"{observed_key!r} != {expected_key!r}"
            )

        if result.get("status") != "completed":
            continue

        value = result.get("macro_st_rae")

        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False

        if not finite:
            continue

        reusable.add(idx)

    result = [
        idx
        for idx in range(D007_TOTAL_CELLS)
        if idx not in reusable
    ]

    return result


def aggregate_and_observe(root: str | Path, trial_index: int) -> dict:
    return D007ProductionController(root).complete_trial(trial_index)


def incomplete_cells(
    root: str | Path,
    trial_index: int,
    *,
    fidelity: str,
) -> list[int]:
    """
    Return missing canonical D007 cell indices for a declared fidelity.
    """
    if fidelity == "reduced":
        expected_indices = reduced_cell_indices()
    elif fidelity == "full":
        expected_indices = list(range(D007_TOTAL_CELLS))
    else:
        raise ValueError(
            "D007 fidelity must be exactly 'reduced' or 'full'."
        )

    tdir = trial_dir(
        Path(root),
        trial_index,
    )

    result = [
        idx
        for idx in expected_indices
        if not (
            cell_dir(tdir, idx)
            / "result.json"
        ).is_file()
    ]

    return result


def run_rehearsal(root: str | Path) -> dict:
    root = Path(root)
    state = initialize_run(root)
    proposal = propose_configuration(root)
    prepare_trial_cells(root, 1)
    blocked_second_proposal = False
    try:
        propose_configuration(root)
    except RuntimeError:
        blocked_second_proposal = True
    for idx in range(D007_TOTAL_CELLS - 1):
        write_dummy_cell_result(root, 1, idx, 0.5 + idx / 1000)
    blocked_149 = False
    try:
        aggregate_and_observe(root, 1)
    except RuntimeError:
        blocked_149 = True
    write_dummy_cell_result(root, 1, D007_TOTAL_CELLS - 1, 0.5 + (D007_TOTAL_CELLS - 1) / 1000)
    observation = aggregate_and_observe(root, 1)
    duplicate_blocked = False
    try:
        aggregate_and_observe(root, 1)
    except RuntimeError:
        duplicate_blocked = True
    next_proposal = propose_configuration(root)
    prepare_trial_cells(root, 2)
    write_dummy_cell_result(root, 2, 0, 0.1, status="failed")
    failure_blocked = False
    try:
        collect_cell_results(root, 2)
    except RuntimeError:
        failure_blocked = True
    state_after = load_state(root)
    return {"initial_state": state, "first_proposal": proposal, "blocked_second_proposal_while_incomplete": blocked_second_proposal, "blocked_149_of_150": blocked_149, "observation": observation, "duplicate_observation_blocked": duplicate_blocked, "next_proposal": next_proposal, "failure_blocked": failure_blocked, "state_after": state_after, "incomplete_trial2_count": len(incomplete_cells(root, 2))}


def run_reduced_production(
    root: str | Path,
    *,
    packed_slurm: str | Path,
    submit: bool,
) -> dict:
    """
    Run the amended sequential D007 O30 search.

    When submit=False, perform preflight only and stop before proposing
    the next trial or submitting any Slurm job.
    """
    root = Path(root)
    packed_slurm = Path(packed_slurm)

    if not packed_slurm.is_file():
        raise RuntimeError(
            f"Missing D007 packed Slurm wrapper: {packed_slurm}"
        )

    controller = D007ReducedProductionController(root)

    if controller.state.get("metric") != D007_REDUCED_METRIC:
        raise RuntimeError(
            "D007 amended production state has wrong metric."
        )

    if controller.state.get("search_fidelity") != "reduced":
        raise RuntimeError(
            "D007 amended production state has wrong search fidelity."
        )

    if controller.state.get("search_cell_count") != 30:
        raise RuntimeError(
            "D007 amended production state has wrong search cell count."
        )

    if not submit:
        result = {
            "status": "preflight_passed",
            "submit": False,
            "metric": controller.state["metric"],
            "observation_count": len(
                controller.state["observations"]
            ),
            "next_trial_index": controller.state["next_trial_index"],
            "pending_trial_index": controller.state[
                "pending_trial_index"
            ],
            "packed_slurm": str(packed_slurm),
            "trial_proposed": False,
            "job_submitted": False,
        }

        return result

    while (
        controller.state["next_trial_index"]
        <= D007_CONTROLLER_CONFIG.total_trials
    ):
        trial_index = controller.state["next_trial_index"]
        tdir = trial_dir(root, trial_index)

        if controller.state.get("pending_trial_index") is None:
            controller.propose_next()

        if controller.state.get("pending_trial_index") != trial_index:
            raise RuntimeError(
                "D007 amended controller pending-trial mismatch."
            )

        prepare_trial_cells(
            root,
            trial_index,
            fidelity="reduced",
        )

        missing = incomplete_cells(
            root,
            trial_index,
            fidelity="reduced",
        )

        if missing:
            submission_path = (
                tdir / "packed_array_submitted.json"
            )

            if submission_path.exists():
                raise RuntimeError(
                    "D007 packed array was already submitted but "
                    "reduced cells remain incomplete; human "
                    "adjudication required."
                )

            command = [
                "sbatch",
                "--wait",
                (
                    "--export=ALL,"
                    f"D007_RUN_ROOT={root},"
                    f"D007_TRIAL_INDEX={trial_index}"
                ),
                str(packed_slurm),
            ]

            write_json_exclusive(
                tdir / "packed_array_submission_plan.json",
                {
                    "command": command,
                    "fidelity": "reduced",
                    "cell_count": 30,
                    "canonical_cell_indices": reduced_cell_indices(),
                    "pack_count": 8,
                    "max_fits_per_gpu": 4,
                },
            )

            subprocess.run(
                command,
                check=True,
            )

            write_json_exclusive(
                submission_path,
                {
                    "status": "completed",
                    "command": command,
                },
            )

        remaining = incomplete_cells(
            root,
            trial_index,
            fidelity="reduced",
        )

        if remaining:
            raise RuntimeError(
                "D007 packed execution returned with incomplete "
                f"reduced cells: {remaining}"
            )

        controller.complete_trial(trial_index)
        controller = D007ReducedProductionController(root)

    result = controller.state

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rehearse")
    p.add_argument("--root", required=True)
    p = sub.add_parser("run-production")
    p.add_argument("--root", required=True)
    p.add_argument("--cell-slurm", required=True)

    p = sub.add_parser("run-reduced-production")
    p.add_argument("--root", required=True)
    p.add_argument("--packed-slurm", required=True)
    p.add_argument(
        "--preflight-only",
        action="store_true",
    )
    args = parser.parse_args()
    if args.cmd == "rehearse":
        root = Path(args.root)
        if root.exists():
            raise SystemExit(f"rehearsal root exists: {root}")
        root.mkdir(parents=True)
        print(json.dumps(run_rehearsal(root), indent=2, sort_keys=True))
    elif args.cmd == "run-reduced-production":
        result = run_reduced_production(
            args.root,
            packed_slurm=args.packed_slurm,
            submit=not args.preflight_only,
        )
        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
        )

    elif args.cmd == "run-production":
        root = Path(args.root)
        controller = D007ProductionController(root)
        while controller.state["next_trial_index"] <= D007_CONTROLLER_CONFIG.total_trials:
            trial_index = controller.state["next_trial_index"]
            tdir = trial_dir(root, trial_index)
            if not (tdir / "proposal.json").exists():
                controller.propose_next()
            prepare_trial_cells(root, trial_index)
            if not (tdir / "array_submitted.json").exists():
                command = [
                    "sbatch",
                    "--wait",
                    f"--export=ALL,D007_RUN_ROOT={root},D007_TRIAL_INDEX={trial_index}",
                    args.cell_slurm,
                ]
                write_json_exclusive(tdir / "array_submission_plan.json", {"command": command, "cell_count": D007_TOTAL_CELLS})
                subprocess.run(command, check=True)
                write_json_exclusive(tdir / "array_submitted.json", {"status": "completed", "command": command})
            controller.complete_trial(trial_index)
            controller = D007ProductionController(root)
        print(json.dumps(load_state(root), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
