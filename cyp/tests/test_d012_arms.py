from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from d012_arms import (
    D012_PLAN,
    bounded_smoke,
    load_d012_instances,
    read_tsv,
    scheduler_budget_for_cell,
)


def test_qualified_plans_are_consumed_and_scheduler_budget_matches() -> None:
    budget = scheduler_budget_for_cell("rep_01_fold_01", 20261101)

    assert budget["b_updates_per_epoch"] == "388"
    assert budget["b_total_steps"] == "19400"
    assert budget["b_warmup_steps"] == "3880"
    assert budget["c_updates_per_epoch"] == "857"
    assert budget["c_total_steps"] == "42850"
    assert budget["c_warmup_steps"] == "8570"

    assert (D012_PLAN / "d012_arm_b_epoch_plan.tsv").exists()
    assert (D012_PLAN / "d012_arm_c_epoch_plan.tsv").exists()


def test_no_protected_labels_enter_auxiliary_training_plans() -> None:
    instances = load_d012_instances()
    assert len(instances) == 4664
    assert all(row["available_for_d012_auxiliary_population"] == "true" for row in instances)

    firewall = read_tsv(D012_PLAN / "d012_manifest_firewall.tsv")
    rep01 = next(row for row in firewall if row["manifest_id"] == "rep_01_fold_01")
    removed_keys = set(rep01["removed_full_standard_inchikeys"].split(";"))
    removed_keys.discard("")

    b_rows = [
        row for row in read_tsv(D012_PLAN / "d012_arm_b_epoch_plan.tsv")
        if row["manifest_id"] == "rep_01_fold_01" and row["training_seed"] == "20261101"
    ]
    c_rows = [
        row for row in read_tsv(D012_PLAN / "d012_arm_c_epoch_plan.tsv")
        if row["manifest_id"] == "rep_01_fold_01" and row["training_seed"] == "20261101"
    ]

    b_keys = {key for row in b_rows for key in row["full_standard_inchikeys"].split(";")}
    c_query_keys = {key for row in c_rows for key in row["query_full_standard_inchikeys"].split(";")}

    assert not (b_keys & removed_keys)
    assert not (c_query_keys & removed_keys)


def test_bounded_smoke_exercises_b_and_c_paths() -> None:
    result = bounded_smoke("rep_01_fold_01", 20261101)

    assert result["arm_b"]["target_update_encoder_changed"] is True
    assert result["arm_b"]["validation_used_for_checkpoint_selection"] is True
    assert result["arm_b"]["validation_score"] >= 0

    assert result["arm_c"]["encoder_grad_l1"] > 0
    assert result["arm_c"]["target_adaptation"]["encoder_unchanged"] is True
    assert result["arm_c"]["target_adaptation"]["head_changed"] is True
    assert result["arm_c"]["validation_used_for_checkpoint_selection"] is False
    assert result["arm_c"]["target_adaptation"]["validation_score"] >= 0

    assert result["target_scaling"]["fit_on_training_names_only"] is True
    assert result["qualified_plan_consumed"]["b_updates_per_epoch"] == 388
    assert result["qualified_plan_consumed"]["c_updates_per_epoch"] == 857


def test_deterministic_replay_and_changed_seed() -> None:
    first = bounded_smoke("rep_01_fold_01", 20261101)
    second = bounded_smoke("rep_01_fold_01", 20261101)
    changed = bounded_smoke("rep_01_fold_01", 20261102)

    assert first == second
    assert first["qualified_plan_consumed"] != changed["qualified_plan_consumed"] or first["arm_b"]["auxiliary_loss"] != changed["arm_b"]["auxiliary_loss"]
    assert first["scheduler_budget"]["retained_instance_count"] == changed["scheduler_budget"]["retained_instance_count"]


def test_smoke_result_is_json_serializable(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(bounded_smoke(), sort_keys=True), encoding="utf-8")
    assert path.read_text(encoding="utf-8").startswith("{")
