from __future__ import annotations

import importlib.util
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT = Path(__file__).with_name("d012_implementation_qualification.py")
SPEC = importlib.util.spec_from_file_location(
    "d012_implementation_qualification",
    SCRIPT,
)
d012 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(d012)

OUTPUTS = d012.build_qualification()


def test_no_validation_or_public_labels_are_read() -> None:
    summary = OUTPUTS["summary"]
    source = SCRIPT.read_text(encoding="utf-8")

    assert summary["boundaries"]["validation_potency_labels_accessed"] is False
    assert summary["boundaries"]["public_challenge_labels_accessed"] is False
    assert "TRAIN_inhibition" not in source
    assert "unblinded" not in source.lower()
    assert "TEST-BLINDED" not in source


def test_validation_witnesses_map_uniquely_and_fail_closed() -> None:
    bridge = d012.build_openadmet_identity_bridge()
    for path in d012.manifest_paths():
        manifest = d012.load_manifest(path)
        validation_keys, unmapped = d012.validation_keys_for_manifest(manifest, bridge)
        assert validation_keys
        assert unmapped == []

    try:
        d012.validation_keys_for_manifest({"validation_names": ["missing"]}, bridge)
    except d012.FirewallClosedError:
        pass
    else:
        raise AssertionError("missing witness did not fail closed")

    ambiguous_rows = [
        {
            "source_family": "openadmet",
            "source_id": "OCNT-X",
            "identity_state": "represented",
            "full_standard_inchikey": "AAAAAAAAAAAAAA-UHFFFAOYSA-N",
        },
        {
            "source_family": "openadmet",
            "source_id": "OCNT-X",
            "identity_state": "represented",
            "full_standard_inchikey": "BBBBBBBBBBBBBB-UHFFFAOYSA-N",
        },
    ]
    try:
        d012.build_openadmet_identity_bridge(ambiguous_rows)
    except d012.FirewallClosedError:
        pass
    else:
        raise AssertionError("ambiguous witness did not fail closed")


def test_firewall_removes_every_matching_key_and_reproduces_invariants() -> None:
    summary = OUTPUTS["summary"]
    firewall = summary["firewall"]
    manifest_rows = OUTPUTS["manifest_rows"]

    assert firewall["manifest_count"] == 50
    assert firewall["cell_count"] == 150
    assert firewall["unmapped_validation_names"] == 0
    assert firewall["manifests_with_at_least_one_overlap"] == 49
    assert firewall["unique_auxiliary_keys_ever_in_validation"] == 33
    assert firewall["removed_instances_min"] == 0
    assert firewall["removed_instances_max"] == 14
    assert firewall["retained_instances_min"] == 4650
    assert firewall["retained_instances_max"] == 4664
    assert firewall["all_tasks_survive_every_manifest"] is True
    assert firewall["global_minimum_post_mask_task_size"] == 14
    assert max(int(row["tasks_below_15"]) for row in manifest_rows) <= 1
    assert max(int(row["tasks_below_16"]) for row in manifest_rows) <= 2

    instances = d012.load_auxiliary_instances()
    bridge = d012.build_openadmet_identity_bridge()
    row_with_overlap = next(row for row in manifest_rows if int(row["removed_instance_count"]) > 0)
    manifest = d012.load_manifest(d012.MANIFEST_DIR / f"{row_with_overlap['manifest_id']}.json")
    validation_keys, _ = d012.validation_keys_for_manifest(manifest, bridge)
    retained, removed = d012.apply_firewall(instances, validation_keys)
    removed_keys = {row["full_standard_inchikey"] for row in removed}

    assert all(row["full_standard_inchikey"] not in validation_keys for row in retained)
    assert removed_keys <= validation_keys
    assert len(removed) == int(row_with_overlap["removed_instance_count"])


def test_arm_b_exposes_every_retained_identity_once_and_step_distribution() -> None:
    manifest_rows = OUTPUTS["manifest_rows"]
    b_dist = Counter(int(row["b_updates_per_epoch"]) for row in manifest_rows)
    assert b_dist == Counter({387: 10, 388: 40})

    cell = OUTPUTS["cell_rows"][0]
    b_rows = [
        row for row in OUTPUTS["b_rows"]
        if row["manifest_id"] == cell["manifest_id"]
        and row["training_seed"] == cell["training_seed"]
    ]
    exposures = []
    for row in b_rows:
        exposures.extend(row["instance_ids"].split(";"))
        assert row["loss_reduction"] == "mean_mse"

    assert len(b_rows) == int(cell["b_updates_per_epoch"])
    assert len(exposures) == int(cell["retained_instance_count"])
    assert len(exposures) == len(set(exposures))


def test_arm_c_query_exposure_once_disjoint_and_step_distribution() -> None:
    manifest_rows = OUTPUTS["manifest_rows"]
    c_dist = Counter(int(row["c_updates_per_epoch"]) for row in manifest_rows)
    assert c_dist == Counter({856: 9, 857: 32, 858: 9})

    cell = OUTPUTS["cell_rows"][0]
    c_rows = [
        row for row in OUTPUTS["c_rows"]
        if row["manifest_id"] == cell["manifest_id"]
        and row["training_seed"] == cell["training_seed"]
    ]
    query_exposures = []
    task_sizes = defaultdict(int)
    for row in c_rows:
        query_ids = row["query_instance_ids"].split(";")
        query_keys = row["query_full_standard_inchikeys"].split(";")
        assert len(query_ids) == int(row["query_count"])
        assert len(query_keys) == int(row["query_count"])
        assert row["support_query_disjoint"] == "true"
        assert row["inner_adaptation_parameters"] == "scalar_head_weight_and_bias_only"
        assert row["inner_lr"] == "0.01"
        assert row["inner_steps"] == "1"
        assert row["differentiation"] == "exact_through_adaptation"
        assert row["support_loss"] == "mean_mse"
        assert row["query_loss"] == "mean_mse"
        query_exposures.extend(query_ids)
        task_sizes[row["task_unit_id"]] += int(row["query_count"])

    assert len(c_rows) == int(cell["c_updates_per_epoch"])
    assert len(query_exposures) == int(cell["retained_instance_count"])
    assert len(query_exposures) == len(set(query_exposures))
    assert min(task_sizes.values()) >= 14


def test_deterministic_replay_and_seed_changes_order_not_membership() -> None:
    instances = d012.load_auxiliary_instances()
    bridge = d012.build_openadmet_identity_bridge()
    path = d012.manifest_paths()[0]
    manifest = d012.load_manifest(path)
    validation_keys, _ = d012.validation_keys_for_manifest(manifest, bridge)
    retained, _ = d012.apply_firewall(instances, validation_keys)
    mid = d012.manifest_id(path)

    b_first = d012.construct_arm_b_epoch(retained, manifest_id_value=mid, training_seed=20261101)
    b_replay = d012.construct_arm_b_epoch(retained, manifest_id_value=mid, training_seed=20261101)
    b_changed = d012.construct_arm_b_epoch(retained, manifest_id_value=mid, training_seed=20261102)

    assert b_first == b_replay
    assert b_first != b_changed
    assert sorted(";".join(row["instance_ids"] for row in b_first).split(";")) == sorted(
        ";".join(row["instance_ids"] for row in b_changed).split(";")
    )

    c_first = d012.construct_arm_c_epoch(retained, manifest_id_value=mid, training_seed=20261101)
    c_changed = d012.construct_arm_c_epoch(retained, manifest_id_value=mid, training_seed=20261102)
    assert c_first != c_changed
    assert sorted(";".join(row["query_instance_ids"] for row in c_first).split(";")) == sorted(
        ";".join(row["query_instance_ids"] for row in c_changed).split(";")
    )


def test_scheduler_budgets_use_each_arm_own_update_count() -> None:
    for row in OUTPUTS["cell_rows"]:
        b_u = int(row["b_updates_per_epoch"])
        c_u = int(row["c_updates_per_epoch"])
        assert int(row["b_total_steps"]) == 50 * b_u
        assert int(row["b_warmup_steps"]) == 10 * b_u
        assert int(row["c_total_steps"]) == 50 * c_u
        assert int(row["c_warmup_steps"]) == 10 * c_u

    assert any(row["b_updates_per_epoch"] != row["c_updates_per_epoch"] for row in OUTPUTS["cell_rows"])


def test_no_old_16_molecule_cutoff_and_deterministic_outputs() -> None:
    summary = OUTPUTS["summary"]

    assert summary["boundaries"]["task_size_filtering"] is False
    assert summary["firewall"]["global_minimum_post_mask_task_size"] == 14
    first = d012.write_outputs(d012.build_qualification())
    second = d012.write_outputs(d012.build_qualification())
    assert first == second
