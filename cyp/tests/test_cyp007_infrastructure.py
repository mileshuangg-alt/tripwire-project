from __future__ import annotations

import math

import pytest

from cyp007_infrastructure import (
    D007_CONTROLLER_CONFIG,
    D007_SEARCH_SPACE,
    D007_TRAINING_SEEDS,
    aggregate_d007_objective,
    c0_for_d007,
    config_from_trial,
    controller_observation,
    d007_controller_config,
    d007_search_space,
    generate_d007_split_manifests,
    validate_d007_split_manifests,
    validate_search_space,
    write_split_manifests,
)


SPLIT_ASSIGNMENTS = "cyp/artifacts/cyp_baseline/molecule_split_assignments.csv"


def _synthetic_results():
    rows = []
    value = 0.5
    for repetition in range(1, 6):
        for fold in range(1, 11):
            for seed in D007_TRAINING_SEEDS:
                rows.append(
                    {
                        "repetition": repetition,
                        "fold": fold,
                        "seed": seed,
                        "macro_st_rae": value,
                    }
                )
                value += 0.001
    return rows


def test_search_space_has_exactly_d007_dimensions_and_ranges():
    space = d007_search_space()
    validate_search_space(space)
    assert list(space) == list(D007_SEARCH_SPACE)
    assert len(space) == 10
    assert space["init_lr_ratio"] == {
        "kind": "loguniform",
        "low": 1e-2,
        "high": 1.0,
    }
    assert space["final_lr_ratio"] == {
        "kind": "loguniform",
        "low": 1e-2,
        "high": 1.0,
    }
    assert space["warmup_epochs"] == {
        "kind": "int",
        "low": 1,
        "high": 25,
    }


def test_ratio_to_absolute_lr_and_warmup_match_chemprop_native_fields():
    sampled = {
        "message_passing_depth": 5,
        "message_hidden_dim": 700,
        "ffn_hidden_dim": 900,
        "ffn_num_layers": 2,
        "dropout": 0.15,
        "batch_size": 32,
        "max_lr": 0.004,
        "init_lr_ratio": 0.25,
        "final_lr_ratio": 0.05,
        "warmup_epochs": 11,
    }
    config = config_from_trial(name="D007-T000", sampled=sampled, seed=123)
    assert config.initial_lr == pytest.approx(0.001)
    assert config.final_lr == pytest.approx(0.0002)
    assert config.max_lr == pytest.approx(0.004)
    assert config.warmup_epochs == 11


def test_c0_for_d007_reuses_c0_scientific_settings_with_declared_seed():
    config = c0_for_d007(seed=D007_TRAINING_SEEDS[0])
    assert config.name == "C0"
    assert config.message_passing_depth == 3
    assert config.message_hidden_dim == 300
    assert config.ffn_hidden_dim == 300
    assert config.batch_size == 64
    assert config.seed == D007_TRAINING_SEEDS[0]


def test_generates_exactly_50_cluster_aware_development_splits():
    manifests = generate_d007_split_manifests(SPLIT_ASSIGNMENTS)
    summary = validate_d007_split_manifests(manifests, SPLIT_ASSIGNMENTS)
    assert summary == {
        "manifest_count": 50,
        "development_rows": 4169,
        "protected_test_rows": 736,
        "protected_test_rows_in_d007_partitions": 0,
        "training_seeds": list(D007_TRAINING_SEEDS),
    }
    assert {(m["repetition"], m["fold"]) for m in manifests} == {
        (repetition, fold)
        for repetition in range(1, 6)
        for fold in range(1, 11)
    }


def test_splits_are_deterministic_and_common_for_all_configurations():
    first = generate_d007_split_manifests(SPLIT_ASSIGNMENTS)
    second = generate_d007_split_manifests(SPLIT_ASSIGNMENTS)
    assert first == second
    assert {tuple(m["training_seeds"]) for m in first} == {D007_TRAINING_SEEDS}


def test_split_manifest_write_refuses_existing_artifacts(tmp_path):
    manifests = generate_d007_split_manifests(SPLIT_ASSIGNMENTS)
    output_dir = tmp_path / "splits"
    write_split_manifests(manifests, output_dir)
    assert len(list(output_dir.glob("rep_*_fold_*.json"))) == 50
    with pytest.raises(FileExistsError):
        write_split_manifests(manifests, output_dir)


def test_objective_aggregates_exactly_150_finite_cells():
    rows = _synthetic_results()
    expected = sum(row["macro_st_rae"] for row in rows) / 150
    assert aggregate_d007_objective(rows) == pytest.approx(expected)


def test_objective_fails_closed_on_missing_duplicate_and_nonfinite_cells():
    rows = _synthetic_results()
    with pytest.raises(ValueError, match="Missing"):
        aggregate_d007_objective(rows[:-1])

    duplicate = rows + [dict(rows[0])]
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_d007_objective(duplicate)

    nonfinite = [dict(row) for row in rows]
    nonfinite[0]["macro_st_rae"] = math.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        aggregate_d007_objective(nonfinite)


def test_controller_receives_only_completed_scalar_objective():
    objective = aggregate_d007_objective(_synthetic_results())
    observation = controller_observation(7, objective)
    assert observation == {"trial_index": 7, "objective": objective}
    assert "fold" not in observation
    assert "seed" not in observation


def test_controller_configuration_is_fresh_128_trial_tpe_without_pruning():
    config = d007_controller_config()
    assert config == {
        "total_trials": 128,
        "random_startup_trials": 32,
        "tpe_trials": 96,
        "controller_rng_seed": D007_CONTROLLER_CONFIG.controller_rng_seed,
        "pruning": "none",
        "imported_observations": 0,
        "objective": "mean_validation_macro_st_rae_50_folds_x_3_seeds",
    }
