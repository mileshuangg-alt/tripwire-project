"""Tests for the PXR-002 multitask Chemprop recipe (D017 Amendment 1, D018).

Each test is tied to a concrete failure risk: auxiliary leakage into validation,
holdout or test groups; mis-calibrated task weights under Chemprop's real loss
reduction; a wrong selection metric; descriptor/target scaling leakage; an
incomplete grid. Chemprop-dependent tests skip where Chemprop is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr002_aux_data as aux  # noqa: E402
from pxr.scripts import pxr002_grid as grid  # noqa: E402
from pxr.scripts import pxr002_loss_weights as lw  # noqa: E402

DATA_DIR = REPO_ROOT / "pxr/data/openadmet_pxr_release_2026-11-15"
ASSIGNMENTS = REPO_ROOT / "pxr/artifacts/pxr001_split/pxr001_split_assignments.csv"


def _scaffold_works() -> bool:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles("c1ccccc1CC"), includeChirality=False)
        return True
    except Exception:
        return False


needs_scaffold = pytest.mark.skipif(not _scaffold_works(), reason="RDKit scaffold decomposition unavailable")


# ---------------------------------------------------------------- eligibility

def synthetic_tables():
    primary = pd.DataFrame(
        {
            "Molecule Name": ["P_train", "P_val", "P_hold"],
            "OCNT_ID": ["OCNT-1", "OCNT-2", "OCNT-3"],
            "SMILES": ["c1ccccc1CC", "c1ccncc1CC", "C1CCCCC1CC"],
            "pEC50": [5.0, 5.5, 6.0],
        }
    )
    assignments = pd.DataFrame(
        {"Molecule Name": ["P_train", "P_val", "P_hold"], "split": ["train", "validation", "holdout"]}
    )
    test = pd.DataFrame({"SMILES": ["c1ccc2ccccc2c1C"]})

    def rows(ocnt, smiles, value):
        return [
            {"OCNT_ID": ocnt, "SMILES": smiles, "compound_class": "Library", "concentration_M": c,
             "log2_fc_estimate": value + i}
            for i, c in enumerate([8.251e-06, 3.3e-05])
        ]

    single = pd.DataFrame(
        rows("OCNT-1", "c1ccccc1CC", 0.1)          # primary train, has aux labels
        + rows("OCNT-2", "c1ccncc1CC", 0.2)        # primary validation: aux must not train
        + rows("OCNT-3", "C1CCCCC1CC", 0.3)        # primary holdout: aux must not train
        + rows("OCNT-A", "c1ccccc1CCC", 0.4)       # shares train scaffold only -> eligible
        + rows("OCNT-B", "c1ccncc1CCC", 0.5)       # validation group -> excluded
        + rows("OCNT-C", "C1CCCCC1CCC", 0.6)       # holdout group -> excluded
        + rows("OCNT-D", "c1ccc2ccccc2c1CC", 0.7)  # official-test group -> excluded
        + rows("OCNT-E", "c1ccsc1C", 0.8)          # novel scaffold -> eligible
        + rows("OCNT-F", "CCCCO", 0.9)             # ringless novel -> eligible
        + [{"OCNT_ID": "CTRL", "SMILES": "CCO", "compound_class": "Positive Control",
            "concentration_M": 9.803e-07, "log2_fc_estimate": 3.0}]
    )
    return primary, assignments, test, single


@needs_scaffold
def test_aux_only_group_exclusions_follow_d014_groups():
    primary, assignments, test, single = synthetic_tables()
    elig = aux.build_eligibility(primary, assignments, test, single)
    table = elig["table"].set_index("OCNT_ID")
    for ocnt in ("OCNT-A", "OCNT-E", "OCNT-F"):
        assert table.loc[ocnt, "eligible"], ocnt
    for ocnt in ("OCNT-B", "OCNT-C", "OCNT-D"):
        assert not table.loc[ocnt, "eligible"], ocnt
    assert table.loc["OCNT-D", "reason"] == "aux_only_group_contains_test_structure"
    assert "CTRL" not in table.index, "positive controls must be excluded"


@needs_scaffold
def test_frames_keep_validation_holdout_aux_out_of_training():
    primary, assignments, test, single = synthetic_tables()
    elig = aux.build_eligibility(primary, assignments, test, single)
    train, validation = aux.assemble_frames(primary, assignments, elig)
    assert set(train["row_id"]) == {"P_train", "OCNT-A", "OCNT-E", "OCNT-F"}
    assert list(validation["row_id"]) == ["P_val"]
    assert validation[list(aux.AUX_TASKS)].isna().all().all(), "validation aux labels must be masked"
    assert "P_hold" not in set(train["row_id"]) | set(validation["row_id"])
    # primary-train keeps its own aux labels; aux-only rows carry no primary label
    assert train.loc[train["row_id"].eq("P_train"), "log2fc_8p25uM"].iloc[0] == pytest.approx(0.1)
    assert train.loc[train["role"].eq("aux_only"), "pEC50"].isna().all()
    counts = aux.label_counts(train)
    assert counts["pEC50"] == 1 and counts["log2fc_8p25uM"] == 4 and counts["log2fc_99uM"] == 0


def test_duplicate_library_rows_are_refused():
    _, _, _, single = synthetic_tables()
    duplicated = pd.concat([single, single.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        aux._library_wide(duplicated)


def test_unknown_library_concentration_is_refused():
    _, _, _, single = synthetic_tables()
    single.loc[0, "concentration_M"] = 1.0e-4
    with pytest.raises(ValueError, match="unexpected library concentration"):
        aux._library_wide(single)


@needs_scaffold
@pytest.mark.skipif(not (DATA_DIR.exists() and ASSIGNMENTS.exists()), reason="release not present")
def test_real_data_frozen_groups_reproduce_and_no_leakage():
    primary = pd.read_csv(DATA_DIR / "pxr-challenge_TRAIN.csv")
    assignments = pd.read_csv(ASSIGNMENTS)
    aux.verify_frozen_groups(primary, assignments)
    test = pd.read_csv(DATA_DIR / "pxr-challenge_TEST_BLINDED.csv")
    single = pd.read_csv(DATA_DIR / "pxr-challenge_single_concentration_TRAIN.csv")
    elig = aux.build_eligibility(primary, assignments, test, single)
    train, validation = aux.assemble_frames(primary, assignments, elig)
    assert (len(validation), int(assignments["split"].eq("train").sum())) == (621, 2897)
    assert int(train["role"].eq("primary_train").sum()) == 2897
    held = set(assignments.loc[assignments["split"].ne("train"), "Molecule Name"])
    assert not (set(train["row_id"]) & held)
    # no eligible aux-only molecule may share an exact connectivity block with a non-train
    # primary molecule or a blinded-test molecule
    from pxr.scripts.pxr001_split import scaffold_and_connectivity

    aux_ids = set(train.loc[train["role"].eq("aux_only"), "row_id"])
    aux_smiles = train.loc[train["role"].eq("aux_only"), "SMILES"]
    _, aux_blocks = scaffold_and_connectivity(aux_smiles)
    other = assignments.loc[assignments["split"].ne("train")]
    other_smiles = primary.merge(other[["Molecule Name"]], on="Molecule Name")["SMILES"]
    _, other_blocks = scaffold_and_connectivity(pd.concat([other_smiles, test["SMILES"]]))
    assert not (set(aux_blocks) & set(other_blocks)), f"{len(aux_ids)} aux-only molecules checked"


# ------------------------------------------------------------- loss weighting

def test_calibration_formula_gives_65_35_nominal_allocation():
    counts = {"a": 9737, "b": 8540, "c": 606}
    w = lw.task_weight_vector(2897, counts)
    alloc = lw.nominal_allocation(w, [2897, *counts.values()])
    assert alloc[0] == pytest.approx(0.65) and sum(alloc[1:]) == pytest.approx(0.35)
    assert w[1] == w[2] == w[3], "equal coefficients across auxiliary heads"
    assert alloc[1] / alloc[2] == pytest.approx(9737 / 8540), "count-proportional among auxiliaries"
    assert w[1] != 0.35 and w[1] != 0.35 / 0.65, "raw shares must not be used as weights"


def test_calibration_rejects_empty_head():
    with pytest.raises(ValueError):
        lw.calibrate_aux_coefficient(100, {"a": 10, "b": 0})


def test_chemprop_reduction_matches_calibration_assumption():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("chemprop.nn")
    n_primary, n_aux = 40, [30, 20, 10]
    n = 60
    mask = torch.zeros(n, 4, dtype=torch.bool)
    mask[:n_primary, 0] = True
    start = 0
    for head, count in enumerate(n_aux, start=1):
        mask[start : start + count, head] = True
        start += count
    preds = torch.zeros(n, 4)
    targets = torch.ones(n, 4)
    w = lw.task_weight_vector(n_primary, {"a": n_aux[0], "b": n_aux[1], "c": n_aux[2]})

    def loss(task_weights):
        metric = nn.MSE(task_weights=task_weights)
        metric.update(preds, targets, mask)
        return float(metric.compute())

    # unit squared error everywhere: reduction is sum(w*mask)/sum(mask), i.e. the weights
    # scale the numerator only -- exactly the semantics the calibration assumes
    expected = (w[0] * n_primary + sum(wi * c for wi, c in zip(w[1:], n_aux))) / mask.sum().item()
    assert loss(w) == pytest.approx(expected)
    equal = loss([1.0] * 4)
    assert equal == pytest.approx(1.0), "equal weights: every observed label counts once"
    primary_only = nn.MSE(task_weights=[1.0, 0.0, 0.0, 0.0])
    primary_only.update(preds, targets, mask)
    assert float(primary_only.compute()) == pytest.approx(n_primary / mask.sum().item())


# ------------------------------------------------------------- selection metric

def test_primary_metric_is_primary_column_mae_and_spearman():
    from pxr.scripts.pxr002_train import primary_validation_metrics

    preds = np.array([4.0, 5.0, 6.0, 7.0])
    targets = np.array([4.5, 5.0, 5.0, 8.0])
    m = primary_validation_metrics(preds, targets)
    assert m["mae"] == pytest.approx((0.5 + 0 + 1.0 + 1.0) / 4)
    assert m["spearman"] == pytest.approx(pd.Series(preds).corr(pd.Series(targets), method="spearman"))
    with pytest.raises(ValueError):
        primary_validation_metrics(preds, np.array([1.0, np.nan, 2.0, 3.0]))


# ------------------------------------------------------------- scaling, descriptors

def test_scalers_fit_on_train_only_and_descriptors_finite():
    pytest.importorskip("chemprop")
    pytest.importorskip("descriptastorus")
    from pxr.scripts.pxr002_dataset import TASK_COLUMNS, build_datasets, compute_descriptors

    nan = np.nan
    train = pd.DataFrame(
        {"row_id": ["a", "b", "c", "d"], "role": ["primary_train"] * 2 + ["aux_only"] * 2,
         "SMILES": ["CCOc1ccccc1", "CCNc1ccncc1", "CCCCO", "c1ccsc1C"],
         "pEC50": [4.0, 6.0, nan, nan], "log2fc_8p25uM": [0.0, 1.0, 2.0, nan],
         "log2fc_33uM": [nan, 1.0, 2.0, 3.0], "log2fc_99uM": [nan, nan, 5.0, 7.0]}
    )
    val = pd.DataFrame(
        {"row_id": ["v"], "role": ["primary_validation"], "SMILES": ["CCOc1ccncc1"],
         "pEC50": [100.0], "log2fc_8p25uM": [nan], "log2fc_33uM": [nan], "log2fc_99uM": [nan]}
    )
    sets = build_datasets(
        train, val, compute_descriptors(train["SMILES"].tolist()), compute_descriptors(val["SMILES"].tolist())
    )
    assert sets["y_scaler"].mean_[0] == pytest.approx(5.0), "pEC50 scaler must ignore validation (100.0)"
    assert sets["y_scaler"].mean_[3] == pytest.approx(6.0)
    assert sets["xd_dim"] > 0
    # validation targets stay in original units
    assert np.asarray(sets["validation"].Y)[0, 0] == pytest.approx(100.0)
    assert list(TASK_COLUMNS)[0] == "pEC50"


# ------------------------------------------------------------- grid

def test_grid_is_the_full_36_configuration_product():
    configs = grid.build_grid()
    assert len(configs) == 36 and len({c.name for c in configs}) == 36
    assert {c.hidden_size for c in configs} == {300, 600}
    assert {c.depth for c in configs} == {3, 4, 5}
    assert {c.ffn_layers for c in configs} == {1, 2}
    assert {c.dropout for c in configs} == {0.0, 0.1, 0.2}


def test_selection_ranks_by_mean_mae_and_plans_only_missing_seeds():
    stage_a = {f"c{i}": 1.0 + i / 10 for i in range(10)}
    top = grid.select_top(stage_a)
    assert top == ["c0", "c1", "c2"]
    extra = grid.stage_b_plan(top)
    assert len(extra) == 6 and all(r["seed"] != grid.COMMON_SEED for r in extra)
    ranked = grid.rank_by_mae({"x": [1.0, 1.2, 1.1], "y": [0.9, 1.5, 1.4]})
    assert ranked[0][0] == "x"


# ------------------------------------------------------------- cohort v2 amendment

from pxr.scripts import pxr002_artifact_v2 as v2  # noqa: E402
from pxr.scripts import pxr002_dataset as dset  # noqa: E402

NAMES = [f"d{i}" for i in range(dset.N_DESCRIPTORS)]
BAD_NAMES = list(v2.APPROVED_DESCRIPTORS)


def _v2_inputs(bad_row="OADMET-0003727", bad_names=BAD_NAMES):
    names = list(NAMES)
    for k, n in enumerate(BAD_NAMES):  # place the approved names at distinct positions
        names[39 + 2 * k] = n
    train = pd.DataFrame(
        {
            "row_id": ["OADMET-0000001", bad_row, "OCNT-9"],
            "role": ["primary_train", "primary_train", "aux_only"],
            "SMILES": ["CCO", "CC[P](CC)(CC)=[Au][S]C", "CCN"],
            "pEC50": [5.0, 6.0, np.nan],
            "log2fc_8p25uM": [0.1, 0.2, 0.3],
            "log2fc_33uM": [np.nan, 0.4, 0.5],
            "log2fc_99uM": [np.nan, np.nan, 0.6],
        }
    )
    validation = pd.DataFrame(
        {"row_id": ["V1"], "role": ["primary_validation"], "SMILES": ["CCC"], "pEC50": [5.5],
         "log2fc_8p25uM": [np.nan], "log2fc_33uM": [np.nan], "log2fc_99uM": [np.nan]}
    )
    train_x = np.arange(3 * dset.N_DESCRIPTORS, dtype=float).reshape(3, -1)
    for n in bad_names:
        train_x[1, names.index(n)] = np.nan
    val_x = np.ones((1, dset.N_DESCRIPTORS))
    return train, validation, train_x, val_x, names


def test_amendment_drops_exactly_the_approved_row_and_nothing_else():
    train, validation, train_x, val_x, names = _v2_inputs()
    kept, kept_x, amendment, removed = v2.apply_descriptor_exclusion(train, validation, train_x, val_x, names)
    assert kept["row_id"].tolist() == ["OADMET-0000001", "OCNT-9"]
    assert kept_x.shape == (2, dset.N_DESCRIPTORS) and np.isfinite(kept_x).all()
    np.testing.assert_array_equal(kept_x[0], train_x[0]), "other rows keep their exact cached values and order"
    assert amendment["row_id"].tolist() == ["OADMET-0003727"]
    assert amendment.loc[0, "non_finite_descriptors"] == ";".join(sorted(BAD_NAMES))
    # the amended counts lose exactly the removed molecule's labels
    c1, c2 = aux.label_counts(train), aux.label_counts(kept)
    assert (c1["pEC50"] - c2["pEC50"], c1["log2fc_8p25uM"] - c2["log2fc_8p25uM"], c1["log2fc_33uM"] - c2["log2fc_33uM"]) == (1, 1, 1)
    # recalculated coefficient preserves the nominal 65:35 allocation on the amended counts
    w = lw.task_weight_vector(c2["pEC50"], {t: c2[t] for t in aux.AUX_TASKS})
    alloc = lw.nominal_allocation(w, [c2["pEC50"], *[c2[t] for t in aux.AUX_TASKS]])
    assert alloc[0] == pytest.approx(0.65)


def test_amendment_refuses_any_other_non_finite_row_or_descriptor():
    train, validation, train_x, val_x, names = _v2_inputs()
    other = train_x.copy()
    other[0, 5] = np.nan  # a second, unapproved molecule
    with pytest.raises(ValueError, match="differ from the approved exclusion"):
        v2.apply_descriptor_exclusion(train, validation, other, val_x, names)
    extra = train_x.copy()
    extra[1, 100] = np.nan  # an unapproved extra descriptor on the approved row
    with pytest.raises(ValueError, match="approved set"):
        v2.apply_descriptor_exclusion(train, validation, extra, val_x, names)
    bad_val = val_x.copy()
    bad_val[0, 3] = np.nan
    with pytest.raises(ValueError, match="validation"):
        v2.apply_descriptor_exclusion(train, validation, train_x, bad_val, names)


def test_artifact_v2_roundtrip_hash_check_and_tamper_detection(tmp_path):
    train, validation, train_x, val_x, names = _v2_inputs()
    kept, kept_x, amendment, _ = v2.apply_descriptor_exclusion(train, validation, train_x, val_x, names)
    out = tmp_path / "v2"
    v2.write_artifact_v2(out, kept, validation, kept_x, val_x, amendment, {"cohort_version": "v2"}, ["test"])
    art = dset.load_artifact(out)
    assert len(art["train"]) == 2 and art["train_x"].shape == (2, dset.N_DESCRIPTORS)
    assert art["amendment"]["row_id"].tolist() == ["OADMET-0003727"]
    with pytest.raises(SystemExit):  # an existing artifact is never overwritten
        v2.write_artifact_v2(out, kept, validation, kept_x, val_x, amendment, {}, ["test"])
    with open(out / dset.TRAIN_FRAME, "a") as handle:
        handle.write("tamper\n")
    with pytest.raises(RuntimeError, match="does not match SHA256SUMS"):
        dset.load_artifact(out)


def test_loader_refuses_nonfinite_cache_and_amended_row_in_cohort(tmp_path):
    train, validation, train_x, val_x, names = _v2_inputs()
    kept, kept_x, amendment, _ = v2.apply_descriptor_exclusion(train, validation, train_x, val_x, names)
    nan_x = kept_x.copy()
    nan_x[0, 0] = np.nan
    out = tmp_path / "nan"
    v2.write_artifact_v2(out, kept, validation, nan_x, val_x, amendment, {}, ["test"])
    with pytest.raises(ValueError, match="non-finite"):
        dset.load_artifact(out)
    readded = pd.concat([kept, train.iloc[[1]]], ignore_index=True)
    readded_x = np.vstack([kept_x, np.ones((1, dset.N_DESCRIPTORS))])
    out2 = tmp_path / "readded"
    v2.write_artifact_v2(out2, readded, validation, readded_x, val_x, amendment, {}, ["test"])
    with pytest.raises(ValueError, match="amended-out"):
        dset.load_artifact(out2)


# ------------------------------------------------------------- validation units (pilot finding)

def test_validation_mae_is_in_original_units_not_scaled_units():
    """Regression test for the 2026-11-18 pilot finding.

    Chemprop puts the output transform in train mode during validation, so raw
    predictions are in scaled units. With targets near 5 and an essentially untrained
    model, a correct primary MAE is O(1); the bug reports roughly the target mean (~5).
    """
    pytest.importorskip("chemprop")
    pytest.importorskip("descriptastorus")
    lightning = pytest.importorskip("lightning")
    from chemprop import data

    from pxr.scripts.pxr002_dataset import build_datasets, compute_descriptors
    from pxr.scripts.pxr002_train import PXR002Config, build_model

    smiles = ["CCOc1ccccc1", "CCNc1ccncc1", "CCCCO", "c1ccsc1C", "CCOC(C)=O", "CC(C)Cc1ccccc1", "CCN(CC)CC", "c1ccc2ccccc2c1"]
    rng = np.random.default_rng(0)
    p = 5.0 + rng.normal(0, 0.5, len(smiles))
    train = pd.DataFrame({"row_id": [f"t{i}" for i in range(len(smiles))], "role": "primary_train", "SMILES": smiles,
                          "pEC50": p, "log2fc_8p25uM": np.nan, "log2fc_33uM": np.nan, "log2fc_99uM": np.nan})
    val = train.iloc[:4].copy()
    val["role"] = "primary_validation"
    val["row_id"] = [f"v{i}" for i in range(4)]
    sets = build_datasets(train, val, compute_descriptors(train["SMILES"].tolist()), compute_descriptors(val["SMILES"].tolist()))
    model = build_model(PXR002Config(300, 2, 1, 0.0), [1.0, 0.1, 0.1, 0.1], sets["y_scaler"], sets["xd_scaler"], sets["xd_dim"])
    loader = data.build_dataloader(sets["validation"], batch_size=4, num_workers=0, shuffle=False)
    trainer = lightning.pytorch.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                                        enable_progress_bar=False, enable_model_summary=False)
    trainer.validate(model, loader)
    mae = model.validation_history[-1]["mae"]
    assert mae < 2.0, f"validation MAE {mae} looks like scaled-unit predictions against original-unit targets"
    assert model.primary_mean == pytest.approx(float(sets["y_scaler"].mean_[0]))


# ------------------------------------------------------------- stage A runner

def test_stage_a_runner_covers_the_grid_and_never_reuses_an_output_root(tmp_path):
    from pxr.scripts import pxr002_stage_a_runner as runner

    configs = grid.build_grid()
    commands = [runner.build_command(c, grid.COMMON_SEED, Path("art"), tmp_path / "out", "gpu") for c in configs]
    out_dirs = {cmd[cmd.index("--out-dir") + 1] for cmd in commands}
    assert len(commands) == 36 and len(out_dirs) == 36, "one distinct output directory per configuration"
    for cmd in commands:
        assert cmd[cmd.index("--seed") + 1] == str(grid.COMMON_SEED)
        assert "-I" in cmd and cmd[cmd.index("--accelerator") + 1] == "gpu"
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        runner.run_stage_a(Path("art"), existing, "gpu")


# ------------------------------------------------------------- stage B runner

def _write_stage_a(root, maes):
    import json

    for name, mae in maes.items():
        d = root / name
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps(
            {"configuration": name, "seed": grid.COMMON_SEED, "best_primary_validation_mae": mae}))


def test_stage_b_selects_top_three_and_plans_only_new_seeds(tmp_path):
    from pxr.scripts import pxr002_stage_b_runner as sb

    names = [c.name for c in grid.build_grid()]
    maes = {n: 0.6 + i / 1000 for i, n in enumerate(names)}
    maes[names[7]], maes[names[20]], maes[names[3]] = 0.50, 0.51, 0.52  # deliberately out of order
    _write_stage_a(tmp_path / "a", maes)
    plan = sb.plan_from_stage_a(tmp_path / "a")
    assert plan["top"] == [names[7], names[20], names[3]]
    assert len(plan["extra_runs"]) == 6
    assert all(r["seed"] in grid.STAGE_B_SEEDS and r["seed"] != grid.COMMON_SEED for r in plan["extra_runs"])


def test_stage_b_requires_all_36_stage_a_results(tmp_path):
    from pxr.scripts import pxr002_stage_b_runner as sb

    _write_stage_a(tmp_path / "a", {"h300_d3_f1_p0": 0.55})
    with pytest.raises(RuntimeError, match="expected 36"):
        sb.plan_from_stage_a(tmp_path / "a")


def test_stage_b_aggregate_uses_stage_a_value_plus_new_seeds():
    from pxr.scripts import pxr002_stage_b_runner as sb

    plan = {"top": ["x", "y", "z"], "stage_a_mae": {"x": 0.50, "y": 0.52, "z": 0.53}}
    new = {("x", 20261118): 0.60, ("x", 20261119): 0.62,   # x was best at one seed only
           ("y", 20261118): 0.53, ("y", 20261119): 0.54,
           ("z", 20261118): 0.55, ("z", 20261119): 0.56}
    ranking = sb.aggregate(plan, new)
    # means: y 0.5300, z 0.5467, x 0.5733 -- x has the best single value but the worst mean
    assert [n for n, _ in ranking] == ["y", "z", "x"]
    assert dict(ranking)["x"] == pytest.approx((0.50 + 0.60 + 0.62) / 3)
    assert dict(ranking)["y"] == pytest.approx((0.52 + 0.53 + 0.54) / 3)
    assert dict(ranking)["z"] == pytest.approx((0.53 + 0.55 + 0.56) / 3)
    with pytest.raises(KeyError):
        sb.aggregate(plan, {("x", 20261118): 0.6})


def test_stage_b_refuses_an_existing_output_root(tmp_path):
    from pxr.scripts import pxr002_stage_b_runner as sb

    names = [c.name for c in grid.build_grid()]
    _write_stage_a(tmp_path / "a", {n: 0.6 + i / 1000 for i, n in enumerate(names)})
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        sb.run_stage_b(Path("art"), tmp_path / "a", tmp_path / "out", "gpu")
