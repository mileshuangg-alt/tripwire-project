"""PXR-003 smoke test: fails loudly (no skips). Run locally in tripwire_cyp002.

Reads the real train/validation/holdout-structure inputs, fits tiny versions of both models on a
300-molecule subsample (2 trials, capped trees), runs search -> confirm -> selection -> freeze
assembly, and checks the artifacts' invariants. Reads no holdout labels. Writes nothing to disk.

  python -m pxr.scripts.pxr003_smoke
"""
from __future__ import annotations

import copy

import numpy as np

from pxr.scripts import pxr003_common as common
from pxr.scripts import pxr003_final as final
from pxr.scripts import pxr003_models as models
from pxr.scripts.pxr001_baseline import ID_COLUMN


def main() -> None:
    import lightgbm  # noqa: F401  (fails loudly if missing)

    cfg = copy.deepcopy(common.load_config())
    cfg["lgbm"].update(n_trials=2, n_estimators_cap=50, early_stopping_rounds=10)
    cfg["rf"].update(n_trials=2, n_estimators=20)
    cfg["seeds"]["confirm"] = cfg["seeds"]["confirm"][:2]
    cfg["confirm_top_k"] = 2

    train, validation, holdout_ids, hashes = common.load_cohort()
    assert hashes["train_csv"] == common.EXPECTED_TRAIN_SHA256
    assert (len(train), len(validation), len(holdout_ids)) == (2897, 621, 621)
    assert not set(train[ID_COLUMN]) & set(validation[ID_COLUMN])
    full = common.build_features(train, validation)
    assert full["x_train"].shape == (2897, 2048) and full["x_validation"].shape == (621, 2048)
    small = {"x_train": full["x_train"][:300], "y_train": full["y_train"][:300],
             "x_validation": full["x_validation"][:150], "y_validation": full["y_validation"][:150]}

    structures = common.load_holdout_structures()
    assert list(structures.columns) == [ID_COLUMN, "SMILES"] and len(structures) == 621
    x_holdout = common.holdout_matrix(structures)
    assert x_holdout.shape == (621, 2048)

    selections = {}
    for model in models.MODELS:
        trials = models.run_search(model, cfg, small, threads=4)
        assert len(trials) == 2 and np.isfinite(trials["val_mae"]).all()
        confirm = models.run_confirm(model, cfg, small, threads=4, trials=trials)
        assert len(confirm) == 2 * 2
        best = models.choose_best(confirm)
        chosen = confirm.loc[confirm["trial"].eq(best["trial"])].iloc[0]
        params = {n: (int(chosen[n]) if cfg[model]["ranges"][n][0] == "int" else float(chosen[n]))
                  for n in cfg[model]["ranges"]}
        for name, (_, low, high) in cfg[model]["ranges"].items():
            assert low <= params[name] <= high, (model, name, params[name])
        predict, _ = models.fit_model(model, params, cfg[model], small["x_train"], small["y_train"],
                                      small["x_validation"], small["y_validation"], 1, 4)
        prediction = predict(x_holdout)
        assert prediction.shape == (621,) and np.isfinite(prediction).all()
        selections[model] = {"model": model, "trial": best["trial"], "params": params,
                             "confirm_mean_val_mae": best["confirm_mean_val_mae"],
                             "config_sha256": "smoke", "input_hashes": hashes}

    frozen = final.build_frozen(selections["lgbm"], selections["rf"], "smoke")
    assert set(frozen["models"]) == {"lgbm", "rf"} and len(frozen["seeds_final"]) == 5
    print("PASS: PXR-003 smoke (cohort, features, search, confirmation, selection, freeze, holdout structure prediction)")


if __name__ == "__main__":
    main()
