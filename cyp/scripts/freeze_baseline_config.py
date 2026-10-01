from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TUNING_DIR = REPO_ROOT / "artifacts" / "cyp_baseline" / "tuning"
OUT_PATH = REPO_ROOT / "artifacts" / "cyp_baseline" / "frozen_baseline_config.json"

TARGETS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

EXPECTED = {
    "CYP1A2_pIC50_direct_inhibition": {
        "best_trial": 15,
        "validation_ST_RAE": 0.8878237148303819,
        "best_iteration": 195,
        "learning_rate": 0.007493581432955413,
        "num_leaves": 41,
        "min_child_weight": 0.16006543580783064,
        "subsample": 0.5309983063408124,
        "search_seed_provenance": "explicit_isoform_seed",
    },
    "CYP2C9_pIC50_direct_inhibition": {
        "best_trial": 30,
        "validation_ST_RAE": 0.7991426892733124,
        "best_iteration": 69,
        "learning_rate": 0.07921505531874473,
        "num_leaves": 25,
        "min_child_weight": 0.04277500477345989,
        "subsample": 0.7985177920408508,
        "search_seed_provenance": "explicit_isoform_seed",
    },
    "CYP2D6_pIC50_direct_inhibition": {
        "best_trial": 14,
        "validation_ST_RAE": 0.8730698116139428,
        "best_iteration": 108,
        "learning_rate": 0.04023960497447025,
        "num_leaves": 61,
        "min_child_weight": 0.12177920971919333,
        "subsample": 0.6037007057447841,
        "search_seed_provenance": "explicit_isoform_seed",
    },
    "CYP3A4_pIC50_direct_inhibition": {
        "best_trial": 24,
        "validation_ST_RAE": 0.6547707974523649,
        "best_iteration": 570,
        "learning_rate": 0.0593598469086799,
        "num_leaves": 24,
        "min_child_weight": 0.07660312692852367,
        "subsample": 0.9132423504906277,
        "search_seed_provenance": "pre_seed-map_corrected-run",
    },
}

config = {
    "status": "FROZEN_FOR_HELD_OUT_TEST_EVALUATION",
    "representation": {
        "type": "Morgan",
        "radius": 2,
        "n_bits": 2048,
    },
    "model": {
        "type": "LightGBM_GBDT",
        "n_estimators": 2000,
        "bagging_freq": 1,
        "max_depth": "default",
        "early_stopping_rounds": 50,
    },
    "selection": {
        "metric": "validation_ST-RAE",
        "n_random_trials_per_isoform": 30,
        "test_set_used_during_tuning": False,
    },
    "split": {
        "type": "analog_cluster_holdout",
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "tanimoto_threshold": 0.70,
        "random_seed": 20261001,
        "assignment_artifact": (
            "artifacts/cyp_baseline/"
            "molecule_split_assignments.csv"
        ),
    },
    "isoforms": EXPECTED,
    "provenance_note": (
        "CYP3A4 was not rerun after the explicit isoform-seed map "
        "was introduced. Its existing corrected 30-trial run is retained "
        "as authoritative because no preregistered basis exists for "
        "choosing between alternative searches."
    ),
}

for target in TARGETS:
    trial_csv = TUNING_DIR / f"{target}_trials.csv"
    best_json = TUNING_DIR / f"{target}_best.json"

    if not trial_csv.exists():
        raise SystemExit(f"Missing trial artifact: {trial_csv}")

    if not best_json.exists():
        raise SystemExit(f"Missing winner artifact: {best_json}")

    trials = __import__("pandas").read_csv(trial_csv)
    if len(trials) != 30:
        raise SystemExit(
            f"{target}: expected 30 trials, found {len(trials)}"
        )

with open(OUT_PATH, "w") as f:
    json.dump(config, f, indent=2)

print(f"Saved frozen baseline configuration: {OUT_PATH}")
