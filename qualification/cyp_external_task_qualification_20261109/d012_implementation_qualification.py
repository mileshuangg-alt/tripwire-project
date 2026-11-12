from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/Users/mileshuang/Desktop/tripwire-project")
QUAL = ROOT / "qualification/cyp_external_task_qualification_20261109"
OUT = QUAL / "d012_implementation_qualification"

AUX_POP = QUAL / "d012_auxiliary_population/d012_task_identity_instances.tsv"
IDENTITY_BRIDGE = QUAL / "identity_audit/bounded_representation_audit.tsv"
MANIFEST_DIR = ROOT / "cyp/artifacts/cyp_007/split_manifests"

B_BATCH_SIZE = 16
AUX_EPOCHS = 50
WARMUP_EPOCHS = 10
LR_INITIAL = 1.4283739994469765e-05
LR_MAX = 0.0008245400010426866
LR_FINAL = 0.00020141638331344468


class FirewallClosedError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def stable_rng(*parts: object) -> random.Random:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def split_semicolon(value: str) -> list[str]:
    return [] if value == "" else value.split(";")


def manifest_paths() -> list[Path]:
    paths = sorted(MANIFEST_DIR.glob("rep_*_fold_*.json"))
    if len(paths) != 50:
        raise RuntimeError(f"expected 50 D007 manifests, got {len(paths)}")
    return paths


def manifest_id(path: Path) -> str:
    return path.stem


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_auxiliary_instances() -> list[dict[str, str]]:
    rows = read_tsv(AUX_POP)
    if len(rows) != 4664:
        raise RuntimeError(f"expected 4664 D012 task-identity instances, got {len(rows)}")
    task_units = {row["task_unit_id"] for row in rows}
    if len(task_units) != 175:
        raise RuntimeError(f"expected 175 task units, got {len(task_units)}")
    for row in rows:
        row["instance_id"] = f"{row['task_unit_id']}|{row['full_standard_inchikey']}"
    return rows


def build_openadmet_identity_bridge(rows: list[dict[str, str]] | None = None) -> dict[str, str]:
    bridge_rows = rows if rows is not None else read_tsv(IDENTITY_BRIDGE)
    by_name: dict[str, str] = {}
    for row in bridge_rows:
        if row["source_family"] != "openadmet":
            continue
        name = row["source_id"]
        if row["identity_state"] != "represented":
            raise FirewallClosedError(f"OpenADMET witness {name} is not represented")
        key = row["full_standard_inchikey"]
        if not key:
            raise FirewallClosedError(f"OpenADMET witness {name} has no full key")
        previous = by_name.get(name)
        if previous is not None and previous != key:
            raise FirewallClosedError(f"OpenADMET witness {name} maps ambiguously")
        by_name[name] = key
    if rows is None and len(by_name) != 5655:
        raise RuntimeError(f"expected 5655 OpenADMET identities, got {len(by_name)}")
    return by_name


def validation_keys_for_manifest(manifest: dict[str, object], bridge: dict[str, str]) -> tuple[set[str], list[str]]:
    names = manifest.get("validation_names")
    if not isinstance(names, list):
        raise FirewallClosedError("manifest lacks validation_names list")
    keys = []
    missing = []
    for name in names:
        if not isinstance(name, str):
            raise FirewallClosedError("manifest validation name is not a string")
        key = bridge.get(name)
        if key is None:
            missing.append(name)
        else:
            keys.append(key)
    if missing:
        raise FirewallClosedError(
            "validation Molecule_Name values missing qualified identity witnesses: "
            + ";".join(sorted(missing))
        )
    return set(keys), []


def apply_firewall(
    instances: list[dict[str, str]],
    validation_keys: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    removed = [row for row in instances if row["full_standard_inchikey"] in validation_keys]
    retained = [row for row in instances if row["full_standard_inchikey"] not in validation_keys]
    return retained, removed


def group_by_task(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_unit_id"]].append(row)
    return {key: sorted(value, key=lambda row: row["instance_id"]) for key, value in sorted(grouped.items())}


def construct_arm_b_epoch(
    retained: list[dict[str, str]],
    *,
    manifest_id_value: str,
    training_seed: int,
) -> list[dict[str, str]]:
    rows = []
    for task_unit_id, task_rows in group_by_task(retained).items():
        shuffled = list(task_rows)
        stable_rng("D012", "B", training_seed, manifest_id_value, task_unit_id).shuffle(shuffled)
        for batch_index, start in enumerate(range(0, len(shuffled), B_BATCH_SIZE), 1):
            batch = shuffled[start:start + B_BATCH_SIZE]
            rows.append({
                "manifest_id": manifest_id_value,
                "training_seed": str(training_seed),
                "task_unit_id": task_unit_id,
                "minibatch_index_within_task": str(batch_index),
                "batch_size": str(len(batch)),
                "instance_ids": ";".join(row["instance_id"] for row in batch),
                "full_standard_inchikeys": ";".join(row["full_standard_inchikey"] for row in batch),
                "objective_bearing_exposure_count": str(len(batch)),
                "loss_reduction": "mean_mse",
            })
    return rows


def q_for_task_size(n: int) -> int:
    return max(1, math.ceil(0.20 * n))


def construct_arm_c_epoch(
    retained: list[dict[str, str]],
    *,
    manifest_id_value: str,
    training_seed: int,
) -> list[dict[str, str]]:
    rows = []
    for task_unit_id, task_rows in group_by_task(retained).items():
        shuffled = list(task_rows)
        stable_rng("D012", "C", training_seed, manifest_id_value, task_unit_id).shuffle(shuffled)
        q_size = q_for_task_size(len(shuffled))
        all_keys = {row["full_standard_inchikey"] for row in shuffled}
        for episode_index, start in enumerate(range(0, len(shuffled), q_size), 1):
            query = shuffled[start:start + q_size]
            query_keys = {row["full_standard_inchikey"] for row in query}
            support_keys = sorted(all_keys - query_keys)
            if query_keys & set(support_keys):
                raise RuntimeError("support/query overlap detected")
            rows.append({
                "manifest_id": manifest_id_value,
                "training_seed": str(training_seed),
                "task_unit_id": task_unit_id,
                "episode_index_within_task": str(episode_index),
                "declared_q_size": str(q_size),
                "query_count": str(len(query)),
                "support_count": str(len(support_keys)),
                "query_instance_ids": ";".join(row["instance_id"] for row in query),
                "query_full_standard_inchikeys": ";".join(row["full_standard_inchikey"] for row in query),
                "support_full_standard_inchikeys_sha256": sha256_bytes(";".join(support_keys).encode("utf-8")),
                "support_query_disjoint": "true",
                "inner_adaptation_parameters": "scalar_head_weight_and_bias_only",
                "inner_lr": "0.01",
                "inner_steps": "1",
                "differentiation": "exact_through_adaptation",
                "support_loss": "mean_mse",
                "query_loss": "mean_mse",
                "outer_update_count": "1",
            })
    return rows


def scheduler_budget(u_b: int, u_c: int) -> dict[str, str]:
    return {
        "auxiliary_epochs": str(AUX_EPOCHS),
        "warmup_epochs": str(WARMUP_EPOCHS),
        "lr_initial": repr(LR_INITIAL),
        "lr_max": repr(LR_MAX),
        "lr_final": repr(LR_FINAL),
        "b_updates_per_epoch": str(u_b),
        "b_total_steps": str(AUX_EPOCHS * u_b),
        "b_warmup_steps": str(WARMUP_EPOCHS * u_b),
        "c_updates_per_epoch": str(u_c),
        "c_total_steps": str(AUX_EPOCHS * u_c),
        "c_warmup_steps": str(WARMUP_EPOCHS * u_c),
    }


def build_qualification() -> dict[str, object]:
    instances = load_auxiliary_instances()
    bridge = build_openadmet_identity_bridge()
    aux_keys = {row["full_standard_inchikey"] for row in instances}
    task_units = {row["task_unit_id"] for row in instances}
    manifest_rows = []
    cell_rows = []
    b_rows_all = []
    c_rows_all = []
    removed_keys_ever = set()
    all_retained_sizes = []

    for path in manifest_paths():
        mid = manifest_id(path)
        manifest = load_manifest(path)
        validation_keys, unmapped = validation_keys_for_manifest(manifest, bridge)
        retained, removed = apply_firewall(instances, validation_keys)
        retained_by_task = group_by_task(retained)
        if set(retained_by_task) != task_units:
            raise RuntimeError(f"not all tasks survived manifest {mid}")
        task_sizes = [len(rows) for rows in retained_by_task.values()]
        all_retained_sizes.extend(task_sizes)
        removed_keys = sorted({row["full_standard_inchikey"] for row in removed})
        removed_keys_ever.update(removed_keys)
        overlapping_validation_keys = sorted(aux_keys & validation_keys)
        u_b = sum(math.ceil(len(rows) / B_BATCH_SIZE) for rows in retained_by_task.values())
        u_c = sum(math.ceil(len(rows) / q_for_task_size(len(rows))) for rows in retained_by_task.values())
        training_seeds = manifest.get("training_seeds")
        if training_seeds != [20261101, 20261102, 20261103]:
            raise RuntimeError(f"unexpected training seeds in {mid}: {training_seeds!r}")
        manifest_row = {
            "manifest_id": mid,
            "manifest_path": str(path.relative_to(ROOT)),
            "repetition": str(manifest["repetition"]),
            "fold": str(manifest["fold"]),
            "validation_count": str(manifest["validation_count"]),
            "validation_unique_key_count": str(len(validation_keys)),
            "unmapped_validation_name_count": str(len(unmapped)),
            "aux_validation_overlap_key_count": str(len(overlapping_validation_keys)),
            "removed_instance_count": str(len(removed)),
            "retained_instance_count": str(len(retained)),
            "surviving_task_unit_count": str(len(retained_by_task)),
            "min_retained_task_size": str(min(task_sizes)),
            "max_retained_task_size": str(max(task_sizes)),
            "tasks_below_15": str(sum(size < 15 for size in task_sizes)),
            "tasks_below_16": str(sum(size < 16 for size in task_sizes)),
            "removed_full_standard_inchikeys": ";".join(removed_keys),
            "b_updates_per_epoch": str(u_b),
            "c_updates_per_epoch": str(u_c),
        }
        manifest_rows.append(manifest_row)
        for seed in training_seeds:
            b_rows = construct_arm_b_epoch(retained, manifest_id_value=mid, training_seed=seed)
            c_rows = construct_arm_c_epoch(retained, manifest_id_value=mid, training_seed=seed)
            b_rows_all.extend(b_rows)
            c_rows_all.extend(c_rows)
            cell_rows.append({
                "manifest_id": mid,
                "training_seed": str(seed),
                **scheduler_budget(u_b, u_c),
                "retained_instance_count": str(len(retained)),
                "arm_b_minibatch_rows": str(len(b_rows)),
                "arm_c_episode_rows": str(len(c_rows)),
            })

    removed_counts = [int(row["removed_instance_count"]) for row in manifest_rows]
    retained_counts = [int(row["retained_instance_count"]) for row in manifest_rows]
    b_dist = Counter(int(row["b_updates_per_epoch"]) for row in manifest_rows)
    c_dist = Counter(int(row["c_updates_per_epoch"]) for row in manifest_rows)
    summary = {
        "artifact_type": "D012_first_implementation_qualification_v1",
        "authority": "qualification_only_no_training_no_target_transfer",
        "runtime": {"python": sys.version},
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [AUX_POP, IDENTITY_BRIDGE, *manifest_paths()]
        },
        "global_input": {
            "task_units": len(task_units),
            "task_identity_instances": len(instances),
        },
        "firewall": {
            "manifest_count": len(manifest_rows),
            "cell_count": len(cell_rows),
            "unmapped_validation_names": sum(int(row["unmapped_validation_name_count"]) for row in manifest_rows),
            "manifests_with_at_least_one_overlap": sum(int(row["removed_instance_count"]) > 0 for row in manifest_rows),
            "unique_auxiliary_keys_ever_in_validation": len(removed_keys_ever),
            "removed_instances_min": min(removed_counts),
            "removed_instances_max": max(removed_counts),
            "removed_instances_mean": sum(removed_counts) / len(removed_counts),
            "retained_instances_min": min(retained_counts),
            "retained_instances_max": max(retained_counts),
            "all_tasks_survive_every_manifest": all(int(row["surviving_task_unit_count"]) == 175 for row in manifest_rows),
            "global_minimum_post_mask_task_size": min(all_retained_sizes),
            "max_tasks_below_15_per_manifest": max(int(row["tasks_below_15"]) for row in manifest_rows),
            "max_tasks_below_16_per_manifest": max(int(row["tasks_below_16"]) for row in manifest_rows),
            "validation_labels_read": False,
            "fail_closed_on_missing_or_ambiguous_witness": True,
        },
        "arm_b": {
            "batch_size": B_BATCH_SIZE,
            "loss": "mean_mse_per_task_local_minibatch",
            "updates_per_epoch_distribution": dict(sorted((str(k), v) for k, v in b_dist.items())),
            "objective_bearing_exposure": "each_retained_identity_exactly_once_per_epoch",
            "pooled_global_molecule_sampler": False,
        },
        "arm_c": {
            "q_rule": "max(1, ceil(0.20*N))",
            "support_query_disjoint_by_full_standard_inchikey": True,
            "support_policy": "all_retained_within_task_identities_not_in_query",
            "support_reuse_permitted_and_recorded": True,
            "inner_adaptation": {
                "parameters": "scalar_head_weight_and_bias_only",
                "inner_lr": 0.01,
                "inner_steps": 1,
                "differentiation": "exact_through_adaptation",
            },
            "updates_per_epoch_distribution": dict(sorted((str(k), v) for k, v in c_dist.items())),
            "query_exposure": "each_retained_identity_exactly_once_per_epoch",
        },
        "scheduler": {
            "auxiliary_epochs": AUX_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "lr_initial": LR_INITIAL,
            "lr_max": LR_MAX,
            "lr_final": LR_FINAL,
            "budgets_derived_from_each_arm_own_u_value": True,
            "assume_equal_b_c_step_counts": False,
        },
        "boundaries": {
            "production_training_implemented": False,
            "target_transfer_implemented": False,
            "validation_potency_labels_accessed": False,
            "public_challenge_labels_accessed": False,
            "partition_changes": False,
            "task_size_filtering": False,
            "hyperparameter_tuning": False,
            "d011_modified": False,
        },
    }
    return {
        "manifest_rows": sorted(manifest_rows, key=lambda row: row["manifest_id"]),
        "cell_rows": sorted(cell_rows, key=lambda row: (row["manifest_id"], row["training_seed"])),
        "b_rows": sorted(b_rows_all, key=lambda row: (row["manifest_id"], row["training_seed"], row["task_unit_id"], int(row["minibatch_index_within_task"]))),
        "c_rows": sorted(c_rows_all, key=lambda row: (row["manifest_id"], row["training_seed"], row["task_unit_id"], int(row["episode_index_within_task"]))),
        "summary": summary,
    }


MANIFEST_FIELDS = [
    "manifest_id",
    "manifest_path",
    "repetition",
    "fold",
    "validation_count",
    "validation_unique_key_count",
    "unmapped_validation_name_count",
    "aux_validation_overlap_key_count",
    "removed_instance_count",
    "retained_instance_count",
    "surviving_task_unit_count",
    "min_retained_task_size",
    "max_retained_task_size",
    "tasks_below_15",
    "tasks_below_16",
    "removed_full_standard_inchikeys",
    "b_updates_per_epoch",
    "c_updates_per_epoch",
]

CELL_FIELDS = [
    "manifest_id",
    "training_seed",
    "auxiliary_epochs",
    "warmup_epochs",
    "lr_initial",
    "lr_max",
    "lr_final",
    "b_updates_per_epoch",
    "b_total_steps",
    "b_warmup_steps",
    "c_updates_per_epoch",
    "c_total_steps",
    "c_warmup_steps",
    "retained_instance_count",
    "arm_b_minibatch_rows",
    "arm_c_episode_rows",
]

B_FIELDS = [
    "manifest_id",
    "training_seed",
    "task_unit_id",
    "minibatch_index_within_task",
    "batch_size",
    "instance_ids",
    "full_standard_inchikeys",
    "objective_bearing_exposure_count",
    "loss_reduction",
]

C_FIELDS = [
    "manifest_id",
    "training_seed",
    "task_unit_id",
    "episode_index_within_task",
    "declared_q_size",
    "query_count",
    "support_count",
    "query_instance_ids",
    "query_full_standard_inchikeys",
    "support_full_standard_inchikeys_sha256",
    "support_query_disjoint",
    "inner_adaptation_parameters",
    "inner_lr",
    "inner_steps",
    "differentiation",
    "support_loss",
    "query_loss",
    "outer_update_count",
]


def write_outputs(outputs: dict[str, object]) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "d012_manifest_firewall.tsv": OUT / "d012_manifest_firewall.tsv",
        "d012_cell_scheduler_budgets.tsv": OUT / "d012_cell_scheduler_budgets.tsv",
        "d012_arm_b_epoch_plan.tsv": OUT / "d012_arm_b_epoch_plan.tsv",
        "d012_arm_c_epoch_plan.tsv": OUT / "d012_arm_c_epoch_plan.tsv",
        "d012_implementation_qualification_summary.json": OUT / "d012_implementation_qualification_summary.json",
    }
    write_tsv(paths["d012_manifest_firewall.tsv"], MANIFEST_FIELDS, outputs["manifest_rows"])
    write_tsv(paths["d012_cell_scheduler_budgets.tsv"], CELL_FIELDS, outputs["cell_rows"])
    write_tsv(paths["d012_arm_b_epoch_plan.tsv"], B_FIELDS, outputs["b_rows"])
    write_tsv(paths["d012_arm_c_epoch_plan.tsv"], C_FIELDS, outputs["c_rows"])
    paths["d012_implementation_qualification_summary.json"].write_text(
        json.dumps(outputs["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hash_path = OUT / "SHA256SUMS.txt"
    with hash_path.open("w", encoding="utf-8") as handle:
        for name in sorted(paths):
            handle.write(f"{sha256_file(paths[name])}  {name}\n")
    return {name: sha256_file(path) for name, path in {**paths, "SHA256SUMS.txt": hash_path}.items()}


def main() -> None:
    outputs = build_qualification()
    hashes = write_outputs(outputs)
    summary = outputs["summary"]
    print("=== D012 IMPLEMENTATION QUALIFICATION ===")
    print("manifests/cells:", summary["firewall"]["manifest_count"], summary["firewall"]["cell_count"])
    print("firewall removed min/max:", summary["firewall"]["removed_instances_min"], summary["firewall"]["removed_instances_max"])
    print("retained min/max:", summary["firewall"]["retained_instances_min"], summary["firewall"]["retained_instances_max"])
    print("Arm B updates distribution:", summary["arm_b"]["updates_per_epoch_distribution"])
    print("Arm C updates distribution:", summary["arm_c"]["updates_per_epoch_distribution"])
    print("global minimum post-mask task size:", summary["firewall"]["global_minimum_post_mask_task_size"])
    print("output hashes:", hashes)
    print("QUALIFICATION COMPLETE -- NO PRODUCTION TRAINING PERFORMED")


if __name__ == "__main__":
    main()
