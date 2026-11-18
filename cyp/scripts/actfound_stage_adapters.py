
"""Concrete adapters for the existing ActFound staged implementations.

No scientific defaults, training launch CLI, or implicit resume is introduced.
The caller supplies frozen configurations, input paths, and interpreter paths.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result = digest.hexdigest()
    return result


def read_json(path):
    result = json.loads(Path(path).read_text())
    return result


def write_new_json(path, value):
    # Serialize first: a value that cannot be encoded must not leave a partial
    # file behind (it would also block the exclusive-create retry).
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(text)


def verify_source(initialization):
    record = initialization["source_checkpoint"]
    if not record or sha256(record["path"]) != record["sha256"]:
        raise ValueError("Stage initialization checkpoint missing or changed.")
    result = Path(record["path"])
    return result


def require_mode(initialization, mode):
    if initialization.get("mode") != mode:
        raise ValueError(f"Expected initialization mode {mode}.")
    for key in (
        "import_optimizer_state", "import_scheduler_state", "import_epoch_state"
    ):
        if initialization.get(key) is not False:
            raise ValueError(f"Stage transition must not import {key}.")


def foundation_adapter(
    *, runner, frozen_config_path, repo_root, upstream_dir
):
    """Reuse original initialization, corpus checks, schedule and selection."""
    def execute(*, output_dir, initialization):
        require_mode(initialization, "original_runner_defined_initialization")
        if initialization.get("source_checkpoint") is not None:
            raise ValueError("Foundation must not import a historical checkpoint.")

        output_dir = Path(output_dir)
        config = dict(runner.validate_and_resolve_config(
            Path(frozen_config_path)
        ))
        # The controller owns output_dir; the runner requires a new child.
        config["output_dir"] = str(output_dir / "training")
        config_path = output_dir / "effective_foundation_config.json"
        config["config_path"] = str(config_path)
        write_new_json(config_path, config)

        runner.run_training(
            config, Path(repo_root), Path(upstream_dir), skip_hash=False
        )
        completion = read_json(
            Path(config["output_dir"]) / "completion.json"
        )
        if completion.get("status") != "completed":
            raise ValueError("Foundation training did not complete.")
        checkpoint = completion.get("best_checkpoint")
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError("Foundation did not produce its selected checkpoint.")
        result = {"selected_checkpoint": checkpoint}
        return result
    return execute


def effective_screening_config(args, initialization, transfer, metadata):
    """The JSON-serializable record of the effective screening configuration.

    args.rng is a live numpy RandomState (made by the runner's make_args); its
    seed is already recorded as torch_seed_source_seed, so it is not recorded.
    """
    return {
        "hyperparameters": {
            key: value for key, value in vars(args).items() if key != "rng"
        },
        "source_checkpoint": initialization["source_checkpoint"],
        "transfer": transfer,
        "screen_metadata": metadata,
    }


def screening_adapter(
    *, runner, frozen_config_path, upstream_dir, screen_inputs
):
    """Strict corrected-foundation initialization -> approved screening."""
    def execute(*, output_dir, initialization):
        require_mode(initialization, "stage_transfer_not_resume")
        source = verify_source(initialization)
        config = runner.validate_and_resolve_config(Path(frozen_config_path))
        prepared = runner.prepare_actfound_partitioned_screen_stage(
            **screen_inputs,
            hyperparameters=dict(config["hyperparameters"]),
        )
        args = prepared["args"]
        if args.metatrain_iterations != 80:
            raise ValueError("Screening preparation did not select 80 epochs.")

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Existing upstream validation requires CUDA.")
        upstream = str(Path(upstream_dir).resolve())
        if upstream not in sys.path:
            sys.path.insert(0, upstream)
        from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR

        runner.seed_like_upstream(args)
        regressor = FCNReLUNormNetworkQSAR(
            input_shape=(2, args.dim_w), args=args, meta=True
        ).cuda()
        rates = torch.nn.Parameter(
            torch.ones(args.num_updates + 1, device="cuda") * args.update_lr,
            requires_grad=bool(
                args.learnable_per_layer_per_step_inner_loop_learning_rate
            ),
        )
        # Existing PyTorch is older: do not pass weights_only.
        checkpoint = torch.load(source, map_location="cpu")
        if checkpoint.get("upstream_commit") != runner.PINNED_UPSTREAM_COMMIT:
            raise ValueError("Foundation checkpoint upstream revision mismatch.")
        transfer = runner.transfer_actfound_stage_initialization(
            checkpoint, regressor, rates
        )

        output_dir = Path(output_dir)
        effective = effective_screening_config(
            args, initialization, transfer, prepared["metadata"]
        )
        config_path = output_dir / "effective_screening_config.json"
        write_new_json(config_path, effective)
        write_new_json(output_dir / "transfer.json", transfer)

        selected = runner.fit_actfound_screen_stage(
            regressor=regressor,
            inner_lrs=rates,
            prepared=prepared,
            output_dir=output_dir / "training",
            config={"config_path": str(config_path)},
            authorize_training=True,
        )
        selected_path = Path(selected["checkpoint"])
        if sha256(selected_path) != selected["sha256"]:
            raise ValueError("Selected screening checkpoint hash mismatch.")
        result = {"selected_checkpoint": str(selected_path)}
        return result
    return execute


def run_logged(command, log_path):
    """Execute a stage subprocess; preserve its output and fail visibly."""
    with Path(log_path).open("x") as handle:
        subprocess.run(
            [str(value) for value in command],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def pic50_adapter(
    *, actfound_python, scoring_python, evaluation_dir,
    evaluation_config, qualification_config, counts,
    frozen_train_config, manifest_dir, freeze_record,
    label_table, repo_root, upstream_dir
):
    """Reuse full D013 adaptation/scoring; seed slots remain replicas.

    The existing evaluator owns adaptation and target handling.
    No optimizer, scheduler or epoch state is resumed by this adapter.
    """
    def execute(*, output_dir, initialization):
        require_mode(initialization, "stage_transfer_not_resume")
        checkpoint = verify_source(initialization)
        output_dir = Path(output_dir)
        evaluation_dir_path = Path(evaluation_dir)
        predictions_dir = output_dir / "predictions"
        scores_dir = output_dir / "scores"

        prediction_command = [
            actfound_python, "-I",
            evaluation_dir_path / "run_evaluation.py",
            "--config", evaluation_config,
            "--config-sha256", sha256(evaluation_config),
            "--qualification-config", qualification_config,
            "--counts", counts,
            "--counts-sha256", sha256(counts),
            "--checkpoint", checkpoint,
            "--checkpoint-sha256", sha256(checkpoint),
            "--train-config", frozen_train_config,
            "--train-config-sha256", sha256(frozen_train_config),
            "--manifest-dir", manifest_dir,
            "--freeze-record", freeze_record,
            "--label-table", label_table,
            "--label-table-sha256", sha256(label_table),
            "--repo-root", repo_root,
            "--upstream-dir", upstream_dir,
            "--out-dir", predictions_dir,
        ]
        write_new_json(output_dir / "prediction_command.json", [
            str(value) for value in prediction_command
        ])
        run_logged(prediction_command, output_dir / "predictions.log")
        predictions = predictions_dir / "cyp_adaptation_predictions.json"
        prediction_record = read_json(predictions)
        if (
            prediction_record.get("unique_adaptations") != 200
            or prediction_record.get("seed_slots_expanded_at_scoring") != 3
        ):
            raise ValueError("Unexpected ActFound adaptation/replica topology.")

        scoring_command = [
            scoring_python, "-I",
            evaluation_dir_path / "score_evaluation.py",
            "--config", evaluation_config,
            "--config-sha256", sha256(evaluation_config),
            "--predictions", predictions,
            "--predictions-sha256", sha256(predictions),
            "--repo-root", repo_root,
            "--out-dir", scores_dir,
        ]
        write_new_json(output_dir / "scoring_command.json", [
            str(value) for value in scoring_command
        ])
        run_logged(scoring_command, output_dir / "scoring.log")
        score_path = scores_dir / "cyp_o150_result.json"
        score = read_json(score_path)
        if score.get("cell_count") != 150:
            raise ValueError("Scorer did not produce all 150 evaluation slots.")

        artifacts = [
            "predictions/cyp_adaptation_predictions.json",
            "scores/cyp_o150_result.json",
            "prediction_command.json", "scoring_command.json",
            "predictions.log", "scoring.log",
        ]
        manifest = {
            "status": "completed",
            "artifacts": artifacts,
            "artifact_sha256": {
                name: sha256(output_dir / name) for name in artifacts
            },
            "source_checkpoint": initialization["source_checkpoint"],
            "unique_endpoint_adaptations": 200,
            "evaluation_slots": 150,
            "seed_slots_are_replicates": score["seed_slots_are_replicates"],
            "comparison_is_final": score["comparison_is_final"],
            "o150": score["o150"],
        }
        manifest_path = output_dir / "evaluation_manifest.json"
        write_new_json(manifest_path, manifest)
        result = {"evaluation_manifest": str(manifest_path)}
        return result
    return execute
