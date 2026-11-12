#!/bin/bash
#SBATCH --job-name=p1-d009
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-149
#SBATCH --output=/scratch/user/mhuang8/p1_d009_logs/%A_%a.out
#SBATCH --error=/scratch/user/mhuang8/p1_d009_logs/%A_%a.err

set -euo pipefail

P1=/scratch/user/mhuang8/p1_corehpc_stage_20261107
BASE=/scratch/user/mhuang8/tripwire-d006/d006_bundle
PY=/scratch/user/mhuang8/envs/tripwire_cyp002/bin/python

PROD=/scratch/user/mhuang8/p1_d009_production_20261108
DATA="$BASE/cyp/data/cyp-challenge-TRAIN_inhibition.csv"
SPLITS="$BASE/cyp/artifacts/cyp_007/split_manifests"
T121="$BASE/cyp/artifacts/cyp_007/production_run/attempt_001/trials/trial_121/proposal.json"

SEEDS=(20261101 20261102 20261103)

TASK=${SLURM_ARRAY_TASK_ID}

SEED_INDEX=$(( TASK % 3 ))
FOLD=$(( (TASK / 3) % 10 + 1 ))
REPETITION=$(( TASK / 30 + 1 ))
TRAINING_SEED=${SEEDS[$SEED_INDEX]}

CELL_DIR=$(printf \
  '%s/rep_%02d/fold_%02d/seed_%d' \
  "$PROD" "$REPETITION" "$FOLD" "$TRAINING_SEED")

RUN_JSON="$CELL_DIR/run_cell.json"

echo "=== D009 P1 CELL ==="
date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "repetition=$REPETITION"
echo "fold=$FOLD"
echo "training_seed=$TRAINING_SEED"
echo "cell_dir=$CELL_DIR"

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

cd "$P1"

PYTHONPATH="$P1/cyp/scripts:$BASE/cyp/scripts" \
"$PY" cyp/scripts/p1_pairwise_production.py run-cell \
  --repetition "$REPETITION" \
  --fold "$FOLD" \
  --training-seed "$TRAINING_SEED" \
  --production-root "$PROD" \
  --repository-root "$BASE" \
  --dataset-path "$DATA" \
  --split-manifest-dir "$SPLITS" \
  --t121-config-artifact "$T121" \
  --output-json "$RUN_JSON"

echo
echo "=== CELL COMPLETE ==="
date --iso-8601=seconds
sha256sum \
  "$CELL_DIR/manifest.json" \
  "$CELL_DIR/best.ckpt" \
  "$CELL_DIR/metrics.json" \
  "$CELL_DIR/SHA256SUMS.txt" \
  "$RUN_JSON"
