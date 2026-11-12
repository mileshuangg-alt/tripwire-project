#!/bin/bash
#SBATCH --job-name=d012_bounded_runtime
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_bounded_runtime_%j.out
#SBATCH --error=/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_bounded_runtime_%j.err

set -euo pipefail

DEPLOY_ROOT=/scratch/user/mhuang8/d012_runtime_qualification_20261110
PROJECT_ROOT="${DEPLOY_ROOT}/tripwire-project"
ENV_ROOT=/scratch/user/mhuang8/envs/tripwire_cyp002
ARTIFACT_ROOT="${DEPLOY_ROOT}/artifacts"

mkdir -p "${ARTIFACT_ROOT}"
cd "${PROJECT_ROOT}"

export TRIPWIRE_PROJECT_ROOT="${PROJECT_ROOT}"

"${ENV_ROOT}/bin/python" \
  cyp/scripts/d012_complete_model_runtime_benchmark.py \
  --manifest-id rep_01_fold_01 \
  --training-seed 20261101 \
  --device cuda \
  --output "${ARTIFACT_ROOT}/d012_chemprop_dmpnn_complete_model_runtime_benchmark_rep01_fold01_seed20261101.json"
