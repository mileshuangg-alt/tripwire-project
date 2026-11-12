#!/bin/bash
#SBATCH --job-name=d012_first_cell_b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_first_cell_b_%j.out
#SBATCH --error=/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_first_cell_b_%j.err

set -euo pipefail

DEPLOY_ROOT=/scratch/user/mhuang8/d012_runtime_qualification_20261110
PROJECT_ROOT="${DEPLOY_ROOT}/tripwire-project"
ENV_ROOT=/scratch/user/mhuang8/envs/tripwire_cyp002
ARTIFACT_ROOT="${DEPLOY_ROOT}/artifacts/d012_first_cell_arm_b_rep01_fold01_seed20261101"

mkdir -p "${ARTIFACT_ROOT}"
cd "${PROJECT_ROOT}"

export TRIPWIRE_PROJECT_ROOT="${PROJECT_ROOT}"

"${ENV_ROOT}/bin/python" \
  cyp/scripts/d012_first_cell_runner.py \
  --arm B \
  --manifest-id rep_01_fold_01 \
  --training-seed 20261101 \
  --device cuda \
  --output-dir "${ARTIFACT_ROOT}"
