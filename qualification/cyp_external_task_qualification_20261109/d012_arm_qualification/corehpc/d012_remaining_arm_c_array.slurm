#!/bin/bash
#SBATCH --job-name=d012_remaining_c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=1-149
#SBATCH --output=/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_remaining_c_%A_%a.out
#SBATCH --error=/scratch/user/mhuang8/d012_runtime_qualification_20261110/artifacts/d012_remaining_c_%A_%a.err

set -euo pipefail

DEPLOY_ROOT=/scratch/user/mhuang8/d012_runtime_qualification_20261110
PROJECT_ROOT="${DEPLOY_ROOT}/tripwire-project"
ENV_ROOT=/scratch/user/mhuang8/envs/tripwire_cyp002
CELL_MANIFEST="${PROJECT_ROOT}/qualification/cyp_external_task_qualification_20261109/d012_arm_qualification/d012_remaining_149_cells.tsv"

cd "${PROJECT_ROOT}"
export TRIPWIRE_PROJECT_ROOT="${PROJECT_ROOT}"

ROW=$(awk -F '\t' -v idx="${SLURM_ARRAY_TASK_ID}" 'NR > 1 && $1 == idx {print $0}' "${CELL_MANIFEST}")
if [[ -z "${ROW}" ]]; then
  echo "No D012 remaining-cell row for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

IFS=$'\t' read -r ARRAY_INDEX REPETITION FOLD MANIFEST_ID TRAINING_SEED MANIFEST_PATH MANIFEST_SHA256 <<< "${ROW}"
ARTIFACT_ROOT="${DEPLOY_ROOT}/artifacts/d012_remaining/arm_c/${MANIFEST_ID}_seed${TRAINING_SEED}"
mkdir -p "${ARTIFACT_ROOT}"

if [[ -f "${ARTIFACT_ROOT}/completion.json" && -f "${ARTIFACT_ROOT}/SHA256SUMS.txt" ]]; then
  echo "D012 Arm C cell ${MANIFEST_ID} seed ${TRAINING_SEED} already has completion marker and hashes; skipping."
  exit 0
fi

"${ENV_ROOT}/bin/python" \
  cyp/scripts/d012_first_cell_runner.py \
  --arm C \
  --manifest-id "${MANIFEST_ID}" \
  --training-seed "${TRAINING_SEED}" \
  --device cuda \
  --output-dir "${ARTIFACT_ROOT}"
