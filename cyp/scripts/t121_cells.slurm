#!/bin/bash
#SBATCH --job-name=t121-d015-cells
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-149
#SBATCH --no-requeue
#SBATCH --output=t121_d015_cell_%A_%a.out
#SBATCH --error=t121_d015_cell_%A_%a.err
# Resources are the recorded D007 cell_array topology (RESOURCE_TOPOLOGY in
# cyp007_production_execution.py). One cell per array task; a cell that fails
# leaves INCOMPLETE.json and needs human adjudication, never a blind retry.

set -euo pipefail

: "${PACKAGE_ROOT:?}" "${T121_ENV:?}" "${CELL_RUN_ROOT:?}" "${SCREEN_SELECTION:?}"
: "${SCREEN_SELECTION_SHA256:?}" "${EXECUTOR_SOURCE_SHA256:?}"
: "${DATASET:?}" "${ORIGINAL_SPLIT:?}" "${SPLIT_MANIFEST_ROOT:?}" "${SCREEN_PROPOSAL:?}"
: "${SCREEN_CSV:?}" "${SCREEN_IDENTITY:?}" "${SCREEN_SIMILARITY:?}" "${SCREEN_CONTRACT:?}"

SCRATCH_RUNTIME="/scratch/user/mhuang8/actfound/runtime/t121cell_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SCRATCH_RUNTIME}"/{tmp,xdg_cache,torch,cuda_cache}
export TMPDIR="${SCRATCH_RUNTIME}/tmp" XDG_CACHE_HOME="${SCRATCH_RUNTIME}/xdg_cache"
export TORCH_HOME="${SCRATCH_RUNTIME}/torch" CUDA_CACHE_PATH="${SCRATCH_RUNTIME}/cuda_cache"
export PYTHONNOUSERSITE=1

cd "${PACKAGE_ROOT}"
"${T121_ENV}/bin/python" -I "${PACKAGE_ROOT}/cyp/scripts/t121_production_cell.py" \
  --run-root "${CELL_RUN_ROOT}" --trial-index 121 --cell-index "${SLURM_ARRAY_TASK_ID}" \
  --dataset "${DATASET}" --original-split "${ORIGINAL_SPLIT}" \
  --split-manifest-root "${SPLIT_MANIFEST_ROOT}" \
  --screen-selection "${SCREEN_SELECTION}" --screen-selection-sha256 "${SCREEN_SELECTION_SHA256}" \
  --screen-proposal "${SCREEN_PROPOSAL}" \
  --screen "${SCREEN_CSV}" --identity "${SCREEN_IDENTITY}" \
  --similarity "${SCREEN_SIMILARITY}" --contract "${SCREEN_CONTRACT}" \
  --executor-source-dir "${PACKAGE_ROOT}/executor_source" \
  --executor-source-sha256 "${EXECUTOR_SOURCE_SHA256}" --authorize-training
