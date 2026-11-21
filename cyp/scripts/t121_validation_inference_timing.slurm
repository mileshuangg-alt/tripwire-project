#!/bin/bash
#SBATCH --job-name=t121-valinf-timing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=t121_valinf_timing_%j.out
#SBATCH --error=t121_valinf_timing_%j.err
# One-cell validation-only inference timing and reproduction check (Tripwire-CYP-007).
# Run as a batch job because the interactive srun allocation failed with
# "Insane message length"; a recorded deviation from the interactive-benchmark
# practice, chosen by Miles. Resources equal the updated-T121 cell topology in
# t121_cells.slurm. No training; reads the cell's checkpoint, writes only to a
# NEW OUTPUT_DIR. Nothing here has a default path.

set -euo pipefail

: "${PACKAGE_ROOT:?}" "${T121_ENV:?}" "${CELL_RUN_ROOT:?}" "${DATASET:?}"
: "${ORIGINAL_SPLIT:?}" "${SPLIT_MANIFEST_ROOT:?}" "${EXECUTOR_SOURCE_SHA256:?}"
: "${INFERENCE_SCRIPT:?}" "${INFERENCE_SCRIPT_SHA256:?}" "${OUTPUT_DIR:?}"
: "${CELL_INDEX:?}" "${TOLERANCE:?}"

SCRATCH_RUNTIME="/scratch/user/mhuang8/actfound/runtime/t121valinf_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_RUNTIME}"/{tmp,xdg_cache,torch,cuda_cache}
export TMPDIR="${SCRATCH_RUNTIME}/tmp" XDG_CACHE_HOME="${SCRATCH_RUNTIME}/xdg_cache"
export TORCH_HOME="${SCRATCH_RUNTIME}/torch" CUDA_CACHE_PATH="${SCRATCH_RUNTIME}/cuda_cache"
export PYTHONNOUSERSITE=1

echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

actual="$(sha256sum "${INFERENCE_SCRIPT}" | cut -d' ' -f1)"
if [ "${actual}" != "${INFERENCE_SCRIPT_SHA256}" ]; then
  echo "inference script hash mismatch: ${actual}" >&2
  exit 2
fi

cd "${PACKAGE_ROOT}"
STARTED=$(date +%s)
"${T121_ENV}/bin/python" -I "${INFERENCE_SCRIPT}" \
  --package-root "${PACKAGE_ROOT}" --run-root "${CELL_RUN_ROOT}" --trial-index 121 \
  --cells "${CELL_INDEX}" --dataset "${DATASET}" --original-split "${ORIGINAL_SPLIT}" \
  --split-manifest-root "${SPLIT_MANIFEST_ROOT}" --output-dir "${OUTPUT_DIR}" \
  --executor-source-dir "${PACKAGE_ROOT}/executor_source" \
  --executor-source-sha256 "${EXECUTOR_SOURCE_SHA256}" \
  --tolerance "${TOLERANCE}" --accelerator gpu --authorize-inference
echo "process_wall_seconds=$(( $(date +%s) - STARTED )) end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
