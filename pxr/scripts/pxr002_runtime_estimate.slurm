#!/bin/bash
#SBATCH --job-name=pxr002-runtime-estimate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=pxr002_runtime_estimate_%j.out
#SBATCH --error=pxr002_runtime_estimate_%j.err

# PXR-002 runtime estimate (D018). Measurement only: no training run, no checkpoint,
# no holdout or test data. Miles chose to run this as a batch job because the
# interactive srun allocation failed (2026-11-17); this is a deviation from the
# interactive-benchmark practice and is recorded as such.
# The job verifies every staged file against the reviewed hash list before running.
set -euo pipefail

: "${STAGE_ROOT:?STAGE_ROOT must be set explicitly (no default)}"
: "${PXR_ENV:?PXR_ENV must be set explicitly (no default)}"
: "${HASH_LIST:?HASH_LIST must point to the reviewed pxr002_stage_hashes file}"

SCRATCH_RUNTIME="/scratch/user/mhuang8/pxr002_runtime/${SLURM_JOB_ID:-interactive}"
mkdir -p "${SCRATCH_RUNTIME}"/{tmp,xdg_cache,torch,cuda_cache}
export TMPDIR="${SCRATCH_RUNTIME}/tmp" XDG_CACHE_HOME="${SCRATCH_RUNTIME}/xdg_cache"
export TORCH_HOME="${SCRATCH_RUNTIME}/torch" CUDA_CACHE_PATH="${SCRATCH_RUNTIME}/cuda_cache"
export PYTHONNOUSERSITE=1

echo "=== PXR-002 runtime estimate start $(date -Is) on $(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "${STAGE_ROOT}"
sha256sum -c "${HASH_LIST}"
"${PXR_ENV}/bin/python" -c "import chemprop; print('chemprop', chemprop.__version__)"
"${PXR_ENV}/bin/python" -I pxr/scripts/pxr002_runtime_estimate.py \
  --artifact-dir pxr/artifacts/pxr002_aux_v2 --accelerator gpu
echo "=== PXR-002 runtime estimate end $(date -Is) exit=0"
