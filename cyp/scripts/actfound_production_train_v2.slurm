#!/bin/bash
#SBATCH --job-name=actfound-d013
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --output=actfound_d013_%j.out
#SBATCH --error=actfound_d013_%j.err

# D013 foundation-training wrapper (v2). Everything it needs is passed explicitly;
# nothing falls back to a default path. A failed run is resubmitted from scratch
# into a new output directory (runner v2 refuses an existing one).
set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT must be set explicitly (no default)}"
: "${ACTFOUND_ENV:?ACTFOUND_ENV must be set explicitly (no default)}"
: "${ACTFOUND_CONFIG:?ACTFOUND_CONFIG must point to the reviewed ActFoundProductionConfigV2 JSON}"

if [[ ! -f "${ACTFOUND_CONFIG}" ]]; then
  echo "ACTFOUND_CONFIG does not exist: ${ACTFOUND_CONFIG}" >&2
  exit 2
fi
if [[ ! -x "${ACTFOUND_ENV}/bin/python" ]]; then
  echo "Pinned ActFound Python is missing: ${ACTFOUND_ENV}/bin/python" >&2
  exit 2
fi

# Keep every runtime cache and temp file off the over-quota home directory.
SCRATCH_RUNTIME="/scratch/user/mhuang8/actfound/runtime/${SLURM_JOB_ID:-interactive}"
mkdir -p "${SCRATCH_RUNTIME}"/{tmp,xdg_cache,torch,cuda_cache}
export TMPDIR="${SCRATCH_RUNTIME}/tmp"
export XDG_CACHE_HOME="${SCRATCH_RUNTIME}/xdg_cache"
export TORCH_HOME="${SCRATCH_RUNTIME}/torch"
export CUDA_CACHE_PATH="${SCRATCH_RUNTIME}/cuda_cache"

RUNNER="${PROJECT_ROOT}/cyp/scripts/actfound_production_train_v2.py"
echo "=== D013 wrapper start $(date -Is) on $(hostname) job=${SLURM_JOB_ID:-none}"
echo "wrapper_sha256=$(sha256sum "$0" | cut -c1-64) wrapper_path=$0"
echo "runner_sha256=$(sha256sum "${RUNNER}" | cut -c1-64)"
echo "model_side_sha256=$(sha256sum "${PROJECT_ROOT}/cyp/scripts/actfound_model_side.py" | cut -c1-64)"
echo "config_sha256=$(sha256sum "${ACTFOUND_CONFIG}" | cut -c1-64) config=${ACTFOUND_CONFIG}"
echo "PROJECT_ROOT=${PROJECT_ROOT} ACTFOUND_ENV=${ACTFOUND_ENV}"
echo "TMPDIR=${TMPDIR} XDG_CACHE_HOME=${XDG_CACHE_HOME} TORCH_HOME=${TORCH_HOME} CUDA_CACHE_PATH=${CUDA_CACHE_PATH}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

cd "${PROJECT_ROOT}"

"${ACTFOUND_ENV}/bin/python" \
  "${RUNNER}" \
  --config "${ACTFOUND_CONFIG}" \
  --repo-root "${PROJECT_ROOT}" \
  --upstream-dir "${PROJECT_ROOT}/external/ActFound"

echo "=== D013 wrapper end $(date -Is) exit=0"
