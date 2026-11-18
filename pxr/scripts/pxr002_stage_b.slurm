#!/bin/bash
#SBATCH --job-name=pxr002-stage-b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=pxr002_stage_b_%j.out
#SBATCH --error=pxr002_stage_b_%j.err

# PXR-002 stage B (D018): the three best stage-A configurations by primary-head validation
# MAE, compared across three seeds (the stage A common seed counts as the first), so six
# extra runs. Wall time rationale: stage A job 3113907 measured 114-143 s per run (mean
# about 126 s, epochs 24-27), so six runs take about 13 minutes; 1 h leaves a wide margin
# for startup and node variance. Nothing resumes: a failed run stops the ranking and the
# job exits nonzero. No auxiliary-off ablation, ensemble or holdout/test evaluation here.
set -euo pipefail

: "${STAGE_ROOT:?STAGE_ROOT must be set explicitly (no default)}"
: "${PXR_ENV:?PXR_ENV must be set explicitly (no default)}"
: "${HASH_LIST:?HASH_LIST must point to the reviewed hash list}"
: "${STAGE_A_ROOT:?STAGE_A_ROOT must be the completed stage A output root}"
: "${OUT_ROOT:?OUT_ROOT must be a new output directory (no default)}"
: "${AUTHORIZE_STAGE_B:?set AUTHORIZE_STAGE_B=yes only with explicit launch approval from Miles}"
[ "${AUTHORIZE_STAGE_B}" = "yes" ] || { echo "AUTHORIZE_STAGE_B must be exactly 'yes'"; exit 2; }
[ ! -e "${OUT_ROOT}" ] || { echo "OUT_ROOT already exists: ${OUT_ROOT}"; exit 2; }

SCRATCH_RUNTIME="/scratch/user/mhuang8/pxr002_runtime/${SLURM_JOB_ID:-interactive}"
mkdir -p "${SCRATCH_RUNTIME}"/{tmp,xdg_cache,torch,cuda_cache}
export TMPDIR="${SCRATCH_RUNTIME}/tmp" XDG_CACHE_HOME="${SCRATCH_RUNTIME}/xdg_cache"
export TORCH_HOME="${SCRATCH_RUNTIME}/torch" CUDA_CACHE_PATH="${SCRATCH_RUNTIME}/cuda_cache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

echo "=== PXR-002 stage B start $(date -Is) on $(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "${STAGE_ROOT}"
sha256sum -c "${HASH_LIST}"
"${PXR_ENV}/bin/python" -c "import chemprop; print('chemprop', chemprop.__version__)"
"${PXR_ENV}/bin/python" -I pxr/scripts/pxr002_stage_b_runner.py \
  --artifact-dir pxr/artifacts/pxr002_aux_v2 --stage-a-root "${STAGE_A_ROOT}" \
  --out-root "${OUT_ROOT}" --accelerator gpu
echo "=== PXR-002 stage B end $(date -Is)"
