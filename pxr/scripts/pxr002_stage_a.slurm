#!/bin/bash
#SBATCH --job-name=pxr002-stage-a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=pxr002_stage_a_%j.out
#SBATCH --error=pxr002_stage_a_%j.err

# PXR-002 stage A (D018): the 36-configuration grid, one common seed, run serially in
# one job. Wall time rationale: pilot job 3113883 measured ~5.1 s/epoch for the largest
# corner, so the worst case (36 runs x 100 epochs) is ~5.1 h; 8 h leaves ~55% margin for
# per-run startup (not separately measured) and node variance. The expected case is much
# shorter (early stopping, patience 20). Nothing resumes: a failed run is recorded and the
# rest continue; a timeout leaves the finished runs valid and the rest unrun.
# No stage B, ablation, ensemble or holdout/test evaluation happens here.
set -euo pipefail

: "${STAGE_ROOT:?STAGE_ROOT must be set explicitly (no default)}"
: "${PXR_ENV:?PXR_ENV must be set explicitly (no default)}"
: "${HASH_LIST:?HASH_LIST must point to the reviewed hash list}"
: "${OUT_ROOT:?OUT_ROOT must be a new output directory (no default)}"
: "${AUTHORIZE_STAGE_A:?set AUTHORIZE_STAGE_A=yes only with explicit launch approval from Miles}"
[ "${AUTHORIZE_STAGE_A}" = "yes" ] || { echo "AUTHORIZE_STAGE_A must be exactly 'yes'"; exit 2; }
[ ! -e "${OUT_ROOT}" ] || { echo "OUT_ROOT already exists: ${OUT_ROOT}"; exit 2; }

SCRATCH_RUNTIME="/scratch/user/mhuang8/pxr002_runtime/${SLURM_JOB_ID:-interactive}"
mkdir -p "${SCRATCH_RUNTIME}"/{tmp,xdg_cache,torch,cuda_cache}
export TMPDIR="${SCRATCH_RUNTIME}/tmp" XDG_CACHE_HOME="${SCRATCH_RUNTIME}/xdg_cache"
export TORCH_HOME="${SCRATCH_RUNTIME}/torch" CUDA_CACHE_PATH="${SCRATCH_RUNTIME}/cuda_cache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

echo "=== PXR-002 stage A start $(date -Is) on $(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "${STAGE_ROOT}"
sha256sum -c "${HASH_LIST}"
"${PXR_ENV}/bin/python" -c "import chemprop; print('chemprop', chemprop.__version__)"
"${PXR_ENV}/bin/python" -I pxr/scripts/pxr002_stage_a_runner.py \
  --artifact-dir pxr/artifacts/pxr002_aux_v2 --out-root "${OUT_ROOT}" --accelerator gpu
echo "=== PXR-002 stage A end $(date -Is)"
