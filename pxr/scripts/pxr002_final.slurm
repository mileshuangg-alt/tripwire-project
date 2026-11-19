#!/bin/bash
#SBATCH --job-name=pxr002-final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --no-requeue
#SBATCH --output=pxr002_final_%j.out
#SBATCH --error=pxr002_final_%j.err

# PXR-002 final training and holdout PREDICTION (approved protocol, 2026-11-18): the stage B
# selected configuration, five fresh seeds (20261120-20261124), two variants (multitask,
# aux_off), train split only. Wall time rationale: stages A/B and the ablation measured about
# 2.2 min per run, so ten runs take about 22 minutes, plus prediction (two replicated
# validation passes and one holdout pass per model, a few minutes); 1.5 h leaves a wide
# margin. This job reads NO holdout labels: the holdout artifact is structure-only and
# evaluation happens locally in pxr002_final_eval.py. Nothing resumes.
set -euo pipefail

: "${STAGE_ROOT:?STAGE_ROOT must be set explicitly (no default)}"
: "${PXR_ENV:?PXR_ENV must be set explicitly (no default)}"
: "${HASH_LIST:?HASH_LIST must point to the reviewed hash list}"
: "${STAGE_B_ROOT:?STAGE_B_ROOT must be the completed stage B output root}"
: "${OUT_ROOT:?OUT_ROOT must be a new output directory (no default)}"
: "${AUTHORIZE_FINAL_TRAINING:?set AUTHORIZE_FINAL_TRAINING=yes only with explicit launch approval from Miles}"
[ "${AUTHORIZE_FINAL_TRAINING}" = "yes" ] || { echo "AUTHORIZE_FINAL_TRAINING must be exactly 'yes'"; exit 2; }
[ ! -e "${OUT_ROOT}" ] || { echo "OUT_ROOT already exists: ${OUT_ROOT}"; exit 2; }

SCRATCH_RUNTIME="/scratch/user/mhuang8/pxr002_runtime/${SLURM_JOB_ID:-interactive}"
mkdir -p "${SCRATCH_RUNTIME}"/{tmp,xdg_cache,torch,cuda_cache}
export TMPDIR="${SCRATCH_RUNTIME}/tmp" XDG_CACHE_HOME="${SCRATCH_RUNTIME}/xdg_cache"
export TORCH_HOME="${SCRATCH_RUNTIME}/torch" CUDA_CACHE_PATH="${SCRATCH_RUNTIME}/cuda_cache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

echo "=== PXR-002 final start $(date -Is) on $(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "${STAGE_ROOT}"
sha256sum -c "${HASH_LIST}"
"${PXR_ENV}/bin/python" -c "import chemprop; print('chemprop', chemprop.__version__)"
"${PXR_ENV}/bin/python" -I pxr/scripts/pxr002_final_runner.py \
  --artifact-dir pxr/artifacts/pxr002_aux_v2 --stage-b-root "${STAGE_B_ROOT}" \
  --out-root "${OUT_ROOT}" --accelerator gpu
"${PXR_ENV}/bin/python" -I pxr/scripts/pxr002_final_predict.py \
  --artifact-dir pxr/artifacts/pxr002_aux_v2 --holdout-dir pxr/artifacts/pxr002_holdout_v1 \
  --final-root "${OUT_ROOT}" --out-dir "${OUT_ROOT}/predictions" --accelerator gpu
echo "=== PXR-002 final end $(date -Is)"
