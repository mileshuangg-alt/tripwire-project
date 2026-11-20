#!/bin/bash
#SBATCH --job-name=pxr004-pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=pxr004_pilot_%j.out
#SBATCH --error=pxr004_pilot_%j.err

# PXR-004 pilot (D020): one job per seed, running the direct arm and then the sequential
# arm for that seed on the same node. Three jobs, seeds 20261125, 20261126, 20261127
# (packaging and seeds approved by Miles 2026-11-19). Submit once per seed with SEED set.
# Wall time rationale: job 3132558 timing (feasibility, worst case = every run to 100
# epochs with validation and checkpoint each epoch) gave direct 0.096 h + sequential
# 0.375 h at p90 = 0.47 h for one seed; the 1 h request is about 2.1x that. Patience 20
# will usually stop runs earlier. Nothing resumes: a failure stops the job nonzero and
# leaves partial output in place for inspection. No holdout or official-test use.
# Submit from the staging root, not from inside the extracted tree.
set -euo pipefail

: "${STAGE_ROOT:?STAGE_ROOT must be set explicitly (no default)}"
: "${PXR_ENV:?PXR_ENV must be set explicitly (no default)}"
: "${HASH_LIST:?HASH_LIST must point to the reviewed pxr004_stage_hashes file}"
: "${SEED:?SEED must be one of the approved seeds 20261125, 20261126, 20261127}"
: "${OUT_ROOT:?OUT_ROOT must be a new output directory for this seed (no default)}"
: "${AUTHORIZE_PXR004_PILOT:?set AUTHORIZE_PXR004_PILOT=yes only with explicit launch approval from Miles}"
[ "${AUTHORIZE_PXR004_PILOT}" = "yes" ] || { echo "AUTHORIZE_PXR004_PILOT must be exactly 'yes'"; exit 2; }
case "${SEED}" in 20261125|20261126|20261127) ;; *) echo "SEED ${SEED} is not an approved seed"; exit 2;; esac
[ ! -e "${OUT_ROOT}" ] || { echo "OUT_ROOT already exists: ${OUT_ROOT}"; exit 2; }

SCRATCH_RUNTIME="/scratch/user/mhuang8/pxr004_runtime/${SLURM_JOB_ID:-interactive}"
mkdir -p "${SCRATCH_RUNTIME}"/{tmp,xdg_cache,torch,cuda_cache}
export TMPDIR="${SCRATCH_RUNTIME}/tmp" XDG_CACHE_HOME="${SCRATCH_RUNTIME}/xdg_cache"
export TORCH_HOME="${SCRATCH_RUNTIME}/torch" CUDA_CACHE_PATH="${SCRATCH_RUNTIME}/cuda_cache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

echo "=== PXR-004 pilot start $(date -Is) on $(hostname) job=${SLURM_JOB_ID:-none} seed=${SEED}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "${STAGE_ROOT}"
sha256sum -c "${HASH_LIST}"
"${PXR_ENV}/bin/python" -I -c "import chemprop, torch, lightning, pandas, scipy; print('chemprop', chemprop.__version__, 'torch', torch.__version__, 'lightning', lightning.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__, 'cuda', torch.cuda.is_available())"

for ARM in direct sequential; do
  echo "=== arm ${ARM} seed ${SEED} start $(date -Is)"
  "${PXR_ENV}/bin/python" -I pxr/scripts/pxr004_train.py \
    --arm "${ARM}" --seed "${SEED}" --accelerator gpu --out-dir "${OUT_ROOT}" \
    --artifact-dir pxr/artifacts/pxr002_aux \
    --assignments pxr/artifacts/pxr001_split/pxr001_split_assignments.csv \
    --monitor-dir pxr/artifacts/pxr004_monitor \
    --weights pxr/artifacts/pxr004_weights/chemeleon_mp.pt
  echo "=== arm ${ARM} seed ${SEED} end $(date -Is)"
done
echo "=== PXR-004 pilot end $(date -Is) exit=0"
