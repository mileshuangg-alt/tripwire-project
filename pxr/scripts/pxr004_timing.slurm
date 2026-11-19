#!/bin/bash
#SBATCH --job-name=pxr004-timing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=pxr004_timing_%j.out
#SBATCH --error=pxr004_timing_%j.err

# PXR-004 feasibility timing (D020). Measures speed and memory of the two stages on the
# target L40S; it starts no training run, reads no loss, writes nothing under
# pxr/artifacts and chooses no hyperparameter. Miles chose a batch job because the
# interactive srun allocation on the gpu partition failed again (2026-11-19, jobs 3132497
# and 3132498); this is a recorded deviation from the interactive-benchmark practice.
# Wall time: the script builds the datasets, then times 3 warmup + 20 optimizer steps,
# one validation pass and three checkpoint writes per stage; the 1 h request follows the
# PXR-002 wrappers and is generous for that.
# The job verifies every staged file against the reviewed hash list before running.
# Submit from the staging root, not from inside the extracted tree, so the .out/.err
# files do not land in it.
set -euo pipefail

: "${STAGE_ROOT:?STAGE_ROOT must be set explicitly (no default)}"
: "${PXR_ENV:?PXR_ENV must be set explicitly (no default)}"
: "${HASH_LIST:?HASH_LIST must point to the reviewed pxr004_stage_hashes file}"
: "${OUT_JSON:?OUT_JSON must be a new output file path (no default)}"
[ ! -e "${OUT_JSON}" ] || { echo "OUT_JSON already exists: ${OUT_JSON}"; exit 2; }

SCRATCH_RUNTIME="/scratch/user/mhuang8/pxr004_runtime/${SLURM_JOB_ID:-interactive}"
mkdir -p "${SCRATCH_RUNTIME}"/{tmp,xdg_cache,torch,cuda_cache}
export TMPDIR="${SCRATCH_RUNTIME}/tmp" XDG_CACHE_HOME="${SCRATCH_RUNTIME}/xdg_cache"
export TORCH_HOME="${SCRATCH_RUNTIME}/torch" CUDA_CACHE_PATH="${SCRATCH_RUNTIME}/cuda_cache"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

echo "=== PXR-004 timing start $(date -Is) on $(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "${STAGE_ROOT}"
sha256sum -c "${HASH_LIST}"
"${PXR_ENV}/bin/python" -I -c "import chemprop, torch, lightning, pandas, scipy; print('chemprop', chemprop.__version__, 'torch', torch.__version__, 'lightning', lightning.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__, 'cuda', torch.cuda.is_available())"

EXTRA=()
if [ -n "${REQUESTED_WALL_HOURS:-}" ]; then
  EXTRA=(--requested-wall-hours "${REQUESTED_WALL_HOURS}")
fi
"${PXR_ENV}/bin/python" -I pxr/scripts/pxr004_timing.py \
  --accelerator gpu --out-json "${OUT_JSON}" \
  --artifact-dir pxr/artifacts/pxr002_aux \
  --assignments pxr/artifacts/pxr001_split/pxr001_split_assignments.csv \
  --monitor-dir pxr/artifacts/pxr004_monitor \
  --weights pxr/artifacts/pxr004_weights/chemeleon_mp.pt ${EXTRA[@]+"${EXTRA[@]}"}
echo "=== PXR-004 timing end $(date -Is) exit=0"
