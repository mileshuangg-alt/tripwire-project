#!/bin/bash
#SBATCH --job-name=t121-d015-screen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --output=t121_d015_screen_%j.out
#SBATCH --error=t121_d015_screen_%j.err
# No --time here: the screening wall time is not recorded and benchmarking is
# waived (D015). Pass --time explicitly on the sbatch command line.

# D015 updated-T121 screening stage (50 epochs). The launcher verifies every
# pinned file before importing scientific code. A failed run is resubmitted
# into a new output_dir; nothing resumes.
set -euo pipefail

: "${PACKAGE_ROOT:?PACKAGE_ROOT must be set explicitly (no default)}"
: "${T121_ENV:?T121_ENV must be set explicitly (no default)}"
: "${SCREEN_MANIFEST:?SCREEN_MANIFEST must point to the reviewed manifest}"

SCRATCH_RUNTIME="/scratch/user/mhuang8/actfound/runtime/t121_${SLURM_JOB_ID:-interactive}"
mkdir -p "${SCRATCH_RUNTIME}"/{tmp,xdg_cache,torch,cuda_cache}
export TMPDIR="${SCRATCH_RUNTIME}/tmp" XDG_CACHE_HOME="${SCRATCH_RUNTIME}/xdg_cache"
export TORCH_HOME="${SCRATCH_RUNTIME}/torch" CUDA_CACHE_PATH="${SCRATCH_RUNTIME}/cuda_cache"
export PYTHONNOUSERSITE=1

echo "=== T121 screen start $(date -Is) on $(hostname) job=${SLURM_JOB_ID:-none}"
echo "manifest_sha256=$(sha256sum "${SCREEN_MANIFEST}" | cut -c1-64)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "${PACKAGE_ROOT}"
"${T121_ENV}/bin/python" -I "${PACKAGE_ROOT}/cyp/scripts/t121_screen_chain_run.py" \
  --manifest "${SCREEN_MANIFEST}" --authorize-training
echo "=== T121 screen end $(date -Is) exit=0"
