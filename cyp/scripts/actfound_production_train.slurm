#!/bin/bash
#SBATCH --job-name=actfound-prod
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=actfound_production_%j.out
#SBATCH --error=actfound_production_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nfs/CX900004_DS117/tripwire-project}"
ACTFOUND_ENV="${ACTFOUND_ENV:-/scratch/user/mhuang8/actfound/envs/actfound_c96f055}"
ACTFOUND_CONFIG="${ACTFOUND_CONFIG:-}"

if [[ -z "${ACTFOUND_CONFIG}" ]]; then
  echo "ACTFOUND_CONFIG must point to the reviewed ActFoundProductionConfigV1 JSON" >&2
  exit 2
fi

if [[ ! -f "${ACTFOUND_CONFIG}" ]]; then
  echo "ACTFOUND_CONFIG does not exist: ${ACTFOUND_CONFIG}" >&2
  exit 2
fi

if [[ ! -x "${ACTFOUND_ENV}/bin/python" ]]; then
  echo "Pinned ActFound Python is missing: ${ACTFOUND_ENV}/bin/python" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

"${ACTFOUND_ENV}/bin/python" \
  "${PROJECT_ROOT}/cyp/scripts/actfound_production_train.py" \
  --config "${ACTFOUND_CONFIG}" \
  --repo-root "${PROJECT_ROOT}" \
  --upstream-dir "${PROJECT_ROOT}/external/ActFound"
