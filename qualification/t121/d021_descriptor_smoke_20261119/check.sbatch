#!/bin/bash
#SBATCH --job-name=t121-d021-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/user/mhuang8/tripwire_staging/t121_d021_package_20261119/smoke/slurm-%j.out

# 2-epoch cap for a smoke; this is a wall-time cap, not a runtime estimate.
export PYTHONNOUSERSITE=1
/scratch/user/mhuang8/envs/tripwire_cyp002/bin/python -I \
  /scratch/user/mhuang8/tripwire_staging/t121_d021_package_20261119/qualification/t121/d021_descriptor_smoke_20261119/check.py
