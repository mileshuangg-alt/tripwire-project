#!/bin/bash
#SBATCH --job-name=t121-d021-timing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/user/mhuang8/tripwire_staging/t121_d021_package_20261119/smoke/timing-%j.out
# Batch deviation for the D021 timing only (requires Miles's approval; interactive
# srun fails with "Insane message length"). Same topology as the production cell
# array. The 2-hour value is a wall-time cap, not an estimate.
export PYTHONNOUSERSITE=1
/scratch/user/mhuang8/envs/tripwire_cyp002/bin/python -I \
  /scratch/user/mhuang8/tripwire_staging/t121_d021_package_20261119/smoke/timing_fixed.py \
  --out /scratch/user/mhuang8/tripwire_staging/t121_d021_timing_20261120
