#!/bin/bash
#SBATCH --job-name=af-weighted-validation-check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/user/mhuang8/tripwire_staging/actfound_weighted_validation_check_20261117/gpu_validation_check/slurm-%j.out

export PYTHONNOUSERSITE=1
/scratch/user/mhuang8/actfound/envs/actfound_c96f055/bin/python -I \
  /scratch/user/mhuang8/tripwire_staging/actfound_weighted_validation_check_20261117/gpu_validation_check/check.py
