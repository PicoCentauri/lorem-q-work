#!/bin/bash
#SBATCH --job-name=razor-lr
#SBATCH --output=slurm.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time 24:00:00

module purge
module load cuda/13.2.0
# python 3.12.0 (the python/3.12-base module) deadlocks in
# multiprocessing.resource_tracker under grain's shared-memory prefetch; the
# reentrancy fix landed in 3.12.1. the spack modulepath must be loaded first.
module load 000-all-spack-pkgs/1.1.1-alex
module load python/3.13.8-gcc11.5.0-apqbs3h
source ~/venv/lorem313/bin/activate

export PYTHONUNBUFFERED=1

export DATASETS=..
lorem-train
