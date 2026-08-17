#!/bin/bash
#SBATCH --job-name=cpmace-grace-evaluate
#SBATCH --output=slurm.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time 02:00:00

source /apps/modules/system/init/bash
module purge
module load 000-all-spack-pkgs/1.1.1-alex
module load python/3.13.8-gcc11.5.0-apqbs3h
source ~/venv/grace/bin/activate
export TF_USE_LEGACY_KERAS=1

python evaluate.py
