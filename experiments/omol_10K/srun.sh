#!/bin/bash
#SBATCH --job-name=omol_10K-evaluate
#SBATCH --output=slurm.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition=a40
#SBATCH --time 00:30:00

module purge
module load cuda/13.2.0
source ~/venv/lorem-q/bin/activate

export PYTHONUNBUFFERED=1

python3 evaluate.py
