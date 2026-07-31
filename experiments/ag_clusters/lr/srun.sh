#!/bin/bash
#SBATCH --job-name=ag_clusters-lr
#SBATCH --output=slurm.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time 12:00:00

module purge
module load cuda/13.2.0
source ~/venv/lorem-q/bin/activate

export PYTHONUNBUFFERED=1

export DATASETS=..
lorem-train
