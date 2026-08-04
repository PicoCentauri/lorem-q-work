#!/bin/bash
#SBATCH --job-name=razor-evaluate
#SBATCH --output=slurm.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition=a40
#SBATCH --time 02:00:00

module purge
module load cuda/13.2.0
# see sr/srun.sh -- python 3.12.0 has a multiprocessing.resource_tracker
# deadlock; the spack modulepath must be loaded before the python module.
module load 000-all-spack-pkgs/1.1.1-alex
module load python/3.13.8-gcc11.5.0-apqbs3h
source ~/venv/lorem313/bin/activate

export PYTHONUNBUFFERED=1

python3 evaluate.py
