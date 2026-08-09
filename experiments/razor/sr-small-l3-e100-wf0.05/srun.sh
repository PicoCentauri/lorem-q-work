#!/bin/bash
#SBATCH --job-name=razor-sm-l3-e100w005
#SBATCH --output=slurm.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time 24:00:00

# `module` is a shell function, and an sbatch script is neither a login nor an
# interactive shell -- it only ever resolved by inheriting BASH_FUNC_module
# from an interactive submitting shell, so submitting over a plain `ssh host
# 'sbatch ...'` silently skipped every load below. Source the init explicitly.
# (/etc/profile.d/modules.sh is deliberately empty here; `system` is a symlink
# to the current version.)
source /apps/modules/system/init/bash

module purge
module load cuda/13.2.0
# python 3.12.0 (the python/3.12-base module) deadlocks in
# multiprocessing.resource_tracker under grain's shared-memory prefetch; the
# reentrancy fix landed in 3.12.1. the spack modulepath must be loaded first.
module load 000-all-spack-pkgs/1.1.1-alex
module load python/3.13.8-gcc11.5.0-apqbs3h
# lorem-wf, not lorem313: these are energy+force runs today, but the
# settings.yaml TODO anticipates adding the work function once the weight
# sweep settles, and that needs lorem.LoremQ -- which only exists in the
# charge-conditioning build.
source ~/venv/lorem-wf/bin/activate

export PYTHONUNBUFFERED=1

export DATASETS=..
lorem-train
