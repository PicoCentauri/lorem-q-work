#!/bin/bash
#SBATCH --job-name=cpmace-grace-w500-16-1-huber
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
source /apps/modules/system/init/bash

module purge
module load 000-all-spack-pkgs/1.1.1-alex
module load python/3.13.8-gcc11.5.0-apqbs3h
source ~/venv/grace/bin/activate

# TensorFlow pulls in keras 3.x as a dependency, but tensorpotential needs the
# legacy Keras API bundled with TF. Without this the import picks keras 3 and
# fails. Documented requirement, not a workaround.
export TF_USE_LEGACY_KERAS=1

# tensorflow[and-cuda] ships its own CUDA libraries, so no cuda module is loaded
# on purpose -- loading one alongside risks two CUDA runtimes in one process.

nvidia-smi

# data/ is shared with the parent experiment; build it only if absent
if [ ! -f ../data/train.xyz ]; then
    (cd .. && python prepare.py)
fi

gracemaker input.yaml
