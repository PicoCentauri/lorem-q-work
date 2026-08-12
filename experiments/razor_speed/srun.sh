#!/bin/bash
#SBATCH --job-name=razor-speed
#SBATCH --output=slurm.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time 01:00:00

# see sr/srun.sh -- `module` is a shell function an sbatch script does not
# inherit unless it was submitted from an interactive shell.
source /apps/modules/system/init/bash

module purge
module load cuda/13.2.0
# see sr/srun.sh -- python 3.12.0 has a multiprocessing.resource_tracker
# deadlock; the spack modulepath must be loaded before the python module.
module load 000-all-spack-pkgs/1.1.1-alex
module load python/3.13.8-gcc11.5.0-apqbs3h
# lorem-wf, not lorem313: every variant evaluated here was trained as
# lorem.LoremQ, and load_checkpoint deserialises that class name straight out
# of the checkpoint's model.yaml. lorem313 predates LoremQ and would fail on
# the very first from_dict.
source ~/venv/lorem-wf/bin/activate

export PYTHONUNBUFFERED=1

# XLA's Triton GEMM autotuner fails on this jaxlib build with
# device_type "DEVICE_TYPE_INVALID" -- it cannot identify the GPU, so it has
# no candidate configs and dies ("Autotuning failed", or "No supported config
# found" if autotuning is merely switched off, which is not the same fix).
# Seen on both a40 (these evaluate jobs) and a100 (sr-wf-bec's post-training
# collation), so it is the Triton path rather than a specific card. Routing
# those fusions to cuBLAS instead avoids the autotuner entirely.
# NOTE for a *timing* run: this is a correctness workaround (the Triton GEMM
# autotuner cannot identify the GPU on this jaxlib build and dies), but it is
# also a performance knob -- it routes those fusions to cuBLAS instead. Every
# number here is therefore "with Triton GEMM off". Without it the job does not
# run at all on this stack, so it is not optional, but it should be revisited
# if the jaxlib build is ever fixed.
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 benchmark.py
