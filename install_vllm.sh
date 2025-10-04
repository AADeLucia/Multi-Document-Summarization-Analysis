#!/bin/bash

module purge
module load gcc/9.3.0
module load anaconda3/2024.02-1
module load cuda/12.3.0

conda activate vllm

pip uninstall vllm -y

# Set CUDA paths for 12.3
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDACXX=$(which nvcc)
export CMAKE_CUDA_COMPILER=$(which nvcc)

# Install from source
pip install vllm==0.6.1.post2 --no-binary vllm --no-cache-dir
