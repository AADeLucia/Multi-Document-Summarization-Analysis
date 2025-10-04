#!/bin/bash
#SBATCH --job-name=serve-vllm
#SBATCH --time=2-00:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH -A mdredze1
#SBATCH --output=logs/%x.%j.log

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

module load anaconda3/2024.02-1
conda activate vllm

SLURM_JOB_GPUS=2  # Match --gres=gpu:#
CUDA_VISIBLE_DEVICES="0"
for i in $(seq 1 "$(( SLURM_JOB_GPUS - 1 ))"); do
  CUDA_VISIBLE_DEVICES+=",${i}"
done

export CUDA_VISIBLE_DEVICES
export PROJECT_DIR="${MDS}/vllm-server"
export PORT=22659
export MODEL_NAME="prometheus-eval/prometheus-bgb-8x7b-v2.0"
export MODEL_NAME="mistralai/Mixtral-8x7B-Instruct-v0.1"

echo "Loading ${MODEL_NAME} on ${CUDA_VISIBLE_DEVICES}"

VLLM_LOGGING_LEVEL=DEBUG \
vllm serve "${MODEL_NAME}" \
    --dtype bfloat16 \
    --tensor-parallel-size "${SLURM_JOB_GPUS}" \
    --seed 42 \
    --tokenizer-mode "auto" \
    --uvicorn-log-level "warning" \
    --port ${PORT} \
    --gpu-memory-utilization 0.7 \
    --enforce-eager

#VLLM_LOGGING_LEVEL=DEBUG \
#VLLM_WORKER_MULTIPROC_METHOD=spawn \
#vllm serve "${MODEL_NAME}" \
#    --tensor-parallel-size 1 \
#    --seed 42 \
#    --port ${PORT} \
#    --enforce-eager \
#    --max-model-len 4096 \
#    --gpu-memory-utilization 0.8


