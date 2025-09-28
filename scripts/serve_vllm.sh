#!/bin/bash
#SBATCH --job-name=serve-vllm
#SBATCH --time=2-00:00:00
#SBATCH --partition=ica100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH -A mdredze80_gpu
#SBATCH --output=logs/%x.%j.log

module load anaconda
conda activate mds

SLURM_JOB_GPUS=4  # Match --gres=gpu:#
CUDA_VISIBLE_DEVICES="0"
for i in $(seq 1 "$(( SLURM_JOB_GPUS - 1 ))"); do
  CUDA_VISIBLE_DEVICES+=",${i}"
done

export CUDA_VISIBLE_DEVICES
export PROJECT_DIR="/home/adeluci2/projects/vllm-server"
export PORT=22659
export MODEL_NAME="prometheus-eval/prometheus-bgb-8x7b-v2.0"

echo "Loading ${MODEL_NAME} on ${CUDA_VISIBLE_DEVICES}"

vllm serve "${MODEL_NAME}" \
    --dtype auto \
    --tensor-parallel-size "${SLURM_JOB_GPUS}" \
    --seed 42 \
    --tokenizer-mode "auto" \
    --uvicorn-log-level "warning" \
    --port ${PORT}
