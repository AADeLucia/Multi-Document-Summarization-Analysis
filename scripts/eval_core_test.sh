#!/bin/sh
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=core
#SBATCH --mem=80GB
#SBATCH -A mdredze1_gpu
#SBATCH --output=logs/%x.%A.log

# Environment
module load anaconda
conda activate core

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ];then
  SLURM_ARRAY_TASK_ID=1
fi

PROJECT_DIR="${MDS}/dependencies/adversarial-factuality"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
export API_KEY="123"
export NLI_MODEL="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# 1. Connect to served vLLM
export PORT=22659
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
NODE="icgpu03"
export SERVER="http://${NODE}:${PORT}"

# Settings
DATA="multinews_plus"
SPLIT="val"
MODEL="ground_truth"

export DECOMP_EXAMPLES="${MDS}/data/${DATA}/core_decomp_examples.txt"
export OUTPUT_PATH="${MDS}/results/${MODEL}/${DATA}/${SPLIT}/core_output.jsonl.demo"
export DATASET="${MDS}/data/${DATA}/${SPLIT}.jsonl"
export INPUT_PATH="${MDS}/results/${MODEL}/${DATA}/${SPLIT}/generated_with_ids.jsonl.demo"
export BLEACHED_TEMPLATES="${MDS}/data/${DATA}/core_claim_bleached_templates.txt"

# Run
python "scripts/run_task.py" "configs/core.yaml"
