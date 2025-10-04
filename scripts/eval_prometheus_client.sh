#!/bin/bash
# Should be run AFTER starting the vLLM server with serve_vllm.sh
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=eval-prom
#SBATCH --mem=15GB
#SBATCH -A mdredze1
#SBATCH --output=logs/%x.%A.log

# Source bashrc
source ~/.bashrc

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Environment
module load anaconda3/2024.02-1
conda activate vllm
source "${MDS}/scripts/helper_functions.sh"

# Connect to served vLLM
SERVER="http://c007:22659/v1"
CONFIG="${MDS}/code/experiments.json"
OUTPUT_BASE="${MDS}/results"


# Parse the settings
parse_settings "${CONFIG}"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for split in "${SPLITS[@]}"; do
      INPUT="${MDS}/data/${dataset}/${split}.jsonl"
      OUTPUT_DIR="${OUTPUT_BASE}/${model}/${dataset}/${split}"
      echo "On ${OUTPUT_DIR}"

      python "${MDS}/code/prometheus_evaluator.py" \
        --input-data "${INPUT}" \
        --output-data "${OUTPUT_DIR}/generated.jsonl" \
        --output-dir "${OUTPUT_DIR}" \
        --server-base "${SERVER}" \
        --sleep 2

      if [ $? -ne 0 ];then
        echo "Error during evaluation"
        exit 1
      fi

      eval_generations "${INPUT}" "${OUTPUT_DIR}" "true" "prometheus"

    done
  done
done

python "${MDS}/code/aggregate_results.py" --experiments-config "${CONFIG}" --results-dir "${OUTPUT_BASE}" --ci
