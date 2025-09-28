#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=eval-prom
#SBATCH --mem=15GB
#SBATCH -A mdredze1_gpu
#SBATCH --output=logs/%x.%A.log

module load anaconda
conda activate mds
source "${MDS}/scripts/helper_functions.sh"

# Connect to served vLLM
SERVER="http://icgpu09:22659/v1"
CONFIG="${MDS}/code/experiments_mayarr.json"
OUTPUT_BASE="${MDS}/results_mayarr_sample"

# Parse the settings
parse_settings "${CONFIG}"
export CUDA_VISIBLE_DEVICES=0

counter=0
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for split in "${SPLITS[@]}"; do
      INPUT="${MDS}/data/${dataset}/${split}.sample.jsonl"
      OUTPUT_DIR="${OUTPUT_BASE}/${model}/${dataset}/${split}"
      echo "On ${OUTPUT_DIR}"

      if [ "${counter}" -eq 0 ]; then
        echo "counter = ${counter}"

#      python "${MDS}/code/prometheus_evaluator.py" \
#        --input-data "${INPUT}" \
#        --output-data "${OUTPUT_DIR}/generated.jsonl" \
#        --output-dir "${OUTPUT_DIR}" \
#        --server-base "${SERVER}" \
#        --dimensions coverage \
#        --sleep 2

        eval_generations "${INPUT}" "${OUTPUT_DIR}" "${METRICS[@]}"
      fi

      if [ $? -ne 0 ];then
        echo "Error during evaluation"
        exit 1
      fi

      counter=$((counter+1))
    done
  done
done

python "${MDS}/code/aggregate_results.py" --experiments-config "${CONFIG}" --results-dir "${OUTPUT_BASE}" --ci
