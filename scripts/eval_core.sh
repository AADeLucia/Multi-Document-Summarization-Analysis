#!/bin/sh
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=core
#SBATCH --mem=60GB
#SBATCH -A mdredze80_gpu
#SBATCH --output=logs/%x.%A.%a.log
#SBATCH --array=1-80%1  # Should be {# datasets} x {# models}
#SBATCH --partition=ica100,a100
#SBATCH --gres=gpu:1

# Environment
module load anaconda
conda activate core

source "${MDS}/code/helper_functions.sh"

# Experiment settings
# Loaded from config but can be overridden below
CONFIG="${MDS}/code/experiments.json"

# Parse the settings
parse_settings "${CONFIG}"

# Overwrite
# SPLITS=( "test" )

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ];then
  SLURM_ARRAY_TASK_ID=16
fi

PROJECT_DIR="${MDS}/dependencies/adversarial-factuality"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
export API_KEY="123"
export NLI_MODEL="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
CONFIG="${PROJECT_DIR}/configs/core_gpu.yaml"

# 1. Connect to served vLLM
export PORT=22659
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
NODE="icgpu10"
export SERVER="http://${NODE}:${PORT}"

task_id=1
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    if [ "${task_id}" -eq "${SLURM_ARRAY_TASK_ID}" ];then
      for split in "${SPLITS[@]}"; do
        echo "On ${task_id}-${model}-${dataset}-${split}"
        export DECOMP_EXAMPLES="${MDS}/data/${dataset}/core_decomp_examples.txt"
        export OUTPUT_PATH="${MDS}/results/${model}/${dataset}/${split}/core_output.jsonl"
        export DATASET="${MDS}/data/${dataset}/${split}.jsonl"
        export INPUT_PATH="${MDS}/results/${model}/${dataset}/${split}/generated_with_ids.jsonl"
        export BLEACHED_TEMPLATES="${MDS}/data/${dataset}/core_claim_bleached_templates.txt"

        # Do not run if results are done
        if [ -f "${OUTPUT_PATH}" ]; then
           echo "Generations exist in ${OUTPUT_PATH}. Skipping."
           continue
        fi

        # Run
        # can add a --cache-path but it gets random errors
        python "scripts/run_task.py" "${CONFIG}"

        # Load in evaluations.json
        # INPUT="${MDS}/results/${model}/${dataset}/${split}/generated.jsonl"
        # OUTPUT_DIR="${MDS}/results/${model}/${dataset}/${split}"
        # eval_generations "${INPUT}" "${OUTPUT_DIR}" "core"

        if [ $? -ne 0 ];then
          echo "Error"
          exit 1
        fi

        # python "${MDS}/code/aggregate_results.py" --experiments-config "${CONFIG}"
      done
    fi
    # Increment task counter
    task_id=$(( task_id + 1))
  done
done

echo "Done with task ${SLURM_ARRAY_TASK_ID}"
