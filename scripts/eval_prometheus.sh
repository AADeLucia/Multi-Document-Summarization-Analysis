#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=eval-prom
#SBATCH --partition=a100,h100
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:2
#SBATCH -A mdredze1
#SBATCH --output=logs/%x.%A.%a.log
#SBATCH --array=1-42%10  # Should be {# datasets} x {# models}

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
conda activate mds
source "${MDS}/scripts/helper_functions.sh"

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1

# Experiment settings for this task
# depends on the SLURM_ARRAY_TASK_ID
if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
  SLURM_ARRAY_TASK_ID=1
  SLURM_ARRAY_TASK_MAX=5
fi
CONFIG="${MDS}/code/experiments.json"
params=$(sed -n ${SLURM_ARRAY_TASK_ID}p "${MDS}/code/experiment_array.jsonl")
echo "Running task ${SLURM_ARRAY_TASK_ID} with params: ${params}"
parse_settings_from_json "${params}"
OUTPUT_BASE="${MDS}/results"
mkdir -p "${OUTPUT_BASE}"

for split in "${SPLITS[@]}"; do
  INPUT="${MDS}/data/${DATASET}/${split}.jsonl"
  OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}/${DATASET}/${split}"
  mkdir -p "${OUTPUT_DIR}"

  python "${MDS}/code/prometheus_evaluator.py" \
    --input-data "${INPUT}" \
    --output-data "${OUTPUT_DIR}/generated_with_ids.jsonl" \
    --output-dir "${OUTPUT_DIR}" \
    --n-examples 100 \
    --model-path "prometheus-eval/prometheus-bgb-8x7b-v2.0" \
    --include-reference \
    --batch-size 1 \
    --device "cuda" \
    --torch-dtype "bfloat16" \
    --max-tokens 1024 \
    --temperature 0.9

  if [ $? -ne 0 ];then
    echo "Error during evaluation"
    exit 1
  fi

  # eval_generations "${INPUT}" "${OUTPUT_DIR}" "true" "prometheus"
done

# Check if this is the last array task AND the last running
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX}" ]; then
  echo "This is the last array task. Waiting for other tasks to finish..."
  while true; do
    RUNNING_TASKS=$(squeue -u ${USER} -j ${SLURM_JOB_ID} -h | wc -l)
    if [ "${RUNNING_TASKS}" -eq 1 ]; then
      break
    fi
    sleep 10
  done
  echo "Aggregating results"
  python "${MDS}/code/aggregate_results.py" --experiments-config "${CONFIG}" --results-dir "${OUTPUT_BASE}" --ci
fi

echo "Done with task ${SLURM_ARRAY_TASK_ID}"

