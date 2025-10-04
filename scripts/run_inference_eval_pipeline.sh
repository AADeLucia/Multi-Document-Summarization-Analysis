#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --job-name=mds-pipe
#SBATCH --partition=a100,h100
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
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

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Experiment settings for this task
# depends on the SLURM_ARRAY_TASK_ID
if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
  SLURM_ARRAY_TASK_ID=1
  SLURM_ARRAY_TASK_MAX=1
fi
CONFIG="${MDS}/code/experiments.json"
params=$(sed -n ${SLURM_ARRAY_TASK_ID}p "${MDS}/code/experiment_array.jsonl")
echo "Running task ${SLURM_ARRAY_TASK_ID} with params: ${params}"
parse_settings_from_json "${params}"
OUTPUT_BASE="${MDS}/results"
mkdir -p "${OUTPUT_BASE}"

DO_PREDICT=false
DO_EVAL=true

for split in "${SPLITS[@]}"; do
  INPUT="${MDS}/data/${DATASET}/${split}.jsonl"
  OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}/${DATASET}/${split}"
  mkdir -p "${OUTPUT_DIR}"

  if [ "${DO_PREDICT}" = true ]; then
    infer_model "${MODEL}" "${INPUT}" "${OUTPUT_DIR}" "${DATASET}"

    if [ $? -ne 0 ];then
      echo "Error during inference"
      exit 1
    fi
  fi

  if [ "${DO_EVAL}" = true ]; then
    eval_generations "${INPUT}" "${OUTPUT_DIR}" "false" "${METRICS[@]}"

    if [ $? -ne 0 ];then
      echo "Error during evaluation"
      exit 1
    fi
  fi
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
