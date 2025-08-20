#!/bin/bash
# jq help from https://stackoverflow.com/questions/54087481/assigning-an-array-parsed-with-jq-to-bash-script-array
# https://linuxize.com/post/bash-functions/
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval
##SBATCH --partition=ica100
##SBATCH --gres=gpu:1
#SBATCH --mem=80GB
##SBATCH -A mdredze80_gpu
#SBATCH --output=logs/%x.%A.%a.log
#SBATCH --array=1-14

# Python environment
module load anaconda
conda activate mds
source "${MDS}/code/helper_functions.sh"

# Experiment settings
CONFIG="${MDS}/code/experiments_mayarr.json"
OUTPUT_BASE="${MDS}/results_mayarr"

# Parse the settings
parse_settings "${CONFIG}"

# Override metrics
#METRICS=(
#  "rouge"
#)

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ];then
  SLURM_ARRAY_TASK_ID=1
fi

# Group by model for the task array
task_id=1
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
      if [ "${task_id}" -eq "${SLURM_ARRAY_TASK_ID}" ];then
        for split in "${SPLITS[@]}"; do
          echo "On ${task_id}-${model}-${dataset}-${split}"
          INPUT="${MDS}/data/${dataset}/${split}.jsonl"
          OUTPUT_DIR="${OUTPUT_BASE}/${model}/${dataset}/${split}"
          mkdir -p "${OUTPUT_DIR}"

          python "${MDS}/code/evaluate_generations.py" \
            --input-data "${INPUT}" \
            --output-data "${OUTPUT_DIR}/generated.jsonl" \
            --output-dir "${OUTPUT_DIR}" \
            --metrics "${METRICS[@]}"

          if [ $? -ne 0 ];then
            echo "ERROR on ${model}-${dataset}-${split}"
            continue
          fi

          python "${MDS}/code/aggregate_results.py" --experiments-config "${CONFIG}"
          python "${MDS}/code/prepare_for_annotation.py" --experiments-config "${CONFIG}"
        done
      fi
      # Increment task counter
      task_id=$(( task_id + 1))
  done
done

echo "Done with task ${SLURM_ARRAY_TASK_ID}"
