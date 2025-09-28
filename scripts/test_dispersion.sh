#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=aac
#SBATCH --mem=64GB
#SBATCH --partition=ica100
#SBATCH --gres=gpu:1
#SBATCH -A mdredze80_gpu
#SBATCH --output=logs/%x.%A.%a.log
#SBATCH --array=9,14

# Experiment settings
CONFIG="${MDS}/code/experiments_abridged.json"

# Parse the settings
source "${MDS}/code/helper_functions.sh"
parse_settings "${CONFIG}"
module load anaconda
conda activate multi_mds

export CUDA_VISIBLE_DEVICES=0

# Overwrite splits
SPLITS=( "val" )

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ];then
  SLURM_ARRAY_TASK_ID=9
fi

# Group by (dataset, model) for the task array
task_id=1
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    task_desc="On task ${task_id}: ${model}-${dataset}"
    if [ "${task_id}" -eq "${SLURM_ARRAY_TASK_ID}" ];then
      for split in "${SPLITS[@]}"; do
        echo "${task_desc}"

        INPUT="${MDS}/data/${dataset}/${split}.jsonl"
        OUTPUT_DIR="${MDS}/results/${model}/${dataset}/${split}"

        python "${MDS}/code/evaluate_generations.py" \
            --input-data "${INPUT}" \
            --output-data "${OUTPUT_DIR}/generated.jsonl" \
            --output-dir "${OUTPUT_DIR}" \
            --metrics "dispersion"

        if [ $? -ne 0 ];then
          echo "ERROR on ${task_desc}"
        fi
      done
    fi
    # Increment task counter
    task_id=$(( task_id + 1))
  done
done

echo "Done with task ${SLURM_ARRAY_TASK_ID}"
