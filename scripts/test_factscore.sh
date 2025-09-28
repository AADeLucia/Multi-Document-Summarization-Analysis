#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=factscore
#SBATCH --partition=ica100
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH -A mdredze80_gpu
#SBATCH --output=logs/%x.%A.%a.log
#SBATCH --array=8

# Experiment settings
CONFIG="${MDS}/code/experiments_abridged.json"

# Parse the settings
source "${MDS}/code/helper_functions.sh"
parse_settings "${CONFIG}"
module load anaconda
conda activate mds

# Overwrite splits
SPLITS=( "val" )

export CUDA_VISIBLE_DEVICES=0

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ];then
  SLURM_ARRAY_TASK_ID=3
fi

# Group by (dataset, model) for the task array
task_id=1
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    task_desc="On task ${task_id}: ${model}-${dataset}"
    if [ "${task_id}" -eq "${SLURM_ARRAY_TASK_ID}" ];then
      echo "${task_desc}"
      for split in "${SPLITS[@]}"; do
        INPUT="${MDS}/data/${dataset}/${split}.jsonl"
        OUTPUT_DIR="${MDS}/results/${model}/${dataset}/${split}"
        mkdir -p "${OUTPUT_DIR}"

        echo "FactScore"
        python "${MDS}/code/calculate_factscore.py" \
          --input-data "${INPUT}" \
          --output-data "${OUTPUT_DIR}/generated.jsonl" \
          --output-dir "${OUTPUT_DIR}" \
          --dataset "${dataset}" \
          --mode "single" \
          --model-name "retrieval+llama+npm"

        if [ $? -ne 0 ];then
          echo "Error on ${task_desc} FactScore"
          exit 1
        fi

        echo "FactScore-MD"
        python "${MDS}/code/calculate_factscore.py" \
          --input-data "${INPUT}" \
          --output-data "${OUTPUT_DIR}/generated.jsonl" \
          --output-dir "${OUTPUT_DIR}" \
          --dataset "${dataset}" \
          --mode "multi" \
          --model-name "retrieval+llama"

        if [ $? -ne 0 ];then
          echo "Error on ${task_desc} FactScore-MD"
          exit 1
        fi

        python "${MDS}/code/evaluate_generations.py" \
          --input-data "${INPUT}" \
          --output-data "${OUTPUT_DIR}/generated.jsonl" \
          --output-dir "${OUTPUT_DIR}" \
          --metrics "factscore" "factscore-md"

        if [ $? -ne 0 ];then
          echo "Error on ${task_desc}"
          exit 1
        fi
      done
    fi

    # Increment task counter
    task_id=$(( task_id + 1))

  done
done

echo "Done with task ${SLURM_ARRAY_TASK_ID}"