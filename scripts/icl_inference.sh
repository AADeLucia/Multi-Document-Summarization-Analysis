#!/bin/bash
#SBATCH --job-name=icl
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH -A mdredze1_gpu
#SBATCH --output=logs/%x.%A.%a.log
#SBATCH --array=1-32%1  # Should be {# datasets} x {# datasets} x {# splits}

# Failed due to long context: 17,18,19,20,21,22,23,24

module load anaconda
conda activate mds

source "${MDS}/code/helper_functions.sh"

# Experiment settings
# Loaded from config but can be overridden below
CONFIG="${MDS}/code/experiments.json"

# Parse the settings
parse_settings "${CONFIG}"

# Connect to served vLLM
PORT=22659
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
NODE="icgpu06"
SERVER="http://${NODE}:${PORT}/v1"
OUTPUT_DIR_BASE="${MDS}/results/ICL"
N_EXAMPLES=-1
BATCH=32
SLEEP=3

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ];then
  SLURM_ARRAY_TASK_ID=11
fi

task_id=1
for test_dataset in "${DATASETS[@]}"; do
  for train_dataset in "${DATASETS[@]}"; do
    INPUT_TRAIN="${MDS}/data/${train_dataset}/train.jsonl"
    for split in "${SPLITS[@]}"; do
      #echo "On ${task_id}-${train_dataset}-${test_dataset}-${split}"
      #continue
      if [ "${task_id}" -eq "${SLURM_ARRAY_TASK_ID}" ]; then
        echo "On ${task_id}-${train_dataset}-${test_dataset}-${split}"
        INPUT_TEST="${MDS}/data/${test_dataset}/${split}.jsonl"

        # Single-shot
        OUTPUT_DIR="${OUTPUT_DIR_BASE}-${train_dataset}/${test_dataset}/${split}"
        mkdir -p "${OUTPUT_DIR}"

        python "${MDS}/code/icl_experiments.py" \
          --input-data-train "${INPUT_TRAIN}" \
          --input-data-test "${INPUT_TEST}" \
          --output-dir "${OUTPUT_DIR}" \
          --server-base "${SERVER}" \
          --model-path "${MODEL_NAME}" \
          --num-examples "${N_EXAMPLES}" \
          --batch-size "${BATCH}" \
          --resume \
          --sleep-time "${SLEEP}"

        if [ $? -ne 0 ];then
          echo "Error during inference of ${train_dataset}-${test_dataset}-${split}"
          exit 1
        fi

        # Zero-shot
        if [ "${train_dataset}" == "${test_dataset}" ]; then
          echo "Zero-shot for ${test_dataset}"
          OUTPUT_DIR="${OUTPUT_DIR_BASE}/${test_dataset}/${split}"
          mkdir -p "${OUTPUT_DIR}"

          python "${MDS}/code/icl_experiments.py" \
            --input-data-test "${INPUT_TEST}" \
            --output-dir "${OUTPUT_DIR}" \
            --server-base "${SERVER}" \
            --model-path "${MODEL_NAME}" \
            --num-examples "${N_EXAMPLES}" \
            --batch-size "${BATCH}" \
            --resume \
            --sleep-time "${SLEEP}"

          if [ $? -ne 0 ];then
            echo "Error during zero-shot inference of ${train_dataset}-${split}"
            exit 1
          fi
        fi
      fi
    # Increment task counter
    task_id=$(( task_id + 1))
    done
  done
done

echo "Done with task ${SLURM_ARRAY_TASK_ID}"
