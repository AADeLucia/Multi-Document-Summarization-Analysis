#!/bin/bash
#SBATCH --job-name=icl
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH -A mdredze1_gpu
#SBATCH --output=logs/%x.%A.%a.log
#SBATCH --array=1-8  # Should be {# datasets} x {# datasets} x {# splits}

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
NODE="icgpu07"
SERVER="http://${NODE}:${PORT}/v1"
OUTPUT_DIR_BASE="${MDS}/results/ICL"
N_EXAMPLES=-1

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ];then
  SLURM_ARRAY_TASK_ID=3
fi

task_id=1
for test_dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
      echo "On ${task_id}-${test_dataset}-${split}"
      if [ "${task_id}" -eq "${SLURM_ARRAY_TASK_ID}" ]; then
#        echo "On ${task_id}-${test_dataset}-${split}"
        INPUT_TEST="${MDS}/data/${test_dataset}/${split}.jsonl"

        echo "Zero-shot for ${test_dataset}"
        OUTPUT_DIR="${OUTPUT_DIR_BASE}/${test_dataset}/${split}"
        mkdir -p "${OUTPUT_DIR}"

        python "${MDS}/code/icl_experiments.py" \
          --input-data-test "${INPUT_TEST}" \
          --output-dir "${OUTPUT_DIR}" \
          --server-base "${SERVER}" \
          --model-path "${MODEL_NAME}" \
          --num-examples "${N_EXAMPLES}" \
          --resume

        if [ $? -ne 0 ];then
          echo "Error during zero-shot inference of ${test_dataset}-${split}"
          exit 1
        fi
      fi
    # Increment task counter
    task_id=$(( task_id + 1))
    done
done

echo "Done with task ${SLURM_ARRAY_TASK_ID}"
