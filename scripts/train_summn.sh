#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=summn
#SBATCH --partition=ica100,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH -A mdredze80_gpu
#SBATCH --output=logs/%x.%j.log

module load anaconda
conda activate summn

DO_TRAIN=true
DO_PREDICT=true

export PYTHONPATH="${PYTHONPATH}:${MDS}/dependencies/AnyROUGE"
STAGES=3  # Sloppy, hard-coded from the MultiNews config file
DATASET="multinews_plus"
CONFIG="ConvoSumm"
RUN_NAME="MultiNews_Plus"
MODEL_NAME="SummN-${RUN_NAME}"
PROJECT_PATH="${MDS}/dependencies/Summ-N"
BART_PATH="${PROJECT_PATH}/bart.large.cnn"
OUTPUT_DIR="${MDS}/results/${MODEL_NAME}/${DATASET}"
TRAINED_MODEL_DIR="${MDS}/models/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TRAINED_MODEL_DIR}"

# Train
TOTAL_NUM_UPDATES=30000
WARMUP_UPDATES=1000
LR=3e-5
MAX_TOKENS=2048
UPDATE_FREQ=4
if [ "${DO_TRAIN}" = true ];then
  echo "Training ${MODEL_NAME}"
  python "${PROJECT_PATH}/run.py" \
    --cfg "${CONFIG}.cfg" \
    --mode "train" \
    --dataset-path "${MDS}/data/${DATASET}" \
    --output-path "${TRAINED_MODEL_DIR}" \
    --checkpoint-dir "${TRAINED_MODEL_DIR}" \
    --save-intermediate \
    --cuda-devices 0 \
    --model-path "${BART_PATH}" \
    --workers 1 \
    --generate-batch-size 128 \
    --total-number-update "${TOTAL_NUM_UPDATES}" \
    --warmup-update "${WARMUP_UPDATES}" \
    --learning-rate "${LR}" \
    --max-tokens-batch "${MAX_TOKENS}" \
    --update-freq "${UPDATE_FREQ}"

  if [ $? -ne 0 ];then
    echo "Error running with training"
    exit 1
  fi

fi

if [ "${DO_PREDICT}" = true ];then
  echo "Inference with ${MODEL_NAME}"
  python "${PROJECT_PATH}/run.py" \
    --cfg "${CONFIG}.cfg" \
    --mode "test" \
    --dataset-path "${MDS}/data/${DATASET}" \
    --output-path "${OUTPUT_DIR}" \
    --checkpoint-dir "${TRAINED_MODEL_DIR}" \
    --save-intermediate \
    --cuda-devices 0 \
    --model-path "${BART_PATH}" \
    --workers 1 \
    --generate-batch-size 128

  if [ $? -ne 0 ];then
    echo "Error running inference"
    exit 1
  fi

  for split in "val" "test" ; do
    RESULTS_DIR="${MDS}/results/${MODEL_NAME}/${DATASET}/${split}"
    mkdir -p "${RESULTS_DIR}"

    # Fix the output format
    python "${MDS}/code/parse_summn_output.py" \
      --input-file "${OUTPUT_DIR}/stage_${STAGES}/${split}.hypo" \
      --output-file "${OUTPUT_DIR}/${split}/generated.jsonl"
  done

fi

