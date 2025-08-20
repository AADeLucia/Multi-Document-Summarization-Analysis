#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=convosumm
#SBATCH --partition=ica100,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH -A mdredze1_gpu
#SBATCH --output=logs/%x.%j.log

# Make sure the ConvoSumm version of fairseq is installed
module load anaconda
conda activate convosumm

DO_PREPROCESS=false
DO_TRAIN=true
DO_PREDICT=true

CONVOSUMM_REPO="${MDS}/dependencies/ConvoSumm"
CLPR="${MDS}/models/CLPR"
DATASET="multinews_plus"
DATA_DIR="${MDS}/data/${DATASET}"
MODEL_NAME="ConvoSumm-MultiNews_Plus"
MODEL_DIR="${MDS}/models/${MODEL_NAME}"
RESULTS_DIR="${MDS}/results/${MODEL_NAME}/${DATASET}"
WORKERS=1
TOTAL_NUM_UPDATES=30000
WARMUP_UPDATES=1000
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=1
VALIDATE_EVERY=1000
MIN_LENGTH=0
MAX_LENGTH=1024
TENSORBOARD_DIR="${MODEL_DIR}/tensorboard"
BART_PATH="${MDS}/models/bart.large"
RESTORE_PATH="${BART_PATH}/model.pt"
mkdir -p "${TENSORBOARD_DIR}"
mkdir -p "${RESULTS_DIR}"

export CUDA_VISIBLE_DEVICES=0

if [ "${DO_PREPROCESS}" = true ];then
  echo "Preprocessing"
  mkdir -p "${MODEL_DIR}/bin/${DATASET}"

  # Copy encoder.json and vocab.bpe
  if [ ! -f "${MODEL_DIR}/dict.txt" ]; then
    cp "${CONVOSUMM_REPO}/code/misc"/* "${MODEL_DIR}"
    # Download dict.txt
    wget 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt' -P "${MODEL_DIR}"
  fi

  for split in "train" "val" "test"; do
    echo "On split ${split}"

    python "${MDS}/code/convosumm_arg_classifier.py" \
      --model-path "${CLPR}" \
      --input-data "${DATA_DIR}/${split}.jsonl" \
      --output-dir "${MODEL_DIR}/bin/${DATASET}" \
      --output-file "${split}.jsonl" \
      --batch-size 1024 \
      --resume

    if [ $? -ne 0 ];then
      echo "Error"
      exit 1
    fi

    # Process for Fairseq
    for LANG in "source" "target"
    do
      echo "fairseq processing ${LANG}"

      python "${CONVOSUMM_REPO}/code/fairseq/examples/roberta/multiprocessing_bpe_encoder.py" \
      --encoder-json "${MODEL_DIR}/encoder.json" \
      --vocab-bpe "${MODEL_DIR}/vocab.bpe" \
      --inputs "${MODEL_DIR}/bin/${DATASET}/${split}.${LANG}" \
      --outputs "${MODEL_DIR}/bin/${DATASET}/${split}.bpe.${LANG}" \
      --workers "${WORKERS}" \
      --keep-empty;

      if [ $? -ne 0 ];then
        echo "Error on split ${split} for ${LANG}"
        exit 1
      fi

    done
  done

  echo "Fairseq preprocess"
  fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref "${MODEL_DIR}/bin/${DATASET}/train.bpe" \
    --validpref "${MODEL_DIR}/bin/${DATASET}/val.bpe" \
    --testpref "${MODEL_DIR}/bin/${DATASET}/test.bpe" \
    --destdir "${MODEL_DIR}/bin/${DATASET}" \
    --workers "${WORKERS}" \
    --srcdict "${MODEL_DIR}/dict.txt" \
    --tgtdict "${MODEL_DIR}/dict.txt";

  if [ $? -ne 0 ];then
    echo "Error on split ${split} for ${LANG}"
    exit 1
  fi
fi

if [ "${DO_TRAIN}" = true ];then
  echo "Training"
  # Uncomment when not resuming from checkpoint
  fairseq-train "${MODEL_DIR}/bin/${DATASET}" \
    --ddp-backend=no_c10d \
    --restore-file "${RESTORE_PATH}" \
    --max-tokens "${MAX_TOKENS}" \
    --task "translation" \
    --source-lang "source" \
    --target-lang "target" \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --arch "bart_large" \
    --criterion "label_smoothed_cross_entropy" \
    --label-smoothing 0.1 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-08 \
    --clip-norm 0.01 \
    --lr-scheduler "polynomial_decay" \
    --lr "${LR}" \
    --total-num-update "${TOTAL_NUM_UPDATES}" \
    --warmup-updates "${WARMUP_UPDATES}" \
    --update-freq "${UPDATE_FREQ}" \
    --max-update "${TOTAL_NUM_UPDATES}" \
    --skip-invalid-size-inputs-valid-test \
    --save-dir "${MODEL_DIR}/checkpoints"  \
    --tensorboard-logdir "${TENSORBOARD_DIR}" \
    --find-unused-parameters \
    --save-interval-updates "${VALIDATE_EVERY}" \
    --keep-interval-updates 1 \
    --max-source-positions "${MAX_TOKENS}" \
    --reset-dataloader --reset-optimizer --reset-meters

  if [ $? -ne 0 ];then
    echo "Error training"
    exit 1
  fi
fi

if [ "${DO_PREDICT}" = true ]; then
  # beam, length penalty, min length, max length, batch size, max source positions
  for split in "val" "test"; do
    mkdir -p "${RESULTS_DIR}/${split}"
    OUTPUT_FILE="${RESULTS_DIR}/${split}/${split}.hypo"

    python "${CONVOSUMM_REPO}/code/scripts/inference.py" \
     "${MODEL_DIR}/checkpoints" \
     "checkpoint_best.pt" \
     "${MODEL_DIR}/bin/${DATASET}" \
     "${MODEL_DIR}/bin/${DATASET}/${split}.source" \
     "${OUTPUT_FILE}" \
      1 1 "${MIN_LENGTH}" "${MAX_LENGTH}" 128 "${MAX_TOKENS}" \
      "${MODEL_DIR}/encoder.json" "${MODEL_DIR}/vocab.bpe"

    # Fix the output format
    python "${MDS}/code/parse_summn_output.py" \
      --input-file "${OUTPUT_FILE}" \
      --output-file "${RESULTS_DIR}/${split}/generated.jsonl"

  done
fi
