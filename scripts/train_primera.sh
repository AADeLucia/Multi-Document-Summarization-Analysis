#!/bin/bash
# Modeled after https://github.com/allenai/PRIMER/blob/main/run_bash/test_primer_hf.sh
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=primer
#SBATCH --partition=ica100,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -A mdredze1_gpu
#SBATCH --output=logs/%x.%j.log

module load anaconda
conda activate primer

export CUDA_VISIBLE_DEVICES=0
DATASET="multinews_plus"
PRIMERA_REPO="${MDS}/dependencies/PRIMER"
PRETRAINED_MODEL="allenai/PRIMERA"
MODEL_NAME="PRIMERA-MultiNews_Plus"
OUTPUT_DIR="${MDS}/models/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

METRICS=("rouge" "moverscore" "factscore-md" "factscore" "unieval")
MIN_LENGTH=0
MAX_LENGTH=1024
WORKERS=0

DO_TRAIN=true
DO_PREDICT=true
DO_EVAL=false
DO_FACTSCORE=false

if [ "${DO_TRAIN}" = true ];then
  python "${PRIMERA_REPO}/script/primer_hf_main.py" \
    --gpus 1 \
    --mode "train" \
    --model_name "${MODEL_NAME}" \
    --model_path "${OUTPUT_DIR}"  \
    --primer_path "${PRETRAINED_MODEL}" \
    --data_path "${MDS}/data" \
    --dataset_name "${DATASET}" \
    --saveTopK 3 \
    --lr 3e-5 \
    --label_smoothing 0.1 \
    --warmup_steps 1000 \
    --total_steps 30000 \
    --batch_size 16 \
    --num_workers "${WORKERS}" \
    --progress_bar_refresh_rate 100 \
    --rand_seed 42

  if [ $? -ne 0 ];then
    echo "Error during training"
    exit 1
  fi

fi


if [ "${DO_PREDICT}" = true ];then
  python "${PRIMERA_REPO}/script/primer_hf_main.py" \
    --gpus 1 \
    --mode "test" \
    --model_name "primer-convosumm" \
    --resume_ckpt "${OUTPUT_DIR}/checkpoints/step=714-vloss=4.65-avgr=0.2387.ckpt" \
    --model_path "${OUTPUT_DIR}"  \
    --primer_path "${PRETRAINED_MODEL}" \
    --data_path "${MDS}/data" \
    --dataset_name "${DATASET}" \
    --saveTopK 3 \
    --lr 3e-5 \
    --label_smoothing 0.1 \
    --warmup_steps 200 \
    --total_steps 5000 \
    --batch_size 16 \
    --num_workers "${WORKERS}" \
    --progress_bar_refresh_rate 1 \
    --rand_seed 42 \
    --min_length_tgt "${MIN_LENGTH}" \
    --max_length_tgt "${MAX_LENGTH}"

  for split in "val" "test"; do
    RESULTS_DIR="${MDS}/results/${MODEL_NAME}/${DATASET}/${split}"
    mkdir -p "${RESULTS_DIR}"

    python "${MDS}/code/parse_primera_output.py" \
      --input-dir "${OUTPUT_DIR}/${DATASET}/${split}/generated" \
      --output-file "${RESULTS_DIR}/generated.jsonl"
  done

  if [ $? -ne 0 ];then
    echo "Error during testing"
    exit 1
  fi
fi


# Evaluate
if [ "${DO_EVAL}" = true ];then
    conda activate mds

    for split in "val" "test"; do
      INPUT="${MDS}/data/${DATASET}/${split}.jsonl"
      RESULTS_DIR="${MDS}/results/${MODEL_NAME}/${DATASET}/${split}"

      python "${MDS}/code/prepare_for_annotation.py" \
        --input-data "${INPUT}" \
        --output-data "${RESULTS_DIR}/generated.jsonl" \
        --output-dir "${RESULTS_DIR}"

      # Run once and then re-run after FactScore
      python "${MDS}/code/evaluate_generations.py" \
        --input-data "${INPUT}" \
        --output-data "${RESULTS_DIR}/generated.jsonl" \
        --output-dir "${RESULTS_DIR}" \
        --metrics "${METRICS[@]}"

      if [ "${DO_FACTSCORE}" = true ];then
        python "${MDS}/code/calculate_factscore.py" \
          --input-data "${INPUT}" \
          --output-data "${RESULTS_DIR}/generated.jsonl" \
          --output-dir "${RESULTS_DIR}" \
          --dataset "${DATASET}" \
          --mode "single" \
          --model-name "retrieval+llama+npm"

        python "${MDS}/code/calculate_factscore.py" \
          --input-data "${INPUT}" \
          --output-data "${RESULTS_DIR}/generated.jsonl" \
          --output-dir "${RESULTS_DIR}" \
          --dataset "${DATASET}" \
          --mode "multi" \
          --model-name "retrieval+llama"

        python "${MDS}/code/evaluate_generations.py" \
        --input-data "${INPUT}" \
        --output-data "${RESULTS_DIR}/generated.jsonl" \
        --output-dir "${RESULTS_DIR}" \
        --metrics "${METRICS[@]}"
      fi
    done
fi

echo "Done"