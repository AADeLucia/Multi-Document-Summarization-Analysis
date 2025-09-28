#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=mds-demo
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH -A mdredze1_gpu
#SBATCH --output=%x.%j.log

conda activate mds

export CUDA_VISIBLE_DEVICES=0
DO_PREDICT=true
DO_EVAL=true
SPLIT="demo"
MODELS=(
  "allenai/PRIMERA-multinews"
)
DATASETS=(
  "convosumm/reddit"
)

for model in "${MODELS[@]}"; do
  echo "On model ${model}"
  for dataset in "${DATASETS[@]}"; do
    echo "On dataset ${dataset}"
    INPUT="${MDS}/data/${dataset}/${SPLIT}.jsonl"
    OUTPUT_DIR="${MDS}/results/${model}/${dataset}/${SPLIT}"
    mkdir -p "${OUTPUT_DIR}"

    if [ "${DO_PREDICT}" = true ];then
      python "${MDS}/code/primera_test.py" \
        --input-data "${INPUT}" \
        --model-path "${model}" \
        --output-dir "${OUTPUT_DIR}"

      python "${MDS}/code/prepare_for_annotation.py" \
        --input-data "${INPUT}" \
        --output-data "${OUTPUT_DIR}/generated.jsonl" \
        --output-dir "${OUTPUT_DIR}"
    fi

    if [ "${DO_EVAL}" = true ];then
      python "${MDS}/code/evaluate_generations.py" \
        --input-data "${INPUT}" \
        --output-data "${OUTPUT_DIR}/generated.jsonl" \
        --output-dir "${OUTPUT_DIR}"

#      TMP="${MDS}/dependencies/multi_mds/tmp"
#      rm -r "${TMP}"
#
#      python "${MDS}/code/calculate_dispersion_aac.py" \
#        --input-data "${INPUT}" \
#        --output-data "${OUTPUT_DIR}/generated.jsonl" \
#        --output-dir "${OUTPUT_DIR}" \
#        --cpu
#
#      mkdir -p "${OUTPUT_DIR}/dispersion"
#      mv "${TMP}"/* "${OUTPUT_DIR}/dispersion"

    fi

    if [ $? -ne 0 ];then
      echo "Error"
      exit 1
    fi

  done
done

echo "Done"