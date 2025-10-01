#!/bin/bash
# Functions for the main scripts
# Need to be sourced (e.g. source helper_functions.sh)

parse_settings() {
  local config
  local str
  config="$1"
  
  str="$(jq -r '.datasets | @sh' "${config}")"
  declare -ga DATASETS="($str)"

  str="$(jq -r '.models | @sh' "${config}")"
  declare -ga MODELS="($str)"

  str="$(jq -r '.splits | @sh' "${config}")"
  declare -ga SPLITS="($str)"

  str="$(jq -r '.metrics | @sh' "${config}")"
  declare -ga METRICS="($str)"
}

parse_settings_from_json() {
  local json_str="$1"
  local str

  str="$(jq -n --argjson data "$json_str" '$data.DATASET | @sh ' | sed "s/'//g")"
  declare -ga DATASET="($str)"

  str="$(jq -n --argjson data "$json_str" '$data.MODEL | @sh' | sed "s/'//g")"
  declare -ga MODEL="($str)"

#  str="$(jq -n --argjson data "$json_str" '$data.SPLITS | @sh' | sed "s/'//g")"
#  declare -ga SPLITS="($str)"
#
  str="$(jq -n --argjson data "$json_str" '$data.SPLITS | @sh' | sed "s/[\"']//g")"
  read -a SPLITS <<< "$str"

  str="$(jq -n --argjson data "$json_str" '$data.METRICS | @sh' | sed "s/[\"']//g")"
  read -a METRICS <<< "$str"
}


infer_model() {
  local model="$1"
  local input="$2"
  local output="$3"
  local dataset="$4"
  MODEL_DIR="${MDS}/models/${model}"

  # Skip if file exists
  if [ -f "${output}/generated.jsonl" ]; then
     echo "Generations exist in ${output}. Skipping."
     return
  fi

  # Check split (needed for some code)
  if [[ ${input} =~ val.jsonl ]]; then
    split="val"
  else
    split="test"
  fi

  # Case: PRIMERA model
  if [[ ${model} =~ "PRIMERA" ]]; then
    conda activate primer
    if [[ ${model} =~ "allenai" ]]; then
      model_path="${model}"
    else
      model_path="${MODEL_DIR}"
    fi

    # Remove previous, incomplete generations
    rm -rf "${output}/raw_generations"

    python "${MDS}/dependencies/PRIMER/script/primer_hf_main.py" \
    --gpus 1 \
    --mode "test" \
    --model_path "${model_path}"  \
    --output-dir "${output}" \
    --test-split "${split}" \
    --primer_path "allenai/PRIMERA" \
    --data_path "${MDS}/data" \
    --dataset_name "${dataset}" \
    --batch_size 32 \
    --num_workers 0 \
    --progress_bar_refresh_rate 1 \
    --rand_seed 42 \
    --min_length_tgt 0 \
    --max_length_tgt 1024 \
    --top_p 0.9 \
    --do_sample \
    --applyTriblck

    if [ $? -ne 0 ];then
      echo "Error during inference"
      return 1
    fi

    conda deactivate

    python "${MDS}/code/parse_primera_output.py" \
      --input-dir "${output}/raw_generations" \
      --output-file "${output}/generated.jsonl"

    echo "Parsed generations saved to ${output}/generated.jsonl"
  fi

  # Case: Ground truth
  if [[ "${model}" == "ground_truth" ]]; then
    python "${MDS}/code/model_oracle.py" --results-dir "${output}" --input-file "${input}"
  fi

  # Case: Summ^N
  if [[ "${model}" =~ "SummN" ]]; then
    conda activate summn
    export PYTHONPATH="${PYTHONPATH}:${MDS}/dependencies/AnyROUGE"

    # Use ConvoSumm.cfg for all. Minimal decoding settings.
    python "${MDS}/dependencies/Summ-N/run.py" \
    --cfg "ConvoSumm.cfg" \
    --mode "test" \
    --dataset-path "${MDS}/data/${dataset}" \
    --output-path "${output}" \
    --checkpoint-dir "${MODEL_DIR}" \
    --save-intermediate \
    --cuda-devices "${CUDA_VISIBLE_DEVICES}" \
    --model-path "${MDS}/dependencies/Summ-N/bart.large.cnn" \
    --workers 1 \
    --generate-batch-size 128

    if [ $? -ne 0 ];then
      echo "Error during inference"
      return 1
    fi

    conda deactivate

    # Fix the output format
    python "${MDS}/code/parse_summn_output.py" \
      --input-file "${output}/stage_3/${split}.hypo" \
      --output-file "${output}/generated.jsonl"
  fi

  # Case: ConvoSumm
  if [[ "${model}" == "ConvoSumm-"* ]]; then
    conda activate convosumm

    CONVOSUMM_REPO="${MDS}/dependencies/ConvoSumm"
    mkdir -p "${MODEL_DIR}/bin/${dataset}"

    # 1. Extract arguments
    python "${MDS}/code/convosumm_arg_classifier.py" \
      --model-path "${MDS}/models/CLPR" \
      --input-data "${input}" \
      --output-dir "${MODEL_DIR}/bin/${dataset}" \
      --output-file "${split}.jsonl" \
      --batch-size 512

    if [ $? -ne 0 ];then
      echo "Error during argument extraction"
      return 1
    fi

    # 2. Preprocess for Fairseq
    for LANG in "source" "target"
    do
      echo "fairseq processing ${LANG}"
      python "${CONVOSUMM_REPO}/code/fairseq/examples/roberta/multiprocessing_bpe_encoder.py" \
      --encoder-json "${MODEL_DIR}/encoder.json" \
      --vocab-bpe "${MODEL_DIR}/vocab.bpe" \
      --inputs "${MODEL_DIR}/bin/${dataset}/${split}.${LANG}" \
      --outputs "${MODEL_DIR}/bin/${dataset}/${split}.bpe.${LANG}" \
      --workers 1 \
      --keep-empty;
    done

    if [[ $split == "val" ]]; then
      fairseq-preprocess \
        --source-lang "source" \
        --target-lang "target" \
        --validpref "${MODEL_DIR}/bin/${dataset}/val.bpe" \
        --destdir "${MODEL_DIR}/bin/${dataset}" \
        --workers 1 \
        --srcdict "${MODEL_DIR}/dict.txt" \
        --tgtdict "${MODEL_DIR}/dict.txt";
    else
      fairseq-preprocess \
        --source-lang "source" \
        --target-lang "target" \
        --testpref "${MODEL_DIR}/bin/${dataset}/test.bpe" \
        --destdir "${MODEL_DIR}/bin/${dataset}" \
        --workers 1 \
        --srcdict "${MODEL_DIR}/dict.txt" \
        --tgtdict "${MODEL_DIR}/dict.txt";
    fi

    # 3. Infer
    # beam, length penalty, min length, max length, batch size, max source positions
    OUTPUT_FILE="${output}/${split}.hypo"
    python "${CONVOSUMM_REPO}/code/scripts/inference.py" \
     "${MODEL_DIR}/checkpoints" \
     "checkpoint_best.pt" \
     "${MODEL_DIR}/bin/${dataset}" \
     "${MODEL_DIR}/bin/${dataset}/${split}.source" \
     "${OUTPUT_FILE}" \
     1 1 0 1024 3 0.9 128 2048 \
     "${MODEL_DIR}/encoder.json" "${MODEL_DIR}/vocab.bpe"

    # Fix the output format
    python "${MDS}/code/parse_summn_output.py" \
      --input-file "${OUTPUT_FILE}" \
      --output-file "${output}/generated.jsonl"

    # Copy the argument extraction output for analysis
    cp "${MODEL_DIR}/bin/${dataset}/${split}.jsonl" "${output}/extracted_arguments.jsonl"

    conda deactivate
  fi

  # Case: ICL
  if [[ "${model}" == "ICL"* ]]; then
    BATCH=8
    # Subcase: zero-shot
    if [[ "${model}" == "ICL"* ]]; then
      python "${MDS}/code/icl_experiments.py" \
        "test" \
        --input-data-test "${input}" \
        --output-dir "${output}" \
        --batch-size "${BATCH}"
    else
      python "${MDS}/code/icl_experiments.py" \
        --model-path "${MODEL_DIR}" \
        "test" \
        --input-data-test "${input}" \
        --output-dir "${output}" \
        --batch-size "${BATCH}"
    fi
  fi

}

eval_generations() {
  # Handle accepting an array
  # https://askubuntu.com/a/995110
  local input="$1"
  local output="$2"
  local overwrite="$3"
  shift 3
  local metrics=("$@")

  conda activate mds

  local overwrite_flag=""
  if [ "${overwrite}" = "true" ]; then
    overwrite_flag="--overwrite"
  fi

  python "${MDS}/code/evaluate_generations.py" \
    --input-data "${input}" \
    --output-data "${output}/generated.jsonl" \
    --output-dir "${output}" \
    --metrics "${metrics[@]}" \
    ${overwrite_flag}
}