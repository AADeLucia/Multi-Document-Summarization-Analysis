#!/bin/bash

# Experiment settings
CONFIG="${MDS}/code/experiments.json"

# Parse the settings
source "${MDS}/code/helper_functions.sh"
parse_settings "${CONFIG}"

# Group by (dataset, model) for the task array
task_id=1
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for split in "${SPLITS[@]}"; do
        status_out="${task_id}: ${model}-${dataset}-${split}"
        # Generated?
        OUTPUT_FILE="${MDS}/results/${model}/${dataset}/${split}/generated.jsonl"
        if [ -f "${OUTPUT_FILE}" ]; then
          status_out+=" Generated"
        else
          status_out+=" Not Generated"
        fi

        # Evaluated?
        OUTPUT_FILE="${MDS}/results/${model}/${dataset}/${split}/evaluation.json"
        if [ -f "${OUTPUT_FILE}" ]; then
          status_out+=" Evaluated"
        else
          status_out+=" Not Evaluated"
        fi
        echo "${status_out}"
    done
    # Increment task counter
    task_id=$(( task_id + 1))
  done
done

echo "Done"
