#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

CONFIG="${MDS}/code/experiments.json"
RESULTS_BASE="${MDS}/results"

models=($(jq -r '.models[]' "$CONFIG"))
datasets=($(jq -r '.datasets[]' "$CONFIG"))
splits=($(jq -r '.splits[]' "$CONFIG"))

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
      eval_path="${RESULTS_BASE}/${model}/${dataset}/${split}/evaluation.json"
      old_path="${eval_path}.old"
      if [ -f "$eval_path" ]; then
        mv "$eval_path" "$old_path"
        echo "Moved $eval_path to $old_path"
      fi
    done
  done
done
