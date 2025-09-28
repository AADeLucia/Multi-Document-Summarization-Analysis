#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=train-cl
#SBATCH --mem=60GB
#SBATCH -A mdredze1
#SBATCH --output=logs/%x.%A.%a.log

module load anaconda
conda activate mds
source "${MDS}/scripts/helper_functions.sh"

CONFIG="${MDS}/code/experiments_mayarr.json"

# Parse the settings
parse_settings "${CONFIG}"

for dataset in "${DATASETS[@]}"; do
	training_data="${MDS}/data/${dataset}/train.jsonl"
	model_path="${MDS}/models/ICL-${dataset}"
	mkdir -p "${model_path}"

	python "${MDS}/code/icl_experiments.py" \
		--model-path "${model_path}" \
		"train" \
		--few-shot-k 2 \
		--input-data-train "${training_data}"

done
