#!/bin/bash
# Download datasets for training and evaluation
# Usage: ./download_datasets.sh

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Initialize conda environment
module load anaconda
conda activate mds

# Create data directory if it doesn't exist
DATA_DIR="${MDS}/data"
mkdir -p "${DATA_DIR}"
echo "Data directory: ${DATA_DIR}"

# Download datasets to $DATA_DIR
echo "Downloading ConvoSumm dataset..."
mkdir -p "${DATA_DIR}/convosumm/reddit"
gdown --folder https://drive.google.com/drive/folders/1JYrj7lvNZmp3xjVSKJvigDEn_AzVOx3e -O "${DATA_DIR}/convosumm/data-non-processed"

echo "Downloading MultiNews+ dataset..."
mkdir -p "${DATA_DIR}/multinews_plus"
git clone https://github.com/c-juhwan/multi_news_plus.git "${DATA_DIR}/multinews_plus/repo"

echo "Downloading MultiXScience dataset..."
mkdir -p "${DATA_DIR}/multixscience"
git clone git@github.com:yaolu/Multi-XScience.git "${DATA_DIR}/multixscience/repo"

# Process the datasets
for dataset in "convosumm" "multinews_plus" "multixscience"; do
  echo "Processing dataset: ${dataset}"
  python "${MDS}/code/process_data.py" --dataset "${dataset}"
done

echo "Datasets downloaded and processed successfully."
