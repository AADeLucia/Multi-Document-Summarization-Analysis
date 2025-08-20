#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=led
#SBATCH --partition=ica100
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH -A mdredze1_gpu
#SBATCH --output=logs/%x.%j.log

# Make sure the ConvoSumm version of fairseq is installed
module load anaconda
conda activate led

DATASET="multixscience"
OUTPUT="${MDS}/models/LED-MultiXScience"
mkdir -p "${OUTPUT}"

python "${MDS}/dependencies/longformer/scripts/summarization.py" \
  --data_dir="${MDS}/data/${DATASET}" \
  --save_dir="${OUTPUT}" \
  --batch_size=16 \
  --model_path="allenai/led-large-16384" \
  --gpus=-1 \
  --max_input_len=16384 \
  --max_output_len=1024 \
  --epochs=5 \
  --grad_accum=1 \
  --lr=3e-5 \
  --grad_ckpt \
  --label_smoothing 0.1 \
  --warmup=200 \
  --val_every=0.5 \
  --val_percent_check=0.5 \
  --no_progress_bar \
  --seed 42 \
  --debug

# For running inference
#--test \
#--from_pretrained /gpfs/loomis/project/radev/af726/convosumm/Argument-Graph-Mining/longformer/summarization/longformer-encdec-large-1200_poly_attention_340_label_smoothing_nooverlap_512_said_summary_act_cnndm/_ckpt_epoch_52.ckpt \
