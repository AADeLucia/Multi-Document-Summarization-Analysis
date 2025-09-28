"""
For each model and dataset, calculate how many tokens are omitted from the context window.

Follows the experiments in --experiments-config
"""
import os
import sys
import logging
from argparse import ArgumentParser

from transformers import AutoTokenizer
import pandas as pd
import jsonlines
from tqdm import tqdm

sys.path.append(f"{os.environ['MDS']}/code")
sys.path.append(f"{os.environ['MDS']}/dependencies/PRIMER/script")
from project_utils import load_config
from dataloader import SummarizationDataset as PrimerDataset

# from fairseq.data.encoders.gpt2_bpe import get_encoder

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)

CONVOSUMM_MAX_LENGTH = 2048

#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--experiments-config", default=f"{os.environ['MDS']}/code/experiments.json")
    parser.add_argument("--results-dir", default=f"{os.environ['MDS']}/results")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def get_convosumm_tokenizer(dataset, split):
    # bpe = get_encoder(
    #     f"{os.environ['MDS']}/dependencies/ConvoSumm/code/misc/encoder.json",
    #     f"{os.environ['MDS']}/dependencies/ConvoSumm/code/misc/vocab.bpe",
    # )
    bpe_path = f"{os.environ['MDS']}/models/ConvoSumm-ConvoSumm/bin/{dataset}/{split}.bpe.source"
    with open(bpe_path, "r") as f:
        for line in f.readlines():
            untruncated_ids = line.strip().split()
            # Should this be 2044 since the BPE file does not pad the sequences?
            is_truncated = len(untruncated_ids) > CONVOSUMM_MAX_LENGTH
            num_truncated = max(0, len(untruncated_ids) - CONVOSUMM_MAX_LENGTH)
            yield is_truncated, num_truncated, len(untruncated_ids)


def get_primera_tokenizer(input_data, dataset):
    primer_tokenizer = AutoTokenizer.from_pretrained("allenai/PRIMERA")
    primera_loader = PrimerDataset(
        hf_dataset=input_data,
        dataset_name=dataset,
        join_method="concat_start_wdoc_global",
        tokenizer=primer_tokenizer,
        max_input_len=4096,
        max_output_len=1024,
        mask_num=0,
        num_data=-1,
        rand_seed=42,
        is_test=True,
        dataset_type="val",
    )
    # Calculate for PRIMERA
    for doc_source, doc_tokenized in zip(input_data, primera_loader):
        truncated_ids, _, _ = doc_tokenized
        # Re-tokenize
        untruncated_ids = primer_tokenizer.encode("<doc-sep>".join(doc_source["docs"]), return_tensors="pt")[0]
        is_truncated = len(untruncated_ids) > len(truncated_ids)
        num_truncated = max(0, len(untruncated_ids) - len(truncated_ids))
        yield is_truncated, num_truncated, len(untruncated_ids)


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load config
    experiments_config = load_config(args.experiments_config)
    output_dir = os.path.join(
        args.results_dir,
        os.path.basename(args.experiments_config).split('.')[0]
    )
    os.makedirs(output_dir, exist_ok=True)

    results = []
    # For each dataset
    for dataset in experiments_config["datasets"]:
        # For each split
        for split in experiments_config["splits"]:
            # Load the data
            input_data_file = os.path.join(os.environ["MDS"], "data", dataset, f"{split}.jsonl")
            input_data = []
            with jsonlines.open(input_data_file) as reader:
                input_data.extend([doc for doc in reader.iter()])

            primera_processed = get_primera_tokenizer(input_data, dataset)
            convosumm_processed = get_convosumm_tokenizer(dataset, split)

            for doc_source, doc_primer, doc_convosumm in tqdm(zip(input_data, primera_processed, convosumm_processed), ncols=0, desc=f"{dataset} - {split}"):
                row = {
                    "dataset": dataset,
                    "split": split,
                    "id": doc_source["id"],
                    "model": "PRIMERA",
                    "truncated": doc_primer[0],
                    "truncated_tokens": doc_primer[1],
                    "source_length": doc_primer[2]
                }
                results.append(row)

                row = {
                    "dataset": dataset,
                    "split": split,
                    "id": doc_source["id"],
                    "model": "ConvoSumm",
                    "truncated": doc_convosumm[0],
                    "truncated_tokens": doc_convosumm[1],
                    "source_length": doc_convosumm[2]
                }
                results.append(row)

    output_path = os.path.join(output_dir, "truncation_stats.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    summary = results_df.groupby(["model", "dataset", "split"]).agg({"truncated": "sum"})
    logger.info(summary)


if __name__ == "__main__":
    main()


