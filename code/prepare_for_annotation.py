"""

"""
import logging
from argparse import ArgumentParser
import sys
import os
import jsonlines
import csv
import random
import pandas as pd
import numpy as np
sys.path.append(f"{os.environ['MDS']}/code")
from project_utils import load_generations, load_config


# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)


def sample_document_ids(input_data, n, seed):
    # Not efficient but oh well
    all_ids = []
    with jsonlines.open(input_data) as reader:
        for doc in reader:
            all_ids.append(doc["id"])

    if n == -1:
        sampled_ids = all_ids
    else:
        random.seed(seed)
        sampled_ids = random.choices(all_ids, k=n)
    return set(sampled_ids)


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--experiments-config", default=f"{os.environ['MDS']}/code/experiments_abridged.json")
    parser.add_argument("--n-examples", type=int, default=10, help="Number of samples from each dataset. Set to -1 for all samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load config
    experiments_config = load_config(args.experiments_config)
    output_dir = os.path.join(
        os.environ['MDS'],
        "results",
        os.path.basename(args.experiments_config).split('.')[0]
    )
    os.makedirs(output_dir, exist_ok=True)

    # Group by dataset
    for dataset in experiments_config["datasets"]:
        for split in experiments_config["splits"]:
            input_data = f"{os.environ['MDS']}/data/{dataset}/{split}.jsonl"
            sampled_ids = sample_document_ids(input_data, args.n_examples, args.seed)
            results = {}
            for model in experiments_config["models"]:
                logger.info(f"Processing dataset {dataset} split {split} and model {model}")
                output_data = experiments_config["format"].replace("{model}", model).replace("{dataset}", dataset).replace("{split}", split)
                output_data = f"{os.environ['MDS']}/results/{output_data}/generated.jsonl"
                if not os.path.exists(output_data):
                    logger.warning(f"No results for {model} {dataset} {split}. Skipping.")
                    continue

                try:
                    for example in load_generations(input_data, output_data):
                        if example["id"] not in sampled_ids:
                            continue

                        if example["id"] not in results:
                            results[example["id"]] = {
                                "source": "<s>".join(example["source"]),
                                "target": example["target"]
                            }
                        results[example["id"]][model] = example["generated"]
                except Exception as err:
                    logger.warning(f"{err} encountered on {dataset}-{split}-{model}. Skipping.")
                    continue

            df = pd.DataFrame.from_dict(results, orient="index")
            if "/" in dataset:
                dataset_name = dataset.replace("/", "-")
            else:
                dataset_name = dataset
            output_file = os.path.join(output_dir, f"annotate_{dataset_name}_{split}.csv")
            df.to_csv(output_file)
            logger.info(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
