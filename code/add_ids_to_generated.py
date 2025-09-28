"""Add IDs of examples to generated.jsonl output

"""
import jsonlines
from tqdm import tqdm

from argparse import ArgumentParser
import sys
import os
from typing import List, Dict
import logging

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
sys.path.append(f"{os.environ['MDS']}/code")
from project_utils import load_generations
from project_utils import load_config


def load_jsonlines(file_path: str) -> List[Dict]:
    """Convenience for loading a file in JSON lines format"""
    with jsonlines.open(file_path) as reader:
        out = [r for r in reader.iter()]
    return out


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--experiments-config", default=f"{os.environ['MDS']}/code/experiments.json")
    parser.add_argument("--results-dir", default=f"{os.environ['MDS']}/results")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load config
    experiments_config = load_config(args.experiments_config)

    # For each dataset
    for dataset in experiments_config["datasets"]:
        # For each split
        for split in experiments_config["splits"]:
            # Load ground-truth
            ground_truth_path = os.path.join(
                os.environ["MDS"], "data", dataset, f"{split}.jsonl"
            )
            ground_truth = load_jsonlines(ground_truth_path)
            # For each model
            for model in experiments_config["models"]:
                model_output_path = os.path.join(
                    args.results_dir, model, dataset, split, "generated.jsonl"
                )
                if not os.path.exists(model_output_path):
                    logger.warning(f"{model_output_path} does not exist. Skipping.")
                    continue
                generated = load_jsonlines(model_output_path)
                new_output = os.path.join(
                    args.results_dir, model, dataset, split, "generated_with_ids.jsonl"
                )
                if len(ground_truth) != len(generated):
                    logger.error(
                        f"Mismatch in number of examples between source ({ground_truth_path}) "
                        f"({len(ground_truth)}) and predicted ({model_output_path}) ({len(generated)})"
                    )
                    continue
                with jsonlines.open(new_output, "w") as writer:
                    for source, generated in zip(ground_truth, generated):
                        generated.update({
                            "id": source["id"],
                        })
                        writer.write(generated)
                logger.info(f"Done with {dataset}-{split}-{model}")


if __name__ == "__main__":
    main()

