"""
Ground truth "model" for sanity checking.

Uses the target summary as the generated one.
"""
import logging
from argparse import ArgumentParser
import sys
import os
import jsonlines
import json
from tqdm import tqdm

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)


def generate_from_file(input_file, output_file):
    with jsonlines.open(input_file, "r") as reader:
        with jsonlines.open(output_file, "w") as writer:
            for entry in reader.iter():
                row = {
                    "decoded": entry["target"]
                }
                writer.write(row)


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str, default="generated.jsonl")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--experiments-config", default=f"{os.environ['MDS']}/code/experiments.json")
    parser.add_argument("--results-dir", default=f"{os.environ['MDS']}/results")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.run_all:
        with open(args.experiments_config, "r") as f:
            experiments_config = json.load(f)

        dataset_dir_base = f"{os.environ['MDS']}/data"
        output_dir_base = f"{args.results_dir}/ground_truth"
        for dataset in experiments_config["datasets"]:
            for split in experiments_config["splits"]:
                data_file = os.path.join(dataset_dir_base, dataset, f"{split}.jsonl")
                out_file = os.path.join(output_dir_base, dataset, split, "generated.jsonl")
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                generate_from_file(data_file, out_file)
    else:
        output_file = os.path.join(args.results_dir, args.output_file)
        generate_from_file(args.input_file, output_file)


if __name__ == "__main__":
    main()

