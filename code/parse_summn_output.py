"""
Transform fairseq newline-delimited output to output expected by load_generations
"""
import logging
from argparse import ArgumentParser
import sys
import os
import jsonlines
from tqdm import tqdm

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    with open(args.input_file, "r") as f:
        generations = f.readlines()

    logger.info(f"Read {len(generations)} from {args.input_file}")

    with jsonlines.open(args.output_file, "w") as writer:
        for generation in generations:
            row = {
                "decoded": generation
            }
            writer.write(row)


if __name__ == "__main__":
    main()

