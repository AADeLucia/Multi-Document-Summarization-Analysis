"""

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
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    generations_dict = {}
    for filename in os.listdir(args.input_dir):
        idx = int(filename.split(".txt")[0])
        filename = os.path.join(args.input_dir, filename)
        with open(filename, "r") as f:
            gen = f.read().strip()
        generations_dict[idx] = gen

    # Sort by idx
    sorted_idx = sorted(generations_dict)

    with jsonlines.open(args.output_file, "w") as writer:
        for idx in sorted_idx:
            row = {
                "decoded": generations_dict[idx]
            }
            writer.write(row)


if __name__ == "__main__":
    main()

