"""
This script is a wrapper for the Multi MDS codebase https://github.com/ariecattan/multi_mds/tree/main

Handles the preprocessing and parsing of the score output
"""
import logging
from argparse import ArgumentParser
import sys
import os
import jsonlines
import subprocess
sys.path.append(f"{os.environ['MDS']}/code")
from project_utils import load_generations

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)


class DispersionScorer:
    def __init__(self, path_to_mds, tmp):
        self.path_to_mds = path_to_mds
        if tmp:
            self.output_dir = tmp
        else:
            self.output_dir = f"{self.path_to_mds}/tmp"
        os.makedirs(self.output_dir, exist_ok=True)
        self.raw_data = f"{self.output_dir}/for_eval.jsonl"

    def compute(self, dataset, device):
        logger.info(f"Preparing dataset")
        self.prepare_dataset(dataset)

        # 1. Extract all Open IE tuples from the summary and the source documents.
        logger.info(f"Extracting OpenIE tuples")
        args = [
            "python", f"{self.path_to_mds}/extract_open_ie.py",
            "--raw_data", self.raw_data,
            "--data_dir", self.output_dir
        ]
        # Corresponds to https://docs.allennlp.org/v2.10.1/api/predictors/predictor/#from_path
        if device == "cpu":
            args.extend(["--gpu", "-1"])
        else:
            args.extend(["--gpu", "0"])
        check = subprocess.run(args)
        if check.returncode != 0:
            logger.error(f"Command failed:\n\t{' '.join(check.args)}")
            exit(1)

        # 2. Prepare pairs
        # Output is $data_dir/pairs.pickle
        logger.info(f"Preparing OpenIE pairs")
        output_file = os.path.join(self.output_dir, "pairs.pickle")
        if os.path.exists(output_file):
            logger.info(f"Pairs already exist at {output_file}. Skipping step.")
        else:
            args = [
                "python", f"{self.path_to_mds}/prepare_oie_pairs.py",
                "--data_dir", self.output_dir
            ]
            check = subprocess.run(args)
            if check.returncode != 0:
                logger.error(f"Command failed:\n\t{' '.join(check.args)}")
                exit(1)

        # 3. Compute alignment scores between source and summary propositions for each topic
        # Output is $data_dir/result_npy/<node ID>_scores.npy
        logger.info(f"Aligning propositions")
        out_dir = os.path.join(self.output_dir, "result_npy")
        output_files = os.listdir(out_dir) if os.path.exists(out_dir) else []
        if len(output_files) >= 5:
            logger.info(f"Alignments already exist at {output_file}. Skipping step.")
        else:
            args = [
                "python", f"{self.path_to_mds}/get_superpal_scores.py",
                "--data_dir", self.output_dir,
                "--model", "biu-nlp/superpal",
                "--device_ids", "-1" if device == "cpu" else "0",
                "--batch_size", "1" if device == "cpu" else "128"
            ]
            check = subprocess.run(args)
            if check.returncode != 0:
                logger.error(f"Command failed:\n\t{' '.join(check.args)}")
                exit(1)

        # 4. Build greedy subsets of documents that maximize coverage
        logger.info(f"Building subsets")
        args = [
            "python", f"{self.path_to_mds}/build_greedy_subsets.py",
            "--data_dir", self.output_dir
        ]
        check = subprocess.run(args)
        if check.returncode != 0:
            logger.error(f"Command failed:\n\t{' '.join(check.args)}")
            exit(1)

        # 5. Compute AAC score and save plot in $data_dir/plot.png
        logger.info(f"Computing AAC")
        args = [
            "python", f"{self.path_to_mds}/get_aac_scores.py",
            "--data_dir", self.output_dir
        ]
        check = subprocess.run(args)
        if check.returncode != 0:
            logger.error(f"Command failed:\n\t{' '.join(check.args)}")
            exit(1)

    def prepare_dataset(self, dataset):
        # Write the dataset in the expect format
        with jsonlines.open(self.raw_data, "w") as writer:
            for i, (s, p) in enumerate(zip(dataset["source"], dataset["predictions"])):
                doc = {
                    "topic_id": i,
                    "documents": s,
                    "summary": p
                }
                writer.write(doc)


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path-to-mds", type=str, default=f"{os.environ['MDS']}/dependencies/multi_mds")
    parser.add_argument("--input-data", required=True, help="")
    parser.add_argument("--output-data", required=True, help="")
    parser.add_argument("--output-dir", required=True, help="")
    parser.add_argument("--tmp-dir", help="Default is path-to-mds/tmp")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only even if GPU is available")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # 1. Load the data
    dataset = {"source": [], "predictions": [], "reference": []}
    for example in load_generations(args.input_data, args.output_data):
        dataset["source"].append(example["source"])
        dataset["predictions"].append(example["generated"])
        dataset["reference"].append(example["target"])

    scorer = DispersionScorer(args.path_to_mds, args.tmp_dir)
    scorer.compute(dataset, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", False) and not args.cpu else "cpu")


if __name__ == "__main__":
    main()
