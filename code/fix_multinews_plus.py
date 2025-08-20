"""

"""
import os
import sys

import jsonlines

sys.path.append(f"{os.environ['MDS']}/code")
from project_utils import load_config

data_dir = f"{os.environ['MDS']}/data/multinews_plus"
experiments_config = load_config(f"{os.environ['MDS']}/code/experiments.json")

for split in ["val", "test"]:
    with jsonlines.open(f"{data_dir}/{split}.jsonl", 'r') as reader:
        truth = [d for d in reader.iter()]

    for model in experiments_config["models"]:
        output_path = f"{os.environ['MDS']}/results/{model}/multinews_plus/{split}/generated.jsonl"
        with jsonlines.open(output_path, "r") as reader:
            gen = [d for d in reader.iter()]

        import pdb;pdb.set_trace()
