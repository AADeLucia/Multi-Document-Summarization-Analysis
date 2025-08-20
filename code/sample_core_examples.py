import os
import sys
import random

import jsonlines

sys.path.append(f"{os.environ['MDS']}/code")
from project_utils import load_generations
from project_utils import load_config

random.seed(42)
experiments_config = f"{os.environ['MDS']}/code/experiments.json"
experiments_config = load_config(experiments_config)

N = 10
for dataset in experiments_config["datasets"]:
    print(dataset)

    ground_truth_path = os.path.join(
        os.environ["MDS"], "data", dataset, "train.jsonl"
    )
    with jsonlines.open(ground_truth_path, "r") as reader:
        ground_truth = [o for o in reader.iter()]
    output_path = os.path.join(
        os.environ["MDS"], "data", dataset, "core_examples.jsonl"
    )

    # Sample from dataset
    samples = random.sample(ground_truth, k=N)
    output = []
    for sample in samples:
        # Sample a document
        num_docs = range(len(sample["docs"]))
        doc_id = random.sample(num_docs, k=1)[0]

        # Split document into "sentences" based on periods
        doc = sample["docs"][doc_id].split(".")

        # Remove sentences less than 5 words
        doc = [s for s in doc if len(s.split()) >= 5]

        num_sentences = range(len(doc))
        sentence_id = random.sample(num_sentences, k=1)[0]
        sentence = doc[sentence_id].strip()

        out = {
            "id": sample["id"],
            "doc_id": doc_id,
            "sentence_id": sentence_id,
            "sentence": sentence,
        }
        output.append(out)
        print(f"Claim: {sentence}.\n")

    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(output)
