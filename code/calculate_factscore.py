"""
This script is a wrapper for the FactScore codebase https://github.com/shmsw25/FActScore

Handles the preprocessing and parsing of the score output
"""
import logging
from argparse import ArgumentParser
import sys
import os
import numpy as np
import jsonlines
import json
import subprocess
from argparse import Namespace
sys.path.append(f"{os.environ['MDS']}/code")
from project_utils import load_generations
from factscore.factscorer import FactScorer
from tqdm import tqdm

FACTSCORE_CACHE=f"{os.environ['MDS']}/dependencies/FActScore-MDS/.cache/factscore"

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)


class CustomJSONizer(json.JSONEncoder):
    """
    Sanitize datatypes for JSON for serialization
    From https://stackoverflow.com/a/67572570
    """
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return super().encode(bool(obj))
        if isinstance(obj, np.float32):
            return super().encode(float(obj))
        if isinstance(obj, set):
            return super().encode(list(obj))
        return super().default(obj)


def calculate_factscore(input_data, output_data, dataset, output_dir, mode, model_name, factscore_cache_base=FACTSCORE_CACHE, debug=False):
    # Change FactScore cache depending on dataset and mode
    factscore_cache = os.path.join(factscore_cache_base, mode, dataset)

    # Check if dataset file is created
    data_base = f"{os.environ['MDS']}/data/{dataset}"
    data_path = f"{data_base}/factscore_corpus.jsonl"
    if not os.path.exists(data_path):
        logger.info(f"Creating data file at {data_path} from validation and test sets in {data_base}")
        os.makedirs(f"{os.environ['MDS']}/data/{dataset}", exist_ok=True)
        with jsonlines.open(data_path, "w") as writer:
            for split in ["val", "test"]:
                with jsonlines.open(f"{data_base}/{split}.jsonl", "r") as reader:
                    for doc in tqdm(reader.iter(), ncols=0, desc="Loading data"):
                        doc_text = [d.strip() for d in doc["docs"] if d.strip() != ""]
                        new_doc = {
                            "title": doc["id"],
                            "text": doc_text
                        }
                        writer.write(new_doc)
    else:
        logger.info(f"Data file at {data_path} already exists.")

    # Initialize FactScorer
    logger.info(f"Initializing FactScore with cache {factscore_cache} in {mode} mode")
    fs = FactScorer(
        model_name=model_name,
        data_dir=FACTSCORE_CACHE,
        model_dir=FACTSCORE_CACHE,
        cache_dir=factscore_cache,
        mode=mode
    )

    # this will create a database using your file
    # once DB file is created, you can reuse it by only specifying `db_path`
    fs.register_knowledge_source(dataset, data_path=data_path)

    output_file = f"{output_dir}/evaluation_factscore_{mode}.json"
    logger.info(f"Will save to {output_file}")
    # Load generations
    generations_all = load_generations(
        input_data,
        output_data,
    )
    generations, topics = [], []
    for example in generations_all:
        generations.append(example["generated"])
        topics.append(example["id"])
        if debug and len(topics) > 3:
            break
    logger.info(f"Loaded {len(generations)} data points")
    out = fs.get_score(topics, generations, gamma=10, knowledge_source=dataset)
    logger.info("FActScore = %.1f%%" % (100 * out["score"]))
    if "init_score" in out:
        logger.info("FActScore w/o length penalty = %.1f%%" % (100 * out["init_score"]))
    logger.info("Respond ratio = %.1f%%" % (100 * out["respond_ratio"]))
    logger.info("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    # Summarize the final scores for easy loading
    summary_out = {
        "overall": {
            "score": out["score"],
            "respond_ratio": out["respond_ratio"],
            "init_score": out.get("init_score", 0),
            "num_facts_per_response": out["num_facts_per_response"]
        },
        "individual": []
    }
    for doc_result in out["decisions"]:
        """{
            atom: "", 
            "individual": {
                "prompt":"", 
                "passage":"", 
                "true_score":, 
                "false_score":, 
                "is_supported":
            }
            "lm": {
                "is_supported": "true"
            }, 
            "npm": {
                "is_supported": "false", 
                "true_score": 0.0, 
                "false_score": 1.0
            }, 
            "is_supported": "false"
        }
        """
        num_atoms = len(doc_result)
        score = sum([d["is_supported"] for d in doc_result]) / num_atoms
        score_lm = sum([d["lm"]["is_supported"] for d in doc_result]) / num_atoms
        len_source = len(doc_result[0]["individual"])
        atom_supports, source_coverage = [], []
        for doc_atom in doc_result:
            support = 0
            for i, d in enumerate(doc_atom["individual"]):
                if d["is_supported"]:
                    support += 1
                    source_coverage.append(i)
            atom_supports.append(support / len_source)
        avg_atom_support = np.mean(atom_supports)
        source_coverage = set(source_coverage)
        res = {
            "score": score,
            "score_lm": score_lm,
            "num_atoms": num_atoms,
            "support": avg_atom_support,
            "support_per_atom": atom_supports,
            "num_sources": len_source,
            "covered_sources": source_coverage,
            "coverage": len(source_coverage) / len_source
        }
        summary_out["individual"].append(res)

    with open(output_file, "w") as f:
        json.dump(summary_out, f, cls=CustomJSONizer)

    with open(output_file.replace(".json", "_detailed.json"), "w") as f:
        json.dump(out, f, cls=CustomJSONizer)

    logger.info(f"Wrote results to {output_file}")

    if debug:
        return


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-data", required=True, help="")
    parser.add_argument("--output-data", required=True, help="")
    parser.add_argument("--dataset", required=True, help="")
    parser.add_argument("--factscore-cache", default=FACTSCORE_CACHE, help="")
    parser.add_argument("--model-name", default="retrieval+llama")
    parser.add_argument("--mode", default="multi", choices=["multi", "single"])
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    calculate_factscore(
        input_data=args.input_data,
        output_data=args.output_data,
        dataset=args.dataset,
        output_dir=args.output_dir,
        mode=args.mode,
        model_name=args.model_name,
        factscore_cache_base=args.factscore_cache,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
