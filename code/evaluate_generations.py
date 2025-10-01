"""Evaluate generations

Author: Alexandra DeLucia
"""
from argparse import ArgumentParser
import sys
import os
import json
import re
import logging
from typing import Dict, Any, List, Optional

import pandas as pd
import jsonlines
from tqdm import tqdm
import evaluate
import torch
import numpy as np
import spacy
from bert_score import BERTScorer
nlp = spacy.load("en_core_web_sm")


# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)

sys.path.append(f"{os.environ['MDS']}/code")
from project_utils import load_generations, CustomJSONEncoder
METRICS = ["rouge", "unieval", "moverscore", "bertscore", "compression", "prometheus"]


def eval_rouge(dataset):
    """
    https://huggingface.co/spaces/evaluate-metric/rouge
    :return:
    """
    try:
        rouge = evaluate.load('rouge', keep_in_memory=True)
    except FileNotFoundError as err:
        rouge = evaluate.load(f"{os.environ['HF_HUB_CACHE']}/metrics/rouge", keep_in_memory=True)
    results = rouge.compute(
        predictions=dataset["predictions"],
        references=dataset["reference"],
        use_aggregator=False,
        use_stemmer=True
    )
    overall = {
        key: np.mean(results[key]) for key in results
    }
    individual = []
    for i in range(len(dataset["predictions"])):
        individual.append({
            k: v[i] for k,v in results.items()
        })
    return {
        "individual": individual,
        "overall": overall
    }


def eval_moverscore(
        dataset: Dict[str, Any],
        device: str,
        batch_size: int = 128,
        ngram: int = 1,
        remove_subwords: bool = True
) -> Dict[str, Any]:
    sys.path.append(f"{os.environ['MDS']}/dependencies/emnlp19-moverscore")
    # Use the original version with BERTMNLI to reproduce the results.
    from moverscore import get_idf_dict, word_mover_score
    from collections import defaultdict

    idf_dict_hyp = get_idf_dict(dataset["predictions"])
    idf_dict_ref = get_idf_dict(dataset["reference"])

    raw_scores = word_mover_score(
        dataset["reference"],
        dataset["predictions"],
        idf_dict_ref,
        idf_dict_hyp,
        stop_words=[],
        n_gram=ngram,
        remove_subwords=remove_subwords,
        batch_size=batch_size,
        device=device
    )
    scores = {
        "individual": raw_scores,
        "overall": np.average(raw_scores)
    }
    return scores


def eval_bertscore(
        dataset: Dict[str, Any],
        device: str,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        batch_size: int = 64,
) -> Dict[str, Any]:
    scorer = BERTScorer(
        model_type=model_type,
        device=device,
        nthreads=4,
        lang="en",
        batch_size=batch_size,
        rescale_with_baseline=True
    )
    P, R, F1 = scorer.score(
        dataset["predictions"],
        dataset["reference"],
        verbose=True
    )
    individual_scores = []
    for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist()):
        individual_scores.append({"precision": p, "recall": r, "f1": f})

    scores = {
        "individual": individual_scores,
        "overall": {
            "precision": np.average(P).item(),
            "recall": np.average(R).item(),
            "f1": np.average(F1).item()
        }
    }
    return scores


def eval_unieval(dataset, device):
    """
    https://github.com/maszhongming/UniEval/tree/main
    :param dataset:
    :param device:
    :return:
    """
    sys.path.append(f"{os.environ['MDS']}/dependencies/UniEval")
    from utils import convert_to_json
    from metric.evaluator import get_evaluator

    # Prepare data for pre-trained evaluators
    # UniEval expects a single source document for each prediction
    # concatenate all the source documents
    data = convert_to_json(
        output_list=dataset["predictions"],
        src_list=[" ".join(s) for s in dataset["source"]],
        ref_list=dataset["reference"]
    )
    # Evaluator for summarization task
    evaluator = get_evaluator("summarization", device=device)
    eval_scores = evaluator.evaluate(data, print_result=True, dims=["coherence", "fluency", "relevance"], overall=False)

    # Evaluator for factual consistency task
    evaluator = get_evaluator("fact", device=device)
    eval_scores_fact = evaluator.evaluate(data, print_result=True)

    # Combine scores
    dimensions = ["coherence", "fluency", "relevance", "factuality", "overall"]
    aggregated_scores = {d: [] for d in dimensions}
    for e1, e2 in zip(eval_scores, eval_scores_fact):
        # Add the UniEval "consistency" (i.e., factuality) score to the others
        e1["factuality"] = e2["consistency"]
        e1["overall"] = np.average([e1[d] for d in dimensions if d != "overall"])
        for d in dimensions:
            aggregated_scores[d].append(e1[d])

    # Scores are calculated for individual documents. Aggregate them.
    scores = {
        "overall": {
            d: np.mean(aggregated_scores[d]) for d in dimensions
        },
        "individual": eval_scores
    }
    return scores


# Clean score field
def clean_score(score: object):
    try:
        score = int(score)
    except ValueError as err:
        logger.warning(f"Issue converting '{score}' to integer. {err=}")
        score =  np.nan
    return score


def eval_prometheus(output_dir: str) -> Optional[Dict[str, Any]]:
    output_file = f"{output_dir}/prometheus_aggregated2.json"
    logger.warning(f"Aggregating Prometheus scores.")
    individual_scores = {}
    for filename in os.listdir(output_dir):
        if "prometheus_eval" not in filename:
            continue

        # Skip non-source and non-reference results
        if ("source" not in filename) or ("ref" not in filename):
            continue
        dimension = filename.split("_")[-1].strip(".jsonl")

        df = pd.read_json(
            os.path.join(output_dir, filename),
            lines=True
        )

        df = df.reset_index().rename(columns={"index": "id"})
        df["score"] = df.score.map(clean_score)#.astype(dtype='Int64')
        individual_scores[dimension] = df["score"].values

    if not len(individual_scores):
        return

    # Scores are calculated for individual documents. Aggregate them.
    aggregated_scores = {
        "overall": {
            k: np.nanmean(v) for k, v in individual_scores.items()
        }
    }
    # Re-group the individual scores
    regrouped_scores = []
    for idx in range(len(df)):
        regrouped_scores.append({
            k: v[idx] for k, v in individual_scores.items()
        })
    aggregated_scores["individual"] = regrouped_scores

    with open(output_file, "w") as f:
        json.dump(aggregated_scores, f, cls=CustomJSONEncoder)

    return aggregated_scores


def eval_core(output_dir):
    """Parses output from eval_core.sh if it exists"""
    output_file = f"{output_dir}/core_output.jsonl"
    if not os.path.exists(output_file):
        logger.warning(f"Core results not found at {output_file}. Skipping.")
        return

    scores = {
        "individual": []
    }
    with jsonlines.open(output_file, "r") as reader:
        for item in reader.iter():
            # Calculate the average
            # {"decoded": "", "id": "", "score": 0.0, "meta":
            # {"parsed": 0.0, "raw": "",
            # "claims": [{"text": "", "score": 0.0}]}
            claims = item["meta"].get("claims", [])
            if claims:
                example_scores = [s["score"] for s in claims]
                avg = np.average(example_scores)
            else:
                # Keep record that this example has no claims
                avg = -1

            scores["individual"].append(avg)
    # Don't include the -1s in the overall average
    scores["overall"] = np.average([s for s in scores["individual"] if s != -1])
    return scores


def eval_compression(dataset: Dict[str, List[str]]):
    scores = {
        "individual": [],
        "overall": {}
    }
    for sources, prediction in tqdm(zip(dataset["source"], dataset["predictions"]), desc=f"Compression", ncols=0):
        indiv_score = {}

        source_length = sum([len(parse_sentences(s)) for s in sources])
        summary_length = len(parse_sentences(prediction))
        compression_ratio = summary_length / source_length
        indiv_score["sentence"] = compression_ratio

        source_length = sum([len(s.split()) for s in sources])
        summary_length = len(prediction.split())
        compression_ratio = summary_length / source_length
        indiv_score["token"] = compression_ratio

        scores["individual"].append(indiv_score)

    for mode in ["sentence", "token"]:
        scores["overall"][mode] = np.average([i[mode] for i in scores["individual"]])
    return scores


def parse_sentences(
        passage: str,
) -> List[Dict[str, Any]]:
    doc = nlp(passage)
    sentences = []
    # sent is a spacy span object https://spacy.io/api/span#init
    # span start/end is based on token index (sent.start, sent.end)
    # convert token index to character to match the annotations
    for sent in doc.sents:
        sentences.append({
            "text": sent.text,
            "span_start": sent.start_char,
            "span_end": sent.end_char
        })
    return sentences


def calculate_metric(metric, dataset, device, args):
    if metric == "unieval":
        return eval_unieval(dataset, device)
    if metric == "rouge":
        return eval_rouge(dataset)
    if metric == "moverscore":
        return eval_moverscore(dataset, device)
    if metric == "core":
        return eval_core(args.output_dir)
    if metric == "dispersion":
        return eval_dispersion(dataset, device, args.output_dir)
    if "factscore" in metric:
        setting = "multi" if "-md" in metric else "single"
        return eval_factscore(args.output_dir, setting, args)
    if "prometheus" in metric:
        return eval_prometheus(args.output_dir)
    if metric == "bertscore":
        return eval_bertscore(dataset, device=device)
    if metric == "compression":
        return eval_compression(dataset)
    logger.error(f"Unknown metric {metric}")


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-data", required=True, help="")
    parser.add_argument("--output-data", required=True, help="")
    parser.add_argument("--output-dir", required=True, help="")
    parser.add_argument("--metrics", type=str, nargs="+", default=METRICS, choices=METRICS)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing. Otherwise only adds fields.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only even if GPU is available")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    output_file = f"{args.output_dir}/evaluation.json"

    if not os.path.exists(args.output_data):
        logger.error(f"Output data {args.output_data} does not exist.")
        exit(0)

    # 1. Load the data
    dataset = {"source": [], "predictions": [], "reference": []}
    for example in load_generations(args.input_data, args.output_data):
        dataset["source"].append(example["source"])
        dataset["predictions"].append(example["generated"])
        dataset["reference"].append(example["target"])

    # Load previously calculated scores
    if not args.overwrite and os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_scores = json.loads(f.read())
        logger.info(f"Loaded existing scores from {output_file}")
    else:
        all_scores = {}

    # 2. Run the metrics
    for metric in args.metrics:
        logger.info(f"Evaluating {metric}")
        if metric in all_scores and metric != "prometheus":
            logger.info(f"{metric} scores already calculated. Skipping.")
        else:
            scores = calculate_metric(metric, dataset, device, args)
            if scores:
                all_scores[metric] = scores
                # Save scores as they are calculated
                with open(output_file, "w") as f:
                    f.write(json.dumps(all_scores, sort_keys=True, indent=4, cls=CustomJSONEncoder))

        logger.info(f"{metric}: {all_scores.get(metric, {}).get('overall', 'Metric not found')}")


if __name__ == "__main__":
    main()

