"""
Misc methods for formatting, processing, and plotting data
"""
import sys
import os
from typing import List, Dict, Any

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(f"{os.environ['MDS']}/code")
import project_utils

dataset_to_domain = {
    "convosumm": "Convo",
    "convosumm/reddit": "Convo",
    "multixscience": "Science",
    "multinews": "News",
    "multinews_plus": "News+"
}

wanted_metrics = [
    "unieval_factuality",
    "prometheus_factuality",
    "prometheus_consistency",
    "unieval_relevance",
    "prometheus_relevance",
    "rouge_rouge1",
    "rouge_rougeLsum",
    "moverscore",
    "unieval_fluency",
    "prometheus_fluency",
    "unieval_coherence",
    "prometheus_coherence",
    "compression_sentence",
    "compression_token",
    "prometheus_coverage"
]


def get_model_name(raw_model_name: str) -> str:
    if "PRIMERA" in raw_model_name:
        return "PRIMERA"
    if "ICL" in raw_model_name:
        raw_model_name = raw_model_name.replace("ICL", "Llama 3.1")
    if "Llama 3.1" == raw_model_name:
        raw_model_name = "Llama 3.1 (0-shot)"
    else:
        raw_model_name = raw_model_name.split("-")[0]
    return raw_model_name


def get_train_domain(row: pd.Series) -> str:
    if row.model == "ground_truth":
        return dataset_to_domain[row.dataset]
    if "icl" == row.model.lower():
        return "zero-shot"
    else:
        return dataset_to_domain[row.model.split("-")[-1].lower()]


def get_test_domain(dataset_name: str) -> str:
    return dataset_to_domain[dataset_name]


def load_config(path: str) -> Dict[str, Any]:
    return project_utils.load_config(path)


def load_generations(input_path: str, predictions_path: str) -> List[Dict[str, Any]]:
    return project_utils.load_generations(input_path, predictions_path)

