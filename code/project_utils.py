"""
Miscellaneous utility functions for the codebase
"""
import logging
import sys
import json
from typing import Any, Union, Dict, Iterable

import jsonlines
import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse experiments config
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    metrics = []
    for m in config["metrics_format"]:
        m = tuple(m)
        if m[1] == "":
            m = (m[0], )
        metrics.append(m)

    config["metrics_format"] = metrics
    return config


def load_generations(input_path: str, predictions_path: str) -> Iterable[Dict[str, Any]]:
    # Change to lazy loading?
    dataset = []
    with jsonlines.open(input_path) as reader:
        for doc in reader:
            dataset.append(doc)

    generated = []
    with jsonlines.open(predictions_path) as reader:
        for doc in reader:
            generated.append(doc)

    # Sort both lists by ID to ensure proper matching
    dataset.sort(key=lambda x: x["id"])
    generated.sort(key=lambda x: x["id"])

    if dataset[0]["id"] != generated[0]["id"]:
        raise Exception(f"Mismatch in IDs between source ({input_path}) and predicted ({predictions_path}): {dataset[0]['id']} != {generated[0]['id']}")

    if len(dataset) != len(generated):
        raise Exception(f"Mismatch in number of examples between source ({input_path}) ({len(dataset)}) and predicted ({predictions_path}) ({len(generated)})")

    for source, pred in zip(dataset, generated):
        out = {
            "id": source["id"],
            "source": source["docs"],
            "target": source["target"],
            "generated": pred["decoded"],
            "prob": np.exp(pred.get("sequence_score", 0)),
            "error": pred.get("error")
        }
        yield out


class CustomJSONEncoder(json.JSONEncoder):
  """Custom JSON Encoder to handle various datatypes."""

  def default(self, o) -> Any:
    if isinstance(o, np.integer):
      return int(o)
    if isinstance(o, np.floating):
      return float(o)
    if isinstance(o, (np.ndarray, pd.core.arrays.numeric.NumericArray)):
      return o.tolist()
    # try:
    #     if pd.isna(o):
    #         return None
    # except ValueError as err:
    #     raise ValueError(f"{o=}\n{err=}")
    try:
      sanitized_o = super(CustomJSONEncoder, self).default(o)
    except TypeError as e:
      raise TypeError(f"Failed to encode '{o}'. Error: {e}")
    return sanitized_o
