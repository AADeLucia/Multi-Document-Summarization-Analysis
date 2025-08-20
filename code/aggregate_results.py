"""
Aggregate the results from /results into a CSV file.

Follows the experiments outlined in experiments.json
"""
from argparse import ArgumentParser
import os
import json
import sys
import ast
import logging
from typing import List, Dict, Any, Union

import pandas as pd
import scipy
import numpy as np

sys.path.append(f"{os.environ['MDS']}/code")
from project_utils import load_config

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)


#############
# Helpers
#############
def aggregate_results(data: List[Union[int, float]], calculate_ci: bool = False) -> Dict[str, Any]:
    data = np.array(data).astype("float64")
    res = {
        "mean": np.nanmean(data).item(),
        "std": np.nanstd(data).item()
    }
    if calculate_ci:
        bootstrap_result = scipy.stats.bootstrap(
            (data,),
            np.nanmean,
            confidence_level=0.95,
            n_resamples=1000,
            method='BCa',
            random_state=42
        )
        res["ci"] = (bootstrap_result.confidence_interval.low, bootstrap_result.confidence_interval.high)
        res["ci_std"] = bootstrap_result.standard_error
    return res


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output-format", default="csv", choices=["csv", "latex"])
    parser.add_argument("--experiments-config", default=f"{os.environ['MDS']}/code/experiments.json")
    parser.add_argument("--results-dir", default=f"{os.environ['MDS']}/results")
    parser.add_argument("--ci", help="Calculate 95% confidence interval instead of using only the mean", action="store_true")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load config
    experiments_config = load_config(args.experiments_config)
    output_dir = os.path.join(
        args.results_dir,
        os.path.basename(args.experiments_config).split('.')[0]
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results_df = []
    run_id = 1
    for dataset in experiments_config["datasets"]:
        for model in experiments_config["models"]:
            for split in experiments_config["splits"]:
                row = {
                    ("run_id",): run_id,
                    ("model",): model,
                    ("dataset",): dataset,
                    ("split",): split
                }
                path = experiments_config["format"].replace("{model}", model).replace("{dataset}", dataset).replace("{split}", split)
                path = f"{args.results_dir}/{path}/evaluation.json"
                if os.path.exists(path):
                    with open(path, "r") as f:
                        results = json.load(f)

                    for metric_name, metric_results in results.items():
                        metric_results = metric_results.get("individual")

                        # Skip empty
                        if not metric_results:
                            continue

                        if not isinstance(metric_results[0], dict):
                            if (metric_name,) in experiments_config["metrics_format"]:
                                row[(metric_name,)] = aggregate_results(metric_results, calculate_ci=args.ci)
                        else:
                            # Re-organize dictionaries into per-submetric lists
                            reorg_results = {}
                            for idx, idx_res in enumerate(metric_results):
                                for k, v in idx_res.items():
                                    if (metric_name,k) in experiments_config["metrics_format"]:
                                        reorg_results[k] = reorg_results.get(k, [])
                                        reorg_results[k].append(v)
                            for k, v in reorg_results.items():
                                row[(metric_name,k)] = aggregate_results(v, calculate_ci=args.ci)
                else:
                    logger.warning(f"No results for {model} {dataset} {split}.")
                results_df.append(row)
            run_id += 1

    results_df = pd.DataFrame(results_df)
    cols = pd.MultiIndex.from_tuples(results_df.columns.tolist())
    results_df.columns = cols
    col_order = [("run_id",), ("model",), ("dataset",), ("split",)] + experiments_config["metrics_format"]
    col_order = pd.MultiIndex.from_tuples(col_order)
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    output_path = f"{output_dir}/aggregated_results"
    if args.output_format == "csv":
        results_df.to_csv(f"{output_path}.csv", index=False)
    elif args.output_format == "latex":
        results_df.to_latex(f"{output_path}.tex")
    logger.info(f"Saved aggregated results to {output_dir}")


if __name__ == "__main__":
    main()


