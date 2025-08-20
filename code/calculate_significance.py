"""
Calculate statistical significance of the results

Follows the experiments outlined in --experiments-config
"""
import pandas as pd
from argparse import ArgumentParser
import os
import json
import numpy as np
import sys
import ast
from scipy.stats import wilcoxon
import logging
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


def get_paired_sample(model, test_dataset, split, results_dir, significance_pairs):
    paired = significance_pairs.get(f"{model}/{test_dataset}")
    path = None
    if paired:
        path = os.path.join(results_dir, paired, split, "evaluation.json")
        logger.debug(f"{path=}")
    return path


def calculate_significance(x, y):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    # Subtract x-y
    # "Before performing the test, consider calculating d and adjusting it as necessary to ensure that
    # theoretically identically values are not numerically distinct."
    x = np.array(x)
    y = np.array(y)
    diff = np.around(x - y, decimals=3)
    res = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    return res.pvalue


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--experiments-config", default=f"{os.environ['MDS']}/code/experiments.json")
    parser.add_argument("--sig-pairs", default=f"{os.environ['MDS']}/code/significance_pairs.json")
    parser.add_argument("--results-dir", default=f"{os.environ['MDS']}/results")
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
    with open(args.sig_pairs) as f:
        significance_pairs = json.load(f)

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
                if not os.path.exists(path):
                    logger.warning(f"No results at {path}.")
                    results_df.append(row)
                    continue

                # Get paired sample
                paired_results_path = get_paired_sample(model, dataset, split, args.results_dir, significance_pairs)
                if paired_results_path:
                    with open(paired_results_path, "r") as f:
                        paired_results = json.load(f)

                    with open(path, "r") as f:
                        results = json.load(f)
                else:
                    logger.warning(f"No paired results for {model} {dataset} {split}.")
                    results_df.append(row)
                    continue
                logger.debug(f"Processing {path} and {paired_results_path}.")

                for metric, res in results.items():
                    res = res.get("individual", {})
                    paired_res = paired_results.get(metric, {}).get("individual")
                    if not res or not paired_res:
                        continue

                    if isinstance(res[0], dict):
                        for key in res[0]:
                            if (metric, key) in experiments_config["metrics_format"]:
                                res_k = [r[key] for r in res]
                                paired_res_k = [r[key] for r in paired_res]
                                row[(metric, key)] = calculate_significance(res_k, paired_res_k)
                    else:
                        if (metric,) in experiments_config["metrics_format"]:
                            if len(res) != len(paired_res):
                                logger.error(f"Metric {metric} has different number of results for {path} {len(res)} and {paired_results_path} {len(paired_res)}")
                                row[(metric,)] = 999999
                                continue
                            row[(metric,)] = calculate_significance(res, paired_res)
                results_df.append(row)
            run_id += 1

    results_df = pd.DataFrame(results_df)
    if results_df.empty:
        logger.warning(f"No results.")
        exit(0)
    cols = pd.MultiIndex.from_tuples(results_df.columns.tolist())
    results_df.columns = cols
    col_order = [("run_id",), ("model",), ("dataset",), ("split",)] + experiments_config["metrics_format"]
    col_order = pd.MultiIndex.from_tuples(col_order)
    logger.debug(f"wanted columns: {col_order}\ncolumns: {results_df.columns}")
    results_df = results_df[col_order]

    output_path = f"{output_dir}/aggregated_results_sig.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"{results_df}")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()


