"""
Save datasets into a standard format

{
    "id":
    "docs": []
    "target"
}
"""
import jsonlines
from argparse import ArgumentParser
import logging
import os
import json
import sys
import regex as re
import gzip
import itertools
from typing import Iterable
from tqdm import tqdm

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)

DATASETS = ["convosumm", "multinews", "multixscience", "multinews_plus"]


def flatten_threads(threads):
    """
    Yield items from any nested iterable; see Reference.
    https://stackoverflow.com/a/40857703
    """
    for x in threads:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten_threads(x):
                yield sub_x
        else:
            yield x


def process_convosumm(data_sources=["reddit"]):
    data_dir = f"{os.environ['MDS']}/data/convosumm/convosumm-data-original/data-non-processed"
    output_dir = f"{os.environ['MDS']}/data/convosumm"
    logger.info(f"On ConvoSumm. Writing to {output_dir}")

    for source in data_sources:
        logger.info(f"Processing {source}")

        input_data = f"{data_dir}/{source}.jsonl"
        split_ids_file = f"{data_dir}/{source}.ids.json"

        with open(split_ids_file) as f:
            split_ids = json.load(f)
            split_ids_rev = {}
            for k, v in split_ids.items():
                split_ids_rev.update({v_i: k for v_i in v})

        data = []
        with jsonlines.open(input_data, "r") as reader:
            for entry in reader.iter():
                row = {
                    "id": entry["example_id"],
                    "split": split_ids_rev[entry["example_id"]],
                    "target": entry["summary"]
                }
                docs = []
                # Flatten the threads
                # Add subreddit info to the first doc
                counter = 0
                for t in flatten_threads(entry["thread_data"]):
                    # Remove usernames
                    if t.startswith("t1_") or t.startswith("t2_") or t.startswith("t3_"):
                        continue
                    if counter == 0:
                        t = f"Title: {entry['title']} Subreddit: {entry['subreddit']} {t}"
                    docs.append(t)
                    counter += 1
                row["docs"] = docs
                data.append(row)

        # Write out
        for split in split_ids:
            with jsonlines.open(f"{output_dir}/{source}/{split}.jsonl", "w") as writer:
                for row in data:
                    if row["split"] == split:
                        writer.write(row)


def process_multinews():
    """
    Data description https://github.com/Alex-Fabbri/Multi-News/tree/master?tab=readme-ov-file#data
    Cleaned source from https://drive.google.com/drive/folders/1jwBzXBVv8sfnFrlzPnSUBHEEAbpIUnFq
    Targets from https://drive.google.com/open?id=1uDarzpu2HFc-vjXNJCRv2NIHzakpSGOw
    """
    data_dir = f"{os.environ['MDS']}/data/multinews/multi-news-original"
    output_dir = f"{os.environ['MDS']}/data/multinews"
    os.makedirs(output_dir, exist_ok=True)
    story_delimiter = "|||||"
    logger.info(f"On MultiNews. Writing to {output_dir}")

    for split in ["val", "test", "train"]:
        source_file = f"{data_dir}/{split}.src.cleaned"
        target_file = f"{data_dir}/{split}.tgt"
        id_file = f"{data_dir}/{split}.id"

        f = open(source_file, "r")
        f2 = open(target_file, "r")
        f3 = open(id_file, "r")
        dataset = []
        for src, tgt, idx in zip(f, f2, f3):
            idx = idx.strip()
            tgt = tgt.strip()
            docs = [d.strip() for d in src.split(story_delimiter) if d.strip()]
            # Issue with the dataset release
            # the "cleaned" version has emptylines instead of the bad scrapes
            # we have to remove them
            if len(docs) == 0:
                logger.warning(f"{idx} empty. Skipping.")
                continue
            row = {
                "id": idx,
                "docs": docs,
                "target": tgt
            }
            dataset.append(row)

        # Write out
        output_file = f"{output_dir}/{split}.jsonl"
        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(dataset)
        logger.info(f"Saved {len(dataset)} samples to {output_file} for {split}")


def process_multinews_plus():
    """
    Cleaned version of Multi-News
    Data description https://github.com/c-juhwan/multi_news_plus/tree/master/cleansing/cleansed_data

    Original is in JSON format:
    {
        "idx": 0,
        "document_num": 2,
        "summary": "",
        "split": "train",
        "cleansed_document": [],
        "raw_document": [],
        "reasons": [],
        "vote_count": {
            "0": 0,
            "1": 0
        }
    }
    """
    data_dir = f"{os.environ['MDS']}/dependencies/multi_news_plus/cleansing/cleansed_data"
    output_dir = f"{os.environ['MDS']}/data/multinews_plus"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"On MultiNews+. Writing to {output_dir}")
    
    # Remove existing files
    for split in ["val", "test", "train"]:
        os.remove(f"{output_dir}/{split}.jsonl")

    split_counters = {}
    for file_path in tqdm(os.listdir(data_dir)):
        if not file_path.endswith(".json"):
            continue
        file_path = os.path.join(data_dir, file_path)

        with open(file_path, "r") as f:
            data = json.load(f)

        for entry in data:
            # Skip entries that are not 'multi'
            if len(entry["cleansed_document"]) < 2:
                continue
            row = {
                "id": entry["idx"],
                "docs": entry["cleansed_document"],
                "target": entry["summary"]
            }
            split = entry["split"]
            split_counters[split] = split_counters.get(split, 0) + 1
            output_file_path = os.path.join(output_dir, f"{split}.jsonl")
            with open(output_file_path, "a") as f:
                f.write(json.dumps(row) + "\n")
    for split, count in split_counters.items():
        logger.info(f"Saved {count:,} samples for {split}")
    logger.info("Done.")


def process_multixscience():
    """
    Data description https://github.com/yaolu/Multi-XScience/tree/master
    """
    data_dir = f"{os.environ['MDS']}/data/multixscience/original"
    output_dir = f"{os.environ['MDS']}/data/multixscience"
    os.makedirs(output_dir, exist_ok=True)

    CITE_RE = re.compile(r"@cite_\d{1,}")

    for split in ["val", "test", "train"]:
        source_file = f"{data_dir}/{split}.json.gz"
        new_dataset = []
        with gzip.open(source_file, "r") as f:
            dataset = json.load(f)
        skip_count = 0
        error_count = 0
        for entry in dataset:
            skip = False
            # dict_keys(['aid', 'mid', 'abstract', 'related_work', 'ref_abstract'])
            docs = []
            for cite_key, cite_info in entry["ref_abstract"].items():
                # {'@cite_5': {'mid': '2176959931', 'abstract': }}
                content = cite_info["abstract"].strip()
                if content == "":
                    logger.debug(f"Skipping {cite_key} {cite_info['mid']}. Empty.")
                    error_count += 1
                    skip = True
                docs.append(content)
            if len(docs) < 2:
                skip = True
                logger.debug(f"Entry {entry['aid']} has only one source document.")
            if skip:
                skip_count += 1
                continue
            # Replace paper-specific citations with general cite
            target = CITE_RE.sub("@cite", entry["related_work"])
            row = {
                "id": entry["aid"],  # Use arXiv ID. Leave Microsoft Academic Graph ID
                "docs": docs,
                "target": target
            }
            new_dataset.append(row)

        # Write out
        with jsonlines.open(f"{output_dir}/{split}.jsonl", "w") as writer:
            writer.write_all(new_dataset)

        logger.info(f"Wrote {len(new_dataset)} samples for {split}. Skipped: {skip_count}. Missing abstracts: {error_count}")


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default=DATASETS, type=str, nargs="+", choices=DATASETS, help="")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    for dataset in args.dataset:
        if dataset == "convosumm":
            process_convosumm()
        elif dataset == "multinews":
            process_multinews()
        elif dataset == "multixscience":
            process_multixscience()
        elif dataset == "multinews_plus":
            process_multinews_plus()

