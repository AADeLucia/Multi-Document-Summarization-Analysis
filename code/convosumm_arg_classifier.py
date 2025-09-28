"""
Extract arguments from the argument CLPR model from ConvoSumm.

Output is dataset ready to be used with fairseq

Modified from the following scripts:
- https://github.com/Yale-LILY/ConvoSumm/blob/master/code/scripts/arg_classifier.py
- https://github.com/Yale-LILY/ConvoSumm/blob/master/code/scripts/process_arg_classifier_results.py
"""
import torch
import re
from tqdm import tqdm
import logging
import random
import jsonlines
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from argparse import ArgumentParser
import os
import sys
import pdb
import spacy
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5 * (10**6)

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
NEWLINE_TAGS = re.compile(r"NEWLINE_CHAR")
NEWLINE_RE = re.compile(r"[\n\r]+")
CONVOSUM_NEWLINE = "|NEWLINE|"
def prepare_document(sample):
    sentences = []
    for doc in sample["docs"]:
        doc = doc.strip()
        doc = NEWLINE_TAGS.sub(CONVOSUM_NEWLINE, doc)
        doc = NEWLINE_RE.sub(CONVOSUM_NEWLINE, doc)
        doc_text = [x.text for x in nlp(doc).sents]
        sentences.extend(doc_text)
    return sentences


def divide_chunks(l, n):
    # from https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def classify_sentences(sentences, model, tokenizer, device, batch_size):
    if len(sentences) <= batch_size:
        batches = [sentences]
    else:
        batches = divide_chunks(sentences, batch_size)
    predictions = []
    for batch in batches:
        encoded_inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            classification_logits = model(**encoded_inputs)[0]
            probs = torch.nn.functional.softmax(classification_logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        preds = preds.detach().cpu().tolist()
        predictions.extend(preds)
    return predictions


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to ConvoSumm CLPR model")
    parser.add_argument("--input-data", required=True, help="")
    parser.add_argument("--output-dir", required=True, help="")
    parser.add_argument("--output-file", required=True, help="")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume where previous run left off")
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size for argument identification")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only even if GPU is available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Set CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset = []
    with jsonlines.open(args.input_data) as reader:
        for doc in reader:
            dataset.append(doc)
    logger.info(f"Loaded {len(dataset)} examples from {args.input_data}")

    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    model.eval()
    tokenizer.model_max_length = model.config.max_position_embeddings

    if not os.path.exists(args.output_dir):
        logger.info(f"Creating output directory {args.output_dir}")
        os.mkdir(args.output_dir)

    logger.info(f"Running argument extraction model")
    output_file = os.path.join(args.output_dir, args.output_file)
    output_file_s = output_file.replace(".jsonl", ".source")
    output_file_t = output_file.replace(".jsonl", ".target")

    if args.resume and os.path.exists(output_file):
        mode = "a"
        # Find how many examples are already written
        with open(output_file_t, "rb") as f:
            num_lines = sum(1 for _ in f)
        dataset = dataset[num_lines:]
        logger.info(f"Resuming from previous iteration {output_file}. Starting at {num_lines} lines.")
    else:
        mode = "w"

    writer = jsonlines.open(output_file, mode)
    writer_s = open(output_file_s, mode)
    writer_t = open(output_file_t, mode)
    for sample in tqdm(dataset, ncols=0):
        # Parse text
        sentences = prepare_document(sample)
        logger.debug(f"Processing {len(sentences)}")

        # Classify
        preds = classify_sentences(sentences, model, tokenizer, device, batch_size=args.batch_size)

        cur_data = {
            "texts": sentences,
            "predictions": preds,
            "summary": sample["target"],
            "id": sample["id"]
        }
        filtered_sentences = [sentences[i] for i, p in enumerate(preds) if p != 2]
        source = " ".join(filtered_sentences)
        target = sample["target"]

        # Write out
        writer.write(cur_data)
        writer_s.write(source + "\n")
        writer_t.write(target + "\n")

    # Close files
    writer.close()
    writer_s.close()
    writer_t.close()


if __name__ == "__main__":
    main()

