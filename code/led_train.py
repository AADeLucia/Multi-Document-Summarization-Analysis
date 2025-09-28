"""
Fine-tune Longformer Encoder-Decoder (LED) for multi-document summarization

Paper: https://arxiv.org/pdf/2004.05150
HuggingFace Model: https://huggingface.co/allenai/led-large-16384
Fine-tuning Example: https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing
"""
import random
import logging
from argparse import ArgumentParser
import sys
import os
from dataclasses import dataclass, field
import typing

import jsonlines
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import HfArgumentParser
from transformers import PretrainedTokenizer

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
def collate_fn(
        batch: List[Dict[Any, Any]],
        tokenizer: PretrainedTokenizer
):
    formatted_batch = {
        "input_ids": [],
        "attention_mask": [],
        "global_attention_mask": []
    }
    inputs = ["<s>".join(x['docs']) for x in batch]
    outputs = [x['target'] for x in batch]
    tokenized_inputs = tokenizer(
        inputs,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )
    formatted_batch["input_ids"] = tokenized_inputs["input_ids"]
    formatted_batch["attention_mask"] = tokenized_inputs["attention_mask"]
    tokenized_outputs = tokenizer(
        outputs,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )
    formatted_batch["labels"] = tokenized_outputs["input_ids"]
    # We have to make sure that the PAD token is ignored
    formatted_batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in formatted_batch["labels"]
    ]
    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    return formatted_batch


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = tokenizer(
        batch["abstract"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


#############
# Main
#############
@dataclass
class ExperimentArguments(Seq2SeqTrainingArguments):
    """
    Extend the HuggingFace Trainer to accept experiment-specific arguments

    Example: https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/training_args_seq2seq.py#L28
    """
    data_path: str = field(
        default=f"{os.environ['MDS']}/data/multixscience",
        metadata={"help": "Path to dataset folder in .jsonl format. Assumes train.jsonl and val.jsonl exist if training."}
    )
    test_split: str = field(
        default="val",
        metadata={"help": "Test split for inference-only mode. 'val' or 'test'."}
    )
    model_name_or_path: str = field(
        default="allenai/led-large-16384",
        metadata={"help": "Path to model for training or inference"}
    )
    patience: int = field(
        default=-1,
        metadata={"help": "Stop training when the specified metric worsens for early_stopping_patience evaluation calls."}
    )


def parse_args():
    parser = HfArgumentParser(ExperimentArguments)
    args = parser.parse_args_into_dataclasses()[0]  # Hack because returns tuple
    return args


def main():
    args = parse_args()
    if args.log_level == "debug":
        logger.setLevel(logging.DEBUG)

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    train_dataset = []
    with jsonlines.open(f"{args.data_path}/train.jsonl") as reader:
        for doc in reader:
            train_dataset.append(doc)
    logger.info(f"Loaded {len(train_dataset)} examples from Train split")
    val_dataset = []
    with jsonlines.open(f"{args.data_path}/val.jsonl") as reader:
        for doc in reader:
            val_dataset.append(doc)
    logger.info(f"Loaded {len(val_dataset)} examples from Validation split")

    # Load model
    #model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Set up Trainer
    if args.patience != -1:
        # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/callback#transformers.EarlyStoppingCallback
        callbacks = [
            transformers.EarlyStoppingCallback(early_stopping_patience=args.patience)
        ]
    else:
        callbacks = None
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=lambda x: collate_fn(x, tokenizer),
        callbacks=callbacks
    )

    if args.do_train:
        train_results = trainer.train()
        trainer.save_model()
        trainer.save_state()
        tokenizer.save_pretrained(args.output_dir)

        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)

    if args.do_predict:
        logger.error(f"Predicting not implemented")
        exit(1)


if __name__ == '__main__':
    main()

