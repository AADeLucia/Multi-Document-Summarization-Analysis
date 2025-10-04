"""
Evaluate summaries with Prometheus 2

Paper: https://aclanthology.org/2024.emnlp-main.248/
Original code: https://github.com/prometheus-eval/prometheus-eval/tree/main
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from argparse import ArgumentParser
import os
import sys
import time
import random
import re
import json

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
from prometheus_eval.prompts import (
    ABSOLUTE_PROMPT,
    ABSOLUTE_PROMPT_WO_REF,
    ABS_SYSTEM_PROMPT,
    SCORE_RUBRIC_TEMPLATE
)

sys.path.append(f"{os.environ['MDS']}/code")
from project_utils import load_generations

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__name__)


###################
# Prompts and Rubrics
###################
# Instructions modified from the icl_experiments.py
INSTRUCTION = """
Write a summary that incorporates all of the following documents:

### DOCUMENTS
{source}
""".strip()

RUBRICS = {
    # # Modified from DUC rating
    # "readability": {
    #   "criteria": "How readable/fluent is the summary? The score reflects the fluency and readability of the summary (independently of whether it contains any information that responds to the topic statement) and is based on factors such as the summary's grammaticality, non-redundancy, referential clarity, focus, and structure and coherence.",
    #   "score1_description": "Very poor",
    #   "score2_description": "Poor",
    #   "score3_description": "Barely acceptable",
    #   "score4_description": "Good",
    #   "score5_description": "Very Good"
    # },
    # Modified from prometheus_eval.prompts
    "factuality": {
        "criteria": "Is the model's summary factually correct and well-supported by the DOCUMENTS?",
        "score1_description": "The model's responses are mostly incorrect or based on unfounded information.",
        "score2_description": "The model sometimes provides factually correct responses, but inaccuracies are common.",
        "score3_description": "The model generally provides factually correct information, though some errors occur.",
        "score4_description": "The model often provides factually accurate information with only occasional minor errors.",
        "score5_description": "The model consistently provides responses that are factually correct and well-supported by evidence."
    },
    # Mine + Gemini
    "coverage": {
        "criteria": "Does the model's summary incorporate salient information from all the documents? Do not penalize the summary for issues that are outside the scope of coverage.",
        "score1_description": "The summary misses most salient information and/or fails to draw from many or all of the source documents.",
        "score2_description": "The summary includes some salient information, but frequently omits key points or fails to adequately represent all source documents.",
        "score3_description": "The summary covers a good portion of the salient information and represents most source documents, though some important details or content from some documents may be missing or underrepresented.",
        "score4_description": "The summary incorporates most salient information from all source documents, with only minor omissions of less critical details, and adequately represents all documents.",
        "score5_description": "The summary comprehensively and accurately captures all salient information from all source documents, providing a well-rounded representation that integrates content from all sources effectively."
    },
    # Modified from SummEval
    "relevance": {
        "criteria": "How well does the summary capture the key points of the documents? Consider whether all and only the important aspects are contained in the summary. Only evaluate the summary for relevance. Do not penalize the summary for issues that are outside the scope of relevance.",
        "score1_description": "Very poor",
        "score2_description": "Poor",
        "score3_description": "Barely acceptable",
        "score4_description": "Good",
        "score5_description": "Very Good"
    },
    "consistency": {
        "criteria": "How consistent are the facts in the summary with the facts in the original documents? Consider whether the summary does reproduce all facts accurately and does not make up untrue information. Do not use your own knowledge, and consider the documents as truth. Only evaluate the summary for consistency. Do not penalize the summary for issues that are outside the scope of consistency.",
        "score1_description": "Very poor",
        "score2_description": "Poor",
        "score3_description": "Barely acceptable",
        "score4_description": "Good",
        "score5_description": "Very Good"
    },
    "fluency": {
        "criteria": "What is the quality of individual sentences, are they well-written and grammatically correct? Consider the quality of individual sentences. Only evaluate the summary for fluency. Do not penalize the summary for issues that are outside the scope of fluency.",
        "score1_description": "Very poor",
        "score2_description": "Poor",
        "score3_description": "Barely acceptable",
        "score4_description": "Good",
        "score5_description": "Very Good"
    },
    "coherence": {
        "criteria": "What is the quality of all sentences collectively? How well do they fit together, and do they sound natural? Consider the quality of the summary as a whole. Only evaluate the summary for coherence. Do not penalize the summary for issues that are outside the scope of coherence.",
        "score1_description": "Very poor",
        "score2_description": "Poor",
        "score3_description": "Barely acceptable",
        "score4_description": "Good",
        "score5_description": "Very Good"
    }
}
DIMENSIONS = list(RUBRICS.keys())


###################
# Evaluator Class
###################
class PrometheusEvaluator(object):
    def __init__(
            self,
            model_name: str = "prometheus-eval/prometheus-bgb-8x7b-v2.0",
            include_reference: bool = True,
            dimensions: List[str] = DIMENSIONS,
            batch_size: int = 4,
            temperature: float = 0.9,
            max_tokens: int = 1024,
            seed: int = 42,
            top_p: float = 1.0,
            device: str = "auto",
            torch_dtype: str = "auto",
            repetition_penalty: float = 1.03
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.top_p = top_p
        self.include_reference = include_reference
        self.repetition_penalty = repetition_penalty
        self.dimensions = dimensions

        # Set up device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set up dtype
        if torch_dtype == "auto":
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = getattr(torch, torch_dtype)

        logger.info(f"Loading model {model_name} on {self.device} with dtype {self.torch_dtype}")

        # Load tokenizer and model
        # Hard-code the max tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = 32768 - self.max_tokens
        logger.debug(f"Tokenizer max length: {self.tokenizer.model_max_length}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def format_grader_instance(self, dataset_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format a single grading instance"""
        # 1. Prepare the instruction
        source_docs = dataset_entry["source"]
        docs = "\n\nDocument: ".join(source_docs)
        docs = "Document: " + docs
        instruction = INSTRUCTION.format(source=docs)

        # 2. Get generation and reference
        generation = dataset_entry["generated"]
        reference = dataset_entry["target"]

        # 3. For each dimension, prepare the rubric and prompt
        prompts = []
        for dim in self.dimensions:
            rubric = RUBRICS[dim]
            score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric)
            if self.include_reference:
                prompt = ABSOLUTE_PROMPT.format(
                    instruction=instruction,
                    response=generation,
                    reference_answer=reference,
                    rubric=score_rubric
                )
            else:
                prompt = ABSOLUTE_PROMPT_WO_REF.format(
                    instruction=instruction,
                    response=generation,
                    rubric=score_rubric
                )
            # Git Repo demo prepends prompt with the system prompt
            prompt = ABS_SYSTEM_PROMPT + "\n\n" + prompt

            p_formatted = {
                "id": dataset_entry["id"],
                "dimension": dim,
                "prompt": prompt
            }

            # Check if the input will be truncated
            tokens = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
            if tokens['input_ids'].shape[1] > self.tokenizer.model_max_length:
                logger.warning(
                    f"Input for ID {m['id']} (dimension: {m['dimension']}) will be truncated: "
                    f"{tokens['input_ids'].shape[1]} tokens > {self.tokenizer.model_max_length} max length"
                )
                p_formatted["truncated"] = True

            prompts.append(p_formatted)
        return prompts

    def batch_generate(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate responses for a batch of the dataset"""
        # Prepare inputs
        inputs, meta = [], []
        for dataset_entry in batch:
            formatted = self.format_grader_instance(dataset_entry)
            prompts = [f["prompt"] for f in formatted]
            meta.extend(formatted)
            for prompt in prompts:
                messages = [
                    {"role": "user", "content": prompt},
                ]
                p = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs.append(p)
        tokenized_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized_inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode outputs (skip input tokens)
        input_lengths = tokenized_inputs['input_ids'].shape[1]
        processed_output = []
        for output, m in tqdm(zip(outputs, meta), desc="Decoding outputs", total=len(meta), disable=logger.level > logging.DEBUG):
            out = m.copy()
            generated_tokens = output[input_lengths:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            feedback, score = self._parse_output_absolute(generated_text)

            out["generated_text"] = generated_text
            out["feedback"] = feedback
            out["score"] = score
            processed_output.append(out)
            logger.debug(out)
        return processed_output

    def _parse_output_absolute(self, output: str) -> Tuple[Optional[str], Optional[int]]:
        """Copied from prometheus_eval.parser"""
        pattern = r"""
            (?:                        # Start of non-capturing group
                \[RESULT\]|\[SCORE\]|   # Match [RESULT] or [SCORE]
                Score:?|score:?|        # Match Score: or score:
                Result:?|\[Result\]:?|  # Match Result: or [Result]:
                score\s+of              # Match "score of"
            )                           # End of non-capturing group
            \s*                         # Allow any whitespace
            (?:\(|\[|\s)*               # Allow opening brackets or whitespace
            (\d+)                       # Capture the digit(s)
            (?:                         # Start of non-capturing group
                (?:\)|\]|\s|$)|         # Allow closing brackets, whitespace, or end of string
                (?:/\s*5|               # Allow /5 with optional whitespace
                   \s*out\s*of\s*5)     # or "out of 5" with flexible whitespace
            )?                          # End of non-capturing group
            (?:\s*$)                    # Match from the end of string 
        """
        match = re.search(pattern, output, re.IGNORECASE | re.VERBOSE)

        if match:
            result = int(match.group(1))
            if 1 <= result <= 5:  # Ensure the result is within the valid range
                feedback = output[: match.start()].strip()
                return output, result
        return None, None


class PrometheusDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    """Custom collate function to handle batching of dictionary entries"""
    return batch


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--input-data", required=True, help="")
    parser.add_argument("--output-data", required=True, help="")
    parser.add_argument("--output-dir", required=True, help="Path to save output")
    parser.add_argument("--n-examples", type=int, default=None, help="Number of examples to evaluate")
    parser.add_argument("--include-reference", action="store_true", default=False, help="Whether to include reference in the prompt")
    parser.add_argument("--dimensions", default=DIMENSIONS, choices=DIMENSIONS, nargs="+")
    parser.add_argument("--model-path", default="prometheus-eval/prometheus-bgb-8x7b-v2.0")
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    parser.add_argument("--max-tokens", default=1024, type=int)
    parser.add_argument("--temperature", default=0.9, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--batch-size", default=4, type=int, help="Number of simultaneous API calls")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run model on")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16", "auto"], help="PyTorch dtype for model")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if not os.path.exists(args.output_data):
        logger.error(f"Output data {args.output_data} does not exist.")
        exit(0)

    # Save args
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/prometheus_args.json", 'w') as f:
        json.dump(vars(args), f)

    # 1. Load the data
    full_dataset = [d for d in load_generations(args.input_data, args.output_data)]
    if args.n_examples is not None and args.n_examples < len(full_dataset):
        sampled_examples = random.sample(full_dataset, args.n_examples)
        logger.info(f"Sampled {len(sampled_examples)} examples.")
    else:
        sampled_examples = full_dataset

    # 2. Create dataset and dataloader
    dataset = PrometheusDataset(sampled_examples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 3. Initialize evaluator
    evaluator = PrometheusEvaluator(
        model_name=args.model_path,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        dimensions=args.dimensions,
        device=args.device,
        torch_dtype=args.torch_dtype,
        include_reference=args.include_reference
    )

    # 4. Run evaluation in batches
    # Save for each batch
    output_file = os.path.join(args.output_dir, "prometheus_outputs.jsonl")
    if args.include_reference:
        output_file = output_file.replace(".jsonl", "_with_ref.jsonl")
    with jsonlines.open(output_file, 'w') as writer:
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch_results = evaluator.batch_generate(batch)
            writer.write_all(batch_results)
    logger.info(f"Wrote output to {output_file}")


if __name__ == "__main__":
    main()

