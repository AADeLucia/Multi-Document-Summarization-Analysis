"""Run inference via API call"""
from argparse import ArgumentParser
import logging
import sys
import os
from functools import partial
import random
from pyexpat.errors import messages
from typing import Dict, Any, List, Callable, Optional
import asyncio
import time
from typing import Dict, List, Any
import json

import requests
from openai.types.chat.chat_completion import ChatCompletion
import nest_asyncio
import backoff
import jsonlines
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from openai import AsyncOpenAI
from openai import APIError, BadRequestError, APIConnectionError
from transformers import AutoTokenizer, PreTrainedTokenizer


# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)


SYSTEM_PROMPT = """You are a multi-document summarizer. When given a list of DOCUMENTS, you provide a SUMMARY that incorporates all of the documents."""
SYSTEM_PROMPT_ICL = SYSTEM_PROMPT + """ Example(s) are provided below:

EXAMPLE:
{example}
"""
PROMPT = """Write a summary for the following DOCUMENTS:

DOCUMENTS:
{source}

SUMMARY:
{target}
"""

def format_prompt(
        example: Dict[str, Any],
        include_target: bool = False
) -> str:
    # Current task / input
    docs = "\n\nDocument: ".join(example["docs"])
    docs = "Document: " + docs
    target = example["target"] if include_target else ""
    prompt = PROMPT.format(source=docs, target=target)
    return prompt


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_time=60
)
async def batch_response(messages: List[List[Dict[str, str]]], client: Callable) -> List[ChatCompletion]:
    async_responses = [
        client(
            messages=x,
        )
        for x in messages
    ]
    return await asyncio.gather(*async_responses, return_exceptions=True)


def format_output(
        completions: List[ChatCompletion],
        batch: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    outputs = []
    for ex, response in zip(batch, completions):
        # Catch document-too-long error
        if isinstance(response, BadRequestError):
            output = {
                "id": ex["id"],
                "decoded": "",
                "error": str(response.message),
            }
        # Catch connection error
        elif isinstance(response, APIError):
            logger.error(f"Error. Exiting. {str(response)=}")
            exit(1)
        else:
            output = {
                "id": ex["id"],
                "decoded": response.choices[0].message.content,
            }
        outputs.append(output)
    return outputs


def prepare_input(batch: List[List[Dict[str, str]]], system_prompt: str) -> List[List[Dict[str, Any]]]:
    # Format model input
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_prompt(b)}
        ] for b in batch
    ]
    return messages


#############
# Main
#############
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model-path", help="Path to 'trained' model (saved prompt). Otherwise, default zero-shot prompt is used.")
    subparsers = parser.add_subparsers(dest="mode")

    train = subparsers.add_parser("train", help="Train a model by creating a prompt")
    train.add_argument("--few-shot-k", default=2, type=int, help="Examples to load from --input-data-train as examples in prompt")
    train.add_argument("--input-data-train", required=True, type=str, help="Data to use for the single shot context")

    test = subparsers.add_parser("test", help="Test a model")
    test.add_argument("--input-data-test", required=True, help="JSONLines file for inference data")
    test.add_argument("--output-dir", required=True, help="Path to save output")
    test.add_argument("--server-base", help="API endpoint", default="https://api.together.xyz/v1")
    test.add_argument("--model-path-base", default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", help="Note: an instruction-tuned model is required")
    test.add_argument("--max-completion-tokens", default=1024, type=int)
    test.add_argument("--temperature", default=0.9, type=float)
    test.add_argument("--num-examples", default=-1, type=int, help="Number of examples from the test set to generate. -1 uses all of them.")
    test.add_argument("--resume", action="store_true", help="Resume from pre-existing output file.")
    test.add_argument("--top-p", default=1.0, type=float)
    test.add_argument("--batch-size", default=4, type=int, help="Number of simultaneous API calls")
    test.add_argument("--sleep", default=10, type=int, help="Seconds to sleep between API calls")
    test.add_argument("--simulate", action="store_true", help="Calculate cost by returning number of tokens. Does not call any API.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    random.seed(args.seed)

    if args.mode == "train":
        # Load input data
        with jsonlines.open(args.input_data_train) as reader:
            train_dataset = [d for d in reader.iter()]
        logger.info(f"Loaded {len(train_dataset)} documents from {args.input_data_train}")

        # Select k-shot examples
        shots = random.choices(train_dataset, k=args.few_shot_k)
        logger.info(f"Selected ID=[{[s['id'] for s in shots]}.")

        # Format the prompt
        examples = [
            format_prompt(s, include_target=True) for s in shots
        ]
        example = "\n\nEXAMPLE:\n".join(examples)
        prompt = SYSTEM_PROMPT_ICL.format(example=example)

        # Save
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        output_file = os.path.join(args.model_path, "prompt.txt")
        with open(output_file, "w") as f:
            f.write(prompt)
        logger.info(f"Saved model to {output_file}")
        args_file = os.path.join(args.model_path, "args.json")
        with open(args_file, "w") as f:
            json.dump(args.__dict__, f, indent=2)

    elif args.mode == "test":
        output_file = os.path.join(args.output_dir, "generated.jsonl")

        if not args.simulate:
            args_file = os.path.join(args.output_dir, "args.json")
            with open(args_file, "w") as f:
                json.dump(args.__dict__, f, indent=2)

        # Load data. Skip already generated.
        already_done = set()
        if args.resume and os.path.exists(output_file):
            with jsonlines.open(output_file) as reader:
                already_done = set([d['id'] for d in reader.iter()])
            logger.info(f"Loaded {len(already_done)} completed examples from {output_file}")
        if not args.resume and os.path.exists(output_file):
            logger.info(f"Deleting existing output file {output_file}")
            os.remove(output_file)

        with jsonlines.open(args.input_data_test) as reader:
            test_dataset = [d for d in reader.iter() if d['id'] not in already_done]
        if len(test_dataset) == 0:
            logger.info(f"All examples in {args.input_data_test} are already done. Exiting.")
            sys.exit(0)
        if args.num_examples > 0 and not args.resume:
            test_dataset = test_dataset[:args.num_examples]
        logger.info(f"Loaded {len(test_dataset)} documents from {args.input_data_test}")

        # Set OpenAI's API key and API base to use vLLM's API server
        if "together" in args.server_base:
            api_key = os.environ['TOGETHER_API_KEY']
        elif "openai" in args.server_base:
            api_key = os.environ['OPENAI_API_KEY']
        else:
            api_key = ""
        client = AsyncOpenAI(
            base_url=args.server_base,
            api_key=api_key,
        )
        client_call = partial(
            client.chat.completions.create,
            model=args.model_path_base,
            max_tokens=args.max_completion_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Load model prompt
        if args.model_path:
            with open(os.join(args.model_path, "prompt.txt"), "r") as f:
                system_prompt = f.read()
        else:
            system_prompt = SYSTEM_PROMPT

        simulation_tokens = {
            "input": 0,
            "output": 0
        }
        # Convert
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path_base)
        except Exception as err:
            hf_name = args.model_path_base.replace("Meta-", "").replace("-Turbo", "")
            logger.warning(f"Couldn't find tokenizer for {args.model_path_base}. Trying again with {hf_name}.\n{err=}")
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
        for i in tqdm(range(0, len(test_dataset), args.batch_size), ncols=0):
            batch_input = test_dataset[i:i + args.batch_size]
            messages = prepare_input(batch_input, system_prompt=system_prompt)

            if args.simulate:
                tokens = sum([
                    len(tokenizer.apply_chat_template(m, tokenize=True)) for m in messages
                ])
                simulation_tokens["input"] += tokens
                simulation_tokens["output"] += (args.max_completion_tokens * len(messages))
                continue

            completions = asyncio.run(batch_response(messages, client=client_call))
            batch_output = format_output(completions, batch_input)

            # Save
            with jsonlines.open(output_file, "a") as writer:
                writer.write_all(batch_output)

            time.sleep(args.sleep)

        if args.simulate:
            logger.info(f"{simulation_tokens=}")

    logger.info(f"Done.")


if __name__ == "__main__":
    main()

