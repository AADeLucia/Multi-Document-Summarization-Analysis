"""
Evaluate summaries with Prometheus 2

Paper: https://aclanthology.org/2024.emnlp-main.248/
Original code: https://github.com/prometheus-eval/prometheus-eval/tree/main
"""
import os
from typing import List, Tuple, Dict, Optional, Callable, Any, Union
from functools import partial
import asyncio
from argparse import ArgumentParser
import logging
import sys
import time
import random

from openai import (
    AsyncOpenAI,
    APIError,
    BadRequestError,
    APIConnectionError
)
from openai.types.chat.chat_completion import ChatCompletion
import nest_asyncio
import jsonlines
from prometheus_eval.prompts import (
    ABSOLUTE_PROMPT,
    ABSOLUTE_PROMPT_WO_REF,
    ABS_SYSTEM_PROMPT,
    SCORE_RUBRIC_TEMPLATE,
    REL_SYSTEM_PROMPT
)
import backoff
import requests
from tqdm import tqdm

sys.path.append(f"{os.environ['MDS']}/code")
from project_utils import load_generations

# Fix async errors
nest_asyncio.apply()

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)

# Re-written version of the prompt that works better
RELATIVE_PROMPT_WO_REF = """### Task Description:
You will be given an instruction, a score rubric representing an evaluation criterion, and finally two responses from different language models for comparison.

1. Write a detailed feedback that assesses the relative quality of the responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, select which model performed better according to the score rubric.
3. The output format should look as follows: (write a feedback for criteria) [RESULT] (A or B)
4. Please do not generate any other opening, closing, and explanations.

### Instruction:
{instruction}

### Score Rubric:
{rubric}

### Response from Model A:
{response_A}

### Response from Model B:
{response_B}

---

Please compare the two responses according to the score rubric above.

### Feedback: """

# Instructions modified from the icl_experiments.py
INSTRUCTION_WO_SOURCE = """You are a multi-document summarizer. When given a list of DOCUMENTS, you provide a SUMMARY that incorporates all of the documents."""
INSTRUCTION = INSTRUCTION_WO_SOURCE + """Write a summary that incorporates all of the following documents:

### DOCUMENTS
{source}
"""

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


class PrometheusEvaluator(object):
    def __init__(
            self,
            api_base: str,
            model_name: str = "prometheus-eval/prometheus-bgb-8x7b-v2.0",
            batch_size: int = 8,
            temperature: float = 0.9,
            max_tokens: int = 1024,
            seed: int = 42,
            top_p: float = 1.0,
            repetition_penalty: float = 1.03,
            sleep_time: int = 10
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.batch_size = batch_size
        self.sleep_time = sleep_time
        self.client = AsyncOpenAI(
            base_url=self.api_base
        )
        # Generation settings from prometheus_eval.utils are top-p=0.9 and temperature=1.0
        self.client_call = partial(
            self.client.chat.completions.create,
            model=self.model_name,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty, "best_of": 1}
        )

    def format_prompt_absolute(
            self,
            generation: str,
            instruction: str,
            rubric_data: Dict[str, str],
            reference_generation: Optional[str],
    ) -> str:
        # Format input for model
        score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
        if reference_generation:
            prompt = ABSOLUTE_PROMPT.format(
                instruction=instruction,
                rubric=score_rubric,
                response=generation,
                reference_answer=reference_generation
            )
        else:
            prompt = ABSOLUTE_PROMPT_WO_REF.format(
                instruction=instruction,
                rubric=score_rubric,
                response=generation,
            )
        # Add system prompt as header because Mistral doesn't accept system prompts
        prompt = ABS_SYSTEM_PROMPT + "\n\n" + prompt
        return prompt

    def format_prompt_relative(
            self,
            generation_A: str,
            generation_B: str,
            instruction: str,
            rubric_data: Dict[str, str]
    ) -> str:
        # Format input for model
        score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
        prompt = RELATIVE_PROMPT_WO_REF.format(
            instruction=instruction,
            rubric=score_rubric,
            response_A=generation_A,
            response_B=generation_B
        )
        # Add system prompt as header because Mistral doesn't accept system prompts
        prompt = REL_SYSTEM_PROMPT + "\n\n" + prompt
        return prompt

    def absolute_grade(
            self,
            generation: List[str],
            instruction: Union[str, List[str]],
            rubric_data: Union[Dict[str, str], List[Dict[str, str]]],
            reference_generation: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        # Format prompts
        if not reference_generation:
            reference_generation = [None] * len(generation)
        if isinstance(instruction, str):
            instruction = [instruction] * len(generation)
        if isinstance(rubric_data, Dict):
            rubric_data = [rubric_data] * len(generation)
        prompts = [
            self.format_prompt_absolute(
                generation=gen,
                reference_generation=ref,
                instruction=ins,
                rubric_data=rub
            )
            for gen, ref, ins, rub in zip(generation, reference_generation, instruction, rubric_data)
        ]
        messages = [
            [
                {"role": "user", "content": prompt}
            ] for prompt in prompts
        ]
        grades = self.batch_grade(messages)
        return grades

    def relative_grade(
            self,
            generation_A: List[str],
            generation_B: List[str],
            instruction: Union[str, List[str]],
            rubric_data: Union[Dict[str, str], List[Dict[str, str]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(instruction, str):
            instruction = [instruction] * len(generation_A)
        if isinstance(rubric_data, Dict):
            rubric_data = [rubric_data] * len(generation_A)
        # Format prompts
        prompts = [
            self.format_prompt_relative(
                generation_A=ga,
                generation_B=gb,
                instruction=ins,
                rubric_data=rub
            )
            for ga, gb, ins, rub in zip(generation_A, generation_B, instruction, rubric_data)
        ]
        messages = [
            [
                {"role": "user", "content": prompt}
            ] for prompt in prompts
        ]
        grades = self.batch_grade(messages)
        return grades

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_time=60
    )
    async def _batch(self, messages: List[List[Dict[str, str]]]) -> List[ChatCompletion]:
        async_responses = [
            self.client_call(
                messages=x,
            )
            for x in messages
        ]
        return await asyncio.gather(*async_responses, return_exceptions=True)

    def batch_grade(self, messages: List[List[Dict]]) -> List[Dict[str, str]]:
        grades = []
        for i in tqdm(range(0, len(messages), self.batch_size), ncols=0):
            batch_messages = messages[i:i + self.batch_size]
            completions = asyncio.run(self._batch(batch_messages))
            grades.extend(self.format_output(completions))
            time.sleep(self.sleep_time)
        return grades

    def format_output(self, completions: List[ChatCompletion]) -> List[Dict[str, Any]]:
        output = []
        for completion in completions:
            raw_out = completion.choices[0].message.content
            try:
                feedback, score = [s.strip() for s in raw_out.split("[RESULT]")]
                output.append({
                    "feedback": feedback,
                    "score": score
                })
            except ValueError as err:
                logger.warning(f"Malformed model response:\n{err=}\n{completion=}")
                output.append({
                    "feedback": raw_out,
                    "score": None
                })
                continue
        return output


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

    parser.add_argument("--dimensions", default=DIMENSIONS, choices=DIMENSIONS, nargs="+")
    parser.add_argument("--server-base", required=True, help="API endpoint")
    parser.add_argument("--model-path-base", default="prometheus-eval/prometheus-bgb-8x7b-v2.0")
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    parser.add_argument("--max-tokens", default=1024, type=int)
    parser.add_argument("--temperature", default=0.9, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--batch-size", default=8, type=int, help="Number of simultaneous API calls")
    parser.add_argument("--sleep", default=10, type=int, help="Seconds to sleep between API calls")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if not os.path.exists(args.output_data):
        logger.error(f"Output data {args.output_data} does not exist.")
        exit(0)

    # 1. Load the data
    full_dataset = [d for d in load_generations(args.input_data, args.output_data)]
    if args.n_examples is not None and args.n_examples < len(full_dataset):
        sampled_examples = random.sample(full_dataset, args.n_examples)
        logger.info(f"Sampled {len(sampled_examples)} examples.")
    else:
        sampled_examples = full_dataset

    dataset = {"source": [], "predictions": [], "reference": []}
    for example in sampled_examples:
        dataset["source"].append(example["source"])
        dataset["predictions"].append(example["generated"])
        dataset["reference"].append(example["target"])

    # 2. Initialize evaluator
    evaluator = PrometheusEvaluator(
        model_name=args.model_path_base,
        api_base=args.server_base,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        sleep_time=args.sleep
    )

    # 3. Run evaluation
    include_source = True
    include_reference = True
    for dimension in args.dimensions:
        output_file = f"{args.output_dir}/prometheus_eval{'_source' if include_source else ''}{'_ref' if include_reference else ''}_{dimension}.jsonl"
        if os.path.exists(output_file):
            logger.warning(f"{output_file} exists. Skipping.")
            continue

        if include_source:
            instruction = []
            for source_docs in dataset["source"]:
                docs = "\n\nDocument: ".join(source_docs)
                docs = "Document: " + docs
                ins = INSTRUCTION.format(source=docs)
                instruction.append(ins)
        else:
            instruction = INSTRUCTION_WO_SOURCE
        results = evaluator.absolute_grade(
            generation=dataset["predictions"],
            reference_generation=dataset["reference"] if include_reference else None,
            instruction=instruction,
            rubric_data=RUBRICS[dimension]
        )

        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with jsonlines.open(output_file, 'w') as f:
            for orig, pred, ref, result in zip(sampled_examples, dataset["predictions"], dataset["reference"], results):
                f.write({
                    "id": orig["id"],
                    "prediction": pred,
                    "reference": ref,
                    "score": result["score"],
                    "feedback": result["feedback"]
                })
        logger.info(f"Wrote output to {output_file}")


if __name__ == "__main__":
    main()

