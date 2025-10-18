import json

import tqdm
from dotenv import load_dotenv
from typing import Optional, TypeVar, Callable, Any, Type, Dict, List
import time

from openai import OpenAI

from autoeval.llm_evaluator.prompts import get_evaluation_prompt
from local_datasets.query_datasets.query_dataset import QueryDataset
from base import artifact
from base.component import Component

load_dotenv(".envrc")

openai_client = OpenAI()

T = TypeVar("T")


def try_until_success(
    func: Callable[..., T],
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    base_delay: float = 1.0,
) -> T:
    """
    Repeatedly tries to execute a function that may raise exceptions until it succeeds.
    Uses exponential backoff between retries.

    Args:
        func: The function to execute
        max_retries: Maximum number of retry attempts before giving up
        exceptions: Tuple of exception types to catch and retry on
        base_delay: Initial delay in seconds between retries, doubles after each attempt

    Returns:
        The result of the successful function execution

    Raises:
        The last exception encountered if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise last_exception
            # Calculate delay with exponential backoff: base_delay * 2^attempt
            delay = base_delay * (2**attempt)
            time.sleep(delay)

    # This should never be reached due to the raise in the loop
    raise last_exception if last_exception else RuntimeError(
        "Unexpected error in try_until_success"
    )


def evaluate_answer(
    question: str,
    ground_truth: str,
    proposed_answer: str,
    ground_truth_citations: Optional[list[str]] = None,
) -> dict:
    """
    Evaluates if a proposed answer matches the ground truth answer for a given question.

    Args:
        question (str): The original question being asked
        ground_truth (str): The correct answer to the question
        proposed_answer (str): The answer to evaluate
        ground_truth_citations (Optional[list[str]], optional): Supporting citations for the ground truth. Defaults to None.

    Returns:
        dict: Evaluation result with format:
            {
                "label": "correct" | "possible_correct" | "incorrect" | "abstain",
                "reasoning": str  # Brief explanation of the evaluation
            }
    """
    prompt = get_evaluation_prompt(
        question=question,
        ground_truth=ground_truth,
        proposed_answer=proposed_answer,
        ground_truth_citations=ground_truth_citations,
    )

    def make_api_call():
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response

    response = try_until_success(make_api_call)
    payload = json.loads(response.choices[0].message.content)
    return payload


class LlmEvaluator(Component):
    @classmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        return {
            "queries": artifact.QueryDatasetArtifact,
            "processed_output": artifact.PickleArtifact[List[Dict[str, Any]]],
        }

    @classmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        return [
            ("llm_evaluate_output.pkl", artifact.PickleArtifact[List[Dict[str, str]]]),
        ]

    def _run(self) -> list[Any]:
        queries: QueryDataset = self.get_object("queries")
        processed_output = self.get_object("processed_output")

        all_outputs = []
        for out in tqdm.tqdm(processed_output, total=len(processed_output)):
            query_id = out["id"]
            query = queries[query_id].queries[0]
            question = query["input"]
            unanswerable = query["unanswerable"]
            ground_truth = query["answer"]
            ground_truth_citations = query["texts"]
            proposed_answer = out["prediction_text"]

            if unanswerable:
                ground_truth = "I don't know."

            llm_out = evaluate_answer(
                question, ground_truth, proposed_answer, ground_truth_citations
            )
            all_outputs.append(llm_out)

        return [all_outputs]
