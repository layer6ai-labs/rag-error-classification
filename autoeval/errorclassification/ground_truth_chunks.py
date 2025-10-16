import pathlib
from typing import Any, Type, Dict, List

import numpy as np
import tqdm

from autoeval.errorclassification.utils import openai_call
from local_datasets.document_datasets.clapnq import ClapnqDoc
from base import artifact
from base.artifact import PickleArtifact, QueryDatasetArtifact, DocumentDatasetArtifact
from base.component import Component
from openai import OpenAI
from dotenv import load_dotenv


class GroundTruthChunks(Component):

    BACKGROUND = [
        "You are an expert evaluator for Retrieval Augmented Generation (RAG) systems.",
        "You will help identify chunking-related errors in RAG systems by analyzing",
        "the relationship between queries, ground truth answers, and document chunks.",
    ]

    def __init__(
        self,
        out_path: pathlib.Path,
        openai_model_name: str,
        num_rounds: int = 10,
        **kwargs
    ):
        super().__init__(out_path, **kwargs)
        self.model_name = openai_model_name
        self.num_rounds = num_rounds

        load_dotenv()
        self.client = OpenAI()

    def get_prompt(
        self,
        query: str,
        ground_truth: str,
        ground_truth_chunks: Dict[str, str],
    ) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": "\n".join(self.BACKGROUND) +
                                          "\nYour task is to identify which chunks from the ground truth document are relevant to answering the query."}
        ]

        prompt_list = [
            "## Task: Find Relevant Chunks",
            "Given a query and the ground truth answer, identify which chunks from the ground truth document",
            "are relevant for answering the query. Focus on finding chunks that contain information",
            "necessary to answer the query completely and accurately.",
            "",
            "## Instructions",
            "1. Review ALL available chunks from the ground truth document",
            "2. Identify ALL chunks contain information relevant to answering the query",
            "3. Consider both direct and indirect relevance to the query",
            "4. Select chunks that together provide sufficient information to answer the query",
            "",
            "## Context",
            f"**Query**: {query}",
            f"**Ground Truth Answer**: {ground_truth}",
            "",
            "**Available Chunks from Ground Truth Document**:",
            *[f"[{chunk_id}] {chunk}\n" for chunk_id, chunk in sorted(ground_truth_chunks.items())],
            "",
            "## Output Format",
            "Provide your answer in the following format:",
            "Relevant Chunks: [45_1, 45_4, 45_10]",
        ]

        messages.append({"role": "user", "content": "\n".join(prompt_list)})
        return messages

    @classmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        return {
            "queries": QueryDatasetArtifact,
            "chunks": DocumentDatasetArtifact,
            "llmeval_results": PickleArtifact[List[Dict[str, str]]]
        }

    @classmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        return [
            ("ground_truth_chunks.pkl", PickleArtifact[Dict[str, Dict[str, float]]]),
        ]

    def _run(self) -> List[Any]:
        queries = self.get_object("queries").queries
        llmeval_results = self.get_object("llmeval_results")
        document_dataset = self.get_object("chunks")
        ground_truth_chunks_dict = {}
        for chunk in document_dataset.chunks:
            doc_id = chunk["doc_id"]
            ground_truth_chunks_dict.setdefault(doc_id, []).append(chunk)

        if len(queries) != len(llmeval_results):
            raise ValueError(f"Length of queries ({len(queries)} not equal to Length of LLM"
                             f" Eval results {len(llmeval_results)}")
        incorrect_indices = [i for i, row in enumerate(llmeval_results) if row["label"] == "incorrect"]

        results = {}
        for i in tqdm.tqdm(incorrect_indices, total=len(incorrect_indices)):
            query = queries[i]["input"]
            if isinstance(document_dataset, ClapnqDoc):
                # already contain GT chunks
                results[queries[i]["query_id"]] = {f"{k}_0": 1.0 for k in queries[i]["doc_ids"]}
                continue
            ground_truth = queries[i]["answer"]
            ground_truth_docs = queries[i]["doc_ids"]
            responses = []
            for doc_id in ground_truth_docs:
                ground_truth_chunks = ground_truth_chunks_dict[doc_id]
                ground_truth_chunks = {chunk["chunk_id"]: chunk["text"] for chunk in ground_truth_chunks}
                prompt = self.get_prompt(query, ground_truth, ground_truth_chunks)
                for _ in range(self.num_rounds):
                    response = openai_call(self.client, prompt, max_tokens=4096, model_name=self.model_name)
                    response = self._parse_response(response)
                    responses.extend(response)
            count_dict = dict(zip(*np.unique(responses, return_counts=True)))
            count_dict = {k: float(v / self.num_rounds) for k, v in count_dict.items()}

            self.log.info(f"{i}: {count_dict}")
            results[queries[i]["query_id"]] = count_dict
        return [results]

    @staticmethod
    def _parse_response(response: str) -> List[str]:
        responses = response.split("[", maxsplit=1)
        if len(responses) == 2:
            response = responses[1]
        li = [w.strip() for w in response.strip("]").split(",")]
        return [r for r in li if r != ""]

