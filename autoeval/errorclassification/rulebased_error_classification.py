import pathlib
from typing import Any, Type, Dict, List

import numpy as np
import tqdm

from autoeval.errorclassification.error_categories import RagFailureStage
from autoeval.errorclassification.error_prompts import PromptInputOutput, MultiStagePromptConstructor
from autoeval.errorclassification.utils import openai_call
from baseconfig import LocalRagConfig
from base import artifact
from base.artifact import (
    PickleArtifact,
    QueryDatasetArtifact,
    NumpyQueryArtifact,
    DocumentDatasetArtifact, PandasArtifact,
)
from base.component import Component
from openai import OpenAI
from dotenv import load_dotenv


class RuleBasedErrorClassification(Component):
    def __init__(
        self,
        out_path: pathlib.Path,
        num_rounds: int,
        rerank_topk: int,
        openai_model_name: str,
        concept_covered_threshold: float = 0.8,
        rerank_covered_threshold: float = 0.5,
        ground_truth_chunk_threshold: float = 0.8,
        **kwargs,
    ):
        super().__init__(out_path, **kwargs)
        self.num_rounds = num_rounds
        self.rerank_topk = rerank_topk
        self.model_name = openai_model_name
        self.prompt_constructor = MultiStagePromptConstructor(reranked_top_k=rerank_topk)
        self.concept_covered_threshold = concept_covered_threshold
        self.rerank_covered_threshold = rerank_covered_threshold
        self.ground_truth_chunk_threshold = ground_truth_chunk_threshold

        load_dotenv()
        self.client = OpenAI()
        self.llm_call_func = openai_call

    @classmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        return {
            "queries": QueryDatasetArtifact,
            "processed_output": PickleArtifact[List[Dict[str, Any]]],
            "chunks": DocumentDatasetArtifact,
            "retrieved_chunks": NumpyQueryArtifact,
            "reranked_order": NumpyQueryArtifact,
            "llmeval_results": PickleArtifact[List[Dict[str, str]]],
            "ground_truth_chunks": PickleArtifact[Dict[str, Dict[str, float]]],
            "validated_concepts": PandasArtifact,
        }

    @classmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        return [
            ("error_classes_rule_based.pkl", PickleArtifact[Dict[str, Dict[str, float]]]),
        ]

    def _run(self) -> List[Any]:
        queries = self.get_object("queries").queries
        processed_output = self.get_object("processed_output")
        retrieved_chunks = self.get_object("retrieved_chunks")
        reranked_orders = self.get_object("reranked_order")
        llmeval_results = self.get_object("llmeval_results")
        document_dataset = self.get_object("chunks")
        ground_truth_chunks = self.get_object("ground_truth_chunks")
        validated_concepts = self.get_object("validated_concepts")
        chunk_ids_to_chunks = {chunk["chunk_id"]: chunk for chunk in document_dataset.chunks}
        if len(queries) != len(llmeval_results):
            raise ValueError(
                f"Length of queries ({len(queries)} not equal to Length of LLM"
                f" Eval results {len(llmeval_results)}"
            )
        incorrect_indices = [
            i for i, row in enumerate(llmeval_results) if row["label"] == "incorrect"
        ]

        results = {}
        for i in tqdm.tqdm(incorrect_indices, total=len(incorrect_indices)):
            query = queries[i]["input"]
            query_id = queries[i]["query_id"]
            search_chunk_ids = [chunk["chunk_id"] for chunk in retrieved_chunks[i]]
            reranked_chunk_ids = [
                search_chunk_ids[reranked_orders[i, j]] for j in range(self.rerank_topk)
            ]
            ground_truth_chunk = ground_truth_chunks.get(query_id, {})
            ground_truth_chunk_ids = [k for k, v in ground_truth_chunk.items() if v > self.ground_truth_chunk_threshold]
            ground_truth_chunk_ids = [k for k in ground_truth_chunk_ids if k != "_0"]
            ground_truth_chunk_ids = [k for k in ground_truth_chunk_ids if k in chunk_ids_to_chunks.keys()]

            rerank_contain = [chunk for chunk in reranked_chunk_ids if chunk in ground_truth_chunk_ids]
            exclude = [chunk for chunk in search_chunk_ids if chunk not in reranked_chunk_ids]
            exclude_contain = [chunk for chunk in exclude if chunk in ground_truth_chunk_ids]

            if len(ground_truth_chunk_ids) == 0:
                suggested_stage = RagFailureStage.ANSWER_GENERATION
            else:
                if len(rerank_contain) / len(ground_truth_chunk_ids) > self.rerank_covered_threshold:
                    suggested_stage = RagFailureStage.ANSWER_GENERATION
                else:
                    if len(exclude_contain) / len(ground_truth_chunk_ids) > 0:
                        suggested_stage = RagFailureStage.RERANKING_FAILURES
                    else:
                        valid_concepts = validated_concepts[validated_concepts["query_id"] == int(query_id)]
                        pivoted_valid_concepts = valid_concepts.drop("query_id", axis=1).pivot_table(
                            columns="concept", index="Chunk ID", values="match", aggfunc="any", fill_value=False
                        )
                        concepts_covered = np.any(pivoted_valid_concepts, axis=0)
                        if np.mean(concepts_covered) < self.concept_covered_threshold:
                            suggested_stage = RagFailureStage.DOCUMENT_PREPROCESSING
                        else:
                            suggested_stage = RagFailureStage.RETRIEVAL_FAILURES

            rag_answer = processed_output[i]["prediction_text"]
            ground_truth = queries[i]["answer"]
            ground_truth_docs = queries[i]["doc_ids"]
            relevant_chunks_ids = set(search_chunk_ids + ground_truth_chunk_ids)
            relevant_chunks = {
                chunk_id: chunk_ids_to_chunks[chunk_id]["text"] for chunk_id in relevant_chunks_ids
            }

            responses = []
            for _ in range(self.num_rounds):
                input_dict = {
                    PromptInputOutput.QUERY: query,
                    PromptInputOutput.RELEVANT_CHUNKS: relevant_chunks,
                    PromptInputOutput.SEARCH_CHUNK_IDS: search_chunk_ids,
                    PromptInputOutput.RERANKED_CHUNK_IDS: reranked_chunk_ids,
                    PromptInputOutput.RAG_ANSWER: rag_answer,
                    PromptInputOutput.GROUND_TRUTH: ground_truth,
                    PromptInputOutput.GROUND_TRUTH_DOC_IDS: ground_truth_docs,
                    PromptInputOutput.GROUND_TRUTH_CHUNK_IDS: ground_truth_chunk_ids,
                    PromptInputOutput.SUGGESTED_STAGE: suggested_stage,
                }
                messages = self.prompt_constructor.construct_prompt(input_dict)
                response = self.llm_call_func(self.client, messages, max_tokens=10240, model_name=self.model_name)
                responses.append(response)
            # self.log.info(responses)
            results[query_id] = responses
        return [results]

    @classmethod
    def from_config(cls, out_path: pathlib.Path, config: LocalRagConfig, **kwargs):
        # noinspection PyTypeChecker
        return cls(
            out_path=out_path,
            num_rounds=config.error_classification.num_rounds,
            rerank_topk=config.rerank.rerank_topk,
            openai_model_name=config.error_classification.model_name,
            concept_covered_threshold=config.error_classification.concept_covered_threshold,
            rerank_covered_threshold=config.error_classification.rerank_covered_threshold,
            ground_truth_chunk_threshold=config.error_classification.ground_truth_chunk_threshold,
            **kwargs,
        )
