import enum
import random
from typing import Dict, Any, List

from autoeval.errorclassification.error_categories import RagFailureStage, ERROR_CATEGORIES


class PromptInputOutput(enum.Enum):
    QUERY = "query"
    RELEVANT_CHUNKS = "relevant_chunks"
    SEARCH_CHUNK_IDS = "search_chunk_ids"
    RERANKED_CHUNK_IDS = "reranked_chunk_ids"
    RAG_ANSWER = "rag_answer"
    GROUND_TRUTH = "ground_truth"
    GROUND_TRUTH_DOC_IDS = "ground_truth_docs"
    GROUND_TRUTH_CHUNK_IDS = "ground_truth_chunks"
    FIRST_STAGE_RESULT = "first_stage_results"
    OUTPUT = "output"
    SUGGESTED_STAGE = "suggested_stage"
    OBSERVATION = "observation"
    CHUNKING_SCORES = "chunking_scores"
    RETRIEVAL_SCORES = "retrieval_scores"
    RERANKING_SCORES = "reranking_scores"
    GENERATION_SCORES = "generation_scores"


class MultiStagePromptConstructor:

    BACKGROUND = [
        "You are an expert Retrieval Augmented Generation (RAG) system evaluator.",
        "Background:",
        "Given a query, and a list of documents possibly containing"
        " the answer to the query, a RAG model has tried to answer the query using "
        "the following 4 steps.",
        "  - Chunking: Each document in the document list is split into smaller chunks.",
        "  - Retrieval: A specified number of chunks closest to the query will be retrieved.",
        "  - Reranking: The retrieved chunks are ranked for the second time, according to"
        " how relevant the chunks are to the query. Then, only a specified number of"
        " reranked chunks are retained.",
        "  - Generation: The query and reranked chunks are passed to the generator LLM to"
        " generate an answer.",
    ]

    def __init__(self, reranked_top_k: int):
        super().__init__()
        self.reranked_top_k = reranked_top_k

    @staticmethod
    def get_stage_errors(stage: RagFailureStage):
        return [category for category in ERROR_CATEGORIES if category.stage == stage]

    @staticmethod
    def format_chunk_id(chunk_id):
        doc_id, chunk_num = chunk_id.strip().rsplit("_", maxsplit=1)
        return f"Document {doc_id} Chunk {chunk_num}"

    def construct_chunking_prompt(self, input_dict: Dict[PromptInputOutput, Any]) -> List[Dict[str, str]]:
        query = input_dict[PromptInputOutput.QUERY]
        ground_truth = input_dict[PromptInputOutput.GROUND_TRUTH]
        ground_truth_chunk_ids = input_dict[PromptInputOutput.GROUND_TRUTH_CHUNK_IDS]
        relevant_chunks = input_dict[PromptInputOutput.RELEVANT_CHUNKS]
        ground_truth_chunks = [
            f"[{self.format_chunk_id(chunk_id)}]\n{relevant_chunks[chunk_id]}\n"
            for chunk_id in ground_truth_chunk_ids
        ]
        random.shuffle(ground_truth_chunks)
        errors = self.get_stage_errors(RagFailureStage.DOCUMENT_PREPROCESSING)
        prompt_list = [
            "You will be given a query and the ground truth answer, as well as all the "
            "chunks that belong to the document containing the ground truth answer. "
            "As described in the background, each chunk will be seen as independent text blocks"
            " by the RAG model. Given there is an error in the chunking step, "
            "your job is to determine the best description of the error from the list below.",
            *[f"- {error.name}: {error.description}" for error in errors],
            "",
            "Context:",
            f"**Query**: {query}",
            f"**Ground Truth**: {ground_truth}",
            f"**Document Chunks**:",
            "\n".join(ground_truth_chunks),
            "",
            "**Output format**:",
            "Your answer should only contain one of the followings,"
            ", ".join([error.name for error in errors]),
        ]
        return [
            {"role": "system", "content": "\n".join(self.BACKGROUND)},
            {"role": "user", "content": "\n".join(prompt_list)},
        ]

    def construct_retrieval_prompt(self, input_dict: Dict[PromptInputOutput, Any]) -> List[Dict[str, str]]:
        errors = self.get_stage_errors(RagFailureStage.RETRIEVAL_FAILURES)
        query = input_dict[PromptInputOutput.QUERY]
        ground_truth = input_dict[PromptInputOutput.GROUND_TRUTH]
        ground_truth_doc_ids = input_dict[PromptInputOutput.GROUND_TRUTH_DOC_IDS]
        ground_truth_doc_ids = ", ".join([str(i) for i in ground_truth_doc_ids])
        relevant_chunks = input_dict[PromptInputOutput.RELEVANT_CHUNKS]
        search_chunk_ids = input_dict[PromptInputOutput.SEARCH_CHUNK_IDS]
        search_chunks = [
            f"[{self.format_chunk_id(chunk_id)}]\n{relevant_chunks[chunk_id]}\n"
            for chunk_id in search_chunk_ids
        ]

        prompt_list = [
            "You will be given a query and the ground truth answer. You will also be given IDs"
            " of the documents containing the ground truth and a selection of document chunks"
            " retrieved. "
            "Given there is an error in the retrieval step,"
            "your job is to determine the best description of the error from the list below.",
            *[f"- {error.name}: {error.description}" for error in errors],
            "Context:",
            f"**Query**: {query}",
            f"**Ground Truth**: {ground_truth}",
            f"**Ground Truth Document ID**: {ground_truth_doc_ids}",
            f"**Retrieved Chunks**:",
            "\n".join(search_chunks),
            "",
            "**Output format**:",
            "Your answer should only contain one of the followings:",
            ", ".join([error.name for error in errors]),
        ]
        return [
            {"role": "system", "content": "\n".join(self.BACKGROUND)},
            {"role": "user", "content": "\n".join(prompt_list)},
        ]

    def construct_reranked_prompt(self, input_dict: Dict[PromptInputOutput, Any]) -> List[Dict[str, str]]:
        errors = self.get_stage_errors(RagFailureStage.RERANKING_FAILURES)
        query = input_dict[PromptInputOutput.QUERY]
        ground_truth = input_dict[PromptInputOutput.GROUND_TRUTH]
        ground_truth_doc_ids = input_dict[PromptInputOutput.GROUND_TRUTH_DOC_IDS]
        ground_truth_doc_ids = ", ".join([str(i) for i in ground_truth_doc_ids])
        relevant_chunks = input_dict[PromptInputOutput.RELEVANT_CHUNKS]
        search_chunk_ids = input_dict[PromptInputOutput.SEARCH_CHUNK_IDS]
        reranked_chunk_ids = input_dict[PromptInputOutput.RERANKED_CHUNK_IDS]
        reranked_chunks = []
        for i, chunk_id in enumerate(reranked_chunk_ids):
            search_chunk_position = search_chunk_ids.index(chunk_id) + 1
            reranked_chunks.append(
                f"[{self.format_chunk_id(chunk_id)}] Retrieved #{search_chunk_position}, Reranked #{i + 1}\n{relevant_chunks[chunk_id]}\n"
            )
        for i, chunk_id in enumerate(search_chunk_ids):
            if chunk_id in reranked_chunk_ids:
                continue
            reranked_chunks.append(
                f"[{self.format_chunk_id(chunk_id)}] Retrieved #{i + 1}, Not Reranked\n{relevant_chunks[chunk_id]}\n"
            )

        prompt_list = [
            "You will be given a query and the ground truth answer. You will also be given a"
            " selection of document chunks retrieved."
            f" The retrieved chunks will be reranked so that only {self.reranked_top_k} chunks"
            " are further selected. Given there is an error in the reranking step,"
            "your job is to determine the best description of the error from the list below.",
            *[f"- {error.name}: {error.description}" for error in errors],
            "Context:",
            f"**Query**: {query}",
            f"**Ground Truth**: {ground_truth}",
            f"**Ground Truth Document ID**: {ground_truth_doc_ids}",
            f"**Reranked Chunks**:",
            "\n".join(reranked_chunks),
            "",
            "**Output format**:",
            "Your answer should only contain one of the followings:",
            ", ".join([error.name for error in errors]),
        ]
        return [
            {"role": "system", "content": "\n".join(self.BACKGROUND)},
            {"role": "user", "content": "\n".join(prompt_list)},
        ]

    def construct_generator_prompt(self, input_dict: Dict[PromptInputOutput, Any]) -> List[Dict[str, str]]:
        errors = self.get_stage_errors(RagFailureStage.ANSWER_GENERATION)
        query = input_dict[PromptInputOutput.QUERY]
        ground_truth = input_dict[PromptInputOutput.GROUND_TRUTH]
        incorrect_rag_answer = input_dict[PromptInputOutput.RAG_ANSWER]
        relevant_chunks = input_dict[PromptInputOutput.RELEVANT_CHUNKS]
        reranked_chunk_ids = input_dict[PromptInputOutput.RERANKED_CHUNK_IDS]
        reranked_chunks = [
            f"[{self.format_chunk_id(chunk_id)}]\n{relevant_chunks[chunk_id]}\n"
            for chunk_id in reranked_chunk_ids
        ]
        prompt_list = [
            "You will be given a query, the ground truth answer and an incorrect answer to"
            " the query generated by the RAG model."
            f" You will also be given {self.reranked_top_k} document chunks. "
            " Your job is to determine the reason why the model outputs the incorrect answer, "
            "from the list below. ",
            *[f"- {error.name}: {error.description}" for error in errors],
            "Context:",
            f"**Query**: {query}",
            f"**Ground Truth**: {ground_truth}",
            f"**Incorrect Answer**: {incorrect_rag_answer}",
            f"**Reranked Chunks**:",
            "\n".join(reranked_chunks),
            "",
            "**Output format**:",
            "Your answer should only contain one of the followings:",
            ", ".join([error.name for error in errors]),
        ]
        return [
            {"role": "system", "content": "\n".join(self.BACKGROUND)},
            {"role": "user", "content": "\n".join(prompt_list)},
        ]

    def construct_prompt(self, input_dict) -> List[Dict[str, str]]:
        suggested_stage = input_dict[PromptInputOutput.SUGGESTED_STAGE]
        if suggested_stage == RagFailureStage.DOCUMENT_PREPROCESSING:
            return self.construct_chunking_prompt(input_dict)
        if suggested_stage == RagFailureStage.RETRIEVAL_FAILURES:
            return self.construct_retrieval_prompt(input_dict)
        if suggested_stage == RagFailureStage.RERANKING_FAILURES:
            return self.construct_reranked_prompt(input_dict)
        if suggested_stage == RagFailureStage.ANSWER_GENERATION:
            return self.construct_generator_prompt(input_dict)
        raise ValueError(f"Invalid suggested stage {suggested_stage}")

    def name(self):
        return "multistagev2"

    def extract_answer(self, output: str) -> Dict[PromptInputOutput, Any]:
        return {PromptInputOutput.OUTPUT: output}
