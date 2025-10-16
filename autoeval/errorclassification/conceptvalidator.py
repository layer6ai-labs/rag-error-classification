import pathlib
import re
from typing import Any, Type, List, Dict

import pandas as pd
from tqdm import tqdm

from autoeval.errorclassification.utils import openai_call
from base import artifact
from base.artifact import QueryDatasetArtifact, PickleArtifact, PandasArtifact, \
    DocumentDatasetArtifact
from base.component import Component
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".envrc")


class ConceptValidator(Component):
    def __init__(
        self,
        out_path: pathlib.Path,
        openai_model_name: str,
        ground_truth_threshold: float = 0.8,
        **kwargs,
    ):
        super().__init__(out_path, **kwargs)
        self.ground_truth_threshold = ground_truth_threshold

        self.client = OpenAI()
        self.llm_call_func = openai_call
        self.openai_model_name = openai_model_name

    @classmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        return {
            "queries": QueryDatasetArtifact,
            "llmeval_results": PickleArtifact[List[Dict[str, str]]],
            "ground_truth_chunks": PickleArtifact[Dict[str, Dict[str, float]]],
            "chunks": DocumentDatasetArtifact,
        }

    @classmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        return [
            ("validated_concepts.csv", PandasArtifact)
        ]

    def _run(self) -> list[Any]:
        queries = self.get_object("queries").queries
        llmeval_results = self.get_object("llmeval_results")
        ground_truth_chunks = self.get_object("ground_truth_chunks")
        document_dataset = self.get_object("chunks")
        if len(queries) != len(llmeval_results):
            raise ValueError(
                f"Length of queries ({len(queries)} not equal to Length of LLM"
                f" Eval results {len(llmeval_results)}"
            )
        incorrect_indices = [
            i for i, row in enumerate(llmeval_results) if row["label"] == "incorrect"
        ]
        chunk_ids_to_chunks = {chunk["chunk_id"]: chunk for chunk in document_dataset.chunks}

        dfs = []
        for i in tqdm(incorrect_indices, total=len(incorrect_indices)):
            query = queries[i]["input"]
            ground_truth_chunk = ground_truth_chunks.get(queries[i]["query_id"], {})
            ground_truth_chunk = [k for k, v in ground_truth_chunk.items() if v > self.ground_truth_threshold]
            ground_truth_chunk = [k for k in ground_truth_chunk if k != "_0"]
            for chunk_id in ground_truth_chunk:
                if chunk_id not in chunk_ids_to_chunks:
                    self.log.warning(f"Chunk {chunk_id} is not a valid chunk.")
            ground_truth_chunk_dic = {chunk_id: chunk_ids_to_chunks[chunk_id]["text"] for chunk_id in ground_truth_chunk if chunk_id in chunk_ids_to_chunks}
            if len(ground_truth_chunk_dic) == 0:
                continue
            concepts = self.extract_concept(query)
            parsed_concepts = self.parse_concept(concepts)
            valid_context = self.get_valid_context(parsed_concepts, ground_truth_chunk_dic)
            valid_context["query_id"] = queries[i]["query_id"]
            dfs.append(valid_context)

        full_df = pd.concat(dfs, axis=0, ignore_index=True)
        return [full_df]

    def extract_concept(self, query: str):
        prompt = self.get_concept_prompt(query)
        response = self.llm_call_func(self.client, messages=prompt, max_tokens=1024, model_name=self.openai_model_name)
        return response

    @staticmethod
    def parse_concept(response: str):
        responses = response.split("\n")
        responses = [r.strip() for r in responses]
        return [r for r in responses if r != ""]

    def get_valid_context(self, concepts: List[str], chunks: Dict[str, str]) -> pd.DataFrame:
        res = {}
        pattern = re.compile(r"\[([\d]+_[\d]+)\] (True|False)")
        for concept in concepts:
            prompt = self.get_context_prompt(concept, chunks)
            response = self.llm_call_func(self.client, prompt, max_tokens=1024)
            matches = [pattern.match(line.strip()) for line in response.split("\n")]
            matches = {(concept, m.group(1)): m.group(2) == "True" for m in matches if m is not None}
            res.update(matches)
        if len(res) == 0:
            return pd.DataFrame(columns=["concept", "Chunk ID", "match"])
        df = pd.Series(res, dtype=bool).reset_index()
        df.columns = ["concept", "Chunk ID", "match"]
        return df

    def get_context_prompt(self, concept: str, chunks: Dict[str, str]):
        prompt_list = [
            "You will be given a list of excerpts and a concept. "
            "Your job is to determine whether the concept given is contained in each excerpt."
            "Output a line for excerpt, output \"True\" or \"False\".",
            "",
            "Example 1:",
            "**Concepts**: Peter",
            "**Excerpts**:"
            "[45_3] John is running",
            "[45_6] Peter is walking",
            "**Answer**:",
            "[45_3] False",
            "[45_6] True",
            "",
            "Example 2:",
            "**Concepts**: running",
            "**Excerpts**:"
            "[45_3] John is running",
            "[45_6] Peter is walking",
            "**Answer**:",
            "[45_3] True",
            "[45_6] False",

            "Question:",
            f"**Concepts**: {concept}",
            "**Excerpts**:",
            *[f"[{k}] {v}" for k, v in chunks.items()],
            "**Answer**:",
        ]
        return "\n".join(prompt_list)

    @staticmethod
    def get_concept_prompt(query: str) -> str:
        examples = [
            "Question:\nHow did the senior management changes in March 2021, including the"
            " appointment of a new CEO in January 2021 and the expansion of farmland in"
            " February 2021, contribute to the enhancement of Green Fields"
            " Agriculture Ltd.'s market competitiveness?\n"
            "Concept List:\n"
            "Senior management changes\n"
            "March 2021\n"
            "new CEO\n"
            "January 2021\n"
            "Expansion of farmland\n"
            "February 2021\n"
            "Green Fields Agriculture Ltd.\n"
            "Market competitiveness\n",
            "Question:\nAccording to the judgment of Vandalia, Bayside, Court, summarize the evidence of G. Torres' crimes.\n"
            "Concept List:\n"
            "judgment\n"
            "Vandalia, Bayside, Court\n"
            "crime evidence\n"
            "G. Torres\n",
            "Question:\nCompare the debt restructuring efforts of MediaCorp in 2018 and "
            "GreenTech Solutions Inc. in 2021. Which company reduced more liabilities "
            "through debt restructuring?\n"
            "Concept List:\n"
            "Debt restructuring efforts\n"
            "MediaCorp\n"
            "2018\n"
            "GreenTech Solutions Inc.\n"
            "2021\n"
            "Reducing liabilities through debt restructuring\n"
        ]
        prompt_list = [
            "Consider the following questions. Your task is to list the set of distinct concepts, "
            "or high-level pieces of information, in the 'Context’ that could possibly influence"
            " someone’s answer to the question. Each concept should appear word-to-word in the"
            " question, or a very minor rewording. Here are three examples.",
            "",
            *[f"Example {i+1}\n{example}"for i, example in enumerate(examples)],
            "Please fill out the ‘Concept List’ for the fourth example by providing a numbered "
            "list. You should not restate the ‘Concept List’ header. You should not put dash ('-') "
            "or numbers before each item in the list.",
            "Example 4",
            query,
        ]
        return "\n".join(prompt_list)

