import abc
import json
import logging
import pathlib
from typing import Any, List, Dict

from tqdm import tqdm


class QueryDataset(abc.ABC):
    """
    The list of query is a list of dict, containing at least the following keys.

    "query_id": int/str - The Query ID
    "input": str - The Query text
    "unanswerable": bool - Whether the query is answerable or not.
    "answer": str - The answer (only 1 GT answer so far)
    "doc_ids": list[int] - The GT citation doc IDs. As there could be more than 1 citations, it is
    a list
    "texts": list[str] - The GT citation texts. As there could be more than 1 citations, it is a
    list
    """

    def __init__(self, queries_file: pathlib.Path, queries: List[Dict[str, Any]] | None = None):
        self.queries_file = queries_file
        self.queries = [] if queries is None else queries
        self.log = logging.getLogger(self.__class__.__name__)
        self.load()
        self.texts = [q["input"] for q in self.queries]
        self.index_lookup = None

    @staticmethod
    def check(input_filepath: pathlib.Path, ext: str):
        if not input_filepath.exists():
            raise ValueError(f"File {input_filepath} not present! Please provide accurate file.")
        if input_filepath.suffix != ext:
            raise ValueError(f"File {input_filepath} must be present with extension {ext}")

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError("Subclasses should implement this method")

    def __getitem__(self, idx):
        if isinstance(idx, (int, str)):
            idx = [idx]
        if isinstance(idx, list):
            if isinstance(idx[0], int):
                # slice by indices
                queries = [self.queries[i] for i in idx]
            elif isinstance(idx[0], str):
                # slice by query ids
                if self.index_lookup is None:
                    self.index_lookup = {q["query_id"]: q for q in self.queries}
                queries = [self.index_lookup[query_id] for query_id in idx]
            else:
                raise TypeError(f"Slice indices is a list of {type(idx[0])}")
            return self.__class__(self.queries_file, queries)

        if isinstance(idx, slice):
            return self.__class__(self.queries_file, self.queries[idx])

        raise TypeError(f"Unknown index type {type(idx)}.")

    def __len__(self):
        return len(self.queries)

    @property
    def query_text(self) -> list[str]:
        return self.texts

    @property
    def answers(self) -> list[str]:
        return [q["answer"] for q in self.queries]


class DragonballQueryDataset(QueryDataset):
    """
    Extra schema:

    "keypoint": list[str] - The keypoints of the citations. The length may be different from the
    length of the citations.
    "type": str - The type of the Questions (Multihop, Factual, etc.)
    """

    def load(self):
        if not len(self.queries):
            self.check(input_filepath=self.queries_file, ext=".jsonl")

            self.log.info("Loading queries...")

            num_lines = sum(1 for i in open(self.queries_file, "rb"))
            with open(self.queries_file, encoding="utf8") as fIn:
                for line in tqdm(fIn, total=num_lines):
                    line = json.loads(line)
                    output = line["output"]
                    provenance = output["provenance"][0]
                    q = {
                        "query_id": str(line["id"]),
                        "input": line["input"],
                        "unanswerable": line["metadata"]["query_type"]
                        == "Irrelevant Unsolvable Question",
                        "answer": output["answers"][0],
                        "doc_ids": provenance["id"],
                        "texts": provenance["text"],
                        "keypoint": output["keypoints"],
                        "type": line["metadata"]["query_type"],
                    }
                    self.queries.append(q)

            self.log.info(f"Finished Loading {len(self.queries):} Queries")

        return self.queries


class ClapNQQueryDataset(QueryDataset):
    """
    Extra schema:

    "original_nq_id": str - The original NQ ID for the query.
    """

    def load(self):
        if not len(self.queries):
            self.check(input_filepath=self.queries_file, ext=".jsonl")

            self.log.info("Loading queries...")

            num_lines = sum(1 for i in open(self.queries_file, "rb"))
            with open(self.queries_file, encoding="utf8") as fIn:
                for line in tqdm(fIn, total=num_lines):
                    line = json.loads(line)
                    output = line["output"]
                    if len(output["provenance"][0]) > 0:
                        provenance = output["provenance"][0]
                    else:
                        provenance = {"id": "", "text": ""}
                    q = {
                        "query_id": str(line["id"]),
                        "input": line["input"],
                        "unanswerable": line["unanswerable"],
                        "answer": output["answer"],
                        "doc_ids": [provenance["id"]],
                        "texts": [provenance["text"]],
                        "original_nq_id": line["original_nq_id"],
                    }
                    self.queries.append(q)

            self.log.info(f"Finished Loading {len(self.queries):} Queries")

        return self.queries
