import json
import logging
import pathlib
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, List, Dict

from tqdm.autonotebook import tqdm


class DocumentDataset(ABC):
    def __init__(
        self,
        corpus_file: pathlib.Path,
        load_docs: bool = True,
        load_semantic_chunks: bool = False,
        chinese: bool = False,
        corpus: Dict[str, dict[str, Any]] | None = None,
        chunks: List[Dict[str, Any]] | None = None,
    ):
        self.corpus_file = corpus_file
        self.corpus = {} if corpus is None else corpus
        self.chunks = [] if chunks is None else chunks
        self.chinese = chinese
        self.log = logging.getLogger(self.__class__.__name__)
        self.load(load_docs, load_semantic_chunks)
        self.lookup = None
        self._lookup_index = None

    @staticmethod
    def check(input_filepath: pathlib.Path, ext: str):
        if not input_filepath.exists():
            raise ValueError(f"File {input_filepath} not present! Please provide accurate file.")

        if input_filepath.suffix != ext:
            raise ValueError(f"File {input_filepath} must be present with extension {ext}")

    def load(
        self, load_docs: bool = True, load_semantic_chunks: bool = False
    ) -> dict[Any, dict[str, Any]]:
        if not (load_docs or load_semantic_chunks):
            raise ValueError(
                "At least one of the flags load_docs or load_semantic_chunks should be True"
            )

        if not len(self.corpus):
            self.check(input_filepath=self.corpus_file, ext=".jsonl")

            self.log.info("Loading Corpus...")

            num_lines = sum(1 for _ in open(self.corpus_file, "rb"))
            with open(self.corpus_file, encoding="utf8") as fIn:
                desc = ""
                if load_docs:
                    desc += "Loading Documents."
                if load_semantic_chunks:
                    desc += "Loading Chunks."

                for line in tqdm(fIn, total=num_lines, desc=desc):
                    parsed_line: dict[str, Any] = json.loads(line)
                    self.load_one_doc(parsed_line, load_docs, load_semantic_chunks)
            self.log.info(f"Finished Loading {len(self.corpus):,} Documents")

        return self.corpus

    def get_documents(self) -> Sequence[Dict[str, Any]]:
        return self.corpus.values()

    @abstractmethod
    def load_one_doc(
        self, line_json: dict[str, Any], load_docs: bool = True, load_semantic_chunks: bool = False,
    ):
        """
        Corpus:{id, document_record}. A document record contains:
        1. id: unique identifier for the document
        2. title: title of the document
        3. text: text of the document
        4. meta_data: additional information about the document, optional

        Chunk:{id, chunk_record}. A chunk record contains:
        1. id: unique identifier for the chunk (doc_id_chunk_id) e.g. 1_0
        2. doc_id: unique identifier for the document
        3. chunk_id: unique identifier for the chunk
        4. title: title of the document
        5. text: text of the chunk
        """

    @property
    def chunk_texts(self) -> List[str]:
        return [c["text"] for c in self.chunks]

    @property
    def chunk_ids(self) -> List[str]:
        return [c["chunk_id"] for c in self.chunks]

    @property
    def doc_ids(self) -> List[str]:
        return [c["doc_id"] for c in self.chunks]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        if isinstance(idx, list):
            chunks = [self.chunks[i] for i in idx]
            return self.__class__(self.corpus_file, True, True, self.chinese, self.corpus, chunks)

        if isinstance(idx, slice):
            return self.__class__(self.corpus_file, True, True, self.chinese, self.corpus, self.chunks[idx])

        raise TypeError(f"Unknown index type {type(idx)}.")

    def __len__(self):
        return len(self.corpus)

    def chunk_lookup(self, chunk_id: str) -> Dict[str, Any] | None:
        """
        Returns the chunk given the chunk ID.
        """
        if self.lookup is None:
            self.lookup = {chunk["chunk_id"]: chunk for chunk in self.chunks}

        return self.lookup.get(chunk_id)

    def index_lookup(self, chunk_id: str) -> int | None:
        """
        Returns the chunk index given the chunk ID.
        """
        if self._lookup_index is None:
            self._lookup_index = {chunk["chunk_id"]: i for i, chunk in enumerate(self.chunks)}

        return self._lookup_index.get(chunk_id)
