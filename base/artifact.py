from __future__ import annotations

import abc
import json
import pathlib
import pickle as pkl
from typing import Generic, TypeVar, List

import numpy as np
import pandas as pd

from baseconfig import LocalRagConfig
from local_datasets.document_datasets.dataset_helper import get_document_dataset
from local_datasets.document_datasets.document_dataset import DocumentDataset
from local_datasets.document_datasets.utils import create_chunks
from local_datasets.query_datasets.dataset_helper import get_query_dataset
from local_datasets.query_datasets.query_dataset import QueryDataset

T = TypeVar("T")
U = TypeVar("U")


class Artifact(abc.ABC, Generic[T]):
    """
    A class describing objects that could and should be saved and loaded, whether it is an
    intermediate output or the final output. It contains the path of the objects as well as the
    objects itself.
    """

    def __init__(
        self,
        path: pathlib.Path | None,
        artifact: T | None = None,
    ):
        """
        Constructor

        Args:
            path:
                The path of where the object should be saved, including the file name.
            artifact:
                The underlying object to be saved.
        """
        self.path = path
        if self.path is None and artifact is None:
            raise TypeError("Path must be specified if there is no artifact attached.")

        self.artifact: T | None = artifact

    def load(self) -> Artifact:
        """
        Load the artifact from the path, only if the artifact is not loaded.
        """
        if self.path is None:
            # Nothing to load
            return self
        if self.artifact is None:
            self.artifact = self._load()
        return self

    @abc.abstractmethod
    def _load(self) -> T:
        """
        The implementation of loading the artifact using the current path. Returns the loaded
        object.
        """

    @abc.abstractmethod
    def save(self) -> None:
        """
        The implementation of saving the artifact using the current path.
        """

    @property
    def get(self) -> T:
        """
        Returns the underlying artifact.

        Raises:
            ValueError if the artifact was not loaded yet.
        """
        if self.artifact is None:
            raise ValueError("Artifact is not loaded yet.")
        return self.artifact

    def clear(self):
        """
        Remove the loaded artifact.
        """
        if self.path is None:
            raise ValueError(
                "Cannot clear the artifact is path is None. Remove the whole object instead."
            )
        self.artifact = None


class QueryArtifact(Generic[U]):
    """
    A subclass of the class ``Artifact`` that the artifact contains some form of "records" for
    each query.
    """

    @abc.abstractmethod
    def __getitem__(self, item) -> U:
        """
        Returns the "record" for the query index.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of queries.
        """

    @abc.abstractmethod
    def slice_by_query_ids(self, query_ids: List[str]):
        """
        Returns a new query artifact containing the records by query ids
        """


class NumpyArtifact(Artifact[np.ndarray]):
    """
    An artifact where the underlying artifact is a numpy array.
    """

    def _load(self) -> np.ndarray:
        return np.load(str(self.path), allow_pickle=True)

    def save(self):
        if self.path is None:
            raise ValueError("Attempt to save without a path.")
        if self.artifact is not None:
            np.save(str(self.path), self.artifact)


class NumpyQueryArtifact(NumpyArtifact, QueryArtifact[np.ndarray]):
    """
    An artifact where the underlying artifact is a numpy array of the first dim being the
    queries.
    """

    def __getitem__(self, item) -> np.ndarray:
        self.load()
        return self.artifact[item]

    def __len__(self) -> int:
        self.load()
        return len(self.artifact)

    def slice_by_query_ids(self, query_ids: List[str]):
        raise RuntimeError("No query ids is encoded in numpy query artifact.")


class PandasArtifact(Artifact[pd.DataFrame]):
    """
    An artifact where the underlying artifact is a pandas dataframe.
    """

    def _load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def save(self):
        if self.path is None:
            raise ValueError("Attempt to save without a path.")
        if self.artifact is not None:
            self.artifact.to_csv(self.path, index=False)


class QueryDatasetArtifact(Artifact[QueryDataset], QueryArtifact["QueryDatasetArtifact"]):
    """
    An artifact containing the query dataset. Does NOT support saving.
    """

    def __init__(self, name: str, path: pathlib.Path | None, artifact: QueryDataset | None = None):
        super().__init__(path, artifact)
        self.name = name

    def _load(self) -> QueryDataset:
        return get_query_dataset(self.name, self.path)

    def save(self) -> None:
        raise ValueError("Attempted to save the query dataset! Aborting!")

    @classmethod
    def from_config(cls, config: LocalRagConfig):
        return cls(config.data.query_name, config.data.query_path)

    def __getitem__(self, item) -> QueryDatasetArtifact:
        self.load()
        artifact = self.artifact[item]
        return QueryDatasetArtifact(self.name, self.path, artifact)

    def __len__(self):
        self.load()
        return len(self.artifact)

    def slice_by_query_ids(self, query_ids: List[str]) -> QueryDatasetArtifact:
        return self.__getitem__(query_ids)


class DocumentDatasetArtifact(Artifact[DocumentDataset]):
    """
    An artifact containing the document dataset. Does NOT support saving.
    """

    def __init__(
        self,
        name: str,
        load_semantic_chunks: bool,
        path: pathlib.Path | None,
        chunking_strategy: str = "",
        corpus_chunk_size: int | None = None,
        corpus_overlap_size: int | None = None,
        artifact: DocumentDataset | None = None,
    ):
        super().__init__(path, artifact)
        self.name = name
        self.load_docs = True
        self.load_semantic_chunks = load_semantic_chunks
        self.chunking_strategy = chunking_strategy
        self.corpus_chunk_size = corpus_chunk_size
        self.corpus_overlap_size = corpus_overlap_size

    def _load(self) -> DocumentDataset:
        document_dataset = get_document_dataset(
            self.name,
            self.path,
            load_docs=self.load_docs,
            load_semantic_chunks=self.load_semantic_chunks,
        )

        if not self.load_semantic_chunks:
            document_dataset.chunks = create_chunks(
                document_dataset,
                strategy=self.chunking_strategy,
                chunk_size=self.corpus_chunk_size,
                overlap_size=self.corpus_overlap_size,
            )

        return document_dataset

    def save(self) -> None:
        raise ValueError("Attempted to save the document dataset! Aborting!")

    @classmethod
    def from_config(cls, config: LocalRagConfig):
        return cls(
            config.data.corpus_name,
            config.data.load_semantic_chunks,
            config.data.corpus_path,
            config.data.chunking_strategy,
            config.data.corpus_chunk_size,
            config.data.corpus_overlap_size,
        )

    def __getitem__(self, item) -> DocumentDatasetArtifact:
        self.load()
        artifact = self.artifact[item]
        return DocumentDatasetArtifact(
            self.name,
            self.load_semantic_chunks,
            self.path,
            self.chunking_strategy,
            self.corpus_chunk_size,
            self.corpus_overlap_size,
            artifact,
        )

    def __len__(self):
        self.load()
        return len(self.artifact)


class PandasQueryArtifact(PandasArtifact, QueryArtifact[pd.DataFrame | pd.Series]):
    """
    An artifact where the underlying artifact is a pandas dataframe with number of rows equal
    to the number of queries.
    """

    def __getitem__(self, item) -> pd.DataFrame | pd.Series:
        self.load()
        return self.artifact.loc[item]

    def __len__(self) -> int:
        self.load()
        return len(self.artifact)

    def slice_column(self, cols: list[str]) -> PandasQueryArtifact:
        self.load()
        return PandasQueryArtifact(self.path, self.artifact[cols])

    def slice_by_query_ids(self, query_ids: List[str]) -> PandasQueryArtifact:
        self.load()
        new_artifact = self.artifact.set_index("query_id").loc[query_ids].reset_index()
        return PandasQueryArtifact(self.path, new_artifact)


class PickleArtifact(Artifact, Generic[T]):
    def _load(self) -> T:
        with open(str(self.path), "rb") as f:
            return pkl.load(f)

    def save(self) -> None:
        if self.path is None:
            raise ValueError("Attempted to save without a path.")
        with open(str(self.path), "wb") as f:
            pkl.dump(self.artifact, f)


class JSONArtifact(Artifact):
    def _load(self) -> T:
        with open(self.path, "r") as f:
            return json.load(f)

    def save(self) -> None:
        if self.path is None:
            raise ValueError("Attempt to save without a path.")
        with open(self.path, "w") as f:
            json.dump(self.artifact, f, indent=2)
