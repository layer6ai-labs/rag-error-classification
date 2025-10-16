import logging
from typing import Iterable

import pandas as pd

from base import artifact
from base.artifact import QueryDatasetArtifact


class QueryResults:
    """
    A class which enable efficient printing/logging of the results of the RAG for each query.
    """

    _query_separator = "\u2588" * 40
    _major_separator = "=" * 40
    _minor_separator = "-" * 40

    def __init__(
        self,
        queries: QueryDatasetArtifact,
    ):
        """
        Constructor.

        Args:
            queries:
                The query dataset artifact.
        """
        self.queries = queries
        self.artifacts = []
        self.length = len(queries)
        self.log = logging.getLogger(self.__class__.__name__)

    def add_query_artifacts(
        self,
        name: str,
        query_artifact: artifact.QueryArtifact,
    ) -> None:
        """
        Add the query artifact to be printed to the object. The order of the printing will be
        the order of calling this method.

        Args:
            name:
                The name of the artifact (which will be printed)
            query_artifact:
                The query artifact to be printed.
        """
        if isinstance(query_artifact, artifact.QueryArtifact):
            if self.length != len(query_artifact):
                raise ValueError(
                    f"Query artifacts length ({len(query_artifact)}) does not match"
                    f" the numnber of queries ({self.length})."
                )
            self.artifacts.append((name, query_artifact))

    def print_result(self, indices: Iterable[int] | None = None) -> None:
        """
        Print the results, via a logger.

        Args:
            indices:
                The indices of the queries to be printed. If None, print all.
        """
        if indices is None:
            indices = range(self.length)

        for i in indices:
            query = self.queries.artifact.queries[i]
            self.log.info(self._query_separator)
            self.log.info(query["input"])
            for name, query_artifact in self.artifacts:
                value = query_artifact[i]
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    value = value.to_dict()
                    if "query_id" in value:
                        value.pop("query_id")
                    for key, v in value.items():
                        to_print = f"{key}={v}"
                        for line in to_print.split("\n"):
                            if len(line) > 0:
                                self.log.info(f"{name:20s}: {line}")
                else:
                    # value should be numpy array
                    for line in str(value.tolist()).split("\n"):
                        if len(line) > 0:
                            self.log.info(f"{line}")
            self.log.info(
                f"{'Ground Truth':20s}: {query['answer']}. Unanswerable = {query['unanswerable']}"
            )
            if "para_ids" in query:
                to_print = [f"{d}_{p}" for d, p in zip(query["doc_ids"], query["para_ids"])]
            else:
                to_print = query["doc_ids"]
            self.log.info(f"{'GT Citations':20s}: {to_print}")

    def __len__(self):
        return self.length

    def names(self) -> list[str]:
        """
        Returns the names of the artifacts to be printed.
        """
        return [a[0] for a in self.artifacts]
