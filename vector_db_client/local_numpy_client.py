
from typing import Any, Dict, Tuple
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .typing import ScalarOrArray, Numeric
from .utils import ensure_1d
from .utils import ensure_2d
from .utils import pre_allocate_and_append


class LocalNumpyVectorDBClient:
    def __init__(
        self,
        similarity_func: Callable[
            [NDArray[np.number[Any]], NDArray[np.number[Any]]],  # input type (2 vector arrays)
            NDArray[np.number[Any]],  # output type (array of similarity scores)
        ] = np.inner,
        pre_alloc_memory: bool = True,
    ):
        """
        Simple in-memory VectorDB that uses ``numpy`` arrays for storing vectors and data.

        Args:
            similarity_func: a function that takes two arrays of vectors and returns a similarity scores for each pair
                (the higher the score, the closer the vectors).

                If not provided, a dot product is used.

            pre_alloc_memory: pre-allocate memory on inserts.

                If ``True``, ``add()`` operations will double allocated memory until it can fit the desired data.
                This leads to more memory allocated than is required to hold the data (up to 2x),
                but provides significant speedups for repeated ``add()`` operations.

                If ``False``, every ``add()`` operation will re-allocate memory to exactly fit the desired data,
                and copy original data into new memory along with added records.
                This leads to a significant overhead when ``add()`` is performed. E.g. even adding just 1 record
                to a database with 1M records will lead to (1M + 1) records being copied.
        """
        self.similarity_func = similarity_func
        self._size: int = 0  # data size
        self._vectors: NDArray[np.number[Any]] | None = None
        self._ids: NDArray[str] | None = None
        self._payloads: NDArray[Dict[str, Any]] | None = None
        self.pre_alloc_memory: bool = pre_alloc_memory

    @property
    def vectors(self) -> NDArray[np.number[Any]] | None:
        if self._vectors is None:
            return None
        else:
            return self._vectors[: self._size]

    @property
    def ids(self) -> NDArray[str] | None:
        if self._ids is None:
            return None
        else:
            return self._ids[: self._size]

    @property
    def payloads(self) -> NDArray[Dict[str, Any]] | None:
        if self._payloads is None:
            return None
        else:
            return self._payloads[: self._size]

    def _reshape_inputs(
        self,
        vectors: ScalarOrArray[Numeric],
        ids: ScalarOrArray[str] | None,
        payloads: ScalarOrArray[Dict[str, Any]] | None,
    ) -> Tuple[NDArray[np.number[Any]], NDArray[str], NDArray[Dict[str, Any]]]:  # vectors, ids, payloads
        if ids is None:
            raise ValueError("IDs are required")
        if payloads is None:
            raise ValueError("Data payloads are required")

        vectors = ensure_2d(vectors)
        ids = ensure_1d(ids)
        payloads = ensure_1d(payloads, allow_higher_dim=True)

        if len(ids) != len(vectors):
            raise ValueError(f"Provided number of ids {len(ids)} does not match number of vectors {len(vectors)}")

        if len(payloads) != len(vectors):
            raise ValueError(
                f"Provided number of payloads {len(payloads)} does not match number of vectors {len(vectors)}"
            )

        return vectors, ids, payloads

    def add(
        self,
        vectors: ScalarOrArray[Numeric],
        ids: ScalarOrArray[str] | None = None,
        payloads: ScalarOrArray[Dict[str, Any]] | None = None,
    ):
        vectors, ids, payloads = self._reshape_inputs(vectors, ids, payloads)

        orig_size = self._size

        self._size, self._vectors = pre_allocate_and_append(
            self._vectors, vectors, orig_size=orig_size, pre_allocate_2x=self.pre_alloc_memory, always_copy=True
        )

        _, self._ids = pre_allocate_and_append(
            self._ids, ids, orig_size=orig_size, pre_allocate_2x=self.pre_alloc_memory, always_copy=True
        )

        _, self._payloads = pre_allocate_and_append(
            self._payloads, payloads, orig_size=orig_size, pre_allocate_2x=self.pre_alloc_memory, always_copy=True
        )

    def search(
        self,
        query_vectors: ScalarOrArray[Numeric],
        top_k: int,
    ) -> tuple[NDArray[str], NDArray[Dict[str, Any]], NDArray[np.number[Any]]]:  # ids, payloads, distances
        if self._vectors is None or self._ids is None or self._payloads is None:
            raise ValueError("Cannot do search on uninitialized database")

        if top_k < 0:
            raise ValueError(f"Invalid value for 'top_k': {top_k}. Expecting a non-negative integer")

        query_vectors = ensure_2d(query_vectors)

        if query_vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError(
                f"Invalid shape for 'query_vectors' array: {query_vectors.shape}."
                f" Query vector size {query_vectors.shape[1]} != database vector size {self._vectors.shape[1]}"
            )

        # limit top_k to total number of vectors in the database
        top_k = min(top_k, len(self))

        if len(query_vectors) == 0 or top_k == 0:
            top_k_indices = np.array([], dtype=np.intp).reshape(len(query_vectors), top_k)
            top_distances = np.array([]).reshape(len(query_vectors), top_k)
        else:
            # calculate similarity scores between each query vector and each vector in the database
            sim_scores = self.similarity_func(query_vectors, self.vectors)
            # find top_k in each row
            # this function will find top_k in linear time, but the results will not be sorted
            partitioned = np.argpartition(sim_scores, -top_k, axis=1)
            top_k_indices_unsorted = partitioned[:, -top_k:]
            # now we need to sort top_k from largest to smallest
            top_k_scores = np.take_along_axis(sim_scores, top_k_indices_unsorted, axis=1)
            top_k_scores_sort_order = np.argsort(top_k_scores, axis=1)
            top_k_scores_reverse_sort_order = top_k_scores_sort_order[:, ::-1]
            top_k_indices = np.take_along_axis(top_k_indices_unsorted, top_k_scores_reverse_sort_order, axis=1)
            top_distances = np.take_along_axis(top_k_scores, top_k_scores_reverse_sort_order, axis=1)
        top_ids = self.ids[top_k_indices]
        top_payloads = self.payloads[top_k_indices]
        return top_ids, top_payloads, top_distances

    def __len__(self) -> int:
        return self._size
