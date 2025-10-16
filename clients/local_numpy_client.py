import math
from abc import abstractmethod
from typing import Any, Dict, Tuple, TypeVar, Iterable, Sized, Protocol, Generic, runtime_checkable
from typing import Callable

import numpy as np
from numpy.typing import NDArray


_T = TypeVar("_T", bound=Any)
T = TypeVar("T", covariant=True)


@runtime_checkable
class OrderedIterable(Iterable[T], Sized, Protocol, Generic[T]):
    @abstractmethod
    def __getitem__(self, key: Any, /) -> Any:
        raise IndexError


ScalarOrArray = T | OrderedIterable[T] | OrderedIterable[OrderedIterable[T]]
Numeric = int | float | bool | np.number[Any]


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

        vectors = self.ensure_2d(vectors)
        ids = self.ensure_1d(ids)
        payloads = self.ensure_1d(payloads, allow_higher_dim=True)

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

        self._size, self._vectors = self.pre_allocate_and_append(
            self._vectors, vectors, orig_size=orig_size, pre_allocate_2x=self.pre_alloc_memory, always_copy=True
        )

        _, self._ids = self.pre_allocate_and_append(
            self._ids, ids, orig_size=orig_size, pre_allocate_2x=self.pre_alloc_memory, always_copy=True
        )

        _, self._payloads = self.pre_allocate_and_append(
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

        query_vectors = self.ensure_2d(query_vectors)

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

    @staticmethod
    def ensure_1d(arr: Any, allow_higher_dim: bool = False) -> NDArray[Any]:
        if not isinstance(arr, np.ndarray) or arr.ndim < 1:
            arr = np.atleast_1d(arr)
        if not allow_higher_dim and arr.ndim != 1:
            raise ValueError(f"Expecting at most a 1D array, got {arr.ndim}D array")
        return arr

    @staticmethod
    def ensure_2d(arr: Any, allow_higher_dim: bool = False) -> NDArray[Any]:
        if not isinstance(arr, np.ndarray) or arr.ndim < 2:
            arr = np.atleast_2d(arr)
        if not allow_higher_dim and arr.ndim != 2:
            raise ValueError(f"Expecting at most a 2D array, got {arr.ndim}D array")
        return arr

    @staticmethod
    def pre_allocate_and_append(
        base: NDArray[_T] | None,
        append: NDArray[_T],
        orig_size: int | None = None,
        pre_allocate_2x: bool = True,
        always_copy: bool = False,
    ) -> Tuple[int, NDArray[_T]]:
        """
        Pre-allocate memory to hold data from ``base`` and ``append``,
        and return a new array with values from ``base`` and ``append`` concatenated.

        Args:
            base: a ``numpy`` array with the original data. Can be ``None``.

            append: a ``numpy`` array with the data to append.

            orig_size: an optional integer representing number of records from ``base`` that are considered data
                (if ``base`` have more memory allocated than used).
                If not provided, full ``base`` array is considered as data.

            pre_allocate_2x:
                If ``True``, memory allocation will be doubled until it can fit desired data from ``base`` and ``append``.
                That means that ``2**N`` memory will be allocated where ``N`` is the smallest integer such that
                ``2**N >= min(orig_size, len(base)) + len(append)``.

                If ``False``, only enough memory is allocated to hold both ``base`` and ``append`` arrays.

            always_copy:
                If ``True``, data from ``append`` will always be copied to new memory.

                If ``False``, ``append`` could be returned as is in certain situations.
        """
        if base is not None and not isinstance(base, np.ndarray):
            raise TypeError("'base' must be a numpy array or None")
        if not isinstance(append, np.ndarray):
            raise TypeError("'append' must be a numpy array")

        # check dimensions and data type
        if base is not None:
            if base.ndim != append.ndim:
                raise ValueError(f"Cannot append {append.ndim}D array to {base.ndim}D array")
            if base.shape[1:] != append.shape[1:]:
                raise ValueError(
                    f"Cannot append array of shape {append.shape} to array of shape {base.shape}."
                    f" All dimensions except for the first must match, but {append.shape[1:]} != {base.shape[1:]}"
                )
            if not np.can_cast(append.dtype, base.dtype):
                raise ValueError(
                    f"Attempting to append data of type {append.dtype} to array of type {base.dtype}."
                    f" Cannot safely convert {append.dtype} to {base.dtype}"
                )

        # shortcut when we don't have original data and we can simply return the appended array
        if not pre_allocate_2x and base is None and not always_copy:
            # simply return the "added" array
            return len(append), append

        # now we know that we can append one array to another,
        # so we can proceed to memory allocation

        # available memory (i.e. existing size of base)
        avail_mem = len(base) if base is not None else 0
        # infer original data size if not provided
        if orig_size is None:
            orig_size = avail_mem
        # check that original size makes sense
        if orig_size < 0 or orig_size > avail_mem:
            raise ValueError(f"Value {orig_size} for 'orig_size' is invalid since base array size is {avail_mem}.")
        # required data size
        req_size = orig_size + len(append)
        # if we already have enough memory, we don't need to allocate anything
        if avail_mem >= req_size and base is not None:
            new_arr = base
        else:
            # otherwise, we need to allocate new memory
            if pre_allocate_2x:
                if req_size == 0:
                    req_mem = 0
                else:
                    # need to find N such that 2**N >= req_size
                    req_mem = 2 ** math.ceil(math.log2(req_size))
            else:
                # we simpy need enough to accomodate for both
                req_mem = req_size
            # new data type
            new_dtype = base.dtype if base is not None else append.dtype
            # allocate required memory
            new_arr = np.empty(tuple([req_mem] + list(append.shape[1:])), dtype=new_dtype)

            # now copy original data to it
            if base is not None:
                new_arr[:orig_size] = base[:orig_size]

        # copy data from ``append`` to new array
        new_arr[orig_size:req_size] = append

        return req_size, new_arr
