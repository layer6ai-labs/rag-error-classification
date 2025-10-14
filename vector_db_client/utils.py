import math
from typing import Any
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray


def ensure_1d(arr: Any, allow_higher_dim: bool = False) -> NDArray[Any]:
    if not isinstance(arr, np.ndarray) or arr.ndim < 1:
        arr = np.atleast_1d(arr)
    if not allow_higher_dim and arr.ndim != 1:
        raise ValueError(f"Expecting at most a 1D array, got {arr.ndim}D array")
    return arr


def ensure_2d(arr: Any, allow_higher_dim: bool = False) -> NDArray[Any]:
    if not isinstance(arr, np.ndarray) or arr.ndim < 2:
        arr = np.atleast_2d(arr)
    if not allow_higher_dim and arr.ndim != 2:
        raise ValueError(f"Expecting at most a 2D array, got {arr.ndim}D array")
    return arr


_T = TypeVar("_T", bound=Any)


def pre_allocate_and_append(
    base: NDArray[_T] | None,
    append: NDArray[_T],
    orig_size: int | None = None,
    pre_allocate_2x: bool = True,
    always_copy: bool = False,
) -> tuple[int, NDArray[_T]]:
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
