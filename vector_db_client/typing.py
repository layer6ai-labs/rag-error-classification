from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Sized
from typing import Any
from typing import Generic
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

import numpy as np

T = TypeVar("T", covariant=True)


@runtime_checkable
class OrderedIterable(Iterable[T], Sized, Protocol, Generic[T]):
    @abstractmethod
    def __getitem__(self, key: Any, /) -> Any:
        raise IndexError


ScalarOrArray = T | OrderedIterable[T] | OrderedIterable[OrderedIterable[T]]
Numeric = int | float | bool | np.number[Any]
