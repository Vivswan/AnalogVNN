from typing import Union, Callable

from torch import Tensor

TENSOR_OPERABLE = Union[Tensor, int, float, bool]
TENSOR_CALLABLE = Callable[[TENSOR_OPERABLE], TENSOR_OPERABLE]
