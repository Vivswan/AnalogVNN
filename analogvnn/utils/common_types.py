from typing import Callable, Union

from torch import Tensor

__all__ = ['TENSOR_OPERABLE', 'TENSOR_CALLABLE']

TENSOR_OPERABLE = Union[Tensor, int, float, bool]
TENSOR_CALLABLE = Callable[[TENSOR_OPERABLE], TENSOR_OPERABLE]
