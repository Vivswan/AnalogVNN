from typing import Tuple, Union

import torch
from torch import nn

__all__ = ['to_float_tensor', 'to_nongrad_parameter']


def to_float_tensor(*args) -> Tuple[Union[torch.Tensor, None], ...]:
    """Converts the given arguments to `torch.Tensor` of type `torch.float32`.

    The tensors are not trainable.

    Args:
        *args: the arguments to convert.

    Returns:
        tuple: the converted arguments.
    """
    return tuple((None if i is None else torch.tensor(i, requires_grad=False, dtype=torch.float)) for i in args)


def to_nongrad_parameter(*args) -> Tuple[Union[nn.Parameter, None], ...]:
    """Converts the given arguments to `nn.Parameter` of type `torch.float32`.

    The parameters are not trainable.

    Args:
        *args: the arguments to convert.

    Returns:
        tuple: the converted arguments.
    """
    return tuple((None if i is None else nn.Parameter(i, requires_grad=False)) for i in to_float_tensor(*args))
