from typing import Tuple, Union

import torch
from torch import nn


def to_float_tensor(*args) -> Tuple[Union[torch.Tensor, None], ...]:
    return tuple((None if i is None else torch.tensor(i, requires_grad=False, dtype=torch.float)) for i in args)


def to_nongrad_parameter(*args) -> Tuple[Union[nn.Parameter, None], ...]:
    return tuple((None if i is None else nn.Parameter(i, requires_grad=False)) for i in to_float_tensor(*args))
