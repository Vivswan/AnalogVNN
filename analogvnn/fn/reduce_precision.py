import torch
from torch import Tensor

from analogvnn.utils.common_types import TENSOR_OPERABLE

__all__ = ['reduce_precision', 'stochastic_reduce_precision']


def reduce_precision(x: Tensor, precision: TENSOR_OPERABLE, divide: TENSOR_OPERABLE) -> Tensor:
    x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
    g: Tensor = x * precision
    f = torch.sign(g) * torch.maximum(
        torch.floor(torch.abs(g)),
        torch.ceil(torch.abs(g) - divide)
    ) * (1 / precision)
    return f


def stochastic_reduce_precision(x: Tensor, precision: TENSOR_OPERABLE) -> Tensor:
    g: Tensor = x * precision
    rand_x = torch.rand_like(g, requires_grad=False)

    g_abs = torch.abs(g)
    g_floor = torch.floor(g_abs)
    g_ceil = torch.ceil(g_abs)

    prob_floor = 1 - torch.abs(g_floor - g_abs)
    bool_floor = rand_x <= prob_floor
    do_floor = bool_floor.type(torch.float)
    do_ceil = torch.logical_not(bool_floor).type(torch.float)

    f = torch.sign(g) * (do_floor * g_floor + do_ceil * g_ceil) * (1 / precision)
    return f
