import torch
from torch import Tensor

from analogvnn.utils.common_types import TENSOR_OPERABLE

__all__ = ['reduce_precision', 'stochastic_reduce_precision']


def reduce_precision(x: TENSOR_OPERABLE, precision: TENSOR_OPERABLE, divide: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
    """Takes `x` and reduces its precision to `precision` by rounding to the nearest multiple of `precision`.

    Args:
      x (TENSOR_OPERABLE): Tensor
      precision (TENSOR_OPERABLE): the precision of the quantization.
      divide (TENSOR_OPERABLE): the rounding value that is if divide is 0.5,
         then 0.6 will be rounded to 1.0 and 0.4 will be rounded to 0.0.

    Returns:
      TENSOR_OPERABLE: TENSOR_OPERABLE with the same shape as x, but with values rounded to the nearest
      multiple of precision.
    """

    x = x if isinstance(x, Tensor) else torch.tensor(x, requires_grad=False)
    f = torch.sign(x) * torch.ceil(torch.abs(x * precision) - divide) * (1 / precision)
    return f


def stochastic_reduce_precision(x: TENSOR_OPERABLE, precision: TENSOR_OPERABLE) -> TENSOR_OPERABLE:
    """Takes `x` and reduces its precision by rounding to the nearest multiple of `precision` with stochastic scheme.

    Args:
        x (TENSOR_OPERABLE): Tensor
        precision (TENSOR_OPERABLE): the precision of the quantization.

    Returns:
        TENSOR_OPERABLE: TENSOR_OPERABLE with the same shape as x, but with values rounded to the
        nearest multiple of precision.
    """

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
