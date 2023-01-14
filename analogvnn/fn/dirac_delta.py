import numpy as np

from analogvnn.utils.common_types import TENSOR_OPERABLE

__all__ = ['dirac_delta']


def dirac_delta(x: TENSOR_OPERABLE, a: TENSOR_OPERABLE = 0.001) -> TENSOR_OPERABLE:
    """`dirac_delta` takes `x` and returns the Dirac delta function of `x` with standard deviation of `a`.

    Args:
        x (TENSOR_OPERABLE): Tensor
        a (TENSOR_OPERABLE): standard deviation.

    Returns:
        TENSOR_OPERABLE: TENSOR_OPERABLE with the same shape as x, but with values equal to the Dirac delta function
        of x.
    """

    return 1 / (np.abs(a) * np.sqrt(np.pi)) * np.exp(-((x / a) ** 2))
