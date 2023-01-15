import numpy as np

from analogvnn.utils.common_types import TENSOR_OPERABLE

__all__ = ['gaussian_dirac_delta']


def gaussian_dirac_delta(x: TENSOR_OPERABLE, std: TENSOR_OPERABLE = 0.001) -> TENSOR_OPERABLE:
    """Gaussian Dirac Delta function with standard deviation `std`

    Args:
        x (TENSOR_OPERABLE): Tensor
        std (TENSOR_OPERABLE): standard deviation.

    Returns:
        TENSOR_OPERABLE: TENSOR_OPERABLE with the same shape as x, with values of the Gaussian Dirac Delta function.
    """

    return 1 / (np.abs(std) * np.sqrt(np.pi)) * np.exp(-((x / std) ** 2))
