from abc import ABC

from torch import Tensor

from analogvnn.backward.BackwardModule import BackwardModule
from analogvnn.utils.common_types import TENSORS

__all__ = ['BackwardIdentity']


class BackwardIdentity(BackwardModule, ABC):
    """The backward module that returns the output gradients as the input gradients."""

    def backward(self, *grad_output: Tensor, **grad_output_kwarg: Tensor) -> TENSORS:
        """Returns the output gradients as the input gradients.

        Args:
            *grad_output (Tensor): The gradients of the output of the layer.
            **grad_output_kwarg (Tensor): The gradients of the output of the layer.

        Returns:
            TENSORS: The gradients of the input of the layer.
        """

        if len(grad_output) == 0:
            return None

        if len(grad_output) == 1:
            return grad_output[0]

        return grad_output
