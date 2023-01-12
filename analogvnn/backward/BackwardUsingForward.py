from abc import ABC

from torch import Tensor

from analogvnn.backward.BackwardModule import BackwardModule
from analogvnn.utils.common_types import TENSORS

__all__ = ['BackwardUsingForward']


class BackwardUsingForward(BackwardModule, ABC):
    """The backward module that uses the forward function to compute the backward gradient."""

    def backward(self, *grad_output: Tensor, **grad_output_kwarg: Tensor) -> TENSORS:
        """Computes the backward gradient of inputs with respect to outputs using the forward function.

        Args:
            *grad_output (Tensor): The gradients of the output of the layer.
            **grad_output_kwarg (Tensor): The gradients of the output of the layer.

        Returns:
            TENSORS: The gradients of the input of the layer.
        """
        return self._layer.forward(*grad_output, **grad_output_kwarg)
