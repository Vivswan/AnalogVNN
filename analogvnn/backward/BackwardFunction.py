from __future__ import annotations

from abc import ABC

from torch import nn, Tensor

from analogvnn.backward.BackwardModule import BackwardModule
from analogvnn.utils.common_types import TENSORS, TENSOR_CALLABLE

__all__ = ['BackwardFunction']


class BackwardFunction(BackwardModule, ABC):
    """The backward module that uses a function to compute the backward gradient.

    Attributes:
        _backward_function (TENSOR_CALLABLE): The function used to compute the backward gradient.
    """

    _backward_function: TENSOR_CALLABLE

    def __init__(self, backward_function: TENSOR_CALLABLE, layer: nn.Module = None):
        """Initializes the backward module.

        Args:
            backward_function (TENSOR_CALLABLE): The function used to compute the backward gradient.
            layer (nn.Module): The layer that this backward module is associated with.
        """

        super(BackwardFunction, self).__init__(layer)
        self._backward_function = backward_function

    @property
    def backward_function(self) -> TENSOR_CALLABLE:
        """The function used to compute the backward gradient.

        Returns:
            TENSOR_CALLABLE: The function used to compute the backward gradient.
        """

        return self._backward_function

    @backward_function.setter
    def backward_function(self, backward_function: TENSOR_CALLABLE):
        """Sets the function used to compute the backward gradient with.

        Args:
            backward_function (TENSOR_CALLABLE): The function used to compute the backward gradient with.
        """

        self.set_backward_function(backward_function)

    def set_backward_function(self, backward_function: TENSOR_CALLABLE) -> BackwardFunction:
        """Sets the function used to compute the backward gradient with.

        Args:
            backward_function (TENSOR_CALLABLE): The function used to compute the backward gradient with.

        Returns:
            BackwardFunction: self.
        """

        self._backward_function = backward_function
        return self

    def backward(self, *grad_output: Tensor, **grad_output_kwarg: Tensor) -> TENSORS:
        """Computes the backward gradient of inputs with respect to outputs using the backward function.

        Args:
            *grad_output (Tensor): The gradients of the output of the layer.
            **grad_output_kwarg (Tensor): The gradients of the output of the layer.

        Returns:
            TENSORS: The gradients of the input of the layer.

        Raises:
            NotImplementedError: If the backward function is not set.
        """

        if self._backward_function is None:
            raise ValueError('set backward_function first before invoking backward')

        return self._backward_function(*grad_output, **grad_output_kwarg)
