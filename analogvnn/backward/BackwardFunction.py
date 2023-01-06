from __future__ import annotations

from abc import ABC
from typing import Callable, Optional

from torch import nn, Tensor

from analogvnn.backward.BackwardModule import BackwardModule

__all__ = ['BackwardFunction']


class BackwardFunction(BackwardModule, ABC):
    _backward_function: Callable

    def __init__(self, backward_function: Callable, layer: nn.Module = None):
        super().__init__(layer)
        self._backward_function = backward_function

    @property
    def backward_function(self):
        return self._backward_function

    @backward_function.setter
    def backward_function(self, backward_function: Callable):
        self.set_backward_function(backward_function)

    def set_backward_function(self, backward_function: Callable):
        self._backward_function = backward_function
        return self

    def backward(self, *grad_output, **grad_output_kwarg) -> Optional[Tensor]:
        if self._backward_function is None:
            raise ValueError("set backward_function first before invoking backward")

        return self._backward_function(*grad_output, **grad_output_kwarg)
