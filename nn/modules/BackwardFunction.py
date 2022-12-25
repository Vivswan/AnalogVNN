from __future__ import annotations

from typing import Callable, Union

from torch import nn, Tensor

from nn.modules.BackwardModule import BackwardModule


class BackwardFunction(BackwardModule):
    def __init__(self, backward_function: Callable = None, layer: nn.Module = None):
        super().__init__(layer)
        self._backward_function = None
        self.set_backward_function(backward_function)

    @property
    def backward_function(self):
        return self._backward_function

    @backward_function.setter
    def backward_function(self, backward_function: Callable):
        self.set_backward_function(backward_function)

    def set_backward_function(self, backward_function: Callable):
        if backward_function is None or callable(backward_function):
            self._backward_function = backward_function
        else:
            raise ValueError('"function" must be Callable')

        return self

    def backward(self, *grad_output: Union[None, Tensor], **grad_output_kwarg) -> Union[None, Tensor]:
        if self._backward_function is None:
            raise ValueError("set backward_function first before invoking backward")

        return self._backward_function(*grad_output, **grad_output_kwarg)
