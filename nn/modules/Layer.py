from __future__ import annotations

from typing import Union, Type, Callable

from torch import nn

from nn.graphs.ArgsKwargs import ArgsKwargs
from nn.modules.BackwardFunction import BackwardFunction
from nn.modules.BackwardModule import BackwardModule


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self._inputs = None
        self._outputs = None
        self._backward_module: Union[None, BackwardModule] = None

    def __call__(self, *inputs, **kwargs):
        outputs = super().__call__(*inputs, **kwargs)

        if self.training:
            self._inputs = ArgsKwargs.from_args_kwargs_object(ArgsKwargs(args=inputs, kwargs=kwargs))
            self._outputs = outputs

        return outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def backward_function(self) -> Union[None, BackwardModule]:
        return self._backward_module

    @backward_function.setter
    def backward_function(self, function):
        self.set_backward_function(function)

    def set_backward_function(self, backward_class: Union[BackwardModule, Type[BackwardModule], Callable]) -> Layer:
        if issubclass(backward_class, BackwardModule):
            self._backward_module = backward_class(self)
        elif isinstance(backward_class, BackwardModule):
            self._backward_module = backward_class
        elif callable(backward_class):
            self._backward_module = BackwardFunction(backward_class, self)
        else:
            raise Exception(f"Backward Module is not set for '{self}'")

        return self
