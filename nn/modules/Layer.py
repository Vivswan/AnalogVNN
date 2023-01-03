from __future__ import annotations

from typing import Union, Type, Callable, Sequence, Optional

from torch import nn, Tensor

from nn.graphs.ArgsKwargs import ArgsKwargs
from nn.modules.BackwardFunction import BackwardFunction
from nn.modules.BackwardModule import BackwardModule


class Layer(nn.Module):
    _inputs: Union[None, ArgsKwargs]
    _outputs: Union[None, Tensor, Sequence[Tensor]]
    _backward_module: Optional[BackwardModule]

    def __init__(self):
        super(Layer, self).__init__()
        self._inputs = None
        self._outputs = None
        self._backward_module = None

    def __call__(self, *inputs, **kwargs):
        outputs = super().__call__(*inputs, **kwargs)

        if self.training:
            self._inputs = ArgsKwargs(args=inputs, kwargs=kwargs)
            self._outputs = outputs

        return outputs

    @property
    def inputs(self):
        return ArgsKwargs.from_args_kwargs_object(self._inputs)

    @property
    def outputs(self):
        return self._outputs

    @property
    def backward_function(self) -> Optional[BackwardModule]:
        if self._backward_module is not None:
            return self._backward_module

        if isinstance(self, BackwardModule):
            return self

        return None

    @backward_function.setter
    def backward_function(self, function):
        self.set_backward_function(function)

    def set_backward_function(self, backward_class: Union[BackwardModule, Type[BackwardModule], Callable]) -> Layer:
        if backward_class == self:
            return self

        if issubclass(backward_class, BackwardModule):
            self._backward_module = backward_class(self)
        elif isinstance(backward_class, BackwardModule):
            self._backward_module = backward_class
        elif callable(backward_class):
            self._backward_module = BackwardFunction(backward_class, self)
        else:
            raise Exception(f"Backward Module is not set for '{self}'")

        return self
