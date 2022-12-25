from __future__ import annotations

import abc
from typing import Union, Type

from torch import nn, Tensor

from nn.graphs.ArgsKwargs import ArgsKwargs


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self._inputs = None
        self._outputs = None
        self._backward_module: Union[None, BackwardFunction] = None
        self._parent_module_attr = lambda x: None

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
    def backward_function(self) -> Union[None, BackwardFunction]:
        return self._backward_module

    @backward_function.setter
    def backward_function(self, function):
        self.set_backward_function(function)

    def set_backward_function(self, backward_class: Type[BackwardFunction]) -> Layer:
        if not issubclass(backward_class, BackwardFunction):
            raise Exception(f"Backward Module is not set for '{self}'")

        self._backward_module = backward_class(self)
        return self


class BackwardFunction(abc.ABC):
    def __init__(self, layer: Layer = None):
        self._layer = None
        self.set_layer(layer)

    def backward(self, *grad_output: Union[None, Tensor], **grad_output_kwarg) -> Union[None, Tensor]:
        raise NotImplementedError

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer: Union[None, Layer]):
        self.set_layer(layer)

    def set_layer(self, layer: Union[None, Layer]):
        if layer is not None and not isinstance(layer, Layer):
            raise Exception(f'layer not instance of Layer class')

        self._layer = layer

    def get_parameter(self, name: str) -> Union[None, Tensor]:
        if self._layer is None:
            return None

        if hasattr(self._layer, name):
            return getattr(self._layer, name)

        raise Exception(f'"{name}" is not found')

    @property
    def inputs(self):
        if self._layer is None:
            return None

        return self._layer.inputs

    @property
    def outputs(self):
        if self._layer is None:
            return None

        return self._layer.outputs

    @staticmethod
    def set_grad_of(tensor: Tensor, grad: Tensor):
        if tensor is None or tensor.requires_grad is False:
            return

        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad += grad

        return tensor.grad
