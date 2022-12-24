from __future__ import annotations
import abc
from typing import Union, Type

import torch
from torch import nn, Tensor

from nn.graphs.AcyclicDirectedGraph import AcyclicDirectedGraph
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
            self._inputs = AcyclicDirectedGraph.from_args_kwargs_object(ArgsKwargs(args=inputs, kwargs=kwargs))
            self._outputs = outputs

        return outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def set_backward_module(self, backward_class: Type[BackwardFunction]) -> Layer:
        if not issubclass(backward_class, BackwardFunction):
            raise Exception(f"Backward Module is not set for '{self}'")
        self._backward_module = backward_class(self)
        return self

    def get_backward_module(self) -> Union[None, BackwardFunction]:
        return self._backward_module

    def use(self, *args) -> Layer:
        for i in args:
            if issubclass(i, BackwardFunction):
                self.set_backward_module(i)
        return self


class BackwardFunction(abc.ABC):
    def __init__(self, layer: Layer):
        if not isinstance(layer, Layer):
            raise Exception(f'layer not instance of BaseLayer class')

        self._layer = layer

    def get_parameter(self, name: str) -> Union[None, Tensor]:
        if hasattr(self._layer, name):
            return getattr(self._layer, name)

        raise Exception(f'"{name}" is not found')

    def backward(self, *grad_output: Union[None, Tensor], **grad_output_kwarg) -> Union[None, Tensor]:
        raise NotImplementedError

    @property
    def inputs(self):
        return self._layer.inputs

    @property
    def outputs(self):
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
