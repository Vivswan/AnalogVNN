from __future__ import annotations

import abc
import inspect
from typing import Union, Type, Callable, Any

from torch import nn, Tensor

from nn.graphs.ArgsKwargs import ArgsKwargs


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self._inputs = None
        self._outputs = None
        self._backward_module: Union[None, BackwardModule] = None
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


class BackwardModule(abc.ABC):
    def __init__(self, layer: Layer = None):
        self._layer = None
        self.set_layer(layer)

    def backward(self, *grad_output: Union[None, Tensor], **grad_output_kwarg) -> Union[None, Tensor]:
        raise NotImplementedError

    __call__: Callable[..., Any] = backward

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


class BackwardFunction(BackwardModule):
    def __init__(self, backward_function: Callable = None, layer: Layer = None):
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
