from __future__ import annotations

import abc
from typing import Union, Callable, Any

from torch import nn, Tensor


class BackwardModule(abc.ABC):
    def __init__(self, layer: nn.Module = None):
        self._layer = None
        self.set_layer(layer)

    def backward(self, *grad_output: Union[None, Tensor], **grad_output_kwarg) -> Union[None, Tensor]:
        raise NotImplementedError

    __call__: Callable[..., Any] = backward

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer: Union[None, nn.Module]):
        self.set_layer(layer)

    def set_layer(self, layer: Union[None, nn.Module]):
        if layer is not None and not isinstance(layer, nn.Module):
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
