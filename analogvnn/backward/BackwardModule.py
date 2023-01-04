from __future__ import annotations

import abc
from typing import Callable, Any, Optional

from torch import nn, Tensor


class BackwardModule(abc.ABC):
    _layer: Optional[nn.Module]

    def __init__(self, layer: nn.Module = None):
        self._layer = None
        self.set_layer(layer)

    @abc.abstractmethod
    def backward(self, *grad_output, **grad_output_kwarg):
        raise NotImplementedError

    def _call_impl_backward(self, *grad_output, **grad_output_kwarg):
        return self.backward(*grad_output, **grad_output_kwarg)

    __call__: Callable[..., Any] = _call_impl_backward

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer: Optional[nn.Module]):
        self.set_layer(layer)

    def set_layer(self, layer: Optional[nn.Module]):
        if layer is not None and not isinstance(layer, nn.Module):
            raise Exception(f'layer not instance of Layer class')

        self._layer = layer

    def get_parameter(self, name: str) -> Optional[Tensor]:
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

        try:
            tensor.backward(gradient=grad, inputs=tensor)
        except Exception:
            for key, value in tensor._backward_hooks.items():
                grad = value(grad)

            if tensor.grad is None:
                tensor.grad = grad
            else:
                tensor.grad += grad

        return tensor.grad

    def __getattr__(self, name):
        if not str(name).startswith("__") and self._layer is not None and hasattr(self._layer, name):
            return getattr(self._layer, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
