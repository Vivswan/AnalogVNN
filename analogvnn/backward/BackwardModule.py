from __future__ import annotations

import abc
from typing import Callable, Any, Optional, Sequence

import torch
from torch import nn, Tensor, autograd

__all__ = ['BackwardFunction']


class BackwardModule(abc.ABC):
    _layer: Optional[nn.Module]

    class AutogradBackward(autograd.Function):
        # noinspection PyMethodOverriding
        @staticmethod
        def forward(ctx: Any, backward_module: BackwardModule, _, *args, **kwargs) -> Any:
            ctx.backward_module = backward_module
            return ctx.backward_module._call_impl_forward(*args, **kwargs)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            backward_module: BackwardModule = ctx.backward_module
            results = backward_module._call_impl_backward(*grad_outputs)

            if isinstance(results, Sequence):
                return (None, None, *results)

            return None, None, results

    def __init__(self, layer: nn.Module = None):
        super().__init__()
        self._layer = None
        self._empty_holder_tensor = torch.ones((1,), requires_grad=True)
        if not isinstance(self, nn.Module):
            self.set_layer(layer)

    def forward(self, *inputs, **inputs_kwarg):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

    forward._implemented = False

    @abc.abstractmethod
    def backward(self, *grad_outputs, **grad_output_kwarg):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"backward\" function")

    def _call_impl_backward(self, *grad_output, **grad_output_kwarg):
        return self.backward(*grad_output, **grad_output_kwarg)

    def _call_impl_forward(self, *args, **kwarg):
        return self.forward(*args, **kwarg)

    __call__: Callable[..., Any] = _call_impl_backward

    def auto_apply(self, *args, to_apply=True, **kwargs):
        if to_apply and self.layer.training:
            return BackwardModule.AutogradBackward.apply(self, self._empty_holder_tensor, *args, **kwargs)
        else:
            return self._call_impl_forward(*args, **kwargs)

    def has_forward(self):
        return not hasattr(self.forward, "_implemented")

    @property
    def layer(self):
        return self.get_layer()

    def get_layer(self):
        if isinstance(self, nn.Module):
            return self
        else:
            return self._layer

    def set_layer(self, layer: Optional[nn.Module]):
        if isinstance(self, nn.Module):
            raise Exception(f"layer of Backward Module is set to itself")
        if self._layer is not None:
            raise Exception(f"changing the layer of Backward Module is not allowed")
        if layer is not None and not isinstance(layer, nn.Module):
            raise Exception(f'layer not instance of Layer class')

        self._layer = layer
        return self

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
        if isinstance(self, nn.Module):
            return super(BackwardModule, self).__getattr__(name)
        if not str(name).startswith("__") and self._layer is not None and hasattr(self._layer, name):
            return getattr(self._layer, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
