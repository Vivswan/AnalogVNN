from __future__ import annotations

import abc
from typing import Callable, Any, Optional, Tuple, Type

import torch
from torch import nn, Tensor, autograd

from analogvnn.utils.common_types import TENSORS

__all__ = ['BackwardModule']


class BackwardModule(abc.ABC):
    """Base class for all backward modules.

    A backward module is a module that can be used to compute the backward gradient of a given
    function. It is used to compute the gradient of the input of a function with respect to the output
    of the function.

    Attributes:
        _layer (Optional[nn.Module]): The layer for which the backward gradient is computed.
        _empty_holder_tensor (Tensor): A placeholder tensor which always requires gradient for backward gradient
         computation.
        _autograd_backward (Type[AutogradBackward]): The autograd backward function.
        _disable_autograd_backward (bool): If True the autograd backward function is disabled.
    """

    _layer: Optional[nn.Module]
    _empty_holder_tensor: Tensor = torch.ones((1,), requires_grad=True)
    _autograd_backward: Type[AutogradBackward] = None
    _disable_autograd_backward: bool = False

    # noinspection PyAbstractClass
    class AutogradBackward(autograd.Function):
        """Optimization and proper calculation of gradients when using the autograd engine."""

        # noinspection PyMethodOverriding
        @staticmethod
        def forward(ctx: Any, backward_module: BackwardModule, _: Tensor, *args: Tensor, **kwargs: Tensor) -> TENSORS:
            """Forward pass of the autograd function.

            Args:
                ctx: The context of the autograd function.
                backward_module (BackwardModule): The backward module.
                _ (Tensor): placeholder tensor which always requires grad.
                *args (Tensor): The arguments of the function.
                **kwargs (Tensor): The keyword arguments of the function.

            Returns:
                TENSORS: The output of the function.
            """

            ctx.backward_module = backward_module
            return ctx.backward_module._call_impl_forward(*args, **kwargs)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[None, None, TENSORS]:
            """Backward pass of the autograd function.

            Args:
                ctx: The context of the autograd function.
                *grad_outputs (Tensor): The gradients of the output of the function.

            Returns:
                TENSORS: The gradients of the input of the function.
            """

            backward_module: BackwardModule = ctx.backward_module
            results = backward_module._call_impl_backward(*grad_outputs)

            if isinstance(results, (tuple, list)):
                return (None, None, *results)

            return None, None, results

    def __init__(self, layer: nn.Module = None):
        """Initializes of the BackwardModule class.

        Args:
            layer (nn.Module): The layer for which the backward gradient is computed.
        """

        super(BackwardModule, self).__init__()
        self._layer = None
        self._set_autograd_backward()
        if not isinstance(self, nn.Module):
            self.set_layer(layer)

    def forward(self, *inputs: Tensor, **inputs_kwarg: Tensor) -> TENSORS:
        """Forward pass of the layer.

        Args:
            *inputs (Tensor): The inputs of the layer.
            **inputs_kwarg (Tensor): The keyword inputs of the layer.

        Returns:
            TENSORS: The output of the layer.

        Raises:
            NotImplementedError: If the forward pass is not implemented.
        """

        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

    forward._implemented = False

    @abc.abstractmethod
    def backward(self, *grad_outputs: Tensor, **grad_output_kwarg: Tensor) -> TENSORS:
        """Backward pass of the layer.

        Args:
            *grad_outputs (Tensor): The gradients of the output of the layer.
            **grad_output_kwarg (Tensor): The keyword gradients of the output of the layer.

        Returns:
            TENSORS: The gradients of the input of the layer.

        Raises:
            NotImplementedError: If the backward pass is not implemented.
        """

        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "backward" function')

    def _call_impl_forward(self, *args: Tensor, **kwarg: Tensor) -> TENSORS:
        """Calls Forward pass of the layer.

        Args:
            *inputs (Tensor): The inputs of the layer.
            **inputs_kwarg (Tensor): The keyword inputs of the layer.

        Returns:
            TENSORS: The output of the layer.
        """

        return self.forward(*args, **kwarg)

    def _call_impl_backward(self, *grad_output: Tensor, **grad_output_kwarg: Tensor) -> TENSORS:
        """Calls Backward pass of the layer.

        Args:
            *grad_outputs (Tensor): The gradients of the output of the layer.
            **grad_output_kwarg (Tensor): The keyword gradients of the output of the layer.

        Returns:
            TENSORS: The gradients of the input of the layer.
        """

        return self.backward(*grad_output, **grad_output_kwarg)

    __call__: Callable[..., Any] = _call_impl_backward

    def auto_apply(self, *args: Tensor, to_apply=True, **kwargs: Tensor) -> TENSORS:
        """Applies the backward module to the given layer using the proper method.

        Args:
            *args (Tensor): The inputs of the layer.
            to_apply (bool): if True and is training then the AutogradBackward is applied,
            otherwise the backward module is applied.
            **kwargs (Tensor): The keyword inputs of the layer.

        Returns:
            TENSORS: The output of the layer.
        """

        if to_apply and not self._disable_autograd_backward:
            return self._autograd_backward.apply(self, self._empty_holder_tensor, *args, **kwargs)
        else:
            return self._call_impl_forward(*args, **kwargs)

    def has_forward(self) -> bool:
        """Checks if the forward pass is implemented.

        Returns:
            bool: True if the forward pass is implemented, False otherwise.
        """

        return not hasattr(self.forward, '_implemented')

    @property
    def layer(self) -> Optional[nn.Module]:
        """Gets the layer for which the backward gradient is computed.

        Returns:
            Optional[nn.Module]: layer
        """

        return self.get_layer()

    def get_layer(self) -> Optional[nn.Module]:
        """Gets the layer for which the backward gradient is computed.

        Returns:
            Optional[nn.Module]: layer
        """

        if isinstance(self, nn.Module):
            return self
        else:
            return self._layer

    def set_layer(self, layer: Optional[nn.Module]) -> BackwardModule:
        """Sets the layer for which the backward gradient is computed.

        Args:
            layer (nn.Module): The layer for which the backward gradient is computed.

        Returns:
            BackwardModule: self

        Raises:
            ValueError: If self is a subclass of nn.Module.
            ValueError: If the layer is already set.
            ValueError: If the layer is not an instance of nn.Module.
        """

        if isinstance(self, nn.Module):
            raise ValueError('layer of Backward Module is set to itself')
        if self._layer is not None:
            raise ValueError('changing the layer of Backward Module is not allowed')
        if layer is not None and not isinstance(layer, nn.Module):
            raise ValueError('layer not instance of Layer class')

        self._layer = layer
        self._set_autograd_backward()
        return self

    def _set_autograd_backward(self):
        layer = self.get_layer()
        if layer is None:
            self._autograd_backward = BackwardModule.AutogradBackward
        else:
            # noinspection PyTypeChecker
            self._autograd_backward = type(
                f'{layer.__class__.__name__}AutoGrad',
                (BackwardModule.AutogradBackward,),
                {}
            )
        return self._autograd_backward

    @staticmethod
    def set_grad_of(tensor: Tensor, grad: Tensor) -> Optional[Tensor]:
        """Sets the gradient of the given tensor.

        Args:
            tensor (Tensor): The tensor.
            grad (Tensor): The gradient.

        Returns:
            Optional[Tensor]: the gradient of the tensor.
        """

        if tensor is None or tensor.requires_grad is False:
            return None

        # noinspection PyBroadException
        try:
            tensor.backward(gradient=grad, inputs=tensor)
        except Exception:
            # noinspection PyProtectedMember
            for _, value in tensor._backward_hooks.items():
                grad = value(grad)

            if tensor.grad is None:
                tensor.grad = grad
            else:
                tensor.grad += grad

        return tensor.grad

    def __getattr__(self, name: str) -> Any:
        """Gets the attribute of the layer.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute of the layer.

        Raises:
            AttributeError: If the attribute is not found.
        """

        if isinstance(self, nn.Module) or self == self._layer:
            return super(BackwardModule, self).__getattr__(name)
        if not str(name).startswith('__') and self._layer is not None and hasattr(self._layer, name):
            return getattr(self._layer, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
