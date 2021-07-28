from typing import Union, Callable, Set

import torch
from torch import Tensor

_backward_fn_type = Union[None, Callable[[Union[None, Tensor]], Union[None, Tensor]]]

class BackwardPass:
    def __init__(self, cls):
        self.cls = cls
        self._to = set()
        self._output_hook = None
        self._output: Union[None, Tensor] = None
        self._loss: Union[None, Tensor] = None
        self._backward_fn: _backward_fn_type = None

    def __call__(self, *args, **kwargs):
        if self._loss is None:
            raise Exception("loss is not set.")

        if self._output is None:
            raise Exception("output is not set.")

        self._loss.backward()

    def set_output(self, output: Tensor):
        if self._output_hook is not None:
            self._output_hook.remove()

        self._output =  output.detach_()
        self._output.requires_grad = True
        self._output_hook = output.register_hook(self._backward_pass_hook)
        return output

    def set_loss(self, loss: Tensor):
        self._loss = loss

    def set_backward(self, backward_fn: _backward_fn_type):
        self._backward_fn = backward_fn

    def to(self, module):
        if hasattr(module, "backward") and isinstance(module.backward, BackwardPass):
            self._to.add(module.backward)

        if isinstance(module, BackwardPass):
            self._to.add(module)

    def _backward_pass_hook(self, grad_output: Tensor):
        with torch.no_grad():
            if self._backward_fn is not None:
                grad_output = self._backward_fn(grad_output)

            # print(f"{self.cls.__class__.__name__}: {grad_output}")
            for i in self._to:
                i._backward_pass_hook(grad_output)
