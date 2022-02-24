import inspect
from typing import Union, Callable

import torch
from torch import Tensor

from nn.BaseLayer import BaseLayer
from nn.backward_pass.BackwardFunction import BackwardFunction

_backward_fn_type = Union[BackwardFunction, Callable[[Union[None, Tensor]], Union[None, Tensor]]]


class BaseBackwardPass:
    OUTPUT = "output"
    STOP = "stop"

    def __init__(self, use_autograd_graph: bool = False):
        self.use_autograd_graph: bool = use_autograd_graph

        self._output: Union[None, Tensor] = None
        self._loss: Union[None, Tensor] = None

        self._output_hook = None

    def __call__(self, *args, **kwargs):
        if self._output is None:
            raise Exception("output is not set.")

        if self._loss is None:
            raise Exception("loss is not set.")

        self._loss.backward()
        if not self.use_autograd_graph:
            self._backward_pass(self._output.grad)
        self._output = None
        self._loss = None

    def set_output(self, output: Tensor):
        if self._output_hook is not None:
            self._output_hook.remove()

        if self.use_autograd_graph:
            self._output = output
        else:
            self._output = output.detach()
            self._output.requires_grad = True
        return self._output

    def set_loss(self, loss: Tensor):
        self._loss = loss

    @staticmethod
    def get_backward_function(module):
        if isinstance(module, BaseLayer):
            if module.get_backward_module() is not None:
                return module.get_backward_module().backward
        if isinstance(module, BackwardFunction):
            return module.backward
        if inspect.ismethod(module) or inspect.isfunction(module):
            return module
        return None

    def add_relation(self, start_at, *args: Union[str, BackwardFunction]):
        raise NotImplemented

    def compile(self):
        raise NotImplemented

    @torch.no_grad()
    def _backward_pass(self, grad_output: Tensor):
        raise NotImplemented
