from typing import Union, Callable

import torch
from torch import Tensor

from nn.backward_pass.BackwardFunction import BackwardFunction

_backward_fn_type = Union[BackwardFunction, Callable[[Union[None, Tensor]], Union[None, Tensor]]]


class BaseBackwardPass:
    OUTPUT = "output"

    def __init__(self, use_default_graph: bool = False):
        self.use_default_graph: bool = use_default_graph

        self._output: Union[None, Tensor] = None
        self._loss: Union[None, Tensor] = None

        self._output_hook = None

    def __call__(self, *args, **kwargs):
        if self._output is None:
            raise Exception("output is not set.")

        if self._loss is None:
            raise Exception("loss is not set.")

        self._loss.backward()

    def set_output(self, output: Tensor):
        if self._output_hook is not None:
            self._output_hook.remove()

        if self.use_default_graph:
            self._output = output
            # self._output_hook = output.register_hook(self._backward_pass_hook)
        else:
            self._output = output.detach_()
            self._output.requires_grad = True
            self._output_hook = output.register_hook(self._backward_pass_hook)
        return output

    def set_loss(self, loss: Tensor):
        self._loss = loss

    def _completed_backward_pass(self):
        self._output_hook.remove()
        self._output = None
        self._loss = None

    def add_relation(self, start_at, *args: Union[str, BackwardFunction]):
        raise NotImplemented

    def compile(self):
        raise NotImplemented

    @torch.no_grad()
    def _backward_pass_hook(self, grad_output: Tensor):
        raise NotImplemented
