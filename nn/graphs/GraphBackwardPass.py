import inspect
from typing import Callable

import torch
from torch import Tensor

from nn.graphs.BackwardFunction import BackwardFunction
from nn.modules.Layer import Layer
from typing import Union, Dict, Set, Tuple

_backward_fn_type = Union[BackwardFunction, Callable[[Union[None, Tensor]], Union[None, Tensor]]]


class GraphBackwardPass:
    OUTPUT = "output"
    STOP = "stop"

    def __init__(self, use_autograd_graph: bool = False):
        self.use_autograd_graph: bool = use_autograd_graph

        self._input: Union[None, Tensor] = None
        self._output: Union[None, Tensor] = None
        self._loss: Union[None, Tensor] = None

        self._output_hook = None
        self.relation_dict: Dict[Union[_backward_fn_type, str], Set[_backward_fn_type]] = {}

    def __call__(self, gradient=None, *args, **kwargs):
        if self._output is None:
            raise Exception("output is not set.")

        if self._loss is None:
            raise Exception("loss is not set.")

        if len(gradient) == 0:
            gradient = None
        elif len(gradient) == 1:
            gradient = gradient[0]

        result = self._loss.backward(gradient=gradient)
        if not self.use_autograd_graph:
            result = self._backward_pass(self._output.grad)
        self._output = None
        self._loss = None

        return result

    def set_inputs(self, *inputs):
        self._input = inputs
        return self._input

    def get_inputs(self):
        return self._input

    def get_output(self):
        return self._output

    @torch.no_grad()
    def set_output(self, output: Tensor):
        if self._output_hook is not None:
            self._output_hook.remove()

        if self.use_autograd_graph:
            self._output = output
        else:
            self._output = output.detach()
            self._output.requires_grad = True
        return self._output

    def get_loss(self):
        return self._loss

    def set_loss(self, loss: Tensor):
        self._loss = loss

    @staticmethod
    def get_backward_function(module):
        if isinstance(module, Layer):
            if module.get_backward_module() is not None:
                return module.get_backward_module().backward
        if isinstance(module, BackwardFunction):
            return module.backward
        if inspect.ismethod(module) or inspect.isfunction(module):
            return module
        return None

    def add_connection(self, *args: Union[str, BackwardFunction]):
        for i, from_fn in enumerate(args[:-1]):
            if from_fn not in self.relation_dict:
                self.relation_dict[from_fn] = set()

            self.relation_dict[from_fn].add(args[i + 1])

    def compile(self):
        visited = set()
        to_visit = set()

        to_visit.add(self.OUTPUT)
        while len(to_visit) > 0:
            node = to_visit.pop()

            if node in visited:
                raise Exception("loop detected in backward pass.")

            visited.add(node)
            if node not in self.relation_dict:
                continue
            for i in self.relation_dict[node]:
                to_visit.add(i)

    @torch.no_grad()
    def _backward_pass(self, grad_output: Tensor):
        to_visit_with: Set[Tuple[Union[_backward_fn_type, str], Tensor]] = set()
        to_visit_with.add((self.OUTPUT, grad_output))

        while len(to_visit_with) > 0:
            function_id, function_grad_output = to_visit_with.pop()

            if function_id not in self.relation_dict:
                continue

            for module in self.relation_dict[function_id]:
                if module == self.STOP:
                    continue
                backward_fn = self.get_backward_function(module)
                if backward_fn is None:
                    raise NotImplementedError

                to_visit_with.add((module, backward_fn(function_grad_output)))
