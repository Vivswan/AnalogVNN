from typing import Union, Callable, Set

import torch
from torch import Tensor

_backward_fn_type = Union[None, Callable[[Union[None, Tensor]], Union[None, Tensor]]]


class BackwardPass:
    OUTPUT = "output"

    def __init__(self):
        self._output: Union[None, Tensor] = None
        self._loss: Union[None, Tensor] = None

        self._output_hook = None
        self.relation_dict = {}

    def __call__(self, *args, **kwargs):
        if self._loss is None:
            raise Exception("loss is not set.")

        if self._output is None:
            raise Exception("output is not set.")

        self._loss.backward()

    def set_output(self, output: Tensor):
        if self._output_hook is not None:
            self._output_hook.remove()

        self._output = output.detach_()
        self._output.requires_grad = True
        self._output_hook = output.register_hook(self._backward_pass_hook)
        return output

    def set_loss(self, loss: Tensor):
        self._loss = loss

    def add_relation(self, from_fn: Union[str, _backward_fn_type], to_fn: _backward_fn_type):
        if from_fn not in self.relation_dict:
            self.relation_dict[from_fn] = set()

        self.relation_dict[from_fn].add(to_fn)

    def _backward_pass_hook(self, grad_output: Tensor):
        with torch.no_grad():
            to_visit_with = set()
            to_visit_with.add((self.OUTPUT, grad_output))

            while len(to_visit_with) > 0:
                function_id, function_grad_output = to_visit_with.pop()

                if function_id not in self.relation_dict:
                    continue

                for func in self.relation_dict[function_id]:
                    new_pair = (func, func(function_grad_output))
                    to_visit_with.add(new_pair)

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
