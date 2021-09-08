import inspect
from typing import Set, List

import torch
from torch import Tensor

from nn.backward_pass.BackwardFunction import BackwardFunction
from nn.backward_pass.BaseBackwardPass import BaseBackwardPass, _backward_fn_type


class TreeBackwardPass(BaseBackwardPass):
    def __init__(self):
        super(TreeBackwardPass, self).__init__()
        self.relation_set: Set[List[_backward_fn_type]] = set()

    def add_relation(self, start_at: str, *args: _backward_fn_type):
        if start_at != self.OUTPUT:
            raise Exception("tree backward relation must originate at output")

        if len(args) == 0:
            return

        self.relation_set.add(list(args))

    def compile(self):
        pass

    @torch.no_grad()
    def _backward_pass_hook(self, grad_output: Tensor):
        for backward_list in self.relation_set:
            current_grad_output = grad_output
            for func in backward_list:
                if isinstance(func, BackwardFunction):
                    current_grad_output = func.backward(current_grad_output)
                elif inspect.ismethod(func) or inspect.isfunction(func):
                    current_grad_output = func(current_grad_output)
                else:
                    raise NotImplementedError

        self._completed_backward_pass()
