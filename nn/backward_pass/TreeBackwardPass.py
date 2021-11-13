from typing import Set, List

import torch
from torch import Tensor

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
    def _backward_pass(self, grad_output: Tensor):
        for backward_list in self.relation_set:
            current_grad_output = grad_output
            for module in backward_list:
                backward_fn = self.get_backward_function(module)

                if backward_fn is None:
                    raise NotImplementedError

                current_grad_output = backward_fn(current_grad_output)
