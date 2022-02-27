from typing import Union, Dict, Set, Tuple

import torch
from torch import Tensor

from nn.backward_pass.BackwardFunction import BackwardFunction
from nn.backward_pass.BaseBackwardPass import BaseBackwardPass, _backward_fn_type


class GraphBackwardPass(BaseBackwardPass):
    def __init__(self):
        super(GraphBackwardPass, self).__init__()
        self.relation_dict: Dict[Union[_backward_fn_type, str], Set[_backward_fn_type]] = {}

    def add_relation(self, *args: Union[str, BackwardFunction]):
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
