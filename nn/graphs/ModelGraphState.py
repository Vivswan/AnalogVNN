from typing import Union, Dict, Optional

import torch
from torch import Tensor

from nn.graphs.ArgsKwargs import InputOutput
from nn.graphs.GraphEnum import GraphEnum


class ModelGraphState:
    allow_loops: bool
    use_autograd_graph: bool
    forward_input_output_graph: Optional[Dict[Union[GraphEnum, torch.nn.Module], InputOutput]]
    _loss: Optional[Tensor]

    INPUT = GraphEnum.INPUT
    OUTPUT = GraphEnum.OUTPUT
    STOP = GraphEnum.STOP

    def __init__(self, use_autograd_graph: bool = False, allow_loops=False):
        self.allow_loops = allow_loops
        self.use_autograd_graph = use_autograd_graph

        self.forward_input_output_graph = None
        self._loss = None

    def ready_for_forward(self, exception=False):
        pass

    def ready_for_backward(self, exception=False):
        if exception:
            if self.output is None:
                raise Exception("output is not set.")

            if self._loss is None:
                raise Exception("loss is not set.")
        else:
            return not (self.output is None or self._loss is None)

    @property
    def inputs(self):
        if self.INPUT not in self.forward_input_output_graph:
            return None
        return self.forward_input_output_graph[self.INPUT].inputs

    @property
    def output(self):
        if self.OUTPUT not in self.forward_input_output_graph:
            return None
        return self.forward_input_output_graph[self.OUTPUT].outputs

    @property
    def loss(self):
        return self._loss

    def set_loss(self, loss: Union[Tensor, None]):
        self._loss = loss
        return self._loss
