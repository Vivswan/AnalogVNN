import enum

import torch
from torch import Tensor
from typing import Union

from nn.graphs.TensorFlowGraphEnum import TensorFlowGraphEnum


class ModelGraphState:
    INPUT = TensorFlowGraphEnum.INPUT
    OUTPUT = TensorFlowGraphEnum.OUTPUT
    STOP = TensorFlowGraphEnum.STOP

    def __init__(self, allow_loops=False, use_autograd_graph: bool = False):
        self.allow_loops = allow_loops
        self.use_autograd_graph: bool = use_autograd_graph

        self.input: Union[None, Tensor] = None
        self._output: Union[None, Tensor] = None
        self.loss: Union[None, Tensor] = None
        self.output_hook = None

    def ready_for_forward(self, exception=False):
        pass

    def ready_for_backward(self, exception=False):
        if exception:
            if self.output is None:
                raise Exception("output is not set.")

            if self.loss is None:
                raise Exception("loss is not set.")
        else:
            return not (self.output is None or self.loss is None)

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, output: Tensor):
        with torch.no_grad():
            if self.output_hook is not None:
                self.output_hook.remove()

            if self.use_autograd_graph or output is None:
                self._output = output
            else:
                self._output = output.detach()
                self._output.requires_grad = True

    def set_outputs(self, output: Tensor):
        self.output = output
        return self.output
