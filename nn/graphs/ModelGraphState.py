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

        self._inputs: Union[None, Tensor] = None
        self._output: Union[None, Tensor] = None
        self._loss: Union[None, Tensor] = None

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
        return self._inputs

    @property
    def output(self):
        return self._output

    @property
    def loss(self):
        return self._loss

    def set_inputs(self, *inputs):
        self._inputs = inputs
        return self._inputs

    def set_outputs(self, output: Union[Tensor, None]):
        with torch.no_grad():
            if self.use_autograd_graph or output is None:
                self._output = output
            else:
                self._output = output.detach()
                self._output.requires_grad = True
        return self._output

    def set_loss(self, loss: Union[Tensor, None]):
        self._loss = loss
        return self._loss
