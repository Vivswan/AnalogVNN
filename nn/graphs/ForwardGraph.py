import enum
import inspect
from typing import Union

import networkx as nx
import torch
from torch import Tensor

from nn.graphs.BackwardFunction import BackwardFunction
from nn.graphs.ModelGraphState import ModelGraphState
from nn.graphs.TensorFlowGraph import TensorFlowGraph
from nn.layers.Linear import Linear
from nn.modules.Layer import Layer


class ForwardGraph(TensorFlowGraph):
    def __call__(self, inputs, is_training, **kwargs):
        self._graph_state.ready_for_forward(exception=True)
        outputs = self._pass(inputs)
        if len(outputs) == 1:
            outputs = outputs[0][1]

        if is_training:
            self._graph_state.input = inputs
            outputs = self._graph_state.set_outputs(outputs)

        return outputs

    def compile(self, is_static=True, **kwargs):
        if not self.graph.has_node(self.INPUT):
            raise Exception("INPUT doesn't exist in the forward graph")

        if not self.graph.has_node(self.OUTPUT):
            raise Exception("OUTPUT doesn't exist in the forward graph")

        return super().compile(self.INPUT, is_static)

    def _pass(self, inputs, **kwargs):
        return super()._pass(inputs, self.INPUT, self.OUTPUT)
