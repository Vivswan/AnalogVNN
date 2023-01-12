from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor

from analogvnn.graph.AcyclicDirectedGraph import AcyclicDirectedGraph
from analogvnn.graph.ArgsKwargs import ArgsKwargs, InputOutput, ArgsKwargsOutput
from analogvnn.graph.GraphEnum import GraphEnum
from analogvnn.utils.common_types import TENSORS

__all__ = ['ForwardGraph']


class ForwardGraph(AcyclicDirectedGraph):
    """The forward graph."""

    def __call__(self, inputs: TENSORS, is_training: bool) -> ArgsKwargsOutput:
        """Forward pass through the forward graph.

        Args:
            inputs (TENSORS): Input to the graph
            is_training (bool): Is training or not

        Returns:
            ArgsKwargsOutput: Output of the graph
        """
        self.graph_state.ready_for_forward(exception=True)
        outputs = self.calculate(inputs, is_training)
        return outputs

    def compile(self, is_static: bool = True):
        """Compile the graph.

        Args:
            is_static (bool): If True, the graph is not changing during runtime and will be cached.

        Returns:
            ForwardGraph: self.

        Raises:
            ValueError: If no forward pass has been performed yet.
        """
        if not self.graph.has_node(self.INPUT):
            raise ValueError("INPUT doesn't exist in the forward graph. Please preform a forward pass first.")

        if not self.graph.has_node(self.OUTPUT):
            raise ValueError("OUTPUT doesn't exist in the forward graph. Please preform a forward pass first.")

        return super().compile(is_static=is_static)

    def calculate(
            self,
            inputs: TENSORS,
            is_training: bool = True,
            **kwargs
    ) -> ArgsKwargsOutput:
        """Calculate the output of the graph.

        Args:
            inputs (TENSORS): Input to the graph
            is_training (bool): Is training or not
            **kwargs: Additional arguments

        Returns:
            ArgsKwargsOutput: Output of the graph
        """
        if not isinstance(inputs, Sequence):
            inputs = (inputs,)

        if not self.graph_state.use_autograd_graph and is_training:
            for i in inputs:
                i.requires_grad = True

        input_output_graph = self._pass(self.INPUT, *inputs)
        if is_training:
            self.graph_state.forward_input_output_graph = input_output_graph

        outputs = input_output_graph[self.OUTPUT].outputs
        return ArgsKwargs.from_args_kwargs_object(outputs)

    def _pass(self, from_node: GraphEnum, *inputs: Tensor) -> Dict[GraphEnum, InputOutput]:
        """Perform the forward pass through the graph.

        Args:
            from_node (GraphEnum): The node to  start the forward pass from
            *inputs (Tensor): Input to the graph

        Returns:
            Dict[GraphEnum, InputOutput]: The input and output of each node
        """
        static_graph = self._create_static_sub_graph(from_node)
        input_output_graph = {
            from_node: InputOutput(inputs=ArgsKwargs(args=[*inputs]))
        }
        for module, predecessors in static_graph:
            if module != from_node:
                inputs = self.parse_args_kwargs(input_output_graph, module, predecessors)
                if not self.graph_state.use_autograd_graph:
                    inputs.args = [self._detach_tensor(i) for i in inputs.args]
                    inputs.kwargs = {k: self._detach_tensor(v) for k, v in inputs.kwargs.items()}
                input_output_graph[module] = InputOutput(inputs=inputs)

            if isinstance(module, GraphEnum):
                input_output_graph[module].outputs = input_output_graph[module].inputs
                continue

            outputs = module(
                *input_output_graph[module].inputs.args,
                **input_output_graph[module].inputs.kwargs
            )
            input_output_graph[module].outputs = ArgsKwargs.to_args_kwargs_object(outputs)

        return input_output_graph

    @staticmethod
    def _detach_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Detach the tensor from the autograd graph.

        Args:
            tensor (torch.Tensor): Tensor to detach

        Returns:
            torch.Tensor: Detached tensor
        """
        tensor: torch.Tensor = tensor.detach()
        tensor.requires_grad = True
        return tensor
