from typing import Dict, Any, Sequence, Tuple, Union

import torch
from torch import Tensor

from nn.graphs.AcyclicDirectedGraph import AcyclicDirectedGraph
from nn.graphs.GraphEnum import GraphEnum
from nn.graphs.InputOutput import InputOutput
from nn.graphs.ArgsKwargs import ArgsKwargs
from nn.graphs.ModelGraphState import ModelGraphState


class ForwardGraph(AcyclicDirectedGraph):
    def __call__(self, inputs, is_training):
        self.graph_state.ready_for_forward(exception=True)
        outputs = self.calculate_graph(inputs, is_training)
        if len(outputs) == 1:
            outputs = outputs[0][1]

        return outputs

    def compile(self, is_static=True, **kwargs):
        if not self.graph.has_node(self.INPUT):
            raise Exception("INPUT doesn't exist in the forward graph")

        if not self.graph.has_node(self.OUTPUT):
            raise Exception("OUTPUT doesn't exist in the forward graph")

        return super().compile(self.INPUT, is_static)

    def calculate_graph(self, inputs: Union[Tensor, Sequence[Tensor]], is_training=True, **kwargs):
        if not isinstance(inputs, Sequence):
            inputs = (inputs,)

        if not self.graph_state.use_autograd_graph and is_training:
            for i in inputs:
                i.requires_grad = True

        if is_training:
            input_output_graph = self._pass(self.INPUT, *inputs)
            self.graph_state.forward_input_output_graph = input_output_graph
        else:
            with torch.no_grad():
                input_output_graph = self._pass(self.INPUT, *inputs)

        output = input_output_graph[self.OUTPUT].outputs
        if len(output.kwargs.keys()) > 0:
            return output
        elif len(output.args) > 1:
            return output.args
        elif len(output.args) == 1:
            return output.args[0]
        else:
            return None

    def _pass(self, from_node: GraphEnum, *inputs: Tensor) -> Dict[Any, InputOutput]:
        static_graph = self._static_graph or self._create_sub_graph(from_node)
        input_output_graph = {
            from_node: InputOutput(inputs=ArgsKwargs(args=[*inputs]))
        }
        for module, predecessors in static_graph:
            if module != from_node:
                inputs = self.get_args_kwargs(input_output_graph, module, predecessors)
                if not self.graph_state.use_autograd_graph:
                    inputs.args = [self._detach_tensor(i) for i in inputs.args]
                    inputs.kwargs = {k: self._detach_tensor(v) for k, v in inputs.kwargs.items()}
                input_output_graph[module] = InputOutput(inputs=inputs)

            if isinstance(module, GraphEnum):
                input_output_graph[module].outputs = input_output_graph[module].inputs
                # self.print_inputs_outputs(input_output_graph, module)
                continue

            outputs = module(
                *input_output_graph[module].inputs.args,
                **input_output_graph[module].inputs.kwargs
            )
            input_output_graph[module].outputs = self.output_to_args_kwargs(outputs)
            # self.print_inputs_outputs(input_output_graph, module)

        return input_output_graph

    @staticmethod
    def _detach_tensor(tensor: torch.Tensor) -> torch.Tensor:
        tensor: torch.Tensor = tensor.detach()
        tensor.requires_grad = True
        return tensor
