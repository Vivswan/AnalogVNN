from typing import Dict, Any

import torch

from nn.graphs.AcyclicDirectedGraph import AcyclicDirectedGraph
from nn.graphs.GraphEnum import GraphEnum
from nn.graphs.InputOutput import ArgsKwargs, InputOutput
from nn.graphs.ModelGraphState import ModelGraphState


class ForwardGraph(AcyclicDirectedGraph):
    def __call__(self, inputs, is_training, **kwargs):
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

    def calculate_graph(self, inputs, is_training=True, **kwargs):
        if is_training:
            input_output_graph = self._pass(inputs, self.INPUT)
            self.graph_state.forward_input_output_graph = input_output_graph
        else:
            with torch.no_grad():
                input_output_graph = self._pass(inputs, self.INPUT)

        output = input_output_graph[self.OUTPUT].outputs
        if len(output.kwargs.keys()) > 0:
            return output
        elif len(output.args) > 1:
            return output.args
        elif len(output.args) == 1:
            return output.args[0]
        else:
            return None

    def _pass(self, inputs, from_node) -> Dict[Any, InputOutput]:
        static_graph = self._static_graph or self._create_sub_graph(from_node)
        input_output_graph = {
            from_node: InputOutput(inputs=ArgsKwargs(args=[inputs]))
        }
        for module, predecessors in static_graph:
            if module != from_node:
                inputs = self.get_args_kwargs(input_output_graph, module, predecessors)
                input_output_graph[module] = InputOutput(inputs=inputs)

            if isinstance(module, GraphEnum):
                input_output_graph[module].outputs = input_output_graph[module].inputs
                self.print_inputs_outputs(input_output_graph, module)
                continue

            outputs = module(
                *input_output_graph[module].inputs.args,
                **input_output_graph[module].inputs.kwargs
            )
            input_output_graph[module].outputs = self.output_to_args_kwargs(outputs)
            self.print_inputs_outputs(input_output_graph, module)

        return input_output_graph


if __name__ == '__main__':
    gb = ForwardGraph(ModelGraphState())


    def l3(x):
        return x * 2


    def l2(x, y):
        return x * y * 3


    def l1(x, y, z):
        return x * y * z * 5


    gb.add_connection(ForwardGraph.INPUT, l3, name="x")
    gb.add_connection(l3, l2, name="y")
    gb.add_connection(ForwardGraph.INPUT, l2, name="x")
    gb.add_connection(ForwardGraph.INPUT, l1, name="x")
    gb.add_connection(l3, l1, name="y")
    gb.add_connection(l2, l1, name="z")
    gb.add_connection(l1, ForwardGraph.OUTPUT, name="OUTPUT")

    gb.compile(is_static=True)
    print(gb._pass(1))
