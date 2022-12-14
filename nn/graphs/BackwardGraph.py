import copy
from typing import Dict, Any

import torch

from nn.graphs.AcyclicDirectedGraph import AcyclicDirectedGraph
from nn.graphs.GraphEnum import GraphEnum
from nn.graphs.InputOutput import ArgsKwargs, InputOutput


class BackwardGraph(AcyclicDirectedGraph):
    def __call__(self, gradient=None, *args, **kwargs):
        self.graph_state.ready_for_backward(exception=True)

        if len(gradient) == 0:
            gradient = None
        elif len(gradient) == 1:
            gradient = gradient[0]

        if self.graph_state.use_autograd_graph:
            result = self.graph_state.loss.backward(
                gradient=gradient,
                inputs=self.graph_state.output
            )
        else:
            self.graph_state.loss.backward(
                gradient=gradient,
                inputs=self.graph_state.output
            )
            result = self.calculate_graph()

        self.graph_state.set_loss(None)

        return result

    def from_forward(self, forward_graph):
        if isinstance(forward_graph, AcyclicDirectedGraph):
            forward_graph = forward_graph.graph
        self.graph = forward_graph.reverse(copy=True)
        for u, v, d in self.graph.edges(data=True):
            dd = copy.deepcopy(d)
            d.clear()
            d["in_arg"] = dd["out_arg"]
            d["in_kwarg"] = dd["out_kwarg"]
            d["out_arg"] = dd["in_arg"]
            d["out_kwarg"] = dd["in_kwarg"]
            d["label"] = " ".join(dd["label"].split(" ")[::-1])

        return self

    def compile(self, is_static=True, **kwargs):
        if not self.graph.has_node(self.OUTPUT):
            raise Exception("OUTPUT doesn't exist in the forward graph")

        return super().compile(self.OUTPUT, is_static)

    @torch.no_grad()
    def calculate_graph(self):
        if self.graph_state.forward_input_output_graph is None:
            raise Exception("No forward pass has been performed yet")

        input_output_graph = self._pass(self.OUTPUT)
        self.graph_state.forward_input_output_graph = None

        if self.INPUT in input_output_graph:
            inputs = input_output_graph[self.INPUT].outputs
            if len(inputs.kwargs.keys()) > 0:
                return inputs
            elif len(inputs.args) > 1:
                return inputs.args
            elif len(inputs.args) == 1:
                return inputs.args[0]

        return None

    def _pass(self, from_node) -> Dict[Any, InputOutput]:
        static_graph = self._static_graph or self._create_sub_graph(from_node)
        from_node_inputs = self.graph_state.forward_input_output_graph[from_node].inputs
        input_output_graph = {
            from_node: InputOutput(inputs=ArgsKwargs(
                args=[i.grad for i in from_node_inputs.args],
                kwargs={k: v.grad for k, v in from_node_inputs.kwargs.items()}
            ))
        }
        for module, predecessors in static_graph:
            if module != from_node:
                inputs = self.get_args_kwargs(input_output_graph, module, predecessors)
                input_output_graph[module] = InputOutput(inputs=inputs)

            if isinstance(module, GraphEnum):
                input_output_graph[module].outputs = input_output_graph[module].inputs
                print()
                self.print_inputs_outputs(input_output_graph, module)
                continue

            outputs = self.gradient_wrapper(
                module,
                input_output_graph[module]
            )
            input_output_graph[module].outputs = self.output_to_args_kwargs(outputs)
            self.print_inputs_outputs(input_output_graph, module)

        return input_output_graph

    def gradient_wrapper(self, module, input_output):
        # if isinstance(module, Layer):
        #     if module.get_backward_module() is not None:
        #         return module.get_backward_module().backward
        # if isinstance(module, BackwardFunction):
        #     return module.backward
        # if inspect.ismethod(module) or inspect.isfunction(module):
        #     return module
        # raise NotImplementedError
        module_inputs = self.graph_state.forward_input_output_graph[module].inputs
        module_outputs = self.graph_state.forward_input_output_graph[module].outputs
        outputs = module_outputs.args + list(module_outputs.kwargs.values())
        inputs = module_inputs.args + list(module_inputs.kwargs.values())
        outputs_grads = [i.grad for i in outputs]
        # grad_dict = {}

        if len(inputs) == 0:
            return ArgsKwargs()

        # for i in range(len(self.forward_inputs_outputs.outputs.args)):
        #     outputs.append(self.forward_inputs_outputs.outputs.args[i])
        #     outputs_grads.append(args[i])
        # for i in self.forward_inputs_outputs.outputs.kwargs:
        #     outputs.append(self.forward_inputs_outputs.outputs.kwargs[i])
        #     outputs_grads.append(kwargs[i])

        out_grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=outputs_grads,
            retain_graph=True,
        )
        for i, j in zip(inputs, out_grads):
            if i.grad is None:
                i.grad = j
            else:
                i.grad += j
        # print()
        # print(f"inputs: {inputs}")
        # print(f"outputs: {outputs}")
        # print(f"grad_outputs: {outputs_grads}")
        # for i, v in enumerate(out_grads):
        #     grad_dict[inputs[i]] = v

        grad_output = ArgsKwargs(
            args=[i.grad for i in module_inputs.args],
            kwargs={key: value.grad for key, value in module_inputs.kwargs.items()}
        )
        return grad_output
