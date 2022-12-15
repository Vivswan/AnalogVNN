import copy
import uuid
from typing import Dict, Any

import networkx as nx
import torch

from nn.graphs.AcyclicDirectedGraph import AcyclicDirectedGraph
from nn.graphs.GraphEnum import GraphEnum
from nn.graphs.InputOutput import ArgsKwargs, InputOutput


class ArgsKwargsCalculator:
    def __init__(self, module):
        self.locations = {}
        self.module = module

    def __repr__(self):
        return f"AccumulateGrad"
        # return f"ArgsKwargsCalculator({self.module})"

    def grad(self, grad_outputs_args_kwargs: ArgsKwargs, forward_input_output_graph):
        grad_inputs_args = {}
        grad_inputs_kwargs = {}
        for key, grad_output in grad_outputs_args_kwargs.kwargs.items():
            location = self.locations[key]
            forward_out_arg = location['in_arg']
            forward_out_kwarg = location['in_kwarg']
            forward_in_arg = location['out_arg']
            forward_in_kwarg = location['out_kwarg']
            # print(out_kwarg, out_arg, value)

            # 0 - not allowed

            # 4
            if forward_out_arg is True and isinstance(forward_in_arg, int) and not isinstance(forward_in_arg, bool):
                forward_inputs = forward_input_output_graph[location["from"]].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.args
                forward_out_arg = forward_inputs.index(forward_outputs[forward_in_arg])
                grad_output = grad_output[forward_out_arg]

            # 7
            if forward_out_arg is True and isinstance(forward_in_kwarg, str):
                forward_inputs = forward_input_output_graph[location["from"]].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.kwargs
                forward_out_arg = forward_inputs.index(forward_outputs[forward_in_kwarg])
                grad_output = grad_output[forward_out_arg]

            # 1
            if forward_out_arg is True and forward_in_arg is True:
                forward_inputs = forward_input_output_graph[location["from"]].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.args
                for i in range(len(forward_inputs)):
                    value_index = forward_outputs.index(forward_inputs[i]) if forward_inputs[i] in forward_outputs else -1
                    if value_index == -1:
                        continue
                    if value_index not in grad_inputs_args:
                        grad_inputs_args[value_index] = torch.zeros_like(grad_output[i])
                    grad_inputs_args[value_index] += grad_output[i]
                continue

            # 2
            if forward_out_arg is True and forward_in_kwarg is True:
                forward_inputs = forward_input_output_graph[location["from"]].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.kwargs
                for i in forward_outputs:
                    value_index = forward_inputs.index(forward_outputs[i])

                    if i not in grad_inputs_kwargs:
                        grad_inputs_kwargs[i] = torch.zeros_like(grad_output[value_index])
                    grad_inputs_kwargs[i] += grad_output[value_index]
                continue

            # 3
            if forward_out_kwarg is True and forward_in_kwarg is True:
                for i in grad_output:
                    if i not in grad_inputs_kwargs:
                        grad_inputs_kwargs[i] = torch.zeros_like(grad_output[i])

                    grad_inputs_kwargs[i] += grad_output[i]
                continue

            # 8 & 9
            if forward_in_kwarg is not None and isinstance(forward_in_kwarg, str):
                if forward_in_kwarg not in grad_inputs_kwargs:
                    grad_inputs_kwargs[forward_in_kwarg] = torch.zeros_like(grad_output)

                grad_inputs_kwargs[forward_in_kwarg] += grad_output
                continue

            # 5 & 6
            if forward_in_arg is not None and isinstance(forward_in_arg, int) and not isinstance(forward_in_arg, bool):
                if forward_in_arg not in grad_inputs_args:
                    grad_inputs_args[forward_in_arg] = torch.zeros_like(grad_output)
                grad_inputs_args[forward_in_arg] += grad_output
                continue

            raise NotImplementedError("WTF!Why!")

        return ArgsKwargs(
            args=[grad_inputs_args[i] for i in sorted(list(grad_inputs_args.keys()))],
            kwargs=grad_inputs_kwargs
        )


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

    def from_forward(self, forward_graph, real_label=False):
        if isinstance(forward_graph, AcyclicDirectedGraph):
            forward_graph = forward_graph.graph

        graph = forward_graph.reverse(copy=True)
        for _, _, attr in graph.edges(data=True):
            attr["in_arg"], attr["out_arg"] = attr["out_arg"], attr["in_arg"]
            attr["in_kwarg"], attr["out_kwarg"] = attr["out_kwarg"], attr["in_kwarg"]
            attr["label"] = " ".join(attr["label"].split(" ")[::-1])

        new_graph = nx.MultiDiGraph()
        for v in graph.nodes():
            if v == self.OUTPUT:
                continue
            all_predecessors = list(graph.predecessors(v))
            akc = ArgsKwargsCalculator(v)
            for u in all_predecessors:
                for key, attr in graph.get_edge_data(u, v).items():
                    if attr["in_arg"] is None or attr["in_kwarg"] is None:
                        uuid_str = str(uuid.uuid4()).replace("-", "")
                        new_graph.add_edge(u, akc, **{
                            "in_arg": attr["in_arg"],
                            "in_kwarg": attr["in_kwarg"],
                            "out_arg": None,
                            "out_kwarg": uuid_str,
                            "label": " ".join(attr["label"].split(" ")[:-1] + ["{" + uuid_str + "}"]) if real_label else attr["label"],
                        })
                        akc.locations[uuid_str] = {
                            **attr,
                            "from": u,
                        }
                    else:
                        uuid_str = str(uuid.uuid4()).replace("-", "")
                        new_graph.add_edge(u, akc, **{
                            "in_arg": True,
                            "in_kwarg": None,
                            "out_arg": None,
                            "out_kwarg": uuid_str,
                            "label": "[] -> {" + uuid_str + "}" if real_label else "[] -> []",
                        })
                        akc.locations[uuid_str] = {
                            **attr,
                            "in_kwarg": None,
                            "out_kwarg": None,
                            "from": u,
                        }
                        uuid_str = str(uuid.uuid4()).replace("-", "")
                        new_graph.add_edge(u, akc, **{
                            "in_arg": None,
                            "in_kwarg": True,
                            "out_arg": None,
                            "out_kwarg": uuid_str,
                            "label": "{} -> {" + uuid_str + "}" if real_label else "{} -> {}",
                        })
                        akc.locations[uuid_str] = {
                            **attr,
                            "in_arg": None,
                            "out_arg": None,
                            "from": u,
                        }

            new_graph.add_edge(akc, v, **{
                "in_arg": True,
                "in_kwarg": True,
                "out_arg": True,
                "out_kwarg": True,
                # "label": f"* -> *",
                "len": 0,
            })
            new_graph.nodes[v]["fillcolor"] = "lightblue"
        self.graph = new_graph
        # self.graph = graph
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

    def gradient_wrapper(self, module, grads):
        # if isinstance(module, Layer):
        #     if module.get_backward_module() is not None:
        #         return module.get_backward_module().backward
        # if isinstance(module, BackwardFunction):
        #     return module.backward
        # if inspect.ismethod(module) or inspect.isfunction(module):
        #     return module
        # raise NotImplementedError
        if module in self.graph_state.forward_input_output_graph:
            module_inputs_outputs = self.graph_state.forward_input_output_graph[module]
        else:
            module_inputs_outputs = None

        grad_dict = {}
        if isinstance(module, ArgsKwargsCalculator):
            return module.grad(
                forward_input_output_graph=self.graph_state.forward_input_output_graph,
                grad_outputs_args_kwargs=grads.inputs,
            )

        inputs = module_inputs_outputs.inputs.args + list(module_inputs_outputs.inputs.kwargs.values())
        outputs = []
        outputs_grads = []

        for i in range(len(module_inputs_outputs.outputs.args)):
            outputs.append(module_inputs_outputs.outputs.args[i])
            outputs_grads.append(grads.inputs.args[i])

        for i in module_inputs_outputs.outputs.kwargs:
            outputs.append(module_inputs_outputs.outputs.kwargs[i])
            outputs_grads.append(grads.inputs.kwargs[i])

        out_grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=outputs_grads,
            retain_graph=True,
        )
        # print()
        # print(f"inputs: {inputs}")
        # print(f"outputs: {outputs}")
        # print(f"grad_outputs: {outputs_grads}")
        for i, v in enumerate(out_grads):
            grad_dict[inputs[i]] = v

        grad_output = ArgsKwargs(
            args=[grad_dict[i] for i in module_inputs_outputs.inputs.args],
            kwargs={key: grad_dict[value] for key, value in module_inputs_outputs.inputs.kwargs.items()}
        )
        return grad_output
