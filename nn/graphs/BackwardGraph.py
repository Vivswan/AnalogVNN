import inspect
import uuid
from typing import Dict, Any

import networkx as nx
import torch

from nn.graphs.AccumulateGrad import AccumulateGrad
from nn.graphs.AcyclicDirectedGraph import AcyclicDirectedGraph
from nn.graphs.GraphEnum import GraphEnum
from nn.graphs.InputOutput import InputOutput
from nn.graphs.ArgsKwargs import ArgsKwargs
from nn.modules.Layer import BackwardFunction, Layer


class BackwardGraph(AcyclicDirectedGraph):
    def __call__(self, gradient=None):
        self.graph_state.ready_for_backward(exception=True)

        if len(gradient) == 0:
            gradient = None
        elif len(gradient) == 1:
            gradient = gradient[0]

        if self.graph_state.use_autograd_graph:
            result = self.graph_state.loss.backward(
                gradient=gradient,
                inputs=self.graph_state.output.args
            )
        else:
            grad_outputs = torch.autograd.grad(
                outputs=self.graph_state.loss,
                inputs=self.graph_state.output.args,
                grad_outputs=gradient,
                retain_graph=True
            )
            result = self.calculate_graph(*grad_outputs)

        self.graph_state.set_loss(None)

        return result

    def from_forward(self, forward_graph):
        if isinstance(forward_graph, AcyclicDirectedGraph):
            forward_graph = forward_graph.graph

        graph = forward_graph.reverse(copy=True)
        for _, _, attr in graph.edges(data=True):
            attr["in_arg"], attr["out_arg"] = attr["out_arg"], attr["in_arg"]
            attr["in_kwarg"], attr["out_kwarg"] = attr["out_kwarg"], attr["in_kwarg"]
            attr["label"] = AcyclicDirectedGraph._create_edge_label(**attr)

        new_graph = nx.MultiDiGraph()
        for v in graph.nodes():
            if v == self.OUTPUT:
                continue
            all_predecessors = list(graph.predecessors(v))

            if len(all_predecessors) == 1 and len(graph.get_edge_data(all_predecessors[0], v)) == 1:
                attr = graph.get_edge_data(all_predecessors[0], v)[0]
                if attr["in_arg"] == attr["in_kwarg"] == attr["in_arg"] == attr["in_arg"] == True:
                    new_graph.add_edge(all_predecessors[0], v, **attr)
                    continue

            akc = AccumulateGrad(v)
            for u in all_predecessors:
                for key, attr in graph.get_edge_data(u, v).items():
                    if attr["in_arg"] is None or attr["in_kwarg"] is None:
                        uuid_str = str(uuid.uuid4()).replace("-", "")
                        new_graph.add_edge(u, akc, **{
                            "in_arg": attr["in_arg"],
                            "in_kwarg": attr["in_kwarg"],
                            "out_arg": None,
                            "out_kwarg": uuid_str,
                            "real_label": " ".join(attr["label"].split(" ")[:-1] + ["{" + uuid_str + "}"]),
                            "label": attr["label"]
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
                            "real_label": "[] -> {" + uuid_str + "}",
                            "label": "[] -> []",
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
                            "real_label": "{} -> {" + uuid_str + "}",
                            "label": "{} -> {}",
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

        for v in graph.nodes():
            new_graph.nodes[v]["fillcolor"] = "lightblue"
        self.graph = new_graph
        # self.graph = graph
        return self

    def compile(self, is_static=True, **kwargs):
        if not self.graph.has_node(self.OUTPUT):
            raise Exception("OUTPUT doesn't exist in the forward graph")

        return super().compile(self.OUTPUT, is_static)

    @torch.no_grad()
    def calculate_graph(self, *args, **kwargs):
        if self.graph_state.forward_input_output_graph is None:
            raise Exception("No forward pass has been performed yet")

        input_output_graph = self._pass(self.OUTPUT, *args, **kwargs)
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

    def _pass(self, from_node, *args, **kwargs) -> Dict[Any, InputOutput]:
        static_graph = self._static_graph or self._create_sub_graph(from_node)
        input_output_graph = {
            from_node: InputOutput(inputs=ArgsKwargs(
                args=args,
                kwargs=kwargs
            ))
        }
        for module, predecessors in static_graph:
            if module != from_node:
                inputs = self.get_args_kwargs(input_output_graph, module, predecessors)
                input_output_graph[module] = InputOutput(inputs=inputs)

            if isinstance(module, GraphEnum):
                input_output_graph[module].outputs = input_output_graph[module].inputs
                # print()
                # self.print_inputs_outputs(input_output_graph, module)
                continue

            outputs = self.gradient_wrapper(
                module,
                input_output_graph[module]
            )
            input_output_graph[module].outputs = self.output_to_args_kwargs(outputs)
            # self.print_inputs_outputs(input_output_graph, module)

        return input_output_graph

    def gradient_wrapper(self, module, grad_outputs):
        if module in self.graph_state.forward_input_output_graph:
            module_inputs_outputs = self.graph_state.forward_input_output_graph[module]
        else:
            module_inputs_outputs = None

        if grad_outputs.inputs.is_empty():
            return ArgsKwargs()

        if isinstance(module, AccumulateGrad):
            return module.grad(
                forward_input_output_graph=self.graph_state.forward_input_output_graph,
                grad_outputs_args_kwargs=grad_outputs.inputs,
            )

        if isinstance(module, BackwardFunction):
            grad_inputs = module.backward(*grad_outputs.inputs.args, **grad_outputs.inputs.kwargs)
            return self.output_to_args_kwargs(grad_inputs)

        if isinstance(module, Layer) and module.get_backward_module() is not None:
            grad_inputs = module.get_backward_module().backward(*grad_outputs.inputs.args, **grad_outputs.inputs.kwargs)
            return self.output_to_args_kwargs(grad_inputs)

        if (inspect.ismethod(module) or inspect.isfunction(module)) and not inspect.isclass(module):
            grad_inputs = module(*grad_outputs.inputs.args, **grad_outputs.inputs.kwargs)
            return self.output_to_args_kwargs(grad_inputs)

        grad_dict = {}
        inputs = module_inputs_outputs.inputs.args + list(module_inputs_outputs.inputs.kwargs.values())
        outputs = []
        outputs_grads = []

        if len(inputs) == 0:
            return ArgsKwargs(
                args=[torch.zeros_like(i) for i in module_inputs_outputs.inputs.args],
                kwargs={key: torch.zeros_like(value) for key, value in module_inputs_outputs.inputs.kwargs.items()}
            )

        for i in range(len(module_inputs_outputs.outputs.args)):
            outputs.append(module_inputs_outputs.outputs.args[i])
            outputs_grads.append(grad_outputs.inputs.args[i])

        for i in module_inputs_outputs.outputs.kwargs:
            outputs.append(module_inputs_outputs.outputs.kwargs[i])
            outputs_grads.append(grad_outputs.inputs.kwargs[i])

        out_grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=outputs_grads,
            retain_graph=True,
            allow_unused=True
        )
        # print()
        # print(f"inputs: {inputs}")
        # print(f"outputs: {outputs}")
        # print(f"grad_outputs: {outputs_grads}")
        for i, v in enumerate(out_grads):
            grad_dict[inputs[i]] = v

        return ArgsKwargs(
            args=[grad_dict[i] for i in module_inputs_outputs.inputs.args],
            kwargs={key: grad_dict[value] for key, value in module_inputs_outputs.inputs.kwargs.items()}
        )
