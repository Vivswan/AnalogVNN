from __future__ import annotations

import uuid
from typing import Dict, Union, Callable, List, Tuple

import networkx as nx
import torch
from torch import nn

from analogvnn.backward.BackwardModule import BackwardModule
from analogvnn.graph.AccumulateGrad import AccumulateGrad
from analogvnn.graph.AcyclicDirectedGraph import AcyclicDirectedGraph
from analogvnn.graph.ArgsKwargs import ArgsKwargs, InputOutput, ArgsKwargsOutput
from analogvnn.graph.GraphEnum import GraphEnum, GRAPH_NODE_TYPE
from analogvnn.nn.module.Layer import Layer
from analogvnn.utils.common_types import TENSORS

__all__ = ['BackwardGraph']


class BackwardGraph(AcyclicDirectedGraph):
    """The backward graph."""

    def __call__(self, gradient: TENSORS = None) -> ArgsKwargsOutput:
        """Backward pass through the backward graph.

        Args:
            gradient (TENSORS): gradient of the loss function w.r.t. the output of the forward graph

        Returns:
            ArgsKwargsOutput: gradient of the inputs function w.r.t. loss
        """

        self.graph_state.ready_for_backward(exception=True)

        if len(gradient) == 0:
            gradient = None
        elif len(gradient) == 1:
            gradient = gradient[0]

        if self.graph_state.use_autograd_graph:
            result = self.graph_state.loss.backward(gradient=gradient)
        else:
            grad_outputs = torch.autograd.grad(
                outputs=self.graph_state.loss,
                inputs=self.graph_state.outputs.args,
                grad_outputs=gradient,
                retain_graph=True
            )
            result = self.calculate(*grad_outputs)

        self.graph_state.set_loss(None)

        return result

    def compile(self, is_static=True):
        """Compile the graph.

        Args:
            is_static (bool): If True, the graph is not changing during runtime and will be cached.

        Returns:
            BackwardGraph: self.

        Raises:
            ValueError: If no forward pass has been performed yet.
        """

        if not self.graph.has_node(self.OUTPUT):
            raise ValueError("OUTPUT doesn't exist in the forward graph. Please preform a forward pass first.")

        return super().compile(is_static=is_static)

    def from_forward(self, forward_graph: Union[AcyclicDirectedGraph, nx.DiGraph]) -> BackwardGraph:  # noqa: C901
        """Create a backward graph from inverting forward graph.

        Args:
            forward_graph (Union[AcyclicDirectedGraph, nx.DiGraph]): The forward graph.

        Returns:
            BackwardGraph: self.
        """

        if isinstance(forward_graph, AcyclicDirectedGraph):
            forward_graph = forward_graph.graph

        graph = forward_graph.reverse(copy=True)
        for _, _, attr in graph.edges(data=True):
            attr['in_arg'], attr['out_arg'] = attr['out_arg'], attr['in_arg']
            attr['in_kwarg'], attr['out_kwarg'] = attr['out_kwarg'], attr['in_kwarg']
            attr['label'] = AcyclicDirectedGraph._create_edge_label(**attr)

        new_graph = nx.MultiDiGraph()
        for v in graph.nodes():
            if v == self.OUTPUT:
                continue
            all_predecessors = list(graph.predecessors(v))

            if len(all_predecessors) == 1 and len(graph.get_edge_data(all_predecessors[0], v)) == 1:
                attr = graph.get_edge_data(all_predecessors[0], v)[0]
                if attr['in_arg'] == attr['in_kwarg'] == attr['in_arg'] == attr['in_arg'] is True:
                    new_graph.add_edge(all_predecessors[0], v, **attr)
                    continue

            akc = AccumulateGrad(v)
            for u in all_predecessors:
                for _, attr in graph.get_edge_data(u, v).items():
                    if attr['in_arg'] is None or attr['in_kwarg'] is None:
                        uuid_str = str(uuid.uuid4()).replace('-', '')
                        new_graph.add_edge(u, akc, **{
                            'in_arg': attr['in_arg'],
                            'in_kwarg': attr['in_kwarg'],
                            'out_arg': None,
                            'out_kwarg': uuid_str,
                            'real_label': ' '.join(attr['label'].split(' ')[:-1] + ['{' + uuid_str + '}']),
                            'label': attr['label']
                        })
                        akc.input_output_connections[uuid_str] = {
                            **attr,
                            'from': u,
                        }
                    else:
                        uuid_str = str(uuid.uuid4()).replace('-', '')
                        new_graph.add_edge(u, akc, **{
                            'in_arg': True,
                            'in_kwarg': None,
                            'out_arg': None,
                            'out_kwarg': uuid_str,
                            'real_label': '[] -> {' + uuid_str + '}',
                            'label': '[] -> []',
                        })
                        akc.input_output_connections[uuid_str] = {
                            **attr,
                            'in_kwarg': None,
                            'out_kwarg': None,
                            'from': u,
                        }
                        uuid_str = str(uuid.uuid4()).replace('-', '')
                        new_graph.add_edge(u, akc, **{
                            'in_arg': None,
                            'in_kwarg': True,
                            'out_arg': None,
                            'out_kwarg': uuid_str,
                            'real_label': '{} -> {' + uuid_str + '}',
                            'label': '{} -> {}',
                        })
                        akc.input_output_connections[uuid_str] = {
                            **attr,
                            'in_arg': None,
                            'out_arg': None,
                            'from': u,
                        }

            new_graph.add_edge(akc, v, **{
                'in_arg': True,
                'in_kwarg': True,
                'out_arg': True,
                'out_kwarg': True,
                'len': 0,
            })

        for v in graph.nodes():
            new_graph.nodes[v]['fillcolor'] = 'lightblue'
        self.graph = new_graph
        return self

    @torch.no_grad()
    def calculate(self, *args, **kwargs) -> ArgsKwargsOutput:
        """Calculate the gradient of the whole graph w.r.t. loss.

        Args:
            *args: The gradients args of outputs.
            **kwargs: The gradients kwargs of outputs.

        Returns:
            ArgsKwargsOutput: The gradient of the inputs function w.r.t. loss.

        Raises:
            ValueError: If no forward pass has been performed yet.
        """

        if self.graph_state.forward_input_output_graph is None:
            raise ValueError('No forward pass has been performed yet. Please preform a forward pass first.')

        input_output_graph = self._pass(self.OUTPUT, *args, **kwargs)
        self.graph_state.forward_input_output_graph = None

        if self.INPUT in input_output_graph:
            return ArgsKwargs.from_args_kwargs_object(input_output_graph[self.INPUT].outputs)
        else:
            return None

    def _pass(self, from_node: GRAPH_NODE_TYPE, *args, **kwargs) -> Dict[GRAPH_NODE_TYPE, InputOutput]:
        """Perform the backward pass through the graph.

        Args:
            from_node (GRAPH_NODE_TYPE): The node to start the backward pass from.
            *args: The gradients args of outputs.
            **kwargs: The gradients kwargs of outputs.

        Returns:
            Dict[GRAPH_NODE_TYPE, InputOutput]: The input and output gradients of each node.
        """

        static_graph: List[Tuple[GRAPH_NODE_TYPE, List[GRAPH_NODE_TYPE]]] = self._create_static_sub_graph(from_node)
        input_output_graph: Dict[GRAPH_NODE_TYPE, InputOutput] = {
            from_node: InputOutput(inputs=ArgsKwargs(
                args=args,
                kwargs=kwargs
            ))
        }
        for module, predecessors in static_graph:
            if module != from_node:
                inputs = self.parse_args_kwargs(input_output_graph, module, predecessors)
                input_output_graph[module] = InputOutput(inputs=inputs)

            if isinstance(module, GraphEnum):
                input_output_graph[module].outputs = input_output_graph[module].inputs
                continue

            outputs = self._calculate_gradients(
                module,
                input_output_graph[module]
            )
            input_output_graph[module].outputs = ArgsKwargs.to_args_kwargs_object(outputs)

        return input_output_graph

    def _calculate_gradients(  # noqa: C901
            self,
            module: Union[AccumulateGrad, Layer, BackwardModule, Callable],
            grad_outputs: InputOutput
    ) -> ArgsKwargs:
        """Calculate the gradient of a module w.r.t. outputs of the module using the output's gradients.

        Args:
            module (Union[AccumulateGrad, Layer, BackwardModule, Callable]): The module to calculate the gradient of.
            grad_outputs (InputOutput): The gradients of the output of the module.

        Returns:
            ArgsKwargs: The input gradients of the module.
        """

        if module in self.graph_state.forward_input_output_graph:
            module_inputs_outputs = self.graph_state.forward_input_output_graph[module]
        else:
            module_inputs_outputs = None

        if grad_outputs.inputs.is_empty():
            return ArgsKwargs()

        if isinstance(module, AccumulateGrad):
            return module.grad(
                grad_outputs_args_kwargs=grad_outputs.inputs,
                forward_input_output_graph=self.graph_state.forward_input_output_graph,
            )

        if isinstance(module, Layer) and isinstance(module.backward_function, BackwardModule):
            module = module.backward_function

        if isinstance(module, BackwardModule):
            grad_inputs = module._call_impl_backward(*grad_outputs.inputs.args, **grad_outputs.inputs.kwargs)
            return ArgsKwargs.to_args_kwargs_object(grad_inputs)

        grad_dict = {}
        inputs = module_inputs_outputs.inputs.args + list(module_inputs_outputs.inputs.kwargs.values())
        outputs = []
        outputs_grads = []
        module_parameters = []

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

        if isinstance(module, nn.Module):
            module_parameters = list(module.parameters())
            inputs += module_parameters

        out_grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=outputs_grads,
            retain_graph=True,
            allow_unused=True
        )
        for i, v in enumerate(out_grads):
            grad_dict[inputs[i]] = v

        for i in module_parameters:
            if grad_dict[i] is None:
                continue

            if i.grad is None:
                i.grad = grad_dict[i]
            else:
                i.grad += grad_dict[i]

        return ArgsKwargs(
            args=[grad_dict[i] for i in module_inputs_outputs.inputs.args],
            kwargs={key: grad_dict[value] for key, value in module_inputs_outputs.inputs.kwargs.items()}
        )
