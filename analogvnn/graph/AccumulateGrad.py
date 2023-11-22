from typing import Dict, Union, Callable, List

import torch
from torch import nn

from analogvnn.graph.ArgsKwargs import ArgsKwargs, InputOutput
from analogvnn.graph.GraphEnum import GRAPH_NODE_TYPE

__all__ = ['AccumulateGrad']


def _get_index(tensor: torch.Tensor, tensor_list: List[torch.Tensor]) -> int:
    return [(i.shape == tensor.shape and torch.all(torch.eq(i, tensor))) for i in tensor_list].index(True)


class AccumulateGrad:
    """AccumulateGrad is a module that accumulates the gradients of the outputs of the module it is attached to.

     It has no parameters of its own.

    Attributes:
        module (nn.Module): Module to accumulate gradients for.
        input_output_connections (Dict[str, Dict[str, Union[None, bool, int, str, GRAPH_NODE_TYPE]]]): input/output
        connections.
    """

    input_output_connections: Dict[str, Dict[str, Union[None, bool, int, str, GRAPH_NODE_TYPE]]]
    module: Union[nn.Module, Callable]

    def __init__(self, module: Union[nn.Module, Callable]):
        """Initialize the module.

        Args:
            module (Union[nn.Module, Callable]): Module from which to accumulate gradients.
        """

        super().__init__()
        self.input_output_connections = {}
        self.module = module

    def __repr__(self):
        """Return a string representation of the module.

        Returns:
            str: String representation of the module.
        """

        return f'{self.__class__.__name__}({self.module})'

    def __call__(  # noqa: C901
            self,
            grad_outputs_args_kwargs: ArgsKwargs,
            forward_input_output_graph: Dict[GRAPH_NODE_TYPE, InputOutput]
    ) -> ArgsKwargs:
        """Calculate and Accumulate the output gradients of the module.

        Args:
            grad_outputs_args_kwargs (ArgsKwargs): The output gradients from previous modules (predecessors).
            forward_input_output_graph (Dict[GRAPH_NODE_TYPE, InputOutput]): The input and output from forward pass.

        Returns:
            ArgsKwargs: The output gradients.
        """

        grad_inputs_args = {}
        grad_inputs_kwargs = {}
        for key, grad_output in grad_outputs_args_kwargs.kwargs.items():
            location = self.input_output_connections[key]
            forward_out_arg: Union[None, int, bool] = location['in_arg']
            forward_out_kwarg: Union[None, str, bool] = location['in_kwarg']
            forward_in_arg: Union[None, int, bool] = location['out_arg']
            forward_in_kwarg: Union[None, str, bool] = location['out_kwarg']
            predecessor: GRAPH_NODE_TYPE = location['from']

            # 0 - not allowed

            # 4
            if forward_out_arg is True and isinstance(forward_in_arg, int) and not isinstance(forward_in_arg, bool):
                forward_inputs = forward_input_output_graph[predecessor].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.args
                forward_out_arg = _get_index(forward_outputs[forward_in_arg], forward_inputs)
                grad_output = grad_output[forward_out_arg]

            # 7
            if forward_out_arg is True and isinstance(forward_in_kwarg, str):
                forward_inputs = forward_input_output_graph[predecessor].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.kwargs
                forward_out_arg = _get_index(forward_outputs[forward_in_kwarg], forward_inputs)
                grad_output = grad_output[forward_out_arg]

            # 1
            if forward_out_arg is True and forward_in_arg is True:
                forward_inputs = forward_input_output_graph[predecessor].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.args
                for i in range(len(forward_inputs)):
                    if forward_inputs[i] not in forward_outputs:
                        continue

                    value_index = _get_index(forward_inputs[i], forward_outputs)
                    if value_index not in grad_inputs_args:
                        grad_inputs_args[value_index] = torch.zeros_like(grad_output[i])
                    grad_inputs_args[value_index] += grad_output[i]
                continue

            # 2
            if forward_out_arg is True and forward_in_kwarg is True:
                forward_inputs = forward_input_output_graph[predecessor].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.kwargs
                for i in forward_outputs:
                    value_index = _get_index(forward_outputs[i], forward_inputs)

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

            raise NotImplementedError('WTF!Why!')

        return ArgsKwargs(
            args=[grad_inputs_args[i] for i in sorted(grad_inputs_args.keys())],
            kwargs=grad_inputs_kwargs
        )

    grad = __call__
    """Alias for __call__."""
