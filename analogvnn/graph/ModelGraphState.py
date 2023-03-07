from __future__ import annotations

from typing import Union, Dict, Optional

from torch import Tensor

from analogvnn.graph.ArgsKwargs import InputOutput, ArgsKwargs
from analogvnn.graph.GraphEnum import GraphEnum, GRAPH_NODE_TYPE

__all__ = ['ModelGraphState']


class ModelGraphState:
    """The state of a model graph.

    Attributes:
        allow_loops (bool): if True, the graph is allowed to contain loops.
        forward_input_output_graph (Optional[Dict[GRAPH_NODE_TYPE, InputOutput]]): the input and output of the
        forward pass.
        use_autograd_graph (bool): if True, the autograd graph is used to calculate the gradients.
        _loss (Tensor): the loss.
        INPUT (GraphEnum): GraphEnum.INPUT
        OUTPUT (GraphEnum): GraphEnum.OUTPUT
        STOP (GraphEnum): GraphEnum.STOP

    Properties:
        input (Tensor): the input of the forward pass.
        output (Tensor): the output of the forward pass.
        loss (Tensor): the loss.
    """

    allow_loops: bool
    use_autograd_graph: bool
    forward_input_output_graph: Optional[Dict[GRAPH_NODE_TYPE, InputOutput]]
    _loss: Optional[Tensor]

    INPUT = GraphEnum.INPUT
    OUTPUT = GraphEnum.OUTPUT
    STOP = GraphEnum.STOP

    def __init__(self, use_autograd_graph: bool = False, allow_loops=False):
        """Initialize the state.

        Args:
            use_autograd_graph: If True, the autograd graph is used to calculate the gradients.
            allow_loops: If True, the graph is allowed to contain loops.
        """

        super().__init__()
        self.allow_loops = allow_loops
        self.use_autograd_graph = use_autograd_graph

        self.forward_input_output_graph = None
        self._loss = None

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def ready_for_forward(self, exception: bool = False) -> bool:
        """Check if the state is ready for forward pass.

        Args:
            exception (bool): If True, an exception is raised if the state is not ready for forward pass.

        Returns:
            bool: True if the state is ready for forward pass.

        Raises:
            RuntimeError: If the state is not ready for forward pass and exception is True.
        """

        return True

    def ready_for_backward(self, exception: bool = False) -> bool:
        """Check if the state is ready for backward pass.

        Args:
            exception (bool): if True, raise an exception if the state is not ready for backward pass.

        Returns:
            bool: True if the state is ready for backward pass.

        Raises:
            RuntimeError: if the state is not ready for backward pass and exception is True.
        """

        if exception:
            if self.outputs is None:
                raise RuntimeError('output is not set.')

            if self._loss is None:
                raise RuntimeError('loss is not set.')

        return not (self.outputs is None or self._loss is None)

    @property
    def inputs(self) -> Optional[ArgsKwargs]:
        """Get the inputs.

        Returns:
            ArgsKwargs: the inputs.
        """

        if self.INPUT not in self.forward_input_output_graph:
            return None
        return self.forward_input_output_graph[self.INPUT].inputs

    @property
    def outputs(self) -> Optional[ArgsKwargs]:
        """Get the output.

        Returns:
            ArgsKwargs: the output.
        """

        if self.OUTPUT not in self.forward_input_output_graph:
            return None
        return self.forward_input_output_graph[self.OUTPUT].outputs

    @property
    def loss(self):
        """Get the loss.

        Returns:
            Tensor: the loss.
        """

        return self._loss

    def set_loss(self, loss: Union[Tensor, None]) -> ModelGraphState:
        """Set the loss.

        Args:
            loss (Tensor): the loss.

        Returns:
            ModelGraphState: self.
        """

        self._loss = loss
        return self
