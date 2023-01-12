from __future__ import annotations

from analogvnn.graph.BackwardGraph import BackwardGraph
from analogvnn.graph.ForwardGraph import ForwardGraph
from analogvnn.graph.ModelGraphState import ModelGraphState

__all__ = ['ModelGraph']


class ModelGraph(ModelGraphState):
    """Store model's graph.

    Attributes:
        forward_graph (ForwardGraph): store model's forward graph.
        backward_graph (BackwardGraph): store model's backward graph.
    """

    forward_graph: ForwardGraph
    backward_graph: BackwardGraph

    def __init__(self, use_autograd_graph: bool = False, allow_loops: bool = False):
        """Initialize the model graph.

        Args:
            use_autograd_graph: If True, the autograd graph is used to calculate the gradients.
            allow_loops: If True, the graph is allowed to contain loops.
        """
        super(ModelGraph, self).__init__(use_autograd_graph, allow_loops)
        self.forward_graph = ForwardGraph(self)
        self.backward_graph = BackwardGraph(self)

    def compile(self, is_static: bool = True, auto_backward_graph: bool = False) -> ModelGraph:
        """Compile the model graph.

        Args:
            is_static (bool): If True, the model graph is static.
            auto_backward_graph (bool): If True, the backward graph is automatically created.

        Returns:
            ModelGraph: self.
        """
        self.forward_graph.compile(is_static=is_static)

        if auto_backward_graph:
            self.backward_graph.from_forward(self.forward_graph)

        self.backward_graph.compile(is_static=is_static)
        return self
