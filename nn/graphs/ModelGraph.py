from nn.graphs.GraphBackwardPass import GraphBackwardPass
from nn.graphs.ForwardGraph import ForwardGraph
from nn.graphs.ModelGraphState import ModelGraphState


class ModelGraph(ModelGraphState):
    def __init__(self, allow_loops=False, use_autograd_graph: bool = False):
        super().__init__(allow_loops, use_autograd_graph)
        self.forward_graph = ForwardGraph(self)
        self.backward_graph = GraphBackwardPass()

    def compile(self, is_static=True):
        # self.forward_graph.compile(is_static=is_static)
        # self.backward_graph.compile(is_static=is_static)
        self.backward_graph.compile()
