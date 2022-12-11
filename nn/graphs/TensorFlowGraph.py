import abc

import networkx as nx

from nn.graphs.ModelGraphState import ModelGraphState
from nn.graphs.TensorFlowGraphEnum import TensorFlowGraphEnum


class TensorFlowGraph(abc.ABC):
    INPUT = TensorFlowGraphEnum.INPUT
    OUTPUT = TensorFlowGraphEnum.OUTPUT
    STOP = TensorFlowGraphEnum.STOP

    def __init__(self, graph_state: ModelGraphState = None):
        self.graph = nx.DiGraph()
        self._graph_state: ModelGraphState = graph_state

        self._static_graph = None

    @abc.abstractmethod
    def __call__(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def add_connection(self, *args):
        for i in range(1, len(args)):
            self.graph.add_edge(args[i - 1], args[i])

    def compile(self, from_node, is_static=True):
        for i in nx.simple_cycles(self.graph):
            s = f"There is cyclic loop between {i}"
            if not self._graph_state.allow_loops:
                raise Exception(s)
            else:
                print(f"Warning: {s}")

        if is_static:
            self._static_graph = nx.dfs_successors(self.graph, from_node)
        else:
            self._static_graph = False

    def _pass(self, inputs, from_node, to_node=None, search_module_function=None):
        if len(inputs) == 1:
            inputs = inputs[0]
        results = []

        static_graph = self._static_graph or nx.dfs_successors(self.graph, from_node)
        to_visit_now = []
        to_visit_next = [(from_node, inputs)]

        while len(to_visit_now) > 0 or len(to_visit_next) > 0:
            if len(to_visit_now) == 0:
                to_visit_now = to_visit_next
                to_visit_next = []

            last_module, last_module_output = to_visit_now.pop(0)

            if last_module not in static_graph:
                continue

            for next_module in static_graph[last_module]:
                if next_module == to_node:
                    results.append((last_module, last_module_output))

                if next_module is None or isinstance(next_module, TensorFlowGraphEnum):
                    continue

                if search_module_function is not None:
                    grad_function = search_module_function(next_module)
                    if next_module is None:
                        raise NotImplementedError
                else:
                    grad_function = next_module

                next_module_output = grad_function(last_module_output)
                if next_module_output is None:
                    continue

                to_visit_next.append((next_module, next_module_output))

        return results
