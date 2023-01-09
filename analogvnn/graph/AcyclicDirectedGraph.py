from __future__ import annotations

import abc
import typing
from typing import Dict, Union, List, Tuple

import networkx as nx

from analogvnn.graph.ArgsKwargs import ArgsKwargs, InputOutput
from analogvnn.graph.GraphEnum import GraphEnum, GRAPH_NODE_TYPE
from analogvnn.graph.ModelGraphState import ModelGraphState
from analogvnn.graph.to_graph_viz_digraph import to_graphviz_digraph

if typing.TYPE_CHECKING:
    pass

__all__ = ['AcyclicDirectedGraph']


class AcyclicDirectedGraph(abc.ABC):
    """The base class for all acyclic directed graphs.

    Attributes:
        graph (nx.MultiDiGraph): The graph.
        graph_state (ModelGraphState): The graph state.
        _is_static (bool): If True, the graph is not changing during runtime and will be cached.
        _static_graphs (Dict[GRAPH_NODE_TYPE, List[Tuple[GRAPH_NODE_TYPE, List[GRAPH_NODE_TYPE]]]]): The static graphs.
        INPUT (GraphEnum): GraphEnum.INPUT
        OUTPUT (GraphEnum): GraphEnum.OUTPUT
        STOP (GraphEnum): GraphEnum.STOP

    """
    graph: nx.MultiDiGraph
    graph_state: ModelGraphState
    _is_static: bool
    _static_graphs: Dict[GRAPH_NODE_TYPE, List[Tuple[GRAPH_NODE_TYPE, List[GRAPH_NODE_TYPE]]]]

    INPUT = GraphEnum.INPUT
    OUTPUT = GraphEnum.OUTPUT
    STOP = GraphEnum.STOP

    def __init__(self, graph_state: ModelGraphState = None):
        """Create a new graph.

        Args:
            graph_state (ModelGraphState): The graph state.

        Raises:
            NotImplementedError: If allow_loops is True, since this is not implemented yet.
        """
        self.graph = nx.MultiDiGraph()
        self.graph_state = graph_state
        self._is_static = False
        self._static_graphs = {}

        if self.graph_state.allow_loops:
            raise NotImplementedError("Loops are not implemented yet. Coming soon...")

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Performs pass through the graph

        Args:
            *args: Arguments
            **kwargs: Keyword arguments

        Raises:
            NotImplementedError: since method is abstract
        """
        raise NotImplementedError

    def add_connection(self, *args: GRAPH_NODE_TYPE):
        """Add a connection between nodes.

        Args:
            *args: The nodes.

        Returns:
            AcyclicDirectedGraph: self.
        """
        for i in range(1, len(args)):
            self.add_edge(args[i - 1], args[i])
        return self

    def add_edge(
            self,
            u_of_edge: GRAPH_NODE_TYPE,
            v_of_edge: GRAPH_NODE_TYPE,
            in_arg: Union[None, int, bool] = None,
            in_kwarg: Union[None, str, bool] = None,
            out_arg: Union[None, int, bool] = None,
            out_kwarg: Union[None, str, bool] = None
    ):
        """Add an edge to the graph.

        Args:
            u_of_edge (GRAPH_NODE_TYPE): The source node.
            v_of_edge (GRAPH_NODE_TYPE): The target node.
            in_arg (Union[None, int, bool]): The input argument.
            in_kwarg (Union[None, str, bool]): The input keyword argument.
            out_arg (Union[None, int, bool]): The output argument.
            out_kwarg (Union[None, str, bool]): The output keyword argument.

        Returns:
            AcyclicDirectedGraph: self.
        """
        attr = self.check_edge_parameters(in_arg, in_kwarg, out_arg, out_kwarg)
        existing_edges = self.graph.get_edge_data(u_of_edge, v_of_edge)

        if existing_edges is not None:
            to_remove = []
            for key, edge in existing_edges.items():
                if not (
                        edge["out_arg"] == attr["out_arg"] is not None
                        or
                        edge["out_kwarg"] == attr["out_kwarg"] is not None
                ):
                    continue
                to_remove.append(key)
            for key in to_remove:
                self.graph.remove_edge(u_of_edge, v_of_edge, key=key)

        self.graph.add_edge(u_of_edge, v_of_edge, **attr)
        self.graph.nodes[u_of_edge]["fillcolor"] = "lightblue"
        self.graph.nodes[v_of_edge]["fillcolor"] = "lightblue"
        return self

    @staticmethod
    def check_edge_parameters(
            in_arg: Union[None, int, bool],
            in_kwarg: Union[None, str, bool],
            out_arg: Union[None, int, bool],
            out_kwarg: Union[None, str, bool]
    ) -> Dict[str, Union[None, int, str, bool]]:
        """Check the edge's in and out parameters.

        Args:
            in_arg (Union[None, int, bool]): The input argument.
            in_kwarg (Union[None, str, bool]): The input keyword argument.
            out_arg (Union[None, int, bool]): The output argument.
            out_kwarg (Union[None, str, bool]): The output keyword argument.

        Returns:
            Dict[str, Union[None, int, str, bool]]: Dict of valid edge's in and out parameters.

        Raises:
            ValueError: If in and out parameters are invalid.
        """
        # @@@ in_arg: None    in_kwarg: None  out_arg: None   out_kwarg: None   0
        # @@  in_arg: True    in_kwarg: None  out_arg: True   out_kwarg: None   1
        #     in_arg: None    in_kwarg: True  out_arg: True   out_kwarg: None   2
        # @   in_arg: None    in_kwarg: True  out_arg: None   out_kwarg: True   3
        # @   in_arg: 0       in_kwarg: None  out_arg: True   out_kwarg: None   4
        #     in_arg: 0       in_kwarg: None  out_arg: 0      out_kwarg: None   5
        #     in_arg: 0       in_kwarg: None  out_arg: None   out_kwarg: a      6
        #     in_arg: None    in_kwarg: a     out_arg: True   out_kwarg: None   7
        #     in_arg: None    in_kwarg: a     out_arg: 0      out_kwarg: None   8
        # @@  in_arg: None    in_kwarg: a     out_arg: None   out_kwarg: a      9
        if out_arg is not None and out_kwarg is not None:
            raise ValueError('both "out_arg" and "out_kwarg" can\'t be present at the same time')
        if in_arg is not None and in_kwarg is not None:
            raise ValueError('both "in_arg" and "in_kwarg" can\'t be present at the same time')

        if in_arg is None and in_kwarg is None and (out_arg is True or out_kwarg is True):
            in_arg = out_arg
            in_kwarg = out_kwarg
        if in_arg is None and in_kwarg is None and (out_arg is not None or out_kwarg is not None):
            in_arg = 0

        if in_arg is True or in_kwarg is True:
            #  All -> All
            if in_arg not in [True, None]:
                raise ValueError(f'Invalid value for in_arg: "{in_arg}')
            if in_kwarg not in [True, None]:
                raise ValueError(f'Invalid value for in_kwarg: "{in_kwarg}')
            if out_arg not in [True, None]:
                raise ValueError(f'Invalid value for out_arg: "{out_arg}')
            if out_kwarg not in [True, None]:
                raise ValueError(f'Invalid value for out_kwarg: "{out_kwarg}')

            if in_arg is True and out_kwarg is True:
                raise ValueError(f'Invalid value in_arg -> out_kwarg')

            if (in_arg is True or in_kwarg is True) and (out_arg is None and out_kwarg is None):
                out_arg = in_arg
                out_kwarg = in_kwarg
        elif in_arg is not None or in_kwarg is not None:
            # one -> one
            if in_arg is not None and (not isinstance(in_arg, int) or in_arg < 0):
                raise ValueError('"in_arg" must be a number >= 0')
            if out_arg is not None and (not isinstance(out_arg, int) or out_arg < 0):
                raise ValueError('"out_arg" must be a number >= 0')
            if in_kwarg is not None and not isinstance(in_kwarg, str):
                raise ValueError('"in_kwarg" must be a string')
            if out_kwarg is not None and not isinstance(out_kwarg, str):
                raise ValueError('"out_kwarg" must be a string')

            if in_arg is not None and (out_arg is None and out_kwarg is None):
                out_arg = True
            if in_kwarg is not None and (out_arg is None and out_kwarg is None):
                out_kwarg = in_kwarg
        else:
            in_arg = True
            out_arg = True
            in_kwarg = True
            out_kwarg = True

        attr = {
            "in_arg": in_arg,
            "in_kwarg": in_kwarg,
            "out_arg": out_arg,
            "out_kwarg": out_kwarg,
        }
        attr["label"] = AcyclicDirectedGraph._create_edge_label(**attr)

        return attr

    # noinspection PyUnusedLocal
    @staticmethod
    def _create_edge_label(
            in_arg: Union[None, int, bool] = None,
            in_kwarg: Union[None, str, bool] = None,
            out_arg: Union[None, int, bool] = None,
            out_kwarg: Union[None, str, bool] = None,
            **kwargs
    ) -> str:
        """Create the edge's label.

        Args:
            in_arg (Union[None, int, bool]): The input argument.
            in_kwarg (Union[None, str, bool]): The input keyword argument.
            out_arg (Union[None, int, bool]): The output argument.
            out_kwarg (Union[None, str, bool]): The output keyword argument.

        Returns:
            str: The edge's label.
        """
        label = ""
        if in_arg == in_kwarg == out_arg == out_kwarg is True:
            return "* -> *"

        if in_arg is True:
            label += "[]"
        elif in_arg is not None:
            label += "[" + str(in_arg) + "]"
        if in_kwarg is True:
            label += "{}"
        elif in_kwarg is not None:
            label += "{" + str(in_kwarg) + "}"

        label += " -> "
        if out_arg is True:
            label += "[]"
        elif out_arg is not None:
            label += "[" + str(out_arg) + "]"
        if out_kwarg is True:
            label += "{}"
        elif out_kwarg is not None:
            label += "{" + str(out_kwarg) + "}"

        return label

    def compile(self, is_static: bool = True):
        """Compile the graph.

        Args:
            is_static (bool): If True, the graph will be compiled as a static graph.

        Returns:
            AcyclicDirectedGraph: The compiled graph.

        Raises:
            ValueError: If the graph is not acyclic.
        """
        for i in nx.simple_cycles(self.graph):
            raise ValueError(f'Cycle detected: {i}')

        self.graph = self._reindex_out_args(self.graph)
        self._is_static = is_static
        self._static_graphs = {}
        return self

    @staticmethod
    def _reindex_out_args(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Reindex the output arguments.

        Args:
            graph (nx.MultiDiGraph): The graph.

        Returns:
            nx.MultiDiGraph: The graph with re-indexed output arguments.
        """
        # noinspection PyTypeChecker
        graph: nx.MultiDiGraph = graph.copy()

        for v in graph.nodes():
            args_index = []
            for u in graph.predecessors(v):
                for _, edge_data in graph.get_edge_data(u, v).items():
                    if isinstance(edge_data["out_arg"], int) and not isinstance(edge_data["out_arg"], bool):
                        args_index.append(edge_data)

            if len(args_index) == 0:
                continue

            args_index = sorted(args_index, key=lambda x: x["out_arg"])
            for index, value in enumerate(args_index):
                value["out_arg"] = index
                value["label"] = AcyclicDirectedGraph._create_edge_label(**value)

        return graph

    def _create_static_sub_graph(
            self,
            from_node: GRAPH_NODE_TYPE
    ) -> List[Tuple[GRAPH_NODE_TYPE, List[GRAPH_NODE_TYPE]]]:
        """Create a static sub graph connected to the given node.

        Args:
            from_node (GRAPH_NODE_TYPE): The node.

        Returns:
            List[Tuple[GRAPH_NODE_TYPE, List[GRAPH_NODE_TYPE]]]: The static sub graph.
        """
        if self._is_static and from_node in self._static_graphs:
            return self._static_graphs[from_node]

        nodes = nx.descendants(self.graph, from_node)
        nodes.add(from_node)

        sub_graph: nx.MultiDiGraph = self.graph.subgraph(nodes)
        sorted_graph = nx.topological_sort(sub_graph)
        dependent_sorted_graph = []

        for i in sorted_graph:
            dependent_sorted_graph.append((i, list(sub_graph.predecessors(i))))

        if self._is_static:
            self._static_graphs[from_node] = dependent_sorted_graph

        return self._static_graphs[from_node]

    def parse_args_kwargs(
            self,
            input_output_graph: Dict[GRAPH_NODE_TYPE, InputOutput],
            module: GRAPH_NODE_TYPE,
            predecessors: List[GRAPH_NODE_TYPE]
    ) -> ArgsKwargs:
        """Parse the arguments and keyword arguments.

        Args:
            input_output_graph (Dict[GRAPH_NODE_TYPE, InputOutput]): The input output graph.
            module (GRAPH_NODE_TYPE): The module.
            predecessors (List[GRAPH_NODE_TYPE]): The predecessors.

        Returns:
            ArgsKwargs: The arguments and keyword arguments.
        """
        args = {}
        extra_args = []
        kwargs = {}
        for p in predecessors:
            edge_data = self.graph.get_edge_data(p, module)
            for k, v in edge_data.items():
                previous_outputs = input_output_graph[p].outputs
                in_arg = v["in_arg"]
                in_kwarg = v["in_kwarg"]
                out_arg = v["out_arg"]
                out_kwarg = v["out_kwarg"]

                # 0
                if in_arg is True and out_arg is True and in_kwarg is True and out_kwarg is True:
                    extra_args += previous_outputs.args
                    kwargs.update(previous_outputs.kwargs)
                    continue
                # 1
                if in_arg is True and out_arg is True:
                    extra_args += previous_outputs.args
                    continue
                # 3
                if in_kwarg is True and out_kwarg is True:
                    kwargs.update(previous_outputs.kwargs)
                    continue
                # 2
                if in_kwarg is True and out_arg is True:
                    extra_args += list(previous_outputs.kwargs.values())
                    continue
                # Backward - 0
                if in_arg is True and out_kwarg is not None:
                    kwargs[out_kwarg] = previous_outputs.args
                    continue
                # Backward - 0
                if in_kwarg is True and out_kwarg is not None:
                    kwargs[out_kwarg] = previous_outputs.kwargs
                    continue

                # 4, 5 & 6
                if in_arg is not None:
                    previous_outputs = previous_outputs.args[in_arg]
                # 7, 8 & 9
                if in_kwarg is not None:
                    previous_outputs = previous_outputs.kwargs[in_kwarg]

                # 4 & 7
                if out_arg is True:
                    extra_args.append(previous_outputs)
                    continue
                # 5 & 8
                if out_arg is not None:
                    args[out_arg] = previous_outputs
                    continue
                # 6 & 9
                if out_kwarg is not None:
                    kwargs[out_kwarg] = previous_outputs
                    continue

                raise NotImplementedError("WTF!Why!")

        args = [args[k] for k in sorted(args.keys())] + extra_args
        inputs = ArgsKwargs(
            args=args,
            kwargs=kwargs
        )
        return inputs

    # @staticmethod
    # def print_inputs_outputs(input_output_graph, module):
    #     if len(input_output_graph[module].inputs.args) > 0:
    #         print(f"{module} :i: {input_output_graph[module].inputs.args}")
    #     if len(input_output_graph[module].inputs.kwargs.keys()) > 0:
    #         print(f"{module} :i: {input_output_graph[module].inputs.kwargs}")
    #     if len(input_output_graph[module].outputs.args) > 0:
    #         print(f"{module} :o: {input_output_graph[module].outputs.args}")
    #     if len(input_output_graph[module].outputs.kwargs.keys()) > 0:
    #         print(f"{module} :o: {input_output_graph[module].outputs.kwargs}")

    def render(self, *args, real_label: bool = False, **kwargs) -> str:
        """Save the source to file and render with the Graphviz engine.

        Args:
            *args: Arguments to pass to graphviz render function.
            real_label: If True, the real label will be used instead of the label.
            **kwargs: Keyword arguments to pass to graphviz render function.

        Returns:
            str: The (possibly relative) path of the rendered file.
        """
        return to_graphviz_digraph(self.graph, real_label=real_label).render(*args, **kwargs)
