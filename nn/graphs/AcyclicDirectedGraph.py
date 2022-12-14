import abc

import networkx as nx

from nn.graphs.GraphEnum import GraphEnum
from nn.graphs.InputOutput import ArgsKwargs
from nn.graphs.ModelGraphState import ModelGraphState


class AcyclicDirectedGraph(abc.ABC):
    INPUT = GraphEnum.INPUT
    OUTPUT = GraphEnum.OUTPUT
    STOP = GraphEnum.STOP

    def __init__(self, graph_state: ModelGraphState = None):
        self.graph = nx.MultiDiGraph()
        self.graph_state: ModelGraphState = graph_state
        self._static_graph = None

        if self.graph_state.allow_loops:
            raise NotImplementedError("Loops are not implemented yet. Coming soon...")

    @abc.abstractmethod
    def __call__(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def add_connection(self, *args):
        for i in range(1, len(args)):
            self.add_edge(args[i - 1], args[i])
        return self

    def add_edge(
            self,
            u_of_edge,
            v_of_edge,
            in_arg=None,
            in_kwarg=None,
            out_arg=None,
            out_kwarg=None,
    ):
        # @@@ in_arg: None    in_kwarg: None  out_arg: None   out_kwarg: None

        # @@  in_arg: True    in_kwarg: None  out_arg: True   out_kwarg: None
        #     in_arg: True    in_kwarg: None  out_arg: None   out_kwarg: True
        #     in_arg: None    in_kwarg: True  out_arg: True   out_kwarg: None
        # @   in_arg: None    in_kwarg: True  out_arg: None   out_kwarg: True

        # @   in_arg: 0       in_kwarg: None  out_arg: True   out_kwarg: None
        #     in_arg: 0       in_kwarg: None  out_arg: 0      out_kwarg: None
        #     in_arg: 0       in_kwarg: None  out_arg: None   out_kwarg: 0
        #     in_arg: None    in_kwarg: 0     out_arg: True   out_kwarg: None
        #     in_arg: None    in_kwarg: 0     out_arg: 0      out_kwarg: None
        # @@  in_arg: None    in_kwarg: 0     out_arg: None   out_kwarg: 0

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

            if (in_arg is True or in_kwarg is True) and (out_arg is None and out_kwarg is None):
                out_arg = in_arg
                out_kwarg = in_kwarg

            label = f""
            if in_arg:
                label += "[]"
            if in_kwarg:
                label += "{}"
            label += " -> "
            if out_arg:
                label += "[]"
            if out_kwarg:
                label += "{}"

        elif in_arg is not None or in_kwarg is not None:
            # one -> one
            if out_arg is not None and isinstance(out_arg, int) and out_arg < 0:
                raise ValueError('"out_arg" must be a number >= 0')
            if in_arg is not None and isinstance(in_arg, int) and in_arg < 0:
                raise ValueError('"in_arg" must be a number >= 0')

            if in_arg is not None and (out_arg is None and out_kwarg is None):
                out_arg = 0
            if in_kwarg is not None and (out_arg is None and out_kwarg is None):
                out_kwarg = in_kwarg

            label = f""
            if in_arg is not None:
                label += "[" + str(in_arg) + "]"
            if in_kwarg is not None:
                label += "{" + str(in_kwarg) + "}"
            label += " -> "
            if out_arg is not None:
                label += "[" + str(out_arg) + "]"
            if out_kwarg is not None:
                label += "{" + str(out_kwarg) + "}"
        else:
            in_arg = True
            out_arg = True
            in_kwarg = True
            out_kwarg = True
            label = f"* -> *"

        self.graph.add_edge(u_of_edge, v_of_edge, **{
            "in_arg": in_arg,
            "in_kwarg": in_kwarg,
            "out_arg": out_arg,
            "out_kwarg": out_kwarg,
            "label": label,
        })

        return self

    def _create_sub_graph(self, from_node):
        nodes = nx.descendants(self.graph, from_node)
        nodes.add(from_node)
        sub_graph: nx.DiGraph = self.graph.subgraph(nodes)
        sorted_graph = nx.topological_sort(sub_graph)
        dependent_sorted_graph = []
        for i in sorted_graph:
            dependent_sorted_graph.append((i, list(sub_graph.predecessors(i))))
        return dependent_sorted_graph

    def compile(self, from_node, is_static=True):
        for i in nx.simple_cycles(self.graph):
            raise Exception(f"There is cyclic loop between {i}")

        if is_static:
            self._static_graph = self._create_sub_graph(from_node)
        else:
            self._static_graph = False

    @staticmethod
    def output_to_args_kwargs(outputs):
        if isinstance(outputs, ArgsKwargs):
            pass
        elif isinstance(outputs, dict):
            outputs = ArgsKwargs(kwargs=outputs)
        elif isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
            outputs = ArgsKwargs(args=outputs[0], kwargs=outputs[1])
        else:
            outputs = ArgsKwargs(args=outputs)
        return outputs

    def get_args_kwargs(self, input_output_graph, module, predecessors) -> ArgsKwargs:
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

                if in_arg is True and out_arg is True and in_kwarg is True and out_kwarg is True:
                    extra_args += previous_outputs.args
                    kwargs.update(previous_outputs.kwargs)
                    continue

                if in_arg is True and out_arg is True:
                    extra_args += previous_outputs.args
                    continue
                if in_kwarg is True and out_kwarg is True:
                    kwargs.update(previous_outputs.kwargs)
                    continue
                if in_arg is True and out_kwarg is True:
                    kwargs.update({i: v for i, v in enumerate(previous_outputs.args)})
                    continue
                if in_kwarg is True and out_arg is True:
                    extra_args += list(previous_outputs.kwargs.values())
                    continue

                if in_arg is not None:
                    previous_outputs = previous_outputs.args[in_arg]
                if in_kwarg is not None:
                    previous_outputs = previous_outputs.kwargs[in_kwarg]

                if out_arg is True:
                    extra_args.append(previous_outputs)
                    continue
                if out_arg is not None:
                    args[out_arg] = previous_outputs
                if out_kwarg is not None:
                    kwargs[out_kwarg] = previous_outputs

        args = [args[k] for k in sorted(args.keys())] + extra_args
        inputs = ArgsKwargs(
            args=args,
            kwargs=kwargs
        )
        return inputs


    @staticmethod
    def print_inputs_outputs(input_output_graph, module):
        if len(input_output_graph[module].inputs.args) > 0:
            print(f"{module} :i: {input_output_graph[module].inputs.args}")
        if len(input_output_graph[module].inputs.kwargs.keys()) > 0:
            print(f"{module} :i: {input_output_graph[module].inputs.kwargs}")
        if len(input_output_graph[module].outputs.args) > 0:
            print(f"{module} :o: {input_output_graph[module].outputs.args}")
        if len(input_output_graph[module].outputs.kwargs.keys()) > 0:
            print(f"{module} :o: {input_output_graph[module].outputs.kwargs}")
