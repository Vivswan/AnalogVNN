import networkx as nx
import torch
from graphviz import Digraph, Source
from pygraphviz import AGraph, graphviz
from torch import nn

from nn.graphs.BackwardGraph import BackwardGraph
from nn.graphs.ForwardGraph import ForwardGraph
from nn.graphs.GraphEnum import GraphEnum
from nn.graphs.ModelGraphState import ModelGraphState
from nn.layers.Linear import Linear
from nn.utils.make_dot import make_dot


class ModelGraph(ModelGraphState):
    def __init__(self, use_autograd_graph: bool = False, allow_loops=False):
        super().__init__(use_autograd_graph, allow_loops)
        self.forward_graph = ForwardGraph(self)
        self.backward_graph = BackwardGraph(self)

    def compile(self, is_static=True, auto_backward_graph=True):
        self.forward_graph.compile(is_static=is_static)

        if auto_backward_graph:
            self.backward_graph.from_forward(self.forward_graph)

        self.backward_graph.compile(is_static=is_static)


def to_digraph(N, real_label=False):
    """Returns a pygraphviz graph from a NetworkX graph N.

    Parameters
    ----------
    N : NetworkX graph
      A graph created with NetworkX

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> A = nx.nx_agraph.to_agraph(K5)

    Notes
    -----
    If N has an dict N.graph_attr an attempt will be made first
    to copy properties attached to the graph (see from_agraph)
    and then updated with the calling arguments if any.

    """
    try:
        import pygraphviz
    except ImportError as e:
        raise ImportError("requires pygraphviz " "http://pygraphviz.github.io/") from e
    strict = nx.number_of_selfloops(N) == 0 and not N.is_multigraph()
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')
    A = Digraph(name=N.name, strict=strict, node_attr=node_attr, graph_attr=dict(size="12,12"), format="svg")

    # default graph attributes
    A.graph_attr.update(N.graph.get("graph", {}))
    A.node_attr.update(N.graph.get("node", {}))
    A.edge_attr.update(N.graph.get("edge", {}))

    A.graph_attr.update(
        (k, v) for k, v in N.graph.items() if k not in ("graph", "node", "edge")
    )

    # add nodes
    for n, nodedata in N.nodes(data=True):
        attr = {k: str(v) for k, v in nodedata.items()}
        if isinstance(n, GraphEnum):
            attr["fillcolor"] = 'lightblue'
        attr["name"] = str(id(n))
        attr["label"] = str(n).strip()
        A.node(**attr)

    for u, v, edgedata in N.edges(data=True):
        attr = {k: str(v) for k, v in edgedata.items()}
        attr["tail_name"] = str(id(u))
        attr["head_name"] = str(id(v))

        if real_label and "real_label" in attr:
            attr["label"] = attr["real_label"]
        else:
            attr["label"] = attr["label"].replace("->", "→") if "label" in attr else None
        A.edge(**attr)

    return A


if __name__ == '__main__':
    mg = ModelGraph()
    # l1 = torch.nn.Linear(1, 1, bias=False)
    l1 = nn.Linear(1, 1, bias=False)
    l1.weight.data = torch.ones_like(l1.weight.data) * 2

    def l2(*x):
        return torch.add(*x), torch.sub(*x)


    def l3(x, y):
        return {"a": torch.sub(x, y), "b": torch.add(x, y)}


    def l4(x, y, z, a, b):
        return {"c": x + y + z + a + b}

    # l1 :: 1 -> 2
    # l2 :: (2, 1) -> (3, 1)
    # l3 :: (2, 3) -> {a: -1, b: 5}
    # l4 :: (-1, 5, 2, 3, 1) -> {c: 10}
    mg.forward_graph.add_edge(mg.INPUT, l1, in_arg=0)
    mg.forward_graph.add_edge(mg.INPUT, l2)
    mg.forward_graph.add_edge(l1, l2, out_arg=1)
    mg.forward_graph.add_edge(l1, l3, out_arg=0)
    mg.forward_graph.add_edge(l1, l3, out_arg=0)
    mg.forward_graph.add_edge(l2, l3, in_arg=1, out_arg=1)
    mg.forward_graph.add_edge(l2, l3, in_arg=0, out_arg=1)
    mg.forward_graph.add_edge(l3, l4, in_kwarg=True, out_arg=True)
    # mg.forward_graph.add_edge(l3, l4, in_kwarg="b", out_kwarg="y")
    mg.forward_graph.add_edge(l1, l4, out_kwarg="z")
    mg.forward_graph.add_edge(l2, l4, out_kwarg="a")
    mg.forward_graph.add_edge(l2, l4, in_arg=1, out_kwarg="b")
    mg.forward_graph.add_edge(l4, mg.OUTPUT, in_kwarg="c", out_arg=0)

    mg.compile(is_static=True, auto_backward_graph=True)
    to_digraph(mg.forward_graph.graph).render("../../_data/forward", format="svg", cleanup=True)
    to_digraph(mg.backward_graph.graph, real_label=True).render("../../_data/backward", format="svg", cleanup=True)
    inputs = torch.ones((1, 1), requires_grad=True)

    print()
    print("Starting Forward Pass ::")
    output = mg.forward_graph.calculate_graph(inputs)
    make_dot(output, params={
        "input": inputs,
        "output": output,
        "l1.weight": l1.weight,
    }).render("../../_data/model_graph", format="svg", cleanup=True)

    # print(f"output: {output}")
    # print("Starting Backward Pass ::")
    # # output.backward(torch.ones((1, 1)), retain_graph=True)
    # print(mg.backward_graph.calculate_graph(torch.ones((1, 1))))
    print()
    print("Grads ::")
    output = mg.forward_graph.calculate_graph(torch.ones((1, 1), requires_grad=True))
    for k, v in reversed(list(mg.forward_input_output_graph.items())):
        if len(v.outputs.args) > 0:
            grad = torch.autograd.grad(outputs=output, grad_outputs=torch.ones((1, 1)), inputs=v.outputs.args, retain_graph=True)
            print(f"{k} :o: {grad}")
            # print(f"inputs: {v.outputs.args}")
            # print()
        if len(v.outputs.kwargs.keys()) > 0:
            grad = {vk: vv for vk, vv in zip(
                list(v.outputs.kwargs.keys()),
                torch.autograd.grad(outputs=output, grad_outputs=torch.ones((1, 1)), inputs=list(v.outputs.kwargs.values()), retain_graph=True)
            )}
            print(f"{k} :o: {grad}")
            # print(f"inputs: {v.outputs.kwargs}")
            # print()
        if len(v.inputs.args) > 0:
            grad = torch.autograd.grad(outputs=output, grad_outputs=torch.ones((1, 1)), inputs=v.inputs.args, retain_graph=True)
            print(f"{k} :i: {grad}")
            # print(f"inputs: {v.inputs.args}")
            # print()
        if len(v.inputs.kwargs.keys()) > 0:
            grad = {vk: vv for vk, vv in zip(
                list(v.inputs.kwargs.keys()),
                torch.autograd.grad(outputs=output, grad_outputs=torch.ones((1, 1)), inputs=list(v.inputs.kwargs.values()), retain_graph=True)
            )}
            print(f"{k} :i: {grad}")
            # print(f"inputs: {v.inputs.kwargs}")
            # print()

    print()
    print("Starting Backward Pass ::")
    # output.backward(torch.ones((1, 1)), retain_graph=True)
    output = mg.forward_graph.calculate_graph(torch.ones((1, 1), requires_grad=True))
    output.grad = torch.ones((1, 1))
    print(mg.backward_graph.calculate_graph())
