import networkx as nx

from nn.graphs.GraphEnum import GraphEnum


def to_digraph(from_graph, real_label=False):
    """Returns a pygraphviz graph from a NetworkX graph N.

    Parameters
    ----------
    from_graph : NetworkX graph
      A graph created with NetworkX
    real_label: bool

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
    try:
        from graphviz import Digraph
    except ImportError as e:
        raise ImportError("requires graphviz " "http://pygraphviz.github.io/") from e
    strict = nx.number_of_selfloops(from_graph) == 0 and not from_graph.is_multigraph()
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')
    new_graph = Digraph(
        name=from_graph.name,
        strict=strict,
        node_attr=node_attr,
        graph_attr=dict(size="12,12"),
        format="svg"
    )

    # default graph attributes
    new_graph.graph_attr.update(from_graph.graph.get("graph", {}))
    new_graph.node_attr.update(from_graph.graph.get("node", {}))
    new_graph.edge_attr.update(from_graph.graph.get("edge", {}))

    new_graph.graph_attr.update(
        (k, v) for k, v in from_graph.graph.items() if k not in ("graph", "node", "edge")
    )

    # add nodes
    for n, nodedata in from_graph.nodes(data=True):
        attr = {k: str(v) for k, v in nodedata.items()}
        if isinstance(n, GraphEnum):
            attr["fillcolor"] = 'lightblue'
        attr["name"] = str(id(n))
        attr["label"] = str(n).strip()
        new_graph.node(**attr)

    for u, v, edgedata in from_graph.edges(data=True):
        attr = {k: str(v) for k, v in edgedata.items()}
        attr["tail_name"] = str(id(u))
        attr["head_name"] = str(id(v))

        if real_label and "real_label" in attr:
            attr["label"] = attr["real_label"]
        else:
            attr["label"] = attr["label"].replace("->", "→") if "label" in attr else None
        new_graph.edge(**attr)

    return new_graph
