from __future__ import annotations

import torch
from torch import nn

from analogvnn.graph.BackwardGraph import BackwardGraph
from analogvnn.graph.ForwardGraph import ForwardGraph
from analogvnn.graph.ModelGraphState import ModelGraphState
from analogvnn.utils.make_dot import make_dot

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


if __name__ == '__main__':
    mg = ModelGraph()
    # l1 = torch.analogvnn.Linear(1, 1, bias=False)
    l1 = nn.Linear(1, 1, bias=False)
    l1.weight.data = torch.ones_like(l1.weight.data) * 2


    def l2(*x):
        return torch.add(*x), torch.sub(*x)


    def l3(x, y):
        return {"a": torch.sub(x, y), "b": torch.add(x, y)}


    def l4(x, y, z, a, b):
        return ((x + y) + (a + b)) + z


    def l5(x):
        return {"c": x * 0.5}


    # l1 :: 1 -> 2
    # l2 :: (1, 2) -> (3, -1)
    # l3 :: (2, 3) -> {a: -1, b: 5}
    # l4 :: (-1, 5, 2, 3, 1) -> 10
    # l5 :: 10 -> {c: 5}

    # l5 :: {c: 1} -> 0.5
    # l4 :: 0.5 -> (0.5, 0.5, 0.5, 0.5, 0.5)
    # l3 :: (0.5, 0.5) -> (1, 0)
    # l2 :: (0.5, 0.5) -> (1, 0)
    # l1 :: 0.5 + 1 + 0 -> 3
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
    mg.forward_graph.add_edge(l4, l5)
    mg.forward_graph.add_edge(l5, mg.OUTPUT, in_kwarg="c", out_arg=0)

    mg.compile(is_static=True, auto_backward_graph=True)
    mg.forward_graph.render("../../_data/forward", real_label=True)
    mg.backward_graph.render("../../_data/backward")

    print()
    print("Starting Forward Pass ::")
    output = mg.forward_graph.calculate(torch.ones((1, 1), dtype=torch.float))
    print(f"output = {output}")

    print()
    print("Grads ::")
    mg.use_autograd_graph = True
    inputs = torch.ones((1, 1), dtype=torch.float, requires_grad=True)
    output = mg.forward_graph.calculate(inputs)
    make_dot(output, params={
        "input": inputs,
        "output": output,
        "l1.weight": l1.weight,
    }).render("../../_data/model_graph", format="svg", cleanup=True)
    for k in reversed(list(mg.forward_input_output_graph)):

        output = mg.forward_graph.calculate(torch.ones((1, 1), dtype=torch.float, requires_grad=True))
        v = mg.forward_input_output_graph[k]
        print(f"{k} :o: ", end="")
        if len(v.outputs.args) > 0:
            grad = torch.autograd.grad(outputs=output, grad_outputs=torch.ones((1, 1), dtype=torch.float),
                                       inputs=v.outputs.args)
            print(f"{grad}, ", end="")

        output = mg.forward_graph.calculate(torch.ones((1, 1), dtype=torch.float, requires_grad=True))
        v = mg.forward_input_output_graph[k]
        if len(v.outputs.kwargs.keys()) > 0:
            grad = {vk: vv for vk, vv in zip(
                list(v.outputs.kwargs.keys()),
                torch.autograd.grad(outputs=output, grad_outputs=torch.ones((1, 1), dtype=torch.float),
                                    inputs=list(v.outputs.kwargs.values()))
            )}
            print(f"{grad}, ", end="")

        print()

        output = mg.forward_graph.calculate(torch.ones((1, 1), dtype=torch.float, requires_grad=True))
        v = mg.forward_input_output_graph[k]
        print(f"{k} :i: ", end="")
        if len(v.inputs.args) > 0:
            grad = torch.autograd.grad(outputs=output, grad_outputs=torch.ones((1, 1), dtype=torch.float),
                                       inputs=v.inputs.args)
            print(f"{grad}, ", end="")

        output = mg.forward_graph.calculate(torch.ones((1, 1), dtype=torch.float, requires_grad=True))
        v = mg.forward_input_output_graph[k]
        if len(v.inputs.kwargs.keys()) > 0:
            grad = {vk: vv for vk, vv in zip(
                list(v.inputs.kwargs.keys()),
                torch.autograd.grad(outputs=output, grad_outputs=torch.ones((1, 1), dtype=torch.float),
                                    inputs=list(v.inputs.kwargs.values()))
            )}
            print(f"{grad}, ", end="")

        print()

    print()
    print("Starting Backward Pass ::")
    mg.use_autograd_graph = False
    output = mg.forward_graph.calculate(torch.ones((1, 1), dtype=torch.float))
    print(mg.backward_graph.calculate(torch.ones((1, 1), dtype=torch.float)))
