import torch
from torch import nn

from nn.graphs.BackwardGraph import BackwardGraph
from nn.graphs.ForwardGraph import ForwardGraph
from nn.graphs.ModelGraphState import ModelGraphState
from nn.graphs.to_graph_viz_digraph import to_digraph
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
        return x + y + z + a + b

    def l5(x):
        return {"c": x * 0.5}

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
    mg.forward_graph.add_edge(l4, l5)
    mg.forward_graph.add_edge(l5, mg.OUTPUT, in_kwarg="c", out_arg=0)

    mg.compile(is_static=True, auto_backward_graph=True)
    mg.forward_graph.render("../../_data/forward", real_label=True)
    mg.backward_graph.render("../../_data/backward", real_label=True)
    inputs = torch.ones((1, 1))

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
    output = mg.forward_graph.calculate_graph(torch.ones((1, 1)))
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
    print(mg.backward_graph.calculate_graph(torch.ones((1, 1))))
