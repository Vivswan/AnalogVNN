import inspect

import torch

from nn.graphs.BackwardFunction import BackwardFunction
from nn.graphs.ModelGraphState import ModelGraphState
from nn.graphs.TensorFlowGraph import TensorFlowGraph
from nn.layers.Linear import Linear
from nn.modules.Layer import Layer


class BackwardGraph(TensorFlowGraph):
    def __call__(self, gradient=None, *args, **kwargs):
        self._graph_state.ready_for_backward(exception=True)

        if len(gradient) == 0:
            gradient = None
        elif len(gradient) == 1:
            gradient = gradient[0]

        result = self._graph_state.loss.backward(gradient=gradient)
        if not self._graph_state.use_autograd_graph:
            result = self._pass(self._graph_state.output.grad)

        self._graph_state.set_outputs(None)
        self._graph_state.set_loss(None)

        return result

    def compile(self, is_static=True, **kwargs):
        if not self.graph.has_node(self.OUTPUT):
            raise Exception("OUTPUT doesn't exist in the forward graph")

        return super().compile(self.OUTPUT, is_static)

    @staticmethod
    def get_backward_function(module):
        if isinstance(module, Layer):
            if module.get_backward_module() is not None:
                return module.get_backward_module().backward
        if isinstance(module, BackwardFunction):
            return module.backward
        if inspect.ismethod(module) or inspect.isfunction(module):
            return module
        return None

    @torch.no_grad()
    def _pass(self, grad_output, **kwargs):
        return super()._pass(grad_output, self.OUTPUT, self.INPUT, search_module_function=self.get_backward_function)


if __name__ == '__main__':
    gb = BackwardGraph(ModelGraphState(allow_loops=True))
    l1 = Linear(1, 1)
    l2 = Linear(2, 2)
    l3 = Linear(3, 3)
    gb.add_connection(BackwardGraph.OUTPUT, l3, l2, l1, l2)
    gb.compile(is_static=True)
    gb._pass(None)
