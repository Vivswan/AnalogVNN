from typing import Union

from torch import nn, Tensor

from nn.modules.Layer import Layer, BackwardFunction


class BackwardWrapper(Layer, BackwardFunction):
    def __init__(self, layer: nn.Module):
        super(BackwardWrapper, self).__init__()
        self._layer = layer
        self.input = None
        self.output = None

    def forward(self, x: Tensor):
        if self._parent_module_attr("use_autograd_graph") or not self.training:
            y = self._layer(x)
        else:
            x_dash = self.save_x(x)
            x_dash.requires_grad = True
            y = self._layer(x_dash)
            self.save_y(y, attached=True)
        return y

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        self.y.backward(gradient=grad_output)
        return self.x.grad
