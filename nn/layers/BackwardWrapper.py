from typing import Union

import torch
from torch import nn, Tensor

from nn.layers.BaseLayer import BaseLayer, BackwardFunction
from nn.modules.FullSequential import FullSequential


class BackwardWrapper(BaseLayer, BackwardFunction):
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


if __name__ == '__main__':
    bw: FullSequential = FullSequential(BackwardWrapper(nn.Flatten(start_dim=0)))
    bw.compile()

    X: Tensor = torch.rand((2, 2), requires_grad=True)
    Y: Tensor = bw(X - torch.tensor(0.5, requires_grad=False))
    print("X\t\t\t\t:", X)
    print("Y\t\t\t\t:", Y)
    bw.backward.set_loss(Y)
    bw.backward(gradient=torch.flatten(X, start_dim=0) - Y)
    print("X.grad\t\t\t:", X.grad)
