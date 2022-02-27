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
        if self.parent_module is not None and hasattr(self.parent_module, "use_autograd_graph") and not self.parent_module.use_autograd_graph:
            x = self.save_x(x)
            x.requires_grad = True
            y = self._layer(x)
            self.save_y(y, attached=True)
        else:
            y = self._layer(x)
        return y

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        self.y.backward(gradient=grad_output)
        return self.x.grad


if __name__ == '__main__':
    bw: FullSequential = FullSequential(BackwardWrapper(nn.Flatten(start_dim=0)))
    bw.compile()

    X: Tensor = torch.rand((2, 2), requires_grad=True)
    Y: Tensor = bw.output(X - torch.tensor(0.5, requires_grad=False))
    print("X\t\t\t\t:", X)
    print("Y\t\t\t\t:", Y)
    bw.backward.loss = Y
    bw.backward(gradient=torch.flatten(X, start_dim=0) - Y)
    print("X.grad\t\t\t:", X.grad)
