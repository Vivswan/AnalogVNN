import math
from typing import Optional

import torch
from torch import Tensor, nn

from nn.layers.activations.Activation import Activation


class PReLU(Activation):
    __constants__ = ['alpha', '_zero']
    alpha: nn.Parameter
    _zero: nn.Parameter

    def __init__(self, alpha: float):
        super(PReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self._zero = nn.Parameter(torch.tensor(0), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.minimum(self._zero, x) * self.alpha + torch.maximum(self._zero, x)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        x = self.inputs
        grad = (x < 0).type(torch.float) * self.alpha + (x >= 0).type(torch.float)
        return grad_output * grad

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        return nn.init.kaiming_uniform(tensor, a=math.sqrt(5), nonlinearity="leaky_relu")

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        return nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), nonlinearity="leaky_relu")


class ReLU(PReLU):
    def __init__(self):
        super(ReLU, self).__init__(alpha=0)

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        return nn.init.kaiming_uniform(tensor, a=math.sqrt(5), nonlinearity="relu")

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        return nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), nonlinearity="relu")


class LeakyReLU(PReLU):
    def __init__(self):
        super(LeakyReLU, self).__init__(alpha=0.01)
