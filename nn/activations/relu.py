import math
from typing import Union

import torch
from torch import Tensor, nn

from nn.activations.init_implementation import InitImplement
from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer
from nn.utils.is_using_cuda import get_device

ZERO_TENSOR = torch.tensor(0, device=get_device())


class PReLU(BaseLayer, BackwardFunction, InitImplement):
    __constants__ = ['alpha']
    alpha: nn.Parameter

    def __init__(self, alpha: float):
        super(PReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        self.save_tensor("input", x)
        return torch.minimum(ZERO_TENSOR, x) * self.alpha + torch.maximum(ZERO_TENSOR, x)

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        grad = (x < 0).type(torch.float) * self.alpha + (x >= 0).type(torch.float)
        return grad_output * grad

    def initialise(self, tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor, gain=nn.init.calculate_gain('leaky_relu', param=float(self.alpha)))

    def initialise_(self, tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('leaky_relu', param=float(self.alpha)))


class ReLU(PReLU):
    def __init__(self):
        super(ReLU, self).__init__(alpha=0)

    def initialise(self, tensor: Tensor) -> Tensor:
        return nn.init.kaiming_uniform(tensor, a=math.sqrt(5), nonlinearity="relu")

    def initialise_(self, tensor: Tensor) -> Tensor:
        return nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), nonlinearity="relu")


class LeakyReLU(PReLU):
    def __init__(self):
        super(LeakyReLU, self).__init__(alpha=0.01)

    def initialise(self, tensor: Tensor) -> Tensor:
        return nn.init.kaiming_uniform(tensor, a=math.sqrt(5), nonlinearity="leaky_relu")

    def initialise_(self, tensor: Tensor) -> Tensor:
        return nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), nonlinearity="leaky_relu")
