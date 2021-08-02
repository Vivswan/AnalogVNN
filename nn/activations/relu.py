from typing import Union

import torch
from torch import Tensor

from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer

ZERO_TENSOR = torch.tensor(0)


class PReLU(BaseLayer, BackwardFunction):
    __constants__ = ['alpha']
    alpha: float

    def __init__(self, alpha: float):
        super(PReLU, self).__init__()
        self.alpha = torch.tensor(alpha)

    def forward(self, x: Tensor) -> Tensor:
        self.save_tensor("input", x)
        return torch.minimum(ZERO_TENSOR, x) * self.alpha + torch.maximum(ZERO_TENSOR, x)

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        grad = (x < 0).type(torch.float) * self.alpha + (x >= 0).type(torch.float)
        return grad_output * grad


class ReLU(PReLU):
    def __init__(self):
        super(ReLU, self).__init__(alpha=0)


class LeakyReLU(PReLU):
    def __init__(self):
        super(LeakyReLU, self).__init__(alpha=0.01)
