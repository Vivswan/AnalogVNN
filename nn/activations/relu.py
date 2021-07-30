from typing import Union

import torch
from torch import Tensor

from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer


class PReLU(BaseLayer, BackwardFunction):
    __constants__ = ['alpha']
    alpha: float

    def __init__(self, alpha: float):
        super(PReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return (
            (x < 0).type(torch.float) * self.alpha * x +
            (x >= 0).type(torch.float) * x
        )

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return grad_output * (
            (grad_output < 0).type(torch.float) * self.alpha +
            (grad_output >= 0).type(torch.float)
        )


class ReLU(PReLU):
    def __init__(self):
        super(ReLU, self).__init__(alpha=0)


class LeakyReLU(PReLU):
    def __init__(self):
        super(LeakyReLU, self).__init__(alpha=0.01)
