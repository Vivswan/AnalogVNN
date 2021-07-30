from typing import Union

import torch
from torch import Tensor

from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer


class SELU(BaseLayer, BackwardFunction):
    __constants__ = ['alpha', 'scale_factor']
    alpha: float
    scale_factor: float

    def __init__(self, alpha: float, scale_factor: float = 1.):
        super(SELU, self).__init__()
        self.alpha = alpha
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        return self.scale_factor * (
            (x <= 0).type(torch.float) * self.alpha * (torch.exp(x) - 1) +
            (x > 0).type(torch.float) * x
        )

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return grad_output * (
            (grad_output < 0).type(torch.float) * self.alpha * torch.exp(grad_output) +
            (grad_output >= 0).type(torch.float)
        )


class ELU(SELU):
    def __init__(self, alpha: float):
        super(ELU, self).__init__(alpha=alpha, scale_factor=1.)
