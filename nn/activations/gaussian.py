import math
from typing import Union

import torch
from torch import Tensor

from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer


class Gaussian(BaseLayer, BackwardFunction):
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-torch.pow(x, 2))

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return -2 * grad_output * torch.exp(-torch.pow(grad_output, 2))


class GeLU(BaseLayer, BackwardFunction):
    def forward(self, x: Tensor) -> Tensor:
        return (1 / 2) * x * (1 + torch.erf(x / math.sqrt(2)))

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return (1 / 2) * (
                (1 + torch.erf(grad_output / math.sqrt(2))) +
                grad_output * ((2 / math.sqrt(math.pi)) * torch.exp(-torch.pow(grad_output, 2)))
        )
