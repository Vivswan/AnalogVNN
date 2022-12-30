import math
from typing import Optional

import torch
from torch import Tensor

from nn.layers.activations.Activation import Activation


class Gaussian(Activation):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return torch.exp(-torch.pow(x, 2))

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        x = self.inputs
        grad = -2 * x * torch.exp(-torch.pow(x, 2))
        return grad_output * grad


class GeLU(Activation):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return (1 / 2) * x * (1 + torch.erf(x / math.sqrt(2)))

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        x = self.inputs
        grad = (1 / 2) * (
                (1 + torch.erf(x / math.sqrt(2))) + x * ((2 / math.sqrt(math.pi)) * torch.exp(-torch.pow(x, 2)))
        )
        return grad_output * grad
