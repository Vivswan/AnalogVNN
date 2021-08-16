import math
from typing import Union

import torch
from torch import Tensor

from nn.activations.activation import Activation


class Gaussian(Activation):
    def forward(self, x: Tensor) -> Tensor:
        self.save_tensor("input", x)
        return torch.exp(-torch.pow(x, 2))

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        grad = -2 * x * torch.exp(-torch.pow(x, 2))
        return grad_output * grad


class GeLU(Activation):
    def forward(self, x: Tensor) -> Tensor:
        self.save_tensor("input", x)
        return (1 / 2) * x * (1 + torch.erf(x / math.sqrt(2)))

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        grad = (1 / 2) * (
                (1 + torch.erf(x / math.sqrt(2))) + x * ((2 / math.sqrt(math.pi)) * torch.exp(-torch.pow(x, 2))))
        return grad_output * grad
