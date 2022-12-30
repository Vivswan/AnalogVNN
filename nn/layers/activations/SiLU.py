from typing import Optional

import torch
from torch import Tensor

from nn.layers.activations.Activation import Activation


class SiLU(Activation):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x / (1 + torch.exp(-x))

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        x = self.inputs
        neg_e = torch.exp(-x)
        grad = (1 + neg_e + x * neg_e) / torch.pow(1 + neg_e, 2)
        return grad_output * grad
