from typing import Union

import torch
from torch import Tensor

from nn.layers.activations.Activation import Activation


class SiLU(Activation):
    def forward(self, x: Tensor) -> Tensor:
        self.save_tensor("input", x)
        return x / (1 + torch.exp(-x))

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        neg_e = torch.exp(-x)
        grad = (1 + neg_e + x * neg_e) / torch.pow(1 + neg_e, 2)
        return grad_output * grad
