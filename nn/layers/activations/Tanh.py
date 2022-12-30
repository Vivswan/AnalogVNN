from typing import Optional

import torch
from torch import Tensor, nn

from nn.layers.activations.Activation import Activation


class Tanh(Activation):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return torch.tanh(x)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        x = self.inputs
        grad = 1 - torch.pow(torch.tanh(x), 2)
        return grad_output * grad

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor, gain=nn.init.calculate_gain('tanh'))

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('tanh'))
