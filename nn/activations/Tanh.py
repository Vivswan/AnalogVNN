from typing import Union

import torch
from torch import Tensor, nn

from nn.activations.Activation import Activation


class Tanh(Activation):
    def forward(self, x: Tensor) -> Tensor:
        self.save_tensor("input", x)
        return torch.tanh(x)

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        grad = 1 - torch.pow(torch.tanh(x), 2)
        return grad_output * grad

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor, gain=nn.init.calculate_gain('tanh'))

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('tanh'))
