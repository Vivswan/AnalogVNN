from typing import Union

import torch
from torch import Tensor, nn

from nn.layers.activations.Activation import Activation


class Logistic(Activation):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-x))

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.inputs
        grad = self.forward(x) * (1 - self.forward(x))
        return grad_output * grad

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor, gain=nn.init.calculate_gain('sigmoid'))

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('sigmoid'))


class Sigmoid(Logistic):
    pass
