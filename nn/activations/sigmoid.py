from typing import Union

import torch
from torch import Tensor, nn

from nn.activations.init_implementation import InitImplement
from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer


class Logistic(BaseLayer, BackwardFunction, InitImplement):
    def forward(self, x: Tensor) -> Tensor:
        self.save_tensor("input", x)
        return 1 / (1 + torch.exp(-x))

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        grad = self.forward(x) * (1 - self.forward(x))
        return grad_output * grad

    def initialise(self, tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor, gain=nn.init.calculate_gain('sigmoid'))

    def initialise_(self, tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('sigmoid'))


class Sigmoid(Logistic):
    pass


class SiLU(BaseLayer, BackwardFunction):
    def forward(self, x: Tensor) -> Tensor:
        self.save_tensor("input", x)
        return x / (1 + torch.exp(-x))

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        neg_e = torch.exp(-x)
        grad = (1 + neg_e + x * neg_e) / torch.pow(1 + neg_e, 2)
        return grad_output * grad


class Tanh(BaseLayer, BackwardFunction, InitImplement):
    def forward(self, x: Tensor) -> Tensor:
        self.save_tensor("input", x)
        return torch.tanh(x)

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        grad = 1 - torch.pow(torch.tanh(x), 2)
        return grad_output * grad

    def initialise(self, tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor, gain=nn.init.calculate_gain('tanh'))

    def initialise_(self, tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('tanh'))
