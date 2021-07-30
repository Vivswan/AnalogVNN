from typing import Union

import torch
from torch import Tensor

from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer


class Logistic(BaseLayer, BackwardFunction):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-x))

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return self.forward(grad_output) * (1 - self.forward(grad_output))


class Sigmoid(Logistic):
    pass


class SiLU(BaseLayer, BackwardFunction):
    def forward(self, x: Tensor) -> Tensor:
        return x / (1 + torch.exp(-x))

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        neg_e = torch.exp(-grad_output)
        return (1 + neg_e + grad_output * neg_e) / torch.pow(1 + neg_e, 2)


class Tanh(BaseLayer, BackwardFunction):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x)

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return 1 - torch.pow(torch.tanh(grad_output), 2)
