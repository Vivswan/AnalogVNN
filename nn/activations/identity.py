from typing import Union

import torch
from torch import Tensor

from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer


class Identity(BaseLayer, BackwardFunction):
    def forward(self, x: Tensor) -> Tensor:
        return x

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return grad_output


class BinaryStep(BaseLayer, BackwardFunction):
    def forward(self, x: Tensor) -> Tensor:
        return (x >= 0).type(torch.float)

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return torch.zeros_like(grad_output)
