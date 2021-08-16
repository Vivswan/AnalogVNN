from typing import Union

import torch
from torch import Tensor

from nn.activations.activation import Activation


class Identity(Activation):
    def forward(self, x: Tensor) -> Tensor:
        return x

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        if bool(torch.any(torch.isnan(grad_output))):
            raise ValueError
        return grad_output


class BinaryStep(Activation):
    def forward(self, x: Tensor) -> Tensor:
        return (x >= 0).type(torch.float)

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return torch.zeros_like(grad_output)
