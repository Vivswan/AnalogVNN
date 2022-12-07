from typing import Union

import torch
from torch import Tensor

from nn.layers.activations.Activation import Activation


class BinaryStep(Activation):
    def forward(self, x: Tensor) -> Tensor:
        return (x >= 0).type(torch.float)

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return torch.zeros_like(grad_output)
