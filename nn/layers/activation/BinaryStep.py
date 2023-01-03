from typing import Optional

import torch
from torch import Tensor

from nn.layers.activation.Activation import Activation


class BinaryStep(Activation):
    def forward(self, x: Tensor) -> Tensor:
        return (x >= 0).type(torch.float)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        return torch.zeros_like(grad_output)
