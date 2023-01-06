from typing import Optional

import torch
from torch import Tensor

from analogvnn.nn.normalize.Normalize import Normalize

__all__ = ['Clamp', 'Clamp01']


class Clamp(Normalize):
    @staticmethod
    def forward(x: Tensor):
        return torch.clamp(x, min=-1, max=1)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        x = self.inputs
        grad = ((-1 <= x) * (x <= 1.)).type(torch.float)
        return grad_output * grad


class Clamp01(Normalize):
    @staticmethod
    def forward(x: Tensor):
        return torch.clamp(x, min=0, max=1)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        x = self.inputs
        grad = ((0 <= x) * (x <= 1.)).type(torch.float)
        return grad_output * grad
