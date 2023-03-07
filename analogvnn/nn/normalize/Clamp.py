from typing import Optional

import torch
from torch import Tensor

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.nn.normalize.Normalize import Normalize

__all__ = ['Clamp', 'Clamp01']


class Clamp(Normalize, BackwardIdentity):
    """Implements the clamp normalization function with range [-1, 1]."""

    @staticmethod
    def forward(x: Tensor):
        """Forward pass of the clamp normalization function with range [-1, 1].

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """

        return torch.clamp(x, min=-1, max=1)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """Backward pass of the clamp normalization function with range [-1, 1].

        Args:
            grad_output (Optional[Tensor]): the gradient of the output tensor.

        Returns:
            Optional[Tensor]: the gradient of the input tensor.
        """

        x = self.inputs
        grad = ((-1 <= x) * (x <= 1.)).type(torch.float)
        return grad_output * grad


class Clamp01(Normalize, BackwardIdentity):
    """Implements the clamp normalization function with range [0, 1]."""

    @staticmethod
    def forward(x: Tensor):
        """Forward pass of the clamp normalization function with range [0, 1].

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """

        return torch.clamp(x, min=0, max=1)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """Backward pass of the clamp normalization function with range [0, 1].

        Args:
            grad_output (Optional[Tensor]): the gradient of the output tensor.

        Returns:
            Optional[Tensor]: the gradient of the input tensor.
        """

        x = self.inputs
        grad = ((0 <= x) * (x <= 1.)).type(torch.float)
        return grad_output * grad
