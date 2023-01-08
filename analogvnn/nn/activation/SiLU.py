from typing import Optional

import torch
from torch import Tensor

from analogvnn.nn.activation.Activation import Activation

__all__ = ['SiLU']


class SiLU(Activation):
    """Implements the SiLU activation function.
    """

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        """forward pass of the SiLU

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return x / (1 + torch.exp(-x))

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """backward pass of the SiLU

        Args:
            grad_output (Optional[Tensor]): the gradient of the output tensor.

        Returns:
            Optional[Tensor]: the gradient of the input tensor.
        """
        x = self.inputs
        neg_e = torch.exp(-x)
        grad = (1 + neg_e + x * neg_e) / torch.pow(1 + neg_e, 2)
        return grad_output * grad
