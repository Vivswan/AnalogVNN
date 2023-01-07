from typing import Optional

import torch
from torch import Tensor

from analogvnn.nn.activation.Activation import Activation

__all__ = ['BinaryStep']


class BinaryStep(Activation):
    """Implements the binary step activation function.
    """

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        """forward pass of the binary step activation function.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return (x >= 0).type(torch.float)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """backward pass of the binary step activation function.

        Args:
            grad_output (Optional[Tensor]): the gradient of the output tensor.

        Returns:
            Optional[Tensor]: the gradient of the input tensor.
        """
        return torch.zeros_like(grad_output)
