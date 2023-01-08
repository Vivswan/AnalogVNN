import math
from typing import Optional

import torch
from torch import Tensor

from analogvnn.nn.activation.Activation import Activation

__all__ = ['Gaussian', 'GeLU']


class Gaussian(Activation):
    """Implements the Gaussian activation function.
    """

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        """forward pass of the Gaussian activation function.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return torch.exp(-torch.pow(x, 2))

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """backward pass of the Gaussian activation function.

        Args:
            grad_output (Optional[Tensor]): the gradient of the output tensor.

        Returns:
            Optional[Tensor]: the gradient of the input tensor.
        """
        x = self.inputs
        grad = -2 * x * torch.exp(-torch.pow(x, 2))
        return grad_output * grad


class GeLU(Activation):
    """Implements the Gaussian error linear unit (GeLU) activation function.
    """

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        """forward pass of the Gaussian error linear unit (GeLU) activation function.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return (1 / 2) * x * (1 + torch.erf(x / math.sqrt(2)))

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """backward pass of the Gaussian error linear unit (GeLU) activation function.

        Args:
            grad_output (Optional[Tensor]): the gradient of the output tensor.

        Returns:
            Optional[Tensor]: the gradient of the input tensor.
        """
        x = self.inputs
        grad = (1 / 2) * (
                (1 + torch.erf(x / math.sqrt(2))) + x * ((2 / math.sqrt(math.pi)) * torch.exp(-torch.pow(x, 2)))
        )
        return grad_output * grad
