from typing import Optional

import torch
from torch import Tensor, nn

from analogvnn.nn.activation.Activation import Activation

__all__ = ['Tanh']


class Tanh(Activation):
    """Implements the tanh activation function."""

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        """Forward pass of the tanh activation function.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """

        return torch.tanh(x)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """Backward pass of the tanh activation function.

        Args:
            grad_output (Optional[Tensor]): the gradient of the output tensor.

        Returns:
            Optional[Tensor]: the gradient of the input tensor.
        """

        x = self.inputs
        grad = 1 - torch.pow(torch.tanh(x), 2)
        return grad_output * grad

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        """Initialisation of tensor using xavier uniform, gain associated with tanh.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """

        return nn.init.xavier_uniform(tensor, gain=nn.init.calculate_gain('tanh'))

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        """In-place initialisation of tensor using xavier uniform, gain associated with tanh.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """

        return nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('tanh'))
