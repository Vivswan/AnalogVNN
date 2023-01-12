import math
from typing import Optional

import torch
from torch import Tensor, nn

from analogvnn.nn.activation.Activation import Activation

__all__ = ['PReLU', 'ReLU', 'LeakyReLU']


class PReLU(Activation):
    """Implements the parametric rectified linear unit (PReLU) activation function.

    Attributes:
        alpha (float): the slope of the negative part of the activation function.
        _zero (Tensor): placeholder tensor of zero.
    """

    __constants__ = ['alpha', '_zero']
    alpha: nn.Parameter
    _zero: nn.Parameter

    def __init__(self, alpha: float):
        """Initialize the parametric rectified linear unit (PReLU) activation function.

        Args:
            alpha (float): the slope of the negative part of the activation function.
        """
        super(PReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self._zero = nn.Parameter(torch.tensor(0), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the parametric rectified linear unit (PReLU) activation function.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return torch.minimum(self._zero, x) * self.alpha + torch.maximum(self._zero, x)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """Backward pass of the parametric rectified linear unit (PReLU) activation function.

        Args:
            grad_output (Optional[Tensor]): the gradient of the output tensor.

        Returns:
            Optional[Tensor]: the gradient of the input tensor.
        """
        x = self.inputs
        grad = (x < 0).type(torch.float) * self.alpha + (x >= 0).type(torch.float)
        return grad_output * grad

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        """Initialisation of tensor using kaiming uniform, gain associated with PReLU activation function.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.kaiming_uniform(tensor, a=math.sqrt(5), nonlinearity='leaky_relu')

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        """In-place initialisation of tensor using kaiming uniform, gain associated with PReLU activation function.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), nonlinearity='leaky_relu')


class ReLU(PReLU):
    """Implements the rectified linear unit (ReLU) activation function.

    Attributes:
        alpha (float): 0
    """

    def __init__(self):
        """Initialize the rectified linear unit (ReLU) activation function."""
        super(ReLU, self).__init__(alpha=0)

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        """Initialisation of tensor using kaiming uniform, gain associated with ReLU activation function.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.kaiming_uniform(tensor, a=math.sqrt(5), nonlinearity='relu')

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        """In-place initialisation of tensor using kaiming uniform, gain associated with ReLU activation function.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), nonlinearity='relu')


class LeakyReLU(PReLU):
    """Implements the leaky rectified linear unit (LeakyReLU) activation function.

    Attributes:
        alpha (float): 0.01
    """

    def __init__(self):
        """Initialize the leaky rectified linear unit (LeakyReLU) activation function."""
        super(LeakyReLU, self).__init__(alpha=0.01)
