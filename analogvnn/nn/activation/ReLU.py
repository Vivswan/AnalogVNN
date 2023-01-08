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
        """initialize the parametric rectified linear unit (PReLU) activation function.

        Args:
            alpha (float): the slope of the negative part of the activation function.
        """
        super(PReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self._zero = nn.Parameter(torch.tensor(0), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """forward pass of the parametric rectified linear unit (PReLU) activation function.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return torch.minimum(self._zero, x) * self.alpha + torch.maximum(self._zero, x)

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """backward pass of the parametric rectified linear unit (PReLU) activation function.

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
        """initialize the tensor using kaiming uniform initialization with gain associated
        with the parametric rectified linear unit (PReLU) activation function.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.kaiming_uniform(tensor, a=math.sqrt(5), nonlinearity="leaky_relu")

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        """in-place initialize the tensor using kaiming uniform initialization with gain associated
        with the parametric rectified linear unit (PReLU) activation function.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), nonlinearity="leaky_relu")


class ReLU(PReLU):
    """Implements the rectified linear unit (ReLU) activation function.
    Attributes:
        alpha (float): 0
    """

    def __init__(self):
        """initialize the rectified linear unit (ReLU) activation function.
        """
        super(ReLU, self).__init__(alpha=0)

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        """initialize the tensor using kaiming uniform initialization with gain associated
        with the rectified linear unit (ReLU) activation function.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.kaiming_uniform(tensor, a=math.sqrt(5), nonlinearity="relu")

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        return nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), nonlinearity="relu")


class LeakyReLU(PReLU):
    """Implements the leaky rectified linear unit (LeakyReLU) activation function.

    Attributes:
        alpha (float): 0.01
    """

    def __init__(self):
        """initialize the leaky rectified linear unit (LeakyReLU) activation function.
        """
        super(LeakyReLU, self).__init__(alpha=0.01)
