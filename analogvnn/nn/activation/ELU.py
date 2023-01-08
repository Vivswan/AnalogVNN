from typing import Optional

import torch
from torch import Tensor, nn

from analogvnn.nn.activation.Activation import Activation

__all__ = ['SELU', 'ELU']


class SELU(Activation):
    """Implements the scaled exponential linear unit (SELU) activation function.

    Attributes:
        alpha (nn.Parameter): the alpha parameter.
        scale_factor (nn.Parameter): the scale factor parameter.
    """
    __constants__ = ['alpha', 'scale_factor']
    alpha: nn.Parameter
    scale_factor: nn.Parameter

    def __init__(self, alpha: float = 1.0507, scale_factor: float = 1.):
        """initialize the scaled exponential linear unit (SELU) activation function.

        Args:
            alpha (float): the alpha parameter.
            scale_factor (float): the scale factor parameter.
        """
        super(SELU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.scale_factor = nn.Parameter(torch.tensor(scale_factor), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """forward pass of the scaled exponential linear unit (SELU) activation function.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return self.scale_factor * (
                (x <= 0).type(torch.float) * self.alpha * (torch.exp(x) - 1) +
                (x > 0).type(torch.float) * x
        )

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """backward pass of the scaled exponential linear unit (SELU) activation function.

        Args:
            grad_output (Optional[Tensor]): the gradient of the output tensor.

        Returns:
            Optional[Tensor]: the gradient of the input tensor.
        """
        x = self.inputs
        grad = self.scale_factor * ((x < 0).type(torch.float) * self.alpha * torch.exp(x) + (x >= 0).type(torch.float))
        return grad_output * grad

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        """initialize the tensor using xavier uniform initialization with gain associated
        with the scaled exponential linear unit (SELU) activation function.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.xavier_uniform(tensor, gain=nn.init.calculate_gain('selu'))

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        """in-place initialize the tensor using xavier uniform initialization with gain associated
        with the scaled exponential linear unit (SELU) activation function.

        Args:
            tensor (Tensor): the tensor to be initialized.

        Returns:
            Tensor: the initialized tensor.
        """
        return nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('selu'))


class ELU(SELU):
    """Implements the exponential linear unit (ELU) activation function.

    Attributes:
        alpha (nn.Parameter): 1.0507
        scale_factor (nn.Parameter): 1.
    """

    def __init__(self, alpha: float = 1.0507):
        """initialize the exponential linear unit (ELU) activation function.

        Args:
            alpha (float): the alpha parameter.
        """
        super(ELU, self).__init__(alpha=alpha, scale_factor=1.)
