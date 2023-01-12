from typing import Optional

from torch import Tensor

from analogvnn.nn.activation.Activation import Activation

__all__ = ['Identity']


class Identity(Activation):
    """Implements the identity activation function.

    Attributes:
        name (str): the name of the activation function.
    """

    name: Optional[str]

    def __init__(self, name=None):
        """Initialize the identity activation function.

        Args:
            name (str): the name of the activation function.
        """
        super(Identity, self).__init__()
        self.name = name

    def extra_repr(self) -> str:
        """Extra __repr__ of the identity activation function.

        Returns:
            str: the extra representation of the identity activation function.
        """
        if self.name is not None:
            return f'name={self.name}'
        else:
            return ''

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        """Forward pass of the identity activation function.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor same as the input tensor.
        """
        return x

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """Backward pass of the identity activation function.

        Args:
            grad_output (Optional[Tensor]): the gradient of the output tensor.

        Returns:
            Optional[Tensor]: the gradient of the input tensor same as the gradient of the output tensor.
        """
        return grad_output
