import math
from typing import Optional

import torch
from torch import nn, Tensor

from analogvnn.backward.BackwardModule import BackwardModule
from analogvnn.fn.to_matrix import to_matrix
from analogvnn.nn.module.Layer import Layer

__all__ = ['Linear', 'LinearBackpropagation']


class LinearBackpropagation(BackwardModule):
    """The backpropagation module of a linear layer."""

    def forward(self, x: Tensor):
        """Forward pass of the linear layer.

        Args:
            x (Tensor): The input of the linear layer.

        Returns:
            Tensor: The output of the linear layer.
        """

        if self.bias is not None:
            y = torch.addmm(self.bias, x, self.weight.t())
        else:
            y = torch.mm(x, self.weight.t())

        return y

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        """Backward pass of the linear layer.

        Args:
            grad_output (Optional[Tensor]): The gradient of the output.

        Returns:
            Optional[Tensor]: The gradient of the input.
        """

        grad_output = to_matrix(grad_output)

        weight = to_matrix(self.weight)
        grad_input = grad_output @ weight

        self.set_grad_of(self.weight, torch.mm(grad_output.t(), self.inputs))
        self.set_grad_of(self.bias, grad_output.sum(0))
        return grad_input


class Linear(Layer):
    """A linear layer.

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        weight (nn.Parameter): The weight of the layer.
        bias (nn.Parameter): The bias of the layer.
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: nn.Parameter
    bias: Optional[nn.Parameter]

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Create a new linear layer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool): True if the layer has a bias.
        """

        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.set_backward_function(LinearBackpropagation)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the layer."""

        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        """Extra representation of the linear layer.

        Returns:
            str: The extra representation of the linear layer.
        """

        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
