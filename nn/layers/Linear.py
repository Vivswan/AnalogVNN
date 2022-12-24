import math
from typing import Union

import torch
from torch import nn, Tensor

from nn.fn.BackwardIdentity import BackwardFunction
from nn.fn.to_matrix import to_matrix
from nn.modules.Layer import Layer


class LinearBackpropagation(BackwardFunction):
    @property
    def weight(self):
        return self.get_parameter("weight")

    @property
    def bias(self):
        return self.get_parameter("bias")

    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        grad_output = to_matrix(grad_output)

        weight = to_matrix(self.weight if weight is None else weight)
        grad_input = grad_output @ weight

        self.set_grad_of(self.weight, torch.mm(grad_output.t(), self.inputs))
        self.set_grad_of(self.bias, grad_output.sum(0))
        return grad_input


class Linear(Layer):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    weight: nn.Parameter
    bias: Union[None, nn.Parameter]

    def __init__(
            self,
            in_features,
            out_features,
            bias=True
    ):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.use(LinearBackpropagation)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor):
        if self.bias is not None:
            y = torch.addmm(self.bias, x, self.weight.t())
        else:
            y = torch.mm(x, self.weight.t())

        return y

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
