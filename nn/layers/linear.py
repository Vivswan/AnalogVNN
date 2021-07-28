from typing import Union

import torch
from torch import nn, Tensor

from nn.Sequential import Sequential
from nn.layers.base_layer import BaseLayer
from utils.helper_functions import to_matrix


class Linear(BaseLayer):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    _x: Union[None, Tensor]
    weight: nn.Parameter
    bias: Union[None, nn.Parameter]
    fixed_FA_weight: nn.Parameter

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self._x: Union[None, Tensor] = None

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.fixed_FA_weight = nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.constant_(self.bias, 1)
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight)
        nn.init.kaiming_uniform_(self.fixed_FA_weight)
        self.backward.set_backward(self.backward_backpropagation)

    def forward(self, x: Tensor):
        self._x = x.clone()
        y = x @ self.weight.t()
        if self.bias is not None:
            y += self.bias
        return y

    def backward_backpropagation(self, grad_output):
        # print(f"grad_output: {grad_output}")
        grad_output = to_matrix(grad_output)
        weight = to_matrix(self.weight)
        grad_input = grad_output @ weight

        if self.weight.requires_grad:
            self.weight.grad = grad_output.t().mm(to_matrix(self._x))
        if self.bias is not None and self.bias.requires_grad:
            self.bias.grad = grad_output.sum(0)

        self._x = None
        return grad_input

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'