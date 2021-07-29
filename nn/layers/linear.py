from typing import Union

import torch
from torch import nn, Tensor

from nn.layers.base_layer import BaseLayer
from utils.helper_functions import to_matrix


class Linear(BaseLayer):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    _x: Union[None, Tensor]
    weight: nn.Parameter
    bias: Union[None, nn.Parameter]
    _fixed_fa_weight: nn.Parameter

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self._x: Union[None, Tensor] = None

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)

        self._fixed_fa_weight = nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=False)
        self._fa_mean: float = 0
        self._fa_std: float = 1.
        self._fa_fixed: bool = True
        self.set_feedforward_alignment_params(self._fa_mean, self._fa_std, self._fa_fixed)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.constant_(self.bias, 1)
        else:
            self.bias = None

        self.set_default_backward(self.backpropagation)

    def forward(self, x: Tensor):
        self._x = x.clone()
        y = x @ self.weight.t()
        if self.bias is not None:
            y += self.bias
        return y

    def set_feedforward_alignment_params(self, mean: Union[None, float] = None, std: Union[None, float] = None, is_fixed: Union[None, bool] = None):
        if mean is not None:
            self._fa_mean = mean
        if std is not None:
            self._fa_std = std
        if is_fixed is not None:
            self._fa_fixed = is_fixed

        nn.init.normal_(self._fixed_fa_weight, mean=self._fa_mean, std=self._fa_std)

    def backpropagation(self, grad_output, weight=None):
        if weight is None:
            weight = self.weight
        weight = to_matrix(weight)
        grad_output = to_matrix(grad_output)
        grad_input = grad_output @ weight

        if self.weight.requires_grad:
            self.weight.grad = grad_output.t().mm(to_matrix(self._x))
        if self.bias is not None and self.bias.requires_grad:
            self.bias.grad = grad_output.sum(0)

        self._x = None
        return grad_input

    def feedforward_alignment(self, grad_output):
        if self._fa_fixed:
            weight = self._fixed_fa_weight
        else:
            weight = torch.ones_like(self._fixed_fa_weight)
            weight.normal_(self._fa_mean, self._fa_std)

        return self.backpropagation(grad_output, weight)


    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'