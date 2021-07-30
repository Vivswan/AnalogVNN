from typing import Union

import torch
from torch import nn, Tensor

from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer
from utils.helper_functions import to_matrix


class LinearBackpropagation(BackwardFunction):
    @property
    def x(self):
        return self.get_tensor("_x")

    @property
    def weight(self):
        return self.get_tensor("weight")

    @property
    def bias(self):
        return self.get_tensor("bias")

    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        x = to_matrix(self.x)
        grad_output = to_matrix(grad_output)
        weight = to_matrix(self.weight if weight is None else weight)

        grad_input = grad_output @ weight

        if self.weight.requires_grad:
            self.weight.grad = grad_output.t() @ x
        if self.bias is not None and self.bias.requires_grad:
            self.bias.grad = grad_output.sum(0)

        return grad_input


class LinearFeedforwardAlignment(LinearBackpropagation):
    def __init__(self, layer):
        super(LinearFeedforwardAlignment, self).__init__(layer)
        self.mean: float = 0
        self.std: float = 1.
        self.is_fixed: bool = True
        self._fixed_weight = None

    def generate_random_weight(self, size = None):
        tensor = torch.rand(size if size is not None else self.weight.size())
        tensor.requires_grad = False
        tensor.normal_(mean=self.mean, std=self.std)
        return tensor

    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        if self._fixed_weight is None:
            self._fixed_weight = self.generate_random_weight()

        if weight is None:
            if self.is_fixed:
                weight = self._fixed_weight
            else:
                weight = self.generate_random_weight()

        return super(LinearFeedforwardAlignment, self).backward(grad_output, weight)


class LinearDirectFeedforwardAlignment(LinearFeedforwardAlignment):
    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        grad_output = to_matrix(grad_output)
        x = to_matrix(self.x)

        size = (grad_output.size()[1], self.weight.size()[0])

        if self._fixed_weight is None:
            self._fixed_weight = self.generate_random_weight(size)

        if weight is None:
            if self.is_fixed:
                weight = self._fixed_weight
            else:
                weight = self.generate_random_weight(size)

        weight = self.weight if weight is None else weight

        grad_output = grad_output @ weight

        if self.weight.requires_grad:
            self.weight.grad = grad_output.t() @ x
        if self.bias is not None and self.bias.requires_grad:
            self.bias.grad = grad_output.sum(0)

        return grad_output


class Linear(BaseLayer):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    _x: Union[None, Tensor]
    weight: nn.Parameter
    bias: Union[None, nn.Parameter]

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self._x: Union[None, Tensor] = None

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.constant_(self.bias, 1)
        else:
            self.bias = None

        self.backpropagation = LinearBackpropagation(self)
        self.feedforward_alignment = LinearFeedforwardAlignment(self)
        self.direct_feedforward_alignment = LinearDirectFeedforwardAlignment(self)

    def forward(self, x: Tensor):
        self._x = x.clone()
        self._x.detach_()
        self._x.requires_grad = False
        y = x @ self.weight.t()
        if self.bias is not None:
            y += self.bias
        return y

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'