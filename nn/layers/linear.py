from typing import Union

import torch
from torch import nn, Tensor

from nn.backward_pass import BackwardFunction
from nn.layers.base_layer import BaseLayer
from utils.helper_functions import to_matrix

class LinearBackpropagation(BackwardFunction):
    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        x = self.get_matrix("_x")
        weight = self.get_matrix("weight") if weight is None else weight
        bias = self.get_matrix("bias")

        grad_output = to_matrix(grad_output)
        grad_input = grad_output @ weight

        if weight.requires_grad:
            self.set_grad("weight", grad_output.t() @ x)
        if bias is not None and bias.requires_grad:
            self.set_grad("bias", grad_output.sum(0))

        return grad_input


class LinearFeedforwardAlignment(LinearBackpropagation):
    def __init__(self, layer):
        super(LinearFeedforwardAlignment, self).__init__(layer)
        self.mean: float = 0
        self.std: float = 1.
        self.is_fixed: bool = True
        self._fixed_weight = None

    def generate_random_weight(self):
        tensor = torch.rand_like(self.get_matrix("weight"))
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
    def backward(self, grad_output: Union[None, Tensor], **kwargs) -> Union[None, Tensor]:
        pass

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
        y = x @ self.weight.t()
        if self.bias is not None:
            y += self.bias
        return y

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'