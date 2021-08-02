import math
from typing import Union

import torch
from torch import nn, Tensor

from nn.activations.relu import ReLU, LeakyReLU
from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer
from nn.utils.is_using_cuda import get_device
from utils.helper_functions import to_matrix


class LinearBackpropagation(BackwardFunction):
    @property
    def x(self):
        return self.get_tensor("input")

    @property
    def weight(self):
        return self.get_tensor("weight")

    @property
    def bias(self):
        return self.get_tensor("bias")

    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        grad_output = to_matrix(grad_output)

        if self.activation is not None:
            grad_output = self.activation.backward(grad_output)

        weight = to_matrix(self.weight if weight is None else weight)
        grad_input = grad_output @ weight

        if self.weight.requires_grad:
            self.weight.grad = grad_output.t() @ to_matrix(self.x)
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

    def generate_random_weight(self, size=None, device=get_device()):
        tensor = torch.rand(size if size is not None else self.weight.size(), device=device)
        tensor.requires_grad = False
        tensor.normal_(mean=self.mean, std=self.std)
        return tensor

    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        if self._fixed_weight is None:
            self._fixed_weight = self.generate_random_weight(device=grad_output.device)

        if weight is None:
            if self.is_fixed:
                weight = self._fixed_weight
            else:
                weight = self.generate_random_weight(device=grad_output.device)

        return super(LinearFeedforwardAlignment, self).backward(grad_output, weight)


class LinearDirectFeedforwardAlignment(LinearFeedforwardAlignment):
    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        grad_output = to_matrix(grad_output)

        x = to_matrix(self.x)
        size = (grad_output.size()[1], self.weight.size()[0])

        if self._fixed_weight is None:
            self._fixed_weight = self.generate_random_weight(size, device=grad_output.device)

        if self.is_fixed:
            dfa_weight = self._fixed_weight
        else:
            dfa_weight = self.generate_random_weight(size)

        grad_output = grad_output @ dfa_weight
        if self.activation is not None:
            grad_output = self.activation.backward(grad_output)

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

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.backpropagation = LinearBackpropagation(self)
        self.feedforward_alignment = LinearFeedforwardAlignment(self)
        self.direct_feedforward_alignment = LinearDirectFeedforwardAlignment(self)

    def reset_parameters(self) -> None:
        if isinstance(self.activation, nn.ReLU) or isinstance(self.activation, ReLU):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), nonlinearity="relu")
        elif isinstance(self.activation, nn.LeakyReLU) or isinstance(self.activation, LeakyReLU):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), nonlinearity="leaky_relu")
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor):
        y = x @ self.weight.t()
        if self.bias is not None:
            y += self.bias

        self.save_tensor("input", x)
        self.save_tensor("output", y)

        if self.activation:
            y = self.activation(y)

        return y

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
