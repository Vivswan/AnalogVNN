import math
from typing import Union

import torch
from torch import nn, Tensor

from nn.BaseLayer import BaseLayer
from nn.backward_pass.BackwardFunction import BackwardFunction
from nn.utils.is_using_cuda import get_device
from utils.helper_functions import to_matrix


def generate_random_weight(mean, std, size, device=get_device()):
    tensor = None
    while tensor is None:
        try:
            tensor = torch.normal(mean=mean, std=std, size=size, device=device)
            torch.linalg.pinv(tensor)
        except:
            pass
    tensor.requires_grad = False
    return tensor


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

    def reset_parameters(self):
        super(LinearFeedforwardAlignment, self).reset_parameters()
        self._fixed_weight = None

    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        if weight is None:
            if not self.is_fixed or self._fixed_weight is None:
                self._fixed_weight = generate_random_weight(mean=self.mean, std=self.std, size=self.weight.size(),
                                                            device=grad_output.device)
            weight = self._fixed_weight

        return super(LinearFeedforwardAlignment, self).backward(grad_output, weight)


class LinearDirectFeedforwardAlignment(LinearFeedforwardAlignment):
    def pre_backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        grad_output = to_matrix(grad_output)

        size = (grad_output.size()[1], self.weight.size()[0])

        if not self.is_fixed or self._fixed_weight is None:
            self._fixed_weight = generate_random_weight(mean=self.mean, std=self.std, size=size,
                                                        device=grad_output.device)

        grad_output = grad_output @ self._fixed_weight
        return grad_output

    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        grad_output = to_matrix(grad_output)

        if self.weight.requires_grad:
            self.weight.grad = grad_output.t() @ to_matrix(self.x)
        if self.bias is not None and self.bias.requires_grad:
            self.bias.grad = grad_output.sum(0)

        return grad_output


class LinearWeightFeedforwardAlignment(LinearFeedforwardAlignment):
    def previous_layer(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        grad_output = to_matrix(grad_output)

        size = (grad_output.size()[1], self.weight.size()[0])

        if self._fixed_weight is None:
            self._fixed_weight = generate_random_weight(size, device=grad_output.device)

        if self.is_fixed:
            weight = self._fixed_weight
        else:
            weight = generate_random_weight(size, device=grad_output.device)

        grad_output = grad_output @ weight
        return grad_output

    def backward(self, grad_output: Union[None, Tensor], weight: Union[None, Tensor] = None) -> Union[None, Tensor]:
        grad_output = to_matrix(grad_output)

        if self.weight.requires_grad:
            self.weight.grad = grad_output.t() @ to_matrix(self.x)
        if self.bias is not None and self.bias.requires_grad:
            self.bias.grad = grad_output.sum(0)

        return grad_output


class Linear(BaseLayer):
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

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor):
        y = torch.mm(x, self.weight.t())
        if self.bias is not None:
            y += self.bias

        self.save_tensor("input", x)
        self.save_tensor("output", y)
        return y

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
