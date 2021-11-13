from typing import Union

import torch
from torch import Tensor, nn

from nn.activations.Activation import Activation


class SELU(Activation):
    __constants__ = ['alpha', 'scale_factor']
    alpha: nn.Parameter
    scale_factor: nn.Parameter

    def __init__(self, alpha: float = 1.0507, scale_factor: float = 1.):
        super(SELU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.scale_factor = nn.Parameter(torch.tensor(scale_factor), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        self.save_tensor("input", x)
        return self.scale_factor * (
                (x <= 0).type(torch.float) * self.alpha * (torch.exp(x) - 1) +
                (x > 0).type(torch.float) * x
        )

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        x = self.get_tensor("input")
        grad = self.scale_factor * ((x < 0).type(torch.float) * self.alpha * torch.exp(x) + (x >= 0).type(torch.float))
        return grad_output * grad

    @staticmethod
    def initialise(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor, gain=nn.init.calculate_gain('selu'))

    @staticmethod
    def initialise_(tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('selu'))


class ELU(SELU):
    def __init__(self, alpha: float = 1.0507):
        super(ELU, self).__init__(alpha=alpha, scale_factor=1.)
