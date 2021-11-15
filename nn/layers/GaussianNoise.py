from typing import Union

import torch
from torch import Tensor, nn

from nn.BaseLayer import BaseLayer
from nn.backward_pass.BackwardFunction import BackwardIdentity


class GaussianNoise(BaseLayer, BackwardIdentity):
    __constants__ = ['std']

    def __init__(self, std: Union[None, int, float] = None, leakage: Union[None, int, float] = None,
                 precision: Union[None, int] = None):
        super(GaussianNoise, self).__init__()
        if std is None and leakage is None:
            raise ValueError("Invalid arguments not found std or leakage")

        tensor_sqrt_2 = torch.sqrt(torch.tensor(2, requires_grad=False))
        self.leakage = None
        self.precision = None
        if std is not None:
            self.std = torch.tensor(std, requires_grad=False)

        if leakage is not None:
            if precision is None:
                raise ValueError("Invalid arguments 'precision' not found with leakage")

            self.leakage = torch.tensor(leakage, requires_grad=False)
            self.precision = torch.tensor(precision, requires_grad=False)
            self.std = 1 / (2 * self.precision * torch.erfinv(1 - self.leakage) * tensor_sqrt_2)

        self.std = nn.Parameter(self.std, requires_grad=False)
        self.leakage = nn.Parameter(self.leakage, requires_grad=False)
        self.precision = nn.Parameter(self.precision, requires_grad=False)

    def extra_repr(self) -> str:
        return f'std={self.std}, leakage={self.leakage}, precision={self.precision}'

    def forward(self, x: Tensor) -> Tensor:
        return torch.normal(mean=x, std=self.leakage)


if __name__ == '__main__':
    x = torch.randn((2, 2))
    print(GaussianNoise(leakage=0.1, precision=1)(x))
