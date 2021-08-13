import torch
from torch import nn, Tensor

from nn.backward_pass import BackwardFunction
from nn.base_layer import BaseLayer


class ReducePrecision(BaseLayer, BackwardFunction):
    __constants__ = ['precision', 'divide']
    precision: nn.Parameter
    divide: nn.Parameter
    shift: nn.Parameter

    def __init__(self, precision: int = 8, divide: float = 0.5, shift: float = 0, stochastic=False):
        # TODO: stochastic
        super(ReducePrecision, self).__init__()
        if precision < 1:
            raise ValueError("precision has to be more than 0, but got {}".format(precision))

        if precision != int(precision):
            raise ValueError("precision must be int, but got {}".format(precision))

        if not (0 <= divide <= 1):
            raise ValueError("divide must be between 0 and 1, but got {}".format(divide))

        if not (-1 <= shift <= 1):
            raise ValueError("shift must be between -1 and 1, but got {}".format(shift))

        self.precision = nn.Parameter(torch.tensor(precision), requires_grad=False)
        self.divide = nn.Parameter(torch.tensor(divide), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(shift), requires_grad=False)

    @property
    def step_size(self) -> float:
        return 1 / self.precision

    def extra_repr(self) -> str:
        return f'precision={self.precision}, divide={self.divide}'

    def forward(self, x: Tensor):
        if self.training:
            x = x * self.precision - self.shift * self.divide

            raise NotImplemented
        else:
            return x

    def backward(self, grad_output: Tensor):
        return grad_output


if __name__ == '__main__':
    for i in range(-10, 10):
        c = torch.tensor(i / 10)
        p = 1
        d = 0.2
        s = 0
        ct = c * p - s * d
        floor_ct_d = torch.sign(c) * torch.floor(torch.abs(ct) + d)
        floor_ct = torch.sign(c) * torch.floor(torch.abs(ct))
        floor_ct_nd = torch.sign(c) * torch.floor(torch.abs(ct) - d)

        ceil_ct_d = torch.sign(c) * torch.ceil(torch.abs(ct) + d)
        ceil_ct = torch.sign(c) * torch.ceil(torch.abs(ct))
        ceil_ct_nd = torch.sign(c) * torch.ceil(torch.abs(ct) - d)
        print(f"{c:.2f} : {floor_ct_d:.2f}, {floor_ct:.2f}, {floor_ct_nd:.2f}, {ceil_ct_d:.2f}, {ceil_ct:.2f}, {ceil_ct_nd:.2f}")
