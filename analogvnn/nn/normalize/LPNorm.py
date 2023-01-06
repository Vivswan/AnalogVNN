import torch
from torch import nn, Tensor

from analogvnn.nn.normalize.Normalize import Normalize

__all__ = ['LPNorm', 'LPNormW', 'L1Norm', 'L2Norm', 'L1NormW', 'L2NormW', 'L1NormM', 'L2NormM', 'L1NormWM', 'L2NormWM']

class LPNorm(Normalize):
    __constants__ = ['p', 'make_max_1']
    p: nn.Parameter
    make_max_1: nn.Parameter

    def __init__(self, p: int, make_max_1=False):
        super(LPNorm, self).__init__()
        self.p = nn.Parameter(torch.tensor(p), requires_grad=False)
        self.make_max_1 = nn.Parameter(torch.tensor(make_max_1), requires_grad=False)

    def forward(self, x: Tensor):
        norm = x
        if len(x.shape) > 1:
            norm = torch.flatten(norm, start_dim=1)

        norm = torch.norm(norm, self.p, -1)
        norm = torch.clamp(norm, min=1e-4)
        x = torch.div(x.T, norm).T

        if self.make_max_1:
            x = torch.div(x, torch.max(torch.abs(x)))

        return x


class LPNormW(LPNorm):
    def forward(self, x: Tensor):
        norm = torch.norm(x, self.p)
        norm = torch.clamp(norm, min=1e-4)
        x = torch.div(x, norm)

        if self.make_max_1:
            x = torch.div(x, torch.max(torch.abs(x)))

        return x


class L1Norm(LPNorm):
    def __init__(self):
        super(L1Norm, self).__init__(p=1, make_max_1=False)


class L2Norm(LPNorm):
    def __init__(self):
        super(L2Norm, self).__init__(p=2, make_max_1=False)


class L1NormW(LPNormW):
    def __init__(self):
        super(L1NormW, self).__init__(p=1, make_max_1=False)


class L2NormW(LPNormW):
    def __init__(self):
        super(L2NormW, self).__init__(p=2, make_max_1=False)


class L1NormM(LPNorm):
    def __init__(self):
        super(L1NormM, self).__init__(p=1, make_max_1=True)


class L2NormM(LPNorm):
    def __init__(self):
        super(L2NormM, self).__init__(p=2, make_max_1=True)


class L1NormWM(LPNormW):
    def __init__(self):
        super(L1NormWM, self).__init__(p=1, make_max_1=True)


class L2NormWM(LPNormW):
    def __init__(self):
        super(L2NormWM, self).__init__(p=2, make_max_1=True)
