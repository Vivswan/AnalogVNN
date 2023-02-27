import torch
from torch import nn, Tensor

from analogvnn.backward.BackwardIdentity import BackwardIdentity
from analogvnn.nn.normalize.Normalize import Normalize

__all__ = ['LPNorm', 'LPNormW', 'L1Norm', 'L2Norm', 'L1NormW', 'L2NormW', 'L1NormM', 'L2NormM', 'L1NormWM', 'L2NormWM']


class LPNorm(Normalize, BackwardIdentity):
    """Implements the row-wise Lp normalization function.

    Attributes:
        p (int): the pth power of the Lp norm.
        make_max_1 (bool): if True, the maximum absolute value of the output tensor will be 1.
    """

    __constants__ = ['p', 'make_max_1']
    p: nn.Parameter
    make_max_1: nn.Parameter

    def __init__(self, p: int, make_max_1=False):
        """Initializes the row-wise Lp normalization function.

        Args:
            p (int): the pth power of the Lp norm.
            make_max_1 (bool): if True, the maximum absolute value of the output tensor will be 1.
        """

        super().__init__()
        self.p = nn.Parameter(torch.tensor(p), requires_grad=False)
        self.make_max_1 = nn.Parameter(torch.tensor(make_max_1), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of row-wise Lp normalization function.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """

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
    """Implements the whole matrix Lp normalization function."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of whole matrix Lp normalization function.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """

        norm = torch.norm(x, self.p)
        norm = torch.clamp(norm, min=1e-4)
        x = torch.div(x, norm)

        if self.make_max_1:
            x = torch.div(x, torch.max(torch.abs(x)))

        return x


class L1Norm(LPNorm):
    """Implements the row-wise L1 normalization function."""

    def __init__(self):
        """Initializes the row-wise L1 normalization function."""

        super().__init__(p=1, make_max_1=False)


class L2Norm(LPNorm):
    """Implements the row-wise L2 normalization function."""

    def __init__(self):
        """Initializes the row-wise L2 normalization function."""

        super().__init__(p=2, make_max_1=False)


class L1NormW(LPNormW):
    """Implements the whole matrix L1 normalization function."""

    def __init__(self):
        """Initializes the whole matrix L1 normalization function."""

        super().__init__(p=1, make_max_1=False)


class L2NormW(LPNormW):
    """Implements the whole matrix L2 normalization function."""

    def __init__(self):
        """Initializes the whole matrix L2 normalization function."""

        super().__init__(p=2, make_max_1=False)


class L1NormM(LPNorm):
    """Implements the row-wise L1 normalization function with maximum absolute value of 1."""

    def __init__(self):
        """Initializes the row-wise L1 normalization function with maximum absolute value of 1."""

        super().__init__(p=1, make_max_1=True)


class L2NormM(LPNorm):
    """Implements the row-wise L2 normalization function with maximum absolute value of 1."""

    def __init__(self):
        """Initializes the row-wise L2 normalization function with maximum absolute value of 1."""

        super().__init__(p=2, make_max_1=True)


class L1NormWM(LPNormW):
    """Implements the whole matrix L1 normalization function with maximum absolute value of 1."""

    def __init__(self):
        """Initializes the whole matrix L1 normalization function with maximum absolute value of 1."""

        super().__init__(p=1, make_max_1=True)


class L2NormWM(LPNormW):
    """Implements the whole matrix L2 normalization function with maximum absolute value of 1."""

    def __init__(self):
        """Initializes the whole matrix L2 normalization function with maximum absolute value of 1."""

        super().__init__(p=2, make_max_1=True)
