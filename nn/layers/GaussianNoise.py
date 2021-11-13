from typing import Callable, Union

import torch
from torch import Tensor

from nn.BaseLayer import BaseLayer
from nn.backward_pass.BackwardFunction import BackwardFunction


class TensorFunctions:
    @staticmethod
    def x_to_x(x: Tensor) -> Tensor:
        return x

    @staticmethod
    def constant(constant: float) -> Callable[[Tensor], Tensor]:
        return lambda x: torch.mul(torch.ones(x.size()), constant)


class GaussianNoise(BaseLayer, BackwardFunction):
    __constants__ = ['mean', 'std']
    mean: Callable[[Tensor], Tensor]
    std: Callable[[Tensor], Tensor]

    def __init__(
            self,
            mean: Callable[[Tensor], Tensor] = TensorFunctions.x_to_x,
            std: Callable[[Tensor], Tensor] = TensorFunctions.constant(1)
    ):
        super(GaussianNoise, self).__init__()

        if not callable(mean):
            raise ValueError("mean must be callable")

        if not callable(std):
            raise ValueError("std must be callable")

        if not isinstance(mean(torch.zeros(1)), Tensor):
            raise ValueError("mean must return a Tensor")

        if not isinstance(std(torch.zeros(1)), Tensor):
            raise ValueError("mean must return a Tensor")

        self.mean = mean
        self.std = std

    def extra_repr(self) -> str:
        return f'mean={self.mean.__name__}, std={self.std.__name__}'

    def forward(self, x: Tensor, force: bool = False) -> Tensor:
        if self.training or force:
            return torch.normal(mean=x, std=torch.abs(x))
        return x

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return self.forward(grad_output, force=True)
