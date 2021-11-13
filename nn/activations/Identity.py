from typing import Union

from torch import Tensor

from nn.activations.Activation import Activation


class Identity(Activation):
    def forward(self, x: Tensor) -> Tensor:
        return x

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return grad_output
