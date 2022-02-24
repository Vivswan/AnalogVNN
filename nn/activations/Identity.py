from typing import Union

from torch import Tensor

from nn.activations.Activation import Activation


class Identity(Activation):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def extra_repr(self) -> str:
        if self.name is not None:
            return f'name={self.name}'
        else:
            return ''

    def forward(self, x: Tensor) -> Tensor:
        return x

    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return grad_output
