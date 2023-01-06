from typing import Optional

from torch import Tensor

from analogvnn.nn.activation.Activation import Activation

__all__ = ['Identity']


class Identity(Activation):
    name: Optional[str]

    def __init__(self, name=None):
        super(Identity, self).__init__()
        self.name = name

    def extra_repr(self) -> str:
        if self.name is not None:
            return f'name={self.name}'
        else:
            return ''

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        return grad_output
