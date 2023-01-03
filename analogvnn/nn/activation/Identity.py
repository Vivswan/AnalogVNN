from typing import Optional

from torch import Tensor

from analogvnn.nn.activation.Activation import Activation


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

    def forward(self, x: Tensor) -> Tensor:
        return x

    def backward(self, grad_output: Optional[Tensor]) -> Optional[Tensor]:
        return grad_output
