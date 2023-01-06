from typing import Type, Optional

from torch import Tensor

from analogvnn.backward.BackwardModule import BackwardModule
from analogvnn.nn.module.Layer import Layer

__all__ = ['BackwardUsingForward']


class BackwardUsingForward(BackwardModule):
    def backward(self: Type[Layer], *grad_output, **grad_output_kwarg) -> Optional[Tensor]:
        return self._layer.forward(*grad_output, **grad_output_kwarg)
