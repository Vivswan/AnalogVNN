from typing import Type, Union

from torch import Tensor

from nn.modules.BackwardModule import BackwardModule
from nn.modules.Layer import Layer


class BackwardUsingForward(BackwardModule):
    def backward(self: Type[Layer], grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        mode = self._layer.training

        self._layer.training = True
        result = self._layer.forward(grad_output)
        self._layer.training = mode

        return result
