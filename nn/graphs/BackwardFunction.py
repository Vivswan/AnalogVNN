from typing import Union, Type

from torch import Tensor

from nn.modules.Layer import Layer, BackwardFunction


class BackwardUsingForward(BackwardFunction):
    def backward(self: Type[Layer], grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        mode = self._layer.training

        self._layer.training = True
        result = self._layer.forward(grad_output)
        self._layer.training = mode

        return result


class BackwardIdentity(BackwardFunction):
    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return grad_output
