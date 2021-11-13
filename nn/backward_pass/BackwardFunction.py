from typing import Union, Type

from torch import Tensor

from nn.BaseLayer import BaseLayer, BackwardFunction


class BackwardUsingForward(BackwardFunction):
    def backward(self: Type[BaseLayer], grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        mode = self.training

        self.training = True
        result = self.forward(grad_output)
        self.training = mode

        return result


class BackwardIdentity(BackwardFunction):
    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return grad_output
