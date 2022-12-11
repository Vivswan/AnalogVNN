from typing import Union

from torch import Tensor

from nn.modules.Layer import BackwardFunction


class BackwardIdentity(BackwardFunction):
    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return grad_output
