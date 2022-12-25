from typing import Union

from torch import Tensor

from nn.modules.Layer import BackwardModule


class BackwardIdentity(BackwardModule):
    def backward(self, grad_output: Union[None, Tensor]) -> Union[None, Tensor]:
        return grad_output
