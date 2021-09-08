from abc import ABC

from torch import Tensor, nn

from nn.BaseLayer import BaseLayer
from nn.backward_pass.BackwardFunction import BackwardFunction


class InitImplement:
    def initialise(self, tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor)

    def initialise_(self, tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor)


class Activation(BaseLayer, BackwardFunction, InitImplement, ABC):
    pass
