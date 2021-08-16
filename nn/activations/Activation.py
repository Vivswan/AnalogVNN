from abc import ABC

from torch import Tensor, nn

from nn.BackwardFunction import BackwardFunction
from nn.BaseLayer import BaseLayer


class InitImplement:
    def initialise(self, tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform(tensor)

    def initialise_(self, tensor: Tensor) -> Tensor:
        return nn.init.xavier_uniform_(tensor)


class Activation(BaseLayer, BackwardFunction, InitImplement, ABC):
    def activation(self):
        pass
